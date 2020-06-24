
#include <pbrt/gpu/accel.h>

#include <pbrt/genscene.h>
#include <pbrt/gpu.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/lights.h>
#include <pbrt/loopsubdiv.h>
#include <pbrt/materials.h>
#include <pbrt/plymesh.h>
#include <pbrt/textures.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/log.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/stats.h>

#include <mutex>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define OPTIX_CHECK(EXPR)                                               \
        do {                                                            \
            OptixResult res = EXPR;                                     \
            if (res != OPTIX_SUCCESS)                                   \
                LOG_FATAL("OptiX call " #EXPR " failed with code %d: \"%s\"", \
                          int(res), optixGetErrorString(res));          \
        } while(false) /* eat semicolon */


namespace pbrt {

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) RaygenRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) MissRecord {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) GPUAccel::TriangleHitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshRecord rec;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) GPUAccel::BilinearPatchHitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    BilinearMeshRecord rec;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) GPUAccel::QuadricHitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    QuadricRecord rec;
};

GPUAccel::~GPUAccel() {}

extern "C" { extern const unsigned char PBRT_EMBEDDED_PTX[]; }

STAT_MEMORY_COUNTER("Memory/Acceleration structures", gpuBVHBytes);

OptixTraversableHandle
GPUAccel::buildBVH(const std::vector<OptixBuildInput> &buildInputs) const {
    ProfilerScope _(ProfilePhase::AccelConstruction);

    // Figure out memory requirements.
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = (OPTIX_BUILD_FLAG_ALLOW_COMPACTION |
                               OPTIX_BUILD_FLAG_PREFER_FAST_TRACE);
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext, &accelOptions,
                                             buildInputs.data(), buildInputs.size(),
                                             &blasBufferSizes));

    uint64_t *compactedSizeBufferPtr = alloc.new_object<uint64_t>();
    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = (CUdeviceptr)compactedSizeBufferPtr;

    // Allocate buffers.
    void *tempBuffer;
    CUDA_CHECK(cudaMalloc(&tempBuffer, blasBufferSizes.tempSizeInBytes));
    void *outputBuffer;
    CUDA_CHECK(cudaMalloc(&outputBuffer, blasBufferSizes.outputSizeInBytes));

    // Build.
    OptixTraversableHandle traversableHandle{0};
    OPTIX_CHECK(
        optixAccelBuild(optixContext, cudaStream, &accelOptions, buildInputs.data(),
                        buildInputs.size(), CUdeviceptr(tempBuffer),
                        blasBufferSizes.tempSizeInBytes,
                        CUdeviceptr(outputBuffer), blasBufferSizes.outputSizeInBytes,
                        &traversableHandle, &emitDesc, 1));

    CUDA_CHECK(cudaDeviceSynchronize());

    gpuBVHBytes += *compactedSizeBufferPtr;

    // Compact
    void *asBuffer;
    CUDA_CHECK(cudaMalloc(&asBuffer, *compactedSizeBufferPtr));

    OPTIX_CHECK(optixAccelCompact(optixContext, cudaStream, traversableHandle,
                                  CUdeviceptr(asBuffer), *compactedSizeBufferPtr,
                                  &traversableHandle));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(tempBuffer));
    CUDA_CHECK(cudaFree(outputBuffer));
    alloc.delete_object(compactedSizeBufferPtr);

    return traversableHandle;
}

static MaterialHandle getMaterial(const ShapeSceneEntity &shape,
                                  const std::map<std::string, MaterialHandle> &namedMaterials,
                                  const std::vector<MaterialHandle> &materials) {
    if (!shape.materialName.empty()) {
        auto iter = namedMaterials.find(shape.materialName);
        if (iter == namedMaterials.end())
            ErrorExit(&shape.loc, "%s: material not defined", shape.materialName);
        return iter->second;
    } else {
        CHECK_NE(shape.materialIndex, -1);
        return materials[shape.materialIndex];
    }
}

static FloatTextureHandle getAlphaTexture(const ShapeSceneEntity &shape,
                                          const std::map<std::string, FloatTextureHandle> &floatTextures) {
    std::string alphaTexName = shape.parameters.GetTexture("alpha");
    if (alphaTexName.empty())
        return nullptr;

    auto iter = floatTextures.find(alphaTexName);
    if (iter == floatTextures.end())
        ErrorExit(&shape.loc, "%s: alpha texture not defined.", alphaTexName);

    FloatTextureHandle alphaTextureHandle = iter->second;

    if (!BasicTextureEvaluator().Matches({alphaTextureHandle}, {})) {
        Warning(&shape.loc, "%s: alpha texture too complex for BasicTextureEvaluator "
                "(need fallback path). Ignoring for now.", alphaTexName);
        alphaTextureHandle = nullptr;
    }

    return alphaTextureHandle;
}

static int
getOptixGeometryFlags(FloatTextureHandle alphaTextureHandle,
                      MaterialHandle materialHandle) {
    if (materialHandle.HasSubsurfaceScattering())
        return OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
    else if (alphaTextureHandle || materialHandle.IsTransparent())
        return OPTIX_GEOMETRY_FLAG_NONE;
    else
        return OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
}

OptixTraversableHandle
GPUAccel::createGASForTriangles(const std::vector<ShapeSceneEntity> &shapes,
                             const OptixProgramGroup &intersectPG,
                             const OptixProgramGroup &shadowPG,
                             const OptixProgramGroup &randomHitPG,
                             const std::map<std::string, FloatTextureHandle> &floatTextures,
                             const std::map<std::string, MaterialHandle> &namedMaterials,
                             const std::vector<MaterialHandle> &materials,
                             const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
                             Bounds3f *gasBounds) {
    std::mutex mutex;
    std::vector<OptixBuildInput> buildInputs;
    std::vector<CUdeviceptr> pDeviceDevicePtrs;
    std::vector<uint32_t> triangleInputFlags;
    {
    ProfilerScope _(ProfilePhase::ShapeConstruction);

    ParallelFor(0, shapes.size(),
        [&](int64_t shapeIndex) {
            const auto &shape = shapes[shapeIndex];
            if (shape.name == "trianglemesh" || shape.name == "plymesh" ||
                shape.name == "loopsubdiv") {
                OptixBuildInput input = {};
                memset(&input, 0, sizeof(input));

                TriangleMesh *mesh = nullptr;
                if (shape.name == "trianglemesh") {
                    mesh = TriangleMesh::Create(shape.worldFromObject, shape.reverseOrientation,
                                                shape.parameters, &shape.loc, alloc);
                    CHECK(mesh != nullptr);
                } else if (shape.name == "loopsubdiv") {
                    mesh = CreateLoopSubdivMesh(shape.worldFromObject, shape.reverseOrientation,
                                                shape.parameters, &shape.loc, alloc);
                    CHECK(mesh != nullptr);
                } else {
                    CHECK_EQ(shape.name, "plymesh");
                    std::string filename = ResolveFilename(shape.parameters.GetOneString("plyfile", ""));
                    pstd::optional<PLYMesh> plyMesh = ReadPLYMesh(filename); // todo: alloc
                    if (!plyMesh)
                        return;

                    std::vector<int> indices = std::move(plyMesh->triIndices);
                    if (plyMesh->quadIndices.size()) {
                        // Convert quads back to pairs of triangles..
                        LOG_VERBOSE("Converting %d PLY quads to tris", plyMesh->quadIndices.size());
                        indices.reserve(indices.size() + 3 * plyMesh->quadIndices.size() / 2);

                        for (size_t i = 0; i < plyMesh->quadIndices.size(); i += 4) {
                            indices.push_back(plyMesh->quadIndices[i]);   // 0, 1, 2 of original
                            indices.push_back(plyMesh->quadIndices[i+1]);
                            indices.push_back(plyMesh->quadIndices[i+3]);

                            indices.push_back(plyMesh->quadIndices[i]);   // 0, 2, 3 of original
                            indices.push_back(plyMesh->quadIndices[i+3]);
                            indices.push_back(plyMesh->quadIndices[i+2]);
                        }
                    }

                    std::vector<Vector3f> s;  // can't pass {} for default through new_object...
                    mesh = alloc.new_object<TriangleMesh>(*shape.worldFromObject, shape.reverseOrientation,
                                                          indices, plyMesh->p, s, plyMesh->n,
                                                          plyMesh->uv, plyMesh->faceIndices);
                }

                Bounds3f bounds;
                for (size_t i = 0; i < mesh->nVertices; ++i)
                    bounds = Union(bounds, mesh->p[i]);

                input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

                input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                input.triangleArray.vertexStrideInBytes = sizeof(Point3f);
                input.triangleArray.numVertices = mesh->nVertices;
                // input.triangleArray.vertexBuffers is set later, once we're
                // done and pDeviceDevicePtrs isn't in danger of being
                // realloced...

                input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
                input.triangleArray.indexStrideInBytes = 3 *sizeof(int);
                input.triangleArray.numIndexTriplets = mesh->nTriangles;
                input.triangleArray.indexBuffer = CUdeviceptr(mesh->vertexIndices);

                input.triangleArray.numSbtRecords = 1;
                input.triangleArray.sbtIndexOffsetBuffer = CUdeviceptr(nullptr);
                input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
                input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

                FloatTextureHandle alphaTextureHandle = getAlphaTexture(shape, floatTextures);
                MaterialHandle materialHandle = getMaterial(shape, namedMaterials, materials);

                int flags = getOptixGeometryFlags(alphaTextureHandle, materialHandle);

                TriangleHitgroupRecord hgRecord;
                OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
                hgRecord.rec.mesh = mesh;
                hgRecord.rec.material = materialHandle;
                hgRecord.rec.alphaTexture = alphaTextureHandle;
                hgRecord.rec.areaLights = nullptr;
                if (shape.lightIndex != -1) {
                    // Note: this will hit if we try to have an instance as an area light.
                    auto iter = shapeIndexToAreaLights.find(shapeIndex);
                    CHECK(iter != shapeIndexToAreaLights.end());
                    // FIXME: needs updating when it's not just tri meshes...
                    CHECK_EQ(iter->second->size(), mesh->nTriangles);
                    hgRecord.rec.areaLights = iter->second->data();
                }

                /////////////////////////////////////////////////////////////////
                // With Mutex here and below...
                std::lock_guard<std::mutex> lock(mutex);

                *gasBounds = Union(*gasBounds, bounds);

                pDeviceDevicePtrs.push_back(CUdeviceptr(mesh->p));
                buildInputs.push_back(input);
                triangleInputFlags.push_back(flags);

                triangleIntersectHGRecords->push_back(hgRecord);

                OPTIX_CHECK(optixSbtRecordPackHeader(randomHitPG, &hgRecord));
                triangleRandomHitHGRecords->push_back(hgRecord);

                OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
                triangleShadowHGRecords->push_back(hgRecord);
            }
        });

    if (buildInputs.empty())
        return {};

    for (size_t i = 0; i < buildInputs.size(); ++i)
        buildInputs[i].triangleArray.vertexBuffers = &pDeviceDevicePtrs[i];

    CHECK_EQ(buildInputs.size(), triangleInputFlags.size());
    // Wire these up now so that the push_backs don't mess up the pointers...
    for (size_t i = 0; i < buildInputs.size(); ++i)
        buildInputs[i].triangleArray.flags = &triangleInputFlags[i];

    }

    return buildBVH(buildInputs);
}

OptixTraversableHandle
GPUAccel::createGASForBLPs(const std::vector<ShapeSceneEntity> &shapes,
                           const OptixProgramGroup &intersectPG,
                           const OptixProgramGroup &shadowPG,
                           const OptixProgramGroup &randomHitPG,
                           const std::map<std::string, FloatTextureHandle> &floatTextures,
                           const std::map<std::string, MaterialHandle> &namedMaterials,
                           const std::vector<MaterialHandle> &materials,
                           const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
                           Bounds3f *gasBounds) {
    std::vector<OptixBuildInput> buildInputs;
    pstd::vector<OptixAabb> shapeAABBs(alloc);
    std::vector<CUdeviceptr> aabbPtrs;
    std::vector<uint32_t> flags;
    {
    ProfilerScope _(ProfilePhase::ShapeConstruction);

    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &shape = shapes[shapeIndex];
        if (shape.name != "bilinearmesh")
            continue;

        BilinearPatchMesh *mesh =
            BilinearPatchMesh::Create(shape.worldFromObject, shape.reverseOrientation,
                                      shape.parameters, &shape.loc, alloc);
        CHECK(mesh != nullptr);

        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        buildInput.aabbArray.numSbtRecords = 1;
        buildInput.aabbArray.numPrimitives = mesh->nVertices;
        // aabbBuffers and flags pointers are set when we're done
        buildInputs.push_back(buildInput);

        Bounds3f shapeBounds;
        for (size_t i = 0; i < mesh->nVertices; ++i)
            shapeBounds = Union(shapeBounds, mesh->p[i]);

        OptixAabb aabb = { shapeBounds.pMin.x, shapeBounds.pMin.y, shapeBounds.pMin.z,
                           shapeBounds.pMax.x, shapeBounds.pMax.y, shapeBounds.pMax.z };
        shapeAABBs.push_back(aabb);

        *gasBounds = Union(*gasBounds, shapeBounds);

        MaterialHandle materialHandle = getMaterial(shape, namedMaterials, materials);
        FloatTextureHandle alphaTextureHandle = getAlphaTexture(shape, floatTextures);

        flags.push_back(getOptixGeometryFlags(alphaTextureHandle, materialHandle));

        BilinearPatchHitgroupRecord hgRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
        hgRecord.rec.mesh = mesh;
        hgRecord.rec.material = materialHandle;
        hgRecord.rec.alphaTexture = alphaTextureHandle;
        hgRecord.rec.areaLights = nullptr;
        if (shape.lightIndex != -1) {
            auto iter = shapeIndexToAreaLights.find(shapeIndex);
            // Note: this will hit if we try to have an instance as an area light.
            CHECK(iter != shapeIndexToAreaLights.end());
            CHECK_EQ(iter->second->size(), mesh->nPatches);
            hgRecord.rec.areaLights = iter->second->data();
        }
        bilinearPatchIntersectHGRecords->push_back(hgRecord);

        OPTIX_CHECK(optixSbtRecordPackHeader(randomHitPG, &hgRecord));
        bilinearPatchRandomHitHGRecords->push_back(hgRecord);

        OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
        bilinearPatchShadowHGRecords->push_back(hgRecord);
    }

    if (buildInputs.empty())
        return {};

    for (size_t i = 0; i < shapeAABBs.size(); ++i)
        aabbPtrs.push_back(CUdeviceptr(&shapeAABBs[i]));

    CHECK_EQ(buildInputs.size(), flags.size());
    for (size_t i = 0; i < buildInputs.size(); ++i) {
        buildInputs[i].aabbArray.aabbBuffers = &aabbPtrs[i];
        buildInputs[i].aabbArray.flags = &flags[i];
    }
    }

    return buildBVH(buildInputs);
}

OptixTraversableHandle GPUAccel::createGASForQuadrics(
        const std::vector<ShapeSceneEntity> &shapes,
        const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds) {
    std::vector<OptixBuildInput> buildInputs;
    pstd::vector<OptixAabb> shapeAABBs(alloc);
    std::vector<CUdeviceptr> aabbPtrs;
    std::vector<unsigned int> flags;
    {
    ProfilerScope _(ProfilePhase::ShapeConstruction);

    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &shape = shapes[shapeIndex];
        if (shape.name != "sphere" && shape.name != "cylinder" && shape.name != "disk")
            continue;

        pstd::vector<ShapeHandle> shapeHandles =
            ShapeHandle::Create(shape.name, shape.worldFromObject, shape.objectFromWorld,
                                shape.reverseOrientation, shape.parameters,
                                &shape.loc, alloc);
        if (shapeHandles.empty())
            continue;
        CHECK_EQ(1, shapeHandles.size());
        ShapeHandle shapeHandle = shapeHandles[0];

        OptixBuildInput buildInput = {};
        memset(&buildInput, 0, sizeof(buildInput));

        buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        buildInput.aabbArray.numSbtRecords = 1;
        buildInput.aabbArray.numPrimitives = 1;
        // aabbBuffers and flags pointers are set when we're done

        buildInputs.push_back(buildInput);

        Bounds3f shapeBounds = shapeHandle.WorldBound();
        OptixAabb aabb = { shapeBounds.pMin.x, shapeBounds.pMin.y, shapeBounds.pMin.z,
                           shapeBounds.pMax.x, shapeBounds.pMax.y, shapeBounds.pMax.z };
        shapeAABBs.push_back(aabb);

        *gasBounds = Union(*gasBounds, shapeBounds);

        // Find alpha texture, if present.
        MaterialHandle materialHandle = getMaterial(shape, namedMaterials, materials);
        FloatTextureHandle alphaTextureHandle = getAlphaTexture(shape, floatTextures);
        flags.push_back(getOptixGeometryFlags(alphaTextureHandle, materialHandle));

        QuadricHitgroupRecord hgRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
        hgRecord.rec.shape = shapeHandle;
        hgRecord.rec.material = materialHandle;
        hgRecord.rec.alphaTexture = alphaTextureHandle;
        hgRecord.rec.areaLight = nullptr;
        if (shape.lightIndex != -1) {
            auto iter = shapeIndexToAreaLights.find(shapeIndex);
            // Note: this will hit if we try to have an instance as an area light.
            CHECK(iter != shapeIndexToAreaLights.end());
            CHECK_EQ(iter->second->size(), 1);
            hgRecord.rec.areaLight = (*iter->second)[0];
        }
        quadricIntersectHGRecords->push_back(hgRecord);

        OPTIX_CHECK(optixSbtRecordPackHeader(randomHitPG, &hgRecord));
        quadricRandomHitHGRecords->push_back(hgRecord);

        OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
        quadricShadowHGRecords->push_back(hgRecord);
    }

    if (buildInputs.empty())
        return {};

    for (size_t i = 0; i < shapeAABBs.size(); ++i)
        aabbPtrs.push_back(CUdeviceptr(&shapeAABBs[i]));

    CHECK_EQ(buildInputs.size(), flags.size());
    for (size_t i = 0; i < buildInputs.size(); ++i) {
        buildInputs[i].aabbArray.aabbBuffers = &aabbPtrs[i];
        buildInputs[i].aabbArray.flags = &flags[i];
    }
    }

    return buildBVH(buildInputs);
}

static CUDAMemoryResource cudaMemoryResource;

GPUAccel::GPUAccel(const GeneralScene &scene, CUstream cudaStream,
                   const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights)
    : alloc(&cudaMemoryResource), cudaStream(cudaStream),
      triangleIntersectHGRecords(alloc.new_object<pstd::vector<TriangleHitgroupRecord>>(alloc)),
      triangleShadowHGRecords(alloc.new_object<pstd::vector<TriangleHitgroupRecord>>(alloc)),
      triangleRandomHitHGRecords(alloc.new_object<pstd::vector<TriangleHitgroupRecord>>(alloc)),
      bilinearPatchIntersectHGRecords(alloc.new_object<pstd::vector<BilinearPatchHitgroupRecord>>(alloc)),
      bilinearPatchShadowHGRecords(alloc.new_object<pstd::vector<BilinearPatchHitgroupRecord>>(alloc)),
      bilinearPatchRandomHitHGRecords(alloc.new_object<pstd::vector<BilinearPatchHitgroupRecord>>(alloc)),
      quadricIntersectHGRecords(alloc.new_object<pstd::vector<QuadricHitgroupRecord>>(alloc)),
      quadricShadowHGRecords(alloc.new_object<pstd::vector<QuadricHitgroupRecord>>(alloc)),
      quadricRandomHitHGRecords(alloc.new_object<pstd::vector<QuadricHitgroupRecord>>(alloc)) {
    CUcontext cudaContext;
    CU_CHECK(cuCtxGetCurrent(&cudaContext));
    CHECK(cudaContext != nullptr);

    paramsPool.resize(256);   // should be plenty
    for (ParamBufferState &ps : paramsPool) {
        void *ptr;
        CUDA_CHECK(cudaMalloc(&ptr, sizeof(RayIntersectParameters)));
        ps.ptr = (CUdeviceptr)ptr;
        CUDA_CHECK(cudaEventCreate(&ps.finishedEvent));
    }

    // Create OptiX context
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));

    LOG_VERBOSE("Optix successfully initialized");

    // OptiX module
    OptixModuleCompileOptions moduleCompileOptions = {};
    // TODO: REVIEW THIS
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#else
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#endif

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 4;
    // OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.exceptionFlags = (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                             OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                             OPTIX_EXCEPTION_FLAG_DEBUG);
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.overrideUsesMotionBlur = false;
    pipelineLinkOptions.maxTraceDepth = 2;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    const std::string ptxCode((const char *)PBRT_EMBEDDED_PTX);

    char log[4096];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
        optixContext, &moduleCompileOptions, &pipelineCompileOptions,
        ptxCode.c_str(), ptxCode.size(), log, &logSize, &optixModule));
    LOG_VERBOSE("%s", log);

    // Optix program groups...
    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroup raygenPGClosest;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = optixModule;
        desc.raygen.entryFunctionName = "__raygen__findClosest";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &raygenPGClosest));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup missPGNoOp;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = optixModule;
        desc.miss.entryFunctionName = "__miss__noop";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &missPGNoOp));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup hitPGTriangle;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = optixModule;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__triangle";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__triangle";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGTriangle));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup hitPGBilinearPatch;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = optixModule;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__bilinearPatch";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__bilinearPatch";
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__bilinearPatch";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGBilinearPatch));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup hitPGQuadric;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = optixModule;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__quadric";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__quadric";
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__quadric";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGQuadric));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup raygenPGShadow;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = optixModule;
        desc.raygen.entryFunctionName = "__raygen__shadow";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &raygenPGShadow));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup missPGShadow;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = optixModule;
        desc.miss.entryFunctionName = "__miss__shadow";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &missPGShadow));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup anyhitPGShadowTriangle;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__shadowTriangle";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &anyhitPGShadowTriangle));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup anyhitPGShadowBilinearPatch;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__bilinearPatch";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__shadowBilinearPatch";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &anyhitPGShadowBilinearPatch));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup anyhitPGShadowQuadric;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__quadric";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__shadowQuadric";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &anyhitPGShadowQuadric));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup raygenPGRandomHit;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = optixModule;
        desc.raygen.entryFunctionName = "__raygen__randomHit";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &raygenPGRandomHit));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup hitPGRandomHitTriangle;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__randomHitTriangle";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGRandomHitTriangle));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup hitPGRandomHitBilinearPatch;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__bilinearPatch";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__randomHitBilinearPatch";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGRandomHitBilinearPatch));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup hitPGRandomHitQuadric;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__quadric";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__randomHitQuadric";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGRandomHitQuadric));
        LOG_VERBOSE("%s", log);
    }

    // Optix pipeline...
    OptixProgramGroup allPGs[] = { raygenPGClosest, missPGNoOp,
                                   hitPGTriangle, hitPGBilinearPatch, hitPGQuadric,
                                   raygenPGShadow, missPGShadow,
                                   anyhitPGShadowTriangle, anyhitPGShadowBilinearPatch,
                                   anyhitPGShadowQuadric,
                                   raygenPGRandomHit, hitPGRandomHitTriangle,
                                   hitPGRandomHitBilinearPatch, hitPGRandomHitQuadric };
    OPTIX_CHECK(optixPipelineCreate(optixContext, &pipelineCompileOptions,
                                    &pipelineLinkOptions, allPGs,
                                    sizeof(allPGs) / sizeof(allPGs[0]),
                                    log, &logSize, &optixPipeline));
    LOG_VERBOSE("%s", log);

    OPTIX_CHECK(optixPipelineSetStackSize(optixPipeline,
                                          /* [in] The direct stack size requirement for
                                             direct callables invoked from IS or AH. */
                                          2 * 1024,
                                          /* [in] The direct stack size requirement for
                                             direct callables invoked from RG, MS, or
                                             CH.  */
                                          2 * 1024,
                                          /* [in] The continuation stack requirement. */
                                          2 * 1024,
                                          /* [in] The maximum depth of a traversable
                                             graph passed to trace. */
                                          2));  // NOTE: this has to increase if there are motion xforms I think

    // Shader binding tables...
    // Hitgroups are done as meshes are processed

    // Closest intersection
    RaygenRecord *raygenClosestRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGClosest, raygenClosestRecord));
    triangleIntersectSBT.raygenRecord = (CUdeviceptr)raygenClosestRecord;
    bilinearPatchIntersectSBT.raygenRecord = (CUdeviceptr)raygenClosestRecord;
    quadricIntersectSBT.raygenRecord = (CUdeviceptr)raygenClosestRecord;

    MissRecord *missNoOpRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGNoOp, missNoOpRecord));
    triangleIntersectSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    triangleIntersectSBT.missRecordStrideInBytes = sizeof(MissRecord);
    triangleIntersectSBT.missRecordCount = 1;
    bilinearPatchIntersectSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    bilinearPatchIntersectSBT.missRecordStrideInBytes = sizeof(MissRecord);
    bilinearPatchIntersectSBT.missRecordCount = 1;
    quadricIntersectSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    quadricIntersectSBT.missRecordStrideInBytes = sizeof(MissRecord);
    quadricIntersectSBT.missRecordCount = 1;

    // Shadow
    RaygenRecord *raygenShadowRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGShadow, raygenShadowRecord));
    triangleShadowSBT.raygenRecord = (CUdeviceptr)raygenShadowRecord;
    bilinearPatchShadowSBT.raygenRecord = (CUdeviceptr)raygenShadowRecord;
    quadricShadowSBT.raygenRecord = (CUdeviceptr)raygenShadowRecord;

    MissRecord *missShadowRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGShadow, missShadowRecord));
    triangleShadowSBT.missRecordBase = (CUdeviceptr)missShadowRecord;
    triangleShadowSBT.missRecordStrideInBytes = sizeof(MissRecord);
    triangleShadowSBT.missRecordCount = 1;
    bilinearPatchShadowSBT.missRecordBase = (CUdeviceptr)missShadowRecord;
    bilinearPatchShadowSBT.missRecordStrideInBytes = sizeof(MissRecord);
    bilinearPatchShadowSBT.missRecordCount = 1;
    quadricShadowSBT.missRecordBase = (CUdeviceptr)missShadowRecord;
    quadricShadowSBT.missRecordStrideInBytes = sizeof(MissRecord);
    quadricShadowSBT.missRecordCount = 1;

    // Random hit
    RaygenRecord *raygenRandomHitRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGRandomHit, raygenRandomHitRecord));
    triangleRandomHitSBT.raygenRecord = (CUdeviceptr)raygenRandomHitRecord;
    triangleRandomHitSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    triangleRandomHitSBT.missRecordStrideInBytes = sizeof(MissRecord);
    triangleRandomHitSBT.missRecordCount = 1;

    bilinearPatchRandomHitSBT.raygenRecord = (CUdeviceptr)raygenRandomHitRecord;
    bilinearPatchRandomHitSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    bilinearPatchRandomHitSBT.missRecordStrideInBytes = sizeof(MissRecord);
    bilinearPatchRandomHitSBT.missRecordCount = 1;

    quadricRandomHitSBT.raygenRecord = (CUdeviceptr)raygenRandomHitRecord;
    quadricRandomHitSBT.missRecordBase = (CUdeviceptr)missNoOpRecord;
    quadricRandomHitSBT.missRecordStrideInBytes = sizeof(MissRecord);
    quadricRandomHitSBT.missRecordCount = 1;

    // Textures
    std::map<std::string, FloatTextureHandle> floatTextures;
    std::map<std::string, SpectrumTextureHandle> spectrumTextures;
    scene.CreateTextures(&floatTextures, &spectrumTextures, alloc, true);

    // Materials
    std::map<std::string, MaterialHandle> namedMaterials;
    std::vector<MaterialHandle> materials;
    scene.CreateMaterials(floatTextures, spectrumTextures, alloc, &namedMaterials,
                          &materials);

    OptixTraversableHandle triangleGASTraversable =
        createGASForTriangles(scene.shapes, hitPGTriangle, anyhitPGShadowTriangle,
                              hitPGRandomHitTriangle,
                              floatTextures, namedMaterials, materials,
                              shapeIndexToAreaLights, &bounds);
    OptixTraversableHandle bilinearPatchGASTraversable =
        createGASForBLPs(scene.shapes, hitPGBilinearPatch, anyhitPGShadowBilinearPatch,
                         hitPGRandomHitBilinearPatch,
                         floatTextures, namedMaterials, materials,
                         shapeIndexToAreaLights, &bounds);
    OptixTraversableHandle quadricGASTraversable =
        createGASForQuadrics(scene.shapes, hitPGQuadric, anyhitPGShadowQuadric,
                             hitPGRandomHitQuadric,
                             floatTextures, namedMaterials, materials,
                             shapeIndexToAreaLights, &bounds);

    pstd::vector<OptixInstance> triangleInstances(alloc), bilinearPatchInstances(alloc);
    pstd::vector<OptixInstance> quadricInstances(alloc);

    OptixInstance gasInstance = {};
    float identity[12] = { 1, 0, 0, 0,
                           0, 1, 0, 0,
                           0, 0, 1, 0 };
    memcpy(gasInstance.transform, identity, 12 * sizeof(float));
    gasInstance.visibilityMask = 255;
    gasInstance.sbtOffset = 0;
    gasInstance.flags = OPTIX_INSTANCE_FLAG_NONE;  // TODO: OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
    if (triangleGASTraversable) {
        gasInstance.traversableHandle = triangleGASTraversable;
        triangleInstances.push_back(gasInstance);
    }
    if (bilinearPatchGASTraversable) {
        gasInstance.traversableHandle = bilinearPatchGASTraversable;
        bilinearPatchInstances.push_back(gasInstance);
    }
    if (quadricGASTraversable) {
        gasInstance.traversableHandle = quadricGASTraversable;
        quadricInstances.push_back(gasInstance);
    }

    // Create GASs for instance definitions
    // TODO: better name here...
    struct Instance {
        OptixTraversableHandle handle;
        Bounds3f bounds;
        int sbtOffset;
    };
    std::map<std::string, Instance> instanceMap;
    for (const auto &def : scene.instanceDefinitions) {
        if (!def.second.animatedShapes.empty())
            Warning("Ignoring %d animated shapes in instance \"%s\".",
                    def.second.animatedShapes.size(), def.first);

        Instance inst;
        inst.sbtOffset = triangleIntersectHGRecords->size();
        inst.handle = createGASForTriangles(def.second.shapes, hitPGTriangle, anyhitPGShadowTriangle,
                                            hitPGRandomHitTriangle,
                                            floatTextures, namedMaterials, materials, {},
                                            &inst.bounds);
        instanceMap[def.first] = inst;
    }

    // Create OptixInstances for instances
    for (const auto &inst : scene.instances) {
        if (instanceMap.find(inst.name) == instanceMap.end())
            ErrorExit(&inst.loc, "%s: object instance not defined.", inst.name);

        if (inst.worldFromInstance == nullptr) {
            Warning(&inst.loc, "%s: object instance has animated transformation. TODO",
                    inst.name);
            continue;
        }

        const Instance &in = instanceMap[inst.name];
        if (!in.handle) {
            //Warning(&inst.loc, "Skipping instance of empty instance definition");
            continue;
        }

        bounds = Union(bounds, (*inst.worldFromInstance)(instanceMap[inst.name].bounds));

        OptixInstance optixInstance = {};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                optixInstance.transform[4*i+j] = inst.worldFromInstance->GetMatrix()[i][j];
        optixInstance.visibilityMask = 255;
        optixInstance.sbtOffset = instanceMap[inst.name].sbtOffset;
        optixInstance.flags = OPTIX_INSTANCE_FLAG_NONE;   // TODO: OPTIX_INSTANCE_FLAG_DISABLE_ANYHIT
        optixInstance.traversableHandle = instanceMap[inst.name].handle;
        triangleInstances.push_back(optixInstance);
    }

    auto buildIAS = [this](const pstd::vector<OptixInstance> &instances) -> OptixTraversableHandle {
        if (instances.empty()) return {};

        // TODO: if instances.size() == 1 -> rootTriangleTraversable = instances[0].traversableHandle
        // But beware sbt offset, visibility mask, and xform...
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        buildInput.instanceArray.instances = CUdeviceptr(instances.data());
        buildInput.instanceArray.numInstances = instances.size();
        std::vector<OptixBuildInput> buildInputs = { buildInput };

        return buildBVH(buildInputs);
    };

    triangleTraversable = buildIAS(triangleInstances);
    bilinearPatchTraversable = buildIAS(bilinearPatchInstances);
    quadricTraversable = buildIAS(quadricInstances);

    if (!scene.animatedShapes.empty())
        Warning("Ignoring %d animated shapes", scene.animatedShapes.size());

    triangleIntersectSBT.hitgroupRecordBase = (CUdeviceptr)triangleIntersectHGRecords->data();
    triangleIntersectSBT.hitgroupRecordStrideInBytes = sizeof(TriangleHitgroupRecord);
    triangleIntersectSBT.hitgroupRecordCount = triangleIntersectHGRecords->size();

    triangleShadowSBT.hitgroupRecordBase = (CUdeviceptr)triangleShadowHGRecords->data();
    triangleShadowSBT.hitgroupRecordStrideInBytes = sizeof(TriangleHitgroupRecord);
    triangleShadowSBT.hitgroupRecordCount = triangleShadowHGRecords->size();

    triangleRandomHitSBT.hitgroupRecordBase = (CUdeviceptr)triangleRandomHitHGRecords->data();
    triangleRandomHitSBT.hitgroupRecordStrideInBytes = sizeof(TriangleHitgroupRecord);
    triangleRandomHitSBT.hitgroupRecordCount = triangleRandomHitHGRecords->size();

    bilinearPatchIntersectSBT.hitgroupRecordBase = (CUdeviceptr)bilinearPatchIntersectHGRecords->data();
    bilinearPatchIntersectSBT.hitgroupRecordStrideInBytes = sizeof(BilinearPatchHitgroupRecord);
    bilinearPatchIntersectSBT.hitgroupRecordCount = bilinearPatchIntersectHGRecords->size();

    bilinearPatchShadowSBT.hitgroupRecordBase = (CUdeviceptr)bilinearPatchShadowHGRecords->data();
    bilinearPatchShadowSBT.hitgroupRecordStrideInBytes = sizeof(BilinearPatchHitgroupRecord);
    bilinearPatchShadowSBT.hitgroupRecordCount = bilinearPatchShadowHGRecords->size();

    bilinearPatchRandomHitSBT.hitgroupRecordBase = (CUdeviceptr)bilinearPatchRandomHitHGRecords->data();
    bilinearPatchRandomHitSBT.hitgroupRecordStrideInBytes = sizeof(BilinearPatchHitgroupRecord);
    bilinearPatchRandomHitSBT.hitgroupRecordCount = bilinearPatchRandomHitHGRecords->size();

    quadricIntersectSBT.hitgroupRecordBase = (CUdeviceptr)quadricIntersectHGRecords->data();
    quadricIntersectSBT.hitgroupRecordStrideInBytes = sizeof(QuadricHitgroupRecord);
    quadricIntersectSBT.hitgroupRecordCount = quadricIntersectHGRecords->size();

    quadricShadowSBT.hitgroupRecordBase = (CUdeviceptr)quadricShadowHGRecords->data();
    quadricShadowSBT.hitgroupRecordStrideInBytes = sizeof(QuadricHitgroupRecord);
    quadricShadowSBT.hitgroupRecordCount = quadricShadowHGRecords->size();

    quadricRandomHitSBT.hitgroupRecordBase = (CUdeviceptr)quadricRandomHitHGRecords->data();
    quadricRandomHitSBT.hitgroupRecordStrideInBytes = sizeof(QuadricHitgroupRecord);
    quadricRandomHitSBT.hitgroupRecordCount = quadricRandomHitHGRecords->size();
}

GPUAccel::ParamBufferState &GPUAccel::getParamBuffer(const RayIntersectParameters &params) const {
    CHECK(nextParamOffset < paramsPool.size());

    ParamBufferState &pbs = paramsPool[nextParamOffset];
    if (++nextParamOffset == paramsPool.size())
        nextParamOffset = 0;
    if (!pbs.used)
        pbs.used = true;
    else
        CUDA_CHECK(cudaEventSynchronize(pbs.finishedEvent));

    CUDA_CHECK(cudaMemcpy((void *)pbs.ptr, &params, sizeof(params), cudaMemcpyHostToDevice));

    return pbs;
}

std::pair<cudaEvent_t, cudaEvent_t> GPUAccel::IntersectClosest(
        int maxRays, const cuda::std::atomic<int> *numActiveRays, const int *rayIndexToPixelIndex,
        const Point3fSOA *rayo, const Vector3fSOA *rayd, Float *tMax,
        SurfaceInteraction *intersections) const {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (triangleTraversable) {
        RayIntersectParameters params;
        params.traversable = triangleTraversable;
        params.numActiveRays = numActiveRays;
        params.rayIndexToPixelIndex = rayIndexToPixelIndex;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.intersections = intersections;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching triangle intersect closest");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &triangleIntersectSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect closest");
#endif
    }

    if (bilinearPatchTraversable) {
        RayIntersectParameters params;
        params.traversable = bilinearPatchTraversable;
        params.numActiveRays = numActiveRays;
        params.rayIndexToPixelIndex = rayIndexToPixelIndex;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.intersections = intersections;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching bilinear patch intersect closest");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &bilinearPatchIntersectSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync bilinear patch intersect closest");
#endif
    }

    if (quadricTraversable) {
        RayIntersectParameters params;
        params.traversable = quadricTraversable;
        params.numActiveRays = numActiveRays;
        params.rayIndexToPixelIndex = rayIndexToPixelIndex;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.intersections = intersections;

        ParamBufferState &pbs = getParamBuffer(params);

#ifndef NDEBUG
        LOG_VERBOSE("Launching quadric intersect closest");
#endif

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &quadricIntersectSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync quadric intersect closest");
#endif
    }
    cudaEventRecord(stop);

    return std::make_pair(start, stop);
};

std::pair<cudaEvent_t, cudaEvent_t> GPUAccel::IntersectShadow(
        int maxRays, const cuda::std::atomic<int> *numActiveRays,
        const Point3fSOA *rayo, const Vector3fSOA *rayd,
        Float *tMax, int *occluded) const {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (triangleTraversable) {
        RayIntersectParameters params;
        params.traversable = triangleTraversable;
        params.numActiveRays = numActiveRays;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.occluded = occluded;

        ParamBufferState &pbs = getParamBuffer(params);

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &triangleShadowSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect shadow");
#endif
    }

    if (bilinearPatchTraversable) {
        RayIntersectParameters params;
        params.traversable = bilinearPatchTraversable;
        params.numActiveRays = numActiveRays;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.occluded = occluded;

        ParamBufferState &pbs = getParamBuffer(params);

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &bilinearPatchShadowSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync bilinear patch intersect shadow");
#endif
    }

    if (quadricTraversable) {
        RayIntersectParameters params;
        params.traversable = quadricTraversable;
        params.numActiveRays = numActiveRays;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.occluded = occluded;

        ParamBufferState &pbs = getParamBuffer(params);

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &quadricShadowSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync quadric intersect shadow");
#endif
    }
    cudaEventRecord(stop);

    return std::make_pair(start, stop);
};

std::pair<cudaEvent_t, cudaEvent_t> GPUAccel::IntersectOneRandom(
        int maxRays, const cuda::std::atomic<int> *numActiveRays, const MaterialHandle *materialHandleArray,
        const Point3fSOA *rayo, const Vector3fSOA *rayd, Float *tMax,
        WeightedReservoirSampler<SurfaceInteraction, Float> *reservoirSamplers) const {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    if (triangleTraversable) {
        RayIntersectParameters params;
        params.traversable = triangleTraversable;
        params.numActiveRays = numActiveRays;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.materialArray = materialHandleArray;
        params.reservoirSamplerArray = reservoirSamplers;

        ParamBufferState &pbs = getParamBuffer(params);

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &triangleRandomHitSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect random");
#endif
    }
    if (bilinearPatchTraversable) {
        RayIntersectParameters params;
        params.traversable = bilinearPatchTraversable;
        params.numActiveRays = numActiveRays;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.materialArray = materialHandleArray;
        params.reservoirSamplerArray = reservoirSamplers;

        ParamBufferState &pbs = getParamBuffer(params);

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &bilinearPatchRandomHitSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync bilinearPatch intersect random");
#endif
    }
    if (quadricTraversable) {
        RayIntersectParameters params;
        params.traversable = quadricTraversable;
        params.numActiveRays = numActiveRays;
        params.rayo = rayo;
        params.rayd = rayd;
        params.tMax = tMax;
        params.materialArray = materialHandleArray;
        params.reservoirSamplerArray = reservoirSamplers;

        ParamBufferState &pbs = getParamBuffer(params);

        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, pbs.ptr,
                                sizeof(RayIntersectParameters), &quadricRandomHitSBT,
                                maxRays, 1, 1));
        CUDA_CHECK(cudaEventRecord(pbs.finishedEvent));

#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync quadric intersect shadow");
#endif
    }

    cudaEventRecord(stop);

    return std::make_pair(start, stop);
}

} // namespace pbrt
