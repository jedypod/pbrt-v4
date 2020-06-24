
#include <pbrt/gpu/accel.h>

#include <pbrt/genscene.h>
#include <pbrt/gpu.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/lights.h>
#include <pbrt/loopsubdiv.h>
#include <pbrt/plymesh.h>
#include <pbrt/textures.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/log.h>
#include <pbrt/util/pstd.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>

#define OPTIX_CHECK(EXPR)                                               \
        do {                                                            \
            OptixResult res = EXPR;                                     \
            if (res != OPTIX_SUCCESS)                                   \
                LOG_ERROR("OptiX call " #EXPR " failed with code %d: \"%s\"", \
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
    MeshRecord rec;
};

struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) GPUAccel::ShapeHitgroupRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    ShapeRecord rec;
};

GPUAccel::~GPUAccel() {}

extern "C" { extern const unsigned char PBRT_EMBEDDED_PTX[]; }

OptixTraversableHandle
GPUAccel::buildBVH(const std::vector<OptixBuildInput> &buildInputs) const {
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
    // TODO: are these necessary?
    CUDA_CHECK(cudaDeviceSynchronize());

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

OptixTraversableHandle
GPUAccel::createGASForTriangles(const std::vector<ShapeSceneEntity> &shapes,
                             const OptixProgramGroup &intersectPG,
                             const OptixProgramGroup &shadowPG,
                             const std::map<std::string, FloatTextureHandle> &floatTextures,
                             const std::map<std::string, MaterialHandle> &namedMaterials,
                             const std::vector<MaterialHandle> &materials,
                             const std::map<int, pstd::vector<LightHandle *> *> &shapeIndexToAreaLights,
                             Bounds3f *gasBounds) {
    std::vector<OptixBuildInput> buildInputs;
    std::vector<CUdeviceptr> pDeviceDevicePtrs;
    std::vector<uint32_t> triangleInputFlags;

    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &shape = shapes[shapeIndex];
        if (shape.name == "trianglemesh" || shape.name == "plymesh" ||
            shape.name == "loopsubdiv") {
            OptixBuildInput input = {};
            memset(&input, 0, sizeof(input));

            TriangleMesh *mesh = nullptr;
            if (shape.name == "trianglemesh") {
                mesh = TriangleMesh::Create(shape.worldFromObject, shape.reverseOrientation,
                                            shape.parameters, alloc);
                CHECK(mesh != nullptr);
            } else if (shape.name == "loopsubdiv") {
                mesh = CreateLoopSubdivMesh(shape.worldFromObject, shape.reverseOrientation,
                                            shape.parameters, alloc);
                CHECK(mesh != nullptr);
            } else {
                CHECK_EQ(shape.name, "plymesh");
                std::string filename = ResolveFilename(shape.parameters.GetOneString("plyfile", ""));
                pstd::optional<PLYMesh> plyMesh = ReadPLYMesh(filename); // todo: alloc
                if (!plyMesh)
                    continue;

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

            for (size_t i = 0; i < mesh->nVertices; ++i)
                *gasBounds = Union(*gasBounds, mesh->p[i]);

            pDeviceDevicePtrs.push_back(CUdeviceptr(mesh->p));

            input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

            input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            input.triangleArray.vertexStrideInBytes = sizeof(Point3f);
            input.triangleArray.numVertices = mesh->nVertices;
            // input.triangleArray.vertexBuffers is set later, once we're
            // done and pDeviceDevicePtrs isn't in danger of being
            // realloced...

            input.triangleArray.indexFormat =  OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            input.triangleArray.indexStrideInBytes = 3 *sizeof(int);
            input.triangleArray.numIndexTriplets = mesh->nTriangles;
            input.triangleArray.indexBuffer = CUdeviceptr(mesh->vertexIndices);

            input.triangleArray.numSbtRecords = 1;
            input.triangleArray.sbtIndexOffsetBuffer = CUdeviceptr(nullptr);
            input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
            input.triangleArray.sbtIndexOffsetStrideInBytes = 0;

            buildInputs.push_back(input);

            // Find alpha texture, if present.
            FloatTextureHandle alphaTextureHandle = nullptr;
            std::string alphaTexName = shape.parameters.GetTexture("alpha");
            if (!alphaTexName.empty()) {
                auto iter = floatTextures.find(alphaTexName);
                if (iter == floatTextures.end())
                    ErrorExit(&shape.loc, "%s: alpha texture not defined.", alphaTexName);

                alphaTextureHandle = iter->second;

                if (!BasicTextureEvaluator().Matches({alphaTextureHandle}, {})) {
                    Warning(&shape.loc, "%s: alpha texture too complex for BasicTextureEvaluator "
                            "(need fallback path). Ignoring for now.", alphaTexName);
                    alphaTextureHandle = nullptr;
                }
            }

            // Get Material...
            MaterialHandle materialHandle = nullptr;
            if (!shape.materialName.empty()) {
                auto iter = namedMaterials.find(shape.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&shape.loc, "%s: material not defined", shape.materialName);
                materialHandle = iter->second;
            } else {
                CHECK_NE(shape.materialIndex, -1);
                materialHandle = materials[shape.materialIndex];
            }

            int flags = alphaTextureHandle ? OPTIX_GEOMETRY_FLAG_NONE :
                OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
            triangleInputFlags.push_back(flags);

            TriangleHitgroupRecord hgRecord;
            OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
            hgRecord.rec.mesh = mesh;
            hgRecord.rec.materialHandle = alloc.new_object<MaterialHandle>(materialHandle);
            hgRecord.rec.alphaTextureHandle = alphaTextureHandle ?
                alloc.new_object<FloatTextureHandle>(alphaTextureHandle) : nullptr;
            hgRecord.rec.areaLights = nullptr;
            if (shape.lightIndex != -1) {
                // Note: this will hit if we try to have an instance as an area light.
                auto iter = shapeIndexToAreaLights.find(shapeIndex);
                CHECK(iter != shapeIndexToAreaLights.end());
                // FIXME: needs updating when it's not just tri meshes...
                CHECK_EQ(iter->second->size(), mesh->nTriangles);
                hgRecord.rec.areaLights = iter->second->data();
            }
            triIntersectHGRecords->push_back(hgRecord);

            OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
            triShadowHGRecords->push_back(hgRecord);
        }
    }

    if (buildInputs.empty())
        return {};

    for (size_t i = 0; i < buildInputs.size(); ++i)
        buildInputs[i].triangleArray.vertexBuffers = &pDeviceDevicePtrs[i];

    CHECK_EQ(buildInputs.size(), triangleInputFlags.size());
    // Wire these up now so that the push_backs don't mess up the pointers...
    for (size_t i = 0; i < buildInputs.size(); ++i)
        buildInputs[i].triangleArray.flags = &triangleInputFlags[i];

    return buildBVH(buildInputs);
}

OptixTraversableHandle GPUAccel::createGASForShape(
        const std::string &shapeName,
        const std::vector<ShapeSceneEntity> &shapes,
        const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<int, pstd::vector<LightHandle *> *> &shapeIndexToAreaLights,
        pstd::vector<ShapeHitgroupRecord> *intersectHGRecords,
        pstd::vector<ShapeHitgroupRecord> *shadowHGRecords,
        Bounds3f *gasBounds) {
    std::vector<OptixBuildInput> buildInputs;
    pstd::vector<OptixAabb> shapeAABBs(alloc);
    std::vector<CUdeviceptr> aabbPtrs;
    std::vector<unsigned int> flags;

    for (size_t shapeIndex = 0; shapeIndex < shapes.size(); ++shapeIndex) {
        const auto &shape = shapes[shapeIndex];
        if (shape.name != shapeName)
            continue;

        // TODO: fix so xforms come in in GPU memory
        // (Note that this messes up xform sharing...0
        pstd::vector<ShapeHandle> shapeHandles =
            ShapeHandle::Create(shape.name,
                                alloc.new_object<Transform>(*shape.worldFromObject),
                                alloc.new_object<Transform>(*shape.objectFromWorld),
                                shape.reverseOrientation,
                                shape.parameters, alloc, shape.loc);
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
        FloatTextureHandle alphaTextureHandle = nullptr;
        std::string alphaTexName = shape.parameters.GetTexture("alpha");
        if (!alphaTexName.empty()) {
            auto iter = floatTextures.find(alphaTexName);
            if (iter == floatTextures.end())
                ErrorExit(&shape.loc, "%s: alpha texture not defined.", alphaTexName);

            alphaTextureHandle = iter->second;

            if (!BasicTextureEvaluator().Matches({alphaTextureHandle}, {})) {
                Warning(&shape.loc, "%s: alpha texture too complex for BasicTextureEvaluator "
                        "(need fallback path). Ignoring for now.", alphaTexName);
                alphaTextureHandle = nullptr;
            }
        }
        if (!alphaTextureHandle)
            flags.push_back(OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
        else
            flags.push_back(OPTIX_GEOMETRY_FLAG_NONE);

        // Get Material...
        MaterialHandle materialHandle = nullptr;
        if (!shape.materialName.empty()) {
            auto iter = namedMaterials.find(shape.materialName);
            if (iter == namedMaterials.end())
                ErrorExit(&shape.loc, "%s: material not defined", shape.materialName);
            materialHandle = iter->second;
        } else {
            CHECK_NE(shape.materialIndex, -1);
            materialHandle = materials[shape.materialIndex];
        }

        ShapeHitgroupRecord hgRecord;
        OPTIX_CHECK(optixSbtRecordPackHeader(intersectPG, &hgRecord));
        hgRecord.rec.shapeHandle = alloc.new_object<ShapeHandle>(shapeHandle);
        hgRecord.rec.materialHandle = alloc.new_object<MaterialHandle>(materialHandle);
        hgRecord.rec.alphaTextureHandle = alphaTextureHandle ?
            alloc.new_object<FloatTextureHandle>(alphaTextureHandle) : nullptr;
        hgRecord.rec.areaLight = nullptr;
        if (shape.lightIndex != -1) {
            auto iter = shapeIndexToAreaLights.find(shapeIndex);
            // Note: this will hit if we try to have an instance as an area light.
            CHECK(iter != shapeIndexToAreaLights.end());
            CHECK_EQ(iter->second->size(), 1);
            hgRecord.rec.areaLight = (*iter->second)[0];
        }
        intersectHGRecords->push_back(hgRecord);

        OPTIX_CHECK(optixSbtRecordPackHeader(shadowPG, &hgRecord));
        shadowHGRecords->push_back(hgRecord);
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

    return buildBVH(buildInputs);
}

static CUDAMemoryResource cudaMemoryResource;

GPUAccel::GPUAccel(const GeneralScene &scene, CUstream cudaStream,
                   const std::map<int, pstd::vector<LightHandle *> *> &shapeIndexToAreaLights)
    : alloc(&cudaMemoryResource), cudaStream(cudaStream),
      triIntersectHGRecords(alloc.new_object<pstd::vector<TriangleHitgroupRecord>>(alloc)),
      triShadowHGRecords(alloc.new_object<pstd::vector<TriangleHitgroupRecord>>(alloc)),
      sphereIntersectHGRecords(alloc.new_object<pstd::vector<ShapeHitgroupRecord>>(alloc)),
      sphereShadowHGRecords(alloc.new_object<pstd::vector<ShapeHitgroupRecord>>(alloc)) {
    CUcontext cudaContext;
    CU_CHECK(cuCtxGetCurrent(&cudaContext));
    CHECK(cudaContext != nullptr);

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
    pipelineCompileOptions.pipelineLaunchParamsVariableName =
        "optixLaunchParams";

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

    OptixProgramGroup hitPGSphere;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = optixModule;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__sphere";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__sphere";
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &hitPGSphere));
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
        desc.hitgroup.entryFunctionNameAH = "__anyhit__shadow_triangle";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &anyhitPGShadowTriangle));
        LOG_VERBOSE("%s", log);
    }

    OptixProgramGroup anyhitPGShadowSphere;
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleIS = optixModule;
        desc.hitgroup.entryFunctionNameIS = "__intersection__sphere";
        desc.hitgroup.moduleAH = optixModule;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__sphere";
        OPTIX_CHECK(optixProgramGroupCreate(optixContext, &desc, 1, &pgOptions,
                                            log, &logSize, &anyhitPGShadowSphere));
        LOG_VERBOSE("%s", log);
    }

    // Optix pipeline...
    OptixProgramGroup allPGs[] = { raygenPGClosest, missPGNoOp,
                                   hitPGTriangle, hitPGSphere,
                                   raygenPGShadow, missPGShadow,
                                   anyhitPGShadowTriangle, anyhitPGShadowSphere };
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
    sphereIntersectSBT.raygenRecord = (CUdeviceptr)raygenClosestRecord;

    MissRecord *missRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGNoOp, missRecord));
    triangleIntersectSBT.missRecordBase = (CUdeviceptr)missRecord;
    triangleIntersectSBT.missRecordStrideInBytes = sizeof(MissRecord);
    triangleIntersectSBT.missRecordCount = 1;
    sphereIntersectSBT.missRecordBase = (CUdeviceptr)missRecord;
    sphereIntersectSBT.missRecordStrideInBytes = sizeof(MissRecord);
    sphereIntersectSBT.missRecordCount = 1;

    // Shadow
    RaygenRecord *raygenShadowRecord = alloc.new_object<RaygenRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPGShadow, raygenShadowRecord));
    triangleShadowSBT.raygenRecord = (CUdeviceptr)raygenShadowRecord;
    sphereShadowSBT.raygenRecord = (CUdeviceptr)raygenShadowRecord;

    MissRecord *missShadowRecord = alloc.new_object<MissRecord>();
    OPTIX_CHECK(optixSbtRecordPackHeader(missPGShadow, missShadowRecord));
    triangleShadowSBT.missRecordBase = (CUdeviceptr)missShadowRecord;
    triangleShadowSBT.missRecordStrideInBytes = sizeof(MissRecord);
    triangleShadowSBT.missRecordCount = 1;
    sphereShadowSBT.missRecordBase = (CUdeviceptr)missShadowRecord;
    sphereShadowSBT.missRecordStrideInBytes = sizeof(MissRecord);
    sphereShadowSBT.missRecordCount = 1;

    // Textures
    // TODO: this is copied directly from cpurender.cpp
    std::map<std::string, FloatTextureHandle> floatTextures;
    std::map<std::string, SpectrumTextureHandle> spectrumTextures;
    for (const auto &tex : scene.floatTextures) {
        if (tex.second.worldFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                    "Using start transform.");
        // TODO: Texture could hold a texture pointer...
        Transform worldFromTexture = *tex.second.worldFromObject.startTransform;

        TextureParameterDictionary texDict(&tex.second.parameters, &floatTextures,
                                           &spectrumTextures);
        FloatTextureHandle t = FloatTextureHandle::Create(tex.second.texName, worldFromTexture,
                                                          texDict, alloc, tex.second.loc, true);
        floatTextures[tex.first] = std::move(t);
    }
    for (const auto &tex : scene.spectrumTextures) {
        if (tex.second.worldFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                    "Using start transform.");

        Transform worldFromTexture = *tex.second.worldFromObject.startTransform;

        TextureParameterDictionary texDict(&tex.second.parameters, &floatTextures,
                                           &spectrumTextures);
        SpectrumTextureHandle t = SpectrumTextureHandle::Create(tex.second.texName, worldFromTexture,
                                                                texDict, alloc, tex.second.loc, true);
        spectrumTextures[tex.first] = std::move(t);
    }

    // Materials
    std::map<std::string, MaterialHandle> namedMaterials;
    std::vector<MaterialHandle> materials;
    scene.CreateMaterials(floatTextures, spectrumTextures, alloc, &namedMaterials,
                          &materials);

    OptixTraversableHandle triangleGASTraversable =
        createGASForTriangles(scene.shapes, hitPGTriangle, anyhitPGShadowTriangle,
                              floatTextures, namedMaterials, materials,
                              shapeIndexToAreaLights, &bounds);
    OptixTraversableHandle sphereGASTraversable =
        createGASForShape("sphere", scene.shapes, hitPGSphere, anyhitPGShadowSphere,
                          floatTextures, namedMaterials, materials,
                          shapeIndexToAreaLights, sphereIntersectHGRecords,
                          sphereShadowHGRecords, &bounds);

    // TODO: device mem, right?
    pstd::vector<OptixInstance> triangleInstances(alloc), sphereInstances(alloc);

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
    if (sphereGASTraversable) {
        gasInstance.traversableHandle = sphereGASTraversable;
        sphereInstances.push_back(gasInstance);
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
        if (!def.second.second.empty())
            Warning("Ignoring %d animated shapes in instance \"%s\".",
                    def.second.second.size(), def.first);

        Instance inst;
        inst.sbtOffset = triIntersectHGRecords->size();
        inst.handle = createGASForTriangles(def.second.first, hitPGTriangle, anyhitPGShadowTriangle,
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
    sphereTraversable = buildIAS(sphereInstances);

    if (!scene.animatedShapes.empty())
        Warning("Ignoring %d animated shapes", scene.animatedShapes.size());

    triangleIntersectSBT.hitgroupRecordBase = (CUdeviceptr)triIntersectHGRecords->data();
    triangleIntersectSBT.hitgroupRecordStrideInBytes = sizeof(TriangleHitgroupRecord);
    triangleIntersectSBT.hitgroupRecordCount = triIntersectHGRecords->size();

    triangleShadowSBT.hitgroupRecordBase = (CUdeviceptr)triShadowHGRecords->data();
    triangleShadowSBT.hitgroupRecordStrideInBytes = sizeof(TriangleHitgroupRecord);
    triangleShadowSBT.hitgroupRecordCount = triShadowHGRecords->size();

    sphereIntersectSBT.hitgroupRecordBase = (CUdeviceptr)sphereIntersectHGRecords->data();
    sphereIntersectSBT.hitgroupRecordStrideInBytes = sizeof(ShapeHitgroupRecord);
    sphereIntersectSBT.hitgroupRecordCount = sphereIntersectHGRecords->size();

    sphereShadowSBT.hitgroupRecordBase = (CUdeviceptr)sphereShadowHGRecords->data();
    sphereShadowSBT.hitgroupRecordStrideInBytes = sizeof(ShapeHitgroupRecord);
    sphereShadowSBT.hitgroupRecordCount = sphereShadowHGRecords->size();
}


std::pair<cudaEvent_t, cudaEvent_t> GPUAccel::IntersectClosest(
        int maxRays, const int *numActiveRays, const int *rayIndexToPixelIndex,
        const Point3fSOA *rayo, const Vector3fSOA *rayd,
        SurfaceInteraction *intersections) const {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (triangleTraversable) {
        // Leaks, but whatever.
        // NOTE: could allocate one of these and reuse it for all triangle
        // IntersectClosest() calls...
        LaunchParams *lp = alloc.allocate_object<LaunchParams>();
        lp->traversable = triangleTraversable;
        lp->numActiveRays = numActiveRays;
        lp->rayIndexToPixelIndex = rayIndexToPixelIndex;
        lp->rayo = rayo;
        lp->rayd = rayd;
        lp->intersections = intersections;

#ifndef NDEBUG
        LOG_VERBOSE("Launching triangle intersect closest");
#endif
        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr)lp,
                                sizeof(LaunchParams), &triangleIntersectSBT,
                                maxRays, 1, 1));
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect closest");
#endif
    }
    if (sphereTraversable) {
        LaunchParams *lp = alloc.allocate_object<LaunchParams>();
        lp->traversable = sphereTraversable;
        lp->numActiveRays = numActiveRays;
        lp->rayIndexToPixelIndex = rayIndexToPixelIndex;
        lp->rayo = rayo;
        lp->rayd = rayd;
        lp->intersections = intersections;

#ifndef NDEBUG
        LOG_VERBOSE("Launching sphere intersect closest");
#endif
        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr)lp,
                                sizeof(LaunchParams), &sphereIntersectSBT,
                                maxRays, 1, 1));
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync sphere intersect closest");
#endif
    }
    cudaEventRecord(stop);

    return std::make_pair(start, stop);
};

std::pair<cudaEvent_t, cudaEvent_t> GPUAccel::IntersectShadow(
        int maxRays, const int *numActiveRays,
        const Point3fSOA *rayo, const Vector3fSOA *rayd,
        const Float *tMax, uint8_t *occluded) const {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    if (triangleTraversable) {
        LaunchParams *lp = alloc.allocate_object<LaunchParams>();
        lp->traversable = triangleTraversable;
        lp->numActiveRays = numActiveRays;
        lp->rayo = rayo;
        lp->rayd = rayd;
        lp->tMax = tMax;
        lp->occluded = occluded;

#ifndef NDEBUG
        LOG_VERBOSE("Launching triangle intersect shadow");
#endif
        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr)lp,
                                sizeof(LaunchParams), &triangleShadowSBT,
                                maxRays, 1, 1));
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync triangle intersect shadow");
#endif
    }

    if (sphereTraversable) {
        LaunchParams *lp = alloc.allocate_object<LaunchParams>();
        lp->traversable = sphereTraversable;
        lp->numActiveRays = numActiveRays;
        lp->rayo = rayo;
        lp->rayd = rayd;
        lp->tMax = tMax;
        lp->occluded = occluded;

#ifndef NDEBUG
        LOG_VERBOSE("Launching sphere intersect shadow");
#endif
        OPTIX_CHECK(optixLaunch(optixPipeline, cudaStream, (CUdeviceptr)lp,
                                sizeof(LaunchParams), &sphereShadowSBT,
                                maxRays, 1, 1));
#ifndef NDEBUG
        CUDA_CHECK(cudaDeviceSynchronize());
        LOG_VERBOSE("Post-sync sphere intersect shadow");
#endif
    }
    cudaEventRecord(stop);

    return std::make_pair(start, stop);
};

} // namespace pbrt
