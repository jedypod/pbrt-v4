
#ifndef PBRT_GPU_ACCEL_H
#define PBRT_GPU_ACCEL_H

#include <pbrt/pbrt.h>

#include <pbrt/genscene.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/util/pstd.h>

#include <utility>
#include <map>
#include <string>
#include <vector>
#include <cuda/std/atomic>

#include <cuda_runtime.h>
#include <optix.h>

namespace pbrt {

class Point3fSOA;
class Vector3fSOA;

class GPUAccel {
 public:
    GPUAccel(const GeneralScene &scene, CUstream cudaStream,
             const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights);
    ~GPUAccel();

    Bounds3f Bounds() const { return bounds; }

    std::pair<cudaEvent_t, cudaEvent_t> IntersectClosest(
        int maxRays, const cuda::std::atomic<int> *numActiveRays, const int *rayIndexToPixelIndex,
        const Point3fSOA *rayo, const Vector3fSOA *rayd, Float *tMax,
        SurfaceInteraction *intersections) const;

    std::pair<cudaEvent_t, cudaEvent_t> IntersectShadow(
        int maxRays, const cuda::std::atomic<int> *numActiveRays,
        const Point3fSOA *rayo, const Vector3fSOA *rayd,
        Float *tMax, int *occluded) const;

    std::pair<cudaEvent_t, cudaEvent_t> IntersectOneRandom(
        int maxRays, const cuda::std::atomic<int> *numActiveRays, const MaterialHandle *materialHandleArray,
        const Point3fSOA *rayo, const Vector3fSOA *rayd, Float *tMax,
        WeightedReservoirSampler<SurfaceInteraction, Float> *reservoirSamplers) const;

 private:
    struct TriangleHitgroupRecord;
    struct BilinearPatchHitgroupRecord;
    struct QuadricHitgroupRecord;

    OptixTraversableHandle createGASForTriangles(
        const std::vector<ShapeSceneEntity> &shapes,
        const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForBLPs(
        const std::vector<ShapeSceneEntity> &shapes,
        const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForQuadrics(
        const std::vector<ShapeSceneEntity> &shapes,
        const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG,
        const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs) const;

    Bounds3f bounds;
    mutable pstd::pmr::polymorphic_allocator<pstd::byte> alloc;
    CUstream cudaStream;
    OptixDeviceContext optixContext;
    OptixModule optixModule;
    OptixPipeline optixPipeline;

    struct ParamBufferState {
        bool used = false;
        cudaEvent_t finishedEvent;
        CUdeviceptr ptr = 0;
    };
    mutable std::vector<ParamBufferState> paramsPool;
    mutable size_t nextParamOffset = 0;

    ParamBufferState &getParamBuffer(const RayIntersectParameters &) const;

    pstd::vector<TriangleHitgroupRecord> *triangleIntersectHGRecords;
    pstd::vector<TriangleHitgroupRecord> *triangleShadowHGRecords;
    pstd::vector<TriangleHitgroupRecord> *triangleRandomHitHGRecords;
    OptixShaderBindingTable triangleIntersectSBT = {}, triangleShadowSBT = {};
    OptixShaderBindingTable triangleRandomHitSBT = {};
    OptixTraversableHandle triangleTraversable = {};

    pstd::vector<BilinearPatchHitgroupRecord> *bilinearPatchIntersectHGRecords;
    pstd::vector<BilinearPatchHitgroupRecord> *bilinearPatchShadowHGRecords;
    pstd::vector<BilinearPatchHitgroupRecord> *bilinearPatchRandomHitHGRecords;
    OptixShaderBindingTable bilinearPatchIntersectSBT = {}, bilinearPatchShadowSBT = {};
    OptixShaderBindingTable bilinearPatchRandomHitSBT = {};
    OptixTraversableHandle bilinearPatchTraversable = {};

    pstd::vector<QuadricHitgroupRecord> *quadricIntersectHGRecords;
    pstd::vector<QuadricHitgroupRecord> *quadricShadowHGRecords;
    pstd::vector<QuadricHitgroupRecord> *quadricRandomHitHGRecords;
    OptixShaderBindingTable quadricIntersectSBT = {}, quadricShadowSBT = {};
    OptixShaderBindingTable quadricRandomHitSBT = {};
    OptixTraversableHandle quadricTraversable = {};
};

} // namespace pbrt

#endif // PBRT_GPU_ACCEL_H
