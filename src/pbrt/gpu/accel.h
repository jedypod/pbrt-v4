// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_GPU_ACCEL_H
#define PBRT_GPU_ACCEL_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu/optix.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/parsedscene.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 2)
  #include <cuda/std/std/atomic>
#else
  #include <cuda/std/atomic>
#endif
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <optix.h>

namespace pbrt {

template <typename RayIndex>
class alignas(128) RayQueue {
  public:
    RayQueue(Allocator alloc, int maxRays)
        : maxRays(maxRays), origins(alloc, maxRays), directions(alloc, maxRays) {
        Float *tm = alloc.allocate_object<Float>(maxRays);  // leaks
        tMax = pstd::MakeSpan(tm, maxRays);
        MediumHandle *mh = alloc.allocate_object<MediumHandle>(maxRays);  // leaks
        mediumHandles = pstd::MakeSpan(mh, maxRays);
    }

    PBRT_GPU
    RayIndex Add(const Ray &ray, Float tm) {
        RayIndex rayIndex(count.fetch_add(1, cuda::std::memory_order_relaxed));
        DCHECK_LT((int)rayIndex, maxRays);
        origins[rayIndex] = ray.o;
        directions[rayIndex] = ray.d;
        tMax[rayIndex] = tm;
        mediumHandles[rayIndex] = ray.medium;
        return rayIndex;
    }

    PBRT_GPU
    void SetRay(RayIndex index, const Ray &ray, Float tm) {
        DCHECK_LT((int)index, maxRays);
        DCHECK_LT((int)index, Size());
        origins[index] = ray.o;
        directions[index] = ray.d;
        tMax[index] = tm;
        mediumHandles[index] = ray.medium;
    }

    PBRT_GPU
    Ray GetRay(RayIndex index, Float *tm = nullptr) const {
        DCHECK_LT((int)index, Size());
        if (tm)
            *tm = tMax[index];
        return Ray(origins[index], directions[index], 0.f /* time */,
                   mediumHandles[index]);
    }

    PBRT_GPU
    void Reset() { count.store(0, cuda::std::memory_order_relaxed); }

    PBRT_GPU
    int Size() const { return count.load(cuda::std::memory_order_relaxed); }

    PBRT_GPU
    void SetTMax(RayIndex index, Float tm) const {
        DCHECK_LT((int)index, Size());
        tMax[index] = tm;
    }

  private:
    cuda::std::atomic<int> count{0};
    int pad[128 - sizeof(int)];  // needed?

    int maxRays;
    Point3fSOA<RayIndex> origins;
    Vector3fSOA<RayIndex> directions;
    mutable TypedIndexSpan<Float, RayIndex> tMax;
    TypedIndexSpan<MediumHandle, RayIndex> mediumHandles;
};

class GPUAccel {
  public:
    GPUAccel(const ParsedScene &scene, Allocator alloc, CUstream cudaStream,
             const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
             const std::map<std::string, MediumHandle> &media);

    Bounds3f Bounds() const { return bounds; }

    std::pair<cudaEvent_t, cudaEvent_t> IntersectClosest(
        const RayQueue<PathRayIndex> *pathRays, int maxRays,
        TypedIndexSpan<SurfaceInteraction, PixelIndex> intersections,
        TypedIndexSpan<PixelIndex, PathRayIndex> rayIndexToPixelIndex,
        TypedIndexSpan<InteractionType, PathRayIndex> interactionType) const;

    std::pair<cudaEvent_t, cudaEvent_t> IntersectShadow(
        const RayQueue<ShadowRayIndex> *rays, int maxRays,
        TypedIndexSpan<SampledSpectrum, ShadowRayIndex> shadowRayTr) const;

    std::pair<cudaEvent_t, cudaEvent_t> IntersectShadowTr(
        const RayQueue<ShadowRayIndex> *rays, int maxRays,
        TypedIndexSpan<SampledSpectrum, ShadowRayIndex> shadowRayTr,
        TypedIndexSpan<PixelIndex, ShadowRayIndex> shadowRayIndexToPixelIndex,
        TypedIndexSpan<SampledWavelengths, PixelIndex> lambda,
        TypedIndexSpan<RNG *, PixelIndex> rng) const;

    std::pair<cudaEvent_t, cudaEvent_t> IntersectOneRandom(
        const RayQueue<SSRayIndex> *rays, int maxRays,
        TypedIndexSpan<MaterialHandle, SSRayIndex> materialHandles,
        TypedIndexSpan<WeightedReservoirSampler<SurfaceInteraction>, SSRayIndex>
            reservoirSamplers) const;

  private:
    struct HitgroupRecord;

    OptixTraversableHandle createGASForTriangles(
        const std::vector<ShapeSceneEntity> &shapes, const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG, const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<std::string, MediumHandle> &media,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForBLPs(
        const std::vector<ShapeSceneEntity> &shapes, const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG, const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<std::string, MediumHandle> &media,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForQuadrics(
        const std::vector<ShapeSceneEntity> &shapes, const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG, const OptixProgramGroup &randomHitPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<std::string, MediumHandle> &media,
        const std::map<int, pstd::vector<LightHandle> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs);

    Allocator alloc;
    Bounds3f bounds;
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

    // TODO: why can't this just be a regular vector rather than a pointer
    // to one?
    pstd::vector<HitgroupRecord> *intersectHGRecords;
    pstd::vector<HitgroupRecord> *shadowHGRecords;
    pstd::vector<HitgroupRecord> *randomHitHGRecords;
    OptixShaderBindingTable intersectSBT = {}, shadowSBT = {}, shadowTrSBT = {};
    OptixShaderBindingTable randomHitSBT = {};
    OptixTraversableHandle rootTraversable = {};
};

}  // namespace pbrt

#endif  // PBRT_GPU_ACCEL_H
