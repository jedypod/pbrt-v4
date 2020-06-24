
#ifndef PBRT_GPU_ACCEL_H
#define PBRT_GPU_ACCEL_H

#include <pbrt/pbrt.h>

#include <pbrt/genscene.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/util/pstd.h>

#include <utility>
#include <map>
#include <string>

#include <cuda_runtime.h>
#include <optix.h>

namespace pbrt {

class Point3fSOA;
class Vector3fSOA;

class GPUAccel {
 public:
    GPUAccel(const GeneralScene &scene, CUstream cudaStream,
             const std::map<int, pstd::vector<LightHandle *> *> &shapeIndexToAreaLights);
    ~GPUAccel();

    Bounds3f Bounds() const { return bounds; }

    std::pair<cudaEvent_t, cudaEvent_t> IntersectClosest(
        int maxRays, const int *numActiveRays, const int *rayIndexToPixelIndex,
        const Point3fSOA *rayo, const Vector3fSOA *rayd,
        SurfaceInteraction *intersections) const;

    std::pair<cudaEvent_t, cudaEvent_t> IntersectShadow(
        int mayRays, const int *numActiveRays,
        const Point3fSOA *rayo, const Vector3fSOA *rayd,
        const Float *tMax, uint8_t *occluded) const;

 private:
    struct TriangleHitgroupRecord;
    struct ShapeHitgroupRecord;

    OptixTraversableHandle createGASForTriangles(
        const std::vector<ShapeSceneEntity> &shapes,
        const OptixProgramGroup &intersectPG,
        const OptixProgramGroup &shadowPG,
        const std::map<std::string, FloatTextureHandle> &floatTextures,
        const std::map<std::string, MaterialHandle> &namedMaterials,
        const std::vector<MaterialHandle> &materials,
        const std::map<int, pstd::vector<LightHandle *> *> &shapeIndexToAreaLights,
        Bounds3f *gasBounds);

    OptixTraversableHandle createGASForShape(
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
        Bounds3f *gasBounds);

    OptixTraversableHandle buildBVH(const std::vector<OptixBuildInput> &buildInputs) const;

    Bounds3f bounds;
    mutable pstd::pmr::polymorphic_allocator<pstd::byte> alloc;
    CUstream cudaStream;
    OptixDeviceContext optixContext;
    OptixModule optixModule;
    OptixPipeline optixPipeline;

    pstd::vector<TriangleHitgroupRecord> *triIntersectHGRecords, *triShadowHGRecords;
    OptixShaderBindingTable triangleIntersectSBT = {}, triangleShadowSBT = {};
    OptixTraversableHandle triangleTraversable = {};

    // TODO: put these in a little struct and keep a map<string /* shape
    // name */, little struct>.  Or maybe just keep a vector--not sure the
    // name association needs to be maintained...
    pstd::vector<ShapeHitgroupRecord> *sphereIntersectHGRecords, *sphereShadowHGRecords;
    OptixShaderBindingTable sphereIntersectSBT = {}, sphereShadowSBT = {};
    OptixTraversableHandle sphereTraversable = {};
};

} // namespace pbrt

#endif // PBRT_GPU_ACCEL_H
