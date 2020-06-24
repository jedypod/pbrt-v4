
#ifndef PBRT_OPTIX_H
#define PBRT_OPTIX_H

#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/gpu.h>
#include <pbrt/interaction.h>
#include <pbrt/util/sampling.h>

#include <cuda/std/atomic>

#include <optix.h>

namespace pbrt {

class TriangleMesh;
class BilinearPatchMesh;

struct TriangleMeshRecord {
    const TriangleMesh *mesh;
    MaterialHandle material;
    FloatTextureHandle alphaTexture;
    LightHandle *areaLights;
};

struct BilinearMeshRecord {
    const BilinearPatchMesh *mesh;
    MaterialHandle material;
    FloatTextureHandle alphaTexture;
    LightHandle *areaLights;
};

struct QuadricRecord {
    ShapeHandle shape;
    MaterialHandle material;
    FloatTextureHandle alphaTexture;
    LightHandle areaLight;
};

struct RayIntersectParameters {
    OptixTraversableHandle traversable;

    const cuda::std::atomic<int> *numActiveRays;
    const Point3fSOA *rayo;
    const Vector3fSOA *rayd;

    // akk types of rays
    float *tMax;

    // intersection rays
    const int *rayIndexToPixelIndex;
    SurfaceInteraction *intersections;

    // shadow rays
    int *occluded;

    // "one random" rays for subsurface...
    const MaterialHandle *materialArray;
    WeightedReservoirSampler<SurfaceInteraction, Float> *reservoirSamplerArray;
};

} // namespace pbrt

#endif // PBRT_OPTIX_H
