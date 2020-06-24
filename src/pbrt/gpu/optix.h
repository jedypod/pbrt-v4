
#ifndef PBRT_OPTIX_H
#define PBRT_OPTIX_H

#include <pbrt/pbrt.h>

#include <pbrt/gpu.h>
#include <pbrt/base.h>
#include <pbrt/shapes.h>

#include <optix.h>

namespace pbrt {

struct MeshRecord {
    const TriangleMesh *mesh;
    MaterialHandle *materialHandle;
    FloatTextureHandle *alphaTextureHandle;
    LightHandle **areaLights;
};

struct ShapeRecord {
    const ShapeHandle *shapeHandle;
    MaterialHandle *materialHandle;
    FloatTextureHandle *alphaTextureHandle;
    LightHandle *areaLight;
};

struct LaunchParams {
    OptixTraversableHandle traversable;

    const int *numActiveRays;
    const Point3fSOA *rayo;
    const Vector3fSOA *rayd;

    // intersection rays
    const int *rayIndexToPixelIndex;
    SurfaceInteraction *intersections;

    // shadow rays only
    const float *tMax;
    uint8_t *occluded;
};

} // namespace pbrt

#endif // PBRT_OPTIX_H
