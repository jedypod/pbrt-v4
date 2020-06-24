// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_OPTIX_H
#define PBRT_OPTIX_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/base/texture.h>
#include <pbrt/gpu/pathintegrator.h>  // SOA and *RayIndex stuff
#include <pbrt/interaction.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>

#include <optix.h>

namespace pbrt {

class TriangleMesh;
class BilinearPatchMesh;
template <typename RayIndex>
class RayQueue;

struct TriangleMeshRecord {
    const TriangleMesh *mesh;
    MaterialHandle material;
    FloatTextureHandle alphaTexture;
    LightHandle *areaLights;
    MediumInterface *mediumInterface;
};

struct BilinearMeshRecord {
    const BilinearPatchMesh *mesh;
    MaterialHandle material;
    FloatTextureHandle alphaTexture;
    LightHandle *areaLights;
    MediumInterface *mediumInterface;
};

struct QuadricRecord {
    ShapeHandle shape;
    MaterialHandle material;
    FloatTextureHandle alphaTexture;
    LightHandle areaLight;
    MediumInterface *mediumInterface;
};

struct RayIntersectParameters {
    OptixTraversableHandle traversable;

    // intersection rays
    const RayQueue<PathRayIndex> *pathRays = nullptr;
    TypedIndexSpan<SurfaceInteraction, PixelIndex> intersections;
    TypedIndexSpan<PixelIndex, PathRayIndex> rayIndexToPixelIndex;
    TypedIndexSpan<InteractionType, PathRayIndex> interactionType;

    // shadow rays
    const RayQueue<ShadowRayIndex> *shadowRays = nullptr;
    TypedIndexSpan<SampledSpectrum, ShadowRayIndex> shadowRayLd;
    TypedIndexSpan<SampledSpectrum, ShadowRayIndex> shadowRayPDFUni, shadowRayPDFLight;
    TypedIndexSpan<PixelIndex, ShadowRayIndex> shadowRayIndexToPixelIndex;
    TypedIndexSpan<SampledWavelengths, PixelIndex> lambda;
    TypedIndexSpan<RNG *, PixelIndex> rng;

    // "one random" rays for subsurface...
    const RayQueue<SSRayIndex> *randomHitRays = nullptr;
    TypedIndexSpan<MaterialHandle, SSRayIndex> materials;
    TypedIndexSpan<WeightedReservoirSampler<SurfaceInteraction>, SSRayIndex>
        reservoirSamplers;
};

}  // namespace pbrt

#endif  // PBRT_OPTIX_H
