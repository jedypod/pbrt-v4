// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// core/primitive.cpp*
#include <pbrt/cpu/primitive.h>

#include <pbrt/cpu/accelerators.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Primitives", primitiveMemory);

Bounds3f PrimitiveHandle::CameraWorldBound() const {
    switch (Tag()) {
    case TypeIndex<SimplePrimitive>():
        return Cast<SimplePrimitive>()->CameraWorldBound();
    case TypeIndex<GeometricPrimitive>():
        return Cast<GeometricPrimitive>()->CameraWorldBound();
    case TypeIndex<TransformedPrimitive>():
        return Cast<TransformedPrimitive>()->CameraWorldBound();
    case TypeIndex<AnimatedPrimitive>():
        return Cast<AnimatedPrimitive>()->CameraWorldBound();
    case TypeIndex<BVHAccel>():
        return Cast<BVHAccel>()->CameraWorldBound();
    case TypeIndex<KdTreeAccel>():
        return Cast<KdTreeAccel>()->CameraWorldBound();
    default:
        LOG_FATAL("Unhandled primitive type: %d", Tag());
        return {};
    }
}

pstd::optional<ShapeIntersection> PrimitiveHandle::Intersect(const Ray &r,
                                                             Float tMax) const {
    switch (Tag()) {
    case TypeIndex<SimplePrimitive>():
        return Cast<SimplePrimitive>()->Intersect(r, tMax);
    case TypeIndex<GeometricPrimitive>():
        return Cast<GeometricPrimitive>()->Intersect(r, tMax);
    case TypeIndex<TransformedPrimitive>():
        return Cast<TransformedPrimitive>()->Intersect(r, tMax);
    case TypeIndex<AnimatedPrimitive>():
        return Cast<AnimatedPrimitive>()->Intersect(r, tMax);
    case TypeIndex<BVHAccel>():
        return Cast<BVHAccel>()->Intersect(r, tMax);
    case TypeIndex<KdTreeAccel>():
        return Cast<KdTreeAccel>()->Intersect(r, tMax);
    default:
        LOG_FATAL("Unhandled primitive type");
        return {};
    }
}

bool PrimitiveHandle::IntersectP(const Ray &r, Float tMax) const {
    switch (Tag()) {
    case TypeIndex<SimplePrimitive>():
        return Cast<SimplePrimitive>()->IntersectP(r, tMax);
    case TypeIndex<GeometricPrimitive>():
        return Cast<GeometricPrimitive>()->IntersectP(r, tMax);
    case TypeIndex<TransformedPrimitive>():
        return Cast<TransformedPrimitive>()->IntersectP(r, tMax);
    case TypeIndex<AnimatedPrimitive>():
        return Cast<AnimatedPrimitive>()->IntersectP(r, tMax);
    case TypeIndex<BVHAccel>():
        return Cast<BVHAccel>()->IntersectP(r, tMax);
    case TypeIndex<KdTreeAccel>():
        return Cast<KdTreeAccel>()->IntersectP(r, tMax);
    default:
        LOG_FATAL("Unhandled primitive type");
        return {};
    }
}

// TransformedPrimitive Method Definitions
TransformedPrimitive::TransformedPrimitive(PrimitiveHandle primitive,
                                           const Transform *cameraWorldFromPrimitive)
    : primitive(primitive), cameraWorldFromPrimitive(cameraWorldFromPrimitive) {
    primitiveMemory += sizeof(*this);
}

pstd::optional<ShapeIntersection> TransformedPrimitive::Intersect(const Ray &r,
                                                                  Float tMax) const {
    // Compute _ray_ after transformation by _cameraWorldFromPrimitive_
    Ray ray = cameraWorldFromPrimitive->ApplyInverse(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si)
        return {};
    CHECK_LT(si->tHit, 1.001 * tMax);

    // Transform instance's intersection data to world space
    si->intr = (*cameraWorldFromPrimitive)(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool TransformedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = cameraWorldFromPrimitive->ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// AnimatedPrimitive Method Definitions
AnimatedPrimitive::AnimatedPrimitive(PrimitiveHandle p,
                                     const AnimatedTransform &cameraWorldFromPrimitive)
    : primitive(p), cameraWorldFromPrimitive(cameraWorldFromPrimitive) {
    primitiveMemory += sizeof(*this);
    CHECK(cameraWorldFromPrimitive.IsAnimated());
}

pstd::optional<ShapeIntersection> AnimatedPrimitive::Intersect(const Ray &r,
                                                               Float tMax) const {
    // Compute _ray_ after transformation by _cameraWorldFromPrimitive_
    Transform interpCameraWorldFromPrimitive =
        cameraWorldFromPrimitive.Interpolate(r.time);
    Ray ray = interpCameraWorldFromPrimitive.ApplyInverse(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si)
        return {};

    // Transform instance's intersection data to world space
    si->intr = interpCameraWorldFromPrimitive(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool AnimatedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = cameraWorldFromPrimitive.ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// GeometricPrimitive Method Definitions
GeometricPrimitive::GeometricPrimitive(ShapeHandle shape, MaterialHandle material,
                                       LightHandle areaLight,
                                       const MediumInterface &mediumInterface,
                                       FloatTextureHandle alpha)
    : shape(shape),
      material(material),
      areaLight(areaLight),
      mediumInterface(mediumInterface),
      alpha(alpha) {
    primitiveMemory += sizeof(*this);
}

Bounds3f GeometricPrimitive::CameraWorldBound() const {
    return shape.CameraWorldBound();
}

bool GeometricPrimitive::IntersectP(const Ray &r, Float tMax) const {
    if (material && material.IsTransparent())
        return false;
    if (alpha)
        // We need to do a full intersection test to see if the point is
        // alpha-masked.
        // Intersect() checks the regular alpha texture.
        return Intersect(r, tMax).has_value();
    else
        return shape.IntersectP(r, tMax);
}

pstd::optional<ShapeIntersection> GeometricPrimitive::Intersect(const Ray &r,
                                                                Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si)
        return {};
    CHECK_LT(si->tHit, 1.001 * tMax);

    // Test intersection against alpha texture, if present
    if (alpha && alpha.Evaluate(si->intr) == 0) {
        // Ignore this hit and trace a new ray.
        Ray rNext = si->intr.SpawnRay(r.d);
        pstd::optional<ShapeIntersection> siNext = Intersect(rNext, tMax - si->tHit);
        if (siNext)
            // The returned t value has to account for both ray segments.
            siNext->tHit += si->tHit;
        return siNext;
    }

    si->intr.areaLight = areaLight;
    si->intr.material = material;

    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0.);
    // Initialize _SurfaceInteraction::mediumInterface_ after _Shape_
    // intersection
    if (mediumInterface.IsMediumTransition())
        si->intr.mediumInterface = &mediumInterface;
    else
        si->intr.medium = r.medium;

    return si;
}

// SimplePrimitive Method Definitions
SimplePrimitive::SimplePrimitive(ShapeHandle shape, MaterialHandle material)
    : shape(shape), material(material) {
    primitiveMemory += sizeof(*this);
}

Bounds3f SimplePrimitive::CameraWorldBound() const {
    return shape.CameraWorldBound();
}

bool SimplePrimitive::IntersectP(const Ray &r, Float tMax) const {
    if (material && material.IsTransparent())
        return false;
    return shape.IntersectP(r, tMax);
}

pstd::optional<ShapeIntersection> SimplePrimitive::Intersect(const Ray &r,
                                                             Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si)
        return {};

    CHECK_LT(si->tHit, 1.001 * tMax);
    si->intr.areaLight = nullptr;
    si->intr.material = material;

    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0.);
    si->intr.medium = r.medium;
    return si;
}

}  // namespace pbrt
