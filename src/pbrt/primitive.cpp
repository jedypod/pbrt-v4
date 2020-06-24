
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


// core/primitive.cpp*
#include <pbrt/primitive.h>

#include <pbrt/base.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/log.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/check.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/taggedptr.h>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Primitives", primitiveMemory);

Primitive::~Primitive() {}

Bounds3f PrimitiveHandle::WorldBound() const {
    switch (Tag()) {
    case TypeIndex<SimplePrimitive>():
        return Cast<SimplePrimitive>()->WorldBound();
    case TypeIndex<GeometricPrimitive>():
        return Cast<GeometricPrimitive>()->WorldBound();
    case TypeIndex<TransformedPrimitive>():
        return Cast<TransformedPrimitive>()->WorldBound();
    case TypeIndex<Primitive>():
        return Cast<Primitive>()->WorldBound();
    default:
        LOG_FATAL("Unhandled primitive type: %d", Tag());
        return {};
    }
}

pstd::optional<ShapeIntersection> PrimitiveHandle::Intersect(const Ray &r, Float tMax) const {
    switch (Tag()) {
    case TypeIndex<SimplePrimitive>():
        return Cast<SimplePrimitive>()->Intersect(r, tMax);
    case TypeIndex<GeometricPrimitive>():
        return Cast<GeometricPrimitive>()->Intersect(r, tMax);
    case TypeIndex<TransformedPrimitive>():
        return Cast<TransformedPrimitive>()->Intersect(r, tMax);
    case TypeIndex<Primitive>():
        return Cast<Primitive>()->Intersect(r, tMax);
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
    case TypeIndex<Primitive>():
        return Cast<Primitive>()->IntersectP(r, tMax);
    default:
        LOG_FATAL("Unhandled primitive type");
        return {};
    }
}

// TransformedPrimitive Method Definitions
TransformedPrimitive::TransformedPrimitive(PrimitiveHandle primitive,
                                           const Transform *worldFromPrimitive)
    : primitive(primitive), worldFromPrimitive(worldFromPrimitive) {
    primitiveMemory += sizeof(*this);
}

pstd::optional<ShapeIntersection> TransformedPrimitive::Intersect(const Ray &r,
                                                            Float tMax) const {
    // Compute _ray_ after transformation by _worldFromPrimitive_
    Ray ray = Inverse(*worldFromPrimitive)(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si) return {};
    CHECK_LT(si->tHit, 1.001 * tMax);

    // Transform instance's intersection data to world space
    si->intr = (*worldFromPrimitive)(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool TransformedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = worldFromPrimitive->ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// AnimatedPrimitive Method Definitions
AnimatedPrimitive::AnimatedPrimitive(PrimitiveHandle p,
                                     const AnimatedTransform &worldFromPrimitive)
    : primitive(p), worldFromPrimitive(worldFromPrimitive) {
    primitiveMemory += sizeof(*this);
    CHECK(worldFromPrimitive.IsAnimated());
}

pstd::optional<ShapeIntersection> AnimatedPrimitive::Intersect(const Ray &r,
                                                         Float tMax) const {
    // Compute _ray_ after transformation by _worldFromPrimitive_
    Transform interpWorldFromPrimitive = worldFromPrimitive.Interpolate(r.time);
    Ray ray = Inverse(interpWorldFromPrimitive)(r, &tMax);
    pstd::optional<ShapeIntersection> si = primitive.Intersect(ray, tMax);
    if (!si) return {};

    // Transform instance's intersection data to world space
    if (!interpWorldFromPrimitive.IsIdentity())
        si->intr = interpWorldFromPrimitive(si->intr);
    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0);
    return si;
}

bool AnimatedPrimitive::IntersectP(const Ray &r, Float tMax) const {
    Ray ray = worldFromPrimitive.ApplyInverse(r, &tMax);
    return primitive.IntersectP(ray, tMax);
}

// GeometricPrimitive Method Definitions
GeometricPrimitive::GeometricPrimitive(ShapeHandle shape,
                                       MaterialHandle material,
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

GeometricPrimitive::~GeometricPrimitive() {}

Bounds3f GeometricPrimitive::WorldBound() const { return shape.WorldBound(); }

bool GeometricPrimitive::IntersectP(const Ray &r, Float tMax) const {
    if (material && material.IsTransparent()) return false;
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
    if (!si) return {};
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

SimplePrimitive::~SimplePrimitive() {}

Bounds3f SimplePrimitive::WorldBound() const { return shape.WorldBound(); }

bool SimplePrimitive::IntersectP(const Ray &r, Float tMax) const {
    if (material && material.IsTransparent()) return false;
    return shape.IntersectP(r, tMax);
}

pstd::optional<ShapeIntersection> SimplePrimitive::Intersect(const Ray &r,
                                                             Float tMax) const {
    pstd::optional<ShapeIntersection> si = shape.Intersect(r, tMax);
    if (!si) return {};

    CHECK_LT(si->tHit, 1.001 * tMax);
    si->intr.areaLight = nullptr;
    si->intr.material = material;

    CHECK_GE(Dot(si->intr.n, si->intr.shading.n), 0.);
    si->intr.medium = r.medium;
    return si;
}

}  // namespace pbrt
