// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_PRIMITIVE_H
#define PBRT_CORE_PRIMITIVE_H

// core/primitive.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/shape.h>
#include <pbrt/base/texture.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>

#include <memory>

namespace pbrt {

class SimplePrimitive;
class GeometricPrimitive;
class TransformedPrimitive;
class AnimatedPrimitive;
class BVHAccel;
class KdTreeAccel;

class PrimitiveHandle
    : public TaggedPointer<SimplePrimitive, GeometricPrimitive, TransformedPrimitive,
                           AnimatedPrimitive, BVHAccel, KdTreeAccel> {
  public:
    using TaggedPointer::TaggedPointer;

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &r, Float tMax = Infinity) const;
};

// GeometricPrimitive Declarations
class alignas(8) GeometricPrimitive {
  public:
    // GeometricPrimitive Public Methods
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    GeometricPrimitive(ShapeHandle shape, MaterialHandle material, LightHandle areaLight,
                       const MediumInterface &mediumInterface,
                       FloatTextureHandle alpha = nullptr);

  private:
    // GeometricPrimitive Private Data
    ShapeHandle shape;
    MaterialHandle material;
    LightHandle areaLight;
    MediumInterface mediumInterface;
    FloatTextureHandle alpha;
};

// SimplePrimitive Declarations
// More compact representation for the common case of only a shape and
// a material, with everything else null.
class alignas(8) SimplePrimitive {
  public:
    // SimplePrimitive Public Methods
    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    SimplePrimitive(ShapeHandle shape, MaterialHandle material);

  private:
    // SimplePrimitive Private Data
    ShapeHandle shape;
    MaterialHandle material;
};

// TransformedPrimitive Declarations
class alignas(8) TransformedPrimitive {
  public:
    // TransformedPrimitive Public Methods
    TransformedPrimitive(PrimitiveHandle primitive,
                         const Transform *renderFromPrimitive);
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    Bounds3f Bounds() const {
        return (*renderFromPrimitive)(primitive.Bounds());
    }

  private:
    // TransformedPrimitive Private Data
    PrimitiveHandle primitive;
    const Transform *renderFromPrimitive;
};

// AnimatedPrimitive Declarations
class alignas(8) AnimatedPrimitive {
  public:
    // AnimatedPrimitive Public Methods
    AnimatedPrimitive(PrimitiveHandle primitive,
                      const AnimatedTransform &renderFromPrimitive);
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    Bounds3f Bounds() const {
        return renderFromPrimitive.MotionBounds(primitive.Bounds());
    }

  private:
    // AnimatedPrimitive Private Data
    PrimitiveHandle primitive;
    AnimatedTransform renderFromPrimitive;
};

}  // namespace pbrt

#endif  // PBRT_CORE_PRIMITIVE_H
