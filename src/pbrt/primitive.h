
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_PRIMITIVE_H
#define PBRT_CORE_PRIMITIVE_H

// core/primitive.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/transform.h>
#include <pbrt/util/taggedptr.h>

#include <memory>

namespace pbrt {

class SimplePrimitive;
class GeometricPrimitive;
class TransformedPrimitive;
class Primitive;

class PrimitiveHandle : public TaggedPointer<SimplePrimitive, GeometricPrimitive,
                                              TransformedPrimitive, Primitive> {
public:
    using TaggedPointer::TaggedPointer;
    PrimitiveHandle() = default;
    PrimitiveHandle(TaggedPointer<SimplePrimitive, GeometricPrimitive,
                                   TransformedPrimitive, Primitive> tp)
        : TaggedPointer(tp) { }

    Bounds3f WorldBound() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax = Infinity) const;
    bool IntersectP(const Ray &r, Float tMax = Infinity) const;
};

// GeometricPrimitive Declarations
// NOTE: doesn't inherit from Primitive!!
class GeometricPrimitive {
  public:
    // GeometricPrimitive Public Methods
    Bounds3f WorldBound() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    GeometricPrimitive(ShapeHandle shape,
                       MaterialHandle material,
                       LightHandle areaLight,
                       const MediumInterface &mediumInterface,
                       FloatTextureHandle alpha = nullptr);
    ~GeometricPrimitive();

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
// NOTE: doesn't inherit from Primitive!!
class SimplePrimitive {
  public:
    // SimplePrimitive Public Methods
    Bounds3f WorldBound() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    SimplePrimitive(ShapeHandle shape, MaterialHandle material);
    ~SimplePrimitive();

  private:
    // SimplePrimitive Private Data
    ShapeHandle shape;
    MaterialHandle material;
};

// TransformedPrimitive Declarations
class TransformedPrimitive {
  public:
    // TransformedPrimitive Public Methods
    TransformedPrimitive(PrimitiveHandle primitive,
                         const Transform *worldFromPrimitive);
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    Bounds3f WorldBound() const {
        return (*worldFromPrimitive)(primitive.WorldBound());
    }

  private:
    // TransformedPrimitive Private Data
    PrimitiveHandle primitive;
    const Transform *worldFromPrimitive;
};

// AnimatedPrimitive Declarations
class AnimatedPrimitive : public Primitive {
  public:
    // AnimatedPrimitive Public Methods
    AnimatedPrimitive(PrimitiveHandle primitive,
                      const AnimatedTransform &worldFromPrimitive);
    pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax) const;
    bool IntersectP(const Ray &r, Float tMax) const;
    Bounds3f WorldBound() const {
        return worldFromPrimitive.MotionBounds(primitive.WorldBound());
    }

  private:
    // AnimatedPrimitive Private Data
    PrimitiveHandle primitive;
    AnimatedTransform worldFromPrimitive;
};

}  // namespace pbrt

#endif  // PBRT_CORE_PRIMITIVE_H
