
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

#ifndef PBRT_SHAPES_CYLINDER_H
#define PBRT_SHAPES_CYLINDER_H

// shapes/cylinder.h*
#include "shape.h"

#include "geometry.h"
#include <memory>

namespace pbrt {

// Cylinder Declarations
class Cylinder : public TransformedShape {
  public:
    // Cylinder Public Methods
    Cylinder(std::shared_ptr<const Transform> ObjectToWorld,
             std::shared_ptr<const Transform> WorldToObject,
             bool reverseOrientation, Float radius, Float zMin, Float zMax,
             Float phiMax)
        : TransformedShape(ObjectToWorld, WorldToObject, reverseOrientation),
          radius(radius),
          zMin(std::min(zMin, zMax)),
          zMax(std::max(zMin, zMax)),
          phiMax(Radians(Clamp(phiMax, 0, 360))) {}
    Bounds3f ObjectBound() const;
    bool Intersect(const Ray &ray, Float *tHit, SurfaceInteraction *isect) const;
    bool IntersectP(const Ray &ray) const;
    Float Area() const;
    Interaction Sample(const Point2f &u, Float *pdf) const;

  protected:
    // Cylinder Private Data
    const Float radius, zMin, zMax, phiMax;
};

std::shared_ptr<Cylinder> CreateCylinderShape(
    std::shared_ptr<const Transform> ObjectToWorld,
    std::shared_ptr<const Transform> WorldToObject, bool reverseOrientation,
    const ParamSet &params);

}  // namespace pbrt

#endif  // PBRT_SHAPES_CYLINDER_H
