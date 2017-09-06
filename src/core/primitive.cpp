
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
#include "primitive.h"

#include "util/stats.h"
#include "util/geometry.h"
#include "interaction.h"
#include "shape.h"
#include <glog/logging.h>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Primitives", primitiveMemory);

// Primitive Method Definitions
Primitive::~Primitive() {}
const AreaLight *Aggregate::GetAreaLight() const {
    LOG(FATAL) <<
        "Aggregate::GetAreaLight() method"
        "called; should have gone to GeometricPrimitive";
    return nullptr;
}

const Material *Aggregate::GetMaterial() const {
    LOG(FATAL) <<
        "Aggregate::GetMaterial() method"
        "called; should have gone to GeometricPrimitive";
    return nullptr;
}

const ParamSet *Aggregate::GetAttributes() const {
    LOG(FATAL) << "Aggregate::GetAttributes() method called; should have gone "
        "to GeometricPrimitive";
    return nullptr;
}

void Aggregate::ComputeScatteringFunctions(SurfaceInteraction *isect,
                                           MemoryArena &arena,
                                           TransportMode mode) const {
    LOG(FATAL) <<
        "Aggregate::ComputeScatteringFunctions() method"
        "called; should have gone to GeometricPrimitive";
}

// TransformedPrimitive Method Definitions
TransformedPrimitive::TransformedPrimitive(std::shared_ptr<Primitive> &primitive,
                                           const AnimatedTransform &PrimitiveToWorld)
    : primitive(primitive), PrimitiveToWorld(PrimitiveToWorld) {
    primitiveMemory += sizeof(*this);
}

bool TransformedPrimitive::Intersect(const Ray &r,
                                     SurfaceInteraction *isect) const {
    // Compute _ray_ after transformation by _PrimitiveToWorld_
    Transform InterpolatedPrimToWorld = PrimitiveToWorld.Interpolate(r.time);
    Ray ray = Inverse(InterpolatedPrimToWorld)(r);
    if (!primitive->Intersect(ray, isect)) return false;
    r.tMax = ray.tMax;
    // Transform instance's intersection data to world space
    if (!InterpolatedPrimToWorld.IsIdentity())
        *isect = InterpolatedPrimToWorld(*isect);
    CHECK_GE(Dot(isect->n, isect->shading.n), 0);
    return true;
}

bool TransformedPrimitive::IntersectP(const Ray &r) const {
    Transform InterpolatedPrimToWorld = PrimitiveToWorld.Interpolate(r.time);
    Transform InterpolatedWorldToPrim = Inverse(InterpolatedPrimToWorld);
    return primitive->IntersectP(InterpolatedWorldToPrim(r));
}

const ParamSet *TransformedPrimitive::GetAttributes() const {
    return primitive->GetAttributes();
}

// GeometricPrimitive Method Definitions
GeometricPrimitive::GeometricPrimitive(const std::shared_ptr<Shape> &shape,
                                       const std::shared_ptr<Material> &material,
                                       const std::shared_ptr<AreaLight> &areaLight,
                                       const MediumInterface &mediumInterface,
                                       const std::shared_ptr<Texture<Float>> &alpha,
                                       const std::shared_ptr<Texture<Float>> &shadowAlpha)
    : shape(shape),
      material(material),
      areaLight(areaLight),
      mediumInterface(mediumInterface),
      alpha(alpha),
      shadowAlpha(shadowAlpha) {
    primitiveMemory += sizeof(*this);
}

Bounds3f GeometricPrimitive::WorldBound() const { return shape->WorldBound(); }

bool GeometricPrimitive::IntersectP(const Ray &r) const {
    if (alpha || shadowAlpha) {
        // We need to do a full intersection test to see if the point is
        // alpha-masked.
        SurfaceInteraction isect;
        // Intersect() checks the regular alpha texture.
        if (!Intersect(r, &isect)) return false;

        // But what it thinks is an intersection may be masked by the
        // shadow alpha texture..
        if (shadowAlpha && shadowAlpha->Evaluate(isect) == 0)
            return IntersectP(isect.SpawnRay(r.d));
        else
            return true;
    } else
        return shape->IntersectP(r);
}

bool GeometricPrimitive::Intersect(const Ray &r,
                                   SurfaceInteraction *isectOut) const {
    Float tHit;
    SurfaceInteraction isect;
    if (!shape->Intersect(r, &tHit, &isect)) return false;

    // Test intersection against alpha texture, if present
    if (alpha && alpha->Evaluate(isect) == 0) {
        // Ignore this hit and trace a new ray.
        Ray rNext = isect.SpawnRay(r.d);
        if (Intersect(rNext, isectOut)) {
            // The returned t value has to account for both ray segments.
            r.tMax = tHit + rNext.tMax;
            return true;
        } else
            return false;
    }

    r.tMax = tHit;
    isect.primitive = this;
    CHECK_GE(Dot(isect.n, isect.shading.n), 0.);
    // Initialize _SurfaceInteraction::mediumInterface_ after _Shape_
    // intersection
    if (mediumInterface.IsMediumTransition())
        isect.mediumInterface = mediumInterface;
    else
        isect.mediumInterface = MediumInterface(r.medium);
    *isectOut = isect;
    return true;
}

const AreaLight *GeometricPrimitive::GetAreaLight() const {
    return areaLight.get();
}

const Material *GeometricPrimitive::GetMaterial() const {
    return material.get();
}

const ParamSet *GeometricPrimitive::GetAttributes() const {
    return shape->GetAttributes();
}

void GeometricPrimitive::ComputeScatteringFunctions(
    SurfaceInteraction *isect, MemoryArena &arena, TransportMode mode) const {
    ProfilePhase p(Prof::ComputeScatteringFuncs);
    if (material)
        material->ComputeScatteringFunctions(isect, arena, mode);
    CHECK_GE(Dot(isect->n, isect->shading.n), 0.);
}

// SimplePrimitive Method Definitions
SimplePrimitive::SimplePrimitive(const std::shared_ptr<Shape> &shape,
                                 const std::shared_ptr<Material> &material)
    : shape(shape),
      material(material) {
    primitiveMemory += sizeof(*this);
}

Bounds3f SimplePrimitive::WorldBound() const { return shape->WorldBound(); }

bool SimplePrimitive::IntersectP(const Ray &r) const {
    return shape->IntersectP(r);
}

bool SimplePrimitive::Intersect(const Ray &r,
                                SurfaceInteraction *isectOut) const {
    Float tHit;
    SurfaceInteraction isect;
    if (!shape->Intersect(r, &tHit, &isect)) return false;

    r.tMax = tHit;
    isect.primitive = this;
    CHECK_GE(Dot(isect.n, isect.shading.n), 0.);
    isect.mediumInterface = MediumInterface(r.medium);
    *isectOut = isect;
    return true;
}

const AreaLight *SimplePrimitive::GetAreaLight() const {
    return nullptr;
}

const Material *SimplePrimitive::GetMaterial() const {
    return material.get();
}

const ParamSet *SimplePrimitive::GetAttributes() const {
    return shape->GetAttributes();
}

void SimplePrimitive::ComputeScatteringFunctions(
    SurfaceInteraction *isect, MemoryArena &arena, TransportMode mode) const {
    ProfilePhase p(Prof::ComputeScatteringFuncs);
    material->ComputeScatteringFunctions(isect, arena, mode);
    CHECK_GE(Dot(isect->n, isect->shading.n), 0.);
}

}  // namespace pbrt
