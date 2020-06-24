
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


// core/scene.cpp*
#include <pbrt/scene.h>

#include <pbrt/shapes.h>
#include <pbrt/util/check.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

STAT_COUNTER("Intersections/Regular ray intersection tests",
             nIntersectionTests);
STAT_COUNTER("Intersections/Shadow ray intersection tests", nShadowTests);

// Scene Method Definitions
pstd::optional<ShapeIntersection> Scene::Intersect(const Ray &ray, Float tMax) const {
    ++nIntersectionTests;
    DCHECK_NE(ray.d, Vector3f(0,0,0));
    if (aggregate)
        return aggregate.Intersect(ray, tMax);
    else
        return {};
}

bool Scene::IntersectP(const Ray &ray, Float tMax) const {
    ++nShadowTests;
    DCHECK_NE(ray.d, Vector3f(0,0,0));
    if (aggregate)
        return aggregate.IntersectP(ray, tMax);
    else
        return false;
}

pstd::optional<ShapeIntersection> Scene::IntersectTr(Ray ray, Float tMax, SamplerHandle sampler,
                                                     const SampledWavelengths &lambda,
                                                     SampledSpectrum *Tr) const {
    *Tr = SampledSpectrum(1.f);
    while (true) {
        pstd::optional<ShapeIntersection> si = Intersect(ray, tMax);
        // Accumulate beam transmittance for ray segment
        if (ray.medium != nullptr) *Tr *= ray.medium->Tr(ray, si ? si->tHit : tMax, lambda, sampler);

        // Initialize next ray segment or terminate transmittance computation
        if (!si) return {};
        if (si->intr.material) return si;

        ray = si->intr.SpawnRay(ray.d);
        tMax -= si->tHit;
    }
}

std::string Scene::ToString() const {
    std::string s = StringPrintf("[ Scene aggregate: %s worldBound: %s lights[%d]: [ ",
                                 aggregate, worldBound, lights.size());
    for (const auto &l : lights)
        s += StringPrintf("%s, ", l.ToString());
    s += StringPrintf("] infiniteLights[%d]: [ ", infiniteLights.size());
    for (const auto &l : infiniteLights)
        s += StringPrintf("%s, ", l.ToString());
    return s + " ]";
}

}  // namespace pbrt
