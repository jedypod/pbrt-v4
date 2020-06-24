
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

#ifndef PBRT_CORE_SCENE_H
#define PBRT_CORE_SCENE_H

// core/scene.h*
#include <pbrt/pbrt.h>

#include <pbrt/primitive.h>
#include <pbrt/base.h>

#include <memory>

namespace pbrt {

// Scene Declarations
class Scene {
  public:
    // Scene Public Methods
    Scene(PrimitiveHandle a, std::vector<LightHandle> l)
        : lights(std::move(l)), aggregate(std::move(a)) {
        // Scene Constructor Implementation
        if (aggregate)
            worldBound = aggregate.WorldBound();
        for (auto &light : lights) {
            light.Preprocess(worldBound);
            if (light->type == LightType::Infinite)
                infiniteLights.push_back(light);
        }
    }
    const Bounds3f &WorldBound() const { return worldBound; }
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax = Infinity) const;
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;
    pstd::optional<ShapeIntersection> IntersectTr(Ray ray, Float tMax, Sampler &sampler,
                                                  const SampledWavelengths &lambda,
                                                  SampledSpectrum *transmittance) const;

    // Scene Public Data
    std::vector<LightHandle> lights;
    // Store infinite light sources separately for cases where we only want
    // to loop over them.
    std::vector<LightHandle> infiniteLights;

    std::string ToString() const;

  private:
    // Scene Private Data
    PrimitiveHandle aggregate;
    Bounds3f worldBound;
};

}  // namespace pbrt

#endif  // PBRT_CORE_SCENE_H
