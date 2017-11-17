
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

#ifndef PBRT_INTEGRATORS_VOLPATH_H
#define PBRT_INTEGRATORS_VOLPATH_H

// integrators/volpath.h*
#include <pbrt/core/pbrt.h>
#include <pbrt/core/integrator.h>
#include <pbrt/core/lightdistrib.h>

#include <memory>

namespace pbrt {

// VolPathIntegrator Declarations
class VolPathIntegrator : public SamplerIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, const Scene &scene,
                      std::shared_ptr<const Camera> camera,
                      std::unique_ptr<Sampler> sampler,
                      const Bounds2i &pixelBounds, Float rrThreshold = 1,
                      const std::string &lightSampleStrategy = "spatial")
        : SamplerIntegrator(scene, camera, std::move(sampler), pixelBounds),
          maxDepth(maxDepth),
          rrThreshold(rrThreshold) {
              lightDistribution =
                  CreateLightSampleDistribution(lightSampleStrategy, scene);
          }

    Spectrum Li(const RayDifferential &ray, Sampler &sampler,
                MemoryArena &arena, int depth) const;

    void AfterWave() { lightDistribution->AfterWave(); }

  private:
    // VolPathIntegrator Private Data
    const int maxDepth;
    const Float rrThreshold;
    std::unique_ptr<LightDistribution> lightDistribution;
};

std::unique_ptr<VolPathIntegrator> CreateVolPathIntegrator(
    const ParamSet &params, const Scene &scene,
    std::shared_ptr<const Camera> camera, std::unique_ptr<Sampler> sampler);

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_VOLPATH_H
