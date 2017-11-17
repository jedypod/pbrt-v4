
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

#ifndef PBRT_CORE_INTEGRATOR_H
#define PBRT_CORE_INTEGRATOR_H

// core/integrator.h*
#include <pbrt/core/pbrt.h>

#include <pbrt/util/bounds.h>
#include <pbrt/util/geometry.h>
#include <pbrt/core/sampling.h>
#include <pbrt/core/sampler.h>

#include <memory>
#include <vector>

namespace pbrt {

// Integrator Declarations
class Integrator {
  public:
    // Integrator Interface
    virtual ~Integrator();
    virtual void Render() = 0;

 protected:
    Integrator(const Scene &scene) : scene(scene) {}
    const Scene &scene;
};

Spectrum EstimateLd(const Interaction &it, const Scene &scene,
                    Sampler &sampler, const LightDistribution &lightDistrib,
                    bool handleMedia);

// SamplerIntegrator Declarations
class SamplerIntegrator : public Integrator {
  public:
    // SamplerIntegrator Public Methods
   SamplerIntegrator(const Scene &scene,
                     std::shared_ptr<const Camera> camera,
                     std::unique_ptr<Sampler> sampler,
                     const Bounds2i &pixelBounds)
       : Integrator(scene), camera(camera), initialSampler(std::move(sampler)),
         pixelBounds(pixelBounds) {}
    void Render();
    virtual Spectrum Li(const RayDifferential &ray, Sampler &sampler,
                        MemoryArena &arena, int depth = 0) const = 0;
    virtual void AfterWave() {}

    Spectrum SpecularReflect(const RayDifferential &ray,
                             const SurfaceInteraction &isect,
                             Sampler &sampler, MemoryArena &arena,
                             int depth) const;
    Spectrum SpecularTransmit(const RayDifferential &ray,
                              const SurfaceInteraction &isect,
                              Sampler &sampler, MemoryArena &arena,
                              int depth) const;

  protected:
    // SamplerIntegrator Protected Data
    std::shared_ptr<const Camera> camera;

  private:
    // SamplerIntegrator Private Data
    std::unique_ptr<Sampler> initialSampler;
    const Bounds2i pixelBounds;
};

}  // namespace pbrt

#endif  // PBRT_CORE_INTEGRATOR_H
