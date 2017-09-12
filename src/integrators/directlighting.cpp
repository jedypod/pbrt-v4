
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


// integrators/directlighting.cpp*
#include "integrators/directlighting.h"

#include "camera.h"
#include "error.h"
#include "film.h"
#include "interaction.h"
#include "paramset.h"
#include "sampler.h"
#include "util/stats.h"

namespace pbrt {

// DirectLightingIntegrator Method Definitions
void DirectLightingIntegrator::Preprocess(const Scene &scene,
                                          Sampler &sampler) {
    // Compute number of samples to use for each light
    for (const auto &light : scene.lights)
        nLightSamples.push_back(1);  // nSamples parameter removed

    // Request samples for sampling all lights
    for (int i = 0; i < maxDepth; ++i) {
        for (size_t j = 0; j < scene.lights.size(); ++j) {
            sampler.Request2DArray(nLightSamples[j]);
            sampler.Request2DArray(nLightSamples[j]);
        }
    }
}

Spectrum DirectLightingIntegrator::Li(const RayDifferential &ray,
                                      const Scene &scene, Sampler &sampler,
                                      MemoryArena &arena, int depth) const {
    ProfilePhase p(Prof::SamplerIntegratorLi);
    Spectrum L(0.f);
    // Find closest ray intersection or return background radiance
    SurfaceInteraction isect;
    if (!scene.Intersect(ray, &isect)) {
        for (const auto &light : scene.lights) L += light->Le(ray);
        return L;
    }

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, arena);
    if (!isect.bsdf)
        return Li(isect.SpawnRay(ray.d), scene, sampler, arena, depth);
    Vector3f wo = isect.wo;
    // Compute emitted light if ray hit an area light source
    L += isect.Le(wo);
    if (scene.lights.size() > 0)
        // Compute direct lighting for _DirectLightingIntegrator_ integrator
        L += UniformSampleAllLights(isect, scene, sampler, nLightSamples);

    if (depth + 1 < maxDepth) {
        Vector3f wi;
        // Trace rays for specular reflection and refraction
        L += SpecularReflect(ray, isect, scene, sampler, arena, depth);
        L += SpecularTransmit(ray, isect, scene, sampler, arena, depth);
    }
    return L;
}

std::unique_ptr<DirectLightingIntegrator> CreateDirectLightingIntegrator(
    const ParamSet &params, std::unique_ptr<Sampler> sampler,
    std::shared_ptr<const Camera> camera) {
    int maxDepth = params.GetOneInt("maxdepth", 5);
    gtl::ArraySlice<int> pb = params.GetIntArray("pixelbounds");
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (!pb.empty()) {
        if (pb.size() != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  (int)pb.size());
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Empty())
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }
    return std::make_unique<DirectLightingIntegrator>(
        maxDepth, camera, std::move(sampler), pixelBounds);
}

}  // namespace pbrt
