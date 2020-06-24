
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

#ifndef PBRT_INTEGRATORS_WHITTED_H
#define PBRT_INTEGRATORS_WHITTED_H

// integrators.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/bsdf.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsampling.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/scene.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

SampledSpectrum LdSampleLights(const SurfaceInteraction &intr, const Scene &scene,
                               const SampledWavelengths &lambda,
                               SamplerHandle sampler, MemoryArena &arena,
                               const LightSampler &lightSampler,
                               bool handleMedia = false);
SampledSpectrum LdSampleLights(const MediumInteraction &intr, const Scene &scene,
                               const SampledWavelengths &lambda,
                               SamplerHandle sampler, MemoryArena &arena,
                               const LightSampler &lightSampler);
SampledSpectrum LdSampleLightsAndBSDF(const SurfaceInteraction &intr, const Scene &scene,
                                      const SampledWavelengths &lambda,
                                      SamplerHandle sampler, MemoryArena &arena,
                                      const LightSampler &lightSampler);

SampledSpectrum Tr(const Scene &scene, const SampledWavelengths &lambda,
                   SamplerHandle sampler, const Interaction &p0, const Interaction &p1);

class ImageTileIntegrator : public Integrator {
public:
    ImageTileIntegrator(const Scene &scene,
                        std::unique_ptr<const Camera> c,
                        SamplerHandle sampler)
        : Integrator(scene), camera(std::move(c)), initialSampler(std::move(sampler))
    {}

    void Render();
    virtual void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                                     SamplerHandle sampler, MemoryArena &arena,
                                     MaterialBuffer &materialBuffer) = 0;

  protected:
    // ImageTileIntegrator Protected Data
    std::unique_ptr<const Camera> camera;
    SamplerHandle initialSampler;
};

// RayIntegrator Declarations
class RayIntegrator : public ImageTileIntegrator {
  public:
    // RayIntegrator Public Methods
    RayIntegrator(const Scene &scene,
                  std::unique_ptr<const Camera> c,
                  SamplerHandle sampler)
        : ImageTileIntegrator(scene, std::move(c), std::move(sampler))
    {}

    void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                             SamplerHandle sampler, MemoryArena &arena,
                             MaterialBuffer &materialBuffer) final;
    virtual SampledSpectrum Li(RayDifferential ray,
                               const SampledWavelengths &lambda,
                               SamplerHandle sampler,
                               MemoryArena &arena,
                               MaterialBuffer &materialBuffer,
                               pstd::optional<VisibleSurface> *visibleSurface = nullptr) const = 0;
};

// WhittedIntegrator Declarations
class WhittedIntegrator : public RayIntegrator {
  public:
    // WhittedIntegrator Public Methods
    WhittedIntegrator(int maxDepth, const Scene &scene,
                      std::unique_ptr<const Camera> c,
                      SamplerHandle sampler)
        : RayIntegrator(scene, std::move(c), std::move(sampler)),
          maxDepth(maxDepth) {}
    SampledSpectrum Li(RayDifferential ray,
                       const SampledWavelengths &lambda,
                       SamplerHandle sampler,
                       MemoryArena &arena,
                       MaterialBuffer &materialBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface = nullptr) const;

    static std::unique_ptr<WhittedIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler,
        const FileLoc *loc);

    std::string ToString() const;

  private:
    SampledSpectrum WhittedLi(RayDifferential ray,
                              const SampledWavelengths &lambda,
                              SamplerHandle sampler,
                              MemoryArena &arena,
                              MaterialBuffer &materialBuffer,
                              int depth) const;

    // WhittedIntegrator Private Data
    int maxDepth;
};

// SimplePathIntegrator Declarations
class SimplePathIntegrator : public RayIntegrator {
  public:
    // SimplePathIntegrator Public Methods
    SimplePathIntegrator(int maxDepth, bool sampleLights, bool sampleBSDF,
                         const Scene &scene, std::unique_ptr<const Camera> camera,
                         SamplerHandle sampler);

    SampledSpectrum Li(RayDifferential ray,
                       const SampledWavelengths &lambda,
                       SamplerHandle sampler,
                       MemoryArena &arena,
                       MaterialBuffer &materialBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<SimplePathIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SimplePathIntegrator Private Data
    int maxDepth;
    bool sampleLights, sampleBSDF;
    UniformLightSampler lightSampler;
};

// LightPathIntegrator Declarations
class LightPathIntegrator : public ImageTileIntegrator {
  public:
    // LightPathIntegrator Public Methods
    LightPathIntegrator(int maxDepth, const Scene &scene,
                        std::unique_ptr<const Camera> camera,
                        SamplerHandle sampler);

    void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                             SamplerHandle sampler, MemoryArena &arena,
                             MaterialBuffer &materialBuffer);

    static std::unique_ptr<LightPathIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    std::string ToString() const;

  private:
    // LightPathIntegrator Private Data
    int maxDepth;
    std::unique_ptr<PowerLightSampler> lightSampler;
};

// PathIntegrator Declarations
class PathIntegrator : public RayIntegrator {
  public:
    // PathIntegrator Public Methods
    PathIntegrator(int maxDepth, const Scene &scene,
                   std::unique_ptr<const Camera> camera,
                   SamplerHandle sampler,
                   Float rrThreshold = 1,
                   const std::string &lightSampleStrategy = "bvh",
                   bool regularize = true);

    SampledSpectrum Li(RayDifferential ray,
                       const SampledWavelengths &lambda,
                       SamplerHandle sampler,
                       MemoryArena &arena,
                       MaterialBuffer &materialBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<PathIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    std::string ToString() const;

  private:
    // PathIntegrator Private Data
    int maxDepth;
    Float rrThreshold;
    std::unique_ptr<LightSampler> lightSampler;
    bool regularize;
};

// VolPathIntegrator Declarations
class VolPathIntegrator : public RayIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, const Scene &scene,
                      std::unique_ptr<const Camera> c,
                      SamplerHandle sampler,
                      Float rrThreshold = 1,
                      const std::string &lightSampleStrategy = "bvh",
                      bool regularize = true)
      : RayIntegrator(scene, std::move(c), std::move(sampler)),
        maxDepth(maxDepth),
        rrThreshold(rrThreshold),
        lightSampler(LightSampler::Create(lightSampleStrategy, scene.lights,
                                          Allocator())),
        regularize(regularize) { }

    SampledSpectrum Li(RayDifferential ray,
                       const SampledWavelengths &lambda,
                       SamplerHandle sampler, MemoryArena &arena,
                       MaterialBuffer &materialBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<VolPathIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    std::string ToString() const;

  private:
    // VolPathIntegrator Private Data
    const int maxDepth;
    const Float rrThreshold;
    std::unique_ptr<LightSampler> lightSampler;
    bool regularize;
};

// AOIntegrator Declarations
class AOIntegrator : public RayIntegrator {
  public:
    // AOIntegrator Public Methods
    AOIntegrator(bool cosSample, Float maxDist, const Scene &scene,
                 std::unique_ptr<const Camera> camera,
                 SamplerHandle sampler, SpectrumHandle illuminant);
    SampledSpectrum Li(RayDifferential ray,
                       const SampledWavelengths &lambda,
                       SamplerHandle sampler, MemoryArena &arena,
                       MaterialBuffer &materialBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<AOIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        SpectrumHandle illuminant,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    std::string ToString() const;

  private:
    bool cosSample;
    Float maxDist;
    SpectrumHandle illuminant;
};

// BDPT Declarations
class BDPTIntegrator : public RayIntegrator {
  public:
    // BDPTIntegrator Public Methods
    BDPTIntegrator(const Scene &scene, std::unique_ptr<const Camera> c,
                   SamplerHandle sampler, int maxDepth,
                   bool visualizeStrategies, bool visualizeWeights,
                   const std::string &lightSampleStrategy = "power",
                   bool regularize = true)
        : RayIntegrator(scene, std::move(c), std::move(sampler)),
          maxDepth(maxDepth),
          visualizeStrategies(visualizeStrategies),
          visualizeWeights(visualizeWeights),
          lightSampleStrategy(lightSampleStrategy),
          regularize(regularize) {}

    void Render();
    SampledSpectrum Li(RayDifferential ray,
                       const SampledWavelengths &lambda,
                       SamplerHandle sampler,
                       MemoryArena &arena,
                       MaterialBuffer &materialBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<BDPTIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    std::string ToString() const;

  private:
    // BDPTIntegrator Private Data
    int maxDepth;
    bool visualizeStrategies;
    bool visualizeWeights;
    std::string lightSampleStrategy;
    bool regularize;
    std::unique_ptr<FixedLightSampler> lightSampler;
    std::vector<std::unique_ptr<Film>> weightFilms;
};

// MLT Declarations
class MLTSampler;

class MLTIntegrator : public Integrator {
  public:
    // MLTIntegrator Public Methods
    MLTIntegrator(const Scene &scene, std::unique_ptr<const Camera> c,
                  int maxDepth, int nBootstrap, int nChains,
                  int mutationsPerPixel, Float sigma, Float largeStepProbability,
                  bool regularize)
        : Integrator(scene),
          camera(std::move(c)),
          maxDepth(maxDepth),
          nBootstrap(nBootstrap),
          nChains(nChains),
          mutationsPerPixel(mutationsPerPixel),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          regularize(regularize) {}
    void Render();

    static std::unique_ptr<MLTIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, const FileLoc *loc);

    std::string ToString() const;

  private:
    static const int cameraStreamIndex;
    static const int lightStreamIndex;
    static const int connectionStreamIndex;
    static const int nSampleStreams;

    SampledSpectrum L(MemoryArena &arena, MaterialBuffer &materialBuffer,
                      MLTSampler &sampler, int k,
                      Point2f *pRaster, SampledWavelengths *lambda);

    // MLTIntegrator Private Data
    std::unique_ptr<const Camera> camera;
    int maxDepth;
    int nBootstrap;
    int nChains;
    int mutationsPerPixel;
    Float sigma, largeStepProbability;
    std::unique_ptr<FixedLightSampler> lightSampler;
    bool regularize;
};

// SPPM Declarations
class SPPMIntegrator : public Integrator {
  public:
    // SPPMIntegrator Public Methods
    SPPMIntegrator(const Scene &scene, std::unique_ptr<const Camera> c,
                   int nIterations, int photonsPerIteration, int maxDepth,
                   Float initialSearchRadius, bool regularize, int seed,
                   const RGBColorSpace *colorSpace)
        : Integrator(scene),
          camera(std::move(c)),
          initialSearchRadius(initialSearchRadius),
          nIterations(nIterations),
          maxDepth(maxDepth),
          photonsPerIteration(photonsPerIteration > 0
                                  ? photonsPerIteration
                                  : camera->film->pixelBounds.Area()),
          regularize(regularize),
          colorSpace(colorSpace),
          digitPermutationsSeed(seed) {}

    void Render();

    static std::unique_ptr<SPPMIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        const RGBColorSpace *colorSpace, std::unique_ptr<const Camera> camera, const FileLoc *loc);

    std::string ToString() const;

  private:
    // SPPMIntegrator Private Data
    std::unique_ptr<const Camera> camera;
    Float initialSearchRadius;
    int nIterations;
    int maxDepth;
    int photonsPerIteration;
    bool regularize;
    const RGBColorSpace *colorSpace;
    int digitPermutationsSeed;
};

class RISIntegrator : public Integrator {
 public:
    enum class LightMISStrategy { One, SqrtM, M };
    RISIntegrator(int N, int M, int nSpatioResamples, bool useMIS,
                  LightMISStrategy lightMISStrategy,
                  const std::string &lightSampleStrategy,
                  const Scene &scene,
                  std::unique_ptr<const Camera> c,
                  int spp);

    static std::unique_ptr<RISIntegrator> Create(
        const ParameterDictionary &dict, const Scene &scene,
        std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc);

    void Render();
    std::string ToString() const;

 private:
    int N, M, rootM;
    int nSpatioResamples;
    bool useMIS;
    LightMISStrategy lightMISStrategy;
    std::unique_ptr<const Camera> camera;
    int spp;
    std::string lightSampleStrategy;
};

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_H
