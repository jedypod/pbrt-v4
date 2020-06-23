// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_INTEGRATORS_WHITTED_H
#define PBRT_INTEGRATORS_WHITTED_H

// integrators.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/sampler.h>
#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/cpu/primitive.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

// Integrator Declarations
class Integrator {
  public:
    // Integrator Interface
    virtual ~Integrator();

    static std::unique_ptr<Integrator> Create(const std::string &name,
                                              const ParameterDictionary &parameters,
                                              CameraHandle camera, SamplerHandle sampler,
                                              PrimitiveHandle aggregate,
                                              std::vector<LightHandle> lights,
                                              const RGBColorSpace *colorSpace,
                                              const FileLoc *loc);

    virtual void Render() = 0;
    virtual std::string ToString() const = 0;

    const Bounds3f &SceneBounds() const { return sceneBounds; }
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const;
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    bool Unoccluded(const Interaction &p0, const Interaction &p1) const {
        return !IntersectP(p0.SpawnRayTo(p1), 1 - ShadowEpsilon);
    }

    SampledSpectrum Tr(const Interaction &p0, const Interaction &p1,
                       const SampledWavelengths &lambda, RNG &rng) const;

    std::vector<LightHandle> lights;
    // Store infinite light sources separately for cases where we only want
    // to loop over them.
    std::vector<LightHandle> infiniteLights;

  protected:
    Integrator(PrimitiveHandle aggregate, std::vector<LightHandle> l)
        : lights(std::move(l)), aggregate(aggregate) {
        // Scene Constructor Implementation
        if (aggregate)
            sceneBounds = aggregate.Bounds();
        for (auto &light : lights) {
            light.Preprocess(sceneBounds);
            if (light.Type() == LightType::Infinite)
                infiniteLights.push_back(light);
        }
    }

    PrimitiveHandle aggregate;
    Bounds3f sceneBounds;
};

class ImageTileIntegrator : public Integrator {
  public:
    ImageTileIntegrator(CameraHandle camera, SamplerHandle sampler,
                        PrimitiveHandle aggregate, std::vector<LightHandle> lights)
        : Integrator(aggregate, lights), camera(camera), initialSampler(sampler) {}

    void Render();
    virtual void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                                     SamplerHandle sampler,
                                     ScratchBuffer &scratchBuffer) = 0;

  protected:
    // ImageTileIntegrator Protected Data
    CameraHandle camera;
    SamplerHandle initialSampler;
};

// RayIntegrator Declarations
class RayIntegrator : public ImageTileIntegrator {
  public:
    // RayIntegrator Public Methods
    RayIntegrator(CameraHandle camera, SamplerHandle sampler, PrimitiveHandle aggregate,
                  std::vector<LightHandle> lights)
        : ImageTileIntegrator(camera, sampler, aggregate, lights) {}

    void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                             SamplerHandle sampler, ScratchBuffer &scratchBuffer) final;
    virtual SampledSpectrum Li(
        RayDifferential ray, const SampledWavelengths &lambda, SamplerHandle sampler,
        ScratchBuffer &scratchBuffer,
        pstd::optional<VisibleSurface> *visibleSurface = nullptr) const = 0;
};

// WhittedIntegrator Declarations
class WhittedIntegrator : public RayIntegrator {
  public:
    // WhittedIntegrator Public Methods
    WhittedIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                      PrimitiveHandle aggregate, std::vector<LightHandle> lights)
        : RayIntegrator(camera, sampler, aggregate, lights), maxDepth(maxDepth) {}

    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface = nullptr) const;

    static std::unique_ptr<WhittedIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    SampledSpectrum WhittedLi(RayDifferential ray, const SampledWavelengths &lambda,
                              SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                              int depth) const;

    // WhittedIntegrator Private Data
    int maxDepth;
};

// SimplePathIntegrator Declarations
class SimplePathIntegrator : public RayIntegrator {
  public:
    // SimplePathIntegrator Public Methods
    SimplePathIntegrator(int maxDepth, bool sampleLights, bool sampleBSDF,
                         CameraHandle camera, SamplerHandle sampler,
                         PrimitiveHandle aggregate, std::vector<LightHandle> lights);

    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<SimplePathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

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
    LightPathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                        PrimitiveHandle aggregate, std::vector<LightHandle> lights);

    void EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                             SamplerHandle sampler, ScratchBuffer &scratchBuffer);

    static std::unique_ptr<LightPathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

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
    PathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                   PrimitiveHandle aggregate, std::vector<LightHandle> lights,
                   Float rrThreshold = 1, const std::string &lightSampleStrategy = "bvh",
                   bool regularize = true);

    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<PathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    SampledSpectrum SampleLd(const SurfaceInteraction &intr,
                             const SampledWavelengths &lambda,
                             SamplerHandle sampler) const;

    // PathIntegrator Private Data
    int maxDepth;
    Float rrThreshold;
    LightSamplerHandle lightSampler;
    bool regularize;
};

// SimpleVolPathIntegrator Declarations
class SimpleVolPathIntegrator : public RayIntegrator {
  public:
    SimpleVolPathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                            PrimitiveHandle aggregate, std::vector<LightHandle> lights);

    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<SimpleVolPathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    int maxDepth;
    bool sampleLights, samplePhase;
};

// VolPathIntegrator Declarations
class VolPathIntegrator : public RayIntegrator {
  public:
    // VolPathIntegrator Public Methods
    VolPathIntegrator(int maxDepth, CameraHandle camera, SamplerHandle sampler,
                      PrimitiveHandle aggregate, std::vector<LightHandle> lights,
                      Float rrThreshold = 1,
                      const std::string &lightSampleStrategy = "bvh",
                      bool regularize = true)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          rrThreshold(rrThreshold),
          lightSampler(
              LightSamplerHandle::Create(lightSampleStrategy, lights, Allocator())),
          regularize(regularize) {}

    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<VolPathIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    SampledSpectrum SampleLd(const Interaction &intr,
                             const SampledWavelengths &lambda,
                             SamplerHandle sampler,
                             const SampledSpectrum &beta,
                             const SampledSpectrum &pathPDF) const;

    // VolPathIntegrator Private Data
    const int maxDepth;
    const Float rrThreshold;
    LightSamplerHandle lightSampler;
    bool regularize;
};

// AOIntegrator Declarations
class AOIntegrator : public RayIntegrator {
  public:
    // AOIntegrator Public Methods
    AOIntegrator(bool cosSample, Float maxDist, CameraHandle camera,
                 SamplerHandle sampler, PrimitiveHandle aggregate,
                 std::vector<LightHandle> lights, SpectrumHandle illuminant);

    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<AOIntegrator> Create(
        const ParameterDictionary &parameters, SpectrumHandle illuminant,
        CameraHandle camera, SamplerHandle sampler, PrimitiveHandle aggregate,
        std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    bool cosSample;
    Float maxDist;
    SpectrumHandle illuminant;
};

// BDPT Declarations
struct Vertex;
class BDPTIntegrator : public RayIntegrator {
  public:
    // BDPTIntegrator Public Methods
    BDPTIntegrator(CameraHandle camera, SamplerHandle sampler, PrimitiveHandle aggregate,
                   std::vector<LightHandle> lights, int maxDepth,
                   bool visualizeStrategies, bool visualizeWeights,
                   const std::string &lightSampleStrategy = "power",
                   bool regularize = true)
        : RayIntegrator(camera, sampler, aggregate, lights),
          maxDepth(maxDepth),
          visualizeStrategies(visualizeStrategies),
          visualizeWeights(visualizeWeights),
          lightSampleStrategy(lightSampleStrategy),
          regularize(regularize) {}

    void Render();
    SampledSpectrum Li(RayDifferential ray, const SampledWavelengths &lambda,
                       SamplerHandle sampler, ScratchBuffer &scratchBuffer,
                       pstd::optional<VisibleSurface> *visibleSurface) const;

    static std::unique_ptr<BDPTIntegrator> Create(
        const ParameterDictionary &parameters, CameraHandle camera, SamplerHandle sampler,
        PrimitiveHandle aggregate, std::vector<LightHandle> lights, const FileLoc *loc);

    std::string ToString() const;

  private:
    // BDPTIntegrator Private Data
    int maxDepth;
    bool visualizeStrategies;
    bool visualizeWeights;
    std::string lightSampleStrategy;
    bool regularize;
    LightSamplerHandle lightSampler;
    mutable std::vector<FilmHandle> weightFilms;
};

// MLT Declarations
class MLTSampler;

class MLTIntegrator : public Integrator {
  public:
    // MLTIntegrator Public Methods
    MLTIntegrator(CameraHandle camera, PrimitiveHandle aggregate,
                  std::vector<LightHandle> lights, int maxDepth, int nBootstrap,
                  int nChains, int mutationsPerPixel, Float sigma,
                  Float largeStepProbability, bool regularize)
        : Integrator(aggregate, lights),
          camera(camera),
          maxDepth(maxDepth),
          nBootstrap(nBootstrap),
          nChains(nChains),
          mutationsPerPixel(mutationsPerPixel),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          regularize(regularize) {}
    void Render();

    static std::unique_ptr<MLTIntegrator> Create(const ParameterDictionary &parameters,
                                                 CameraHandle camera,
                                                 PrimitiveHandle aggregate,
                                                 std::vector<LightHandle> lights,
                                                 const FileLoc *loc);

    std::string ToString() const;

  private:
    static const int cameraStreamIndex;
    static const int lightStreamIndex;
    static const int connectionStreamIndex;
    static const int nSampleStreams;

    SampledSpectrum L(ScratchBuffer &scratchBuffer, MLTSampler &sampler, int k,
                      Point2f *pRaster, SampledWavelengths *lambda);

    // MLTIntegrator Private Data
    CameraHandle camera;
    int maxDepth;
    int nBootstrap;
    int nChains;
    int mutationsPerPixel;
    Float sigma, largeStepProbability;
    LightSamplerHandle lightSampler;
    bool regularize;
};

// SPPM Declarations
class SPPMIntegrator : public Integrator {
  public:
    // SPPMIntegrator Public Methods
    SPPMIntegrator(CameraHandle camera, PrimitiveHandle aggregate,
                   std::vector<LightHandle> lights, int nIterations,
                   int photonsPerIteration, int maxDepth, Float initialSearchRadius,
                   bool regularize, int seed, const RGBColorSpace *colorSpace)
        : Integrator(aggregate, lights),
          camera(camera),
          initialSearchRadius(initialSearchRadius),
          nIterations(nIterations),
          maxDepth(maxDepth),
          photonsPerIteration(photonsPerIteration > 0
                                  ? photonsPerIteration
                                  : camera.GetFilm().PixelBounds().Area()),
          regularize(regularize),
          colorSpace(colorSpace),
          digitPermutationsSeed(seed) {}

    void Render();

    static std::unique_ptr<SPPMIntegrator> Create(const ParameterDictionary &parameters,
                                                  const RGBColorSpace *colorSpace,
                                                  CameraHandle camera,
                                                  PrimitiveHandle aggregate,
                                                  std::vector<LightHandle> lights,
                                                  const FileLoc *loc);

    std::string ToString() const;

  private:
    SampledSpectrum SampleLd(const SurfaceInteraction &intr,
                             const SampledWavelengths &lambda,
                             SamplerHandle sampler,
                             LightSamplerHandle lightSampler) const;

    // SPPMIntegrator Private Data
    CameraHandle camera;
    Float initialSearchRadius;
    int nIterations;
    int maxDepth;
    int photonsPerIteration;
    bool regularize;
    const RGBColorSpace *colorSpace;
    int digitPermutationsSeed;
};

}  // namespace pbrt

#endif  // PBRT_INTEGRATORS_H
