
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

#ifndef PBRT_SAMPLERS_HALTON_H
#define PBRT_SAMPLERS_HALTON_H

// samplers/halton.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <atomic>
#include <limits>
#include <memory>
#include <string>

namespace pbrt {

// HaltonSampler Declarations
class HaltonSampler final : public Sampler {
  public:
    // HaltonSampler Public Methods
    HaltonSampler(int samplesPerPixel, const Point2i &fullResolution, int seed = 0);
    PBRT_HOST_DEVICE
    ~HaltonSampler();

    static std::unique_ptr<HaltonSampler> Create(const ParameterDictionary &dict,
                                                 const Point2i &fullResolution);

    PBRT_HOST_DEVICE
    void ImplStartPixelSample(const Point2i &p, int pixelSample);
    PBRT_HOST_DEVICE
    Float ImplGet1D(const SamplerState &s);
    PBRT_HOST_DEVICE
    Point2f ImplGet2D(const SamplerState &s);

    std::unique_ptr<Sampler> Clone();
    std::string ToString() const;

  private:
    // HaltonSampler Private Data
    pstd::vector<DigitPermutation> *digitPermutations;
    bool ownDigitPermutations;
    int digitPermutationsSeed;

    HaltonPixelIndexer haltonPixelIndexer;
    Point2i pixelForIndex = Point2i(std::numeric_limits<int>::max(),
                                    std::numeric_limits<int>::max());

    RNG rng;  // If we run out of dimensions..
};


// PaddedSobolSampler Declarations
class PaddedSobolSampler final : public Sampler {
  public:
    // PaddedSobolSampler Public Methods
    PaddedSobolSampler(int samplesPerPixel, RandomizeStrategy randomizeStrategy);
    static std::unique_ptr<PaddedSobolSampler> Create(const ParameterDictionary &dict);

    PBRT_HOST_DEVICE
    Float ImplGet1D(const SamplerState &s);
    PBRT_HOST_DEVICE
    Point2f ImplGet2D(const SamplerState &s);

    std::unique_ptr<Sampler> Clone();
    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE_INLINE
    Float generateSample(pstd::span<const uint32_t> C, uint32_t a, uint32_t hash) const {
        switch (randomizeStrategy) {
        case RandomizeStrategy::None:
            return SampleGeneratorMatrix(C, a, NoRandomizer());
        case RandomizeStrategy::Xor:
            return SampleGeneratorMatrix(C, a, XORScrambler(hash));
        case RandomizeStrategy::Owen:
            return SampleGeneratorMatrix(C, a, OwenScrambler(hash));
        default:
            LOG_FATAL("Unhandled randomization strategy");
            return {};
        }
    }

    RandomizeStrategy randomizeStrategy;
};


class PMJ02BNSampler final : public Sampler {
  public:
    // PMJ02BNSampler Public Methods
    PMJ02BNSampler(int samplesPerPixel, Allocator alloc = {});

    static std::unique_ptr<PMJ02BNSampler> Create(const ParameterDictionary &dict);

    PBRT_HOST_DEVICE
    void ImplStartPixelSample(const Point2i &p, int pixelSample);
    PBRT_HOST_DEVICE
    Float ImplGet1D(const SamplerState &s);
    PBRT_HOST_DEVICE
    Point2f ImplGet2D(const SamplerState &s);

    std::unique_ptr<Sampler> Clone();
    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE_INLINE
    int pixelSampleOffset(Point2i p) const {
        DCHECK(p.x >= 0 && p.y >= 0);
        int px = p.x % pixelTileSize, py = p.y % pixelTileSize;
        return (px + py * pixelTileSize) * samplesPerPixel;
    }

    // PMJ02BNSampler Private Data
    int pixelTileSize;
    int pmjInstance;
    pstd::vector<Point2f> *pixelSamples;
};

class RandomSampler final : public Sampler {
  public:
    RandomSampler(int ns, int seed = 0);
    static std::unique_ptr<RandomSampler> Create(const ParameterDictionary &dict);

    PBRT_HOST_DEVICE
    void ImplStartPixelSample(const Point2i &p, int pixelSample);
    PBRT_HOST_DEVICE
    Float ImplGet1D(const SamplerState &s);
    PBRT_HOST_DEVICE
    Point2f ImplGet2D(const SamplerState &s);

    std::unique_ptr<Sampler> Clone();
    std::string ToString() const;

  private:
    int seed;
    RNG rng;
};

// SobolSampler Declarations
class SobolSampler final : public Sampler {
  public:
    // SobolSampler Public Methods
    SobolSampler(int samplesPerPixel, const Point2i &fullResolution,
                 RandomizeStrategy randomizeStrategy)
        : Sampler(RoundUpPow2(samplesPerPixel)),
          fullResolution(fullResolution),
          randomizeStrategy(randomizeStrategy) {
        if (!IsPowerOf2(samplesPerPixel))
            Warning("Non power-of-two sample count rounded up to %d "
                    "for SobolSampler.",
                    this->samplesPerPixel);
        resolution = RoundUpPow2(std::max(fullResolution.x, fullResolution.y));
        log2Resolution = Log2Int(resolution);
    }
    static std::unique_ptr<SobolSampler> Create(const ParameterDictionary &dict,
                                                const Point2i &fullResolution);

    PBRT_HOST_DEVICE
    void ImplStartPixelSample(const Point2i &p, int pixelSample);
    PBRT_HOST_DEVICE
    Float ImplGet1D(const SamplerState &s);
    PBRT_HOST_DEVICE
    Point2f ImplGet2D(const SamplerState &s);

    std::unique_ptr<Sampler> Clone();
    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE_INLINE
    Float sampleDimension(int dimension) const {
        if (dimension < 2 || randomizeStrategy == RandomizeStrategy::None)
            return SobolSample(sequenceIndex, dimension, NoRandomizer());

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson) {
            uint32_t hash = MixBits(dimension);
            return SobolSample(sequenceIndex, dimension,
                               CranleyPattersonRotator(hash));
        } else if (randomizeStrategy == RandomizeStrategy::Xor) {
            // Only use the dimension! (Want the same scrambling over all
            // pixels).
            uint32_t hash = MixBits(dimension);
            return SobolSample(sequenceIndex, dimension, XORScrambler(hash));
        } else {
            DCHECK(randomizeStrategy == RandomizeStrategy::Owen);
            uint32_t seed = MixBits(dimension);  // Only dimension!
            return SobolSample(sequenceIndex, dimension, OwenScrambler(seed));
        }
    }

    // SobolSampler Private Data
    Point2i fullResolution;
    int resolution, log2Resolution;

    Point2i pixelForIndex = Point2i(std::numeric_limits<int>::max(),
                                    std::numeric_limits<int>::max());
    int pixelSampleForIndex;
    int64_t sequenceIndex; // offset into Sobol sequence for current sample in current pixel
    RNG rng;  // If we run out of dimensions..
    RandomizeStrategy randomizeStrategy;
};


// StratifiedSampler Declarations
class StratifiedSampler final : public Sampler {
  public:
    // StratifiedSampler Public Methods
    StratifiedSampler(int xPixelSamples, int yPixelSamples, bool jitter, int seed = 0)
        : Sampler(xPixelSamples * yPixelSamples),
          xPixelSamples(xPixelSamples),
          yPixelSamples(yPixelSamples),
          jitter(jitter),
          seed(seed) {}
    static std::unique_ptr<StratifiedSampler> Create(const ParameterDictionary &dict);

    PBRT_HOST_DEVICE
    void ImplStartPixelSample(const Point2i &p, int sampleIndex);
    PBRT_HOST_DEVICE
    Float ImplGet1D(const SamplerState &s);
    PBRT_HOST_DEVICE
    Point2f ImplGet2D(const SamplerState &s);

    std::unique_ptr<Sampler> Clone();
    std::string ToString() const;

  private:
    // StratifiedSampler Private Data
    int xPixelSamples, yPixelSamples;
    bool jitter;
    int seed;
    RNG rng;
};

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_STRATIFIED_H
