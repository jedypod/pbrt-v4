
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
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pmj02tables.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <limits>
#include <memory>
#include <string>

namespace pbrt {

// HaltonSampler Declarations
class alignas(128) HaltonSampler {
  public:
    // HaltonSampler Public Methods
    HaltonSampler(int samplesPerPixel, const Point2i &fullResolution,
                  pstd::vector<DigitPermutation> *digitPermutations = nullptr,
                  Allocator alloc = {});

    static HaltonSampler *Create(const ParameterDictionary &dict,
                                 const Point2i &fullResolution,
                                 const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int index) {
        if (p != pixel)
            // Compute Halton sample offset for _currentPixel_
            haltonPixelIndexer.SetPixel(p);
        haltonPixelIndexer.SetPixelSample(index);

        pixel = p;
        sampleIndex = index;
        dimension = 0;

        rng.SetSequence(p.x + p.y * 65536);
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_HOST_DEVICE
    Float Get1D() {
        if (dimension >= PrimeTableSize)
            return rng.Uniform<Float>();

        // Note (for book): it's about 5x faster to use precomputed tables than
        // to generate the permutations on the fly...
        // TODO: check on GPU though...
        int dim = dimension++;
        return ScrambledRadicalInverse(dim, haltonPixelIndexer.SampleIndex(),
                                       (*digitPermutations)[dim]);
    }

    PBRT_HOST_DEVICE
    Point2f Get2D() {
        if (dimension == 0) {
            dimension += 2;
            return haltonPixelIndexer.SampleFirst2D();
        } else {
            if (dimension + 1 >= PrimeTableSize)
                return {rng.Uniform<Float>(), rng.Uniform<Float>()};

            int dim = dimension;
            dimension += 2;
            return {ScrambledRadicalInverse(dim, haltonPixelIndexer.SampleIndex(),
                                            (*digitPermutations)[dim]),
                    ScrambledRadicalInverse(dim + 1, haltonPixelIndexer.SampleIndex(),
                                            (*digitPermutations)[dim + 1])};
        }
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // HaltonSampler Private Data
    pstd::vector<DigitPermutation> *digitPermutations;

    HaltonPixelIndexer haltonPixelIndexer;
    Point2i pixel = Point2i(std::numeric_limits<int>::max(),
                            std::numeric_limits<int>::max());
    int sampleIndex = 0;
    int dimension = 0;
    int samplesPerPixel;

    RNG rng;  // If we run out of dimensions..
};


// PaddedSobolSampler Declarations
class alignas(128) PaddedSobolSampler {
  public:
    // PaddedSobolSampler Public Methods
    PaddedSobolSampler(int samplesPerPixel, RandomizeStrategy randomizeStrategy);

    static PaddedSobolSampler *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                      Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int index) {
        pixel = p;
        sampleIndex = index;
        dimension = 0;
    }

    PBRT_HOST_DEVICE
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^
                                ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16)
#ifdef __CUDA_ARCH__
                                ^ PbrtOptionsGPU.seed
#else
                                ^ PbrtOptions.seed
#endif
                                );
        int dim = dimension++;

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
            return SampleGeneratorMatrix(CVanDerCorput, index,
                                         CranleyPattersonRotator(BlueNoise(dim, pixel.x, pixel.y)));
        else
            return generateSample(CVanDerCorput, index, hash >> 32);
    }

    PBRT_HOST_DEVICE
    Point2f Get2D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^
                                ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16)
#ifdef __CUDA_ARCH__
                                ^ PbrtOptionsGPU.seed
#else
                                ^ PbrtOptions.seed
#endif
                                );
        int dim = dimension;
        dimension += 2;

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
            return {SampleGeneratorMatrix(CSobol[0], index,
                                          CranleyPattersonRotator(BlueNoise(dim, pixel.x, pixel.y))),
                    SampleGeneratorMatrix(CSobol[1], index,
                                          CranleyPattersonRotator(BlueNoise(dim + 1, pixel.x, pixel.y)))};
        else
            // Note: we're reusing the low 32 bits of the hash both for the
            // permutation and for the random scrambling in the first
            // dimension. This should(?) be fine.
            return {generateSample(CSobol[0], index, hash >> 8), generateSample(CSobol[1], index, hash >> 32)};
    }


    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
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

    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;
    int samplesPerPixel;

    RandomizeStrategy randomizeStrategy;
};

class alignas(128) PMJ02BNSampler {
  public:
    // PMJ02BNSampler Public Methods
    PMJ02BNSampler(int samplesPerPixel, Allocator alloc = {});

    static PMJ02BNSampler *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                  Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int index) {
        pixel = p;
        sampleIndex = index;
        dimension = 0;
        pmjInstance = 0;
    }

    PBRT_HOST_DEVICE
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^
                                ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16)
#ifdef __CUDA_ARCH__
                                ^ PbrtOptionsGPU.seed
#else
                                ^ PbrtOptions.seed
#endif
                                );

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);
        Float cpOffset = BlueNoise(dimension, pixel.x, pixel.y);
        Float u = (index + cpOffset) / samplesPerPixel;
        if (u >= 1) u -= 1;
        ++dimension;
        return std::min(u, OneMinusEpsilon);
    }

    PBRT_HOST_DEVICE
    Point2f Get2D() {
        // Don't start permuting until the second time through: when we
        // permute, that breaks the progressive part of the pattern and in
        // turn, convergence is similar to random until the very end. This way,
        // we generally do well for intermediate images as well.
        int index = sampleIndex;
        if (pmjInstance >= nPMJ02bnSets) {
            uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^
                                    ((uint64_t)pixel.y << 32) ^
                                    ((uint64_t)dimension << 16)
#ifdef __CUDA_ARCH__
                                ^ PbrtOptionsGPU.seed
#else
                                ^ PbrtOptions.seed
#endif
                                    );
            index = PermutationElement(sampleIndex, samplesPerPixel, hash);
        }

        if (dimension == 0) {
            // special case the pixel sample
            int offset = pixelSampleOffset(Point2i(pixel));
            dimension += 2;
            return (*pixelSamples)[offset + index];
        } else {
            Vector2f cpOffset(BlueNoise(dimension, pixel.x, pixel.y),
                              BlueNoise(dimension + 1, pixel.x, pixel.y));
            Point2f u = GetPMJ02BNSample(pmjInstance++, index) + cpOffset;
            if (u.x >= 1) u.x -= 1;
            if (u.y >= 1) u.y -= 1;
            dimension += 2;
            return {std::min(u.x, OneMinusEpsilon), std::min(u.y, OneMinusEpsilon)};
        }
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE_INLINE
    int pixelSampleOffset(Point2i p) const {
        DCHECK(p.x >= 0 && p.y >= 0);
        int px = p.x % pixelTileSize, py = p.y % pixelTileSize;
        return (px + py * pixelTileSize) * samplesPerPixel;
    }

    // PMJ02BNSampler Private Data
    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;

    int samplesPerPixel;
    int pixelTileSize;
    int pmjInstance;
    pstd::vector<Point2f> *pixelSamples;
};

class alignas(128) RandomSampler {
  public:
    RandomSampler(int samplesPerPixel, int seed = 0)
        : samplesPerPixel(samplesPerPixel), seed(seed) { }

    static RandomSampler *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                 Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int pixelSample) {
        rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
        // Assume we won't use more than 64k sample dimensions in a pixel...
        rng.Advance(pixelSample * 65536);
    }

    PBRT_HOST_DEVICE
    Float Get1D() {
        // TODO: (here and elsewhere) profiling..
        return rng.Uniform<Float>();
    }

    PBRT_HOST_DEVICE
    Point2f Get2D() {
        return Point2f{rng.Uniform<Float>(), rng.Uniform<Float>()};
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    int samplesPerPixel;
    int seed;
    RNG rng;
};

// SobolSampler Declarations
class alignas(128) SobolSampler {
  public:
    // SobolSampler Public Methods
    SobolSampler(int spp, const Point2i &fullResolution,
                 RandomizeStrategy randomizeStrategy)
        : samplesPerPixel(RoundUpPow2(spp)),
          randomizeStrategy(randomizeStrategy) {
        if (!IsPowerOf2(spp))
            Warning("Non power-of-two sample count rounded up to %d "
                    "for SobolSampler.",
                    samplesPerPixel);
        resolution = RoundUpPow2(std::max(fullResolution.x, fullResolution.y));
    }
    static SobolSampler *Create(const ParameterDictionary &dict, const Point2i &fullResolution,
                                const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int index) {
        DCHECK_LT(sampleIndex, samplesPerPixel);
        pixel = p;
        sampleIndex = index;
        dimension = 0;

        sequenceIndex = SobolIntervalToIndex(Log2Int(resolution), sampleIndex, pixel);

        rng.SetSequence(pixel.x + pixel.y * 65536);
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_HOST_DEVICE
    Float Get1D() {
        if (dimension >= NSobolDimensions)
            return rng.Uniform<Float>();

        return sampleDimension(dimension++);
    }

    PBRT_HOST_DEVICE
    Point2f Get2D() {
        if (dimension + 1 >= NSobolDimensions)
            return {rng.Uniform<Float>(), rng.Uniform<Float>()};

        Point2f u(sampleDimension(dimension), sampleDimension(dimension + 1));

        if (dimension == 0) {
            // Remap Sobol$'$ dimensions used for pixel samples
            for (int dim = 0; dim < 2; ++dim) {
                u[dim] = u[dim] * resolution;
                CHECK_RARE(1e-7, u[dim] - pixel[dim] < 0);
                CHECK_RARE(1e-7, u[dim] - pixel[dim] > 1);
                u[dim] = Clamp(u[dim] - pixel[dim], (Float)0, OneMinusEpsilon);
            }
        }

        dimension += 2;
        return u;
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE
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
    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;

    int samplesPerPixel;

    int resolution;

    int64_t sequenceIndex; // offset into Sobol sequence for current sample in current pixel
    RNG rng;  // If we run out of dimensions..
    RandomizeStrategy randomizeStrategy;
};


// StratifiedSampler Declarations
class alignas(128) StratifiedSampler {
  public:
    // StratifiedSampler Public Methods
    StratifiedSampler(int xPixelSamples, int yPixelSamples, bool jitter, int seed = 0)
        : xPixelSamples(xPixelSamples),
          yPixelSamples(yPixelSamples),
          jitter(jitter),
          seed(seed) {}

    static StratifiedSampler *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                     Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return xPixelSamples * yPixelSamples; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int index) {
        pixel = p;
        sampleIndex = index;
        dimension = 0;

        rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
        // Assume we won't use more than 64k sample dimensions in a pixel...
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_HOST_DEVICE
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^
                                ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16)
#ifdef __CUDA_ARCH__
                                ^ PbrtOptionsGPU.seed
#else
                                ^ PbrtOptions.seed
#endif
                                );
        ++dimension;

        int stratum = PermutationElement(sampleIndex, SamplesPerPixel(), hash);
        Float delta = jitter ? rng.Uniform<Float>() : 0.5f;
        return (stratum + delta) / SamplesPerPixel();
    }

    PBRT_HOST_DEVICE
    Point2f Get2D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^
                                ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16)
#ifdef __CUDA_ARCH__
                                ^ PbrtOptionsGPU.seed
#else
                                ^ PbrtOptions.seed
#endif
                                );
        dimension += 2;

        int stratum = PermutationElement(sampleIndex, SamplesPerPixel(), hash);
        int x = stratum % xPixelSamples;
        int y = stratum / xPixelSamples;
        Float dx = jitter ? rng.Uniform<Float>() : 0.5f;
        Float dy = jitter ? rng.Uniform<Float>() : 0.5f;
        return {(x + dx) / xPixelSamples, (y + dy) / yPixelSamples};
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // StratifiedSampler Private Data
    Point2i pixel;
    int sampleIndex = 0;
    int dimension = 0;

    int xPixelSamples, yPixelSamples;
    bool jitter;
    int seed;
    RNG rng;
};

// MLTSampler Declarations
class MLTSampler {
  public:
    // MLTSampler Public Methods
    MLTSampler(int mutationsPerPixel, int rngSequenceIndex, Float sigma,
               Float largeStepProbability, int streamCount)
        : mutationsPerPixel(mutationsPerPixel),
          rng(rngSequenceIndex),
          sigma(sigma),
          largeStepProbability(largeStepProbability),
          streamCount(streamCount) {}

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const { return mutationsPerPixel; }

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int sampleIndex) {
    }

    PBRT_HOST_DEVICE
    Float Get1D();

    PBRT_HOST_DEVICE
    Point2f Get2D();

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);

    PBRT_HOST_DEVICE
    void StartIteration();
    PBRT_HOST_DEVICE
    void Accept();
    PBRT_HOST_DEVICE
    void Reject();
    PBRT_HOST_DEVICE
    void StartStream(int index);
    PBRT_HOST_DEVICE
    int GetNextIndex() { return streamIndex + streamCount * sampleIndex++; }

    std::string DumpState() const;

    std::string ToString() const {
        return StringPrintf("[ MLTSampler rng: %s sigma: %f largeStepProbability: %f "
                            "streamCount: %d X: %s currentIteration: %d largeStep: %s "
                            "lastLargeStepIteration: %d streamIndex: %d sampleIndex: %d ] ",
                            rng, sigma, largeStepProbability, streamCount, X,
                            currentIteration, largeStep, lastLargeStepIteration, streamIndex,
                            sampleIndex);
    }

  protected:
    // MLTSampler Private Declarations
    struct PrimarySample {
        Float value = 0;
        // PrimarySample Public Methods
        PBRT_HOST_DEVICE
        void Backup() {
            valueBackup = value;
            modifyBackup = lastModificationIteration;
        }
        PBRT_HOST_DEVICE
        void Restore() {
            value = valueBackup;
            lastModificationIteration = modifyBackup;
        }

        std::string ToString() const {
            return StringPrintf("[ PrimarySample lastModificationIteration: %d "
                                "valueBackup: %f modifyBackup: %d ]",
                                lastModificationIteration, valueBackup, modifyBackup);
        }

        // PrimarySample Public Data
        int64_t lastModificationIteration = 0;
        Float valueBackup = 0;
        int64_t modifyBackup = 0;
    };

    // MLTSampler Private Methods
    PBRT_HOST_DEVICE
    void EnsureReady(int index);

    // MLTSampler Private Data
    int mutationsPerPixel;
    RNG rng;
    Float sigma, largeStepProbability;
    int streamCount;
    pstd::vector<PrimarySample> X;
    int64_t currentIteration = 0;
    bool largeStep = true;
    int64_t lastLargeStepIteration = 0;
    int streamIndex, sampleIndex;
};

class DebugMLTSampler : public MLTSampler {
public:
    static DebugMLTSampler Create(pstd::span<const std::string> state, int nSampleStreams);

    Float Get1D() {
        int index = GetNextIndex();
        CHECK_LT(index, u.size());
        return u[index];
    }

    Point2f Get2D() {
        return {Get1D(), Get1D()};
    }

    std::string ToString() const {
        return StringPrintf("[ DebugMLTSampler %s u: %s ]",
                            ((const MLTSampler *)this)->ToString(), u);
    }

private:
    DebugMLTSampler(int nSampleStreams)
        : MLTSampler(1, 0, 0.5, 0.5, nSampleStreams) { }

    std::vector<Float> u;
};

inline void SamplerHandle::StartPixelSample(const Point2i &p, int sampleIndex) {
    ProfilerScope _(ProfilePhase::StartPixelSample);
    switch (Tag()) {
    case TypeIndex<HaltonSampler>():
        Cast<HaltonSampler>()->StartPixelSample(p, sampleIndex);
        break;
    case TypeIndex<PaddedSobolSampler>():
        Cast<PaddedSobolSampler>()->StartPixelSample(p, sampleIndex);
        break;
    case TypeIndex<PMJ02BNSampler>():
        Cast<PMJ02BNSampler>()->StartPixelSample(p, sampleIndex);
        break;
    case TypeIndex<RandomSampler>():
        Cast<RandomSampler>()->StartPixelSample(p, sampleIndex);
        break;
    case TypeIndex<SobolSampler>():
        Cast<SobolSampler>()->StartPixelSample(p, sampleIndex);
        break;
    case TypeIndex<StratifiedSampler>():
        Cast<StratifiedSampler>()->StartPixelSample(p, sampleIndex);
        break;
    case TypeIndex<MLTSampler>():
        Cast<MLTSampler>()->StartPixelSample(p, sampleIndex);
        break;
    default:
        LOG_FATAL("Unhandled Sampler type");
    }
}

inline int SamplerHandle::SamplesPerPixel() const {
    switch (Tag()) {
    case TypeIndex<HaltonSampler>():
        return Cast<HaltonSampler>()->SamplesPerPixel();
    case TypeIndex<PaddedSobolSampler>():
        return Cast<PaddedSobolSampler>()->SamplesPerPixel();
    case TypeIndex<PMJ02BNSampler>():
        return Cast<PMJ02BNSampler>()->SamplesPerPixel();
    case TypeIndex<RandomSampler>():
        return Cast<RandomSampler>()->SamplesPerPixel();
    case TypeIndex<SobolSampler>():
        return Cast<SobolSampler>()->SamplesPerPixel();
    case TypeIndex<StratifiedSampler>():
        return Cast<StratifiedSampler>()->SamplesPerPixel();
    case TypeIndex<MLTSampler>():
        return Cast<MLTSampler>()->SamplesPerPixel();
    default:
        LOG_FATAL("Unhandled Sampler type");
        return {};
    }
}

inline Float SamplerHandle::Get1D() {
    ProfilerScope _(ProfilePhase::GetSample);
    switch (Tag()) {
    case TypeIndex<HaltonSampler>():
        return Cast<HaltonSampler>()->Get1D();
    case TypeIndex<PaddedSobolSampler>():
        return Cast<PaddedSobolSampler>()->Get1D();
    case TypeIndex<PMJ02BNSampler>():
        return Cast<PMJ02BNSampler>()->Get1D();
    case TypeIndex<RandomSampler>():
        return Cast<RandomSampler>()->Get1D();
    case TypeIndex<SobolSampler>():
        return Cast<SobolSampler>()->Get1D();
    case TypeIndex<StratifiedSampler>():
        return Cast<StratifiedSampler>()->Get1D();
    case TypeIndex<MLTSampler>():
        return Cast<MLTSampler>()->Get1D();
    default:
        LOG_FATAL("Unhandled Sampler type");
        return {};
    }
}

inline Point2f SamplerHandle::Get2D() {
    ProfilerScope _(ProfilePhase::GetSample);
    switch (Tag()) {
    case TypeIndex<HaltonSampler>():
        return Cast<HaltonSampler>()->Get2D();
    case TypeIndex<PaddedSobolSampler>():
        return Cast<PaddedSobolSampler>()->Get2D();
    case TypeIndex<PMJ02BNSampler>():
        return Cast<PMJ02BNSampler>()->Get2D();
    case TypeIndex<RandomSampler>():
        return Cast<RandomSampler>()->Get2D();
    case TypeIndex<SobolSampler>():
        return Cast<SobolSampler>()->Get2D();
    case TypeIndex<StratifiedSampler>():
        return Cast<StratifiedSampler>()->Get2D();
    case TypeIndex<MLTSampler>():
        return Cast<MLTSampler>()->Get2D();
    default:
        LOG_FATAL("Unhandled Sampler type");
        return {};
    }
}

inline CameraSample SamplerHandle::GetCameraSample(const Point2i &pPixel,
                                                   FilterHandle filter) {
    FilterSample fs = filter.Sample(Get2D());
#ifdef __CUDA_ARCH__
    if (PbrtOptionsGPU.disablePixelJitter) {
#else
    if (PbrtOptions.disablePixelJitter) {
#endif
        fs.p = Point2f(0, 0);
        fs.weight = 1;
    }

    CameraSample cs;
    cs.pFilm = pPixel + fs.p + Vector2f(0.5, 0.5);
    cs.time = Get1D();
    cs.pLens = Get2D();
    cs.weight = fs.weight;
    return cs;
}

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_STRATIFIED_H
