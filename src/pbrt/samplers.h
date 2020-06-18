// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLERS_HALTON_H
#define PBRT_SAMPLERS_HALTON_H

// samplers/halton.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/sampler.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pmj02tables.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
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

    static HaltonSampler *Create(const ParameterDictionary &parameters,
                                 const Point2i &fullResolution, const FileLoc *loc,
                                 Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    // HaltonSampler Private Data
    pstd::vector<DigitPermutation> *digitPermutations;

    HaltonPixelIndexer haltonPixelIndexer;
    Point2i pixel =
        Point2i(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
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

    static PaddedSobolSampler *Create(const ParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index) {
        pixel = p;
        sampleIndex = index;
        dimension = 0;

        rng.SetSequence(p.x + p.y * 65536);
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        int dim = dimension++;

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
            return SampleGeneratorMatrix(
                CVanDerCorput, index,
                CranleyPattersonRotator(BlueNoise(dim, pixel.x, pixel.y)));
        else
            return generateSample(CVanDerCorput, index, hash >> 32);
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        int dim = dimension;
        dimension += 2;

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
            return {SampleGeneratorMatrix(
                        CSobol[0], index,
                        CranleyPattersonRotator(BlueNoise(dim, pixel.x, pixel.y))),
                    SampleGeneratorMatrix(
                        CSobol[1], index,
                        CranleyPattersonRotator(BlueNoise(dim + 1, pixel.x, pixel.y)))};
        else
            // Note: we're reusing the low 32 bits of the hash both for the
            // permutation and for the random scrambling in the first
            // dimension. This should(?) be fine.
            return {generateSample(CSobol[0], index, hash >> 8),
                    generateSample(CSobol[1], index, hash >> 32)};
    }

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    PBRT_CPU_GPU
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
    RNG rng;
};

class alignas(128) PMJ02BNSampler {
  public:
    // PMJ02BNSampler Public Methods
    PMJ02BNSampler(int samplesPerPixel, Allocator alloc = {});

    static PMJ02BNSampler *Create(const ParameterDictionary &parameters,
                                  const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index) {
        pixel = p;
        sampleIndex = index;
        dimension = 0;
        pmjInstance = 0;

        rng.SetSequence(p.x + p.y * 65536);
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);

        int index = PermutationElement(sampleIndex, samplesPerPixel, hash);
        Float cpOffset = BlueNoise(dimension, pixel.x, pixel.y);
        Float u = (index + cpOffset) / samplesPerPixel;
        if (u >= 1)
            u -= 1;
        ++dimension;
        return std::min(u, OneMinusEpsilon);
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        // Don't start permuting until the second time through: when we
        // permute, that breaks the progressive part of the pattern and in
        // turn, convergence is similar to random until the very end. This way,
        // we generally do well for intermediate images as well.
        int index = sampleIndex;
        if (pmjInstance >= nPMJ02bnSets) {
            uint64_t hash =
                MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                        ((uint64_t)dimension << 16) ^ GetOptions().seed);
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
            if (u.x >= 1)
                u.x -= 1;
            if (u.y >= 1)
                u.y -= 1;
            dimension += 2;
            return {std::min(u.x, OneMinusEpsilon), std::min(u.y, OneMinusEpsilon)};
        }
    }

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    PBRT_CPU_GPU
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
    RNG rng;
};

class alignas(128) RandomSampler {
  public:
    RandomSampler(int samplesPerPixel, int seed = 0)
        : samplesPerPixel(samplesPerPixel), seed(seed) {}

    static RandomSampler *Create(const ParameterDictionary &parameters,
                                 const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int pixelSample) {
        rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
        // Assume we won't use more than 64k sample dimensions in a pixel...
        rng.Advance(pixelSample * 65536);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        // TODO: (here and elsewhere) profiling..
        return rng.Uniform<Float>();
    }

    PBRT_CPU_GPU
    Point2f Get2D() { return Point2f{rng.Uniform<Float>(), rng.Uniform<Float>()}; }

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

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
        : samplesPerPixel(RoundUpPow2(spp)), randomizeStrategy(randomizeStrategy) {
        if (!IsPowerOf2(spp))
            Warning("Non power-of-two sample count rounded up to %d "
                    "for SobolSampler.",
                    samplesPerPixel);
        resolution = RoundUpPow2(std::max(fullResolution.x, fullResolution.y));
    }
    static SobolSampler *Create(const ParameterDictionary &parameters,
                                const Point2i &fullResolution, const FileLoc *loc,
                                Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return samplesPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index) {
        DCHECK_LT(sampleIndex, samplesPerPixel);
        pixel = p;
        sampleIndex = index;
        dimension = 0;

        sequenceIndex = SobolIntervalToIndex(Log2Int(resolution), sampleIndex, pixel);

        rng.SetSequence(pixel.x + pixel.y * 65536);
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        if (dimension >= NSobolDimensions)
            return rng.Uniform<Float>();

        return sampleDimension(dimension++);
    }

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);
    std::string ToString() const;

  private:
    PBRT_CPU_GPU
    Float sampleDimension(int dimension) const {
        if (dimension < 2 || randomizeStrategy == RandomizeStrategy::None)
            return SobolSample(sequenceIndex, dimension, NoRandomizer());

        if (randomizeStrategy == RandomizeStrategy::CranleyPatterson) {
            uint32_t hash = MixBits(dimension);
            return SobolSample(sequenceIndex, dimension, CranleyPattersonRotator(hash));
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

    int64_t sequenceIndex;  // offset into Sobol sequence for current sample in
                            // current pixel
    RNG rng;                // If we run out of dimensions..
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

    static StratifiedSampler *Create(const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return xPixelSamples * yPixelSamples; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int index) {
        pixel = p;
        sampleIndex = index;
        dimension = 0;

        rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
        // Assume we won't use more than 64k sample dimensions in a pixel...
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_CPU_GPU
    Float Get1D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        ++dimension;

        int stratum = PermutationElement(sampleIndex, SamplesPerPixel(), hash);
        Float delta = jitter ? rng.Uniform<Float>() : 0.5f;
        return (stratum + delta) / SamplesPerPixel();
    }

    PBRT_CPU_GPU
    Point2f Get2D() {
        uint64_t hash = MixBits(((uint64_t)pixel.x << 48) ^ ((uint64_t)pixel.y << 32) ^
                                ((uint64_t)dimension << 16) ^ GetOptions().seed);
        dimension += 2;

        int stratum = PermutationElement(sampleIndex, SamplesPerPixel(), hash);
        int x = stratum % xPixelSamples;
        int y = stratum / xPixelSamples;
        Float dx = jitter ? rng.Uniform<Float>() : 0.5f;
        Float dy = jitter ? rng.Uniform<Float>() : 0.5f;
        return {(x + dx) / xPixelSamples, (y + dy) / yPixelSamples};
    }

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

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

    PBRT_CPU_GPU
    int SamplesPerPixel() const { return mutationsPerPixel; }

    PBRT_CPU_GPU
    void StartPixelSample(const Point2i &p, int sampleIndex) {
        rng.SetSequence(p.x + p.y * 65536);
        rng.Advance(sampleIndex * 65536);
    }

    PBRT_CPU_GPU
    Float Get1D();

    PBRT_CPU_GPU
    Point2f Get2D();

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);

    PBRT_CPU_GPU
    void StartIteration();
    PBRT_CPU_GPU
    void Accept();
    PBRT_CPU_GPU
    void Reject();
    PBRT_CPU_GPU
    void StartStream(int index);
    PBRT_CPU_GPU
    int GetNextIndex() { return streamIndex + streamCount * sampleIndex++; }

    std::string DumpState() const;

    PBRT_CPU_GPU
    RNG &GetRNG() { return rng; }

    std::string ToString() const {
        return StringPrintf(
            "[ MLTSampler rng: %s sigma: %f largeStepProbability: %f "
            "streamCount: %d X: %s currentIteration: %d largeStep: %s "
            "lastLargeStepIteration: %d streamIndex: %d sampleIndex: %d ] ",
            rng, sigma, largeStepProbability, streamCount, X, currentIteration, largeStep,
            lastLargeStepIteration, streamIndex, sampleIndex);
    }

  protected:
    // MLTSampler Private Declarations
    struct PrimarySample {
        Float value = 0;
        // PrimarySample Public Methods
        PBRT_CPU_GPU
        void Backup() {
            valueBackup = value;
            modifyBackup = lastModificationIteration;
        }
        PBRT_CPU_GPU
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
    PBRT_CPU_GPU
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
    static DebugMLTSampler Create(pstd::span<const std::string> state,
                                  int nSampleStreams);

    PBRT_CPU_GPU
    Float Get1D() {
        int index = GetNextIndex();
        CHECK_LT(index, u.size());
#ifdef PBRT_IS_GPU_CODE
        return 0;
#else
        return u[index];
#endif
    }

    PBRT_CPU_GPU
    Point2f Get2D() { return {Get1D(), Get1D()}; }

    std::string ToString() const {
        return StringPrintf("[ DebugMLTSampler %s u: %s ]",
                            ((const MLTSampler *)this)->ToString(), u);
    }

  private:
    DebugMLTSampler(int nSampleStreams) : MLTSampler(1, 0, 0.5, 0.5, nSampleStreams) {}

    std::vector<Float> u;
};

inline void SamplerHandle::StartPixelSample(const Point2i &p, int sampleIndex) {
    auto start = [&](auto ptr) { return ptr->StartPixelSample(p, sampleIndex); };
    return Apply<void>(start);
}

inline int SamplerHandle::SamplesPerPixel() const {
    auto spp = [&](auto ptr) { return ptr->SamplesPerPixel(); };
    return Apply<int>(spp);
}

inline Float SamplerHandle::Get1D() {
    auto get = [&](auto ptr) { return ptr->Get1D(); };
    return Apply<Float>(get);
}

inline Point2f SamplerHandle::Get2D() {
    auto get = [&](auto ptr) { return ptr->Get2D(); };
    return Apply<Point2f>(get);
}

inline RNG &SamplerHandle::GetRNG() {
    auto rng = [&](auto ptr) -> RNG & { return ptr->GetRNG(); };
    return Apply<RNG &>(rng);
}

inline CameraSample SamplerHandle::GetCameraSample(const Point2i &pPixel,
                                                   FilterHandle filter) {
    FilterSample fs = filter.Sample(Get2D());
    if (GetOptions().disablePixelJitter) {
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

inline int SamplerHandle::GetDiscrete1D(int n) {
    return std::min<int>(Get1D() * n, n - 1);
}

}  // namespace pbrt

#endif  // PBRT_SAMPLERS_STRATIFIED_H
