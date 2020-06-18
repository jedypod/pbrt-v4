// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SAMPLING_LOWDISCREPANCY_H
#define PBRT_SAMPLING_LOWDISCREPANCY_H

// sampling/lowdiscrepancy.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/shuffle.h>
#include <pbrt/util/sobolmatrices.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <memory>
#include <string>

namespace pbrt {

class DigitPermutation {
  public:
    DigitPermutation() = default;
    DigitPermutation(int base, uint32_t seed, Allocator alloc) : base(base) {
        CHECK_LT(base, 65536);  // uint16_t

        // Same computation that ScrambledRadicalInverseSpecialized does...
        // It would be nice to do this in closed form, but it's a little
        // messy to work out, given floating-point math, rounding, and all
        // that...
        nDigits = 0;
        Float invBase = (Float)1 / (Float)base;
        Float invBaseN = 1;
        while (1 - invBaseN < 1) {
            ++nDigits;
            invBaseN *= invBase;
        }

        permutations = alloc.allocate_object<uint16_t>(nDigits * base);

        for (int digitIndex = 0; digitIndex < nDigits; ++digitIndex) {
            uint32_t digitSeed = (base * 32 + digitIndex) ^ seed;
            for (int digitValue = 0; digitValue < base; ++digitValue)
                Perm(digitIndex, digitValue) =
                    PermutationElement(digitValue, base, digitSeed);
        }
    }

    PBRT_CPU_GPU
    int Permute(int digitIndex, int digitValue) const {
        DCHECK_LT(digitIndex, nDigits);
        DCHECK_LT(digitValue, base);
        return Perm(digitIndex, digitValue);
    }

    std::string ToString() const;

    int base;

  private:
    PBRT_CPU_GPU
    uint16_t &Perm(int digitIndex, int digitValue) {
        return permutations[digitIndex * base + digitValue];
    }
    PBRT_CPU_GPU
    uint16_t Perm(int digitIndex, int digitValue) const {
        return permutations[digitIndex * base + digitValue];
    }

    int nDigits;
    // indexed by [digitIndex * base + digitValue]
    uint16_t *permutations;
};

// Low Discrepancy Declarations
PBRT_CPU_GPU
Float RadicalInverse(int baseIndex, uint64_t a);
pstd::vector<DigitPermutation> *ComputeRadicalInversePermutations(uint32_t seed,
                                                                  Allocator alloc = {});
PBRT_CPU_GPU
Float ScrambledRadicalInverse(int baseIndex, uint64_t a, const DigitPermutation &perm);
PBRT_CPU_GPU
Float ScrambledRadicalInverse(int baseIndex, uint64_t a, uint32_t seed);

// Low Discrepancy Inline Functions
template <int base>
PBRT_CPU_GPU inline uint64_t InverseRadicalInverse(uint64_t inverse, int nDigits) {
    uint64_t index = 0;
    for (int i = 0; i < nDigits; ++i) {
        uint64_t digit = inverse % base;
        inverse /= base;
        index = index * base + digit;
    }
    return index;
}

class HaltonPixelIndexer {
  public:
    HaltonPixelIndexer(const Point2i &fullResolution);

    PBRT_CPU_GPU
    void SetPixel(const Point2i &p) {
        pixelSampleForIndex = 0;

        int sampleStride = baseScales[0] * baseScales[1];
        if (sampleStride > 1) {
            Point2i pm(Mod(p[0], MaxHaltonResolution), Mod(p[1], MaxHaltonResolution));
            for (int i = 0; i < 2; ++i) {
                uint64_t dimOffset =
                    (i == 0) ? InverseRadicalInverse<2>(pm[i], baseExponents[i])
                             : InverseRadicalInverse<3>(pm[i], baseExponents[i]);
                pixelSampleForIndex +=
                    dimOffset * (sampleStride / baseScales[i]) * multInverse[i];
            }
            pixelSampleForIndex %= sampleStride;
        }
    }

    PBRT_CPU_GPU
    void SetPixelSample(int pixelSample) {
        int sampleStride = baseScales[0] * baseScales[1];
        sampleIndex = pixelSampleForIndex + pixelSample * sampleStride;
    }

    PBRT_CPU_GPU
    Point2f SampleFirst2D() const {
        return {RadicalInverse(0, sampleIndex >> baseExponents[0]),
                RadicalInverse(1, sampleIndex / baseScales[1])};
    };

    PBRT_CPU_GPU
    int64_t SampleIndex() const { return sampleIndex; }

    std::string ToString() const {
        return StringPrintf("[ HaltonPixelIndexer pixelSampleForIndex: %d "
                            "sampleIndex: %d baseScales: %s baseExponents: %s "
                            "multInverse[0]: %d multInverse[1]: %d ]",
                            pixelSampleForIndex, sampleIndex, baseScales, baseExponents,
                            multInverse[0], multInverse[1]);
    }

  private:
    static constexpr int MaxHaltonResolution = 128;

    int64_t pixelSampleForIndex;
    int64_t sampleIndex;  // offset into Halton sequence for current sample
                          // in current pixel.

    // note: these could all be uint8_t
    Point2i baseScales, baseExponents;
    int multInverse[2];
};

class Halton128PixelIndexer {
  public:
    PBRT_CPU_GPU
    void SetPixel(const Point2i &p) {
        static constexpr int baseScales[2] = {128, 243};
        static constexpr int baseExponents[2] = {7, 5};
        static constexpr int multInverse[2] = {59, 131};

        pixelSampleForIndex = 0;

        int sampleStride = baseScales[0] * baseScales[1];
        Point2i pm(Mod(p[0], MaxHaltonResolution), Mod(p[1], MaxHaltonResolution));
        for (int i = 0; i < 2; ++i) {
            uint64_t dimOffset = (i == 0)
                                     ? InverseRadicalInverse<2>(pm[i], baseExponents[i])
                                     : InverseRadicalInverse<3>(pm[i], baseExponents[i]);
            pixelSampleForIndex +=
                dimOffset * (sampleStride / baseScales[i]) * multInverse[i];
        }
        pixelSampleForIndex %= sampleStride;
    }

    PBRT_CPU_GPU
    void SetPixelSample(int pixelSample) {
        static constexpr int baseScales[2] = {128, 243};
        static constexpr int baseExponents[2] = {7, 5};
        static constexpr int multInverse[2] = {59, 131};

        int sampleStride = baseScales[0] * baseScales[1];
        sampleIndex = pixelSampleForIndex + pixelSample * sampleStride;
    }

    PBRT_CPU_GPU
    Point2f SampleFirst2D() const {
        static constexpr int baseScales[2] = {128, 243};
        static constexpr int baseExponents[2] = {7, 5};
        static constexpr int multInverse[2] = {59, 131};

        return {RadicalInverse(0, sampleIndex >> baseExponents[0]),
                RadicalInverse(1, sampleIndex / baseScales[1])};
    };

    PBRT_CPU_GPU
    int64_t SampleIndex() const { return sampleIndex; }

  private:
    static constexpr int MaxHaltonResolution = 128;

    uint32_t pixelSampleForIndex;
    uint32_t sampleIndex;
};

PBRT_CPU_GPU
inline uint32_t MultiplyGenerator(pstd::span<const uint32_t> C, uint32_t a) {
    uint32_t v = 0;
    for (int i = 0; a != 0; ++i, a >>= 1)
        if (a & 1)
            v ^= C[i];
    return v;
}

PBRT_CPU_GPU
inline uint32_t OwenScramble(uint32_t v, uint32_t hash) {
    // Expect already reversed?
    v = ReverseBits32(v ^ hash);
    // Must be even numbers!
    v ^= v * 0x9207662f0u;
    v ^= v << 3;
    v ^= v * 0xfe932074u;
    v ^= v << 5;
    v ^= v * 0x94b41206u;
    v ^= v << 2;
    return ReverseBits32(v);
}

enum class RandomizeStrategy { None, CranleyPatterson, Xor, Owen };

std::string ToString(RandomizeStrategy r);

struct CranleyPattersonRotator {
    PBRT_CPU_GPU
    CranleyPattersonRotator(Float v) : offset(v * (1ull << 32)) {}
    PBRT_CPU_GPU
    CranleyPattersonRotator(uint32_t offset) : offset(offset) {}

    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return v + offset; }

    uint32_t offset;
};

struct XORScrambler {
    PBRT_CPU_GPU
    XORScrambler(uint32_t s) : s(s) {}

    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return s ^ v; }

    uint32_t s;
};

struct OwenScrambler {
    PBRT_CPU_GPU
    OwenScrambler(uint32_t seed) : seed(seed) {}

    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return OwenScramble(v, seed); }

    uint32_t seed;
};

struct NoRandomizer {
    PBRT_CPU_GPU
    uint32_t operator()(uint32_t v) const { return v; }
};

template <typename R>
PBRT_CPU_GPU inline Float SampleGeneratorMatrix(pstd::span<const uint32_t> C, uint32_t a,
                                                R randomizer) {
#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
    return std::min(randomizer(MultiplyGenerator(C, a)) * Float(2.3283064365386963e-10),
                    OneMinusEpsilon);
#else
    return std::min(randomizer(MultiplyGenerator(C, a)) * Float(0x1p-32),
                    OneMinusEpsilon);
#endif
}

PBRT_CPU_GPU
inline Float SampleGeneratorMatrix(pstd::span<const uint32_t> C, uint32_t a) {
    return SampleGeneratorMatrix(C, a, NoRandomizer());
}

PBRT_CONST uint32_t CVanDerCorput[32] = {
    // clang-format off
  0b10000000000000000000000000000000,
  0b1000000000000000000000000000000,
  0b100000000000000000000000000000,
  0b10000000000000000000000000000,
  // Remainder of Van Der Corput generator matrix entries
  0b1000000000000000000000000000,
  0b100000000000000000000000000,
  0b10000000000000000000000000,
  0b1000000000000000000000000,
  0b100000000000000000000000,
  0b10000000000000000000000,
  0b1000000000000000000000,
  0b100000000000000000000,
  0b10000000000000000000,
  0b1000000000000000000,
  0b100000000000000000,
  0b10000000000000000,
  0b1000000000000000,
  0b100000000000000,
  0b10000000000000,
  0b1000000000000,
  0b100000000000,
  0b10000000000,
  0b1000000000,
  0b100000000,
  0b10000000,
  0b1000000,
  0b100000,
  0b10000,
  0b1000,
  0b100,
  0b10,
  0b1,
    // clang-format on
};

PBRT_CPU_GPU
inline uint64_t SobolIntervalToIndex(uint32_t m, uint64_t frame, const Point2i &p) {
    if (m == 0)
        return frame;

    const uint32_t m2 = m << 1;
    uint64_t index = uint64_t(frame) << m2;

    uint64_t delta = 0;
    for (int c = 0; frame; frame >>= 1, ++c)
        if (frame & 1)  // Add flipped column m + c + 1.
            delta ^= VdCSobolMatrices[m - 1][c];

    // flipped b
    uint64_t b = (((uint64_t)((uint32_t)p.x) << m) | ((uint32_t)p.y)) ^ delta;

    for (int c = 0; b; b >>= 1, ++c)
        if (b & 1)  // Add column 2 * m - c.
            index ^= VdCSobolMatricesInv[m - 1][c];

    return index;
}

PBRT_CPU_GPU
inline uint32_t SobolSampleBits32(int64_t a, int dimension) {
    uint32_t v = 0;
    for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
        if (a & 1)
            v ^= SobolMatrices32[i];
    return v;
}

template <typename R>
PBRT_CPU_GPU inline float SobolSampleFloat(int64_t a, int dimension, R randomizer) {
    if (dimension >= NSobolDimensions)
        LOG_FATAL("Integrator has consumed too many Sobol' dimensions; you "
                  "may want to use a Sampler without a dimension limit like "
                  "\"02sequence.\"");

    uint32_t v = randomizer(SobolSampleBits32(a, dimension));
#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
    return std::min(v * 2.3283064365386963e-10f /* 1/2^32 */, FloatOneMinusEpsilon);
#else
    return std::min(v * 0x1p-32f /* 1/2^32 */, FloatOneMinusEpsilon);
#endif
}

PBRT_CPU_GPU
inline uint64_t SobolSampleBits64(int64_t a, int dimension) {
    if (dimension >= NSobolDimensions)
        LOG_FATAL("Integrator has consumed too many Sobol' dimensions; you "
                  "may want to use a Sampler without a dimension limit like "
                  "\"02sequence.\"");

    uint64_t v = 0;
    for (int i = dimension * SobolMatrixSize; a != 0; a >>= 1, i++)
        if (a & 1)
            v ^= SobolMatrices64[i];
    return v;
}

template <typename R>
PBRT_CPU_GPU inline double SobolSampleDouble(int64_t a, int dimension, R randomizer) {
    uint64_t v = SobolSampleBits64(a, dimension);
    // FIXME? We just scramble the high bits here...
    uint32_t vs = randomizer(v >> 32);
    v = (uint64_t(vs) << 32) | (v & 0xffffffff);
    return std::min(v * (1.0 / (1ULL << SobolMatrixSize)), DoubleOneMinusEpsilon);
}

template <typename R>
PBRT_CPU_GPU inline Float SobolSample(int64_t index, int dimension, R randomizer) {
#ifdef PBRT_FLOAT_AS_DOUBLE
    return SobolSampleDouble(index, dimension, randomizer);
#else
    return SobolSampleFloat(index, dimension, randomizer);
#endif
}

}  // namespace pbrt

#endif  // PBRT_LOWDISCREPANCY_H
