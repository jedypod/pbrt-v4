
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/util/float.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>
#include <pbrt/samplers.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/shuffle.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>

using namespace pbrt;

TEST(Sampling, InvertUniformHemisphere) {
    for (Point2f u : Uniform2D(1000)) {
        Vector3f v = SampleUniformHemisphere(u);
        Point2f up = InvertUniformHemisphereSample(v);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertCosineHemisphere) {
    for (Point2f u : Uniform2D(1000)) {
        Vector3f v = SampleCosineHemisphere(u);
        Point2f up = InvertCosineHemisphereSample(v);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertUniformSphere) {
    for (Point2f u : Uniform2D(1000)) {
        Vector3f v = SampleUniformSphere(u);
        Point2f up = InvertUniformSphereSample(v);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertUniformTriangle) {
    for (Point2f u : Uniform2D(1000)) {
        pstd::array<Float, 3> b = SampleUniformTriangle(u);
        Point2f up = InvertUniformTriangleSample(b);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << b[0] << ", " << b[1] << ", " << b[2] << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << b[0] << ", " << b[1] << ", " << b[2] << " -> " << up;
    }
}

TEST(Sampling, InvertUniformCone) {
    RNG rng;
    for (Point2f u : Uniform2D(1000)) {
        Float cosThetaMax = rng.Uniform<Float>();
        Vector3f v = SampleUniformCone(u, cosThetaMax);
        Point2f up = InvertUniformConeSample(v, cosThetaMax);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << v << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << v << " -> " << up;
    }
}

TEST(Sampling, InvertUniformDiskPolar) {
    for (Point2f u : Uniform2D(1000)) {
        Point2f p = SampleUniformDiskPolar(u);
        Point2f up = InvertUniformDiskPolarSample(p);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << p << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << p << " -> " << up;
    }
}

TEST(Sampling, InvertUniformDiskConcentric) {
    for (Point2f u : Uniform2D(1000)) {
        Point2f p = SampleUniformDiskConcentric(u);
        Point2f up = InvertUniformDiskConcentricSample(p);

        EXPECT_LT(std::abs(u.x - up.x), 1e-3) << "u " << u << " -> " << p << " -> " << up;
        EXPECT_LT(std::abs(u.y - up.y), 1e-3) << "u " << u << " -> " << p << " -> " << up;
    }
}

TEST(LowDiscrepancy, RadicalInverse) {
    for (int a = 0; a < 1024; ++a) {
        EXPECT_EQ(ReverseBits32(a) * 2.3283064365386963e-10f,
                  RadicalInverse(0, a));
    }
}

#if 0
TEST(LowDiscrepancy, ScrambledRadicalInverse) {
    for (int dim = 0; dim < 128; ++dim) {
        RNG rng(dim);
        // Random permutation table
        const int base = Primes[dim];

        std::vector<uint16_t> perm;
        for (int i = 0; i < base; ++i) perm.push_back(base - 1 - i);
        Shuffle(pstd::MakeSpan(perm), rng);

        for (const uint32_t index : {0, 1, 2, 1151, 32351, 4363211, 681122}) {
            // First, compare to the pbrt-v2 implementation.
            {
                Float val = 0;
                Float invBase = 1. / base, invBi = invBase;
                uint32_t n = index;
                while (n > 0) {
                    uint32_t d_i = perm[n % base];
                    val += d_i * invBi;
                    n *= invBase;
                    invBi *= invBase;
                }
                // For the case where the permutation table permutes the digit 0
                // to
                // another digit, account for the infinite sequence of that
                // digit
                // trailing at the end of the radical inverse value.
                val += perm[0] * base / (base - 1.0f) * invBi;

                EXPECT_NEAR(val, ScrambledRadicalInverse(dim, index, perm),
                            1e-5);
            }

            {
                // Now also check against a totally naive "loop over all the
                // bits in
                // the index" approach, regardless of hitting zero...
                Float val = 0;
                Float invBase = 1. / base, invBi = invBase;

                uint32_t a = index;
                for (int i = 0; i < 32; ++i) {
                    uint32_t d_i = perm[a % base];
                    a /= base;
                    val += d_i * invBi;
                    invBi *= invBase;
                }
                EXPECT_NEAR(val, ScrambledRadicalInverse(dim, index, perm),
                            1e-5);
            }
        }
    }
}
#endif

TEST(LowDiscrepancy, GeneratorMatrix) {
    uint32_t C[32];
    uint32_t Crev[32];
    // Identity matrix, column-wise
    for (int i = 0; i < 32; ++i) {
        C[i] = 1 << i;
        Crev[i] = ReverseBits32(C[i]);
    }

    for (int a = 0; a < 128; ++a) {
        // Make sure identity generator matrix matches van der Corput
        EXPECT_EQ(a, MultiplyGenerator(C, a));
        EXPECT_EQ(RadicalInverse(0, a), ReverseBits32(MultiplyGenerator(C, a)) *
                                            2.3283064365386963e-10f);
        EXPECT_EQ(RadicalInverse(0, a), SampleGeneratorMatrix(Crev, a));
    }

    // Random / goofball generator matrix
    RNG rng;
    for (int i = 0; i < 32; ++i) {
        C[i] = rng.Uniform<uint32_t>();
        Crev[i] = ReverseBits32(C[i]);
    }
    for (int a = 0; a < 1024; ++a) {
        EXPECT_EQ(ReverseBits32(MultiplyGenerator(C, a)),
                  MultiplyGenerator(Crev, a));
    }
}

TEST(LowDiscrepancy, Sobol) {
    // Check that float and double variants match (as float values).
    for (int i = 0; i < 256; ++i) {
        for (int dim = 0; dim < 100; ++dim) {
            EXPECT_EQ(SobolSampleFloat(i, dim, NoRandomizer()),
                      (float)SobolSampleDouble(i, dim, NoRandomizer()));
        }
    }

    // Make sure first dimension is the regular base 2 radical inverse
    for (int i = 0; i < 8192; ++i) {
        EXPECT_EQ(SobolSampleFloat(i, 0, NoRandomizer()),
                  ReverseBits32(i) * 2.3283064365386963e-10f);
    }
}

// Make sure all samplers give the same sample values if we go back to the
// same pixel / sample index.
TEST(Sampler, ConsistentValues) {
    constexpr int rootSpp = 4;
    constexpr int spp = rootSpp * rootSpp;
    Point2i resolution(100, 101);

    std::vector<SamplerHandle> samplers;
    samplers.push_back(new HaltonSampler(spp, resolution));
    samplers.push_back(new RandomSampler(spp));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::None));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::CranleyPatterson));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::Xor));
    samplers.push_back(new PaddedSobolSampler(spp, RandomizeStrategy::Owen));
    samplers.push_back(new PMJ02BNSampler(spp));
    samplers.push_back(new StratifiedSampler(rootSpp, rootSpp, true));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::None));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::CranleyPatterson));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::Xor));
    samplers.push_back(new SobolSampler(spp, resolution, RandomizeStrategy::Owen));

    for (auto &sampler : samplers) {
        std::vector<Float> s1d[spp];
        std::vector<Point2f> s2d[spp];

        for (int s = 0; s < spp; ++s) {
            sampler.StartPixelSample({1, 5}, s);
            for (int i = 0; i < 10; ++i) {
                s2d[s].push_back(sampler.Get2D());
                s1d[s].push_back(sampler.Get1D());
            }
        }

        // Go somewhere else and generate some samples, just to make sure
        // things are shaken up.
        sampler.StartPixelSample({0, 6}, 10);
        sampler.Get2D();
        sampler.Get2D();
        sampler.Get1D();

        // Now go back and generate samples again, but enumerate them in a
        // different order to make sure the sampler is doing the right
        // thing.
        for (int s = spp - 1; s >= 0; --s) {
            sampler.StartPixelSample({1, 5}, s);
            for (int i = 0; i < s2d[s].size(); ++i) {
                EXPECT_EQ(s2d[s][i], sampler.Get2D());
                EXPECT_EQ(s1d[s][i], sampler.Get1D());
            }
        }
    }
}

static void checkElementary(const char *name, std::vector<Point2f> samples,
                            int logSamples) {
    for (int i = 0; i <= logSamples; ++i) {
        // Check one set of elementary intervals: number of intervals
        // in each dimension.
        int nx = 1 << i, ny = 1 << (logSamples - i);

        std::vector<int> count(1 << logSamples, 0);
        for (const Point2f &s : samples) {
            // Map the sample to an interval
            Float x = nx * s.x, y = ny * s.y;
            EXPECT_GE(x, 0);
            EXPECT_LT(x, nx);
            EXPECT_GE(y, 0);
            EXPECT_LT(y, ny);
            int index = (int)std::floor(y) * nx + (int)std::floor(x);
            EXPECT_GE(index, 0);
            EXPECT_LT(index, count.size());

            // This should be the first time a sample has landed in its
            // interval.
            EXPECT_EQ(0, count[index]) << "Sampler " << name << " with interval " <<
                nx << " x " << ny;
            ++count[index];
        }
    }
}

static void checkElementarySampler(const char *name, SamplerHandle sampler,
                                   int logSamples) {
    // Get all of the samples for a pixel.
    int spp = sampler.SamplesPerPixel();
    std::vector<Point2f> samples;
    for (int i = 0; i < spp; ++i) {
        sampler.StartPixelSample(Point2i(0, 0), i);
        samples.push_back(sampler.Get2D());
    }

    checkElementary(name, samples, logSamples);
}

// TODO: check Halton (where the elementary intervals are (2^i, 3^j)).

TEST(PaddedSobolSampler, ElementaryIntervals) {
    for (auto rand : { RandomizeStrategy::None, RandomizeStrategy::Owen,
                       RandomizeStrategy::Xor })
        for (int logSamples = 2; logSamples <= 10; ++logSamples)
            checkElementarySampler(
                "PaddedSobolSampler",
                new PaddedSobolSampler(1 << logSamples, rand),
                logSamples);
}

TEST(SobolUnscrambledSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; ++logSamples)
        checkElementarySampler(
            "Sobol Unscrambled",
            new SobolSampler(1 << logSamples, Point2i(1, 1), RandomizeStrategy::None),
            logSamples);
}

TEST(SobolXORScrambledSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; ++logSamples)
        checkElementarySampler(
            "Sobol XOR Scrambled",
            new SobolSampler(1 << logSamples, Point2i(1, 1), RandomizeStrategy::Xor),
            logSamples);
}

TEST(SobolOwenScrambledSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; ++logSamples)
        checkElementarySampler(
            "Sobol Owen Scrambled",
            new SobolSampler(1 << logSamples, Point2i(1, 1), RandomizeStrategy::Owen),
            logSamples);
}

TEST(PMJ02BNSampler, ElementaryIntervals) {
    for (int logSamples = 2; logSamples <= 10; logSamples += 2)
        checkElementarySampler(
            "PMJ02BNSampler",
            new PMJ02BNSampler(1 << logSamples), logSamples);
}

TEST(CranleyPattersonRotator, Basics) {
    auto toFixed = [](Float v) { return uint32_t(v * 0x1p+32); };
    auto fromFixed = [](uint32_t v) { return Float(v) * 0x1p-32; };
    EXPECT_EQ(0, toFixed(0));
    EXPECT_EQ(0x80000000, toFixed(0.5f));
    EXPECT_EQ(0x40000000, toFixed(0.25f));
    EXPECT_EQ(fromFixed(0x80000000), 0.5f);
    EXPECT_EQ(fromFixed(0xc0000000), 0.75f);
    for (int i = 1; i < 31; ++i) {
        Float v = 1.f / (1 << i);
        EXPECT_EQ(toFixed(v), 1u << (32 - i));
        EXPECT_EQ(fromFixed(1u << (32 - i)), v);
    }

    EXPECT_EQ(toFixed(0.5), CranleyPattersonRotator(0.5f)(0));
    EXPECT_EQ(toFixed(0.5), CranleyPattersonRotator(0.25f)(toFixed(0.25)));
    EXPECT_EQ(toFixed(0.5), CranleyPattersonRotator(toFixed(0.25f))(toFixed(0.25)));
    EXPECT_EQ(toFixed(0.75), CranleyPattersonRotator(toFixed(0.5f))(toFixed(0.25)));
    EXPECT_EQ(toFixed(0.375f), CranleyPattersonRotator(toFixed(0.25f))(toFixed(0.125)));
}

TEST(Sobol, IntervalToIndex) {
    for (int logRes = 0; logRes < 8; ++logRes) {
        int res = 1 << logRes;
        for (int y = 0; y < res; ++y) {
            for (int x = 0; x < res; ++x) {
                // For each pixel sample.
                bool sawCorner = false;
                for (int s = 0; s < 16; ++s) {
                    uint64_t seqIndex = SobolIntervalToIndex(logRes, s, {x, y});
                    Point2f samp(SobolSample(seqIndex, 0, NoRandomizer()),
                                 SobolSample(seqIndex, 1, NoRandomizer()));
                    Point2f ss(samp[0] * res, samp[1] * res);
                    // Make sure all samples aren't landing at the
                    // lower-left corner.
                    if (ss == Point2f(x, y)) {
                        EXPECT_FALSE(sawCorner) << "Multi corner " << Point2i(x, y) <<
                            ", res " << res << ", samp " << s << ", seq index " <<
                            seqIndex << ", got " << ss << ", from samp " << samp;
                        sawCorner = true;
                    }
                    // Most of the time, the Sobol sample should be within the
                    // pixel's extent. Due to rounding, it may rarely be at the
                    // upper end of the extent; the check here is written carefully
                    // to only accept points just at the upper limit but not into
                    // the next pixel.
                    EXPECT_TRUE(Point2i(x, y) == Point2i(ss) ||
                                (x == int(ss.x)       && Float(y + 1) == ss.y ||
                                 Float(x + 1) == ss.x && y == int(ss.y))) <<
                        "Pixel " << Point2i(x, y) << ", sample " << s << ", got " << ss;
                }
            }
        }
    }
}

TEST(Sobol, IntervalToIndexRandoms) {
    RNG rng;
    for (int i = 0; i < 100000; ++i) {
        int logRes = rng.Uniform<int>(16);
        int res = 1 << logRes;
        int x = rng.Uniform<int>(res), y = rng.Uniform<int>(res);
        int s = rng.Uniform<int>(8192);

        uint64_t seqIndex = SobolIntervalToIndex(logRes, s, {x, y});
        Point2f samp(SobolSample(seqIndex, 0, NoRandomizer()),
                     SobolSample(seqIndex, 1, NoRandomizer()));
        Point2f ss(int(samp[0] * res), int(samp[1] * res));
        // Most of the time, the Sobol sample should be within the
        // pixel's extent. Due to rounding, it may rarely be at the
        // upper end of the extent; the check here is written carefully
        // to only accept points just at the upper limit but not into
        // the next pixel.
        EXPECT_TRUE(Point2i(x, y) == Point2i(ss) ||
                    (x == int(ss.x)       && Float(y + 1) == ss.y ||
                     Float(x + 1) == ss.x && y == int(ss.y))) <<
            "Pixel " << Point2i(x, y) << ", sample " << s << ", got " << ss;
    }
}

TEST(Distribution1D, Discrete) {
    // Carefully chosen distribution so that transitions line up with
    // (inverse) powers of 2.
    Distribution1D dist({0.f, 1.f, 0.f, 3.f});
    EXPECT_EQ(4, dist.size());

    EXPECT_EQ(0, dist.DiscretePDF(0));
    EXPECT_EQ(.25, dist.DiscretePDF(1));
    EXPECT_EQ(0, dist.DiscretePDF(2));
    EXPECT_EQ(.75, dist.DiscretePDF(3));

    Float pdf, uRemapped;
    EXPECT_EQ(1, dist.SampleDiscrete(0., &pdf));
    EXPECT_EQ(0.25, pdf);
    EXPECT_EQ(1, dist.SampleDiscrete(0.125, &pdf, &uRemapped));
    EXPECT_EQ(0.25, pdf);
    EXPECT_FLOAT_EQ(0.5, uRemapped);
    EXPECT_EQ(1, dist.SampleDiscrete(.24999, &pdf));
    EXPECT_EQ(0.25, pdf);
    EXPECT_EQ(3, dist.SampleDiscrete(.250001, &pdf));
    EXPECT_EQ(0.75, pdf);
    EXPECT_EQ(3, dist.SampleDiscrete(0.625, &pdf, &uRemapped));
    EXPECT_EQ(0.75, pdf);
    EXPECT_FLOAT_EQ(0.5, uRemapped);
    EXPECT_EQ(3, dist.SampleDiscrete(OneMinusEpsilon, &pdf));
    EXPECT_EQ(0.75, pdf);
    EXPECT_EQ(3, dist.SampleDiscrete(1., &pdf));
    EXPECT_EQ(0.75, pdf);

    // Compute the interval to test over.
    Float u = .25, uMax = .25;
    for (int i = 0; i < 20; ++i) {
        u = NextFloatDown(u);
        uMax = NextFloatUp(uMax);
    }
    // We should get a stream of hits in the first interval, up until the
    // cross-over point at 0.25 (plus/minus fp slop).
    for (; u < uMax; u = NextFloatUp(u)) {
        int interval = dist.SampleDiscrete(u);
        if (interval == 3) break;
        EXPECT_EQ(1, interval);
    }
    EXPECT_LT(u, uMax);
    // And then all the rest should be in the third interval
    for (; u <= uMax; u = NextFloatUp(u)) {
        int interval = dist.SampleDiscrete(u);
        EXPECT_EQ(3, interval);
    }
}

TEST(Distribution1D, Continuous) {
    Distribution1D dist({1.f, 1.f, 2.f, 4.f, 8.f});
    EXPECT_EQ(5, dist.size());

    Float pdf;
    int offset;
    EXPECT_EQ(0., dist.SampleContinuous(0., &pdf, &offset));
    EXPECT_FLOAT_EQ(dist.size() * 1. / 16., pdf);
    EXPECT_EQ(0, offset);

    // Right at the bounary between the 4 and the 8 segments.
    EXPECT_FLOAT_EQ(.8, dist.SampleContinuous(0.5, &pdf, &offset));

    // Middle of the 8 segment
    EXPECT_FLOAT_EQ(.9, dist.SampleContinuous(0.75, &pdf, &offset));
    EXPECT_FLOAT_EQ(dist.size() * 8. / 16., pdf);
    EXPECT_EQ(4, offset);

    EXPECT_FLOAT_EQ(0., dist.SampleContinuous(0., &pdf));
    EXPECT_FLOAT_EQ(1., dist.SampleContinuous(1., &pdf));
}

TEST(Distribution1D, Range) {
    auto values = Sample1DFunction([](Float x) { return 1 + x; }, 65536, 4, -1.f, 3.f);
    Distribution1D dist(values, -1.f, 3.f);
    // p(x) = (1+x) / 8
    // xi = int_{-1}^x p(x) ->
    // xi = 1/16 x^2 + x/8 + 1/16 ->
    // Solve 0 = 1/16 x^2 + x/8 + 1/16 - xi to sample

    for (Float u : Uniform1D(100)) {
        Float pd;
        Float xd = dist.SampleContinuous(u, &pd);

        Float t0, t1;
        ASSERT_TRUE(Quadratic(1./16., 1./8., 1./16 - u, &t0, &t1));
        Float xa = (t0 >= -1 && t0 <= 3) ? t0 : t1;
        Float pa = (1 + xa) / 8;

        EXPECT_LT(std::abs(xd - xa) / xa, 2e-3) << xd << " vs " << xa;
        EXPECT_LT(std::abs(pd - pa) / pa, 2e-3) << pd << " vs " << pa;
    }
}

TEST(Distribution1D, FromFuncL1) {
    auto f = [](Float v) { return v * v; };
    auto values = Sample1DFunction(f, 8, 1024*1024, 0.f, 1.f, Norm::L1);
    Distribution1D dSampled(values);

    std::vector<Float> exact = { Float(0.00195313),
                                 Float(0.0136719),
                                 Float(0.0371094),
                                 Float(0.0722656),
                                 Float(0.119141),
                                 Float(0.177734),
                                 Float(0.248047),
                                 Float(0.330078) };
    Distribution1D dExact(exact);
    Distribution1D::TestCompareDistributions(dSampled, dExact);
}

TEST(Distribution1D, FromFuncL2) {
    auto f = [](Float v) { return v * v; };
    auto values = Sample1DFunction(f, 8, 1024*1024, 0.f, 1.f, Norm::L2);
    Distribution1D dSampled(values);

    std::vector<Float> exact = { Float(0.00552427),
                                 Float(0.0307578),
                                 Float(0.0802447),
                                 Float(0.154383),
                                 Float(0.253214),
                                 Float(0.376746),
                                 Float(0.52498),
                                 Float(0.697919) };

    Distribution1D dExact(exact);
    Distribution1D::TestCompareDistributions(dSampled, dExact);
}

TEST(Distribution1D, FromFuncLInfinity) {
    auto f = [](Float v) { return v * v; };
    auto values = Sample1DFunction(f, 8, 1024*1024, 0.f, 1.f, Norm::LInfinity);
    Distribution1D dSampled(values);

    // pdf = 3v^2
    std::vector<Float> exact = { 3 * f(1./8.),
                                 3 * f(2./8.),
                                 3 * f(3./8.),
                                 3 * f(4./8.),
                                 3 * f(5./8.),
                                 3 * f(6./8.),
                                 3 * f(7./8.),
                                 3 * f(8./8.) };

    Distribution1D dExact(exact);
    Distribution1D::TestCompareDistributions(dSampled, dExact);
}

TEST(Distribution1D, InverseUniform) {
    std::vector<Float> values = { Float(1), Float(1), Float(1) };

    Distribution1D dist(values);
    EXPECT_EQ(0, *dist.Inverse(0));
    EXPECT_EQ(0.5, *dist.Inverse(0.5));
    EXPECT_EQ(0.75, *dist.Inverse(0.75));

    Distribution1D dist2(values, -1, 3);
    EXPECT_EQ(0, *dist2.Inverse(-1));
    EXPECT_EQ(0.25, *dist2.Inverse(0));
    EXPECT_EQ(0.5, *dist2.Inverse(1));
    EXPECT_EQ(0.75,* dist2.Inverse(2));
    EXPECT_EQ(1, *dist2.Inverse(3));
}

TEST(Distribution1D, InverseGeneral) {
    std::vector<Float> values = { Float(0), Float(0.25), Float(0.5), Float(0.25) };

    Distribution1D dist(values);
    EXPECT_EQ(0, *dist.Inverse(0));
    EXPECT_EQ(1, *dist.Inverse(1));
    EXPECT_EQ(0.25, *dist.Inverse(0.5));
    EXPECT_EQ(0.5, *dist.Inverse(0.625));
    EXPECT_EQ(0.75, *dist.Inverse(0.75));
    EXPECT_FLOAT_EQ(0.825, *dist.Inverse(0.825));

    Distribution1D dist2(values, -1, 3);
    EXPECT_EQ(0, *dist2.Inverse(-1));
    EXPECT_EQ(1, *dist2.Inverse(3));
    EXPECT_EQ(0.25, *dist2.Inverse(Lerp(0.5, -1, 3)));
    EXPECT_EQ(0.5, *dist2.Inverse(Lerp(0.625, -1, 3)));
    EXPECT_EQ(0.75, *dist2.Inverse(Lerp(0.75, -1, 3)));
    EXPECT_FLOAT_EQ(0.825, *dist2.Inverse(Lerp(0.825, -1, 3)));
}

TEST(Distribution1D, InverseRandoms) {
    std::vector<Float> values = { Float(0), Float(1.25), Float(0.5), Float(0.25), Float(3.7) };

    Distribution1D dist(values);
    for (Float u : Uniform1D(100)) {
        Float v = dist.SampleContinuous(u);
        auto inv = dist.Inverse(v);
        ASSERT_TRUE(inv.has_value());
        Float err = std::min(std::abs(*inv - u), std::abs(*inv - u) / u);
        EXPECT_LT(err, 1e-4) << "u " << u << " vs inv " << *inv;
    }

    Distribution1D dist2(values, -1, 3);
    for (Float u : Uniform1D(100)) {
        Float v = dist.SampleContinuous(u);
        auto inv = dist.Inverse(v);
        ASSERT_TRUE(inv.has_value());
        Float err = std::min(std::abs(*inv - u), std::abs(*inv - u) / u);
        EXPECT_LT(err, 1e-4) << "u " << u << " vs inv " << *inv;
    }
}

TEST(Distribution2D, InverseUniform) {
    std::vector<Float> values = { Float(1), Float(1), Float(1),
                                  Float(1), Float(1), Float(1) };

    Distribution2D dist(values, 3, 2);
    EXPECT_EQ(Point2f(0, 0), *dist.Inverse(Point2f(0, 0)));
    EXPECT_EQ(Point2f(1, 1), *dist.Inverse(Point2f(1, 1)));
    EXPECT_EQ(Point2f(0.5, 0.5), *dist.Inverse(Point2f(0.5, 0.5)));
    EXPECT_EQ(Point2f(0.25, 0.75), *dist.Inverse(Point2f(0.25, 0.75)));

    Bounds2f domain(Point2f(-1, -0.5), Point2f(3, 1.5));
    Distribution2D dist2(values, 3, 2, domain);
    EXPECT_EQ(Point2f(0, 0), *dist2.Inverse(domain.Lerp(Point2f(0, 0))));
    EXPECT_EQ(Point2f(1, 1), *dist2.Inverse(domain.Lerp(Point2f(1, 1))));
    EXPECT_EQ(Point2f(0.5, 0.5), *dist2.Inverse(domain.Lerp(Point2f(0.5, 0.5))));
    EXPECT_EQ(Point2f(0.25, 0.75), *dist2.Inverse(domain.Lerp(Point2f(0.25, 0.75))));
}

TEST(Distribution2D, InverseRandoms) {
    int nx = 4, ny = 5;
    std::vector<Float> values;
    RNG rng;
    for (int i = 0; i < nx * ny; ++i)
        values.push_back(rng.Uniform<Float>());

    Distribution2D dist(values, nx, ny);
    for (Point2f u : Uniform2D(100)) {
        Point2f v = dist.SampleContinuous(u);
        auto inv = dist.Inverse(v);
        ASSERT_TRUE(inv.has_value());
        Point2f err(std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
                    std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
        EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
        EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
    }

    Bounds2f domain(Point2f(-1, -0.5), Point2f(3, 1.5));
    Distribution2D dist2(values, nx, ny, domain);
    for (Point2f u : Uniform2D(100, 235351)) {
        Point2f v = dist2.SampleContinuous(u);
        auto inv = dist2.Inverse(v);
        ASSERT_TRUE(inv.has_value());
        Point2f err(std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
                    std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
        EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
        EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
    }
}

TEST(Distribution2D, FromFuncL1) {
    auto f = [](Float x, Float y) { return x * x * y; }; // x^2 y
    auto values = Sample2DFunction(f, 4, 2, 1024*1024, Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1);
    Distribution2D dSampled(values, 4, 2);

    std::vector<Float> exact = { Float(0.0006510416653645833),
                                 Float(0.004557291657552083),
                                 Float(0.012369791641927086),
                                 Float(0.024088541618489584),
                                 Float(0.0019531249960937497),
                                 Float(0.013671874972656246),
                                 Float(0.037109374925781244),
                                 Float(0.07226562485546874) };
    Distribution2D dExact(exact, 4, 2);
    Distribution2D::TestCompareDistributions(dSampled, dExact);
}

TEST(Distribution2D, FromFuncL2) {
    auto f = [](Float x, Float y) { return x * x * y; };
    auto values = Sample2DFunction(f, 4, 2, 1024*1024, Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L2);
    Distribution2D dSampled(values, 4, 2);

    std::vector<Float> exact = { Float(0.002852721649393658),
                                 Float(0.015883281936567687),
                                 Float(0.04143817552308458),
                                 Float(0.07972323533177735),
                                 Float(0.0075475920472202924),
                                 Float(0.04202321402569243),
                                 Float(0.10963510726531212),
                                 Float(0.2109278544917584) };
    Distribution2D dExact(exact, 4, 2);
    Distribution2D::TestCompareDistributions(dSampled, dExact);
}

TEST(Distribution2D, FromFuncLInfinity) {
    auto f = [](Float x, Float y) { return x * x * y; };
    auto values = Sample2DFunction(f, 4, 2, 1, Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::LInfinity);
    Distribution2D dSampled(values, 4, 2);

    std::vector<Float> exact = { Float(Sqr(0.25) * Float(0.5)),
                                 Float(Sqr(0.5) * Float(0.5)),
                                 Float(Sqr(0.75) * Float(0.5)),
                                 Float(Sqr(1) * Float(0.5)),
                                 Float(Sqr(0.25) * Float(1)),
                                 Float(Sqr(0.5) * Float(1)),
                                 Float(Sqr(0.75) * Float(1)),
                                 Float(Sqr(1) * Float(1)) };
    Distribution2D dExact(exact, 4, 2);
    Distribution2D::TestCompareDistributions(dSampled, dExact);
}

TEST(Sampling, SampleDiscrete1D) {
    Float pdf, uRemapped;

    EXPECT_EQ(1, SampleDiscrete({1, 1}, .61, &pdf, &uRemapped));
    EXPECT_EQ(0.5, pdf);
    EXPECT_FLOAT_EQ(.22, uRemapped);

    EXPECT_EQ(0, SampleDiscrete({10, 10}, .4999, &pdf, &uRemapped));
    EXPECT_EQ(0.5, pdf);
    EXPECT_FLOAT_EQ(.9998, uRemapped);

    EXPECT_EQ(0, SampleDiscrete({67, 33}, .66669, &pdf));
    EXPECT_FLOAT_EQ(0.67, pdf);

    EXPECT_EQ(1, SampleDiscrete({67, 33}, .672, &pdf));
    EXPECT_FLOAT_EQ(.33, pdf);

    EXPECT_EQ(0, SampleDiscrete({1, 1, 1, 1}, .24, &pdf, &uRemapped));
    EXPECT_EQ(0.25, pdf);
    EXPECT_FLOAT_EQ(.96, uRemapped);

    EXPECT_EQ(1, SampleDiscrete({1, 1, 1, 1}, .251, &pdf, &uRemapped));
    EXPECT_EQ(0.25, pdf);
    EXPECT_NEAR(.004, uRemapped, 1e-5);

    EXPECT_EQ(2, SampleDiscrete({.125, .125, .625, .125}, .251, &pdf, &uRemapped));
    EXPECT_EQ(5./8., pdf);
    EXPECT_NEAR(.001 / .625, uRemapped, 1e-5);

    EXPECT_EQ(2, SampleDiscrete({1, 1, 5, 1}, .251, &pdf, &uRemapped));
    EXPECT_EQ(5./8., pdf);
    EXPECT_NEAR(.001 * 8. / 5., uRemapped, 1e-5);

    EXPECT_EQ(0, SampleDiscrete({1, 1, 5, 1}, .124, &pdf));
    EXPECT_EQ(1./8., pdf);
}

TEST(Sampling, SampleDiscrete2D) {
    Float values[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    constexpr int nValues = PBRT_ARRAYSIZE(values);
    Float sum = std::accumulate(values, values + nValues, Float(0));
    int nuv[][2] = { { 9, 1 }, { 1, 9 }, { 3, 3 } };

    RNG rng;
    for (const auto uv : nuv) {
        // Make sure we get roughly the expected number of samples in each
        // bucket.
        Distribution2D distrib(values, uv[0], uv[1]);
        pstd::array<int, nValues> counts = { 0 };
        int n = 256*1024;
        for (int i = 0; i < n; ++i) {
            Point2f u{rng.Uniform<Float>(), rng.Uniform<Float>()};
            Float pdf;
            Point2i p = distrib.SampleDiscrete(u, &pdf);
            EXPECT_LT(std::abs(pdf - distrib.DiscretePDF(p)), 1e-5);
            int offset = p[0] + p[1] * uv[0];
            EXPECT_LT(std::abs(pdf - values[offset] / sum), 1e-5);
            counts[offset]++;
        }
        for (int i = 0; i < nValues; ++i) {
            if (values[i] == 0)
                EXPECT_EQ(0, counts[i]);
            else {
                EXPECT_GT(counts[i], .98 * n * values[i] / sum);
                EXPECT_LT(counts[i], 1.02 * n * values[i] / sum);
            }
        }
    }
}

TEST(Sampling, SphericalTriangle) {
    int count = 1024*1024;
    pstd::array<Point3f, 3> v = { Point3f(4, 1, 1), Point3f(-10, 3, 3), Point3f(-2, -8, 10) };
    Float A = 0.5 * Length(Cross(v[1] - v[0], v[2] - v[0]));
    Vector3f N = Normalize(Cross(v[1] - v[0], v[2] - v[0]));
    Point3f p(.5, -.4, .7);

    // Integrate this function over the projection of the triangle given by
    // |v| at the unit sphere surrounding |p|.
    auto f = [](Point3f p) { return p.x * p.y * p.z; };

    Float sphSum = 0, areaSum = 0;
    for (int i = 0; i < count; ++i) {
        Point2f u(RadicalInverse(0, i), RadicalInverse(1, i));

        Float pdf;
        pstd::array<Float, 3> bs = SampleSphericalTriangle(v, p, u, &pdf);
        Point3f pTri = bs[0] * v[0] + bs[1] * v[1] + bs[2] * v[2];
        sphSum += f(pTri) / pdf;

        pstd::array<Float, 3> ba = SampleUniformTriangle(u);
        pdf = 1 / A;
        pTri = ba[0] * v[0] + ba[1] * v[1] + (1 - ba[0] - ba[1]) * v[2];
        areaSum += f(pTri) * AbsDot(N, Normalize(pTri - p)) /
            (pdf * DistanceSquared(p, pTri));
    }
    Float sphInt = sphSum / count;
    Float areaInt = areaSum / count;

    EXPECT_LT(std::abs(areaInt - sphInt), 1e-3);
}

TEST(Sampling, SphericalTriangleInverse) {
    RNG rng;
    auto rp = [&rng](Float low = -10, Float high = 10) {
                  return Point3f(Lerp(rng.Uniform<Float>(), low, high),
                                 Lerp(rng.Uniform<Float>(), low, high),
                                 Lerp(rng.Uniform<Float>(), low, high));
              };

    for (int i = 0; i < 10; ++i) {
        pstd::array<Point3f, 3> v = { rp(), rp(), rp() };
        Point3f p = rp(-1, 1);
        for (int j = 0; j < 10; ++j) {
            Float pdf;
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            pstd::array<Float, 3> bs = SampleSphericalTriangle(v, p, u, &pdf);

            Point3f pTri = bs[0] * v[0] + bs[1] * v[1] + bs[2] * v[2];

            Point2f ui = InvertSphericalTriangleSample(v, p, Normalize(pTri - p));

            auto err = [](Float a, Float ref) {
                           if (ref < 1e-3) return std::abs(a - ref);
                           else return std::abs((a - ref) / ref);
                       };
            // The tolerance has to be fiarly high, unfortunately...
            EXPECT_LT(err(ui[0], u[0]), 0.0025) << u << " vs inverse " << ui;
            EXPECT_LT(err(ui[1], u[1]), 0.0025) << u << " vs inverse " << ui;
        }
    }
}

TEST(Sampling, SphericalQuad) {
    int count = 1024*1024;
    pstd::array<Point3f, 4> v = { Point3f(4, 1, 1), Point3f(6, 1, -2),
                                 Point3f(4, 4, 1), Point3f(6, 4, -2) };
    Float A = Length(v[0] - v[1]) * Length(v[0] - v[2]);
    Vector3f N = Normalize(Cross(v[1] - v[0], v[2] - v[0]));
    Point3f p(.5, -.4, .7);

    // Integrate this function over the projection of the quad given by
    // |v| at the unit sphere surrounding |p|.
    auto f = [](Point3f p) { return p.x * p.y * p.z; };

    Float sphSum = 0, areaSum = 0;
    for (Point2f u : Hammersley2D(count)) {
        Float pdf;
        Point3f pq = SampleSphericalQuad(p, v[0], v[1] - v[0], v[2] - v[0], u, &pdf);
        sphSum += f(pq) / pdf;

        pq = Lerp(u[1], Lerp(u[0], v[0], v[1]), Lerp(u[0], v[2], v[3]));
        pdf = 1 / A;
        areaSum += f(pq) * AbsDot(N, Normalize(pq - p)) /
            (pdf * DistanceSquared(p, pq));
    }
    Float sphInt = sphSum / count;
    Float areaInt = areaSum / count;

    EXPECT_LT(std::abs(areaInt - sphInt), 1e-3) << "area " << areaInt << " sph " << sphInt;
}

TEST(Sampling, SphericalQuadInverse) {
#if 0
    LOG(WARNING) << "bits " << int(FloatToBits(0.00026721583f) - FloatToBits(0.00026713056f));
    {
    Point3f p( 5.5154743, -6.8645816, -1.2982006), s(6, 1, -2);
    Vector3f ex( 0, 3, 0), ey( -2, 0, 3);
    Point2f u(0.031906128, 0.82836914);

    Point3f pq = SampleSphericalQuad(p, s, ex, ey, u);
    Point2f ui = InvertSphericalQuadSample(p, s, ex, ey, pq);
    EXPECT_EQ(u, ui);
    }

    {
        Point3f p(-1.8413692, 3.8777208, 9.158957), s( 6, 4, -2);
        Vector3f ex(0, -3, 0), ey( -2, 0, 3);
        Point2f u (0.11288452, 0.40319824 );
        Point3f pq = SampleSphericalQuad(p, s, ex, ey, u);
        Point2f ui = InvertSphericalQuadSample(p, s, ex, ey, pq);
        EXPECT_EQ(u, ui);
    }
//CO    return;
#endif

    int count = 64*1024;
    Point3f v[2][2] = { { Point3f(4, 1, 1), Point3f(6, 1, -2) },
                        { Point3f(4, 4, 1), Point3f(6, 4, -2) } };

    RNG rng;
    int nTested = 0;
    for (Point2f u : Hammersley2D(count)) {
        int a = rng.Uniform<int>() & 1, b = rng.Uniform<int>() & 1;

        Point3f p(Lerp(rng.Uniform<Float>(), -10, 10),
                  Lerp(rng.Uniform<Float>(), -10, 10),
                  Lerp(rng.Uniform<Float>(), -10, 10));
        Float pdf;
        Point3f pq = SampleSphericalQuad(p, v[a][b], v[!a][b] - v[a][b],
                                         v[a][!b] - v[a][b], u, &pdf);

        Float solidAngle = 1 / pdf;
        if (solidAngle < .01)
            continue;
        ++nTested;
        Point2f ui = InvertSphericalQuadSample(p, v[a][b], v[!a][b] - v[a][b],
                                               v[a][!b] - v[a][b], pq);

        auto err = [](Float a, Float ref) {
                       if (ref < 1e-2) return std::abs(a - ref);
                       else return std::abs((a - ref) / ref);
                   };
        // The tolerance has to be fairly high, unfortunately...
        // FIXME: super high for now to find the really bad cases...
        EXPECT_LT(err(ui[0], u[0]), 0.01) << u << " vs inverse " << ui << ", solid angle " << 1 / pdf;
        // y is pretty good at this point...
        EXPECT_LT(err(ui[1], u[1]), 0.001) << u << " vs inverse " << ui << ", solid angle " << 1 / pdf;
    }
    EXPECT_GT(nTested, count / 2);
}

TEST(Sampling, SmoothStep) {
    Float start = SampleSmoothStep(0, 10, 20);
    // Fairly high slop since lots of values close to the start are close
    // to zero.
    EXPECT_LT(std::abs(start - 10), .1) << start;

    Float end = SampleSmoothStep(1, -10, -5);
    EXPECT_LT(std::abs(end - -5), 1e-5) << end;

    Float mid = SampleSmoothStep(0.5, 0, 1);
    // Solved this numericalla in Mathematica.
    EXPECT_LT(std::abs(mid - 0.733615), 1e-5) << mid;

    for (Float u : Uniform1D(1000)) {
        Float x = SampleSmoothStep(u, -3, 5);
        Float ratio = SmoothStep(x, -3, 5) / SmoothStepPDF(x, -3, 5);
        // SmoothStep over [-3,5] integrates to 4.
        EXPECT_LT(std::abs(ratio - 4), 1e-5) << ratio;

        auto checkErr = [](Float a, Float b) {
                            Float err;
                            if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                            else err = std::abs(2 * (a - b) / (a + b));
                            return err > 1e-2;
                        };
        EXPECT_FALSE(checkErr(u, InvertSmoothStepSample(x, -3, 5)));
    }

    auto ss = [](Float v) { return SmoothStep(v, 0, 1); };
    auto values = Sample1DFunction(ss, 1024, 64*1024, 0.f, 1.f, Norm::L1);
    Distribution1D distrib(values);
    for (Float u : Uniform1D(100, 62351)) {
        Float cx = SampleSmoothStep(u, 0, 1);
        Float cp = SmoothStepPDF(cx, 0, 1);

        Float dp;
        Float dx = distrib.SampleContinuous(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
            ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
            ", distrib PDF = " << dp;
    }
}

TEST(Sampling, Linear) {
    int nBuckets = 32;
    std::vector<int> buckets(nBuckets, 0);

    int ranges[][2] = { { 0, 1 }, { 1, 2 }, {5, 50}, {100, 0}, {75, 50} };
    for (const auto r : ranges) {
        Float f0 = r[0], f1 = r[1];
        int nSamples = 1000000;
        for (int i = 0; i < nSamples; ++i) {
            Float u = (i + .5) / nSamples;
            Float t = SampleLinear(u, f0, f1);
            ++buckets[std::min<int>(t * nBuckets, nBuckets - 1)];
        }

        for (int i = 0; i < nBuckets; ++i) {
            int expected = Lerp(Float(i) / (nBuckets - 1), buckets[0],
                                buckets[nBuckets-1]);
            EXPECT_GE(buckets[i], .99 * expected);
            EXPECT_LE(buckets[i], 1.01 * expected);
        }
    }

    auto lin = [](Float v) { return 1 + 3 * v; };
    auto values = Sample1DFunction(lin, 1024, 64*1024, 0.f, 1.f, Norm::L1);
    Distribution1D distrib(values);
    for (Float u : Uniform1D(100)) {
        Float cx = SampleLinear(u, 1, 4);
        Float cp = LinearPDF(cx, 1, 4);

        Float dp;
        Float dx = distrib.SampleContinuous(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
            ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
            ", distrib PDF = " << dp;
    }

    RNG rng;
    for (Float u : Uniform1D(100)) {
        Float low = rng.Uniform<Float>() * 10;
        Float high = rng.Uniform<Float>() * 10;
        if (low < high) pstd::swap(low, high);
        Float x = SampleLinear(u, low, high);

        auto checkErr = [](Float a, Float b) {
                            Float err;
                            if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                            else err = std::abs(2 * (a - b) / (a + b));
                            return err > 1e-2;
                        };

        EXPECT_FALSE(checkErr(u, InvertLinearSample(x, low, high))) <<
            " u = " << u << " -> x " << x << " -> " <<
            InvertLinearSample(x, low, high) << " (over " << low <<
                " - " << high;
    }
}

TEST(Sampling, QuadraticA) {
    // parabola that just hits zero at x = 0.5, maximum of 0.25 at 0 and 1.
    Float a = 1, b = -1, c = 0.25;

    // Check the pdf
    for (Float x : Uniform1D(100)) {
        Float pdf = QuadraticPDF(x, a, b, c);
        // It integrates to 1/12
        EXPECT_FLOAT_EQ(12 * EvaluatePolynomial(x, c, b, a), pdf);
    }

    // Compare vs finely-tabularized
    auto func = [a, b, c](Float x) { return EvaluatePolynomial(x, c, b, a); };
    auto values = Sample1DFunction(func, 16384, 64, 0.f, 1.f, Norm::L1);
    Distribution1D distrib(values);
    for (Float u : Uniform1D(100, 2362469)) {
        Float cx = SampleQuadratic(u, a, b, c);
        Float cp = QuadraticPDF(cx, a, b, c);

        Float dp;
        Float dx = distrib.SampleContinuous(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
            ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
            ", distrib PDF = " << dp;

        auto checkErr = [](Float a, Float b) {
                            Float err;
                            if (std::min(std::abs(a), std::abs(b)) < 1e-3) err = std::abs(a - b);
                            else err = std::abs(2 * (a - b) / (a + b));
                            return err > 1e-2;
                        };
        EXPECT_FALSE(checkErr(u, InvertQuadraticSample(cx, a, b, c)));
    }
}

TEST(Sampling, QuadraticB) {
    // upside down parabola
    Float a = -2, b = 4, c = 0.25;

    // Check the pdf
    for (Float x : Uniform1D(100)) {
        Float pdf = QuadraticPDF(x, a, b, c);
        // The quadratic integrates to 2.25 - 2/3
        EXPECT_FLOAT_EQ(EvaluatePolynomial(x, c, b, a) / (2.25 - 2./3.), pdf);
    }

    // Compare vs finely-tabularized
    auto func = [a, b, c](Float x) { return EvaluatePolynomial(x, c, b, a); };
    auto values = Sample1DFunction(func, 16384, 64, 0.f, 1.f, Norm::L1);
    Distribution1D distrib(values);
    for (Float u : Uniform1D(100, 3692830)) {
        Float cx = SampleQuadratic(u, a, b, c);
        Float cp = QuadraticPDF(cx, a, b, c);

        Float dp;
        Float dx = distrib.SampleContinuous(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
            ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
            ", distrib PDF = " << dp;


        auto checkErr = [](Float a, Float b) {
                            Float err;
                            if (std::min(std::abs(a), std::abs(b)) < 1e-4) err = std::abs(a - b);
                            else err = std::abs(2 * (a - b) / (a + b));
                            return err > 1e-3;
                        };
        EXPECT_FALSE(checkErr(u, InvertQuadraticSample(cx, a, b, c)));
    }
}

TEST(Sampling, EquiQuadratic) {
    // parabola that just hits zero at x = 0.5, maximum of 0.25 at 0 and 1.
    Float a = 1, b = -1, c = 0.25;
    auto func = [a, b, c](Float x) { return EvaluatePolynomial(x, c, b, a); };

    pstd::array<Float, 3> cf = FitEquiQuadratic(func(0), func(0.5), func(1));

    // Check vs. sampling using the quadratic cofficients.
    for (Float u : Uniform1D(10)) {
        Float x[2] = { SampleQuadratic(u, a, b,c),
                       SampleEquiQuadratic(u, func(0), func(0.5), func(1)) };
        EXPECT_FLOAT_EQ(x[0], x[1]) << u;

        Float pdf[2] = { QuadraticPDF(x[0], a, b, c),
                         EquiQuadraticPDF(x[1], func(0), func(0.5), func(1)) };
        EXPECT_FLOAT_EQ(pdf[0], pdf[1]);

        auto checkErr = [](Float a, Float b) {
                            Float err;
                            if (std::min(std::abs(a), std::abs(b)) < 1e-4) err = std::abs(a - b);
                            else err = std::abs(2 * (a - b) / (a + b));
                            return err > 1e-3;
                        };

        EXPECT_FALSE(checkErr(u, InvertEquiQuadraticSample(x[0], func(0), func(0.5), func(1)))) <<
            StringPrintf("%f vs %f at %f", u, InvertEquiQuadraticSample(x[0], func(0), func(0.5), func(1)), x[0]);
    }
}

TEST(Sampling, EquiQuadraticRandoms) {
    RNG rng;
    for (int i = 0; i < 30; ++i) {
        // Compute random coefficients
        Float a, b, c;
        auto func = [&](Float x) { return EvaluatePolynomial(x, c, b, a); };
        do {
            a = -1 + 2 * rng.Uniform<Float>();
            b = -1 + 2 * rng.Uniform<Float>();
            c = rng.Uniform<Float>();
            // Keep going until we one that's positive over [0,1]...
        } while (a + b + c < 0 || func(-b / (2 * a)) < 0);

        // Compare vs finely-tabularized
        auto values = Sample1DFunction(func, 16384, 64, 0.f, 1.f, Norm::L1);
        Distribution1D distrib(values);
        for (int j = 0; j < 10; ++j) {
            Float u = rng.Uniform<Float>();
            Float cx = SampleEquiQuadratic(u, func(0), func(0.5), func(1));
            Float cp = EquiQuadraticPDF(cx, func(0), func(0.5), func(1));

            Float dp;
            Float dx = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
                ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
                ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                            Float err;
                            if (std::min(std::abs(a), std::abs(b)) < 1e-3) err = std::abs(a - b);
                            else err = std::abs(2 * (a - b) / (a + b));
                            return err > 1e-2;
                        };
            EXPECT_FALSE(checkErr(u, InvertEquiQuadraticSample(cx, func(0),
                                                               func(0.5), func(1)))) <<
                StringPrintf("%f vs %f at %f", u, InvertEquiQuadraticSample(cx, func(0), func(0.5), func(1)), cx);
        }
    }
}

TEST(Sampling, EquiQuadratic2D) {
    RNG rng;
    for (int i = 0; i < 10; ++i) {
        pstd::array<pstd::array<Float, 3>, 3> w;
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                w[j][k] = .25 + rng.Uniform<Float>();

        auto values = Sample2DFunction([&](Float u, Float v) {
            return w[0][0] + v*(-3*w[0][0] + 4*w[0][1] - w[0][2]) + v*v*(2*w[0][0] - 4*w[0][1] + 2*w[0][2]) +
                u*(-3*w[0][0] + 4*w[1][0] - w[2][0]) + u*u*(2*w[0][0] - 4*w[1][0] + 2*w[2][0]) +
                u*v*v*(-6*w[0][0] + 12*w[0][1] - 6*w[0][2] + 8*w[1][0] - 16*w[1][1] + 8*w[1][2] -
                       2*w[2][0] + 4*w[2][1] -
                       2*w[2][2]) + u*u*v*(-6*w[0][0] + 8*w[0][1] - 2*w[0][2] + 12*w[1][0] -
                                           16*w[1][1] + 4*w[1][2] - 6*w[2][0] +
                                           8*w[2][1] - 2*w[2][2]) + u*v*(9*w[0][0] - 12*w[0][1] +
                                                                         3*w[0][2] - 12*w[1][0] +
                                                                         16*w[1][1] - 4*w[1][2] + 3*w[2][0] -
                                                             4*w[2][1] + w[2][2]) + u*u*v*v*
                (4*w[0][0] - 8*w[0][1] + 4*w[0][2] - 8*w[1][0] + 16*w[1][1] -
                 8*w[1][2] + 4*w[2][0] - 8*w[2][1] + 4*w[2][2]);
                                       }, 1024, 1024, 16, Bounds2f(Point2f(0, 0), Point2f(1, 1)), Norm::L1);
        Distribution2D dist(values);

        for (int j = 0; j < 100; ++j) {
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            Float pdf[2];
            Point2f p[2] = { SampleBiquadratic(u, w, &pdf[0]),
                             dist.SampleContinuous(u, &pdf[1]) };

            auto checkErr = [](Float a, Float b) {
                                Float err;
                                if (std::min(std::abs(a), std::abs(b)) < .05) err = std::abs(a - b);
                                else err = std::abs(2 * (a - b) / (a + b));
                                return err > 1e-2;
                            };
            EXPECT_FALSE(checkErr(p[0][0], p[1][0])) << p[0][0] << " vs " << p[1][0] << " at  u = " << u;
            EXPECT_FALSE(checkErr(p[0][1], p[1][1])) << p[0][1] << " vs " << p[1][1] << " at  u = " << u;
            EXPECT_FALSE(checkErr(pdf[0], pdf[1])) << pdf[0] << " vs " << pdf[1] << " at  u = " << u;
            EXPECT_FALSE(checkErr(pdf[0], BiquadraticPDF(p[0], w))) << pdf[0] << " vs " <<
                         BiquadraticPDF(p[0], w) << " at  u = " << u;

            Point2f up = InvertBiquadraticSample(p[0], w);
            EXPECT_FALSE(checkErr(u[0], up[0])) << "orig u " << u << ", got up " << up;
            EXPECT_FALSE(checkErr(u[1], up[1])) << "orig u " << u << ", got up " << up;
        }
    }
}

TEST(Sampling, Bezier2D) {
    RNG rng;
    for (int i = 0; i < 10; ++i) {
        pstd::array<pstd::array<Float, 3>, 3> cp;
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                cp[j][k] = i == 0 ? j+k : .15 + rng.Uniform<Float>();

        auto values = Sample2DFunction([&](Float u, Float v) {
            Float up[3] = { Lerp(v, Lerp(v, cp[0][0], cp[0][1]), Lerp(v, cp[0][1], cp[0][2])),
                            Lerp(v, Lerp(v, cp[1][0], cp[1][1]), Lerp(v, cp[1][1], cp[1][2])),
                            Lerp(v, Lerp(v, cp[2][0], cp[2][1]), Lerp(v, cp[2][1], cp[2][2])) };
            return Lerp(u, Lerp(u, up[0], up[1]), Lerp(u, up[1], up[2]));
                                       }, 1024, 1024, 16, Bounds2f(Point2f(0, 0), Point2f(1, 1)), Norm::L2);
        Distribution2D dist(values);

        for (int j = 0; j < 100; ++j) {
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            Float pdf[2];
            Point2f p[2] = { SampleBezier2D(u, cp, &pdf[0]),
                             dist.SampleContinuous(u, &pdf[1]) };

            auto checkErr = [](Float a, Float b) {
                                Float err;
                                if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                                else err = std::abs(2 * (a - b) / (a + b));
                                return err > 1e-2;
                            };
            EXPECT_FALSE(checkErr(p[0][0], p[1][0])) << p[0][0] << " vs " << p[1][0] << " at  u = " << u;
            EXPECT_FALSE(checkErr(p[0][1], p[1][1])) << p[0][1] << " vs " << p[1][1] << " at  u = " << u;
            EXPECT_FALSE(checkErr(pdf[0], pdf[1])) << pdf[0] << " vs " << pdf[1] << " at  u = " << u;
            EXPECT_FALSE(checkErr(pdf[0], Bezier2DPDF(p[0], cp))) << pdf[0] << " vs " <<
                Bezier2DPDF(p[0], cp) << " at  u = " << u;

            Point2f up = InvertBezier2DSample(p[0], cp);
            EXPECT_FALSE(checkErr(u[0], up[0])) << "orig u " << u << ", got up " << up;
            EXPECT_FALSE(checkErr(u[1], up[1])) << "orig u " << u << ", got up " << up;
        }
    }
}

TEST(Sampling, Tent) {
    // Make sure stratification is preserved at the midpoint of the
    // sampling domain.
    Float dist = std::abs(SampleTent(.501, 1) - SampleTent(.499, 1));
    EXPECT_LT(dist, .01);

    Float rad[] = { Float(1), Float(2.5), Float(.125) };
    RNG rng;
    for (Float radius : rad) {
        auto tent = [&](Float x) {
            return std::max<Float>(0, 1 - std::abs(x) / radius);
        };

        auto values = Sample1DFunction(tent, 8192, 64, -radius, radius);
        Distribution1D distrib(values, -radius, radius);
        for (int i = 0; i < 100; ++i) {
            Float u = rng.Uniform<Float>();
            Float tx = SampleTent(u, radius);
            Float tp = TentPDF(tx, radius);

            Float dp;
            Float dx = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(tx - dx), 3e-3) << "Closed form = " << tx <<
                ", distrib = " << dx;
            EXPECT_LT(std::abs(tp - dp), 3e-3) << "Closed form PDF = " << tp <<
                ", distrib PDF = " << dp;


            auto checkErr = [](Float a, Float b) {
                                Float err;
                                if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                                else err = std::abs(2 * (a - b) / (a + b));
                                return err > 1e-2;
                            };
            EXPECT_FALSE(checkErr(u, InvertTentSample(tx, radius))) <<
                "u " << u << " radius " << radius << " x " << tx <<
                " inverse " << InvertTentSample(tx, radius);
        }
    }
}

TEST(Sampling, CatmullRom) {
    std::vector<Float> nodes = { Float(0), Float(.1), Float(.4), Float(.88), Float(1) };
    std::vector<Float> values = { Float(0), Float(5), Float(2), Float(10), Float(5) };
    std::vector<Float> cdf(values.size());

    IntegrateCatmullRom(nodes, values, pstd::span<Float>(cdf));

    auto cr = [&](Float v) { return CatmullRom(nodes, values, v); };
    Distribution1D distrib(Sample1DFunction(cr, 8192, 1024, 0.f, 1.f, Norm::L1));
    for (Float u : Uniform1D(100)) {
        Float cp;
        Float cx = SampleCatmullRom(nodes, values, cdf, u, nullptr, &cp);

        Float dp;
        Float dx = distrib.SampleContinuous(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
            ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
            ", distrib PDF = " << dp;
    }
}

TEST(Sampling, Bilinear) {
    RNG rng;
    Float quads[][4] = { { 0, .5, 1.3, 4.7 }, { 1, 1, 1, 1 }, { 11, .25, 1, 20 } };
    for (const auto v : quads) {
        auto bilerp = [&](Float x, Float y) {
            return ((1-x)*(1-y) * v[0] + x*(1-y) * v[1] +
                    (1-x)*y * v[2] + x*y * v[3]); };

        auto values = Sample2DFunction(bilerp, 1024, 1024, 16,
                                       Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1);
        Distribution2D distrib(values, 1024, 1024);
        for (Point2f u : Uniform2D(100)) {
            Point2f pb = SampleBilinear(u, {v, 4});
            Float bp = BilinearPDF(pb, {v, 4});

            Float dp;
            Point2f pd = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(pb[0] - pd[0]), 3e-3) << "X: Closed form = " << pb[0] <<
                ", distrib = " << pd[0];
            EXPECT_LT(std::abs(pb[1] - pd[1]), 3e-3) << "Y: Closed form = " << pb[1] <<
                ", distrib = " << pd[1];
            EXPECT_LT(std::abs(bp - dp), 3e-3) << "Closed form PDF = " << bp <<
                ", distrib PDF = " << dp;

            // Check InvertBilinear...
            Point2f up = InvertBilinearSample(pb, {v, 4});
            EXPECT_LT(std::abs(up[0] - u[0]), 3e-3) << "Invert failure: u[0] = " << u[0] <<
                ", p = " << pb[0] << ", up = " << up[0];
            EXPECT_LT(std::abs(up[1] - u[1]), 3e-3) << "Invert failure: u[1] = " << u[1] <<
                ", p = " << pb[1] << ", up = " << up[1];
        }
    }
}

TEST(Sampling, BilinearGrid) {
    // This is itself bilinear...
    using A3 = pstd::array<Float, 3>;
    pstd::array<A3, 3> blGrids[] = {
        pstd::array<A3, 3>{
            A3{ 2.f, 3.f, 4.f },
            A3{ 2.f, 4.f, 6.f },
            A3{ 2.f, 5.f, 8.f }
        },
    };

    for (const auto w : blGrids) {
        for (Point2f u : Hammersley2D(10)) {
            Float bp;
            Point2f pb = SampleBilinearGrid(u, w, &bp);
            Float pdf2 = BilinearGridPDF(pb, w);

            Point2f px = SampleBilinear(u, {w[0][0], w[2][0], w[0][2], w[2][2]});
            Float xpdf = BilinearPDF(px, {w[0][0], w[2][0], w[0][2], w[2][2]});

            EXPECT_LT(std::abs(pb[0] - px[0]), 5e-3) << "X: grid = " << pb[0] <<
                ", bilinear = " << px[0] << ", u = " << u;
            EXPECT_LT(std::abs(pb[1] - px[1]), 5e-3) << "Y: grid = " << pb[1] <<
                ", bilinear = " << px[1] << ", u = " << u;
            EXPECT_LT(std::abs(bp - xpdf), 5e-3) << "grid PDF = " << bp <<
                ", bilinear PDF = " << xpdf << ", u = " << u;
            EXPECT_LT(std::abs(bp - pdf2), 5e-3) << "sample PDF = " << bp <<
                ", PDF = " << pdf2 << ", u = " << u;
        }
    }

    pstd::array<A3, 3> grids[] = {
        pstd::array<A3, 3>{
            A3{ 1.f, 0.f, 0.f },
            A3{ 0.f, 0.f, 0.f },
            A3{ 0.f, 0.f, 0.f }
        },
        pstd::array<A3, 3>{
            A3{ 0.f, 0.f, 0.f },
            A3{ 0.f, 5.f, 0.f },
            A3{ 0.f, 0.f, 0.f }
        },
        pstd::array<A3, 3>{
            A3{ 1.f, 1.f, 1.f },
            A3{ 1.f, 1.f, 1.f },
            A3{ 1.f, 1.f, 1.f }
        },
        pstd::array<A3, 3>{
            A3{ 1.f, 0.f, 1.f },
            A3{ 0.f, 0.f, 0.f },
            A3{ 2.f, 0.f, 2.f }
        },
        pstd::array<A3, 3>{
            A3{ 1.f, 0.f, 2.f },
            A3{ 0.f, 0.f, 0.f },
            A3{ 1.f, 0.f, 2.f }
        },
        pstd::array<A3, 3>{
            A3{ 2.f, 0.f, 2.f },
            A3{ 0.f, 2.f, 0.f },
            A3{ 2.f, 0.f, 2.f }
        },
        pstd::array<A3, 3>{
            A3{ 2.f, 2.f, 2.f },
            A3{ 2.f, 0.f, 2.f },
            A3{ 2.f, 2.f, 2.f }
        },
        pstd::array<A3, 3>{
            A3{ 2.f, 0.f, 0.f },
            A3{ 0.f, 0.f, 0.f },
            A3{ 0.f, 0.f, 2.f }
        },
        pstd::array<A3, 3>{
            A3{ 2.f, 17.f, 1.f },
            A3{ 0.f, 14.f, 7.f },
            A3{ 9.f, 11.f, 2.f }
        },
    };

    for (const auto w : grids) {
        auto eval = [&](Float u, Float v) {
            int uBase = (u >= 0.5), vBase = (v >= 0.5);
            u *= 2;
            if (u >= 1) u -= 1;
            v *= 2;
            if (v >= 1) v -= 1;

            return (1-u)*(1-v)*w[uBase][vBase] + (1-u)*v*w[uBase][vBase+1] +
                u*(1-v)*w[uBase+1][vBase] + u*v*w[uBase+1][vBase+1];
        };

        auto values = Sample2DFunction(eval, 1024, 1024, 16,
                                       Bounds2f(Point2f(0,0), Point2f(1,1)), Norm::L1);
        Distribution2D distrib(values, 1024, 1024);

        Image im(PixelFormat::Float, {1024, 1024}, {"Y"});
        for (int y = 0; y < 1024; ++y)
            for (int x = 0; x < 1024; ++x)
                im.SetChannel({x, y}, 0, values(x, y));
        CHECK(im.Write("yolo.exr"));

        for (Point2f u : Hammersley2D(32)) {
            Float bp;
            Point2f pb = SampleBilinearGrid(u, w, &bp);

            Float dp;
            Point2f pd = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(pb[0] - pd[0]), 5e-3) << "X: Closed form = " << pb[0] <<
                ", distrib = " << pd[0] << ", u = " << u;
            EXPECT_LT(std::abs(pb[1] - pd[1]), 5e-3) << "Y: Closed form = " << pb[1] <<
                ", distrib = " << pd[1] << ", u = " << u;
            EXPECT_LT(std::abs(bp - dp), 5e-3) << "Closed form PDF = " << bp <<
                ", distrib PDF = " << dp << ", u = " << u;
        }
    }
}

TEST(Sampling, Logistic) {
    Float params[][3] = { { 1., -Pi, Pi }, { 5, 0, 3 }, { .25, -5, -3 } };
    for (const auto p : params) {
        Float s = p[0], a = p[1], b = p[2];
        auto logistic = [&](Float v) { return TrimmedLogistic(v, s, a, b); };

        auto values = Sample1DFunction(logistic, 8192, 16, a, b, Norm::L1);
        Distribution1D distrib(values, a, b);
        for (Float u : Uniform1D(100)) {
            Float cx = SampleTrimmedLogistic(u, s, a, b);
            Float cp = TrimmedLogisticPDF(cx, s, a, b);

            Float dp;
            Float dx = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
                ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
                ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                                Float err;
                                if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                                else err = std::abs(2 * (a - b) / (a + b));
                                return err > 1e-2;
                            };
            EXPECT_FALSE(checkErr(u, InvertTrimmedLogisticSample(cx, s, a, b))) <<
                "u = " << u << " -> x = " << cx << " -> ... " <<
                InvertTrimmedLogisticSample(cx, s, a, b);
        }
    }
}

TEST(Sampling, Cauchy) {
    Float params[][4] = { { 0, 1, 0, 1 }, { 1.5, .25, -1, 3 },
                          { 1.25, 3, -4, 0 } /* mu outside range */ };
    for (const auto p : params) {
        Float mu = p[0], sigma = p[1], x0 = p[2], x1 = p[3];
        auto cauchy = [&](Float v) { return TrimmedCauchyPDF(v, x0, x1, mu, sigma); };

        auto values = Sample1DFunction(cauchy, 8192, 16, x0, x1, Norm::L1);
        Distribution1D distrib(values, x0, x1);
        for (Float u : Uniform1D(100)) {
            Float cx = SampleTrimmedCauchy(u, x0, x1, mu, sigma);
            Float cp = TrimmedCauchyPDF(cx, x0, x1, mu, sigma);

            EXPECT_GE(cx, x0);
            EXPECT_LE(cx, x1);

            Float dp;
            Float dx = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
                ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
                ", distrib PDF = " << dp;

            auto checkErr = [](Float a, Float b) {
                                Float err;
                                if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                                else err = std::abs(2 * (a - b) / (a + b));
                                return err > 1e-2;
                            };
            EXPECT_FALSE(checkErr(u, InvertTrimmedCauchySample(cx, x0, x1, mu, sigma))) <<
                "u = " << u << " -> x = " << cx << " -> ... " <<
                InvertTrimmedCauchySample(cx, x0, x1, mu, sigma);
        }
    }
}

TEST(Sampling, Normal) {
    Float params[][2] = { { 0., 1. }, { -.5, .8 }, { .25, .005 }, { 3.6, 1.6 } };
    for (const auto p : params) {
        Float mu = p[0], sigma = p[1];
        auto normal = [&](Float x) {
            return 1 / std::sqrt(2 * Pi * sigma * sigma) *
            std::exp(-Sqr(x - mu) / (2 * sigma * sigma));
        };
        auto values = Sample1DFunction(normal, 8192, 16, mu - 7 * sigma,
                                           mu + 7 * sigma, Norm::L1);
        Distribution1D distrib(values, mu - 7 * sigma, mu + 7 * sigma);

        for (Float u : Uniform1D(100)) {
            Float cx = SampleNormal(u, mu, sigma);
            Float cp = NormalPDF(cx, mu, sigma);

            Float dp;
            Float dx = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
                ", distrib = " << dx;
            EXPECT_LT(std::abs(cp - dp) / dp, .025) << "Closed form PDF = " << cp <<
                ", distrib PDF = " << dp;


            auto checkErr = [](Float a, Float b) {
                                Float err;
                                if (std::min(std::abs(a), std::abs(b)) < 1e-2) err = std::abs(a - b);
                                else err = std::abs(2 * (a - b) / (a + b));
                                return err > 1e-2;
                            };
            EXPECT_FALSE(checkErr(u, InvertNormalSample(cx, mu, sigma))) <<
                " u " << u << " -> x = " << cx << " -> " <<
                InvertNormalSample(cx, mu, sigma) << " with mu " << mu <<
                " and sigma " << sigma;
        }
    }
}

TEST(Sampling, LinearDistribution1D) {
    // First test against simple linear
    auto f = [](Float x) { return Lerp(x, 1, 3); };
    {
        LinearDistribution1D ld({f(0), f(1)});
        Float xpdf;
        Float xl = ld.Sample(.25, &xpdf);
        EXPECT_EQ(SampleLinear(.25, 1, 3), xl);
        EXPECT_EQ(LinearPDF(xl, 1, 3), xpdf);
        EXPECT_EQ(LinearPDF(xl, 1, 3), ld.PDF(xl));
    }

    // Same but sample more
    {
        LinearDistribution1D ld({f(0), f(0.25), f(0.5), f(0.75), f(1)});
        Float xpdf;
        Float xl = ld.Sample(.25, &xpdf);
        EXPECT_EQ(SampleLinear(.25, 1, 3), xl);
        EXPECT_LT(std::abs(LinearPDF(xl, 1, 3) - xpdf), 1e-5);
        EXPECT_LT(std::abs(LinearPDF(xl, 1, 3) - ld.PDF(xl)), 1e-5);
    }
}

TEST(Sampling, LinearDistribution2D) {
    // simple bilerp
    const Float v[4] = { 0, 1, 1, 1 };
//CO    const Float v[4] = { 100, .75, .25, 4 };
    auto f = [v](Float x, Float y) { return Bilerp({x, y}, v); };

#if 0
    {
        LinearDistribution2D ld(2, 2, f);
        Point2f u(.25, .85);
        Float xpdf;
        Point2f xl = ld.Sample(u, &xpdf);
        EXPECT_EQ(SampleBilinear(u, v), xl);
        EXPECT_EQ(BilinearPDF(xl, v), xpdf);
        EXPECT_EQ(BilinearPDF(xl, v), ld.PDF(xl));
    }
#endif
    {
//CO        LinearDistribution2D ld(3, 3, f);
        Distribution2D ld(Sample2DFunction(f, 500, 500, 4), 500, 500);
        Point2f u(.1, .1);
        Float xpdf;
//CO        Point2f xl = ld.Sample(u, &xpdf);
        Point2f xl = ld.SampleContinuous(u, &xpdf);
        // both off
        EXPECT_EQ(SampleBilinear(u, v), xl) << SampleBilinear(u, v) << " - " << xl;
        EXPECT_LT(std::abs(BilinearPDF(xl, v) - xpdf), 1e-5) << BilinearPDF(xl, v) << " - " << xpdf;
        EXPECT_LT(std::abs(BilinearPDF(xl, v) - ld.ContinuousPDF(xl)), 1e-5);
    }

    {
        Distribution2D d2(Sample2DFunction(f, 512, 512, 4), 512, 512);
        LinearDistribution2D ld({f(0, 0), f(0, .5), f(0, 1),
                                 f(0.5, 0), f(0.5, 0.5), f(0.5, 1),
                                 f(1, 0), f(1, 0.5), f(1, 1) }, 3, 3);
        Point2f u(.1, .1);
        Float xpdf;
        Point2f xl = ld.Sample(u, &xpdf);

        constexpr int nc = 2;
    int counts[nc][nc] = { { 0 } };
    int n = 4096;
    int count = 0;
    for (float a = 0.5f / n; a < 1; a += 1.f / n) {
        for (float b = 0.5f / n; b < 1; b += 1.f / n, ++count) {
            Float pdf;
            Point2f x = ld.Sample({a, b}, &pdf);
            ++counts[int(Clamp(x[0] * nc, 0, nc - 1))][int(Clamp(x[1] * nc, 0, nc - 1))];

//CO            x = d2.SampleContinuous({a, b}, &pdf);
            x = SampleBilinear({a, b}, v);
            --counts[int(Clamp(x[0] * nc, 0, nc - 1))][int(Clamp(x[1] * nc, 0, nc - 1))];
        }
    }
    for (int y = 0; y < nc; ++y) {
        for (int x = 0; x < nc; ++x) {
//CO            fprintf(stderr, "%f, ", counts[x][y] / Bilerp({(x + .5f) / nc, (y + .5f) / nc}, v));
            fprintf(stderr, "%d, ", counts[x][y]);
        }
        fprintf(stderr, "\n");
    }
    }
#if 0
    {
        LinearDistribution2D ld(300, 2, f);
        Point2f u(.25, .85);
        Float xpdf;
        Point2f xl = ld.Sample(u, &xpdf);
        // off in both
        EXPECT_EQ(SampleBilinear(u, v), xl) << SampleBilinear(u, v) << " - " << xl;
        EXPECT_LT(std::abs(BilinearPDF(xl, v) - xpdf), 1e-5);
        EXPECT_LT(std::abs(BilinearPDF(xl, v) - ld.PDF(xl)), 1e-5);
    }

    {
        LinearDistribution2D ld(2, 19, f);
        Point2f u(.25, .85);
        Float xpdf;
        Point2f xl = ld.Sample(u, &xpdf);
        // good
        EXPECT_EQ(SampleBilinear(u, v), xl) << SampleBilinear(u, v) << " - " << xl;
        EXPECT_LT(std::abs(BilinearPDF(xl, v) - xpdf), 1e-5);
        EXPECT_LT(std::abs(BilinearPDF(xl, v) - ld.PDF(xl)), 1e-5);
    }

    {
        LinearDistribution2D ld(40, 40, f);
        Point2f u(.25, .85);
        Float xpdf;
        Point2f xl = ld.Sample(u, &xpdf);
        // off in u
        EXPECT_EQ(SampleBilinear(u, v), xl) << SampleBilinear(u, v) << " - " << xl;
        EXPECT_EQ(BilinearPDF(xl, v), xpdf);
        EXPECT_EQ(BilinearPDF(xl, v), ld.PDF(xl));
    }
#endif
}

TEST(Sampling, DynamicDistribution1D) {
    // Check basic agreement. Set up so that values sum to 8 -> can do
    // exact probabilities.
    Float v[] = { 1, 0, 2, 1, 0, 0, 3, 1 };
    Distribution1D d(v);
    DynamicDistribution1D dd(v);
    EXPECT_EQ(PBRT_ARRAYSIZE(v), dd.size());

    Float pd, pdd;
    for (Float u = 0; u < 1; u += .05) {
        EXPECT_EQ(d.SampleDiscrete(u, &pd),
                  dd.SampleDiscrete(u, &pdd)) << u;
        EXPECT_LT(std::abs(pd - pdd), 1e-5) << u;
    }

    // Dynamic update; update just individual ones.
    pstd::swap(v[0], v[5]);
    dd[0] = v[0];
    dd[5] = v[5];
    d = Distribution1D(v);
    dd.Update(0);
    dd.Update(5);
    for (Float u = 0; u < 1; u += .05) {
        EXPECT_EQ(d.SampleDiscrete(u, &pd),
                  dd.SampleDiscrete(u, &pdd)) << u;
        EXPECT_LT(std::abs(pd - pdd), 1e-5) << u;
    }

    // Test UpdateAll()
    v[6] = 0;
    dd[6] = 0;
    v[1] = .5;
    dd[1] = .5;
    v[2] = .5;
    dd[2] = .5;
    d = Distribution1D(v);
    dd.UpdateAll();

    for (Float u = 0; u < 1; u += .05) {
        EXPECT_EQ(d.SampleDiscrete(u, &pd),
                  dd.SampleDiscrete(u, &pdd)) <<
            "u = " << u << "dd = " << dd.ToString();
        EXPECT_LT(std::abs(pd - pdd), 1e-5) <<
            "u = " << u << "dd = " << dd.ToString();
    }

    // Test varying sizes; not just power of 2...
    std::vector<Float> vec;
    RNG rng;
    vec.push_back(1);
    for (int i = 0; i < 40; ++i) {
        Distribution1D d(vec);
        DynamicDistribution1D dd(vec);
        EXPECT_EQ(vec.size(), dd.size());

        // This is a little dicey to test, since due to fp roundoff error,
        // one or the other may reasonably choose one side or the other
        // when on the edge. This value of u seems to get us through the tests.
        // TODO: think about how to properly make this more robust.
        Float u = .75;
        EXPECT_EQ(d.SampleDiscrete(u, &pd),
                  dd.SampleDiscrete(u, &pdd)) <<
            "u = " << u << "dd = " << dd.ToString();
        EXPECT_LT(std::abs(pd - pdd), 1e-5) <<
            "u = " << u << "dd = " << dd.ToString();

        vec.push_back(rng.Uniform<uint32_t>(8));
    }
}

TEST(Sampling, VarianceEstimatorZero) {
    VarianceEstimator<Float> ve;
    for (int i = 0; i < 100; ++i)
        ve.Add(10.);
    EXPECT_EQ(ve.Variance(), 0);
}

TEST(Sampling, VarianceEstimator) {
    VarianceEstimator<double> ve;
    int count = 10000;
    double sum = 0;
    for (Float u : Stratified1D(count)) {
        Float v = Lerp(u, -1, 1);
        ve.Add(v);
        sum += v;
    }

    // f(x) = 0, random variables x_i uniform in [-1,1] ->
    // variance is E[x^2] on [-1,1] == 1/3
    Float err = std::abs(ve.Variance() - 1./3.);
    EXPECT_LT(err, 1e-3) << ve.Variance();

    err = std::abs((sum / count - ve.Mean()) / (sum / count));
    EXPECT_LT(err, 1e-5);
}

TEST(Sampling, VarianceEstimatorMerge) {
    int n = 16;
    std::vector<VarianceEstimator<double>> ve(n);

    RNG rng;
    int count = 10000;
    double sum = 0;
    for (Float u : Stratified1D(count)) {
        Float v = Lerp(u, -1, 1);
        int index = rng.Uniform<int>(ve.size());
        ve[index].Add(v);
        sum += v;
    }

    VarianceEstimator<double> veFinal;
    for (const auto &v : ve)
        veFinal.Add(v);

    // f(x) = 0, random variables x_i uniform in [-1,1] ->
    // variance is E[x^2] on [-1,1] == 1/3
    Float err = std::abs(veFinal.Variance() - 1./3.);
    EXPECT_LT(err, 1e-3) << veFinal.Variance();

    err = std::abs((sum / count - veFinal.Mean()) / (sum / count));
    EXPECT_LT(err, 1e-5);
}

// Make sure that the permute function is in fact a valid permutation.
TEST(Sampling, PermutationElement) {
    for (int len = 2; len < 1024; ++len) {
        for (int iter = 0; iter < 10; ++iter) {
            std::vector<bool> seen(len, false);

            for (int i = 0; i < len; ++i) {
                int offset = PermutationElement(i, len, iter);
                ASSERT_TRUE(offset >= 0 && offset < seen.size()) << offset;
                EXPECT_FALSE(seen[offset]);
                seen[offset] = true;
            }
        }
    }
}

#if 0
TEST(CMJ, Stratification) {
    // Now try with a CMJSampler
    for (int nx = 1; nx < 10; ++nx) {
        for (int ny = 1; ny < 10; ++ny) {
            CMJSampler sampler(nx, ny, true /* jitter */);

            for (int iter = 0; iter < 10; ++iter) {
                constexpr int dims = 10;
                std::vector<bool> seen[dims], seenX[dims], seenY[dims];
                for (int d = 0; d < dims; ++d) {
                    seen[d] = std::vector<bool>(nx * ny, false);
                    seenX[d] = std::vector<bool>(nx * ny, false);
                    seenY[d] = std::vector<bool>(nx * ny, false);
                }

                for (int i = 0; i < nx * ny; ++i) {
                    sampler.StartPixelSample({2, 6}, i);

                    for (int d = 0; d < dims; ++d) {
                        Point2f u = sampler.Get2D();
                        (void)sampler.Get1D(); // yolo
                        int x = u.x * nx, y = u.y * ny;
                        ASSERT_TRUE(x >= 0 && x <= nx);
                        ASSERT_TRUE(y >= 0 && y <= ny);
                        x = std::min(x, nx);
                        y = std::min(y, ny);

                        // Check 2D stratification
                        EXPECT_FALSE(seen[d][x + y * nx]);
                        seen[d][x + y * nx] = true;

                        // Check 1D projections on each axis
                        x = u.x * nx * ny;
                        ASSERT_TRUE(x >= 0 && x <= nx * ny);
                        x = std::min(x, nx * ny);
                        EXPECT_FALSE(seenX[d][x]);
                        seenX[d][x] = true;

                        y = u.y * nx * ny;
                        ASSERT_TRUE(y >= 0 && y <= nx * ny);
                        y = std::min(y, nx * ny);
                        EXPECT_FALSE(seenY[d][y]);
                        seenY[d][y] = true;
                    }

                    // Just in case.
                    sampler.StartPixelSample({3, 6}, i);
                }
            }
        }
    }
}
#endif

TEST(Sampling, HierWarpRealSimple) {
    const Float values[4] = { 0, 2,
                              1, 3 };
    Hierarchical2DWarp warp(values, 2, 2);

    std::vector<int> count(4, 0);
    std::vector<Float> integ(4, 0.);
    int n = 1024*1024;
    for (Point2f u : Uniform2D(n)) {
        Float pdf;
        Point2i p = warp.SampleDiscrete(u, &pdf);
        ASSERT_TRUE(p.x >= 0 && p.x < 2 && p.y >= 0 && p.y < 2) << p;
        EXPECT_GT(pdf, 0);

        Float altPdf = warp.DiscretePDF(p);
        Float err = std::abs(pdf - altPdf) / pdf;
        EXPECT_LE(err, 1e-5);

        int offset = p.x + 2 * p.y;
        ++count[offset];
        integ[offset] += values[offset] / pdf;
    }

    Float norm = n / std::accumulate(values, values + 4, Float(0));
    for (int i = 0; i < 4; ++i) {
        if (values[i] == 0)
            EXPECT_EQ(0, count[i]);
        else {
            EXPECT_GT(count[i], .98 * values[i] * norm);
            EXPECT_LT(count[i], 1.02 * values[i] * norm);
            EXPECT_GT(integ[i] / n, .99 * values[i]);
            EXPECT_LT(integ[i] / n, 1.01 * values[i]);
        }
    }
}

TEST(Sampling, HierWarpSimpleDiscrete) {
    const Float values[16] = { 0, 10, 5, 3,
                               4,  0, 2, 1,
                               1, 16, 8, 2,
                               5,  5, 0, 3 };
    Hierarchical2DWarp warp(values, 4, 4);

    std::vector<int> count(16, 0);
    std::vector<Float> integ(16, 0.);
    int n = 1024*1024;
    for (Point2f u : Uniform2D(n)) {
        Float pdf;
        Point2i p = warp.SampleDiscrete(u, &pdf);
        ASSERT_TRUE(p.x >= 0 && p.x < 4 && p.y >= 0 && p.y < 4) << p;
        EXPECT_GT(pdf, 0);

        Float altPdf = warp.DiscretePDF(p);
        Float err = std::abs(pdf - altPdf) / pdf;
        EXPECT_LE(err, 1e-5);

        int offset = p.x + 4 * p.y;
        ++count[offset];
        integ[offset] += values[offset] / pdf;
    }

    Float norm = n / std::accumulate(values, values + 16, Float(0));
    for (int i = 0; i < 16; ++i) {
        if (values[i] == 0)
            EXPECT_EQ(0, count[i]);
        else {
            EXPECT_GT(count[i], .98 * values[i] * norm);
            EXPECT_LT(count[i], 1.02 * values[i] * norm);
            EXPECT_GT(integ[i] / n, .99 * values[i]);
            EXPECT_LT(integ[i] / n, 1.01 * values[i]);
        }
    }
}

TEST(Sampling, HierWarpSimpleContinuous) {
    const Float values[16] = { 0, 10, 5, 3,
                               4,  0, 2, 1,
                               1, 16, 8, 2,
                               5,  5, 0, 3 };
    Hierarchical2DWarp warp(values, 4, 4);

    std::vector<int> count(16, 0);
    std::vector<Float> integ(16, 0.);
    int n = 1024*1024;
    for (Point2f u : Uniform2D(n)) {
        Float pdf;
        Point2f p = warp.SampleContinuous(u, &pdf);
        ASSERT_TRUE(p.x >= 0 && p.x < 1 && p.y >= 0 && p.y < 1) << p;
        EXPECT_GT(pdf, 0);

        Float altPdf = warp.ContinuousPDF(p);
        Float err = std::abs(pdf - altPdf) / pdf;
        EXPECT_LE(err, 1e-5);

        int offset = int(4 * p.x) + 4 * int(4 * p.y);
        ++count[offset];
        integ[offset] += values[offset] / pdf;
    }

    Float norm = n / std::accumulate(values, values + 16, Float(0));
    for (int i = 0; i < 16; ++i) {
        if (values[i] == 0)
            EXPECT_EQ(0, count[i]);
        else {
            EXPECT_GT(count[i], .98 * values[i] * norm);
            EXPECT_LT(count[i], 1.02 * values[i] * norm);
            EXPECT_GT(16 * integ[i] / n, .99 * values[i]);
            EXPECT_LT(16 * integ[i] / n, 1.01 * values[i]);
        }
    }
}

TEST(Sampling, HierWarpRectPow2) {
    const Float values[16] = { 0, 10, 5, 3,
                               4,  0, 2, 1,
                               1, 16, 8, 2,
                               5,  5, 0, 3 };
    int nx = 1, ny = 16;
    for (; ny > 0; nx *= 2, ny /= 2) {
        Hierarchical2DWarp warp(values, nx, ny);

        std::vector<int> count(16, 0);
        std::vector<Float> integ(16, 0.);
        RNG rng;
        int n = 1024*1024;
        for (Point2f u : Uniform2D(n, nx)) {
            Float pdf;
            Point2i p = warp.SampleDiscrete(u, &pdf);
            ASSERT_TRUE(p.x >= 0 && p.x < nx && p.y >= 0 && p.y < ny) << p;
            EXPECT_GT(pdf, 0);

            Float altPdf = warp.DiscretePDF(p);
            Float err = std::abs(pdf - altPdf) / pdf;
            EXPECT_LE(err, 1e-5);

            int offset = p.x + nx * p.y;
            ++count[offset];
            integ[offset] += values[offset] / pdf;
        }

        Float norm = n / std::accumulate(values, values + 16, Float(0));
        for (int i = 0; i < 16; ++i) {
            if (values[i] == 0)
                EXPECT_EQ(0, count[i]);
            else {
                EXPECT_GT(count[i], .98 * values[i] * norm);
                EXPECT_LT(count[i], 1.02 * values[i] * norm);
                EXPECT_GT(integ[i] / n, .98 * values[i]);
                EXPECT_LT(integ[i] / n, 1.02 * values[i]);
            }
        }
    }
}

TEST(Sampling, HierWarpGeneral) {
    const Float values[21] = { 0, 10,  5, 3, 4, 0, 2,
                               1,  1, 16, 8, 2, 5, 5,
                               0,  3,  7, 2, 9, 0, 0 };
    int nx = 3, ny = 7;
    for (int i = 0; i < 2; ++i) {
        pstd::swap(nx, ny);
        Hierarchical2DWarp warp(values, nx, ny);

        std::vector<int> count(21, 0);
        std::vector<Float> integ(21, 0.);
        int n = 1024*1024;
        for (Point2f u : Uniform2D(n)) {
            Float pdf;
            Point2i p = warp.SampleDiscrete(u, &pdf);
            ASSERT_TRUE(p.x >= 0 && p.x < nx && p.y >= 0 && p.y < ny) << p;
            EXPECT_GT(pdf, 0);

            Float altPdf = warp.DiscretePDF(p);
            Float err = std::abs(pdf - altPdf) / pdf;
            EXPECT_LE(err, 1e-5);

            int offset = p.x + nx * p.y;
            ++count[offset];
            integ[offset] += values[offset] / pdf;
        }

        Float norm = n / std::accumulate(values, values + 21, Float(0));
        for (int i = 0; i < 21; ++i) {
            if (values[i] == 0)
                EXPECT_EQ(0, count[i]);
            else {
                EXPECT_GT(count[i], .98 * values[i] * norm);
                EXPECT_LT(count[i], 1.02 * values[i] * norm);
                EXPECT_GT(integ[i] / n, .98 * values[i]);
                EXPECT_LT(integ[i] / n, 1.02 * values[i]);
            }
        }
    }
}

TEST(Sampling, HierWarpDiscretePdfs) {
    const Float values[21] = { 0, 10,  5, 3, 4, 0, 2,
                               1,  1, 16, 8, 2, 5, 5,
                               0,  3,  7, 2, 9, 0, 0 };

    int nx = 3, ny = 7;
    for (int i = 0; i < 2; ++i) {
        pstd::swap(nx, ny);
        Hierarchical2DWarp warp(values, nx, ny);
        Distribution2D d2d(values, nx, ny);

        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                Float hPdf = warp.DiscretePDF({x, y});
                Float dPdf = d2d.DiscretePDF({x, y});
                if (dPdf == 0) EXPECT_EQ(0, hPdf);
                else {
                    Float err = std::abs(hPdf - dPdf) / dPdf;
                    EXPECT_LT(err, 1e-4) << "hier " << hPdf << ", d2d " << dPdf << " ratio " << dPdf / hPdf <<
                        " @ x = " << x << ", y = " << y;
                }
            }
    }
}

TEST(Sampling, HierWarpContinuousPdfs) {
    const Float values[21] = { 0, 10,  5, 3, 4, 0, 2,
                               1,  1, 16, 8, 2, 5, 5,
                               0,  3,  7, 2, 9, 0, 0 };

    int nx = 3, ny = 7;
    for (int i = 0; i < 4; ++i) {
        Bounds2f domain = (i < 2) ? Bounds2f(Point2f(0,0), Point2f(1,1)) :
            Bounds2f(Point2f(-2.5, 1.), Point2f(2, 2));
        pstd::swap(nx, ny);
        Hierarchical2DWarp warp(values, nx, ny, domain);
        Distribution2D d2d(values, nx, ny, domain);

        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x) {
                Point2f p((x + .5) / nx, (y + .5) / ny);
                p = domain.Lerp(p);

                Float hPdf = warp.ContinuousPDF(p);
                Float dPdf = d2d.ContinuousPDF(p);
                if (dPdf == 0) EXPECT_EQ(0, hPdf);
                else {
                    Float err = std::abs(hPdf - dPdf) / dPdf;
                    EXPECT_LT(err, 1e-4) << "hier " << hPdf << ", d2d " << dPdf << " ratio " << dPdf / hPdf;
                }
            }
    }
}

TEST(Sampling, HierWarpInverseSimple) {
    int nx = 4, ny = 4;
    std::vector<Float> values;
    for (int i = 0; i < nx * ny; ++i)
        values.push_back(10);

    Hierarchical2DWarp dist(values, nx, ny);

    ASSERT_TRUE(dist.Inverse(Point2f(0, 0)).has_value());
    EXPECT_EQ(Point2f(0, 0), *dist.Inverse(Point2f(0, 0)));

    for (Point2f u : Uniform2D(100)) {
        Point2f v = dist.SampleContinuous(u);
        pstd::optional<Point2f> inv = dist.Inverse(v);
        ASSERT_TRUE(inv.has_value());
        Point2f err(std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
                    std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
        EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
        EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
    }
}

TEST(Sampling, HierWarpInverseVaryDimensions) {
    for (int ny = 3; ny < 9; ++ny)
        for (int nx = 3; nx < 9; ++nx) {
            std::vector<Float> values;
            for (int i = 0; i < nx * ny; ++i)
                values.push_back(10);

            Hierarchical2DWarp dist(values, nx, ny);

            ASSERT_TRUE(dist.Inverse(Point2f(0, 0)).has_value());
            EXPECT_EQ(Point2f(0, 0), *dist.Inverse(Point2f(0, 0)));

            for (Point2f u : Uniform2D(100)) {
                Point2f v = dist.SampleContinuous(u);
                pstd::optional<Point2f> inv = dist.Inverse(v);
                ASSERT_TRUE(inv.has_value());
                Point2f err(std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
                            std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
                EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
                EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
            }
        }
}

TEST(Sampling, HierWarpInverseVaryDimensionsRandoms) {
    RNG rng;
    for (int ny = 3; ny < 9; ++ny)
        for (int nx = 3; nx < 9; ++nx) {
            std::vector<Float> values;
            for (int i = 0; i < nx * ny; ++i)
                values.push_back(rng.Uniform<Float>());

            Hierarchical2DWarp dist(values, nx, ny);

            ASSERT_TRUE(dist.Inverse(Point2f(0, 0)).has_value());
            EXPECT_EQ(Point2f(0, 0), *dist.Inverse(Point2f(0, 0)));

            for (Point2f u : Uniform2D(100)) {
                Point2f v = dist.SampleContinuous(u);
                pstd::optional<Point2f> inv = dist.Inverse(v);
                ASSERT_TRUE(inv.has_value());
                Point2f err(std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
                            std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
                EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
                EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
            }
        }
}

TEST(Sampling, HierWarpInverseDomain) {
    RNG rng;
    Bounds2f domain(Point2f(-1, 2.5), Point2f(3, 4.5));
    for (int ny = 3; ny < 9; ++ny)
        for (int nx = 3; nx < 9; ++nx) {
            std::vector<Float> values;
            for (int i = 0; i < nx * ny; ++i)
                values.push_back(rng.Uniform<Float>());

            Hierarchical2DWarp dist(values, nx, ny);

            ASSERT_TRUE(dist.Inverse(Point2f(0, 0)).has_value());
            EXPECT_EQ(Point2f(0, 0), *dist.Inverse(Point2f(0, 0)));

            for (Point2f u : Uniform2D(100)) {
                Point2f v = dist.SampleContinuous(u);
                pstd::optional<Point2f> inv = dist.Inverse(v);
                ASSERT_TRUE(inv.has_value());
                Point2f err(std::min(std::abs((*inv)[0] - u[0]), std::abs((*inv)[0] - u[0]) / u[0]),
                            std::min(std::abs((*inv)[1] - u[1]), std::abs((*inv)[1] - u[1]) / u[1]));
                EXPECT_LT(err.x, 1e-4) << "u.x " << u.x << " vs inv " << inv->x << " at " << v.x;
                EXPECT_LT(err.y, 1e-4) << "u.y " << u.y << " vs inv " << inv->y << " at " << v.y;
            }
        }
}

TEST(WeightedReservoir, Basic) {
    RNG rng;
    constexpr int n = 16;
    float weights[n];
    for (int i = 0; i < n; ++i)
        weights[i] = .01 + Sqr(rng.Uniform<Float>());

    std::atomic<int> count[n] = {};
    int64_t nTrials = 1000000;
    ParallelFor(0, nTrials,
                [&](int64_t start, int64_t end) {
                    RNG rng(3 * start);
                    int localCount[n] = {};

                    for (int64_t i = start; i < end; ++i) {
                        WeightedReservoirSampler<int> wrs(i);
                        int perm[n];
                        for (int j = 0; j < n; ++j)
                            perm[j] = j;

                        for (int j = 0; j < n; ++j) {
                            int index = perm[j];
                            wrs.Add(index, weights[index]);
                        }

                        int index = wrs.GetSample();
                        ASSERT_TRUE(index >= 0 && index < n);
                        ++localCount[index];
                    }

                    for (int i = 0; i < n; ++i)
                        count[i] += localCount[i];
                });

    Float sumW = std::accumulate(std::begin(weights), std::end(weights), Float(0));
    for (int i = 0; i < n; ++i) {
        EXPECT_LE(.98 * count[i] / double(nTrials), weights[i] / sumW);
        EXPECT_GE(1.02 * count[i] / double(nTrials), weights[i] / sumW);
    }
}

TEST(WeightedReservoir, MergeReservoirs) {
    RNG rng(6502);
    constexpr int n = 8;
    float weights[n];
    for (int i = 0; i < n; ++i)
        weights[i] = .01 + rng.Uniform<Float>();

    std::atomic<int> count[n] = {};
    int64_t nTrials = 1000000;
    ParallelFor(0, nTrials,
                [&](int64_t start, int64_t end) {
                    int localCount[n] = {};

                    for (int64_t i = start; i < end; ++i) {
                        WeightedReservoirSampler<int> wrs0(i);
                        WeightedReservoirSampler<int> wrs1(i+1);

                        for (int j = 0; j < n; ++j) {
                            if (j & 1)
                                wrs0.Add(j, weights[j]);
                            else
                                wrs1.Add(j, weights[j]);
                        }

                        wrs0.Merge(wrs1);
                        ++localCount[wrs0.GetSample()];
                    }

                    for (int i = 0; i < n; ++i)
                        count[i] += localCount[i];
                });

    Float sumW = std::accumulate(std::begin(weights), std::end(weights), Float(0));
    for (int i = 0; i < n; ++i) {
        EXPECT_LE(.98 * count[i] / double(nTrials), weights[i] / sumW);
        EXPECT_GE(1.02 * count[i] / double(nTrials), weights[i] / sumW);
    }
}

TEST(Generators, Uniform1D) {
    int count = 0;
    for (Float u : Uniform1D(120)) {
        EXPECT_TRUE(u >= 0 && u < 1);
        ++count;
    }
    EXPECT_EQ(120, count);
}

TEST(Generators, Uniform1DSeed) {
    std::vector<Float> samples;
    for (Float u : Uniform1D(1250))
        samples.push_back(u);

    // Different seed
    int i = 0;
    for (Float u : Uniform1D(samples.size(), 1)) {
        EXPECT_NE(u, samples[i]);
        ++i;
    }
}

TEST(Generators, Uniform2D) {
    int count = 0;
    for (Point2f u : Uniform2D(32)) {
        EXPECT_TRUE(u[0] >= 0 && u[0] < 1 && u[1] >= 0 && u[1] < 1);
        ++count;
    }
    EXPECT_EQ(32, count);
}

TEST(Generators, Uniform2DSeed) {
    std::vector<Point2f> samples;
    for (Point2f u : Uniform2D(83))
        samples.push_back(u);

    // Different seed
    int i = 0;
    for (Point2f u : Uniform2D(samples.size(), 1)) {
        EXPECT_NE(u, samples[i]);
        ++i;
    }
}

TEST(Generators, Uniform3D) {
    int count = 0;
    for (Point3f u : Uniform3D(32)) {
        EXPECT_TRUE(u[0] >= 0 && u[0] < 1 && u[1] >= 0 && u[1] < 1 && u[2] >= 0 && u[2] < 1);
        ++count;
    }
    EXPECT_EQ(32, count);
}

TEST(Generators, Stratified1D) {
    int count = 0, n = 128;  // power of 2
    for (Float u : Stratified1D(n)) {
        EXPECT_TRUE(u >= Float(count) / Float(n) && u < Float(count + 1) / Float(n));
        ++count;
    }
    EXPECT_EQ(n, count);
}

TEST(Generators, Stratified2D) {
    int count = 0, nx = 16, ny = 4;  // power of 2
    for (Point2f u : Stratified2D(nx, ny)) {
        int ix = count % nx;
        int iy = count / nx;
        EXPECT_TRUE(u[0] >= Float(ix) / Float(nx) && u[0] < Float(ix + 1) / Float(nx));
        EXPECT_TRUE(u[1] >= Float(iy) / Float(ny) && u[1] < Float(iy + 1) / Float(ny));
        ++count;
    }
    EXPECT_EQ(nx * ny, count);
}

TEST(Generators, Stratified3D) {
    int count = 0, nx = 4, ny = 32, nz = 8;  // power of 2
    for (Point3f u : Stratified3D(nx, ny, nz)) {
        int ix = count % nx;
        int iy = (count / nx) % ny;
        int iz = count / (nx * ny);
        EXPECT_TRUE(u[0] >= Float(ix) / Float(nx) && u[0] < Float(ix + 1) / Float(nx));
        EXPECT_TRUE(u[1] >= Float(iy) / Float(ny) && u[1] < Float(iy + 1) / Float(ny));
        EXPECT_TRUE(u[2] >= Float(iz) / Float(nz) && u[2] < Float(iz + 1) / Float(nz));
        ++count;
    }
    EXPECT_EQ(nx * ny * nz, count);
}

TEST(Generators, Hammersley2D) {
    int count = 0;
    for (Point2f u : Hammersley2D(32)) {
        EXPECT_EQ((Float)count / 32.f, u[0]);
        EXPECT_EQ(RadicalInverse(0, count), u[1]);
        ++count;
    }
    EXPECT_EQ(32, count);
}

TEST(Generators, Hammersley3D) {
    int count = 0;
    for (Point3f u : Hammersley3D(128)) {
        EXPECT_EQ((Float)count / 128.f, u[0]);
        EXPECT_EQ(RadicalInverse(0, count), u[1]);
        EXPECT_EQ(RadicalInverse(1, count), u[2]);
        ++count;
    }
    EXPECT_EQ(128, count);
}

TEST(Halton, PixelIndexer) {
    Halton128PixelIndexer h128;
    HaltonPixelIndexer h(Point2i(128, 128));
    RNG rng;

    for (int i = 0; i < 100; ++i) {
        Point2i p(rng.Uniform<int>(128), rng.Uniform<int>(128));

        h128.SetPixel(p);
        h.SetPixel(p);

        int sample = rng.Uniform<int>(16384);
        h128.SetPixelSample(sample);
        h.SetPixelSample(sample);

        EXPECT_EQ(h128.SampleFirst2D(), h.SampleFirst2D());
        EXPECT_EQ(h128.SampleIndex(), h.SampleIndex());
    }
}
