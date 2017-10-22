
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "util/interpolation.h"
#include "util/rng.h"
#include "sampling.h"
#include "lowdiscrepancy.h"
#include "samplers/halton.h"
#include "samplers/maxmin.h"
#include "samplers/random.h"
#include "samplers/sobol.h"
#include "samplers/stratified.h"
#include "samplers/zerotwosequence.h"

#include <stdint.h>
#include <algorithm>
#include <iterator>

using namespace pbrt;

TEST(LowDiscrepancy, RadicalInverse) {
    for (int a = 0; a < 1024; ++a) {
        EXPECT_EQ(ReverseBits32(a) * 2.3283064365386963e-10f,
                  RadicalInverse(0, a));
    }
}

TEST(LowDiscrepancy, ScrambledRadicalInverse) {
    for (int dim = 0; dim < 128; ++dim) {
        RNG rng(dim);
        // Random permutation table
        const int base = Primes[dim];

        std::vector<uint16_t> perm;
        for (int i = 0; i < base; ++i) perm.push_back(base - 1 - i);
        Shuffle(absl::MakeSpan(perm), 1, rng);

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
        C[i] = rng.UniformUInt32();
        Crev[i] = ReverseBits32(C[i]);
    }
    for (int a = 0; a < 1024; ++a) {
        EXPECT_EQ(ReverseBits32(MultiplyGenerator(C, a)),
                  MultiplyGenerator(Crev, a));
    }
}

TEST(LowDiscrepancy, GrayCodeSample) {
    uint32_t C[32];
    // Identity matrix, column-wise
    for (int i = 0; i < 32; ++i) C[i] = 1 << i;

    std::vector<Float> v(64, (Float)0);
    GrayCodeSample(C, 0, absl::MakeSpan(v));

    for (int a = 0; a < (int)v.size(); ++a) {
        Float u = MultiplyGenerator(C, a) * 2.3283064365386963e-10f;
        EXPECT_NE(v.end(), std::find(v.begin(), v.end(), u));
    }
}

TEST(LowDiscrepancy, Sobol) {
    // Check that float and double variants match (as float values).
    for (int i = 0; i < 256; ++i) {
        for (int dim = 0; dim < 100; ++dim) {
            EXPECT_EQ(SobolSampleFloat(i, dim, 0),
                      (float)SobolSampleDouble(i, dim, 0));
        }
    }

    // Make sure first dimension is the regular base 2 radical inverse
    for (int i = 0; i < 8192; ++i) {
        EXPECT_EQ(SobolSampleFloat(i, 0, 0),
                  ReverseBits32(i) * 2.3283064365386963e-10f);
    }
}

// Make sure GeneratePixelSamples() isn't called more than it should be.
TEST(PixelSampler, GeneratePixelSamples) {
    class TestSampler : public PixelSampler {
    public:
        TestSampler() : PixelSampler(64, 5) {}
        void GeneratePixelSamples(RNG &rng) { ++calls; }
        std::unique_ptr<Sampler> Clone() { return std::make_unique<TestSampler>(*this); }
        int calls = 0;
    };

    TestSampler ts;
    ts.StartSequence({0, 0}, 0);
    ts.StartSequence({0, 0}, 1);
    ts.StartSequence({0, 0}, 10);
    ts.StartSequence({1, 0}, 4);
    ts.StartSequence({0, 0}, 11);
    EXPECT_EQ(ts.calls, 3);
}

// Make sure all samplers give the same sample values if we go back to the
// same pixel / sample index.
TEST(Sampler, ConsistentValues) {
    constexpr int rootSpp = 4;
    constexpr int spp = rootSpp * rootSpp;
    Bounds2i sampleBounds {{-2, -1}, {100, 101}};

    std::vector<std::unique_ptr<Sampler>> samplers;
    samplers.push_back(std::make_unique<HaltonSampler>(spp, sampleBounds));
    samplers.push_back(std::make_unique<RandomSampler>(spp));
    samplers.push_back(std::make_unique<StratifiedSampler>(rootSpp, rootSpp, true, 4));
    samplers.push_back(std::make_unique<SobolSampler>(spp, sampleBounds));
    samplers.push_back(std::make_unique<ZeroTwoSequenceSampler>(spp));
    samplers.push_back(std::make_unique<MaxMinDistSampler>(spp, 4));

    for (const auto &sampler : samplers) {
        int na1d = sampler->RoundCount(8);
        sampler->Request1DArray(na1d);
        int na2d = sampler->RoundCount(18);
        sampler->Request2DArray(na2d);

        std::vector<Float> s1d[spp], a1d[spp];
        std::vector<Point2f> s2d[spp], a2d[spp];

        for (int s = 0; s < spp; ++s) {
            sampler->StartSequence({1, 5}, s);
            for (int i = 0; i < 10; ++i) {
                s2d[s].push_back(sampler->Get2D());
                s1d[s].push_back(sampler->Get1D());
            }
            auto array1d = sampler->Get1DArray(na1d);
            std::copy(array1d.begin(), array1d.end(), std::back_inserter(a1d[s]));
            auto array2d = sampler->Get2DArray(na2d);
            std::copy(array2d.begin(), array2d.end(), std::back_inserter(a2d[s]));
        }

        // Go somewhere else and generate some samples, just to make sure
        // things are shaken up.
        sampler->StartSequence({0, 6}, 10);
        sampler->Get2D();
        sampler->Get2D();
        sampler->Get1D();

        // Now go back and generate samples again, but enumerate them in a
        // different order to make sure the sampler is doing the right
        // thing.
        for (int s = spp - 1; s >= 0; --s) {
            sampler->StartSequence({1, 5}, s);
            for (int i = 0; i < s2d[s].size(); ++i) {
                EXPECT_EQ(s2d[s][i], sampler->Get2D());
                EXPECT_EQ(s1d[s][i], sampler->Get1D());
            }
            auto array1d = sampler->Get1DArray(na1d);
            for (int i = 0; i < na1d; ++i)
                EXPECT_EQ(array1d[i], a1d[s][i]);

            auto array2d = sampler->Get2DArray(na2d);
            for (int i = 0; i < na2d; ++i)
                EXPECT_EQ(array2d[i], a2d[s][i]);
        }
    }
}

// Make sure samplers that are supposed to generate a single sample in
// each of the elementary intervals actually do so.
// TODO: check Halton (where the elementary intervals are (2^i, 3^j)).
TEST(LowDiscrepancy, ElementaryIntervals) {
    auto checkSampler = [](const char *name, std::unique_ptr<Sampler> sampler,
                           int logSamples) {
        // Get all of the samples for a pixel.
        int spp = sampler->samplesPerPixel;
        std::vector<Point2f> samples;
        for (int i = 0; i < spp; ++i) {
            sampler->StartSequence(Point2i(0, 0), i);
            samples.push_back(sampler->Get2D());
        }

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
                EXPECT_EQ(0, count[index]) << "Sampler " << name;
                ++count[index];
            }
        }
    };

    for (int logSamples = 2; logSamples <= 10; ++logSamples) {
        checkSampler(
            "MaxMinDistSampler",
            std::unique_ptr<Sampler>(new MaxMinDistSampler(1 << logSamples, 2)),
            logSamples);
        checkSampler("ZeroTwoSequenceSampler",
                     std::unique_ptr<Sampler>(
                         new ZeroTwoSequenceSampler(1 << logSamples, 2)),
                     logSamples);
        checkSampler("Sobol", std::unique_ptr<Sampler>(new SobolSampler(
                                  1 << logSamples,
                                  Bounds2i(Point2i(0, 0), Point2i(10, 10)))),
                     logSamples);
    }
}

TEST(MaxMinDist, MinDist) {
    // We use a silly O(n^2) distance check below, so don't go all the way up
    // to 2^16 samples.
    for (int logSamples = 2; logSamples <= 10; ++logSamples) {
        // Store a pixel's worth of samples in the vector s.
        MaxMinDistSampler mm(1 << logSamples, 2);
        int spp = mm.samplesPerPixel;
        std::vector<Point2f> s;
        for (int i = 0; i < spp; ++i) {
            mm.StartSequence(Point2i(0, 0), i);
            s.push_back(mm.Get2D());
        }

        // Distance with toroidal topology
        auto dist = [](const Point2f &p0, const Point2f &p1) {
            Vector2f d = Abs(p1 - p0);
            if (d.x > 0.5) d.x = 1. - d.x;
            if (d.y > 0.5) d.y = 1. - d.y;
            return Length(d);
        };

        Float minDist = Infinity;
        for (size_t i = 0; i < s.size(); ++i) {
            for (size_t j = 0; j < s.size(); ++j) {
                if (i == j) continue;
                minDist = std::min(minDist, dist(s[i], s[j]));
            }
        }

        // Expected minimum distances from Gruenschloss et al.'s paper.
        Float expectedMinDist[17] = {
            0., /* not checked */
            0., /* not checked */
            0.35355, 0.35355, 0.22534, 0.16829, 0.11267,
            0.07812, 0.05644, 0.03906, 0.02816, 0.01953,
            0.01408, 0.00975, 0.00704, 0.00486, 0.00352,
        };
        // Increase the tolerance by a small slop factor.
        EXPECT_GT(minDist, 0.99 * expectedMinDist[logSamples]);
    }
}

TEST(Distribution1D, Discrete) {
    // Carefully chosen distribution so that transitions line up with
    // (inverse) powers of 2.
    Distribution1D dist({0.f, 1.f, 0.f, 3.f});
    EXPECT_EQ(4, dist.Count());

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
    EXPECT_EQ(5, dist.Count());

    Float pdf;
    int offset;
    EXPECT_EQ(0., dist.SampleContinuous(0., &pdf, &offset));
    EXPECT_FLOAT_EQ(dist.Count() * 1. / 16., pdf);
    EXPECT_EQ(0, offset);

    // Right at the bounary between the 4 and the 8 segments.
    EXPECT_FLOAT_EQ(.8, dist.SampleContinuous(0.5, &pdf, &offset));

    // Middle of the 8 segment
    EXPECT_FLOAT_EQ(.9, dist.SampleContinuous(0.75, &pdf, &offset));
    EXPECT_FLOAT_EQ(dist.Count() * 8. / 16., pdf);
    EXPECT_EQ(4, offset);

    EXPECT_FLOAT_EQ(0., dist.SampleContinuous(0., &pdf));
    EXPECT_FLOAT_EQ(1., dist.SampleContinuous(1., &pdf));
}

TEST(Distribution1D, FromFuncL1) {
    auto f = [](Float v) { return v * v; };
    Distribution1D dSampled = Distribution1D::SampleFunction(f, 8, 1024*1024, Norm::L1);

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
    Distribution1D dSampled = Distribution1D::SampleFunction(f, 8, 1024*1024, Norm::L2);

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
    Distribution1D dSampled = Distribution1D::SampleFunction(f, 8, 1024*1024, Norm::LInfinity);

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

TEST(Distribution2D, FromFuncL1) {
    auto f = [](Point2f p) { return p[0] * p[0] * p[1]; }; // x^2 y
    Distribution2D dSampled = Distribution2D::SampleFunction(f, 4, 2, 1024*1024, Norm::L1);

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
    auto f = [](Point2f p) { return p[0] * p[0] * p[1]; }; // x^2 y
    Distribution2D dSampled = Distribution2D::SampleFunction(f, 4, 2, 1024*1024, Norm::L2);

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
    auto f = [](Point2f p) { return p[0] * p[0] * p[1]; }; // x^2 y
    Distribution2D dSampled = Distribution2D::SampleFunction(f, 4, 2, 1024*1024, Norm::LInfinity);

    std::vector<Float> exact = { Sqr(0.25) * Float(0.5),
                                 Sqr(0.5) * Float(0.5),
                                 Sqr(0.75) * Float(0.5),
                                 Sqr(1) * Float(0.5),
                                 Sqr(0.25) * Float(1),
                                 Sqr(0.5) * Float(1),
                                 Sqr(0.75) * Float(1),
                                 Sqr(1) * Float(1) };
    Distribution2D dExact(exact, 4, 2);
    Distribution2D::TestCompareDistributions(dSampled, dExact);
}

TEST(Sampling, SampleDiscrete) {
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

TEST(Sampling, SphericalTriangle) {
    int count = 1024*1024;
    std::array<Point3f, 3> v = { Point3f(4, 1, 1), Point3f(-10, 3, 3), Point3f(-2, -8, 10) };
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
        std::array<Float, 3> bs = SphericalSampleTriangle(v, p, u, &pdf);
        Point3f pTri = bs[0] * v[0] + bs[1] * v[1] + bs[2] * v[2];
        sphSum += f(pTri) / pdf;

        std::array<Float, 3> ba = UniformSampleTriangle(u);
        pdf = 1 / A;
        pTri = ba[0] * v[0] + ba[1] * v[1] + (1 - ba[0] - ba[1]) * v[2];
        areaSum += f(pTri) * AbsDot(N, Normalize(pTri - p)) /
            (pdf * DistanceSquared(p, pTri));
    }
    Float sphInt = sphSum / count;
    Float areaInt = areaSum / count;

    EXPECT_LT(std::abs(areaInt - sphInt), 1e-3);
}

TEST(Sampling, Smoothstep) {
    Float start = SampleSmoothstep(0, 10, 20);
    // Fairly high slop since lots of values close to the start are close
    // to zero.
    EXPECT_LT(std::abs(start - 10), .1) << start;

    Float end = SampleSmoothstep(1, -10, -5);
    EXPECT_LT(std::abs(end - -5), 1e-5) << end;

    Float mid = SampleSmoothstep(0.5, 0, 1);
    // Solved this numericalla in Mathematica.
    EXPECT_LT(std::abs(mid - 0.733615), 1e-5) << mid;

    RNG rng;
    for (int i = 0; i < 100; ++i) {
        Float x = SampleSmoothstep(rng.UniformFloat(), -3, 5);
        Float ratio = Smoothstep(x, -3, 5) / SmoothstepPdf(x, -3, 5);
        // Smoothstep over [-3,5] integrates to 4.
        EXPECT_LT(std::abs(ratio - 4), 1e-5) << ratio;
    }

    auto ss = [](Float v) { return Smoothstep(v, 0, 1); };
    Distribution1D distrib =
        Distribution1D::SampleFunction(ss, 1024, 64*1024, Norm::L1);
    for (int i = 0; i < 100; ++i) {
        Float u = rng.UniformFloat();
        Float cx = SampleSmoothstep(u, 0, 1);
        Float cp = SmoothstepPdf(cx, 0, 1);

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

    RNG rng;
    auto lin = [](Float v) { return 1 + 3 * v; };
    Distribution1D distrib =
        Distribution1D::SampleFunction(lin, 1024, 64*1024, Norm::L1);
    for (int i = 0; i < 100; ++i) {
        Float u = rng.UniformFloat();
        Float cx = SampleLinear(u, 1, 4);
        Float cp = LinearPdf(cx, 1, 4);

        Float dp;
        Float dx = distrib.SampleContinuous(u, &dp);
        EXPECT_LT(std::abs(cx - dx), 3e-3) << "Closed form = " << cx <<
            ", distrib = " << dx;
        EXPECT_LT(std::abs(cp - dp), 3e-3) << "Closed form PDF = " << cp <<
            ", distrib PDF = " << dp;
    }
}

TEST(Sampling, CatmullRom) {
    std::vector<Float> nodes = { Float(0), Float(.1), Float(.4), Float(.88), Float(1) };
    std::vector<Float> values = { Float(0), Float(5), Float(2), Float(10), Float(5) };
    std::vector<Float> cdf(values.size());

    IntegrateCatmullRom(nodes, values, absl::Span<Float>(cdf));

    RNG rng;
    auto cr = [&](Float v) { return CatmullRom(nodes, values, v); };
    Distribution1D distrib =
        Distribution1D::SampleFunction(cr, 8192, 1024, Norm::L1);
    for (int i = 0; i < 100; ++i) {
        Float u = rng.UniformFloat();
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
        auto bilerp = [&](Point2f p) {
            return ((1-p[0])*(1-p[1]) * v[0] + p[0]*(1-p[1]) * v[1] +
                    (1-p[0])*p[1] * v[2] + p[0]*p[1] * v[3]); };

        Distribution2D distrib = Distribution2D::SampleFunction(bilerp, 1024, 1024, 16,
                                                                Norm::L1);
        for (int i = 0; i < 100; ++i) {
            Point2f u{rng.UniformFloat(), rng.UniformFloat()};
            Point2f pb = SampleBilinear(u, {v, 4});
            Float bp = BilinearPdf(pb, {v, 4});

            Float dp;
            Point2f pd = distrib.SampleContinuous(u, &dp);
            EXPECT_LT(std::abs(pb[0] - pd[0]), 3e-3) << "X: Closed form = " << pb[0] <<
                ", distrib = " << pd[0];
            EXPECT_LT(std::abs(pb[1] - pd[1]), 3e-3) << "Y: Closed form = " << pb[1] <<
                ", distrib = " << pd[1];
            EXPECT_LT(std::abs(bp - dp), 3e-3) << "Closed form PDF = " << bp <<
                ", distrib PDF = " << dp;
        }
    }
}

