
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "util/half.h"
#include "util/rng.h"

using namespace pbrt;

TEST(Half, Basics) {
    EXPECT_EQ(Half(0.f).Bits(), kHalfPositiveZero);
    EXPECT_EQ(Half(-0.f).Bits(), kHalfNegativeZero);
    EXPECT_EQ(Half(Infinity).Bits(), kHalfPositiveInfinity);
    EXPECT_EQ(Half(-Infinity).Bits(), kHalfNegativeInfinity);

    EXPECT_TRUE(Half(0.f / 0.f).IsNaN());
    EXPECT_TRUE(Half(-0.f / 0.f).IsNaN());
    EXPECT_FALSE(Half::FromBits(kHalfPositiveInfinity).IsNaN());
}

TEST(Half, ExactConversions) {
    // Test round-trip conversion of integers that are perfectly
    // representable.
    for (Float i = -2048; i <= 2048; ++i) {
        EXPECT_EQ(i, Float(Half(i)));
    }

    // Similarly for some well-behaved floats
    float limit = 1024, delta = 0.5;
    for (int i = 0; i < 10; ++i) {
        for (float f = -limit; f <= limit; f += delta)
            EXPECT_EQ(f, Float(Half(f)));
        limit /= 2;
        delta /= 2;
    }
}

TEST(Half, Randoms) {
    RNG rng;
    // Choose a bunch of random positive floats and make sure that they
    // convert to reasonable values.
    for (int i = 0; i < 1024; ++i) {
        float f = rng.UniformFloat() * 512;
        uint16_t h = Half(f).Bits();
        float fh = Float(Half::FromBits(h));
        if (fh == f) {
            // Very unlikely, but we happened to pick a value exactly
            // representable as a half.
            continue;
        }
        else {
            // The other half value that brackets the float.
            uint16_t hother;
            if (fh > f) {
                // The closest half was a bit bigger; therefore, the half before it
                // s the other one.
                hother = h - 1;
                if (hother > h) {
                    // test for wrapping around zero
                    continue;
                }
            } else {
                hother = h + 1;
                if (hother < h) {
                    // test for wrapping around zero
                    continue;
                }
            }

            // Make sure the two half values bracket the float.
            float fother = Float(Half::FromBits(hother));
            float dh = std::abs(fh - f);
            float dother = std::abs(fother - f);
            if (fh > f)
                EXPECT_LT(fother, f);
            else
                EXPECT_GT(fother, f);

            // Make sure rounding to the other one of them wouldn't have given a
            // closer half.
            EXPECT_LE(dh, dother);
        }
    }
}

TEST(Half, NextUp) {
    Half h = Half::FromBits(kHalfNegativeInfinity);
    int iters = 0;
    while (h.Bits() != kHalfPositiveInfinity) {
        ASSERT_LT(iters, 65536);
        ++iters;

        Half hup = h.NextUp();
        EXPECT_GT((float)hup, (float)h);
        h = hup;
    }
    // NaNs use the maximum exponent and then the sign bit and have a
    // non-zero significand.
    EXPECT_EQ(65536 - (1 << 11), iters);
}

TEST(Half, NextDown) {
    Half h = Half::FromBits(kHalfPositiveInfinity);
    int iters = 0;
    while (h.Bits() != kHalfNegativeInfinity) {
        ASSERT_LT(iters, 65536);
        ++iters;

        Half hdown = h.NextDown();
        EXPECT_LT((float)hdown, (float)h) << hdown.Bits() << " " << h.Bits();
        h = hdown;
    }
    // NaNs use the maximum exponent and then the sign bit and have a
    // non-zero significand.
    EXPECT_EQ(65536 - (1 << 11), iters);
}
