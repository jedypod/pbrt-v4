
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>

#include <cstdint>

using namespace pbrt;

TEST(Pow2, Basics) {
    for (int i = 0; i < 32; ++i) {
        uint32_t ui = 1u << i;
        EXPECT_EQ(true, IsPowerOf2(ui));
        if (ui > 1) {
            EXPECT_EQ(false, IsPowerOf2(ui + 1));
        }
        if (ui > 2) {
            EXPECT_EQ(false, IsPowerOf2(ui - 1));
        }
    }
}

TEST(CountTrailing, Basics) {
    for (int i = 0; i < 32; ++i) {
        uint32_t ui = 1u << i;
        EXPECT_EQ(i, CountTrailingZeros(ui));
    }
    for (int i = 0; i < 64; ++i) {
        uint64_t ui = 1ull << i;
        EXPECT_EQ(i, CountTrailingZeros(ui));
    }
}

TEST(RoundUpPow2, Basics) {
    EXPECT_EQ(RoundUpPow2(7), 8);
    for (int i = 1; i < (1 << 24); ++i)
        if (IsPowerOf2(i))
            EXPECT_EQ(RoundUpPow2(i), i);
        else
            EXPECT_EQ(RoundUpPow2(i), 1 << (Log2Int(i) + 1));

    for (int64_t i = 1; i < (1 << 24); ++i)
        if (IsPowerOf2(i))
            EXPECT_EQ(RoundUpPow2(i), i);
        else
            EXPECT_EQ(RoundUpPow2(i), 1 << (Log2Int(i) + 1));

    for (int i = 0; i < 30; ++i) {
        int v = 1 << i;
        EXPECT_EQ(RoundUpPow2(v), v);
        if (v > 2) EXPECT_EQ(RoundUpPow2(v - 1), v);
        EXPECT_EQ(RoundUpPow2(v + 1), 2 * v);
    }

    for (int i = 0; i < 62; ++i) {
        int64_t v = 1ll << i;
        EXPECT_EQ(RoundUpPow2(v), v);
        if (v > 2) EXPECT_EQ(RoundUpPow2(v - 1), v);
        EXPECT_EQ(RoundUpPow2(v + 1), 2 * v);
    }
}

TEST(PopCount, Basics) {
    EXPECT_EQ(PopCount(uint32_t(0)), 0);
    EXPECT_EQ(PopCount(uint64_t(0)), 0);
    EXPECT_EQ(PopCount(uint32_t(1)), 1);
    EXPECT_EQ(PopCount(uint64_t(1)), 1);

    auto pc = [](uint64_t v) {
                  int count = 0;
                  for (int b = 0; b < 64; ++b)
                      if ((v & (1ull << b)) != 0)
                          ++count;
                  return count;
              };
    RNG rng;
    for (int i = 0; i < 100; ++i) {
        uint32_t v0 = rng.Uniform<uint32_t>();
        uint64_t v1 = rng.Uniform<uint64_t>();
        EXPECT_EQ(pc(v0), PopCount(v0));
        EXPECT_EQ(pc(v1), PopCount(v1));
    }
}

TEST(Morton2, Basics) {
    uint16_t x = 0b01010111, y = 0b11000101;
    uint32_t m = EncodeMorton2(x, y);
    EXPECT_EQ(m, 0b1011000100110111);

#if 0
    for (int x = 0; x <= 65535; ++x)
        for (int y = 0; y <= 65535; ++y) {
            uint32_t m = EncodeMorton2(x, y);
            uint16_t xp, yp;
            DecodeMorton2(m, &xp, &yp);

            EXPECT_EQ(x, xp);
            EXPECT_EQ(y, yp);
        }
#endif

    RNG rng(12351);
    for (int i = 0; i < 100000; ++i) {
        uint16_t x = rng.Uniform<uint32_t>() >> 16;
        uint16_t y = rng.Uniform<uint32_t>() >> 16;
        uint32_t m = EncodeMorton2(x, y);

        uint16_t xp, yp;
        DecodeMorton2(m, &xp, &yp);
        EXPECT_EQ(x, xp);
        EXPECT_EQ(y, yp);
    }
}
