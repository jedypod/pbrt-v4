
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "util/rng.h"

#include <vector>

using namespace pbrt;

TEST(RNG, Reseed) {
    RNG rng(1234);
    std::vector<uint32_t> values;
    for (int i = 0; i < 100; ++i)
        values.push_back(rng.UniformUInt32());

    rng.SetSequence(1234);
    for (int i = 0; i < values.size(); ++i)
        EXPECT_EQ(values[i], rng.UniformUInt32());
}

TEST(RNG, Advance) {
    RNG rng;
    rng.SetSequence(1234);
    std::vector<Float> v;
    for (int i = 0; i < 1000; ++i)
        v.push_back(rng.UniformFloat());

    rng.SetSequence(1234);
    rng.Advance(16);
    EXPECT_EQ(rng.UniformFloat(), v[16]);

    for (int i = v.size() - 1; i >= 0; --i) {
        rng.SetSequence(1234);
        rng.Advance(i);
        EXPECT_EQ(rng.UniformFloat(), v[i]);
    }

    // Switch to another sequence
    rng.SetSequence(32);
    rng.UniformFloat();

    // Go back and check one last time
    for (int i : { 5, 998, 552, 37, 16 }) {
        rng.SetSequence(1234);
        rng.Advance(i);
        EXPECT_EQ(rng.UniformFloat(), v[i]);
    }
}

TEST(RNG, OperatorMinus) {
    RNG ra(1337), rb(1337);
    RNG rng;
    for (int i = 0; i < 10; ++i) {
        int step = 1 + rng.UniformUInt32(1000);
        for (int j = 0; j < step; ++j)
            (void)ra.UniformUInt32();
        EXPECT_EQ(step, ra - rb);
        EXPECT_EQ(-step, rb - ra);

        // Reysnchronize them
        if (rng.UniformUInt32() & 1)
            rb.Advance(step);
        else
            ra.Advance(-step);
        EXPECT_EQ(0, ra - rb);
        EXPECT_EQ(0, rb - ra);
    }
}
