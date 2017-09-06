
#include "tests/gtest/gtest.h"
#include "pbrt.h"
#include "util/parallel.h"
#include <atomic>

using namespace pbrt;

TEST(Parallel, Basics) {
    ParallelInit();

    std::atomic<int> counter{0};
    ParallelFor(0, 1000, [&](int64_t) { ++counter; });
    EXPECT_EQ(1000, counter);

    counter = 0;
    ParallelFor(10, 1010, 19, [&](int64_t start, int64_t end) {
            EXPECT_GT(end, start);
            EXPECT_LE(end - start, 19);
            EXPECT_TRUE(start >= 10 && start < 1010);
            EXPECT_TRUE(end > 10 && end <= 1010);
            for (int64_t i = start; i < end; ++i)
                ++counter;
        });
    EXPECT_EQ(1000, counter);

    counter = 0;
    ParallelFor2D(Bounds2i{{0, 0}, {15, 14}}, [&](Bounds2i b) { ++counter; });
    EXPECT_EQ(15*14, counter);

    ParallelCleanup();
}

TEST(Parallel, DoNothing) {
    ParallelInit();

    std::atomic<int> counter{0};
    ParallelFor(0, 0, [&](int64_t) { ++counter; });
    EXPECT_EQ(0, counter);

    counter = 0;
    ParallelFor2D(Bounds2i{{0, 0}, {0, 0}}, 1, [&](Bounds2i b) { ++counter; });
    EXPECT_EQ(0, counter);

    ParallelCleanup();
}
