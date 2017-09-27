
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "filters/sinc.h"
#include "util/mathutil.h"

using namespace pbrt;

TEST(Sinc, ZeroHandling) {
    Float x = 0;
    Float prev = 1;
    for (int i = 0; i < 10000; ++i) {
        Float cur = LanczosSincFilter::Sinc(x);
        EXPECT_LE(cur, prev);
        x = NextFloatUp(x);
        prev = cur;
    }

    x = -0;
    prev = 1;
    for (int i = 0; i < 10000; ++i) {
        Float cur = LanczosSincFilter::Sinc(x);
        EXPECT_LE(cur, prev);
        x = NextFloatDown(x);
        prev = cur;
    }
}
