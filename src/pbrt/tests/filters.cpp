
#include <gtest/gtest.h>

#include <pbrt/core/pbrt.h>
#include <pbrt/filters/sinc.h>
#include <pbrt/util/mathutil.h>

using namespace pbrt;

TEST(Sinc, ZeroHandling) {
    Float x = 0;
    Float prev = 1;
    for (int i = 0; i < 10000; ++i) {
        Float cur = Sinc(x);
        EXPECT_LE(cur, prev);
        x = NextFloatUp(x);
        prev = cur;
    }

    x = -0;
    prev = 1;
    for (int i = 0; i < 10000; ++i) {
        Float cur = Sinc(x);
        EXPECT_LE(cur, prev);
        x = NextFloatDown(x);
        prev = cur;
    }
}
