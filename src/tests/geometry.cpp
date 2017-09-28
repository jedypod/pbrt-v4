
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "util/geometry.h"
#include "util/rng.h"

using namespace pbrt;

TEST(Vector, AngleBetween) {
    EXPECT_EQ(0, AngleBetween(Vector3f(1, 0, 0), Vector3f(1, 0, 0)));

    EXPECT_EQ(Pi, AngleBetween(Vector3f(0, 0, 1), Vector3f(0, 0, -1)));

    EXPECT_EQ(Pi / 2, AngleBetween(Vector3f(1, 0, 0), Vector3f(0, 1, 0)));

    Vector3f x = Normalize(Vector3f(1, -3, 10));
    EXPECT_EQ(0, AngleBetween(x, x));
    EXPECT_EQ(Pi, AngleBetween(x, -x));

    RNG rng;
    for (int i = 0; i < 10000; ++i) {
        Vector3f a = Normalize(Vector3f(-1 + 2 * rng.UniformFloat(),
                                        -1 + 2 * rng.UniformFloat(),
                                        -1 + 2 * rng.UniformFloat()));
        Vector3f b = Normalize(Vector3f(-1 + 2 * rng.UniformFloat(),
                                        -1 + 2 * rng.UniformFloat(),
                                        -1 + 2 * rng.UniformFloat()));
        Float v[2] = { SafeACos(Dot(a, b)), AngleBetween(a, b) };
        Float err = std::abs(v[0] - v[1]);
        EXPECT_LT(err, 5e-6) << v[1] << ", a: " << a << ", b: " << b;
    }
}
