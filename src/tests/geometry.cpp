
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "core/sampling.h"
#include "util/geometry.h"
#include "util/rng.h"

using namespace pbrt;

TEST(Vector, AngleBetween) {
    EXPECT_EQ(0, AngleBetween(Vector3f(1, 0, 0), Vector3f(1, 0, 0)));

    EXPECT_EQ(Pi, AngleBetween(Vector3f(0, 0, 1), Vector3f(0, 0, -1)));

    EXPECT_FLOAT_EQ(Pi / 2, AngleBetween(Vector3f(1, 0, 0), Vector3f(0, 1, 0)));

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

TEST(Vector, CoordinateSystem) {
    Vector3f a, b;
    CoordinateSystem(Vector3f(1, 0, 0), &a, &b);
    EXPECT_EQ(0, a.x);
    EXPECT_EQ(0, b.x);
    EXPECT_LT(std::abs(Dot(a, b)), 1e-8);

    RNG rng;
    for (int i = 0; i < 100; ++i) {
        Point2f u = { rng.UniformFloat(), rng.UniformFloat() };
        Vector3f v = UniformSampleSphere(u);
        CoordinateSystem(v, &a, &b);
        EXPECT_LT(std::abs(Dot(v, a)), 1e-7);
        EXPECT_LT(std::abs(Dot(v, b)), 1e-7);
        EXPECT_LT(std::abs(Dot(b, a)), 1e-7);
    }
}
