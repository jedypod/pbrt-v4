
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
    // Duff et al 2017 footnote 1
    auto error = [](Vector3f a, Vector3f b, Vector3f c) {
        return (Sqr(Length(a) - 1) + Sqr(Length(b) - 1) + Sqr(Length(c) - 1) +
                Sqr(Dot(a, b)) + Sqr(Dot(b, c)) + Sqr(Dot(c, a))) / 6;
    };

    // Coordinate axes.
    Vector3f a, b;
    for (Vector3f v : { Vector3f(1, 0, 0), Vector3f(-1, 0, 0),
                Vector3f(0, 1, 0), Vector3f(0, -1, 0),
                Vector3f(0, 0, 1), Vector3f(0, 0, -1) }) {
        CoordinateSystem(v, &a, &b);
        for (int c = 0; c < 3; ++c) {
            if (v[c] != 0) {
                EXPECT_EQ(0, a[c]);
                EXPECT_EQ(0, b[c]);
            }
        }
    }

    // Bad vectors from Duff et al
    for (Vector3f v : { Vector3f(0.00038527316, 0.00038460016, -0.99999988079),
                Vector3f(-0.00019813581, -0.00008946839, -0.99999988079) }) {
        CoordinateSystem(v, &a, &b);
        EXPECT_LT(error(v, a, b), 1e-10);
    }

    // Random vectors
    RNG rng;
    for (int i = 0; i < 1000; ++i) {
        Point2f u = { rng.UniformFloat(), rng.UniformFloat() };
        Vector3f v = UniformSampleSphere(u);
        CoordinateSystem(v, &a, &b);
        EXPECT_LT(error(v, a, b), 1e-10);
    }
}
