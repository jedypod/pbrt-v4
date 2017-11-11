
#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>

#include <pbrt/core/pbrt.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>

using namespace pbrt;

TEST(Math, Pow) {
    EXPECT_EQ(Pow<0>(2.f), 1 << 0);
    EXPECT_EQ(Pow<1>(2.f), 1 << 1);
    EXPECT_EQ(Pow<2>(2.f), 1 << 2);
    // Test remainder of pow template powers to 29
    EXPECT_EQ(Pow<3>(2.f), 1 << 3);
    EXPECT_EQ(Pow<4>(2.f), 1 << 4);
    EXPECT_EQ(Pow<5>(2.f), 1 << 5);
    EXPECT_EQ(Pow<6>(2.f), 1 << 6);
    EXPECT_EQ(Pow<7>(2.f), 1 << 7);
    EXPECT_EQ(Pow<8>(2.f), 1 << 8);
    EXPECT_EQ(Pow<9>(2.f), 1 << 9);
    EXPECT_EQ(Pow<10>(2.f), 1 << 10);
    EXPECT_EQ(Pow<11>(2.f), 1 << 11);
    EXPECT_EQ(Pow<12>(2.f), 1 << 12);
    EXPECT_EQ(Pow<13>(2.f), 1 << 13);
    EXPECT_EQ(Pow<14>(2.f), 1 << 14);
    EXPECT_EQ(Pow<15>(2.f), 1 << 15);
    EXPECT_EQ(Pow<16>(2.f), 1 << 16);
    EXPECT_EQ(Pow<17>(2.f), 1 << 17);
    EXPECT_EQ(Pow<18>(2.f), 1 << 18);
    EXPECT_EQ(Pow<19>(2.f), 1 << 19);
    EXPECT_EQ(Pow<20>(2.f), 1 << 20);
    EXPECT_EQ(Pow<21>(2.f), 1 << 21);
    EXPECT_EQ(Pow<22>(2.f), 1 << 22);
    EXPECT_EQ(Pow<23>(2.f), 1 << 23);
    EXPECT_EQ(Pow<24>(2.f), 1 << 24);
    EXPECT_EQ(Pow<25>(2.f), 1 << 25);
    EXPECT_EQ(Pow<26>(2.f), 1 << 26);
    EXPECT_EQ(Pow<27>(2.f), 1 << 27);
    EXPECT_EQ(Pow<28>(2.f), 1 << 28);
    EXPECT_EQ(Pow<29>(2.f), 1 << 29);
}

TEST(Math, NewtonBisection) {
    EXPECT_FLOAT_EQ(
        1, NewtonBisection(0, 10, [](Float x) -> std::pair<Float, Float> {
            return {Float(-1 + x), Float(1)};
        }));
    EXPECT_FLOAT_EQ(
        Pi / 2, NewtonBisection(0, 2, [](Float x) -> std::pair<Float, Float> {
            return {std::cos(x), -std::sin(x)};
        }));

    // The derivative is a lie--pointing in the wrong direction, even--but
    // it should still work.
    Float bad = NewtonBisection(0, 2, [](Float x) -> std::pair<Float, Float> {
        return {std::cos(x), 10 * std::sin(x)};
    });
    EXPECT_LT(std::abs(Pi / 2 - bad), 1e-5);

    // Multiple zeros in the domain; make sure we find one.
    Float zero =
        NewtonBisection(.1, 10.1, [](Float x) -> std::pair<Float, Float> {
            return {std::sin(x), std::cos(x)};
        });
    EXPECT_LT(std::abs(std::sin(zero)), 1e-6);

    // Ill-behaved function with derivatives that go to infinity (and also
    // multiple zeros).
    auto f = [](Float x) -> std::pair<Float, Float> {
        return {
            std::pow(Sqr(std::sin(x)), .05) - 0.3,
            0.1 * std::cos(x) * std::sin(x) / std::pow(Sqr(std::sin(x)), 0.95)};
    };
    zero = NewtonBisection(.01, 9.42477798, f);
    // Extra slop for a messy function.
    EXPECT_LT(std::abs(f(zero).first), 1e-2);

    // Ill-behaved function with derivatives that go to infinity (and also
    // multiple zeros).
    auto fd = [](double x) -> std::pair<double, double> {
        return {
            std::pow(Sqr(std::sin(x)), .05) - 0.3,
            0.1 * std::cos(x) * std::sin(x) / std::pow(Sqr(std::sin(x)), 0.95)};
    };
    double dzero = NewtonBisection(.01, 9.42477798, fd, 0, 1e-10);
    // Expect to come closer via double precision and tighter tolerances
    EXPECT_LT(std::abs(fd(dzero).first), 1e-10);
}

TEST(Math, EvaluatePolynomial) {
    EXPECT_EQ(4, EvaluatePolynomial(100, 4));
    EXPECT_EQ(10, EvaluatePolynomial(2, 4, 3));

    EXPECT_EQ(1.5 + 2.75 * .5 - 4.25 * Pow<2>(.5) + 15.125 * Pow<3>(.5),
              EvaluatePolynomial(.5, 1.5, 2.75, -4.25, 15.125));
}

TEST(Math, KahanSum) {
    // In order of decreasing accuracy...
    long double ldSum = 0;
    KahanSum<double> kahanSumD;
    double doubleSum = 0;
    KahanSum<float> kahanSumF;
    float floatSum = 0;

    RNG rng;
    for (int i = 0; i < 16*1024*1024; ++i) {
        // Hard to sum accurately since the values span many magnitudes.
        float v = std::exp(Lerp(rng.UniformFloat(), -5, 20));
        ldSum += v;
        kahanSumD += v;
        doubleSum += v;
        kahanSumF += v;
        floatSum += v;
    }

    int64_t ldBits = FloatToBits(double(ldSum));
    int64_t kahanDBits = FloatToBits(double(kahanSumD));
    int64_t doubleBits = FloatToBits(doubleSum);
    int64_t kahanFBits = FloatToBits(double(kahanSumF));
    int64_t floatBits = FloatToBits(double(floatSum));

    int64_t kahanDErrorUlps = std::abs(kahanDBits - ldBits);
    int64_t doubleErrorUlps = std::abs(doubleBits - ldBits);
    int64_t kahanFErrorUlps = std::abs(kahanFBits - ldBits);
    int64_t floatErrorUlps = std::abs(floatBits - ldBits);

    // Expect each to be much more accurate than the one before it.
    EXPECT_LT(kahanDErrorUlps * 10000, doubleErrorUlps) <<
        kahanDErrorUlps << " - " << doubleErrorUlps;
    // Less slop between double and Kahan with floats.
    EXPECT_LT(doubleErrorUlps * 1000, kahanFErrorUlps) <<
        doubleErrorUlps << " - " << kahanFErrorUlps;
    EXPECT_LT(kahanFErrorUlps * 10000, floatErrorUlps) <<
        kahanFErrorUlps << " - " << floatErrorUlps;
}
