
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

using namespace pbrt;

TEST(RGBColorSpace, RGBXYZ) {
    for (const RGBColorSpace &cs : { *RGBColorSpace::ACES2065_1, *RGBColorSpace::Rec2020, *RGBColorSpace::sRGB }) {
        XYZ xyz = cs.ToXYZ({1, 1, 1});
        RGB rgb = cs.ToRGB(xyz);
        EXPECT_LT(std::abs(1 - rgb[0]), 1e-4);
        EXPECT_LT(std::abs(1 - rgb[1]), 1e-4);
        EXPECT_LT(std::abs(1 - rgb[2]), 1e-4);
    }
}

TEST(RGBColorSpace, sRGB) {
    const RGBColorSpace &sRGB = *RGBColorSpace::sRGB;

    // Make sure the matrix values are sensible by throwing the x, y, and z
    // basis vectors at it to pull out columns.
    RGB rgb = sRGB.ToRGB({1, 0, 0});
    EXPECT_LT(std::abs(3.2406 - rgb[0]), 1e-3);
    EXPECT_LT(std::abs(-.9689 - rgb[1]), 1e-3);
    EXPECT_LT(std::abs(.0557 - rgb[2]), 1e-3);

    rgb = sRGB.ToRGB({0, 1, 0});
    EXPECT_LT(std::abs(-1.5372 - rgb[0]), 1e-3);
    EXPECT_LT(std::abs(1.8758 - rgb[1]), 1e-3);
    EXPECT_LT(std::abs(-.2040 - rgb[2]), 1e-3);

    rgb = sRGB.ToRGB({0, 0, 1});
    EXPECT_LT(std::abs(-.4986 - rgb[0]), 1e-3);
    EXPECT_LT(std::abs(.0415 - rgb[1]), 1e-3);
    EXPECT_LT(std::abs(1.0570 - rgb[2]), 1e-3);
}

TEST(RGBColorSpace, StdIllumWhitesRGB) {
    XYZ xyz = SpectrumToXYZ(&RGBColorSpace::sRGB->illuminant);
    RGB rgb = RGBColorSpace::sRGB->ToRGB(xyz);
    EXPECT_GE(rgb.r, .99);
    EXPECT_LE(rgb.r, 1.01);
    EXPECT_GE(rgb.g, .99);
    EXPECT_LE(rgb.g, 1.01);
    EXPECT_GE(rgb.b, .99);
    EXPECT_LE(rgb.b, 1.01);
}

TEST(RGBColorSpace, StdIllumWhiteRec2020) {
    XYZ xyz = SpectrumToXYZ(&RGBColorSpace::Rec2020->illuminant);
    RGB rgb = RGBColorSpace::Rec2020->ToRGB(xyz);
    EXPECT_GE(rgb.r, .99);
    EXPECT_LE(rgb.r, 1.01);
    EXPECT_GE(rgb.g, .99);
    EXPECT_LE(rgb.g, 1.01);
    EXPECT_GE(rgb.b, .99);
    EXPECT_LE(rgb.b, 1.01);
}

TEST(RGBColorSpace, StdIllumWhiteACES2065_1) {
    XYZ xyz = SpectrumToXYZ(&RGBColorSpace::ACES2065_1->illuminant);
    RGB rgb = RGBColorSpace::ACES2065_1->ToRGB(xyz);
    EXPECT_GE(rgb.r, .99);
    EXPECT_LE(rgb.r, 1.01);
    EXPECT_GE(rgb.g, .99);
    EXPECT_LE(rgb.g, 1.01);
    EXPECT_GE(rgb.b, .99);
    EXPECT_LE(rgb.b, 1.01);
}

TEST(RGBSpec, MaxValue) {
    RNG rng;
    for (const auto &cs : { *RGBColorSpace::sRGB, *RGBColorSpace::Rec2020, *RGBColorSpace::ACES2065_1 }) {
        for (int i = 0; i < 100; ++i) {
            RGB rgb(rng.Uniform<Float>(), rng.Uniform<Float>(), rng.Uniform<Float>());
            RGBReflectanceSpectrum rs(cs, rgb);

            Float m = rs.MaxValue();
            Float sm = 0;
            for (Float lambda = 360; lambda <= 830; lambda += 1./16.)
                sm = std::max(sm, rs(lambda));
            EXPECT_LT(std::abs((sm - m) / sm), 1e-4) << "sampled " << sm << " MaxValue " << m << " for " << rs;
        }
    }
}

TEST(RGBSpec, RoundTripsRGB) {
    RNG rng;
    const RGBColorSpace &cs = *RGBColorSpace::sRGB;

    for (int i = 0; i < 100; ++i) {
        RGB rgb(rng.Uniform<Float>(),
                rng.Uniform<Float>(),
                rng.Uniform<Float>());
        RGBReflectanceSpectrum rs(cs, rgb);

        ProductSpectrum rsIllum(&rs, &cs.illuminant);
        XYZ xyz = SpectrumToXYZ(&rsIllum);
        RGB rgb2 = cs.ToRGB(xyz);

        // Some error comes from the fact that piecewise linear (at 5nm)
        // CIE curves were used for the optimization while we use piecewise
        // linear at 1nm spacing converted to 1nm constant / densely
        // sampled.
        Float eps = .01;
        EXPECT_LT(std::abs(rgb.r - rgb2.r), eps) << rgb << " vs " << rgb2;
        EXPECT_LT(std::abs(rgb.g - rgb2.g), eps) << rgb << " vs " << rgb2;
        EXPECT_LT(std::abs(rgb.b - rgb2.b), eps) << rgb << " vs " << rgb2;
    }
}

TEST(RGBSpec, RoundTripRec2020) {
    RNG rng;
    const RGBColorSpace &cs = *RGBColorSpace::Rec2020;

    for (int i = 0; i < 100; ++i) {
        RGB rgb(.1 + .7 * rng.Uniform<Float>(),
                .1 + .7 * rng.Uniform<Float>(),
                .1 + .7 * rng.Uniform<Float>());
        RGBReflectanceSpectrum rs(cs, rgb);

        ProductSpectrum rsIllum(&rs, &cs.illuminant);
        XYZ xyz = SpectrumToXYZ(&rsIllum);
        RGB rgb2 = cs.ToRGB(xyz);

        Float eps = .01;
        EXPECT_LT(std::abs(rgb.r - rgb2.r), eps) << rgb << " vs " << rgb2 << " xyz " << xyz;
        EXPECT_LT(std::abs(rgb.g - rgb2.g), eps) << rgb << " vs " << rgb2 << " xyz " << xyz;
        EXPECT_LT(std::abs(rgb.b - rgb2.b), eps) << rgb << " vs " << rgb2 << " xyz " << xyz;
    }
}

TEST(RGBSpec, RoundTripACES) {
    RNG rng;
    const RGBColorSpace &cs = *RGBColorSpace::ACES2065_1;

    for (int i = 0; i < 100; ++i) {
        RGB rgb(.3 + .4 * rng.Uniform<Float>(),
                .3 + .4 * rng.Uniform<Float>(),
                .3 + .4 * rng.Uniform<Float>());
        RGBReflectanceSpectrum rs(cs, rgb);

        ProductSpectrum rsIllum(&rs, &cs.illuminant);
        XYZ xyz = SpectrumToXYZ(&rsIllum);
        RGB rgb2 = cs.ToRGB(xyz);

        Float eps = .01;
        EXPECT_LT(std::abs(rgb.r - rgb2.r), eps) << rgb << " vs " << rgb2 << " xyz " << xyz;
        EXPECT_LT(std::abs(rgb.g - rgb2.g), eps) << rgb << " vs " << rgb2 << " xyz " << xyz;
        EXPECT_LT(std::abs(rgb.b - rgb2.b), eps) << rgb << " vs " << rgb2 << " xyz " << xyz;
    }
}

