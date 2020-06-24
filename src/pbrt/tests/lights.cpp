
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/lights.h>
#include <pbrt/transform.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <cmath>
#include <memory>

using namespace pbrt;

TEST(SpotLight, Power) {
    static ConstantSpectrum I(10.);
    Transform id;
    AnimatedTransform identity(&id);
    SpotLight light(identity, MediumInterface(), &I,
                    60 /* total width */,
                    40 /* falloff start */, Allocator());

    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    SampledSpectrum phi = light.Phi(lambda);

    int nSamples = 1024*1024;
    double phiSampled = 0;
    for (int i = 0 ; i < nSamples; ++i) {
        Vector3f w = SampleUniformSphere({RadicalInverse(0, i), RadicalInverse(1, i)});
        phiSampled += (I.Sample(lambda) * light.Falloff(w))[0];
    }
    phiSampled /= (nSamples * UniformSpherePDF());

    EXPECT_LT(std::abs(phiSampled - phi[0]), 1e-3) <<
        " qmc: " << phiSampled << ", closed-form: " << phi[0];
}

TEST(SpotLight, Sampling) {
    static ConstantSpectrum I(10.);
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);

    int widthStart[][2] = { { 50, 0 }, { 40, 10 }, { 60, 5 }, {70 , 70 } };
    Transform id;
    AnimatedTransform identity(&id);
    for (auto ws : widthStart) {
        SpotLight light(identity, MediumInterface(), &I,
                        ws[0] /* total width */,
                        ws[1] /* falloff start */, Allocator());

        RNG rng;
        for (int i = 0; i < 100; ++i) {
            Point2f u1 { rng.Uniform<Float>(), rng.Uniform<Float>() };
            Point2f u2 { rng.Uniform<Float>(), rng.Uniform<Float>() };
            pstd::optional<LightLeSample> ls = light.Sample_Le(u1, u2, lambda, 0 /* time */);
            EXPECT_TRUE(ls.has_value());
            EXPECT_TRUE(ls->ray.o == Point3f(0, 0, 0));
            EXPECT_EQ(1, ls->pdfPos);
            // Importance should be perfect, so a single sample should
            // compute power with zero variance.
            EXPECT_LT(std::abs(light.Phi(lambda)[0] - (ls->L / ls->pdfDir)[0]), 1e-3);
        }
    }
}

static Image MakeLightImage(Point2i res) {
    Image image(PixelFormat::Float, res, { "R", "G", "B" });
    for (int y = 0; y < res[1]; ++y)
        for (int x = 0; x < res[0]; ++x) {
            Float val = 0;
            if (((x >= 30 && x <= 200) || x > 400) && y >= 40 && y <= 220) {
                val = .2 + std::sin(100 * x * y / Float(res[0] * res[1]));
                val = std::max(Float(0), val);
            }
            image.SetChannels({x, y}, {val, val, val});
        }
    return image;
}

TEST(GoniometricLight, Power) {
    Image image = MakeLightImage({512, 256});
    image = image.SelectChannels(*image.GetChannelDesc({"R"}));

    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    static ConstantSpectrum I(10.);
    auto id = std::make_unique<Transform>();
    AnimatedTransform identity(id.get());
    GoniometricLight light(AnimatedTransform(id.get()), MediumInterface(), &I, std::move(image),
                           RGBColorSpace::sRGB, Allocator());

    SampledSpectrum phi = light.Phi(lambda);

    int sqrtSamples = 1024;
    int nSamples = sqrtSamples * sqrtSamples;
    double phiSampled = 0;
    for (Point2f u : Hammersley2D(nSamples)) {
        Vector3f w = SampleUniformSphere(u);
        phiSampled += light.Scale(w, lambda)[0];
    }
    phiSampled /= (nSamples * UniformSpherePDF());

    EXPECT_LT(std::abs(phiSampled - phi[0]), 1e-3) <<
        " qmc: " << phiSampled << ", closed-form: " << phi[0];
}

static void testPhiVsSampled(LightHandle light, const SampledWavelengths &lambda) {
    double sum = 0;
    int count = 100000;
    for (int i = 0; i < count; ++i) {
        Point2f u1 { RadicalInverse(0, i), RadicalInverse(1, i) };
        Point2f u2 { RadicalInverse(2, i), RadicalInverse(3, i) };
        pstd::optional<LightLeSample> ls = light.Sample_Le(u1, u2, lambda, 0 /* time */);
        if (!ls)
            continue;

        EXPECT_TRUE(ls->ray.o == Point3f(0, 0, 0));
        EXPECT_EQ(1, ls->pdfPos);
        sum += ls->L[0] / ls->pdfDir;
    }
    SampledSpectrum Phi = light.Phi(lambda);
    EXPECT_LT(std::abs(sum / count - Phi[0]) / Phi[0], 1e-2) <<
        Phi[0] << ", sampled " << sum / count;
}

TEST(GoniometricLight, Sampling) {
    Image image = MakeLightImage({512, 256});
    image = image.SelectChannels(*image.GetChannelDesc({"R"}));

    static ConstantSpectrum I(10.);
    static Transform id;
    AnimatedTransform identity(&id);
    GoniometricLight light(identity, MediumInterface(), &I, std::move(image),
                           RGBColorSpace::sRGB, Allocator());
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    testPhiVsSampled(LightHandle(&light), lambda);
}

TEST(ProjectionLight, Power) {
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    for (Point2i res : { Point2i(512, 256), Point2i(300, 900) }) {
        Image image = MakeLightImage(res);

        auto id = std::make_unique<Transform>();
        AnimatedTransform identity(id.get());
        ProjectionLight light(identity, MediumInterface(), std::move(image),
                              RGBColorSpace::sRGB,
                              10 /* scale */,
                              30 /* fov */, Allocator());

        SampledSpectrum phi = light.Phi(lambda);

        int nSamples = 1024*1024;
        double phiSampled = 0;
        for (Point2f u : Hammersley2D(nSamples)) {
            Vector3f w = SampleUniformSphere(u);
            phiSampled += light.Projection(w, lambda)[0];
        }
        phiSampled /= (nSamples * UniformSpherePDF());

        EXPECT_LT(std::abs(phiSampled - phi[0]), 1e-3) << "res: " << res <<
            " qmc: " << phiSampled << ", closed-form: " << phi[0];
    }
}

TEST(ProjectionLight, Sampling) {
    for (Point2i res : { Point2i(512, 256), Point2i(300, 900) }) {
        Image image = MakeLightImage(res);

        SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
        Transform id;
        AnimatedTransform identity(&id);
        ProjectionLight light(identity, MediumInterface(), std::move(image),
                              RGBColorSpace::sRGB,
                              10. /* scale */,
                              30 /* fov */, Allocator());

        testPhiVsSampled(LightHandle(&light), lambda);
    }
}
