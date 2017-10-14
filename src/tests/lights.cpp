
#include "tests/gtest/gtest.h"

#include "pbrt.h"
#include "lowdiscrepancy.h"
#include "image.h"
#include "medium.h"
#include "spectrum.h"
#include "sampling.h"
#include "lights/goniometric.h"
#include "lights/projection.h"
#include "lights/spot.h"
#include "util/transform.h"

#include <cmath>

using namespace pbrt;

TEST(SpotLight, Power) {
    Spectrum I(10.);
    SpotLight light(Transform(), MediumInterface(), I,
                    60 /* total width */,
                    40 /* falloff start */,
                    nullptr);

    Spectrum phi = light.Phi();

    int nSamples = 1024*1024;
    double phiSampled = 0;
    for (int i = 0 ; i < nSamples; ++i) {
        Vector3f w = UniformSampleSphere({RadicalInverse(0, i), RadicalInverse(1, i)});
        phiSampled += (I * light.Falloff(w))[0];
    }
    phiSampled /= (nSamples * UniformSpherePdf());

    EXPECT_LT(std::abs(phiSampled - phi[0]), 1e-3) <<
        " qmc: " << phiSampled << ", closed-form: " << phi[0];
}

TEST(SpotLight, Sampling) {
    Spectrum I(10.);

    int widthStart[][2] = { { 50, 0 }, { 40, 10 }, { 60, 5 }, {70 , 70 } };
    for (auto ws : widthStart) {
        SpotLight light(Transform(), MediumInterface(), I,
                        ws[0] /* total width */,
                        ws[1] /* falloff start */,
                        nullptr);

        RNG rng;
        for (int i = 0; i < 100; ++i) {
            Point2f u1 { rng.UniformFloat(), rng.UniformFloat() };
            Point2f u2 { rng.UniformFloat(), rng.UniformFloat() };
            Ray ray;
            Normal3f n;
            Float pdfPos, pdfDir;
            Spectrum Li = light.Sample_Le(u1, u2, 0 /* time */, &ray, &n, &pdfPos,
                                          &pdfDir);
            EXPECT_TRUE(ray.o == Point3f(0, 0, 0));
            EXPECT_EQ(1, pdfPos);
            // Importance should be perfect, so a single sample should
            // compute power with zero variance.
            EXPECT_LT(std::abs(light.Phi()[0] - (Li / pdfDir)[0]), 1e-3);
        }
    }
}

static Image MakeLightImage(Point2i res) {
    Image image(PixelFormat::Y32, res);
    for (int y = 0; y < image.resolution[1]; ++y)
        for (int x = 0; x < image.resolution[0]; ++x) {
            Float val = 0;
            if (((x >= 30 && x <= 200) || x > 400) && y >= 40 && y <= 220) {
                val = .2 + std::sin(100 * x * y / Float(image.resolution[0] *
                                                        image.resolution[1]));
                val = std::max(Float(0), val);
            }
            image.SetChannel({x, y}, 0, val);
        }
    return image;
}

TEST(GoniometricLight, Power) {
    Image image = MakeLightImage({512, 256});

    Spectrum I(10.);
    GonioPhotometricLight light(Transform(), MediumInterface(), I, std::move(image),
                                nullptr);

    Spectrum phi = light.Phi();

    int nSamples = 1024*1024;
    double phiSampled = 0;
    for (int i = 0 ; i < nSamples; ++i) {
        Vector3f w = UniformSampleSphere({RadicalInverse(0, i), RadicalInverse(1, i)});
        phiSampled += light.Scale(w)[0];
    }
    phiSampled /= (nSamples * UniformSpherePdf());

    EXPECT_LT(std::abs(phiSampled - phi[0]), 1e-3) <<
        " qmc: " << phiSampled << ", closed-form: " << phi[0];
}

TEST(ProjectionLight, Power) {
    for (Point2i res : { Point2i(512, 256), Point2i(30, 90) }) {
        Image image = MakeLightImage(res);

        Spectrum I(10.);
        ProjectionLight light(Transform(), MediumInterface(), I, std::move(image),
                              30 /* fov */, nullptr);

        Spectrum phi = light.Phi();

        int nSamples = 1024*1024;
        double phiSampled = 0;
        for (int i = 0 ; i < nSamples; ++i) {
            Vector3f w = UniformSampleSphere({RadicalInverse(0, i), RadicalInverse(1, i)});
            phiSampled += light.Projection(w)[0];
        }
        phiSampled /= (nSamples * UniformSpherePdf());

        EXPECT_LT(std::abs(phiSampled - phi[0]), 1e-3) << "res: " << res <<
            " qmc: " << phiSampled << ", closed-form: " << phi[0];
    }
}
