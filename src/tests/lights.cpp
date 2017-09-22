
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

    Spectrum phi = light.Power();

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

    Spectrum phi = light.Power();

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

        Spectrum phi = light.Power();

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
