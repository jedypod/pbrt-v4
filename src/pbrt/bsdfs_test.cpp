// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/bsdf.h>
#include <pbrt/interaction.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/shapes.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>

using namespace pbrt;

#if 0
TEST(RoughDielectric, Viz) {
    Vector3f wo = Normalize(Vector3f(1,.3,1));
    TrowbridgeReitzDistribution rough(.5, .5);
    Float eta = 1.5;
    DielectricInterfaceBxDF di(eta, &rough);

    int res = 512;
    Image f(PixelFormat::Float, {res, res}, {"Y"});
    Image samplef(PixelFormat::Float, {res, res}, {"Y"});
    Image sampledist(PixelFormat::Float, {res, res}, {"Y"});
    Image pdf(PixelFormat::Float, {res, res}, {"Y"});

    for (int y = 0; y < res; ++y)
        for (int x = 0; x < res; ++x) {
            Point2f p((x + 0.5) / res, (y + 0.5) / res);
            Vector3f wi = EquiAreaSquareToSphere(p);
            f.SetChannel({x, y}, 0, di.f(wo, wi, TransportMode::Radiance)[0] * std::abs(wi.z));
            pdf.SetChannel({x, y}, 0, di.PDF(wo, wi, TransportMode::Radiance, BxDFReflTransFlags::All));
        }

    for (Point3f u : Hammersley3D(1000000)) {
        Vector3f wi;
        auto bs = di.Sample_f(wo, u[0], {u[1], u[2]}, TransportMode::Radiance,
                              BxDFReflTransFlags::All);
        if (!bs || bs->pdf == 0)
            continue;
        Point2f p = EquiAreaSphereToSquare(bs->wi);
        Point2i pi(Clamp(p.x * res, 0, res - 1),Clamp(p.y * res, 0, res - 1));
        if (pi.x == 430 && pi.y == 140)
            std::cerr << "YOLO\n";
        samplef.SetChannel(pi, 0, bs->f[0] / bs->pdf);
        sampledist.SetChannel(pi, 0, sampledist.GetChannel(pi, 0) + 1e-5);
    }

    // pixel 430,140 seems to never be sampled even though f is small but non zero there...

    f.Write("f.exr");
    samplef.Write("samplef.exr");
    sampledist.Write("sampledist.exr");
    pdf.Write("pdf.exr");
}

// NOTE: need to declare this in bxdfs.cpp and initialize in f() and
// sample_f() methods...
namespace pbrt { extern Vector3f evalWh;}

TEST(RoughDielectric, Interface) {
    TrowbridgeReitzDistribution rough(1, 1);
    Float eta = 1.5;
    DielectricInterfaceBxDF di(eta, &rough);

    RNG rng;

    /*
eval wh [ 0.021775303, -0.9268676, -0.37475643 ], mf wh pdf 0.304875
best sampled wh [ 0.021622226, -0.92727023, -0.37376785 ], dot 0.999999, u [ 0.47623, 0.65802383 ]
No can sample. Max dot = 0.881171 with wi [ -0.4703933, -0.8818077, -0.033843935 ]
    */
    {
        Vector3f wo(0.28132406, -0.5946933, -0.7531246), wi(-0.38194194, -0.8124417, 0.44052106);
        std::cerr << "f " << di.f(wo, wi, TransportMode::Radiance);  //         f 0.361925
        EXPECT_GT(Dot(evalWh, Vector3f(0.021622226, -0.92727023, -0.37376785)), .999) << evalWh;
        auto bs = di.Sample_f(wo, 0.9999, Point2f(0.47623, 0.65802383),
                              TransportMode::Radiance, BxDFReflTransFlags::All);
        std::cerr << "bs->wi " << bs->wi << " vs wi " << wi << ", f = " << bs->f << std::endl;
    }
    {
        Vector3f wo( -0.3891811, 0.0051056407, 0.9211471), wi( -0.7639788, 0.6237078, -0.16530311);
        std::cerr << "f " << di.f(wo, wi, TransportMode::Radiance);  //         f 0.0361824
        EXPECT_GT(Dot(evalWh, Vector3f(-0.79865545, 0.48937848, 0.35022575)), .999);
        auto bs = di.Sample_f(wo, 0.9999, Point2f(0.67361665, 0.6510124),
                              TransportMode::Radiance, BxDFReflTransFlags::All);
        std::cerr << "bs->wi " << bs->wi << " vs wi " << wi << ", f = " << bs->f << std::endl;
//CO        return;
    }
    {
        Vector3f wo( 0.8310706, 0.37243307, 0.41305608), wi( -0.53010356, -0.3259088, -0.78279865);
        std::cerr << "f " << di.f(wo, wi, TransportMode::Radiance);  //         f 0.118993
        //EXPECT_GT(Dot(evalWh, Vector3f(-0.046592813, 0.1510447, 0.98742825)), .999) << evalWh;
        auto bs = di.Sample_f(wo, 0.9999, Point2f(0.74932, 0.22056198),
                              TransportMode::Radiance, BxDFReflTransFlags::All);
        std::cerr << "bs->wi " << bs->wi << " vs wi " << wi << ", f = " << bs->f << std::endl;
//CO        return;
    }

    int fMismatches = 0, cantSamples = 0;
    auto err = [](float a, float b) { return std::abs(a-b) / (2 * std::abs(a + b)); };
    for (int i = 0; i < 20; ++i) {
        Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});

        // Do sample_f and f agree?
        for (int j = 0; j < 10; ++j) {
            Float uc = rng.Uniform<Float>();
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            auto bs = di.Sample_f(wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
            if (!bs || !bs->pdf)
                continue;

            SampledSpectrum f = di.f(wo, bs->wi, TransportMode::Radiance);
            if (err(f[0], bs->f[0]) > .001) {
                ++fMismatches;
                std::cerr << "MISMATCH: f " << f[0] << " vs " << bs->f[0] << std::endl;
            }
        }

        // Given a random direction with f non-zero, can we sample it?
        for (int j = 0; j < 10; ++j) {
            Vector3f wi = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            SampledSpectrum f = di.f(wo, wi, TransportMode::Radiance);
            if (!f) continue;

            std::cerr << "----------" << std::endl;
            std::cerr << "wo " << wo << ", wi " << wi << ", f " << f[0] << std::endl;

            Vector3f wh = evalWh;
            std::cerr << "eval wh " << wh << ", mf wh pdf " <<
                MicrofacetDistributionHandle(&rough).PDF(wo, wh)  << std::endl;

            Float maxDot = -1;
            Vector3f bestWh;
            Point2f bestU;
            for (Point2f u : Hammersley2D(300000)) {
                Vector3f ww = rough.Sample_wm(wo, u);
                if (Dot(ww, wh) > maxDot) {
                    maxDot = Dot(ww, wh);
                    bestWh = ww;
                    bestU = u;
                }
            }
            std::cerr << "best sampled wh " << bestWh << ", dot " << maxDot << ", u " << bestU  << std::endl;
            if (maxDot < .95) std::cerr << "MEH!"  << std::endl;

            maxDot = -1;
            Vector3f bestWi;
            for (Point2f u : Hammersley2D(30000)) {
                Float uc = SameHemisphere(wo, wi) > 0 ? 0. : 0.999999;
                auto bs = di.Sample_f(wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
                if (bs && bs->pdf && bs->f) {
                    Float d = Dot(wi, bs->wi);
                    if (d > maxDot) {
                        maxDot = d;
                        bestWi = bs->wi;
                    }
                }
            }
            if (maxDot < .95) {
                std::cerr << "No can sample. Max dot = " << maxDot << " with wi " << bestWi << std::endl;
                ++cantSamples;
            }
        }
    }

    EXPECT_EQ(0, fMismatches);
    EXPECT_EQ(0, cantSamples);

    return;

    // it's always non-zero (from outside at least), right?
    int nvalid = 0;
    for (int i = 0; i < 1000; ++i) {
        Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
//CO        if (wo.z > 0) wo.z = -wo.z;
        Vector3f wi = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
//CO        if (wi.z < 0) wi.z = -wi.z;
        SampledSpectrum f = di.f(wo, wi, TransportMode::Radiance);
//CO        if (!f) std::cerr << i << ": f is black for " << wo << ", " << wi << std::endl;
        if (!f) continue;
        ++nvalid;

        // How close does sampling get?
        Float maxDot = -1;
        for (int j = 0; j < 100000; ++j) {
            Float uc = rng.Uniform<Float>();
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            auto bs = di.Sample_f(wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
            if (bs && bs->pdf && bs->f)
                maxDot = std::max(maxDot, Dot(wi, bs->wi));
        }
        if (maxDot < .9) {
            std::cerr << maxDot << " max dot for wo " << wo << ", wi " << wi << ", f " << f <<
                (SameHemisphere(wo, wi) ? ", REFL " : ", TRANS") << std::endl;

            // How close can we come with random normals in the top hemisphere?
            maxDot = -1;
            Vector3f bestWm;
            for (int j = 0; j < 100000; ++j) {
                Vector3f wm = SampleUniformHemisphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
                ASSERT_GE(wm.z, 0);
                Vector3f wip;
                Float ep = wo.z > 0 ? (1 / eta) : eta;
                bool tir = !Refract(wo, (Normal3f)wm, ep, &wip);
                if (!tir && Dot(wi, wip) > maxDot) {
                    maxDot = Dot(wi, wip);
                    bestWm = wm;
                }
            }
            std::cerr << "Exhaustive got to " << maxDot << " with wm " << bestWm <<
                " which has pdf " <<
                MicrofacetDistributionHandle(&rough).PDF(wo, bestWm)  << std::endl;
            std::cerr << "wm . wo = " << Dot(wo, bestWm) << std::endl;

            /*
SUSPICIOUS OF BRDF EVAL HERE... RETURNING >0 when SOMETHING SHOULD GO TO 0...

But also, best wm is in same hemisphere as wo: how come it's never sampled
review heitz jcgt stuff, implementation here.

Try pre-improved version of TrowbridgeReitzSample

...but, uniform hemi sample of wh doesn't work, right???
(wait, what, why??? ---> so is the problem that TR sample is fine, but then what we do with it is wrong???

revisit uniform TR wh sampling--seems like it should work??!?

            */
            // How close can the MF distribution get us?
            maxDot = -1;
            for (int j = 0; j < 100000; ++j) {
                Vector3f wm = rough.Sample_wm(wo, {rng.Uniform<Float>(), rng.Uniform<Float>()});
                ASSERT_GE(wm.z, 0);
                Vector3f wip;
                Float ep = wo.z > 0 ? (1 / eta) : eta;
                bool tir = !Refract(wo, (Normal3f)wm, ep, &wip);
                if (!tir && Dot(wi, wip) > maxDot) {
                    maxDot = Dot(wi, wip);
                    bestWm = wm;
                }
            }
            std::cerr << "TR distrib got to " << maxDot << " with wm " << bestWm <<
                " which has pdf " << MicrofacetDistributionHandle(&rough).PDF(wo, bestWm)  << std::endl;
            std::cerr << "Best . wo " << Dot(wo, bestWm)  << std::endl;

            std::cerr << "Redo " << di.f(wo, wi, TransportMode::Radiance)  << std::endl;
        }
    }

    for (int i = 0; i < 5; ++i) {
        Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        if (wo.z < 0) wo.z = -wo.z;

        constexpr int nTheta = 256;
        double sums[nTheta] = { 0. };
        int counts[nTheta] = { 0 };
        for (int j = 0; j < 100000; ++j) {
            Float uc = rng.Uniform<Float>();
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());

            auto bs = di.Sample_f(wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
            if (!bs || bs->pdf == 0 || !bs->f) {
//CO                std::cerr << "Sample failed"  << std::endl;
                continue;
            }

//CO            std::cerr << Dot(wo, bs->wi) << " for wi " << bs->wi  << std::endl;
            Float theta = SphericalTheta(bs->wi);
            int t = Clamp(theta / Pi * nTheta, 0, nTheta-1);
            sums[t] += bs->f[0] / bs->pdf;
            ++counts[t];
        }

        fprintf(stderr, "WO %f, %f, %f\n", wo.x, wo.y, wo.z);
        for (int j = 0; j < nTheta; ++j)
            fprintf(stderr, "%d: count %d avg %f\n", j, counts[j], sums[j] / counts[j]);
        fprintf(stderr, "---------\n");
    }
    fprintf(stderr, "nvalid %d\n", nvalid);
}
#endif

/* The null hypothesis will be rejected when the associated
   p-value is below the significance level specified here. */
#define CHI2_SLEVEL 0.01

/* Resolution of the frequency table discretization. The azimuthal
   resolution is twice this value. */
#define CHI2_THETA_RES 10
#define CHI2_PHI_RES (2 * CHI2_THETA_RES)

/* Number of MC samples to compute the observed frequency table */
#define CHI2_SAMPLECOUNT 1000000

/* Minimum expected bin frequency. The chi^2 test does not
   work reliably when the expected frequency in a cell is
   low (e.g. less than 5), because normality assumptions
   break down in this case. Therefore, the implementation
   will merge such low-frequency cells when they fall below
   the threshold specified here. */
#define CHI2_MINFREQ 5

/* Each provided BSDF will be tested for a few different
   incident directions. The value specified here determines
   how many tests will be executed per BSDF */
#define CHI2_RUNS 5

/// Regularized lower incomplete gamma function (based on code from Cephes)
double RLGamma(double a, double x) {
    const double epsilon = 0.000000000000001;
    const double big = 4503599627370496.0;
    const double bigInv = 2.22044604925031308085e-16;
    if (a < 0 || x < 0)
        throw std::runtime_error("LLGamma: invalid arguments range!");

    if (x == 0)
        return 0.0f;

    double ax = (a * std::log(x)) - x - std::lgamma(a);
    if (ax < -709.78271289338399)
        return a < x ? 1.0 : 0.0;

    if (x <= 1 || x <= a) {
        double r2 = a;
        double c2 = 1;
        double ans2 = 1;

        do {
            r2 = r2 + 1;
            c2 = c2 * x / r2;
            ans2 += c2;
        } while ((c2 / ans2) > epsilon);

        return std::exp(ax) * ans2 / a;
    }

    int c = 0;
    double y = 1 - a;
    double z = x + y + 1;
    double p3 = 1;
    double q3 = x;
    double p2 = x + 1;
    double q2 = z * x;
    double ans = p2 / q2;
    double error;

    do {
        c++;
        y += 1;
        z += 2;
        double yc = y * c;
        double p = (p2 * z) - (p3 * yc);
        double q = (q2 * z) - (q3 * yc);

        if (q != 0) {
            double nextans = p / q;
            error = std::abs((ans - nextans) / nextans);
            ans = nextans;
        } else {
            // zero div, skip
            error = 1;
        }

        // shift
        p3 = p2;
        p2 = p;
        q3 = q2;
        q2 = q;

        // normalize fraction when the numerator becomes large
        if (std::abs(p) > big) {
            p3 *= bigInv;
            p2 *= bigInv;
            q3 *= bigInv;
            q2 *= bigInv;
        }
    } while (error > epsilon);

    return 1.0 - (std::exp(ax) * ans);
}

/// Chi^2 distribution cumulative distribution function
double Chi2CDF(double x, int dof) {
    if (dof < 1 || x < 0) {
        return 0.0;
    } else if (dof == 2) {
        return 1.0 - std::exp(-0.5 * x);
    } else {
        return (Float)RLGamma(0.5 * dof, 0.5 * x);
    }
}

/// Adaptive Simpson integration over an 1D interval
Float AdaptiveSimpson(const std::function<Float(Float)>& f, Float x0, Float x1,
                      Float eps = 1e-6f, int depth = 6) {
    int count = 0;
    /* Define an recursive lambda function for integration over subintervals */
    std::function<Float(Float, Float, Float, Float, Float, Float, Float, Float, int)>
        integrate = [&](Float a, Float b, Float c, Float fa, Float fb, Float fc, Float I,
                        Float eps, int depth) {
            /* Evaluate the function at two intermediate points */
            Float d = 0.5f * (a + b), e = 0.5f * (b + c), fd = f(d), fe = f(e);

            /* Simpson integration over each subinterval */
            Float h = c - a, I0 = (Float)(1.0 / 12.0) * h * (fa + 4 * fd + fb),
                  I1 = (Float)(1.0 / 12.0) * h * (fb + 4 * fe + fc), Ip = I0 + I1;
            ++count;

            /* Stopping criterion from J.N. Lyness (1969)
              "Notes on the adaptive Simpson quadrature routine" */
            if (depth <= 0 || std::abs(Ip - I) < 15 * eps) {
                // Richardson extrapolation
                return Ip + (Float)(1.0 / 15.0) * (Ip - I);
            }

            return integrate(a, d, b, fa, fd, fb, I0, .5f * eps, depth - 1) +
                   integrate(b, e, c, fb, fe, fc, I1, .5f * eps, depth - 1);
        };
    Float a = x0, b = 0.5f * (x0 + x1), c = x1;
    Float fa = f(a), fb = f(b), fc = f(c);
    Float I = (c - a) * (Float)(1.0 / 6.0) * (fa + 4 * fb + fc);
    return integrate(a, b, c, fa, fb, fc, I, eps, depth);
}

/// Nested adaptive Simpson integration over a 2D rectangle
Float AdaptiveSimpson2D(const std::function<Float(Float, Float)>& f, Float x0, Float y0,
                        Float x1, Float y1, Float eps = 1e-6f, int depth = 6) {
    /* Lambda function that integrates over the X axis */
    auto integrate = [&](Float y) {
        return AdaptiveSimpson(std::bind(f, std::placeholders::_1, y), x0, x1, eps,
                               depth);
    };
    Float value = AdaptiveSimpson(integrate, y0, y1, eps, depth);
    return value;
}

/// Generate a histogram of the BSDF density function via MC sampling
void FrequencyTable(const BSDF* bsdf, const Vector3f& wo, RNG& rng, int sampleCount,
                    int thetaRes, int phiRes, Float* target) {
    memset(target, 0, thetaRes * phiRes * sizeof(Float));

    Float factorTheta = thetaRes / Pi, factorPhi = phiRes / (2 * Pi);

    BxDFFlags sampledType;
    Vector3f wi;
    Float pdf;

    for (int i = 0; i < sampleCount; ++i) {
        Float u = rng.Uniform<Float>();
        Point2f sample{rng.Uniform<Float>(), rng.Uniform<Float>()};
        pstd::optional<BSDFSample> bs = bsdf->Sample_f(wo, u, sample);

        if (!bs || bs->IsSpecular())
            continue;

        Vector3f wiL = bsdf->WorldToLocal(bs->wi);

        Point2f coords(SafeACos(wiL.z) * factorTheta,
                       std::atan2(wiL.y, wiL.x) * factorPhi);

        if (coords.y < 0)
            coords.y += 2 * Pi * factorPhi;

        int thetaBin = std::min(std::max(0, (int)std::floor(coords.x)), thetaRes - 1);
        int phiBin = std::min(std::max(0, (int)std::floor(coords.y)), phiRes - 1);

        target[thetaBin * phiRes + phiBin] += 1;
    }
}

// Numerically integrate the probability density function over rectangles in
// spherical coordinates.
void IntegrateFrequencyTable(const BSDF* bsdf, const Vector3f& wo, int sampleCount,
                             int thetaRes, int phiRes, Float* target) {
    memset(target, 0, thetaRes * phiRes * sizeof(Float));

    Float factorTheta = Pi / thetaRes, factorPhi = (2 * Pi) / phiRes;

    for (int i = 0; i < thetaRes; ++i) {
        for (int j = 0; j < phiRes; ++j) {
            *target++ =
                sampleCount *
                AdaptiveSimpson2D(
                    [&](Float theta, Float phi) -> Float {
                        Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
                        Float cosPhi = std::cos(phi), sinPhi = std::sin(phi);
                        Vector3f wiL(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta);
                        return bsdf->PDF(wo, bsdf->LocalToWorld(wiL)) * sinTheta;
                    },
                    i* factorTheta, j* factorPhi, (i + 1) * factorTheta,
                    (j + 1) * factorPhi);
        }
    }
}

/// Write the frequency tables to disk in a format that is nicely plottable by
/// Octave and MATLAB
void DumpTables(const Float* frequencies, const Float* expFrequencies, int thetaRes,
                int phiRes, const char* filename) {
    std::ofstream f(filename);

    f << "frequencies = [ ";
    for (int i = 0; i < thetaRes; ++i) {
        for (int j = 0; j < phiRes; ++j) {
            f << frequencies[i * phiRes + j];
            if (j + 1 < phiRes)
                f << ", ";
        }
        if (i + 1 < thetaRes)
            f << "; ";
    }
    f << " ];" << std::endl << "expFrequencies = [ ";
    for (int i = 0; i < thetaRes; ++i) {
        for (int j = 0; j < phiRes; ++j) {
            f << expFrequencies[i * phiRes + j];
            if (j + 1 < phiRes)
                f << ", ";
        }
        if (i + 1 < thetaRes)
            f << "; ";
    }
    f << " ];" << std::endl
      << "colormap(jet);" << std::endl
      << "clf; subplot(2,1,1);" << std::endl
      << "imagesc(frequencies);" << std::endl
      << "title('Observed frequencies');" << std::endl
      << "axis equal;" << std::endl
      << "subplot(2,1,2);" << std::endl
      << "imagesc(expFrequencies);" << std::endl
      << "axis equal;" << std::endl
      << "title('Expected frequencies');" << std::endl;
    f.close();
}

/// Run A Chi^2 test based on the given frequency tables
std::pair<bool, std::string> Chi2Test(const Float* frequencies,
                                      const Float* expFrequencies, int thetaRes,
                                      int phiRes, int sampleCount, Float minExpFrequency,
                                      Float significanceLevel, int numTests) {
    struct Cell {
        Float expFrequency;
        size_t index;
    };

    /* Sort all cells by their expected frequencies */
    std::vector<Cell> cells(thetaRes * phiRes);
    for (size_t i = 0; i < cells.size(); ++i) {
        cells[i].expFrequency = expFrequencies[i];
        cells[i].index = i;
    }
    std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b) {
        return a.expFrequency < b.expFrequency;
    });

    /* Compute the Chi^2 statistic and pool cells as necessary */
    Float pooledFrequencies = 0, pooledExpFrequencies = 0, chsq = 0;
    int pooledCells = 0, dof = 0;

    for (const Cell& c : cells) {
        if (expFrequencies[c.index] == 0) {
            if (frequencies[c.index] > sampleCount * 1e-5f) {
                /* Uh oh: samples in a c that should be completely empty
                   according to the probability density function. Ordinarily,
                   even a single sample requires immediate rejection of the null
                   hypothesis. But due to finite-precision computations and
                   rounding
                   errors, this can occasionally happen without there being an
                   actual bug. Therefore, the criterion here is a bit more
                   lenient. */

                std::string result =
                    StringPrintf("Encountered %f samples in a c with expected "
                                 "frequency 0. Rejecting the null hypothesis!",
                                 frequencies[c.index]);
                return std::make_pair(false, result);
            }
        } else if (expFrequencies[c.index] < minExpFrequency) {
            /* Pool cells with low expected frequencies */
            pooledFrequencies += frequencies[c.index];
            pooledExpFrequencies += expFrequencies[c.index];
            pooledCells++;
        } else if (pooledExpFrequencies > 0 && pooledExpFrequencies < minExpFrequency) {
            /* Keep on pooling cells until a sufficiently high
               expected frequency is achieved. */
            pooledFrequencies += frequencies[c.index];
            pooledExpFrequencies += expFrequencies[c.index];
            pooledCells++;
        } else {
            Float diff = frequencies[c.index] - expFrequencies[c.index];
            chsq += (diff * diff) / expFrequencies[c.index];
            ++dof;
        }
    }

    if (pooledExpFrequencies > 0 || pooledFrequencies > 0) {
        Float diff = pooledFrequencies - pooledExpFrequencies;
        chsq += (diff * diff) / pooledExpFrequencies;
        ++dof;
    }

    /* All parameters are assumed to be known, so there is no
       additional DF reduction due to model parameters */
    dof -= 1;

    if (dof <= 0) {
        std::string result =
            StringPrintf("The number of degrees of freedom %d is too low!", dof);
        return std::make_pair(false, result);
    }

    /* Probability of obtaining a test statistic at least
       as extreme as the one observed under the assumption
       that the distributions match */
    Float pval = 1 - (Float)Chi2CDF(chsq, dof);

    /* Apply the Sidak correction term, since we'll be conducting multiple
       independent
       hypothesis tests. This accounts for the fact that the probability of a
       failure
       increases quickly when several hypothesis tests are run in sequence. */
    Float alpha = 1.0f - std::pow(1.0f - significanceLevel, 1.0f / numTests);

    if (pval < alpha || !std::isfinite(pval)) {
        std::string result = StringPrintf("Rejected the null hypothesis (p-value = %f, "
                                          "significance level = %f",
                                          pval, alpha);
        return std::make_pair(false, result);
    } else {
        return std::make_pair(true, std::string(""));
    }
}

void TestBSDF(std::function<BSDF*(const SurfaceInteraction&, Allocator)> createBSDF,
              const char* description) {
    const int thetaRes = CHI2_THETA_RES;
    const int phiRes = CHI2_PHI_RES;
    const int sampleCount = CHI2_SAMPLECOUNT;
    Float* frequencies = new Float[thetaRes * phiRes];
    Float* expFrequencies = new Float[thetaRes * phiRes];
    RNG rng;

    int index = 0;
    std::cout.precision(3);

    // Create BSDF, which requires creating a Shape, casting a Ray that
    // hits the shape to get a SurfaceInteraction object.
    BSDF* bsdf = nullptr;
    auto t = std::make_shared<const Transform>(RotateX(-90));
    auto tInv = std::make_shared<const Transform>(Inverse(*t));
    {
        bool reverseOrientation = false;

        std::shared_ptr<Disk> disk = std::make_shared<Disk>(
            t.get(), tInv.get(), reverseOrientation, 0., 1., 0, 360.);
        Point3f origin(0.1, 1,
                       0);  // offset slightly so we don't hit center of disk
        Vector3f direction(0, -1, 0);
        Ray r(origin, direction);
        auto si = disk->Intersect(r);
        ASSERT_TRUE(si.has_value());
        bsdf = createBSDF(si->intr, Allocator());
    }

    for (int k = 0; k < CHI2_RUNS; ++k) {
        /* Randomly pick an outgoing direction on the hemisphere */
        Point2f sample{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Vector3f woL = SampleCosineHemisphere(sample);
        Vector3f wo = bsdf->LocalToWorld(woL);

        FrequencyTable(bsdf, wo, rng, sampleCount, thetaRes, phiRes, frequencies);

        IntegrateFrequencyTable(bsdf, wo, sampleCount, thetaRes, phiRes, expFrequencies);

        std::string filename =
            StringPrintf("/tmp/chi2test_%s_%03i.m", description, ++index);
        DumpTables(frequencies, expFrequencies, thetaRes, phiRes, filename.c_str());

        auto result = Chi2Test(frequencies, expFrequencies, thetaRes, phiRes, sampleCount,
                               CHI2_MINFREQ, CHI2_SLEVEL, CHI2_RUNS);
        EXPECT_TRUE(result.first) << result.second << ", iteration " << k;
    }

    delete[] frequencies;
    delete[] expFrequencies;
}

BSDF* createLambertian(const SurfaceInteraction& si, Allocator alloc) {
    SampledSpectrum Kd(1.);
    return alloc.new_object<BSDF>(
        si, alloc.new_object<DiffuseBxDF>(Kd, SampledSpectrum(0.), 0));
}

BSDF* createMicrofacet(const SurfaceInteraction& si, Allocator alloc, float roughx,
                       float roughy) {
    Float alphax = TrowbridgeReitzDistribution::RoughnessToAlpha(roughx);
    Float alphay = TrowbridgeReitzDistribution::RoughnessToAlpha(roughy);
    MicrofacetDistributionHandle distrib =
        alloc.new_object<TrowbridgeReitzDistribution>(alphax, alphay);
    FresnelHandle fresnel = alloc.new_object<FresnelDielectric>(1.5, true);
    return alloc.new_object<BSDF>(
        si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
    // CO    return alloc.new_object<BSDF>(si,
    // alloc.new_object<DielectricInterface>(1.5, distrib,
    // TransportMode::Radiance));
}

TEST(BSDFSampling, Lambertian) {
    TestBSDF(createLambertian, "Lambertian");
}

TEST(BSDFSampling, TR_VA_0p5) {
    TestBSDF(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            return createMicrofacet(si, alloc, 0.5, 0.5);
        },
        "Trowbridge-Reitz, visible area sample, alpha = 0.5");
}

TEST(BSDFSampling, TR_VA_0p3_0p15) {
    TestBSDF(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            return createMicrofacet(si, alloc, 0.3, 0.15);
        },
        "Trowbridge-Reitz, visible area sample, alpha = 0.3/0.15");
}

///////////////////////////////////////////////////////////////////////////
// Energy Conservation Tests

static void TestEnergyConservation(
    std::function<BSDF*(const SurfaceInteraction&, Allocator)> createBSDF,
    const char* description) {
    RNG rng;

    // Create BSDF, which requires creating a Shape, casting a Ray that
    // hits the shape to get a SurfaceInteraction object.
    auto t = std::make_shared<const Transform>(RotateX(-90));
    auto tInv = std::make_shared<const Transform>(Inverse(*t));

    bool reverseOrientation = false;
    std::shared_ptr<Disk> disk =
        std::make_shared<Disk>(t.get(), tInv.get(), reverseOrientation, 0., 1., 0, 360.);
    Point3f origin(0.1, 1,
                   0);  // offset slightly so we don't hit center of disk
    Vector3f direction(0, -1, 0);
    Ray r(origin, direction);
    auto si = disk->Intersect(r);
    ASSERT_TRUE(si.has_value());
    BSDF* bsdf = createBSDF(si->intr, Allocator());

    for (int i = 0; i < 10; ++i) {
        Point2f uo{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Vector3f woL = SampleUniformHemisphere(uo);
        Vector3f wo = bsdf->LocalToWorld(woL);

        const int nSamples = 16384;
        SampledSpectrum Lo(0.f);
        for (int j = 0; j < nSamples; ++j) {
            Float u = rng.Uniform<Float>();
            Point2f ui{rng.Uniform<Float>(), rng.Uniform<Float>()};
            pstd::optional<BSDFSample> bs = bsdf->Sample_f(wo, u, ui);
            if (bs && bs->pdf > 0)
                Lo += bs->f * AbsDot(bs->wi, si->intr.n) / bs->pdf;
        }
        Lo /= nSamples;

        EXPECT_LT(Lo.MaxComponentValue(), 1.01)
            << description << ": Lo = " << Lo << ", wo = " << wo;
    }
}

TEST(BSDFEnergyConservation, LambertianReflection) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            return alloc.new_object<BSDF>(
                si, alloc.new_object<DiffuseBxDF>(SampledSpectrum(1.f),
                                                  SampledSpectrum(0.), 0));
        },
        "LambertianReflection");
}

TEST(BSDFEnergyConservation, OrenNayar) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            return alloc.new_object<BSDF>(
                si, alloc.new_object<DiffuseBxDF>(SampledSpectrum(1.f),
                                                  SampledSpectrum(0.), 20));
        },
        "Oren-Nayar sigma 20");
}

TEST(BSDFEnergyConservation,
     MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_1_dielectric1_5) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            FresnelHandle fresnel = alloc.new_object<FresnelDielectric>(1.f, 1.5f);
            MicrofacetDistributionHandle distrib =
                alloc.new_object<TrowbridgeReitzDistribution>(0.1, 0.1);
            return alloc.new_object<BSDF>(
                si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
        },
        "MicrofacetReflectionBxDF, Fresnel dielectric, TrowbridgeReitz alpha "
        "0.1");
}

TEST(BSDFEnergyConservation,
     MicrofacetReflectionBxDFTrowbridgeReitz_alpha1_5_dielectric1_5) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            FresnelHandle fresnel = alloc.new_object<FresnelDielectric>(1.f, 1.5f);
            MicrofacetDistributionHandle distrib =
                alloc.new_object<TrowbridgeReitzDistribution>(1.5, 1.5);
            return alloc.new_object<BSDF>(
                si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
        },
        "MicrofacetReflectionBxDF, Fresnel dielectric, TrowbridgeReitz alpha "
        "1.5");
}

TEST(BSDFEnergyConservation,
     MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_01_dielectric1_5) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            FresnelHandle fresnel = alloc.new_object<FresnelDielectric>(1.f, 1.5f);
            MicrofacetDistributionHandle distrib =
                alloc.new_object<TrowbridgeReitzDistribution>(0.01, 0.01);
            return alloc.new_object<BSDF>(
                si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
        },
        "MicrofacetReflectionBxDF, Fresnel dielectric, TrowbridgeReitz alpha "
        "0.01");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_1_conductor) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
            SampledSpectrum etaT = SPDs::MetalAlEta().Sample(lambda);
            SampledSpectrum K = SPDs::MetalAlK().Sample(lambda);
            FresnelHandle fresnel = alloc.new_object<FresnelConductor>(etaT, K);
            MicrofacetDistributionHandle distrib =
                alloc.new_object<TrowbridgeReitzDistribution>(0.1, 0.1);
            return alloc.new_object<BSDF>(
                si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
        },
        "MicrofacetReflectionBxDF, Fresnel conductor, TrowbridgeReitz alpha "
        "0.1");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha1_5_conductor) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
            SampledSpectrum etaT = SPDs::MetalAlEta().Sample(lambda);
            SampledSpectrum K = SPDs::MetalAlK().Sample(lambda);
            FresnelHandle fresnel = alloc.new_object<FresnelConductor>(etaT, K);
            MicrofacetDistributionHandle distrib =
                alloc.new_object<TrowbridgeReitzDistribution>(1.5, 1.5);
            return alloc.new_object<BSDF>(
                si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
        },
        "MicrofacetReflectionBxDF, Fresnel conductor, TrowbridgeReitz alpha "
        "1.5");
}

TEST(BSDFEnergyConservation,
     MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_01_conductor) {
    TestEnergyConservation(
        [](const SurfaceInteraction& si, Allocator alloc) -> BSDF* {
            SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
            SampledSpectrum etaT = SPDs::MetalAlEta().Sample(lambda);
            SampledSpectrum K = SPDs::MetalAlK().Sample(lambda);
            FresnelHandle fresnel = alloc.new_object<FresnelConductor>(etaT, K);

            MicrofacetDistributionHandle distrib =
                alloc.new_object<TrowbridgeReitzDistribution>(0.01, 0.01);
            return alloc.new_object<BSDF>(
                si, alloc.new_object<MicrofacetReflectionBxDF>(distrib, fresnel));
        },
        "MicrofacetReflectionBxDF, Fresnel conductor, TrowbridgeReitz alpha "
        "0.01");
}

// Hair Tests
#if 0
TEST(Hair, Reciprocity) {
  RNG rng;
  for (int i = 0; i < 10; ++i) {
    Hair h(-1 + 2 * rng.Uniform<Float>(), 1.55,
           HairBSDF::SigmaAFromConcentration(.3 + 7.7 * rng.Uniform<Float>()),
           .1 + .9 * rng.Uniform<Float>(),
           .1 + .9 * rng.Uniform<Float>());
    Vector3f wi = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    Spectrum a = h.f(wi, wo) * AbsCosTheta(wo);
    Spectrum b = h.f(wo, wi) * AbsCosTheta(wi);
    EXPECT_EQ(a.y(), b.y()) << h << ", a = " << a << ", b = " << b << ", wi = " << wi
                    << ", wo = " << wo;
  }
}

#endif

TEST(Hair, WhiteFurnace) {
    RNG rng;
    Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    for (Float beta_m = .1; beta_m < 1; beta_m += .2) {
        for (Float beta_n = .1; beta_n < 1; beta_n += .2) {
            // Estimate reflected uniform incident radiance from hair
            Float ySum = 0;

            // More samples for the smooth case, since we're sampling blindly.
            int count = (beta_m < .5 || beta_n < .5) ? 100000 : 20000;

            for (int i = 0; i < count; ++i) {
                SampledWavelengths lambda =
                    SampledWavelengths::SampleImportance(RadicalInverse(0, i));

                Float h = Clamp(-1 + 2. * RadicalInverse(1, i), -.999999, .999999);
                SampledSpectrum sigma_a(0.f);
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);
                Vector3f wi =
                    SampleUniformSphere({RadicalInverse(2, i), RadicalInverse(3, i)});

                SampledSpectrum f =
                    hair.f(wo, wi, TransportMode::Radiance) * AbsCosTheta(wi);
                ySum += f.y(lambda);
            }

            Float avg = ySum / (count * UniformSpherePDF());
            EXPECT_TRUE(avg >= .95 && avg <= 1.05) << avg;
        }
    }
}

TEST(Hair, HOnTheEdge) {
    Vector3f wo(0.54986966, 0.03359017, 0.83457476),
        wi(-0.37383357, -0.91920084, 0.12376696);
    Float h = -1, beta_m = .1, beta_n = .1;
    SampledSpectrum sigma_a(0.f);
    HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);

    SampledSpectrum f = hair.f(wo, wi, TransportMode::Radiance);
}

TEST(Hair, WhiteFurnaceSampled) {
    RNG rng;
    SampledWavelengths lambda = SampledWavelengths::SampleImportance(0.5);
    Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    for (Float beta_m = .1; beta_m < 1; beta_m += .2) {
        for (Float beta_n = .1; beta_n < 1; beta_n += .2) {
            Float ySum = 0;

            int count = 10000;
            for (int i = 0; i < count; ++i) {
                SampledWavelengths lambda =
                    SampledWavelengths::SampleImportance(RadicalInverse(0, i));
                Float h = Clamp(-1 + 2. * RadicalInverse(1, i), -.999999, .999999);

                SampledSpectrum sigma_a(0.f);
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);

                Float uc = RadicalInverse(2, i);
                Point2f u(RadicalInverse(3, i), RadicalInverse(4, i));

                pstd::optional<BSDFSample> bs = hair.Sample_f(
                    wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
                if (bs && bs->pdf > 0) {
                    SampledSpectrum f = bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                    ySum += f.y(lambda);
                }
            }

            Float avg = ySum / count;
            EXPECT_TRUE(avg >= .99 && avg <= 1.01) << avg;
        }
    }
}

TEST(Hair, SamplingWeights) {
    RNG rng;
    SampledWavelengths lambda = SampledWavelengths::SampleImportance(0.5);
    for (Float beta_m = .1; beta_m < 1; beta_m += .2)
        for (Float beta_n = .4; beta_n < 1; beta_n += .2) {
            int count = 10000;
            for (int i = 0; i < count; ++i) {
                Float h = Clamp(-1 + 2. * RadicalInverse(0, i), -.999999, .999999);

                // Check _HairBxDF::Sample\_f()_ sample weight
                SampledSpectrum sigma_a(0.);
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);

                Vector3f wo =
                    SampleUniformSphere({RadicalInverse(1, i), RadicalInverse(2, i)});
                Float uc = RadicalInverse(3, i);
                Point2f u = {RadicalInverse(4, i), RadicalInverse(5, i)};
                pstd::optional<BSDFSample> bs = hair.Sample_f(
                    wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
                if (bs && bs->pdf > 0) {
                    Float sum = 0;
                    int ny = 20;
                    for (Float u : Stratified1D(ny)) {
                        SampledWavelengths lambda =
                            SampledWavelengths::SampleImportance(u);
                        sum += bs->f.y(lambda) * AbsCosTheta(bs->wi) / bs->pdf;
                    }

                    // Verify that hair BSDF sample weight is close to 1 for
                    // _wi_
                    Float avg = sum / ny;
                    EXPECT_GT(avg, 0.99);
                    EXPECT_LT(avg, 1.01);
                }
            }
        }
}

TEST(Hair, SamplingConsistency) {
    RNG rng;
    SampledWavelengths lambda = SampledWavelengths::SampleImportance(0.5);
    for (Float beta_m = .2; beta_m < 1; beta_m += .2)
        for (Float beta_n = .4; beta_n < 1; beta_n += .2) {
            // Declare variables for hair sampling test
            const int count = 64 * 1024;
            SampledSpectrum sigma_a(.25);
            Vector3f wo =
                SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            auto Li = [](const Vector3f& w) { return SampledSpectrum(w.z * w.z); };
            SampledSpectrum fImportance(0.), fUniform(0.);
            for (int i = 0; i < count; ++i) {
                // Compute estimates of scattered radiance for hair sampling
                // test
                Float h = -1 + 2 * rng.Uniform<Float>();
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);
                Vector3f wi;
                Float pdf;
                Float uc = rng.Uniform<Float>();
                Point2f u = {rng.Uniform<Float>(), rng.Uniform<Float>()};
                pstd::optional<BSDFSample> bs = hair.Sample_f(
                    wo, uc, u, TransportMode::Radiance, BxDFReflTransFlags::All);
                if (bs && bs->pdf > 0)
                    fImportance +=
                        bs->f * Li(bs->wi) * AbsCosTheta(bs->wi) / (count * bs->pdf);
                wi = SampleUniformSphere(u);
                fUniform += hair.f(wo, wi, TransportMode::Radiance) * Li(wi) *
                            AbsCosTheta(wi) / (count * UniformSpherePDF());
            }
            // Verify consistency of estimated hair reflected radiance values
            Float err =
                std::abs(fImportance.y(lambda) - fUniform.y(lambda)) / fUniform.y(lambda);
            EXPECT_LT(err, 0.05);
        }
}
