
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/api.h>
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

/*
Refl 0.5

Layers.DirHemiSimple, 1 bounce
eta 1.0001: 0.4999 (1.4% TIR)
eta 1.5: 0.194 (75% TIR)
eta 2: 0.093 (86.6% TIR)

Layers.DirHemi, eval:


Layers.DirHemi, sample, 1 bounce, hack to ignore specular:



path integrator:
    // should be for refl 0.5: eta 1.5 -> .357, eta 2 -> .324
    // not including one-bounce specular reflection off the top: 1.5 -> .32, 2 -> .215

*/

TEST(Layers, DirHemiSimple) {
    Vector3f wo = Normalize(Vector3f(0, 0, 1)); // 1,.3,1));
    float eta = 3;
    float refl = 0.5;

    DielectricInterfaceBxDF top(eta, nullptr, TransportMode::Radiance);
    LambertianBxDF bottom(SampledSpectrum(refl), SampledSpectrum(0), 0);
    LayeredBxDFConfig config;
    config.maxDepth = 10;
    GeneralLayeredBxDF lb(&top, &bottom, 0. /* thickness */,
                          SampledSpectrum(0) /* albedo */,
                          0 /* g */, config);

    double sum = 0, sum2 = 0, sumx = 0;
    int ntir = 0, n2tir = 0;
    // one bounce, ray from (0,0,1)
    int n = 8*1024*1024;

    Float fo = FrDielectric(wo.z, eta);
    RNG rng;
    for (int i = 0; i < n; ++i) {
        // Sample wi hemisphere at bottom
        Vector3f w = SampleUniformHemisphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
        Float pdf = 1.f / (2 * Pi);
        Vector3f wt;
        if (Refract(-w, Normal3f(0,0,-1), eta, &wt)) {
            Float f = FrDielectric(-w.z, eta);
            //if (f == 1) { fprintf(stderr, "meh\n"); }
            sum += (1 - fo) * (1 - f) * (refl / Pi) * AbsCosTheta(wt) / pdf;
        } else {
            ++ntir;
        }

        // Sample wi at top layer
        if (Refract(w, Normal3f(0, 0, 1), 1 / eta, &wt)) {
            Float f = FrDielectric(w.z, eta);
            sum2 += (1 - fo) * (1 - f) * (refl / Pi) * AbsCosTheta(wt) / pdf;
        } else
            ++n2tir;

        // Guo probabilistic
        sumx += lb.f(wo, w)[0] * AbsCosTheta(w) / pdf;
    }
    printf("simple: %f, vs refl %f -> ratio %f. %.2f%% TIR\n", sum / n, refl,
           (sum / n) / refl, 100. * ntir / n);
    printf("simple2: %f, vs refl %f -> ratio %f. %.2f%% TIR\n", sum2 / n, refl,
           (sum2 / n) / refl, 100. * n2tir / n);
    printf("guo f: %f, vs refl %f -> ratio %f\n", sumx / n, refl,
           (sumx / n) / refl);



    // hemispherical-directional reflectance...
    Float fu = 0, fi = 0, fp = 0;
    int ns = 1024*1024;
    for (Point3f u : Uniform3D(ns)) {
        Vector3f wi = SampleCosineHemisphere({u[0], u[1]});
        fu += lb.f(wo, wi)[0] * AbsCosTheta(wi) / CosineHemispherePDF(wi.z);

        auto bs = lb.Sample_f(wo, u[0], {u[1], u[2]}, BxDFReflTransFlags::All);
        if (bs && bs->pdf > 0 && bs->wi.z != 1)  // hack to skip immediate specular refl
            fi += bs->f[0] * AbsCosTheta(bs->wi) / bs->pdf;

        // use PDF
        if (bs && bs->wi.z != 1) { // specular avoid hack
            Float pdf = lb.PDF(wo, bs->wi, BxDFReflTransFlags::All);
            if (pdf > 0)
                fp += bs->f[0] * AbsCosTheta(bs->wi) / pdf;
        }
    }
    // Note imp should be higher due to the top interface...
    fprintf(stderr, "guo f: %f sample_f: %f sample_f/PDF(): %f\n", fu/ns, fi/ns, fp/ns);

}

TEST(Layers, Viz) {
    Vector3f wo = Normalize(Vector3f(0, 0, 1)); // 1,.3,1));
    Float eta = 2;
    DielectricInterfaceBxDF top(eta, nullptr, TransportMode::Radiance);
    LambertianBxDF bottom(SampledSpectrum(0.5), SampledSpectrum(0), 0);
    LayeredBxDFConfig config;
    GeneralLayeredBxDF lb(&top, &bottom, 0. /* thickness */,
                          SampledSpectrum(0) /* albedo */,
                          0 /* g */, config);

    int res = 256;
    Image f(PixelFormat::Float, {res, res}, {"Y"});
    Image samplef(PixelFormat::Float, {res, res}, {"Y"});
    Image samplefPdf(PixelFormat::Float, {res, res}, {"Y"});
    Image samplefCount(PixelFormat::Float, {res, res}, {"Y"});
    Image pdf(PixelFormat::Float, {res, res}, {"Y"});

#if 0
    int os = 64;
    ParallelFor(0, res, [&](int y) {
        for (int x = 0; x < res; ++x) {
            for (Point2f u : Hammersley2D(os)) {
                Point2f p((x + u.x) / res, (y + u.y) / res);
                Vector3f wi = EquiAreaSquareToSphere(p);

                Float fv = lb.f(wo, wi)[0] * AbsCosTheta(wi) / os;
                f.SetChannel({x, y}, 0, f.GetChannel({x, y}, 0) + fv);

                Float pv = lb.PDF(wo, wi, BxDFReflTransFlags::All) / os;
                pdf.SetChannel({x, y}, 0, pdf.GetChannel({x, y}, 0) + pv);
            }
        }
    });
#endif

    std::vector<int> fCount(res*res, 0);
    for (Point3f u : Hammersley3D(100000000)) {
        Vector3f wi;
        auto bs = lb.Sample_f(wo, u[0], {u[1], u[2]}, BxDFReflTransFlags::All);
        if (!bs || bs->pdf == 0)
            continue;

        Point2f p = EquiAreaSphereToSquare(bs->wi);
        Point2i pi(Clamp(p.x * res, 0, res - 1),
                   Clamp(p.y * res, 0, res - 1));

        samplef.SetChannel(pi, 0, samplef.GetChannel(pi, 0) + bs->f[0] * AbsCosTheta(bs->wi));
        ++fCount[pi.x + pi.y * res];

        samplefPdf.SetChannel(pi, 0, samplefPdf.GetChannel(pi, 0) + bs->pdf);
    }

    for (int y = 0; y < res; ++y)
        for (int x = 0; x < res; ++x) {
            int c = fCount[x + y * res];
            if (c == 0) continue;
            samplef.SetChannel({x, y}, 0, samplef.GetChannel({x, y}, 0) / c);
            samplefPdf.SetChannel({x, y}, 0, samplefPdf.GetChannel({x, y}, 0) / c);
            samplefCount.SetChannel({x, y}, 0, c);
        }

    f.Write("f.exr");
    samplef.Write("samplef.exr");
    samplefPdf.Write("samplef-pdf.exr");
    samplefCount.Write("samplef-count.exr");
    pdf.Write("pdf.exr");
}

#if 0
TEST(RoughDielectric, Viz) {
    Vector3f wo = Normalize(Vector3f(1,.3,1));
    TrowbridgeReitzDistribution rough(.5, .5);
    Float eta = 1.5;
    DielectricInterface di(eta, &rough, TransportMode::Radiance);

    int res = 512;
    Image f(PixelFormat::Float, {res, res}, {"Y"});
    Image samplef(PixelFormat::Float, {res, res}, {"Y"});
    Image sampledist(PixelFormat::Float, {res, res}, {"Y"});
    Image pdf(PixelFormat::Float, {res, res}, {"Y"});

    for (int y = 0; y < res; ++y)
        for (int x = 0; x < res; ++x) {
            Point2f p((x + 0.5) / res, (y + 0.5) / res);
            Vector3f wi = EquiAreaSquareToSphere(p);
            f.SetChannel({x, y}, 0, di.f(wo, wi)[0] * std::abs(wi.z));
            pdf.SetChannel({x, y}, 0, di.PDF(wo, wi, BxDFReflTransFlags::All));
        }

    for (Point3f u : Hammersley3D(1000000)) {
        Vector3f wi;
        auto bs = di.Sample_f(wo, u[0], {u[1], u[2]}, BxDFReflTransFlags::All);
        if (!bs || bs->pdf == 0)
            continue;
        Point2f p = EquiAreaSphereToSquare(bs->wi);
        Point2i pi(Clamp(p.x * res, 0, res - 1),Clamp(p.y * res, 0, res - 1));
        samplef.SetChannel(pi, 0, bs->f[0]);
        sampledist.SetChannel(pi, 0, sampledist.GetChannel(pi, 0) + 1e-5);
    }

    f.Write("f.exr");
    samplef.Write("samplef.exr");
    sampledist.Write("sampledist.exr");
    pdf.Write("pdf.exr");
}

TEST(RoughDielectric, Interface) {
    TrowbridgeReitzDistribution rough(1, 1);
    Float eta = 1.5;
    DielectricInterface di(eta, &rough, TransportMode::Radiance);

    RNG rng;

    /*
eval wh [ 0.021775303, -0.9268676, -0.37475643 ], mf wh pdf 0.304875
best sampled wh [ 0.021622226, -0.92727023, -0.37376785 ], dot 0.999999, u [ 0.47623, 0.65802383 ]
No can sample. Max dot = 0.881171 with wi [ -0.4703933, -0.8818077, -0.033843935 ]
    */
    {
        Vector3f wo(0.28132406, -0.5946933, -0.7531246), wi(-0.38194194, -0.8124417, 0.44052106);
        LOG(WARNING) << "f " << di.f(wo, wi);  //         f 0.361925
        EXPECT_GT(Dot(evalWh, Vector3f(0.021622226, -0.92727023, -0.37376785)), .999) << evalWh;
        auto bs = di.Sample_f(wo, 0.9999, Point2f(0.47623, 0.65802383), BxDFReflTransFlags::All);
        LOG(WARNING) << "bs->wi " << bs->wi << " vs wi " << wi << ", f = " << bs->f;
    }
    {
        Vector3f wo( -0.3891811, 0.0051056407, 0.9211471), wi( -0.7639788, 0.6237078, -0.16530311);
        LOG(WARNING) << "f " << di.f(wo, wi);  //         f 0.0361824
        EXPECT_GT(Dot(evalWh, Vector3f(-0.79865545, 0.48937848, 0.35022575)), .999);
        auto bs = di.Sample_f(wo, 0.9999, Point2f(0.67361665, 0.6510124), BxDFReflTransFlags::All);
        LOG(WARNING) << "bs->wi " << bs->wi << " vs wi " << wi << ", f = " << bs->f;
        return;
    }
    {
        Vector3f wo( 0.8310706, 0.37243307, 0.41305608), wi( -0.53010356, -0.3259088, -0.78279865);
        LOG(WARNING) << "f " << di.f(wo, wi);  //         f 0.118993
        //EXPECT_GT(Dot(evalWh, Vector3f(-0.046592813, 0.1510447, 0.98742825)), .999) << evalWh;
        auto bs = di.Sample_f(wo, 0.9999, Point2f(0.74932, 0.22056198), BxDFReflTransFlags::All);
        LOG(WARNING) << "bs->wi " << bs->wi << " vs wi " << wi << ", f = " << bs->f;
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
            auto bs = di.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
            if (!bs || !bs->pdf)
                continue;

            SampledSpectrum f = di.f(wo, bs->wi);
            if (err(f[0], bs->f[0]) > .001) {
                ++fMismatches;
                LOG(WARNING) << "MISMATCH: f " << f[0] << " vs " << bs->f[0];
            }
        }

        // Given a random direction with f non-zero, can we sample it?
        for (int j = 0; j < 10; ++j) {
            Vector3f wi = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            SampledSpectrum f = di.f(wo, wi);
            if (!f) continue;

            LOG(WARNING) << "----------";
            LOG(WARNING) << "wo " << wo << ", wi " << wi << ", f " << f[0];

            Vector3f wh = evalWh;
            LOG(WARNING) << "eval wh " << wh << ", mf wh pdf " << rough.PDF(wo, wh);

            Float maxDot = -1;
            Vector3f bestWh;
            Point2f bestU;
            for (Point2f u : Hammersley2D(300000)) {
                Vector3f ww = rough.Sample_wh(wo, u);
                if (Dot(ww, wh) > maxDot) {
                    maxDot = Dot(ww, wh);
                    bestWh = ww;
                    bestU = u;
                }
            }
            LOG(WARNING) << "best sampled wh " << bestWh << ", dot " << maxDot << ", u " << bestU;
            if (maxDot < .95) LOG(WARNING) << "MEH!";

            maxDot = -1;
            Vector3f bestWi;
            for (Point2f u : Hammersley2D(30000)) {
                Float uc = SameHemisphere(wo, wi) > 0 ? 0. : 0.999999;
                auto bs = di.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
                if (bs && bs->pdf && bs->f) {
                    Float d = Dot(wi, bs->wi);
                    if (d > maxDot) {
                        maxDot = d;
                        bestWi = bs->wi;
                    }
                }
            }
            if (maxDot < .95) {
                LOG(WARNING) << "No can sample. Max dot = " << maxDot << " with wi " << bestWi;
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
        SampledSpectrum f = di.f(wo, wi);
//CO        if (!f) LOG(WARNING) << i << ": f is black for " << wo << ", " << wi;
        if (!f) continue;
        ++nvalid;

        // How close does sampling get?
        Float maxDot = -1;
        for (int j = 0; j < 100000; ++j) {
            Float uc = rng.Uniform<Float>();
            Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
            auto bs = di.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
            if (bs && bs->pdf && bs->f)
                maxDot = std::max(maxDot, Dot(wi, bs->wi));
        }
        if (maxDot < .9) {
            LOG(WARNING) << maxDot << " max dot for wo " << wo << ", wi " << wi << ", f " << f <<
                (SameHemisphere(wo, wi) ? ", REFL " : ", TRANS");

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
            LOG(WARNING) << "Exhaustive got to " << maxDot << " with wm " << bestWm <<
                " which has pdf " << rough.PDF(wo, bestWm);
            LOG(WARNING) << "wm . wo = " << Dot(wo, bestWm);

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
                Vector3f wm = rough.Sample_wh(wo, {rng.Uniform<Float>(), rng.Uniform<Float>()});
                ASSERT_GE(wm.z, 0);
                Vector3f wip;
                Float ep = wo.z > 0 ? (1 / eta) : eta;
                bool tir = !Refract(wo, (Normal3f)wm, ep, &wip);
                if (!tir && Dot(wi, wip) > maxDot) {
                    maxDot = Dot(wi, wip);
                    bestWm = wm;
                }
            }
            LOG(WARNING) << "TR distrib got to " << maxDot << " with wm " << bestWm <<
                " which has pdf " << rough.PDF(wo, bestWm);
            LOG(WARNING) << "Best . wo " << Dot(wo, bestWm);

            LOG(WARNING) << "Redo " << di.f(wo, wi);
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

            auto bs = di.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
            if (!bs || bs->pdf == 0 || !bs->f) {
//CO                LOG(WARNING) << "Sample failed";
                continue;
            }

//CO            LOG(WARNING) << Dot(wo, bs->wi) << " for wi " << bs->wi;
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

TEST(LayeredBxDF, Samplingz) {
    MemoryArena arena;

    BxDFHandle bottom = arena.Alloc<LambertianBxDF>(SampledSpectrum(0.8), SampledSpectrum(0), 0);
    MicrofacetDistributionHandle distrib = arena.Alloc<TrowbridgeReitzDistribution>(0.1, 0.1);
    Float eta = 1.1;
    BxDFHandle top = arena.Alloc<DielectricInterfaceBxDF>(eta, distrib, TransportMode::Radiance);
    LayeredBxDFConfig config;
    GeneralLayeredBxDF *layered =
        arena.Alloc<GeneralLayeredBxDF>(top, bottom, 0 /* thickness */,
                                        SampledSpectrum(0) /* albedo */,
                                        0 /* g */,
                                        config);

    Vector3f wo = Normalize(Vector3f(-1, 0, .25));
    Vector3f wi = Normalize(Vector3f(1, 0, .1));

    LOG_VERBOSE("top->f %s", top.f(wo, wi));
    double fSum = 0;
    int n = 1*1024*1024;
    for (int i = 0; i < n; ++i)
        fSum += (layered->f(wo, wi) - top.f(wo, wi)).Average();
    LOG_VERBOSE("f = %f, vs bottom %f, FT wo %f, FT wi %f --> %f ",
                fSum / n, bottom.f(wo, wi).Average(),
                (1 - FrDielectric(CosTheta(wo), eta)),
                (1 - FrDielectric(CosTheta(wi), eta)),
                (bottom.f(wo, wi).Average() *
                 (1 - FrDielectric(CosTheta(wo), eta)) *
                 (1 - FrDielectric(CosTheta(wi), eta))));
}

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

    if (x == 0) return 0.0f;

    double ax = (a * std::log(x)) - x - std::lgamma(a);
    if (ax < -709.78271289338399) return a < x ? 1.0 : 0.0;

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
    std::function<Float(Float, Float, Float, Float, Float, Float, Float, Float,
                        int)> integrate = [&](Float a, Float b, Float c,
                                              Float fa, Float fb, Float fc,
                                              Float I, Float eps, int depth) {
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
Float AdaptiveSimpson2D(const std::function<Float(Float, Float)>& f, Float x0,
                        Float y0, Float x1, Float y1, Float eps = 1e-6f,
                        int depth = 6) {
    /* Lambda function that integrates over the X axis */
    auto integrate = [&](Float y) {
        return AdaptiveSimpson(std::bind(f, std::placeholders::_1, y), x0, x1,
                               eps, depth);
    };
    Float value = AdaptiveSimpson(integrate, y0, y1, eps, depth);
    return value;
}

/// Generate a histogram of the BSDF density function via MC sampling
void FrequencyTable(const BSDF* bsdf, const Vector3f& wo, RNG& rng,
                    int sampleCount, int thetaRes, int phiRes, Float* target) {
    memset(target, 0, thetaRes * phiRes * sizeof(Float));

    Float factorTheta = thetaRes / Pi, factorPhi = phiRes / (2 * Pi);

    BxDFFlags sampledType;
    Vector3f wi;
    Float pdf;

    for (int i = 0; i < sampleCount; ++i) {
        Float u = rng.Uniform<Float>();
        Point2f sample {rng.Uniform<Float>(), rng.Uniform<Float>()};
        pstd::optional<BSDFSample> bs = bsdf->Sample_f(wo, u, sample);

        if (!bs || bs->IsSpecular()) continue;

        Vector3f wiL = bsdf->WorldToLocal(bs->wi);

        Point2f coords(SafeACos(wiL.z) * factorTheta,
                       std::atan2(wiL.y, wiL.x) * factorPhi);

        if (coords.y < 0) coords.y += 2 * Pi * factorPhi;

        int thetaBin =
            std::min(std::max(0, (int)std::floor(coords.x)), thetaRes - 1);
        int phiBin =
            std::min(std::max(0, (int)std::floor(coords.y)), phiRes - 1);

        target[thetaBin * phiRes + phiBin] += 1;
    }
}

// Numerically integrate the probability density function over rectangles in
// spherical coordinates.
void IntegrateFrequencyTable(const BSDF* bsdf, const Vector3f& wo,
                             int sampleCount, int thetaRes, int phiRes,
                             Float* target) {
    memset(target, 0, thetaRes * phiRes * sizeof(Float));

    Float factorTheta = Pi / thetaRes, factorPhi = (2 * Pi) / phiRes;

    for (int i = 0; i < thetaRes; ++i) {
        for (int j = 0; j < phiRes; ++j) {
            *target++ = sampleCount *
                        AdaptiveSimpson2D(
                            [&](Float theta, Float phi) -> Float {
                                Float cosTheta = std::cos(theta),
                                      sinTheta = std::sin(theta);
                                Float cosPhi = std::cos(phi),
                                      sinPhi = std::sin(phi);
                                Vector3f wiL(sinTheta * cosPhi,
                                             sinTheta * sinPhi, cosTheta);
                                return bsdf->PDF(wo, bsdf->LocalToWorld(wiL)) *
                                       sinTheta;
                            },
                            i* factorTheta, j* factorPhi, (i + 1) * factorTheta,
                            (j + 1) * factorPhi);
        }
    }
}

/// Write the frequency tables to disk in a format that is nicely plottable by
/// Octave and MATLAB
void DumpTables(const Float* frequencies, const Float* expFrequencies,
                int thetaRes, int phiRes, const char* filename) {
    std::ofstream f(filename);

    f << "frequencies = [ ";
    for (int i = 0; i < thetaRes; ++i) {
        for (int j = 0; j < phiRes; ++j) {
            f << frequencies[i * phiRes + j];
            if (j + 1 < phiRes) f << ", ";
        }
        if (i + 1 < thetaRes) f << "; ";
    }
    f << " ];" << std::endl << "expFrequencies = [ ";
    for (int i = 0; i < thetaRes; ++i) {
        for (int j = 0; j < phiRes; ++j) {
            f << expFrequencies[i * phiRes + j];
            if (j + 1 < phiRes) f << ", ";
        }
        if (i + 1 < thetaRes) f << "; ";
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
                                      int phiRes, int sampleCount,
                                      Float minExpFrequency,
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

                std::string result = StringPrintf(
                        "Encountered %f samples in a c with expected "
                        "frequency 0. Rejecting the null hypothesis!",
                        frequencies[c.index]);
                return std::make_pair(false, result);
            }
        } else if (expFrequencies[c.index] < minExpFrequency) {
            /* Pool cells with low expected frequencies */
            pooledFrequencies += frequencies[c.index];
            pooledExpFrequencies += expFrequencies[c.index];
            pooledCells++;
        } else if (pooledExpFrequencies > 0 &&
                   pooledExpFrequencies < minExpFrequency) {
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
        std::string result = StringPrintf(
              "The number of degrees of freedom %d is too low!", dof);
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
      std::string result = StringPrintf(
                "Rejected the null hypothesis (p-value = %f, "
                "significance level = %f",
                pval, alpha);
        return std::make_pair(false, result);
    } else {
        return std::make_pair(true, std::string(""));
    }
}

void TestBSDF(std::function<BSDF *(const SurfaceInteraction &, MemoryArena&)> createBSDF,
              const char* description) {
    MemoryArena arena;

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

        std::shared_ptr<Disk> disk =
            std::make_shared<Disk>(t.get(), tInv.get(), reverseOrientation, 0., 1., 0, 360.);
        Point3f origin(0.1, 1,
                       0);  // offset slightly so we don't hit center of disk
        Vector3f direction(0, -1, 0);
        Ray r(origin, direction);
        auto si = disk->Intersect(r);
        ASSERT_TRUE(si.has_value());
        bsdf = createBSDF(si->intr, arena);
    }

    for (int k = 0; k < CHI2_RUNS; ++k) {
        /* Randomly pick an outgoing direction on the hemisphere */
        Point2f sample {rng.Uniform<Float>(), rng.Uniform<Float>()};
        Vector3f woL = SampleCosineHemisphere(sample);
        Vector3f wo = bsdf->LocalToWorld(woL);

        FrequencyTable(bsdf, wo, rng, sampleCount, thetaRes, phiRes,
                       frequencies);

        IntegrateFrequencyTable(bsdf, wo, sampleCount, thetaRes, phiRes,
                                expFrequencies);

        std::string filename = StringPrintf("/tmp/chi2test_%s_%03i.m",
                                            description, ++index);
        DumpTables(frequencies, expFrequencies, thetaRes, phiRes,
                   filename.c_str());

        auto result =
            Chi2Test(frequencies, expFrequencies, thetaRes, phiRes, sampleCount,
                     CHI2_MINFREQ, CHI2_SLEVEL, CHI2_RUNS);
        EXPECT_TRUE(result.first) << result.second << ", iteration " << k;
    }

    delete[] frequencies;
    delete[] expFrequencies;
}

BSDF *createLambertian(const SurfaceInteraction &si, MemoryArena &arena) {
    SampledSpectrum Kd(1.);
    return arena.Alloc<BSDF>(si, arena.Alloc<LambertianBxDF>(Kd, SampledSpectrum(0.), 0));
}

BSDF *createMicrofacet(const SurfaceInteraction &si, MemoryArena& arena,
                       float roughx, float roughy) {
    Float alphax = TrowbridgeReitzDistribution::RoughnessToAlpha(roughx);
    Float alphay = TrowbridgeReitzDistribution::RoughnessToAlpha(roughy);
    MicrofacetDistributionHandle distrib = arena.Alloc<TrowbridgeReitzDistribution>(alphax, alphay);
    FresnelHandle fresnel = arena.Alloc<FresnelDielectric>(1.5, true);
    return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
//CO    return arena.Alloc<BSDF>(si, arena.Alloc<DielectricInterface>(1.5, distrib, TransportMode::Radiance));
}

TEST(BSDFSampling, Lambertian) { TestBSDF(createLambertian, "Lambertian"); }

TEST(BSDFSampling, TR_VA_0p5) {
    TestBSDF([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        return createMicrofacet(si, arena, 0.5, 0.5);
    }, "Trowbridge-Reitz, visible area sample, alpha = 0.5");
}

TEST(BSDFSampling, TR_VA_0p3_0p15) {
    TestBSDF([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        return createMicrofacet(si, arena, 0.3, 0.15);
    }, "Trowbridge-Reitz, visible area sample, alpha = 0.3/0.15");
}

///////////////////////////////////////////////////////////////////////////
// Energy Conservation Tests

static void TestEnergyConservation(std::function<BSDF *(const SurfaceInteraction &, MemoryArena&)> createBSDF,
                                   const char* description) {
    MemoryArena arena;
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
    BSDF *bsdf = createBSDF(si->intr, arena);

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

        EXPECT_LT(Lo.MaxComponentValue(), 1.01) << description << ": Lo = " << Lo << ", wo = " << wo;
    }
}

TEST(BSDFEnergyConservation, LambertianReflection) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        return arena.Alloc<BSDF>(si, arena.Alloc<LambertianBxDF>(SampledSpectrum(1.f), SampledSpectrum(0.), 0));
        }, "LambertianReflection");
}

TEST(BSDFEnergyConservation, OrenNayar) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        return arena.Alloc<BSDF>(si, arena.Alloc<LambertianBxDF>(SampledSpectrum(1.f), SampledSpectrum(0.), 20));
        }, "Oren-Nayar sigma 20");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_1_dielectric1_5) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        FresnelHandle fresnel = arena.Alloc<FresnelDielectric>(1.f, 1.5f);
        MicrofacetDistributionHandle distrib =
            arena.Alloc<TrowbridgeReitzDistribution>(0.1, 0.1);
        return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
        }, "MicrofacetReflectionBxDF, Fresnel dielectric, TrowbridgeReitz alpha 0.1");
}


TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha1_5_dielectric1_5) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        FresnelHandle fresnel = arena.Alloc<FresnelDielectric>(1.f, 1.5f);
        MicrofacetDistributionHandle distrib =
            arena.Alloc<TrowbridgeReitzDistribution>(1.5, 1.5);
        return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
        }, "MicrofacetReflectionBxDF, Fresnel dielectric, TrowbridgeReitz alpha 1.5");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_01_dielectric1_5) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        FresnelHandle fresnel = arena.Alloc<FresnelDielectric>(1.f, 1.5f);
        MicrofacetDistributionHandle distrib =
            arena.Alloc<TrowbridgeReitzDistribution>(0.01, 0.01);
        return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
        }, "MicrofacetReflectionBxDF, Fresnel dielectric, TrowbridgeReitz alpha 0.01");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_1_conductor) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
        SampledSpectrum etaT = SPDs::MetalAlEta().Sample(lambda);
        SampledSpectrum K = SPDs::MetalAlK().Sample(lambda);
        FresnelHandle fresnel = arena.Alloc<FresnelConductor>(etaT, K);
        MicrofacetDistributionHandle distrib =
            arena.Alloc<TrowbridgeReitzDistribution>(0.1, 0.1);
        return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
        }, "MicrofacetReflectionBxDF, Fresnel conductor, TrowbridgeReitz alpha 0.1");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha1_5_conductor) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
        SampledSpectrum etaT = SPDs::MetalAlEta().Sample(lambda);
        SampledSpectrum K = SPDs::MetalAlK().Sample(lambda);
        FresnelHandle fresnel = arena.Alloc<FresnelConductor>(etaT, K);
        MicrofacetDistributionHandle distrib =
            arena.Alloc<TrowbridgeReitzDistribution>(1.5, 1.5);
        return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
    }, "MicrofacetReflectionBxDF, Fresnel conductor, TrowbridgeReitz alpha 1.5");
}

TEST(BSDFEnergyConservation, MicrofacetReflectionBxDFTrowbridgeReitz_alpha0_01_conductor) {
    TestEnergyConservation([](const SurfaceInteraction &si, MemoryArena& arena) -> BSDF * {
        SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
        SampledSpectrum etaT = SPDs::MetalAlEta().Sample(lambda);
        SampledSpectrum K = SPDs::MetalAlK().Sample(lambda);
        FresnelHandle fresnel = arena.Alloc<FresnelConductor>(etaT, K);

        MicrofacetDistributionHandle distrib =
            arena.Alloc<TrowbridgeReitzDistribution>(0.01, 0.01);
        return arena.Alloc<BSDF>(si, arena.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel));
    }, "MicrofacetReflectionBxDF, Fresnel conductor, TrowbridgeReitz alpha 0.01");
}
