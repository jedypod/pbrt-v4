
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// tests/hair.cpp*
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/materials.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/rng.h>

using namespace pbrt;

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
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    for (Float beta_m = .1; beta_m < 1; beta_m += .2) {
        for (Float beta_n = .1; beta_n < 1; beta_n += .2) {
            // Estimate reflected uniform incident radiance from hair
            SampledSpectrum sum(0.f);
            int count = 300000;
            for (int i = 0; i < count; ++i) {
                Float h = -1 + 2. * rng.Uniform<Float>();
                SampledSpectrum sigma_a(0.f);
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);
                Vector3f wi = SampleUniformSphere(
                    {rng.Uniform<Float>(), rng.Uniform<Float>()});
                sum += hair.f(wo, wi) * AbsCosTheta(wi);
            }
            Float avg = sum.y(lambda) / (count * UniformSpherePDF());
            EXPECT_TRUE(avg >= .95 && avg <= 1.05);
        }
    }
}

TEST(Hair, WhiteFurnaceSampled) {
    RNG rng;
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    Vector3f wo = SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
    for (Float beta_m = .1; beta_m < 1; beta_m += .2) {
        for (Float beta_n = .1; beta_n < 1; beta_n += .2) {
            SampledSpectrum sum(0.f);
            int count = 300000;
            for (int i = 0; i < count; ++i) {
                Float h = -1 + 2. * rng.Uniform<Float>();
                SampledSpectrum sigma_a(0.f);
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);

                Vector3f wi;
                Float pdf;
                Float uc = rng.Uniform<Float>();
                Point2f u = {rng.Uniform<Float>(), rng.Uniform<Float>()};
                pstd::optional<BSDFSample> bs = hair.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
                if (bs && bs->pdf > 0) sum += bs->f * AbsCosTheta(bs->wi) / bs->pdf;
            }
            Float avg = sum.y(lambda) / count;
            EXPECT_TRUE(avg >= .99 && avg <= 1.01) << avg;
        }
    }
}

TEST(Hair, SamplingWeights) {
    RNG rng;
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    for (Float beta_m = .1; beta_m < 1; beta_m += .2)
        for (Float beta_n = .4; beta_n < 1; beta_n += .2) {
            int count = 10000;
            for (int i = 0; i < count; ++i) {
                // Check _HairBxDF::Sample\_f()_ sample weight
                Float h = -1 + 2 * rng.Uniform<Float>();
                SampledSpectrum sigma_a(0.);
                HairBxDF hair(h, 1.55, sigma_a, beta_m, beta_n, 0.f);
                Vector3f wo = SampleUniformSphere(
                    {rng.Uniform<Float>(), rng.Uniform<Float>()});
                Vector3f wi;
                Float pdf;
                Float uc = rng.Uniform<Float>();
                Point2f u = {rng.Uniform<Float>(), rng.Uniform<Float>()};
                pstd::optional<BSDFSample> bs = hair.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
                if (bs && bs->pdf > 0) {
                    // Verify that hair BSDF sample weight is close to 1 for
                    // _wi_
                    EXPECT_GT(bs->f.y(lambda) * AbsCosTheta(bs->wi) / bs->pdf, 0.99);
                    EXPECT_LT(bs->f.y(lambda) * AbsCosTheta(bs->wi) / bs->pdf, 1.01);
                }
            }
        }
}

TEST(Hair, SamplingConsistency) {
    RNG rng;
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    for (Float beta_m = .2; beta_m < 1; beta_m += .2)
        for (Float beta_n = .4; beta_n < 1; beta_n += .2) {
            // Declare variables for hair sampling test
            const int count = 64 * 1024;
            SampledSpectrum sigma_a(.25);
            Vector3f wo =
                SampleUniformSphere({rng.Uniform<Float>(), rng.Uniform<Float>()});
            auto Li = [](const Vector3f &w) { return SampledSpectrum(w.z * w.z); };
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
                pstd::optional<BSDFSample> bs = hair.Sample_f(wo, uc, u, BxDFReflTransFlags::All);
                if (bs && bs->pdf > 0)
                    fImportance += bs->f * Li(bs->wi) * AbsCosTheta(bs->wi) / (count * bs->pdf);
                wi = SampleUniformSphere(u);
                fUniform += hair.f(wo, wi) * Li(wi) * AbsCosTheta(wi) /
                            (count * UniformSpherePDF());
            }
            // Verify consistency of estimated hair reflected radiance values
            Float err = std::abs(fImportance.y(lambda) -
                                 fUniform.y(lambda)) /
                fUniform.y(lambda);
            EXPECT_LT(err, 0.05);
        }
}
