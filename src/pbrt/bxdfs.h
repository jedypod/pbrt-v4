// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_BXDF_H
#define PBRT_CORE_BXDF_H

// core/bxdf.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/interaction.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace pbrt {

class alignas(8) DiffuseBxDF {
  public:
    // Lambertian Public Methods
    PBRT_CPU_GPU
    DiffuseBxDF(const SampledSpectrum &R, const SampledSpectrum &T, Float sigma)
        : R(R), T(T) {
        sigma = Radians(sigma);
        Float sigma2 = sigma * sigma;
        A = 1 - sigma2 / (2 * (sigma2 + 0.33f));
        B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        if (B == 0)
            return SameHemisphere(wo, wi) ? (R * InvPi) : (T * InvPi);

        if ((SameHemisphere(wo, wi) && !R) || (!SameHemisphere(wo, wi) && !T))
            return SampledSpectrum(0.);

        Float sinTheta_i = SinTheta(wi), sinTheta_o = SinTheta(wo);
        // Compute cosine term of Oren-Nayar model
        Float maxCos = 0;
        if (sinTheta_i > 0 && sinTheta_o > 0)
            maxCos = std::max<Float>(0, CosDPhi(wi, wo));

        // Compute sine and tangent terms of Oren-Nayar model
        Float sinAlpha, tanBeta;
        if (AbsCosTheta(wi) > AbsCosTheta(wo)) {
            sinAlpha = sinTheta_o;
            tanBeta = sinTheta_i / AbsCosTheta(wi);
        } else {
            sinAlpha = sinTheta_i;
            tanBeta = sinTheta_o / AbsCosTheta(wo);
        }

        if (SameHemisphere(wo, wi))
            return R * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
        else
            return T * InvPi * (A + B * maxCos * sinAlpha * tanBeta);
    }
    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        const Vector3f &wo, Float uc, const Point2f &u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Float pr = R.MaxComponentValue(), pt = T.MaxComponentValue();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        Float cpdf;
        // TODO: rewrite to a single code path for the GPU. Good chance to
        // discuss divergence.
        if (SampleDiscrete({pr, pt}, uc, &cpdf) == 0) {
            Vector3f wi = SampleCosineHemisphere(u);
            if (wo.z < 0)
                wi.z *= -1;
            Float pdf = AbsCosTheta(wi) * InvPi * cpdf;
            return BSDFSample(f(wo, wi, mode), wi, pdf, BxDFFlags::DiffuseReflection);
        } else {
            Vector3f wi = SampleCosineHemisphere(u);
            if (wo.z > 0)
                wi.z *= -1;
            Float pdf = AbsCosTheta(wi) * InvPi * cpdf;
            return BSDFSample(f(wo, wi, mode), wi, pdf, BxDFFlags::DiffuseTransmission);
        }
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Float pr = R.MaxComponentValue(), pt = T.MaxComponentValue();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return 0;

        if (SameHemisphere(wo, wi))
            return pr / (pr + pt) * AbsCosTheta(wi) * InvPi;
        else
            return pt / (pr + pt) * AbsCosTheta(wi) * InvPi;
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) { return this; }

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return ((R ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset) |
                (T ? BxDFFlags::DiffuseTransmission : BxDFFlags::Unset));
    }

  private:
    // Lambertian Private Data
    SampledSpectrum R, T;
    Float A, B;
};

class alignas(8) DielectricInterfaceBxDF {
  public:
    PBRT_CPU_GPU
    DielectricInterfaceBxDF(Float eta, MicrofacetDistributionHandle distrib)
        : flags(BxDFFlags::Reflection | BxDFFlags::Transmission |
                BxDFFlags((!distrib || distrib.EffectivelySpecular())
                              ? BxDFFlags::Specular
                              : BxDFFlags::Glossy)),
          eta(eta == 1 ? 1.001 : eta),
          distrib(distrib) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        if (!distrib || distrib.EffectivelySpecular())
            return SampledSpectrum(0);

        if (SameHemisphere(wo, wi)) {
            // reflect
            Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
            Vector3f wh = wi + wo;
            // Handle degenerate cases for microfacet reflection
            if (cosTheta_i == 0 || cosTheta_o == 0)
                return SampledSpectrum(0.);
            if (wh.x == 0 && wh.y == 0 && wh.z == 0)
                return SampledSpectrum(0.);
            wh = Normalize(wh);
            Float F = FrDielectric(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))), eta);
            return SampledSpectrum(distrib.D(wh) * distrib.G(wo, wi) * F /
                                   (4 * cosTheta_i * cosTheta_o));
        } else {
            // transmit
            Float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
            if (cosTheta_i == 0 || cosTheta_o == 0)
                return SampledSpectrum(0.);

            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
            Vector3f wh = wo + wi * etap;
            CHECK_RARE(1e-6, LengthSquared(wh) == 0);
            if (LengthSquared(wh) == 0)
                return SampledSpectrum(0.);
            wh = FaceForward(Normalize(wh), Normal3f(0, 0, 1));

            // both on same side?
            // if (Dot(wi, wh) * Dot(wo, wh) > 0) return SampledSpectrum(0.);

            Float F = FrDielectric(Dot(wo, wh), eta);

            Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
            Float factor = (mode == TransportMode::Radiance) ? Sqr(1 / etap) : 1;

            return SampledSpectrum((1 - F) * factor *
                                   std::abs(distrib.D(wh) * distrib.G(wo, wi) *
                                            AbsDot(wi, wh) * AbsDot(wo, wh) /
                                            (cosTheta_i * cosTheta_o * Sqr(sqrtDenom))));
        }
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        const Vector3f &wo, Float uc, const Point2f &u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (wo.z == 0)
            return {};

        if (!distrib) {
            Float F = FrDielectric(CosTheta(wo), eta);

            Float pr = F, pt = 1 - F;
            if (!(sampleFlags & BxDFReflTransFlags::Reflection))
                pr = 0;
            if (!(sampleFlags & BxDFReflTransFlags::Transmission))
                pt = 0;
            if (pr == 0 && pt == 0)
                return {};

            if (uc < pr / (pr + pt)) {
                // reflect
                Vector3f wi(-wo.x, -wo.y, wo.z);
                SampledSpectrum fr(F / AbsCosTheta(wi));
                return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);
            } else {
                // transmit
                // Figure out which $\eta$ is incident and which is transmitted
                bool entering = CosTheta(wo) > 0;
                Float etap = entering ? eta : (1 / eta);

                // Compute ray direction for specular transmission
                Vector3f wi;
                bool tir = !Refract(wo, FaceForward(Normal3f(0, 0, 1), wo), etap, &wi);
                CHECK_RARE(1e-6, tir);
                if (tir)
                    return {};
                SampledSpectrum ft((1 - F) / AbsCosTheta(wi));

                // Account for non-symmetry with transmission to different
                // medium
                if (mode == TransportMode::Radiance)
                    ft /= Sqr(etap);
                return BSDFSample(ft, wi, pt / (pr + pt),
                                  BxDFFlags::SpecularTransmission);
            }
        } else {
            Float compPDF;
            Vector3f wh = distrib.Sample_wm(wo, u);
            Float F = FrDielectric(
                Dot(Reflect(wo, wh), FaceForward(wh, Vector3f(0, 0, 1))), eta);

            Float pr = F, pt = 1 - F;
            if (!(sampleFlags & BxDFReflTransFlags::Reflection))
                pr = 0;
            if (!(sampleFlags & BxDFReflTransFlags::Transmission))
                pt = 0;
            if (pr == 0 && pt == 0)
                return {};

            if (uc < pr / (pr + pt)) {
                // reflect
                // Sample microfacet orientation $\wh$ and reflected direction
                // $\wi$
                Vector3f wi = Reflect(wo, wh);
                CHECK_RARE(1e-6, Dot(wo, wh) <= 0);
                if (!SameHemisphere(wo, wi) || Dot(wo, wh) <= 0)
                    return {};

                // Compute PDF of _wi_ for microfacet reflection
                Float pdf = distrib.PDF(wo, wh) / (4 * Dot(wo, wh)) * pr / (pr + pt);
                CHECK(!std::isnan(pdf));

                // TODO: reuse fragments from f()
                Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
                // Handle degenerate cases for microfacet reflection
                if (cosTheta_i == 0 || cosTheta_o == 0)
                    return {};
                SampledSpectrum f(distrib.D(wh) * distrib.G(wo, wi) * F /
                                  (4 * cosTheta_i * cosTheta_o));
                if (distrib.EffectivelySpecular())
                    return BSDFSample(f / pdf, wi, 1, BxDFFlags::SpecularReflection);
                else
                    return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
            } else {
                // FIXME (make consistent): this etap is 1/etap as used in
                // specular...
                Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
                Vector3f wi;
                bool tir = !Refract(wo, (Normal3f)wh, etap, &wi);
                CHECK_RARE(1e-6, tir);
                if (SameHemisphere(wo, wi))
                    return {};
                if (tir || wi.z == 0)
                    return {};

                // Evaluate BSDF
                // TODO: share fragments with f(), PDF()...
                wh = FaceForward(wh, Normal3f(0, 0, 1));

                Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
                Float factor = (mode == TransportMode::Radiance) ? Sqr(1 / etap) : 1;

                SampledSpectrum f(
                    (1 - F) * factor *
                    std::abs(distrib.D(wh) * distrib.G(wo, wi) * AbsDot(wi, wh) *
                             AbsDot(wo, wh) /
                             (AbsCosTheta(wi) * AbsCosTheta(wo) * Sqr(sqrtDenom))));

                // Compute PDF
                Float dwh_dwi =
                    /*Sqr(etap) * */ AbsDot(wi, wh) /
                    Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
                Float pdf = distrib.PDF(wo, wh) * dwh_dwi * pt / (pr + pt);
                CHECK(!std::isnan(pdf));

                // CO            LOG(WARNING) << "pt/(pr+pt) " << pt / (pr +
                // pt); CO            LOG(WARNING) << "Sample_f: (1-F) " <<
                // (1-F)
                // << ", factor " << factor << CO                ", D " <<
                // distrib.D(wh) << ", G " << distrib.G(wo, wi) << CO ", others
                // "
                // << (AbsDot(wi, wh) * AbsDot(wo, wh) / CO (AbsCosTheta(wi) *
                // AbsCosTheta(wo) * Sqr(sqrtDenom))) << CO                ",
                // pdf " << pdf << ", f*cos/pdf " << f*AbsCosTheta(wi)/pdf;

                if (distrib.EffectivelySpecular())
                    return BSDFSample(f / pdf, wi, 1, BxDFFlags::SpecularTransmission);
                else
                    return BSDFSample(f, wi, pdf, BxDFFlags::GlossyTransmission);
            }
        }
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (!distrib || distrib.EffectivelySpecular())
            return 0;

        if (SameHemisphere(wo, wi)) {
            if (!(sampleFlags & BxDFReflTransFlags::Reflection))
                return 0;

            Vector3f wh = wo + wi;
            CHECK_RARE(1e-6, LengthSquared(wh) == 0);
            CHECK_RARE(1e-6, Dot(wo, wh) < 0);
            if (LengthSquared(wh) == 0 || Dot(wo, wh) <= 0)
                return 0;

            wh = Normalize(wh);

            Float F = FrDielectric(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))), eta);
            CHECK_RARE(1e-6, F == 0);
            Float pr = F, pt = 1 - F;
            if (!(sampleFlags & BxDFReflTransFlags::Transmission))
                pt = 0;

            return distrib.PDF(wo, wh) / (4 * Dot(wo, wh)) * pr / (pr + pt);
        } else {
            if (!(sampleFlags & BxDFReflTransFlags::Transmission))
                return 0;
            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
            Vector3f wh = wo + wi * etap;
            CHECK_RARE(1e-6, LengthSquared(wh) == 0);
            if (LengthSquared(wh) == 0)
                return 0;
            wh = Normalize(wh);

            // both on same side?
            // if (Dot(wi, wh) * Dot(wo, wh) > 0) return 0.;

            Float F = FrDielectric(Dot(wo, FaceForward(wh, Normal3f(0, 0, 1))), eta);
            Float pr = F, pt = 1 - F;
            if (pt == 0)
                return 0;
            if (!(sampleFlags & BxDFReflTransFlags::Reflection))
                pr = 0;

            // Compute change of variables _dwh\_dwi_ for microfacet
            // transmission
            Float dwh_dwi =
                /*Sqr(etap) * */ AbsDot(wi, wh) / Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
            CHECK_RARE(1e-6, (1 - F) == 0);
            return distrib.PDF(wo, wh) * dwh_dwi * pt / (pr + pt);
        }
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        MicrofacetDistributionHandle rd =
            distrib ? distrib.Regularize(scratchBuffer)
                    : scratchBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
        return scratchBuffer.Alloc<DielectricInterfaceBxDF>(eta, rd);
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return flags; }

  private:
    BxDFFlags flags;
    Float eta;
    MicrofacetDistributionHandle distrib;
};

class alignas(8) ThinDielectricBxDF {
  public:
    PBRT_CPU_GPU
    ThinDielectricBxDF(Float eta) : eta(eta) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        return SampledSpectrum(0);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const {
        Float R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
        if (R < 1) {
            // Note this goes to Stokes glass plates...
            // R + TRT + TRRRT + ...
            R += T * T * R / (1 - R * R);
            T = 1 - R;
        }

        Float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission))
            pt = 0;
        if (pr == 0 && pt == 0)
            return {};

        if (uc < pr / (pr + pt)) {
            // reflect
            Vector3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt), BxDFFlags::SpecularReflection);
        } else {
            // transmit
            // Figure out which $\eta$ is incident and which is transmitted
            bool entering = CosTheta(wo) > 0;
            Float etap = entering ? 1 / eta : eta;

            Vector3f wi = -wo;
            SampledSpectrum ft(T / AbsCosTheta(wi));

            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode::Radiance)
                ft *= etap * etap;
            return BSDFSample(ft, wi, pt / (pr + pt), BxDFFlags::SpecularTransmission);
        }
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        return 0;
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        MicrofacetDistributionHandle distrib =
            scratchBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
        return scratchBuffer.Alloc<DielectricInterfaceBxDF>(eta, distrib);
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const {
        return (BxDFFlags::Reflection | BxDFFlags::Transmission | BxDFFlags::Specular);
    }

  private:
    Float eta;
};

class alignas(8) MicrofacetReflectionBxDF {
  public:
    // MicrofacetReflection Public Methods
    PBRT_CPU_GPU
    MicrofacetReflectionBxDF(MicrofacetDistributionHandle distribution,
                             FresnelHandle fresnel)
        : distribution(distribution), fresnel(fresnel) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        if (!SameHemisphere(wo, wi))
            return SampledSpectrum(0);

        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        Vector3f wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosTheta_i == 0 || cosTheta_o == 0)
            return SampledSpectrum(0.);
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return SampledSpectrum(0.);
        wh = Normalize(wh);
        SampledSpectrum F = fresnel.Evaluate(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))));
        return distribution.D(wh) * distribution.G(wo, wi) * F /
               (4 * cosTheta_i * cosTheta_o);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};

        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        if (wo.z == 0)
            return {};
        Vector3f wh = distribution.Sample_wm(wo, u);
        Vector3f wi = Reflect(wo, wh);
        CHECK_RARE(1e-6, Dot(wo, wh) <= 0);
        if (!SameHemisphere(wo, wi) || Dot(wo, wh) <= 0)
            return {};

        // Compute PDF of _wi_ for microfacet reflection
        Float pdf = distribution.PDF(wo, wh) / (4 * Dot(wo, wh));

        // TODO: reuse fragments from f()
        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        // Handle degenerate cases for microfacet reflection
        if (cosTheta_i == 0 || cosTheta_o == 0)
            return {};
        SampledSpectrum F = fresnel.Evaluate(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))));
        SampledSpectrum f = distribution.D(wh) * distribution.G(wo, wi) * F /
                            (4 * cosTheta_i * cosTheta_o);
        return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return 0;

        if (!SameHemisphere(wo, wi))
            return 0;
        Vector3f wh = wo + wi;
        CHECK_RARE(1e-6, LengthSquared(wh) == 0);
        CHECK_RARE(1e-6, Dot(wo, wh) < 0);
        if (LengthSquared(wh) == 0 || Dot(wo, wh) <= 0)
            return 0;

        wh = Normalize(wh);
        return distribution.PDF(wo, wh) / (4 * Dot(wo, wh));
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        return scratchBuffer.Alloc<MicrofacetReflectionBxDF>(
            distribution.Regularize(scratchBuffer), fresnel);
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Glossy); }

  private:
    // MicrofacetReflection Private Data
    MicrofacetDistributionHandle distribution;
    FresnelHandle fresnel;
};

class alignas(8) MicrofacetTransmissionBxDF {
  public:
    // MicrofacetTransmission Public Methods
    PBRT_CPU_GPU
    MicrofacetTransmissionBxDF(const SampledSpectrum &T,
                               MicrofacetDistributionHandle distribution, Float eta)
        : T(T),
          distribution(distribution),
          // Things blow up otherwise since given wo and wi, it's completely
          // indeterminate what wh is...
          eta(eta == 1 ? 1.01 : eta) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const;

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const;

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        return scratchBuffer.Alloc<MicrofacetTransmissionBxDF>(
            SampledSpectrum(1.f), distribution.Regularize(scratchBuffer), eta);
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return (BxDFFlags::Transmission | BxDFFlags::Glossy); }

  private:
    // MicrofacetTransmission Private Data
    SampledSpectrum T;
    MicrofacetDistributionHandle distribution;
    Float eta;
};

class alignas(8) SpecularReflectionBxDF {
  public:
    // SpecularReflection Public Methods
    PBRT_CPU_GPU
    SpecularReflectionBxDF(FresnelHandle fresnel) : fresnel(fresnel) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        return SampledSpectrum(0.f);
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};

        // Compute perfect specular reflection direction
        Vector3f wi(-wo.x, -wo.y, wo.z);
        SampledSpectrum f = fresnel.Evaluate(CosTheta(wi)) / AbsCosTheta(wi);
        return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        return 0;
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        MicrofacetDistributionHandle distrib =
            scratchBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
        return scratchBuffer.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel);
    }

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Specular); }

  private:
    // SpecularReflection Private Data
    FresnelHandle fresnel;
};

struct LayeredBxDFConfig {
    int maxDepth = 10;
    int nSamples = 1;
    bool twoSided = true;
    bool deterministic = false;
};

template <typename TopBxDF = BxDFHandle, typename BottomBxDF = BxDFHandle>
class alignas(8) LayeredBxDF {
  public:
    PBRT_CPU_GPU
    LayeredBxDF(TopBxDF top, BottomBxDF bottom, Float thickness,
                const SampledSpectrum &albedo, Float g, LayeredBxDFConfig config)
        : flags([](BxDFFlags topFlags, BxDFFlags bottomFlags,
                   const SampledSpectrum &albedo) -> BxDFFlags {
              CHECK(IsTransmissive(topFlags) ||
                    IsTransmissive(bottomFlags));  // otherwise, why bother?
              BxDFFlags flags = BxDFFlags::Reflection;
              if (IsSpecular(topFlags) && IsSpecular(bottomFlags) && !albedo)
                  flags = flags | BxDFFlags::Specular;
              else if (IsDiffuse(topFlags) && IsDiffuse(bottomFlags))
                  flags = flags | BxDFFlags::Diffuse;
              else
                  flags = flags | BxDFFlags::Glossy;
              if (IsTransmissive(topFlags) && IsTransmissive(bottomFlags))
                  flags = flags | BxDFFlags::Transmission;
              return flags;
          }(top.Flags(), bottom.Flags(), albedo)),
          top(top),
          bottom(bottom),
          thickness(std::max(thickness, std::numeric_limits<Float>::min())),
          g(g),
          albedo(albedo),
          config(config) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        const Vector3f &wo, Float uc, const Point2f &u, TransportMode mode,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        return scratchBuffer.Alloc<LayeredBxDF<BxDFHandle, BxDFHandle>>(
            top.Regularize(scratchBuffer), bottom.Regularize(scratchBuffer), thickness,
            albedo, g, config);
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return true; }

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return flags; }

  protected:
    PBRT_CPU_GPU
    static Float Tr(Float dz, const Vector3f &w) {
        if (std::abs(dz) <= std::numeric_limits<Float>::min())
            return 1;
        return std::exp(-std::abs(dz) / AbsCosTheta(w));
    }

    BxDFFlags flags;
    TopBxDF top;
    BottomBxDF bottom;
    Float thickness, g;
    SampledSpectrum albedo;
    LayeredBxDFConfig config;
};

class CoatedDiffuseBxDF : public LayeredBxDF<DielectricInterfaceBxDF, DiffuseBxDF> {
  public:
    using LayeredBxDF::LayeredBxDF;

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) {
        // Note: putting this in the ScratchBuffer is wasteful since the
        // CoatedDiffuseBxDF holds copies of the top and bottom BxDF by
        // value. We'll at least skip regularizing the DiffuseBxDF,
        // since it's a no-op anyway...
        BxDFHandle topReg = top.Regularize(scratchBuffer);
        return scratchBuffer.Alloc<CoatedDiffuseBxDF>(
            *topReg.Cast<DielectricInterfaceBxDF>(), bottom, thickness, albedo, g,
            config);
    }
};

// HairBSDF Constants
static const int pMax = 3;
static const Float SqrtPiOver8 = 0.626657069f;

// HairBSDF Declarations
class alignas(8) HairBxDF {
  public:
    // HairBSDF Public Methods
    PBRT_CPU_GPU
    HairBxDF(Float h, Float eta, const SampledSpectrum &sigma_a, Float beta_m,
             Float beta_n, Float alpha);
    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const;
    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const;
    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const;

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) { return this; }

    std::string ToString() const;

    PBRT_CPU_GPU
    static RGBSpectrum SigmaAFromConcentration(Float ce, Float cp);
    PBRT_CPU_GPU
    static SampledSpectrum SigmaAFromReflectance(const SampledSpectrum &c, Float beta_n,
                                                 const SampledWavelengths &lambda);
    PBRT_CPU_GPU
    BxDFFlags Flags() const { return BxDFFlags::GlossyReflection; }

  private:
    // HairBSDF Private Methods
    PBRT_CPU_GPU
    pstd::array<Float, pMax + 1> ComputeApPDF(Float cosThetaO) const;

    // HairBSDF Private Data
    Float h, gamma_o, eta;
    SampledSpectrum sigma_a;
    Float beta_m, beta_n;
    Float v[pMax + 1];
    Float s;
    Float sin2kAlpha[3], cos2kAlpha[3];
};

class alignas(8) MeasuredBxDF {
  public:
    PBRT_CPU_GPU
    MeasuredBxDF(const MeasuredBRDFData *brdfData, const SampledWavelengths &lambda)
        : brdfData(brdfData), lambda(lambda) {}

    static MeasuredBRDFData *BRDFDataFromFile(const std::string &filename,
                                              Allocator alloc);

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const;

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const;
    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const;

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) { return this; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Glossy); }

  private:
    template <typename Value>
    PBRT_CPU_GPU static Value u2theta(Value u) {
        return Sqr(u) * (Pi / 2.f);
    }
    template <typename Value>
    PBRT_CPU_GPU static Value u2phi(Value u) {
        return (2.f * u - 1.f) * Pi;
    }
    template <typename Value>
    PBRT_CPU_GPU static Value theta2u(Value theta) {
        return std::sqrt(theta * (2.f / Pi));
    }
    template <typename Value>
    PBRT_CPU_GPU static Value phi2u(Value phi) {
        return (phi + Pi) / (2.f * Pi);
    }

    const MeasuredBRDFData *brdfData;
    SampledWavelengths lambda;
};

inline SampledSpectrum FresnelHandle::Evaluate(Float cosTheta_i) const {
    auto eval = [&](auto ptr) { return ptr->Evaluate(cosTheta_i); };
    return Apply<SampledSpectrum>(eval);
}

class BSSRDFAdapter {
  public:
    // BSSRDFAdapter Public Methods
    PBRT_CPU_GPU
    BSSRDFAdapter(Float eta)
        : flags(BxDFFlags::Reflection | BxDFFlags::Diffuse), eta(eta) {}

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        if (!SameHemisphere(wo, wi))
            return SampledSpectrum(0.f);

        // Sw()
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        SampledSpectrum f((1 - FrDielectric(CosTheta(wi), eta)) / (c * Pi));

        // Update BSSRDF transmission term to account for adjoint light
        // transport
        if (mode == TransportMode::Radiance)
            f *= Sqr(eta);
        return f;
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        TransportMode mode,
                                        BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};

        // Cosine-sample the hemisphere, flipping the direction if necessary
        Vector3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0)
            wi.z *= -1;
        return BSDFSample(f(wo, wi, mode), wi, PDF(wo, wi, mode, sampleFlags),
                          BxDFFlags::DiffuseReflection);
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
              BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return 0;
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
    }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return false; }

    PBRT_CPU_GPU
    BxDFHandle Regularize(ScratchBuffer &scratchBuffer) { return this; }

    std::string ToString() const;

    PBRT_CPU_GPU
    BxDFFlags Flags() const { return flags; }

  private:
    BxDFFlags flags;
    Float eta;
};

inline SampledSpectrum BxDFHandle::f(const Vector3f &wo, const Vector3f &wi,
                                     TransportMode mode) const {
    auto f = [&](auto ptr) -> SampledSpectrum { return ptr->f(wo, wi, mode); };
    return Apply<SampledSpectrum>(f);
}

inline pstd::optional<BSDFSample> BxDFHandle::Sample_f(
    const Vector3f &wo, Float uc, const Point2f &u, TransportMode mode,
    BxDFReflTransFlags sampleFlags) const {
    auto sample_f = [&](auto ptr) -> pstd::optional<BSDFSample> {
        return ptr->Sample_f(wo, uc, u, mode, sampleFlags);
    };
    return Apply<pstd::optional<BSDFSample>>(sample_f);
}

inline Float BxDFHandle::PDF(const Vector3f &wo, const Vector3f &wi, TransportMode mode,
                             BxDFReflTransFlags sampleFlags) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi, mode, sampleFlags); };
    return Apply<Float>(pdf);
}

inline bool BxDFHandle::SampledPDFIsProportional() const {
    auto approx = [&](auto ptr) { return ptr->SampledPDFIsProportional(); };
    return Apply<bool>(approx);
}

inline BxDFFlags BxDFHandle::Flags() const {
    auto flags = [&](auto ptr) { return ptr->Flags(); };
    return Apply<BxDFFlags>(flags);
}

inline BxDFHandle BxDFHandle::Regularize(ScratchBuffer &scratchBuffer) {
    auto regularize = [&](auto ptr) { return ptr->Regularize(scratchBuffer); };
    return Apply<BxDFHandle>(regularize);
}

extern template class LayeredBxDF<DielectricInterfaceBxDF, DiffuseBxDF>;
extern template class LayeredBxDF<BxDFHandle, BxDFHandle>;

template <typename T>
struct BxDFTraits {};
template <>
struct BxDFTraits<DiffuseBxDF> {
    static const char *name() { return "DiffuseBxDF"; }
};
template <typename TopBxDF, typename BottomBxDF>
struct BxDFTraits<LayeredBxDF<TopBxDF, BottomBxDF>> {
    static const char *name() { return "LayeredBxDF"; }
};
template <>
struct BxDFTraits<DielectricInterfaceBxDF> {
    static const char *name() { return "DielectricInterfaceBxDF"; }
};
template <>
struct BxDFTraits<ThinDielectricBxDF> {
    static const char *name() { return "ThinDielectricBxDF"; }
};
template <>
struct BxDFTraits<SpecularReflectionBxDF> {
    static const char *name() { return "SpecularReflectionBxDF"; }
};
template <>
struct BxDFTraits<HairBxDF> {
    static const char *name() { return "HairBxDF"; }
};
template <>
struct BxDFTraits<MeasuredBxDF> {
    static const char *name() { return "MeasuredBxDF"; }
};
template <>
struct BxDFTraits<MicrofacetReflectionBxDF> {
    static const char *name() { return "MicrofacetReflectionBxDF"; }
};
template <>
struct BxDFTraits<MicrofacetTransmissionBxDF> {
    static const char *name() { return "MicrofacetTransmissionBxDF"; }
};
template <>
struct BxDFTraits<BSSRDFAdapter> {
    static const char *name() { return "BSSRDFAdapter"; }
};
template <>
struct BxDFTraits<CoatedDiffuseBxDF> {
    static const char *name() { return "CoatedDiffuseBxDF"; }
};
template <>
struct BxDFTraits<GeneralLayeredBxDF> {
    static const char *name() { return "GeneralLayeredBxDF"; }
};

}  // namespace pbrt

#endif  // PBRT_BXDF_H
