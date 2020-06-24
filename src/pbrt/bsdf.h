
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_BSDF_H
#define PBRT_CORE_BSDF_H

// core/bsdf.h*
#include <pbrt/pbrt.h>

#include <pbrt/interaction.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/profile.h>
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

inline std::string ToString(TransportMode mode) {
    return mode == TransportMode::Radiance ? "Radiance" : "Importance";
}

// BSDF Inline Functions
// BSDF Declarations
enum class BxDFReflTransFlags {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    All = Reflection | Transmission
};

PBRT_HOST_DEVICE
inline BxDFReflTransFlags operator|(BxDFReflTransFlags a, BxDFReflTransFlags b) {
    return BxDFReflTransFlags((int)a | (int)b);
}

PBRT_HOST_DEVICE
inline int operator&(BxDFReflTransFlags a, BxDFReflTransFlags b) {
    return ((int)a & (int)b);
}

PBRT_HOST_DEVICE
inline BxDFReflTransFlags &operator|=(BxDFReflTransFlags &a, BxDFReflTransFlags b) {
    (int &)a |= int(b);
    return a;
}

std::string ToString(BxDFReflTransFlags flags);

enum class BxDFFlags {
    Unset = 0,
    Reflection = 1 << 0,
    Transmission = 1 << 1,
    Diffuse = 1 << 2,
    Glossy = 1 << 3,
    Specular = 1 << 4,
    DiffuseReflection = Diffuse | Reflection,
    DiffuseTransmission = Diffuse | Transmission,
    GlossyReflection = Glossy | Reflection,
    GlossyTransmission = Glossy | Transmission,
    SpecularReflection = Specular | Reflection,
    SpecularTransmission = Specular | Transmission,
    All = Diffuse | Glossy | Specular | Reflection | Transmission
};

PBRT_HOST_DEVICE
inline BxDFFlags operator|(BxDFFlags a, BxDFFlags b) {
    return BxDFFlags((int)a | (int)b);
}

PBRT_HOST_DEVICE
inline int operator&(BxDFFlags a, BxDFFlags b) {
    return ((int)a & (int)b);
}

PBRT_HOST_DEVICE
inline int operator&(BxDFFlags a, BxDFReflTransFlags b) {
    return ((int)a & (int)b);
}

PBRT_HOST_DEVICE
inline BxDFFlags &operator|=(BxDFFlags &a, BxDFFlags b) {
    (int &)a |= int(b);
    return a;
}

PBRT_HOST_DEVICE
inline bool IsReflective(BxDFFlags flags) { return (flags & BxDFFlags::Reflection) != 0; }
PBRT_HOST_DEVICE
inline bool IsTransmissive(BxDFFlags flags) { return (flags & BxDFFlags::Transmission) != 0; }
PBRT_HOST_DEVICE
inline bool IsDiffuse(BxDFFlags flags) { return (flags & BxDFFlags::Diffuse) != 0; }
PBRT_HOST_DEVICE
inline bool IsGlossy(BxDFFlags flags) { return (flags & BxDFFlags::Glossy) != 0; }
PBRT_HOST_DEVICE
inline bool IsSpecular(BxDFFlags flags) { return (flags & BxDFFlags::Specular) != 0; }

std::string ToString(BxDFFlags flags);

struct BSDFSample {
    BSDFSample() = default;
    PBRT_HOST_DEVICE
    BSDFSample(const SampledSpectrum &f, const Vector3f &wi, Float pdf,
               BxDFFlags flags)
        : f(f), wi(wi), pdf(pdf), flags(flags) {}

    SampledSpectrum f;
    Vector3f wi;
    Float pdf = 0;
    BxDFFlags flags;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    bool IsReflection() const { return pbrt::IsReflective(flags); }
    PBRT_HOST_DEVICE
    bool IsTransmission() const { return pbrt::IsTransmissive(flags); }
    PBRT_HOST_DEVICE
    bool IsDiffuse() const { return pbrt::IsDiffuse(flags); }
    PBRT_HOST_DEVICE
    bool IsGlossy() const { return pbrt::IsGlossy(flags); }
    PBRT_HOST_DEVICE
    bool IsSpecular() const { return pbrt::IsSpecular(flags); }
};

// BxDFHandle Declarations
class LambertianBxDF;
template <typename TopBxDF, typename BottomBxDF> class LayeredBxDF;
class DielectricInterfaceBxDF;
class ThinDielectricBxDF;
class SpecularReflectionBxDF;
class SpecularTransmissionBxDF;
class HairBxDF;
class MeasuredBxDF;
class MixBxDF;
class MicrofacetReflectionBxDF;
class MicrofacetTransmissionBxDF;
class DisneyBxDF;
class SeparableBSSRDFAdapter;
using CoatedDiffuseBxDF = LayeredBxDF<DielectricInterfaceBxDF, LambertianBxDF>;
using GeneralLayeredBxDF = LayeredBxDF<BxDFHandle, BxDFHandle>;

class BxDFHandle : public TaggedPointer<LambertianBxDF, CoatedDiffuseBxDF, GeneralLayeredBxDF,
                                        DielectricInterfaceBxDF,
                                        ThinDielectricBxDF, SpecularReflectionBxDF,
                                        SpecularTransmissionBxDF, HairBxDF, MeasuredBxDF,
                                        MixBxDF, MicrofacetReflectionBxDF,
                                        MicrofacetTransmissionBxDF, DisneyBxDF,
                                        SeparableBSSRDFAdapter>
{
  public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    BxDFHandle(TaggedPointer<LambertianBxDF, CoatedDiffuseBxDF, GeneralLayeredBxDF,
                             DielectricInterfaceBxDF,
                             ThinDielectricBxDF, SpecularReflectionBxDF,
                             SpecularTransmissionBxDF, HairBxDF, MeasuredBxDF,
                             MixBxDF, MicrofacetReflectionBxDF,
                             MicrofacetTransmissionBxDF, DisneyBxDF,
                             SeparableBSSRDFAdapter> tp)
    : TaggedPointer(tp) { }

    // BxDF Interface
    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;
    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &wo, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u2) const;
    PBRT_HOST_DEVICE
    SampledSpectrum rho(pstd::span<const Float> uc1, pstd::span<const Point2f> u1,
                        pstd::span<const Float> uc2, pstd::span<const Point2f> u2) const;

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    void FlipTransportMode();

    PBRT_HOST_DEVICE
    bool PDFIsApproximate() const;

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const;
};

class BSDF {
  public:
    // BSDF Public Methods
    PBRT_HOST_DEVICE
    BSDF(const SurfaceInteraction &si, BxDFHandle bxdf, Float eta = 1)
        : eta(Dot(si.wo, si.n) < 0 ? 1 / eta : eta),
          bxdf(bxdf),
          ng(si.n),
          shadingFrame(Frame::FromXZ(Normalize(si.shading.dpdu),
                                     Vector3f(si.shading.n)))
    { }

    PBRT_HOST_DEVICE
    bool IsNonSpecular() const {
        return (bxdf.Flags() & (BxDFFlags::Diffuse | BxDFFlags::Glossy));
    }
    PBRT_HOST_DEVICE
    bool IsDiffuse() const {
        return (bxdf.Flags() & BxDFFlags::Diffuse);
    }
    PBRT_HOST_DEVICE
    bool IsGlossy() const {
        return (bxdf.Flags() & BxDFFlags::Glossy);
    }
    PBRT_HOST_DEVICE
    bool HasReflection() const {
        return (bxdf.Flags() & BxDFFlags::Reflection);
    }
    PBRT_HOST_DEVICE
    bool HasTransmission() const {
        return (bxdf.Flags() & BxDFFlags::Transmission);
    }

    PBRT_HOST_DEVICE
    Vector3f WorldToLocal(const Vector3f &v) const {
        return shadingFrame.ToLocal(v);
    }
    PBRT_HOST_DEVICE
    Vector3f LocalToWorld(const Vector3f &v) const {
        return shadingFrame.FromLocal(v);
    }

    template <typename BxDF>
    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &woW, const Vector3f &wiW) const {
        ProfilerScope pp(ProfilePhase::BSDFEvaluation);
        Vector3f wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
        if (wo.z == 0) return SampledSpectrum(0.);

        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        return specificBxDF->f(wo, wi);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &woW, const Vector3f &wiW) const {
        ProfilerScope pp(ProfilePhase::BSDFEvaluation);
        Vector3f wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
        if (wo.z == 0) return SampledSpectrum(0.);

        return bxdf.f(wo, wi);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum rho(pstd::span<const Float> uc1, pstd::span<const Point2f> u1,
                        pstd::span<const Float> uc2, pstd::span<const Point2f> u2) const {
        return bxdf.rho(uc1, u1, uc2, u2);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &woWorld, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u) const {
        Vector3f wo = WorldToLocal(woWorld);
        return bxdf.rho(wo, uc, u);
    }

    template <typename BxDF>
    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &woWorld, Float u,
                                        const Point2f &u2,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        ProfilerScope pp(ProfilePhase::BSDFSampling);

        Vector3f wo = WorldToLocal(woWorld);
        if (wo.z == 0) return {};

        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        if (!(specificBxDF->Flags() & sampleFlags))
            return {};

        pstd::optional<BSDFSample> bs = specificBxDF->Sample_f(wo, u, u2, sampleFlags);
        if (!bs || bs->pdf == 0 || !bs->f) return {};
        CHECK_GT(bs->pdf, 0);

        VLOG(2, "For wo = %s, sampled f = %s, pdf = %f, ratio = %s, wi = %s", wo,
             bs->f, bs->pdf, (bs->pdf > 0) ? (bs->f / bs->pdf) : SampledSpectrum(0.),
             bs->wi);

        bs->wi = LocalToWorld(bs->wi);

        return bs;
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &woWorld, Float u,
                                        const Point2f &u2,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        ProfilerScope pp(ProfilePhase::BSDFSampling);

        Vector3f wo = WorldToLocal(woWorld);
        if (wo.z == 0) return {};

        if (!(bxdf.Flags() & sampleFlags))
            return {};

        pstd::optional<BSDFSample> bs = bxdf.Sample_f(wo, u, u2, sampleFlags);
        if (!bs || bs->pdf == 0 || !bs->f) return {};
        CHECK_GT(bs->pdf, 0);

        VLOG(2, "For wo = %s, sampled f = %s, pdf = %f, ratio = %s, wi = %s", wo,
             bs->f, bs->pdf, (bs->pdf > 0) ? (bs->f / bs->pdf) : SampledSpectrum(0.),
             bs->wi);

        bs->wi = LocalToWorld(bs->wi);

        return bs;
    }

    PBRT_HOST_DEVICE
    SampledSpectrum SampleSpecular_f(const Vector3f &wo, Vector3f *wi,
                                     BxDFReflTransFlags sampleFlags) const;

    template <typename BxDF>
    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &woWorld, const Vector3f &wiWorld,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        ProfilerScope pp(ProfilePhase::BSDFPdf);
        Vector3f wo = WorldToLocal(woWorld), wi = WorldToLocal(wiWorld);
        if (wo.z == 0) return 0.;
        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        return specificBxDF->PDF(wo, wi, sampleFlags);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &woWorld, const Vector3f &wiWorld,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        ProfilerScope pp(ProfilePhase::BSDFPdf);
        Vector3f wo = WorldToLocal(woWorld), wi = WorldToLocal(wiWorld);
        if (wo.z == 0) return 0.;
        return bxdf.PDF(wo, wi, sampleFlags);
    }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    void Regularize(MaterialBuffer &materialBuffer) {
        bxdf = bxdf.Regularize(materialBuffer);
    }
    PBRT_HOST_DEVICE
    BxDFHandle GetBxDF() { return bxdf; }
    PBRT_HOST_DEVICE
    bool PDFIsApproximate() const { return bxdf.PDFIsApproximate(); }

    // BSDF Public Data
    const Float eta;

  private:
    // BSDF Private Data
    BxDFHandle bxdf;
    Frame shadingFrame;
    Normal3f ng;
};

class alignas(8) MixBxDF {
  public:
    // MixBxDF Public Methods
    PBRT_HOST_DEVICE
    MixBxDF(Float t, BxDFHandle bxdf0, BxDFHandle bxdf1)
        : t(t), bxdf0(bxdf0), bxdf1(bxdf1) { }

    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &w, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u) const {
        return Lerp(t, bxdf0.rho(w, uc, u), bxdf1.rho(w, uc, u));
    }

    PBRT_HOST_DEVICE
    SampledSpectrum rho(pstd::span<const Float> uc1, pstd::span<const Point2f> u1,
                        pstd::span<const Float> uc2, pstd::span<const Point2f> u2) const {
        return Lerp(t, bxdf0.rho(uc1, u1, uc2, u2), bxdf1.rho(uc1, u1, uc2, u2));
    }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;
    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags) const;
    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return bxdf0.Flags() | bxdf1.Flags(); }

  private:
    Float t;
    BxDFHandle bxdf0, bxdf1;
};

class alignas(8) FresnelConductor {
  public:
    // FresnelConductor Public Methods
    PBRT_HOST_DEVICE
    FresnelConductor(const SampledSpectrum &eta, const SampledSpectrum &k)
        : eta(eta), k(k) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(Float cosTheta_i) const {
        return FrConductor(std::abs(cosTheta_i), eta, k);
    }

    std::string ToString() const;

  private:
    SampledSpectrum eta, k;
};

class alignas(8) FresnelDielectric {
  public:
    // FresnelDielectric Public Methods
    PBRT_HOST_DEVICE
    FresnelDielectric(Float eta, bool opaque = false)
        : eta(eta), opaque(opaque) {}

    PBRT_HOST_DEVICE
    SampledSpectrum Evaluate(Float cosTheta_i) const {
        if (opaque) cosTheta_i = std::abs(cosTheta_i);
        return SampledSpectrum(FrDielectric(cosTheta_i, eta));
    }

    std::string ToString() const;

  private:
    Float eta;
    bool opaque;
};

class DisneyFresnel;

class FresnelHandle : public TaggedPointer<FresnelConductor, FresnelDielectric,
                                           DisneyFresnel> {
public:
    using TaggedPointer::TaggedPointer;
    FresnelHandle(TaggedPointer<FresnelConductor, FresnelDielectric,
                                DisneyFresnel> tp)
        : TaggedPointer(tp) { }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(Float cosTheta_i) const;
};

class alignas(8) LambertianBxDF {
  public:
    // Lambertian Public Methods
    PBRT_HOST_DEVICE
    LambertianBxDF(const SampledSpectrum &R, const SampledSpectrum &T, Float sigma)
        : R(R), T(T) {
        sigma = Radians(sigma);
        Float sigma2 = sigma * sigma;
        A = 1 - sigma2 / (2 * (sigma2 + 0.33f));
        B = 0.45f * sigma2 / (sigma2 + 0.09f);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
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
    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Float pr = R.MaxComponentValue(), pt = T.MaxComponentValue();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
        if (pr == 0 && pt == 0) return {};

        Float cpdf;
        // TODO: rewrite to a single code path for the GPU. Good chance to
        // discuss divergence.
        if (SampleDiscrete({pr, pt}, uc, &cpdf) == 0) {
            Vector3f wi = SampleCosineHemisphere(u);
            if (wo.z < 0) wi.z *= -1;
            Float pdf = AbsCosTheta(wi) * InvPi * cpdf;
            return BSDFSample(f(wo, wi), wi, pdf,
                              BxDFFlags::DiffuseReflection);
        } else {
            Vector3f wi = SampleCosineHemisphere(u);
            if (wo.z > 0) wi.z *= -1;
            Float pdf = AbsCosTheta(wi) * InvPi * cpdf;
            return BSDFSample(f(wo, wi), wi, pdf,
                              BxDFFlags::DiffuseTransmission);
        }
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Float pr = R.MaxComponentValue(), pt = T.MaxComponentValue();
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
        if (pr == 0 && pt == 0) return 0;

        if (SameHemisphere(wo, wi))
            return pr / (pr + pt) * AbsCosTheta(wi) * InvPi;
        else
            return pt / (pr + pt) * AbsCosTheta(wi) * InvPi;
    }

    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &, pstd::span<const Float>,
                        pstd::span<const Point2f>) const {
        return R + T;
    }
    PBRT_HOST_DEVICE
    SampledSpectrum rho(int, pstd::span<const Float>, pstd::span<const Point2f>,
                        pstd::span<const Float>, pstd::span<const Point2f>) const {
        return R + T;
    }
    std::string ToString() const;

    PBRT_HOST_DEVICE
    void FlipTransportMode() { }

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer) { return this; }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return ((R ? BxDFFlags::DiffuseReflection : BxDFFlags::Unset) |
                                      (T ? BxDFFlags::DiffuseTransmission : BxDFFlags::Unset)); }

  private:
    // Lambertian Private Data
    SampledSpectrum R, T;
    Float A, B;
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
    PBRT_HOST_DEVICE
    LayeredBxDF(TopBxDF top, BottomBxDF bottom,
                Float thickness, const SampledSpectrum &albedo,
                Float g, LayeredBxDFConfig config)
        : flags([](BxDFFlags topFlags, BxDFFlags bottomFlags, const SampledSpectrum &albedo) -> BxDFFlags {
            CHECK(IsTransmissive(topFlags) || IsTransmissive(bottomFlags));     // otherwise, why bother?
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
        top(top), bottom(bottom),
        thickness(std::max(thickness, std::numeric_limits<Float>::min())),
        g(g),
        albedo(albedo), config(config) { }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    bool PDFIsApproximate() const { return true; }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return flags; }

private:
    PBRT_HOST_DEVICE_INLINE
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

// MicrofacetDistribution Declarations
class alignas(8) TrowbridgeReitzDistribution {
  public:
    // TrowbridgeReitzDistribution Public Methods
    PBRT_HOST_DEVICE
    static inline Float RoughnessToAlpha(Float roughness);

    PBRT_HOST_DEVICE_INLINE
    TrowbridgeReitzDistribution(Float alpha_x, Float alpha_y)
        : alpha_x(std::max<Float>(1e-4, alpha_x)),
          alpha_y(std::max<Float>(1e-4, alpha_y)) {
    }

    PBRT_HOST_DEVICE_INLINE
    Float D(const Vector3f &wh) const {
        Float tan2Theta = Tan2Theta(wh);
        if (std::isinf(tan2Theta)) return 0.;
        Float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);
        Float e =
            (Cos2Phi(wh) / (alpha_x * alpha_x) + Sin2Phi(wh) / (alpha_y * alpha_y)) *
            tan2Theta;
        return 1 / (Pi * alpha_x * alpha_y * cos4Theta * (1 + e) * (1 + e));
    }

    PBRT_HOST_DEVICE_INLINE
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        return 1 / (1 + Lambda(wo) + Lambda(wi));
    }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Sample_wm(const Point2f &u) const {
        return SampleTrowbridgeReitz(alpha_x, alpha_y, u);
    }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Sample_wm(const Vector3f &wo, const Point2f &u) const {
        bool flip = wo.z < 0;
        Vector3f wm = SampleTrowbridgeReitzVisibleArea(flip ? -wo : wo, alpha_x, alpha_y, u);
        if (flip) wm = -wm;
        return wm;
    }

    std::string ToString() const;

    PBRT_HOST_DEVICE_INLINE
    bool EffectivelySpecular() const { return std::min(alpha_x, alpha_y) < 1e-3; }

    PBRT_HOST_DEVICE
    MicrofacetDistributionHandle Regularize(MaterialBuffer &materialBuffer) const;

    PBRT_HOST_DEVICE_INLINE
    Float Lambda(const Vector3f &w) const {
        Float tan2Theta = Tan2Theta(w);
        if (std::isinf(tan2Theta)) return 0.;
        // Compute _alpha_ for direction _w_
        Float alpha2 = Cos2Phi(w) * alpha_x * alpha_x + Sin2Phi(w) * alpha_y * alpha_y;
        return (-1 + std::sqrt(1.f + alpha2 * tan2Theta)) / 2;
    }

  private:
    // TrowbridgeReitzDistribution Private Data
    Float alpha_x, alpha_y;
};

class alignas(8) DisneyMicrofacetDistribution : public TrowbridgeReitzDistribution {
public:
    DisneyMicrofacetDistribution(Float alpha_x, Float alpha_y)
        : TrowbridgeReitzDistribution(alpha_x, alpha_y) {}

    PBRT_HOST_DEVICE_INLINE
    Float G1(const Vector3f &w) const {
        //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
        return 1 / (1 + Lambda(w));
    }

    PBRT_HOST_DEVICE_INLINE
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        // Disney uses the separable masking-shadowing model.
        return G1(wo) * G1(wi);
    }
};

class MicrofacetDistributionHandle : public TaggedPointer<TrowbridgeReitzDistribution,
                                                          DisneyMicrofacetDistribution> {
  public:
    using TaggedPointer::TaggedPointer;
    MicrofacetDistributionHandle(TaggedPointer<TrowbridgeReitzDistribution,
                                               DisneyMicrofacetDistribution> tp)
        : TaggedPointer(tp) { }

    // MicrofacetDistributionHandle Public Methods
    PBRT_HOST_DEVICE_INLINE
    Float D(const Vector3f &wm) const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->D(wm);
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->D(wm);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    Float D(const Vector3f &w, const Vector3f &wm) const {
        return D(wm) * G1(w) * std::max<Float>(0, Dot(w, wm)) / AbsCosTheta(w);
    }

    PBRT_HOST_DEVICE_INLINE
    Float Lambda(const Vector3f &w) const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->Lambda(w);
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->Lambda(w);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    Float G1(const Vector3f &w) const {
        //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
        return 1 / (1 + Lambda(w));
    }

    PBRT_HOST_DEVICE_INLINE
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->G(wo, wi);
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->G(wo, wi);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Sample_wm(const Point2f &u) const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->Sample_wm(u);
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->Sample_wm(u);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Sample_wm(const Vector3f &wo, const Point2f &u) const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->Sample_wm(wo, u);
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->Sample_wm(wo, u);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    Float PDF(const Vector3f &wo, const Vector3f &wh) const {
        return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
    }

    std::string ToString() const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->ToString();
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->ToString();
        }
    }

    PBRT_HOST_DEVICE_INLINE
    bool EffectivelySpecular() const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->EffectivelySpecular();
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->EffectivelySpecular();
        }
    }

    PBRT_HOST_DEVICE_INLINE
    MicrofacetDistributionHandle Regularize(MaterialBuffer &materialBuffer) const {
        if (Tag() == TypeIndex<TrowbridgeReitzDistribution>())
            return Cast<TrowbridgeReitzDistribution>()->Regularize(materialBuffer);
        else {
            DCHECK_EQ(Tag(), TypeIndex<DisneyMicrofacetDistribution>());
            return Cast<DisneyMicrofacetDistribution>()->Regularize(materialBuffer);
        }
    }
};


// MicrofacetDistribution Inline Methods
inline Float TrowbridgeReitzDistribution::RoughnessToAlpha(Float roughness) {
    if (roughness < 1e-3) return roughness;
    Float x = std::log(roughness);
    return EvaluatePolynomial(x, 1.62142f, 0.819955f, 0.1734f, 0.0171201f,
                              0.000640711f);
}

class alignas(8) DielectricInterfaceBxDF {
public:
    PBRT_HOST_DEVICE
    DielectricInterfaceBxDF(Float eta, MicrofacetDistributionHandle distrib,
                            TransportMode mode)
        : flags(BxDFFlags::Reflection | BxDFFlags::Transmission |
                BxDFFlags((!distrib || distrib.EffectivelySpecular()) ?
                          BxDFFlags::Specular : BxDFFlags::Glossy)),
          eta(eta == 1 ? 1.001 : eta), distrib(distrib), mode(mode) {
    }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;
    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    void FlipTransportMode() {
        mode = (mode == TransportMode::Radiance) ? TransportMode::Importance :
            TransportMode::Radiance;
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return flags; }

private:
    BxDFFlags flags;
    Float eta;
    MicrofacetDistributionHandle distrib;
    TransportMode mode;
};

class alignas(8) ThinDielectricBxDF {
public:
    PBRT_HOST_DEVICE
    ThinDielectricBxDF(Float eta, TransportMode mode)
        : eta(eta), mode(mode) { }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u, BxDFReflTransFlags sampleFlags) const {
        Float R = FrDielectric(CosTheta(wo), eta), T = 1 - R;
        if (R < 1) {
            // Note this goes to Stokes glass plates...
            // R + TRT + TRRRT + ...
            R += T * T * R / (1 - R * R);
            T = 1 - R;
        }

        Float pr = R, pt = T;
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
        if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
        if (pr == 0 && pt == 0) return {};

        if (uc < pr / (pr + pt)) {
            // reflect
            Vector3f wi(-wo.x, -wo.y, wo.z);
            SampledSpectrum fr(R / AbsCosTheta(wi));
            return BSDFSample(fr, wi, pr / (pr + pt),
                              BxDFFlags::SpecularReflection);
        } else {
            // transmit
            // Figure out which $\eta$ is incident and which is transmitted
            bool entering = CosTheta(wo) > 0;
            Float etap = entering ? 1 / eta : eta;

            Vector3f wi = -wo;
            SampledSpectrum ft(T / AbsCosTheta(wi));

            // Account for non-symmetry with transmission to different medium
            if (mode == TransportMode::Radiance) ft *= etap * etap;
            return BSDFSample(ft, wi, pt / (pr + pt),
                              BxDFFlags::SpecularTransmission);
        }
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    void FlipTransportMode() {
        mode = (mode == TransportMode::Radiance) ? TransportMode::Importance :
            TransportMode::Radiance;
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Transmission |
                                      BxDFFlags::Specular); }

private:
    Float eta;
    TransportMode mode;
};

class alignas(8) MicrofacetReflectionBxDF {
  public:
    // MicrofacetReflection Public Methods
    PBRT_HOST_DEVICE
    MicrofacetReflectionBxDF(MicrofacetDistributionHandle distribution,
                             FresnelHandle fresnel)
        : distribution(distribution),
          fresnel(fresnel) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!SameHemisphere(wo, wi)) return SampledSpectrum(0);

        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        Vector3f wh = wi + wo;
        // Handle degenerate cases for microfacet reflection
        if (cosTheta_i == 0 || cosTheta_o == 0) return SampledSpectrum(0.);
        if (wh.x == 0 && wh.y == 0 && wh.z == 0) return SampledSpectrum(0.);
        wh = Normalize(wh);
        SampledSpectrum F = fresnel.Evaluate(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))));
        return distribution.D(wh) * distribution.G(wo, wi) * F /
            (4 * cosTheta_i * cosTheta_o);
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u, BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return {};

        // Sample microfacet orientation $\wh$ and reflected direction $\wi$
        if (wo.z == 0) return {};
        Vector3f wh = distribution.Sample_wm(wo, u);
        Vector3f wi = Reflect(wo, wh);
        CHECK_RARE(1e-6, Dot(wo, wh) <= 0);
        if (!SameHemisphere(wo, wi) || Dot(wo, wh) <= 0) return {};

        // Compute PDF of _wi_ for microfacet reflection
        Float pdf = distribution.PDF(wo, wh) / (4 * Dot(wo, wh));

        // TODO: reuse fragments from f()
        Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
        // Handle degenerate cases for microfacet reflection
        if (cosTheta_i == 0 || cosTheta_o == 0) return {};
        SampledSpectrum F = fresnel.Evaluate(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))));
        SampledSpectrum f = distribution.D(wh) * distribution.G(wo, wi) * F /
            (4 * cosTheta_i * cosTheta_o);
        return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return 0;

        if (!SameHemisphere(wo, wi)) return 0;
        Vector3f wh = wo + wi;
        CHECK_RARE(1e-6, LengthSquared(wh) == 0);
        CHECK_RARE(1e-6, Dot(wo, wh) < 0);
        if (LengthSquared(wh) == 0 || Dot(wo, wh) <= 0)
            return 0;

        wh = Normalize(wh);
        return distribution.PDF(wo, wh) / (4 * Dot(wo, wh));
    }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Glossy); }

  private:
    // MicrofacetReflection Private Data
    MicrofacetDistributionHandle distribution;
    FresnelHandle fresnel;
};

class alignas(8) MicrofacetTransmissionBxDF {
  public:
    // MicrofacetTransmission Public Methods
    PBRT_HOST_DEVICE
    MicrofacetTransmissionBxDF(const SampledSpectrum &T,
                               MicrofacetDistributionHandle distribution,
                               Float eta, TransportMode mode)
        : T(T),
          distribution(distribution),
          // Things blow up otherwise since given wo and wi, it's completely
          // indeterminate what wh is...
          eta(eta == 1 ? 1.01 : eta),
          mode(mode) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u, BxDFReflTransFlags sampleFlags) const;

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    void FlipTransportMode() {
        mode = (mode == TransportMode::Radiance) ? TransportMode::Importance :
            TransportMode::Radiance;
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return (BxDFFlags::Transmission | BxDFFlags::Glossy); }

  private:
    // MicrofacetTransmission Private Data
    SampledSpectrum T;
    MicrofacetDistributionHandle distribution;
    Float eta;
    TransportMode mode;
};

class alignas(8) SpecularReflectionBxDF {
  public:
    // SpecularReflection Public Methods
    PBRT_HOST_DEVICE
    SpecularReflectionBxDF(FresnelHandle fresnel)
        : fresnel(fresnel) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        return SampledSpectrum(0.f);
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return {};

        // Compute perfect specular reflection direction
        Vector3f wi(-wo.x, -wo.y, wo.z);
        SampledSpectrum f = fresnel.Evaluate(CosTheta(wi)) / AbsCosTheta(wi);
        return BSDFSample(f, wi, 1, BxDFFlags::SpecularReflection);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags) const { return 0; }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer) {
        MicrofacetDistributionHandle distrib =
            materialBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
        return materialBuffer.Alloc<MicrofacetReflectionBxDF>(distrib, fresnel);
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Specular); }

  private:
    // SpecularReflection Private Data
    FresnelHandle fresnel;
};

// Only used for Disney FWIW
class alignas(8) SpecularTransmissionBxDF {
  public:
    // SpecularTransmission Public Methods
    PBRT_HOST_DEVICE
    SpecularTransmissionBxDF(const SampledSpectrum &T, Float eta, TransportMode mode)
        : T(T),
          eta(eta),
          mode(mode) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        return SampledSpectrum(0.f);
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags) const;

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags) const { return 0; }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFHandle Regularize(MaterialBuffer &materialBuffer);

    PBRT_HOST_DEVICE
    void FlipTransportMode() {
        mode = (mode == TransportMode::Radiance) ? TransportMode::Importance :
            TransportMode::Radiance;
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return (BxDFFlags::Transmission | BxDFFlags::Specular); }

  private:
    // SpecularTransmission Private Data
    SampledSpectrum T;
    Float eta;
    TransportMode mode;
};

/*

Implementation of the Disney BSDF with Subsurface Scattering, as described in:
http://blog.selfshadow.com/publications/s2015-shading-course/burley/s2015_pbs_disney_bsdf_notes.pdf.

That model is based on the Disney BRDF, described in:
https://disney-animation.s3.amazonaws.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf

Many thanks for Brent Burley and Karl Li for answering many questions about
the details of the implementation.

The initial implementation of the BRDF was adapted from
https://github.com/wdas/brdf/blob/master/src/brdfs/disney.brdf, which is
licensed under a slightly-modified Apache 2.0 license.

*/

///////////////////////////////////////////////////////////////////////////
// DisneyDiffuse

class DisneyDiffuseLobe {
  public:
    PBRT_HOST_DEVICE
    DisneyDiffuseLobe(const SampledSpectrum &R)
        : R(R) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!SameHemisphere(wo, wi)) return SampledSpectrum(0.f);

        Float Fo = SchlickWeight(AbsCosTheta(wo)),
              Fi = SchlickWeight(AbsCosTheta(wi));

        // Diffuse fresnel - go from 1 at normal incidence to .5 at grazing.
        // Burley 2015, eq (4).
        return R * InvPi * (1 - Fo / 2) * (1 - Fi / 2);
    }

    std::string ToString() const;

  private:
    SampledSpectrum R;
};

///////////////////////////////////////////////////////////////////////////
// DisneyFakeSS

// "Fake" subsurface scattering lobe, based on the Hanrahan-Krueger BRDF
// approximation of the BSSRDF.
class DisneyFakeSSLobe {
  public:
    PBRT_HOST_DEVICE
    DisneyFakeSSLobe(const SampledSpectrum &R, Float roughness)
        : R(R),
          roughness(roughness) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!SameHemisphere(wo, wi)) return SampledSpectrum(0.f);

        Vector3f wh = wi + wo;
        if (LengthSquared(wh) == 0) return SampledSpectrum(0.);
        wh = Normalize(wh);
        Float cosThetaD = Dot(wi, wh);

        // Fss90 used to "flatten" retroreflection based on roughness
        Float Fss90 = cosThetaD * cosThetaD * roughness;
        Float Fo = SchlickWeight(AbsCosTheta(wo)),
              Fi = SchlickWeight(AbsCosTheta(wi));
        Float Fss = Lerp(Fo, 1.0, Fss90) * Lerp(Fi, 1.0, Fss90);
        // 1.25 scale is used to (roughly) preserve albedo
        Float ss =
            1.25f * (Fss * (1 / (AbsCosTheta(wo) + AbsCosTheta(wi)) - .5f) + .5f);

        return R * InvPi * ss;
    }

    std::string ToString() const;

  private:
    SampledSpectrum R;
    Float roughness;
};

///////////////////////////////////////////////////////////////////////////
// DisneyRetro

class DisneyRetroLobe {
  public:
    PBRT_HOST_DEVICE
    DisneyRetroLobe(const SampledSpectrum &R, Float roughness)
        : R(R),
          roughness(roughness) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!SameHemisphere(wo, wi)) return SampledSpectrum(0.f);

        Vector3f wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return SampledSpectrum(0.);
        wh = Normalize(wh);
        Float cosThetaD = Dot(wi, wh);

        Float Fo = SchlickWeight(AbsCosTheta(wo)),
              Fi = SchlickWeight(AbsCosTheta(wi));
        Float Rr = 2 * roughness * cosThetaD * cosThetaD;

        // Burley 2015, eq (4).
        return R * InvPi * Rr * (Fo + Fi + Fo * Fi * (Rr - 1));
    }

    std::string ToString() const;

  private:
    SampledSpectrum R;
    Float roughness;
};

///////////////////////////////////////////////////////////////////////////
// DisneySheen

class DisneySheenLobe {
  public:
    PBRT_HOST_DEVICE
    DisneySheenLobe(const SampledSpectrum &R)
        : R(R) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!SameHemisphere(wo, wi)) return SampledSpectrum(0.f);

        Vector3f wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return SampledSpectrum(0.);
        wh = Normalize(wh);
        Float cosThetaD = Dot(wi, wh);

        return R * SchlickWeight(cosThetaD);
    }

    std::string ToString() const;

  private:
    SampledSpectrum R;
};

///////////////////////////////////////////////////////////////////////////
// DisneyClearcoat

class DisneyClearcoatLobe {
  public:
    PBRT_HOST_DEVICE
    DisneyClearcoatLobe(Float weight, Float gloss)
        : weight(weight),
          gloss(gloss) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!SameHemisphere(wo, wi)) return SampledSpectrum(0.f);

        Vector3f wh = wi + wo;
        if (wh.x == 0 && wh.y == 0 && wh.z == 0)
            return SampledSpectrum(0.);
        wh = Normalize(wh);

        // Clearcoat has ior = 1.5 hardcoded -> F0 = 0.04. It then uses the
        // GTR1 distribution, which has even fatter tails than Trowbridge-Reitz
        // (which is GTR2).
        Float Dr = GTR1(AbsCosTheta(wh), gloss);
        Float Fr = FrSchlick(.04, Dot(wo, wh));
        // The geometric term always based on alpha = 0.25.
        Float Gr =
            smithG_GGX(AbsCosTheta(wo), .25) * smithG_GGX(AbsCosTheta(wi), .25);

        return SampledSpectrum(weight * Gr * Fr * Dr / 4);
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return {};

        // TODO: double check all this: there still seem to be some very
        // occasional fireflies with clearcoat; presumably there is a bug
        // somewhere.
        if (wo.z == 0) return {};

        Float alpha2 = gloss * gloss;
        Float cosTheta = SafeSqrt((1 - std::pow(alpha2, 1 - u[0])) / (1 - alpha2));
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Float phi = 2 * Pi * u[1];
        Vector3f wh = SphericalDirection(sinTheta, cosTheta, phi);
        if (!SameHemisphere(wo, wh)) wh = -wh;

        Vector3f wi = Reflect(wo, wh);
        if (!SameHemisphere(wo, wi)) return {};

        return BSDFSample(f(wo, wi), wi, PDF(wo, wi, sampleFlags),
                          BxDFFlags::GlossyReflection);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return 0;
        if (!SameHemisphere(wo, wi)) return 0;

        Vector3f wh = wi + wo;
        CHECK_RARE(1e-6, Dot(wo, wh) < 0);
        if (LengthSquared(wh) == 0 || Dot(wo, wh) < 0) return 0;
        wh = Normalize(wh);

        // The sampling routine samples wh exactly from the GTR1 distribution.
        // Thus, the final value of the PDF is just the value of the
        // distribution for wh converted to a mesure with respect to the
        // surface normal.
        Float Dr = GTR1(AbsCosTheta(wh), gloss);
        return Dr * AbsCosTheta(wh) / (4 * Dot(wo, wh));
    }

    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE
    inline static Float GTR1(Float cosTheta, Float alpha) {
        Float alpha2 = alpha * alpha;
        return (alpha2 - 1) /
            (Pi * std::log(alpha2) * (1 + (alpha2 - 1) * cosTheta * cosTheta));
    }

    // Smith masking/shadowing term.
    PBRT_HOST_DEVICE
    inline static Float smithG_GGX(Float cosTheta, Float alpha) {
        Float alpha2 = alpha * alpha;
        Float cosTheta2 = cosTheta * cosTheta;
        return 1 / (cosTheta + SafeSqrt(alpha2 + cosTheta2 - alpha2 * cosTheta2));
    }

    Float weight, gloss;
};

class alignas(8) DisneyBxDF {
 public:
    PBRT_HOST_DEVICE_INLINE
        DisneyBxDF(DisneyDiffuseLobe *diffuseReflection, DisneyFakeSSLobe *fakeSS,
                   DisneyRetroLobe *retro, DisneySheenLobe *sheen,
                   DisneyClearcoatLobe *clearcoat,
                   MicrofacetReflectionBxDF *glossyReflection,
                   MicrofacetTransmissionBxDF *glossyTransmission,
                   LambertianBxDF *diffuseTransmission,
                   SpecularTransmissionBxDF *subsurfaceBxDF)
        : diffuseReflection(diffuseReflection), fakeSS(fakeSS), retro(retro), sheen(sheen),
          clearcoat(clearcoat), glossyReflection(glossyReflection),
          glossyTransmission(glossyTransmission),
          diffuseTransmission(diffuseTransmission), subsurfaceBxDF(subsurfaceBxDF) {
    }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        SampledSpectrum f(0.);
        if (diffuseReflection)     f += diffuseReflection->f(wo, wi);
        if (fakeSS)                f += fakeSS->f(wo, wi);
        if (retro)                 f += retro->f(wo, wi);
        if (sheen)                 f += sheen->f(wo, wi);
        if (clearcoat)             f += clearcoat->f(wo, wi);
        if (glossyReflection)    f += glossyReflection->f(wo, wi);
        if (glossyTransmission)  f += glossyTransmission->f(wo, wi);
        if (diffuseTransmission)   f += diffuseTransmission->f(wo, wi);
        // subsurfaceBxDF is perfectly specular, so no need to worry about
        // it for evaluation.
        return f;
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags) const {
        auto BSDFSampleCosine = [](const Vector3f &wo, const Point2f &u) {
            Vector3f wi = SampleCosineHemisphere(u);
            if (!SameHemisphere(wo, wi)) wi = -wi;
            return BSDFSample{SampledSpectrum(0.f), wi,
                              CosineHemispherePDF(AbsCosTheta(wi)),
                              BxDFFlags::Diffuse | BxDFFlags::Reflection};
        };

        int n = nLobes(sampleFlags);
        int comp = std::min<int>(uc * n, n - 1);
        uc = (uc * n) - comp;
        CHECK(uc >= 0 && uc < 1);
        bool sampleReflection = sampleFlags & BxDFReflTransFlags::Reflection;
        bool sampleTransmission = sampleFlags & BxDFReflTransFlags::Transmission;

        pstd::optional<BSDFSample> bs;
        if (sampleReflection && diffuseReflection && comp-- == 0)
            bs = BSDFSampleCosine(wo, u);
        else if (sampleReflection && fakeSS && comp-- == 0)
            bs = BSDFSampleCosine(wo, u);
        else if (sampleReflection && retro && comp-- == 0)
            bs = BSDFSampleCosine(wo, u);
        else if (sampleReflection && sheen && comp-- == 0)
            bs = BSDFSampleCosine(wo, u);
        else if (sampleReflection && clearcoat && comp-- == 0)
            bs = clearcoat->Sample_f(wo, uc, u, sampleFlags);
        else if (sampleReflection && glossyReflection && comp-- == 0)
            bs = glossyReflection->Sample_f(wo, uc, u, sampleFlags);
        else if (sampleTransmission && glossyTransmission && comp-- == 0)
            bs = glossyTransmission->Sample_f(wo, uc, u, sampleFlags);
        else if (sampleTransmission && diffuseTransmission && comp-- == 0) {
            bs = BSDFSampleCosine(wo, u);
            bs->wi = -bs->wi;
        } else if (sampleTransmission && subsurfaceBxDF) {
            CHECK_EQ(0, comp);
            bs = subsurfaceBxDF->Sample_f(wo, uc, u, sampleFlags);
        }

        if (bs && !bs->IsSpecular()) {
            bs->f = f(wo, bs->wi);
            bs->pdf = PDF(wo, bs->wi, sampleFlags);
        }
        return bs;
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags) const {
        int nLobes = 0;
        bool sampleReflection = sampleFlags & BxDFReflTransFlags::Reflection;
        bool sampleTransmission = sampleFlags & BxDFReflTransFlags::Transmission;
        Float pdf = 0;

        if (diffuseReflection && sampleReflection) {
            pdf += SameHemisphere(wo, wi) ? CosineHemispherePDF(AbsCosTheta(wi)) : 0;
            ++nLobes;
        }
        if (fakeSS && sampleReflection) {
            pdf += SameHemisphere(wo, wi) ? CosineHemispherePDF(AbsCosTheta(wi)) : 0;
            ++nLobes;
        }
        if (retro && sampleReflection) {
            pdf += SameHemisphere(wo, wi) ? CosineHemispherePDF(AbsCosTheta(wi)) : 0;
            ++nLobes;
        }
        if (sheen && sampleReflection) {
            pdf += SameHemisphere(wo, wi) ? CosineHemispherePDF(AbsCosTheta(wi)) : 0;
            ++nLobes;
        }
        if (clearcoat && sampleReflection) {
            pdf += clearcoat->PDF(wo, wi, sampleFlags);
            ++nLobes;
        }
        if (glossyReflection && sampleReflection) {
            pdf += glossyReflection->PDF(wo, wi, sampleFlags);
            ++nLobes;
        }
        if (glossyTransmission && sampleTransmission) {
            pdf += glossyTransmission->PDF(wo, wi, sampleFlags);
            ++nLobes;
        }
        if (diffuseTransmission && sampleTransmission) {
            pdf += !SameHemisphere(wo, wi) ? CosineHemispherePDF(AbsCosTheta(wi)) : 0;
            ++nLobes;
        }
        // subsurfaceBxDF is specular, so can ignore it

        return pdf / nLobes;
    }

    std::string ToString() const;

    PBRT_HOST_DEVICE_INLINE
    BxDFFlags Flags() const {
        if (clearcoat)
            return (BxDFFlags::Reflection | BxDFFlags::Glossy);
        else
            return (BxDFFlags::Reflection | BxDFFlags::Diffuse);
    }

 private:
    PBRT_HOST_DEVICE
    int nLobes(BxDFReflTransFlags sampleFlags) const {
        bool sampleReflection = sampleFlags & BxDFReflTransFlags::Reflection;
        bool sampleTransmission = sampleFlags & BxDFReflTransFlags::Transmission;
        return (((diffuseReflection && sampleReflection) ? 1 : 0) +
                ((fakeSS && sampleReflection) ? 1 : 0) +
                ((retro && sampleReflection) ? 1 : 0) +
                ((sheen && sampleReflection) ? 1 : 0) +
                ((clearcoat && sampleReflection) ? 1 : 0) +
                ((glossyReflection && sampleReflection) ? 1 : 0) +
                ((glossyTransmission && sampleTransmission) ? 1 : 0) +
                ((diffuseTransmission && sampleTransmission) ? 1 : 0) +
                ((subsurfaceBxDF && sampleTransmission) ? 1 : 0));
    }

    DisneyDiffuseLobe *diffuseReflection;
    DisneyFakeSSLobe *fakeSS;
    DisneyRetroLobe *retro;
    DisneySheenLobe *sheen;
    DisneyClearcoatLobe *clearcoat;
    MicrofacetReflectionBxDF *glossyReflection;
    MicrofacetTransmissionBxDF *glossyTransmission;
    LambertianBxDF *diffuseTransmission;
    SpecularTransmissionBxDF *subsurfaceBxDF;
};

///////////////////////////////////////////////////////////////////////////
// DisneyFresnel

// Specialized Fresnel function used for the disney component, based on
// a mixture between dielectric and the Schlick Fresnel approximation.
class alignas(8) DisneyFresnel {
  public:
    PBRT_HOST_DEVICE_INLINE
    DisneyFresnel(const SampledSpectrum &R0, Float metallic, Float eta)
        : R0(R0), metallic(metallic), eta(eta) {}

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(Float cosI) const {
        return Lerp(metallic,
                    SampledSpectrum(FrDielectric(cosI, eta)),
                    FrSchlick(R0, cosI));
    }

    std::string ToString() const;

  private:
    SampledSpectrum R0;
    Float metallic, eta;
};

// HairBSDF Constants
static const int pMax = 3;
static const Float SqrtPiOver8 = 0.626657069f;

// HairBSDF Declarations
class alignas(8) HairBxDF {
  public:
    // HairBSDF Public Methods
    PBRT_HOST_DEVICE
    HairBxDF(Float h, Float eta, const SampledSpectrum &sigma_a, Float beta_m,
             Float beta_n, Float alpha);
    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;
    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags) const;
    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    static RGBSpectrum SigmaAFromConcentration(Float ce, Float cp);
    PBRT_HOST_DEVICE
    static SampledSpectrum SigmaAFromReflectance(const SampledSpectrum &c,
                                                 Float beta_n,
                                                 const SampledWavelengths &lambda);
    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return BxDFFlags::GlossyReflection; }

  private:
    // HairBSDF Private Methods
    PBRT_HOST_DEVICE
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
    PBRT_HOST_DEVICE
    MeasuredBxDF(const MeasuredBRDFData *brdfData,
                 const SampledWavelengths &lambda)
        : brdfData(brdfData), lambda(lambda) { }

    static MeasuredBRDFData *BRDFDataFromFile(const std::string &filename,
                                              Allocator alloc);

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u, BxDFReflTransFlags sampleFlags) const;
    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return (BxDFFlags::Reflection | BxDFFlags::Glossy); }

private:
    template <typename Value>
    PBRT_HOST_DEVICE
    static Value u2theta(Value u) {
        return Sqr(u) * (Pi / 2.f);
    }
    template <typename Value>
    PBRT_HOST_DEVICE
    static Value u2phi(Value u) {
        return (2.f * u - 1.f) * Pi;
    }
    template <typename Value>
    PBRT_HOST_DEVICE
    static Value theta2u(Value theta) {
        return std::sqrt(theta * (2.f / Pi));
    }
    template <typename Value>
    PBRT_HOST_DEVICE
    static Value phi2u(Value phi) {
        return (phi + Pi) / (2.f * Pi);
    }

    const MeasuredBRDFData *brdfData;
    SampledWavelengths lambda;
};

inline SampledSpectrum FresnelHandle::Evaluate(Float cosTheta_i) const {
    if (Tag() == TypeIndex<FresnelConductor>())
        return Cast<FresnelConductor>()->Evaluate(cosTheta_i);
    else if (Tag() == TypeIndex<FresnelDielectric>())
        return Cast<FresnelDielectric>()->Evaluate(cosTheta_i);
    else {
        DCHECK_EQ(Tag(), TypeIndex<DisneyFresnel>());
        return Cast<DisneyFresnel>()->Evaluate(cosTheta_i);
    }
}

class SeparableBSSRDFAdapter {
  public:
    // SeparableBSSRDFAdapter Public Methods
    PBRT_HOST_DEVICE
    SeparableBSSRDFAdapter(const SeparableBSSRDF *bssrdf)
        : flags(BxDFFlags::Reflection | BxDFFlags::Diffuse), bssrdf(bssrdf) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return flags; }

  private:
    BxDFFlags flags;
    const SeparableBSSRDF *bssrdf;
};

inline SampledSpectrum BxDFHandle::f(const Vector3f &wo, const Vector3f &wi) const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->f(wo, wi);
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->f(wo, wi);
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->f(wo, wi);
    case TypeIndex<SpecularReflectionBxDF>():
        return Cast<SpecularReflectionBxDF>()->f(wo, wi);
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->f(wo, wi);
    case TypeIndex<MicrofacetReflectionBxDF>():
        return Cast<MicrofacetReflectionBxDF>()->f(wo, wi);
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->f(wo, wi);
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->f(wo, wi);
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->f(wo, wi);
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->f(wo, wi);
    case TypeIndex<HairBxDF>():
        return Cast<HairBxDF>()->f(wo, wi);
    case TypeIndex<MeasuredBxDF>():
        return Cast<MeasuredBxDF>()->f(wo, wi);
    case TypeIndex<DisneyBxDF>():
        return Cast<DisneyBxDF>()->f(wo, wi);
    case TypeIndex<SeparableBSSRDFAdapter>():
        return Cast<SeparableBSSRDFAdapter>()->f(wo, wi);
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

inline pstd::optional<BSDFSample> BxDFHandle::Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                                       BxDFReflTransFlags sampleFlags) const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<SpecularReflectionBxDF>():
        return Cast<SpecularReflectionBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<MicrofacetReflectionBxDF>():
        return Cast<MicrofacetReflectionBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<HairBxDF>():
        return Cast<HairBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<MeasuredBxDF>():
        return Cast<MeasuredBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<DisneyBxDF>():
        return Cast<DisneyBxDF>()->Sample_f(wo, uc, u, sampleFlags);
    case TypeIndex<SeparableBSSRDFAdapter>(): {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};

        // Cosine-sample the hemisphere, flipping the direction if necessary
        Vector3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0) wi.z *= -1;
        return BSDFSample(f(wo, wi), wi, PDF(wo, wi, sampleFlags),
                          BxDFFlags::DiffuseReflection);
    }
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

inline Float BxDFHandle::PDF(const Vector3f &wo, const Vector3f &wi,
                             BxDFReflTransFlags sampleFlags) const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<SpecularReflectionBxDF>():
        return Cast<SpecularReflectionBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<MicrofacetReflectionBxDF>():
        return Cast<MicrofacetReflectionBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<HairBxDF>():
        return Cast<HairBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<MeasuredBxDF>():
        return Cast<MeasuredBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<DisneyBxDF>():
        return Cast<DisneyBxDF>()->PDF(wo, wi, sampleFlags);
    case TypeIndex<SeparableBSSRDFAdapter>():
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return 0;
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

extern template class LayeredBxDF<DielectricInterfaceBxDF, LambertianBxDF>;
extern template class LayeredBxDF<BxDFHandle, BxDFHandle>;

}  // namespace pbrt

#endif  // PBRT_BSDF_H
