
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
#include <tuple>

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
class BSSRDFAdapter;
using CoatedDiffuseBxDF = LayeredBxDF<DielectricInterfaceBxDF, LambertianBxDF>;
using GeneralLayeredBxDF = LayeredBxDF<BxDFHandle, BxDFHandle>;

struct RhoHemiDirSample {
    Float u;
    Point2f u2;
};

struct RhoHemiHemiSample {
    Float u;
    Point2f u2[2];
};

class BxDFHandle : public TaggedPointer<LambertianBxDF, CoatedDiffuseBxDF, GeneralLayeredBxDF,
                                        DielectricInterfaceBxDF,
                                        ThinDielectricBxDF, SpecularReflectionBxDF,
                                        SpecularTransmissionBxDF, HairBxDF, MeasuredBxDF,
                                        MixBxDF, MicrofacetReflectionBxDF,
                                        MicrofacetTransmissionBxDF,
                                        BSSRDFAdapter>
{
  public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    BxDFHandle(TaggedPointer<LambertianBxDF, CoatedDiffuseBxDF, GeneralLayeredBxDF,
                             DielectricInterfaceBxDF,
                             ThinDielectricBxDF, SpecularReflectionBxDF,
                             SpecularTransmissionBxDF, HairBxDF, MeasuredBxDF,
                             MixBxDF, MicrofacetReflectionBxDF,
                             MicrofacetTransmissionBxDF,
                             BSSRDFAdapter> tp)
    : TaggedPointer(tp) { }

    // BxDF Interface
    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    // RhoSampler: f(index) -> std::pair<Float, Point2f>
    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &wo, RhoSampler rhoSampler,
                        int nSamples = 16) const;

    // RhoSampler: f(index) -> std::tuple<Float, Point2f, Float, Point2f>
    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(RhoSampler rhoSampler, int nSamples = 16) const;

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
    bool IsSpecular() const {
        return (bxdf.Flags() & BxDFFlags::Specular);
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

    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(RhoSampler rhoSampler, int nSamples = 16) const {
        return bxdf.rho(rhoSampler, nSamples);
    }

    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &woWorld, RhoSampler rhoSampler,
                        int nSamples = 16) const {
        Vector3f wo = WorldToLocal(woWorld);
        return bxdf.rho(wo, rhoSampler, nSamples);
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

    template <typename RhoSampler>
    PBRT_HOST_DEVICE
        SampledSpectrum rho(const Vector3f &w, RhoSampler rhoSampler,
                            int nSamples) const {
        return Lerp(t, bxdf0.rho(w, rhoSampler, nSamples),
                    bxdf1.rho(w, rhoSampler, nSamples));
    }

    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(RhoSampler rhoSampler, int nSamples) const {
        return Lerp(t, bxdf0.rho(rhoSampler, nSamples), bxdf1.rho(rhoSampler, nSamples));
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

class FresnelHandle : public TaggedPointer<FresnelConductor, FresnelDielectric> {
public:
    using TaggedPointer::TaggedPointer;
    FresnelHandle(TaggedPointer<FresnelConductor, FresnelDielectric> tp)
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

    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(const Vector3f &, RhoSampler, int nSamples) const {
        return R + T;
    }

    template <typename RhoSampler>
    PBRT_HOST_DEVICE
    SampledSpectrum rho(RhoSampler rhoSampler, int nSamples) const {
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

class MicrofacetDistributionHandle : public TaggedPointer<TrowbridgeReitzDistribution> {
  public:
    using TaggedPointer::TaggedPointer;
    MicrofacetDistributionHandle(TaggedPointer<TrowbridgeReitzDistribution> tp)
        : TaggedPointer(tp) { }

    // MicrofacetDistributionHandle Public Methods
    PBRT_HOST_DEVICE_INLINE
    Float D(const Vector3f &wm) const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->D(wm);
    }

    PBRT_HOST_DEVICE_INLINE
    Float D(const Vector3f &w, const Vector3f &wm) const {
        return D(wm) * G1(w) * std::max<Float>(0, Dot(w, wm)) / AbsCosTheta(w);
    }

    PBRT_HOST_DEVICE_INLINE
    Float Lambda(const Vector3f &w) const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->Lambda(w);
    }

    PBRT_HOST_DEVICE_INLINE
    Float G1(const Vector3f &w) const {
        //    if (Dot(w, wh) * CosTheta(w) < 0.) return 0.;
        return 1 / (1 + Lambda(w));
    }

    PBRT_HOST_DEVICE_INLINE
    Float G(const Vector3f &wo, const Vector3f &wi) const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->G(wo, wi);
    }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Sample_wm(const Point2f &u) const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->Sample_wm(u);
    }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Sample_wm(const Vector3f &wo, const Point2f &u) const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->Sample_wm(wo, u);
    }

    PBRT_HOST_DEVICE_INLINE
    Float PDF(const Vector3f &wo, const Vector3f &wh) const {
        return D(wh) * G1(wo) * AbsDot(wo, wh) / AbsCosTheta(wo);
    }

    std::string ToString() const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->ToString();
    }

    PBRT_HOST_DEVICE_INLINE
    bool EffectivelySpecular() const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->EffectivelySpecular();
    }

    PBRT_HOST_DEVICE_INLINE
    MicrofacetDistributionHandle Regularize(MaterialBuffer &materialBuffer) const {
        DCHECK_EQ(Tag(), TypeIndex<TrowbridgeReitzDistribution>());
        return Cast<TrowbridgeReitzDistribution>()->Regularize(materialBuffer);
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
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        if (!distrib || distrib.EffectivelySpecular())
            return SampledSpectrum(0);

        if (SameHemisphere(wo, wi)) {
            // reflect
            Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
            Vector3f wh = wi + wo;
            // Handle degenerate cases for microfacet reflection
            if (cosTheta_i == 0 || cosTheta_o == 0) return SampledSpectrum(0.);
            if (wh.x == 0 && wh.y == 0 && wh.z == 0) return SampledSpectrum(0.);
            wh = Normalize(wh);
            Float F = FrDielectric(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))), eta);
            return SampledSpectrum(distrib.D(wh) * distrib.G(wo, wi) * F /
                                   (4 * cosTheta_i * cosTheta_o));
        } else {
            // transmit
            Float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
            if (cosTheta_i == 0 || cosTheta_o == 0) return SampledSpectrum(0.);

            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
            Vector3f wh = wo + wi * etap;
            CHECK_RARE(1e-6, LengthSquared(wh) == 0);
            if (LengthSquared(wh) == 0) return SampledSpectrum(0.);
            wh = FaceForward(Normalize(wh), Normal3f(0, 0, 1));

            Float F = FrDielectric(Dot(wo, wh), eta);

            Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
            Float factor = (mode == TransportMode::Radiance) ? Sqr(1 / etap) : 1;

            return SampledSpectrum((1 - F) * factor *
                                   std::abs(distrib.D(wh) * distrib.G(wo, wi) *
                                            AbsDot(wi, wh) * AbsDot(wo, wh) /
                                            (cosTheta_i * cosTheta_o * Sqr(sqrtDenom))));

        }
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (wo.z == 0) return {};

        if (!distrib) {
            Float F = FrDielectric(CosTheta(wo), eta);

            Float pr = F, pt = 1 - F;
            if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
            if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
            if (pr == 0 && pt == 0) return {};

            if (uc < pr / (pr + pt)) {
                // reflect
                Vector3f wi(-wo.x, -wo.y, wo.z);
                SampledSpectrum fr(F / AbsCosTheta(wi));
                return BSDFSample(fr, wi, pr / (pr + pt),
                                  BxDFFlags::SpecularReflection);
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

                // Account for non-symmetry with transmission to different medium
                if (mode == TransportMode::Radiance) ft /= Sqr(etap);
                return BSDFSample(ft, wi, pt / (pr + pt),
                                  BxDFFlags::SpecularTransmission);
            }
        } else {
            // TODO: sample wh first, then compute fresnel, then choose a lobe...
            // Need that random sample passed in...
            Float compPDF;
            Vector3f wh = distrib.Sample_wm(wo, u);
            Float F = FrDielectric(Dot(Reflect(wo, wh),
                                       FaceForward(wh, Vector3f(0, 0, 1))), eta);

            Float pr = F, pt = 1 - F;
            if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;
            if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;
            if (pr == 0 && pt == 0) return {};

            if (uc < pr / (pr + pt)) {
                // reflect
                // Sample microfacet orientation $\wh$ and reflected direction $\wi$
                Vector3f wi = Reflect(wo, wh);
                CHECK_RARE(1e-6, Dot(wo, wh) <= 0);
                if (!SameHemisphere(wo, wi) || Dot(wo, wh) <= 0) return {};

                // Compute PDF of _wi_ for microfacet reflection
                Float pdf = distrib.PDF(wo, wh) / (4 * Dot(wo, wh)) * pr / (pr + pt);
                CHECK(!std::isnan(pdf));

                // TODO: reuse fragments from f()
                Float cosTheta_o = AbsCosTheta(wo), cosTheta_i = AbsCosTheta(wi);
                // Handle degenerate cases for microfacet reflection
                if (cosTheta_i == 0 || cosTheta_o == 0) return {};
                SampledSpectrum f(distrib.D(wh) * distrib.G(wo, wi) * F /
                                  (4 * cosTheta_i * cosTheta_o));
                if (distrib.EffectivelySpecular())
                    return BSDFSample(f / pdf, wi, 1, BxDFFlags::SpecularReflection);
                else
                    return BSDFSample(f, wi, pdf, BxDFFlags::GlossyReflection);
            } else {
                // FIXME (make consistent): this etap is 1/etap as used in specular...
                Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
                Vector3f wi;
                bool tir = !Refract(wo, (Normal3f)wh, etap, &wi);
                CHECK_RARE(1e-6, tir);
                if (SameHemisphere(wo, wi)) return {};
                if (tir || wi.z == 0) return {};

                // Evaluate BSDF
                // TODO: share fragments with f(), PDF()...
                wh = FaceForward(wh, Normal3f(0, 0, 1));

                Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
                Float factor = (mode == TransportMode::Radiance) ? Sqr(1 / etap) : 1;

                SampledSpectrum f((1 - F) * factor *
                                  std::abs(distrib.D(wh) * distrib.G(wo, wi) *
                                           AbsDot(wi, wh) * AbsDot(wo, wh) /
                                           (AbsCosTheta(wi) * AbsCosTheta(wo) * Sqr(sqrtDenom))));

                // Compute PDF
                Float dwh_dwi = /*Sqr(etap) * */AbsDot(wi, wh) /
                    Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
                Float pdf = distrib.PDF(wo, wh) * dwh_dwi * pt / (pr + pt);
                CHECK(!std::isnan(pdf));

                //CO            LOG(WARNING) << "pt/(pr+pt) " << pt / (pr + pt);
                //CO            LOG(WARNING) << "Sample_f: (1-F) " << (1-F) << ", factor " << factor <<
                //CO                ", D " << distrib.D(wh) << ", G " << distrib.G(wo, wi) <<
                //CO                ", others " << (AbsDot(wi, wh) * AbsDot(wo, wh) /
                //CO                                (AbsCosTheta(wi) * AbsCosTheta(wo) * Sqr(sqrtDenom))) <<
                //CO                ", pdf " << pdf << ", f*cos/pdf " << f*AbsCosTheta(wi)/pdf;

                if (distrib.EffectivelySpecular())
                    return BSDFSample(f / pdf, wi, 1, BxDFFlags::SpecularTransmission);
                else
                    return BSDFSample(f, wi, pdf, BxDFFlags::GlossyTransmission);
            }
        }
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        if (!distrib || distrib.EffectivelySpecular()) return 0;

        if (SameHemisphere(wo, wi)) {
            if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return 0;

            Vector3f wh = wo + wi;
            CHECK_RARE(1e-6, LengthSquared(wh) == 0);
            CHECK_RARE(1e-6, Dot(wo, wh) < 0);
            if (LengthSquared(wh) == 0 || Dot(wo, wh) <= 0)
                return 0;

            wh = Normalize(wh);

            Float F = FrDielectric(Dot(wi, FaceForward(wh, Vector3f(0, 0, 1))), eta);
            CHECK_RARE(1e-6, F == 0);
            Float pr = F, pt = 1 - F;
            if (!(sampleFlags & BxDFReflTransFlags::Transmission)) pt = 0;

            return distrib.PDF(wo, wh) / (4 * Dot(wo, wh)) * pr / (pr + pt);
        } else {
            if (!(sampleFlags & BxDFReflTransFlags::Transmission)) return 0;
            // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
            Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
            Vector3f wh = wo + wi * etap;
            CHECK_RARE(1e-6, LengthSquared(wh) == 0);
            if (LengthSquared(wh) == 0) return 0;
            wh = Normalize(wh);

            Float F = FrDielectric(Dot(wo, FaceForward(wh, Normal3f(0, 0, 1))), eta);
            Float pr = F, pt = 1 - F;
            if (pt == 0) return 0;
            if (!(sampleFlags & BxDFReflTransFlags::Reflection)) pr = 0;

            // Compute change of variables _dwh\_dwi_ for microfacet transmission
            Float dwh_dwi = /*Sqr(etap) * */AbsDot(wi, wh) /
                Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
            CHECK_RARE(1e-6, (1 - F) == 0);
            return distrib.PDF(wo, wh) * dwh_dwi * pt / (pr + pt);
        }
    }

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
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        return SampledSpectrum(0);
    }

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
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const {
        return 0;
    }

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
    else {
        DCHECK_EQ(Tag(), TypeIndex<FresnelDielectric>());
        return Cast<FresnelDielectric>()->Evaluate(cosTheta_i);
    }
}

class BSSRDFAdapter {
  public:
    // BSSRDFAdapter Public Methods
    PBRT_HOST_DEVICE
    BSSRDFAdapter(Float eta, TransportMode mode)
        : flags(BxDFFlags::Reflection | BxDFFlags::Diffuse), eta(eta), mode(mode) {}

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        // Sw()
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        SampledSpectrum f((1 - FrDielectric(CosTheta(wi), eta)) /
                          (c * Pi));

        // Update BSSRDF transmission term to account for adjoint light
        // transport
        if (mode == TransportMode::Radiance)
            f *= Sqr(eta);
        return f;
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc,
                                        const Point2f &u, BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection))
            return {};

        // Cosine-sample the hemisphere, flipping the direction if necessary
        Vector3f wi = SampleCosineHemisphere(u);
        if (wo.z < 0) wi.z *= -1;
        return BSDFSample(f(wo, wi), wi, PDF(wo, wi, sampleFlags),
                          BxDFFlags::DiffuseReflection);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const {
        if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return 0;
        return SameHemisphere(wo, wi) ? AbsCosTheta(wi) * InvPi : 0;
    }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const { return flags; }

  private:
    BxDFFlags flags;
    Float eta;
    TransportMode mode;
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
    case TypeIndex<BSSRDFAdapter>():
        return Cast<BSSRDFAdapter>()->f(wo, wi);
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
    case TypeIndex<BSSRDFAdapter>():
        return Cast<BSSRDFAdapter>()->Sample_f(wo, uc, u, sampleFlags);
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
    case TypeIndex<BSSRDFAdapter>():
        return Cast<BSSRDFAdapter>()->PDF(wo, wi, sampleFlags);
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

template <typename RhoSampler>
inline SampledSpectrum BxDFHandle::rho(const Vector3f &wo, RhoSampler rhoSampler,
                                       int nSamples) const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->rho(wo, rhoSampler, nSamples);
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->rho(wo, rhoSampler, nSamples);

    case TypeIndex<CoatedDiffuseBxDF>():
    case TypeIndex<GeneralLayeredBxDF>():
    case TypeIndex<DielectricInterfaceBxDF>():
    case TypeIndex<ThinDielectricBxDF>():
    case TypeIndex<SpecularReflectionBxDF>():
    case TypeIndex<HairBxDF>():
    case TypeIndex<MeasuredBxDF>():
    case TypeIndex<MicrofacetReflectionBxDF>():
    case TypeIndex<MicrofacetTransmissionBxDF>():
    case TypeIndex<BSSRDFAdapter>(): {
        if (wo.z == 0) return SampledSpectrum(0.f);
        SampledSpectrum r(0.);
        for (int i = 0; i < nSamples; ++i) {
            RhoHemiDirSample sample = rhoSampler(i);
            auto bs = Sample_f(wo, sample.u, sample.u2);
            if (bs && bs->pdf > 0)
                r += bs->f * AbsCosTheta(bs->wi) / bs->pdf;
        }
        return r / nSamples;
    }
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

template <typename RhoSampler>
inline SampledSpectrum BxDFHandle::rho(RhoSampler rhoSampler, int nSamples) const {
    switch (Tag()) {
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->rho(rhoSampler, nSamples);
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->rho(rhoSampler, nSamples);

    case TypeIndex<CoatedDiffuseBxDF>():
    case TypeIndex<GeneralLayeredBxDF>():
    case TypeIndex<DielectricInterfaceBxDF>():
    case TypeIndex<ThinDielectricBxDF>():
    case TypeIndex<SpecularReflectionBxDF>():
    case TypeIndex<HairBxDF>():
    case TypeIndex<MeasuredBxDF>():
    case TypeIndex<MicrofacetReflectionBxDF>():
    case TypeIndex<MicrofacetTransmissionBxDF>():
    case TypeIndex<BSSRDFAdapter>(): {
        SampledSpectrum r(0.f);
        for (int i = 0; i < nSamples; ++i) {
            RhoHemiHemiSample sample = rhoSampler(i);

            Vector3f wo = SampleUniformHemisphere(sample.u2[0]);
            if (wo.z == 0) continue;
            Float pdfo = UniformHemispherePDF();

            auto bs = Sample_f(wo, sample.u, sample.u2[1]);
            if (bs && bs->pdf > 0)
                r += bs->f * AbsCosTheta(bs->wi) * AbsCosTheta(wo) /
                    (pdfo * bs->pdf);
        }
        return r / nSamples;
    }
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

extern template class LayeredBxDF<DielectricInterfaceBxDF, LambertianBxDF>;
extern template class LayeredBxDF<BxDFHandle, BxDFHandle>;

}  // namespace pbrt

#endif  // PBRT_BSDF_H
