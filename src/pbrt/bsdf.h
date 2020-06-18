// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_BSDF_H
#define PBRT_CORE_BSDF_H

// core/bsdf.h*
#include <pbrt/pbrt.h>

#include <pbrt/bxdfs.h>
#include <pbrt/interaction.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

class BSDF {
  public:
    // BSDF Public Methods
    PBRT_CPU_GPU
    BSDF(const SurfaceInteraction &si, BxDFHandle bxdf, Float eta = 1)
        : eta(Dot(si.wo, si.n) < 0 ? 1 / eta : eta),
          bxdf(bxdf),
          ng(si.n),
          shadingFrame(
              Frame::FromXZ(Normalize(si.shading.dpdu), Vector3f(si.shading.n))) {}

    PBRT_CPU_GPU
    bool IsNonSpecular() const {
        return (bxdf.Flags() & (BxDFFlags::Diffuse | BxDFFlags::Glossy));
    }
    PBRT_CPU_GPU
    bool IsDiffuse() const { return (bxdf.Flags() & BxDFFlags::Diffuse); }
    PBRT_CPU_GPU
    bool IsGlossy() const { return (bxdf.Flags() & BxDFFlags::Glossy); }
    PBRT_CPU_GPU
    bool IsSpecular() const { return (bxdf.Flags() & BxDFFlags::Specular); }
    PBRT_CPU_GPU
    bool HasReflection() const { return (bxdf.Flags() & BxDFFlags::Reflection); }
    PBRT_CPU_GPU
    bool HasTransmission() const { return (bxdf.Flags() & BxDFFlags::Transmission); }

    PBRT_CPU_GPU
    Vector3f WorldToLocal(const Vector3f &v) const { return shadingFrame.ToLocal(v); }
    PBRT_CPU_GPU
    Vector3f LocalToWorld(const Vector3f &v) const { return shadingFrame.FromLocal(v); }

    template <typename BxDF>
    PBRT_CPU_GPU SampledSpectrum f(const Vector3f &woW, const Vector3f &wiW,
                                   TransportMode mode = TransportMode::Radiance) const {
        Vector3f wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
        if (wo.z == 0)
            return SampledSpectrum(0.);

        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        return specificBxDF->f(wo, wi, mode) * GBump(woW, wiW, mode);
    }

    PBRT_CPU_GPU
    SampledSpectrum f(const Vector3f &woW, const Vector3f &wiW,
                      TransportMode mode = TransportMode::Radiance) const {
        Vector3f wi = WorldToLocal(wiW), wo = WorldToLocal(woW);
        if (wo.z == 0)
            return SampledSpectrum(0.);

        return bxdf.f(wo, wi, mode) * GBump(woW, wiW, mode);
    }

    PBRT_CPU_GPU
    SampledSpectrum rho(pstd::span<const Float> uc1, pstd::span<const Point2f> u1,
                        pstd::span<const Float> uc2, pstd::span<const Point2f> u2) const {
        return bxdf.rho(uc1, u1, uc2, u2);
    }

    PBRT_CPU_GPU
    SampledSpectrum rho(const Vector3f &woWorld, pstd::span<const Float> uc,
                        pstd::span<const Point2f> u) const {
        Vector3f wo = WorldToLocal(woWorld);
        return bxdf.rho(wo, uc, u);
    }

    template <typename BxDF>
    PBRT_CPU_GPU pstd::optional<BSDFSample> Sample_f(
        const Vector3f &woWorld, Float u, const Point2f &u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = WorldToLocal(woWorld);
        if (wo.z == 0)
            return {};

        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        if (!(specificBxDF->Flags() & sampleFlags))
            return {};

        pstd::optional<BSDFSample> bs =
            specificBxDF->Sample_f(wo, u, u2, mode, sampleFlags);
        if (!bs || bs->pdf == 0 || !bs->f)
            return {};
        CHECK_GT(bs->pdf, 0);

        VLOG(2, "For wo = %s, sampled f = %s, pdf = %f, ratio = %s, wi = %s", wo, bs->f,
             bs->pdf, (bs->pdf > 0) ? (bs->f / bs->pdf) : SampledSpectrum(0.), bs->wi);

        bs->wi = LocalToWorld(bs->wi);
        bs->f *= GBump(woWorld, bs->wi, mode);

        return bs;
    }

    PBRT_CPU_GPU
    pstd::optional<BSDFSample> Sample_f(
        const Vector3f &woWorld, Float u, const Point2f &u2,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = WorldToLocal(woWorld);
        if (wo.z == 0)
            return {};

        if (!(bxdf.Flags() & sampleFlags))
            return {};

        pstd::optional<BSDFSample> bs = bxdf.Sample_f(wo, u, u2, mode, sampleFlags);
        if (!bs || bs->pdf == 0 || !bs->f)
            return {};
        CHECK_GT(bs->pdf, 0);

        VLOG(2, "For wo = %s, sampled f = %s, pdf = %f, ratio = %s, wi = %s", wo, bs->f,
             bs->pdf, (bs->pdf > 0) ? (bs->f / bs->pdf) : SampledSpectrum(0.), bs->wi);

        bs->wi = LocalToWorld(bs->wi);
        bs->f *= GBump(woWorld, bs->wi, mode);

        return bs;
    }

    PBRT_CPU_GPU
    SampledSpectrum SampleSpecular_f(const Vector3f &wo, Vector3f *wi,
                                     BxDFReflTransFlags sampleFlags) const;

    template <typename BxDF>
    PBRT_CPU_GPU Float
    PDF(const Vector3f &woWorld, const Vector3f &wiWorld,
        TransportMode mode = TransportMode::Radiance,
        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = WorldToLocal(woWorld), wi = WorldToLocal(wiWorld);
        if (wo.z == 0)
            return 0.;
        const BxDF *specificBxDF = bxdf.Cast<BxDF>();
        return specificBxDF->PDF(wo, wi, mode, sampleFlags);
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &woWorld, const Vector3f &wiWorld,
              TransportMode mode = TransportMode::Radiance,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        Vector3f wo = WorldToLocal(woWorld), wi = WorldToLocal(wiWorld);
        if (wo.z == 0)
            return 0.;
        return bxdf.PDF(wo, wi, mode, sampleFlags);
    }

    std::string ToString() const;

    PBRT_CPU_GPU
    void Regularize(ScratchBuffer &scratchBuffer) {
        bxdf = bxdf.Regularize(scratchBuffer);
    }

    PBRT_CPU_GPU
    BxDFHandle GetBxDF() { return bxdf; }

    PBRT_CPU_GPU
    bool SampledPDFIsProportional() const { return bxdf.SampledPDFIsProportional(); }

    // BSDF Public Data
    const Float eta;

  private:
    PBRT_CPU_GPU
    Float GBump(const Vector3f &wo, const Vector3f &wi, TransportMode mode) const {
        return 1;  // disable for now...

        Vector3f w = (mode == TransportMode::Radiance) ? wi : wo;
        Normal3f ngf = FaceForward(ng, w);
        Normal3f nsf = FaceForward(Normal3f(shadingFrame.z), ngf);
        Float cosThetaIs = std::max<Float>(0, Dot(nsf, w)), cosThetaIg = Dot(ngf, w);
        Float cosThetaN = Dot(ngf, nsf);
        CHECK_GE(cosThetaIs, 0);
        CHECK_GE(cosThetaIg, 0);
        CHECK_GE(cosThetaN, 0);

        if (cosThetaIs == 0 || cosThetaIg == 0 || cosThetaN == 0)
            return 0;
        Float G = cosThetaIg / (cosThetaIs * cosThetaN);
        if (G >= 1)
            return 1;

        return -G * G * G + G * G + G;
    }

    // BSDF Private Data
    BxDFHandle bxdf;
    Frame shadingFrame;
    Normal3f ng;
    TransportMode mode;
};

}  // namespace pbrt

#endif  // PBRT_BSDF_H
