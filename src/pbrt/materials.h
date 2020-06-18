// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_H
#define PBRT_MATERIALS_H

// materials.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/bssrdf.h>
#include <pbrt/base/material.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>

#include <memory>

namespace pbrt {

struct BumpEvalContext {
    BumpEvalContext() = default;
    PBRT_CPU_GPU
    BumpEvalContext(const SurfaceInteraction &si)
        : p(si.p()),
          dpdu(si.shading.dpdu),
          dpdv(si.shading.dpdv),
          dpdx(si.dpdx),
          dpdy(si.dpdy),
          uv(si.uv),
          dudx(si.dudx),
          dvdx(si.dvdx),
          dudy(si.dudy),
          dvdy(si.dvdy),
          n(si.shading.n),
          dndu(si.shading.dndu),
          dndv(si.shading.dndv) {}
    PBRT_CPU_GPU
    operator TextureEvalContext() const {
        return TextureEvalContext(p, dpdx, dpdy, uv, dudx, dvdx, dudy, dvdy);
    }

    Point3f p;
    Vector3f dpdu, dpdv;
    Vector3f dpdx, dpdy;
    Point2f uv;
    Float dudx, dvdx, dudy, dvdy;
    Normal3f n;
    Normal3f dndu, dndv;
};

template <typename TextureEvaluator>
PBRT_CPU_GPU void Bump(TextureEvaluator texEval, FloatTextureHandle displacement,
                       const BumpEvalContext &ctx, Vector3f *dpdu, Vector3f *dpdv) {
    if (!displacement) {
        *dpdu = ctx.dpdu;
        *dpdv = ctx.dpdv;
        return;
    }

    CHECK(texEval.CanEvaluate({displacement}, {}));

    // Compute offset positions and evaluate displacement texture
    TextureEvalContext shiftedTexCtx = ctx;

    // Shift _shiftedTexCtx_ _du_ in the $u$ direction
    Float du = .5f * (std::abs(ctx.dudx) + std::abs(ctx.dudy));
    // The most common reason for du to be zero is for ray that start from
    // light sources, where no differentials are available. In this case,
    // we try to choose a small enough du so that we still get a decently
    // accurate bump value.
    if (du == 0)
        du = .0005f;
    shiftedTexCtx.p = ctx.p + du * ctx.dpdu;
    shiftedTexCtx.uv = ctx.uv + Vector2f(du, 0.f);
    Float uDisplace = texEval(displacement, shiftedTexCtx);

    // Shift _shiftedTexCtx_ _dv_ in the $v$ direction
    Float dv = .5f * (std::abs(ctx.dvdx) + std::abs(ctx.dvdy));
    if (dv == 0)
        dv = .0005f;
    shiftedTexCtx.p = ctx.p + dv * ctx.dpdv;
    shiftedTexCtx.uv = ctx.uv + Vector2f(0.f, dv);
    Float vDisplace = texEval(displacement, shiftedTexCtx);

    Float displace = texEval(displacement, ctx);

    // Compute bump-mapped differential geometry
    *dpdu = ctx.dpdu + (uDisplace - displace) / du * Vector3f(ctx.n) +
            displace * Vector3f(ctx.dndu);
    *dpdv = ctx.dpdv + (vDisplace - displace) / dv * Vector3f(ctx.n) +
            displace * Vector3f(ctx.dndv);
}

class MaterialBase {
public:
    template <typename TextureEvaluator>
    PBRT_CPU_GPU
    BSSRDFHandle GetBSSRDF(TextureEvaluator texEval, SurfaceInteraction &si,
                           const SampledWavelengths &lambda,
                           ScratchBuffer &scratchBuffer) const {
        return nullptr;
    }

    PBRT_CPU_GPU bool IsTransparent() const { return false; }
    PBRT_CPU_GPU bool HasSubsurfaceScattering() const { return false; }
};

// DielectricMaterial Declarations
class alignas(8) DielectricMaterial : public MaterialBase {
  public:
    // DielectricMaterial Public Methods
    DielectricMaterial(FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                       FloatTextureHandle etaF, SpectrumTextureHandle etaS,
                       FloatTextureHandle displacement, bool remapRoughness)
        : displacement(displacement),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          etaF(etaF),
          etaS(etaS),
          remapRoughness(remapRoughness) {
        CHECK((bool)etaF ^ (bool)etaS);
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({etaF, uRoughness, vRoughness}, {etaS}))
            return nullptr;

        // Compute index of refraction for glass
        Float eta;
        if (etaF)
            eta = texEval(etaF, si);
        else {
            eta = texEval(etaS, si, lambda)[0];
            lambda.TerminateSecondaryWavelengths();
        }

        Float urough = texEval(uRoughness, si), vrough = texEval(vRoughness, si);
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        MicrofacetDistributionHandle distrib =
            (urough != 0 && vrough != 0)
                ? scratchBuffer.Alloc<TrowbridgeReitzDistribution>(urough, vrough)
                : nullptr;

        // Initialize _bsdf_ for smooth or rough dielectric
        return scratchBuffer.Alloc<BSDF>(
            si, scratchBuffer.Alloc<DielectricInterfaceBxDF>(eta, distrib), eta);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static DielectricMaterial *Create(const TextureParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // DielectricMaterial Private Data
    FloatTextureHandle displacement;
    FloatTextureHandle uRoughness, vRoughness;
    FloatTextureHandle etaF;
    SpectrumTextureHandle etaS;
    bool remapRoughness;
};

// ThinDielectricMaterial Declarations
class alignas(8) ThinDielectricMaterial : public MaterialBase {
  public:
    // ThinDielectricMaterial Public Methods
    ThinDielectricMaterial(FloatTextureHandle etaF, SpectrumTextureHandle etaS,
                           FloatTextureHandle displacement)
        : displacement(displacement), etaF(etaF), etaS(etaS) {
        CHECK((bool)etaF ^ (bool)etaS);
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({etaF}, {etaS}))
            return nullptr;

        Float eta;
        if (etaF)
            eta = texEval(etaF, si);
        else {
            eta = texEval(etaS, si, lambda)[0];
            lambda.TerminateSecondaryWavelengths();
        }

        return scratchBuffer.Alloc<BSDF>(si, scratchBuffer.Alloc<ThinDielectricBxDF>(eta),
                                         eta);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    PBRT_CPU_GPU bool IsTransparent() const { return true; }

    static ThinDielectricMaterial *Create(const TextureParameterDictionary &parameters,
                                          const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // ThinDielectricMaterial Private Data
    FloatTextureHandle displacement;
    FloatTextureHandle etaF;
    SpectrumTextureHandle etaS;
};

// HairMaterial Declarations
class alignas(8) HairMaterial : public MaterialBase {
  public:
    // HairMaterial Public Methods
    HairMaterial(SpectrumTextureHandle sigma_a, SpectrumTextureHandle color,
                 FloatTextureHandle eumelanin, FloatTextureHandle pheomelanin,
                 FloatTextureHandle eta, FloatTextureHandle beta_m,
                 FloatTextureHandle beta_n, FloatTextureHandle alpha)
        : sigma_a(sigma_a),
          color(color),
          eumelanin(eumelanin),
          pheomelanin(pheomelanin),
          eta(eta),
          beta_m(beta_m),
          beta_n(beta_n),
          alpha(alpha) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({eumelanin, pheomelanin, eta, beta_m, beta_n, alpha},
                                 {sigma_a, color}))
            return nullptr;

        Float bm = std::max<Float>(1e-2, texEval(beta_m, si));
        Float bn = std::max<Float>(1e-2, texEval(beta_n, si));
        Float a = texEval(alpha, si);
        Float e = texEval(eta, si);

        SampledSpectrum sig_a;
        if (sigma_a)
            sig_a = ClampZero(texEval(sigma_a, si, lambda));
        else if (color) {
            SampledSpectrum c = Clamp(texEval(color, si, lambda), 0, 1);
            sig_a = HairBxDF::SigmaAFromReflectance(c, bn, lambda);
        } else {
            CHECK(eumelanin || pheomelanin);
            sig_a = HairBxDF::SigmaAFromConcentration(
                        std::max(Float(0), eumelanin ? texEval(eumelanin, si) : 0),
                        std::max(Float(0), pheomelanin ? texEval(pheomelanin, si) : 0))
                        .Sample(lambda);
        }

        // Offset along width
        Float h = -1 + 2 * si.uv[1];
        return scratchBuffer.Alloc<BSDF>(
            si, scratchBuffer.Alloc<HairBxDF>(h, e, sig_a, bm, bn, a), e);
    }

    static HairMaterial *Create(const TextureParameterDictionary &parameters,
                                const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return nullptr; }

    std::string ToString() const;

  private:
    // HairMaterial Private Data
    SpectrumTextureHandle sigma_a, color;
    FloatTextureHandle eumelanin, pheomelanin, eta;
    FloatTextureHandle beta_m, beta_n, alpha;
};

// DiffuseMaterial Declarations
class alignas(8) DiffuseMaterial : public MaterialBase {
  public:
    // DiffuseMaterial Public Methods
    DiffuseMaterial(SpectrumTextureHandle reflectance, FloatTextureHandle sigma,
                    FloatTextureHandle displacement)
        : displacement(displacement), reflectance(reflectance), sigma(sigma) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({sigma}, {reflectance}))
            return nullptr;

        // Evaluate textures for _DiffuseMaterial_ material and allocate BRDF
        SampledSpectrum r = Clamp(texEval(reflectance, si, lambda), 0, 1);
        Float sig = Clamp(texEval(sigma, si), 0, 90);
        DiffuseBxDF *bxdf = scratchBuffer.Alloc<DiffuseBxDF>(r, SampledSpectrum(0), sig);
        return scratchBuffer.Alloc<BSDF>(si, bxdf);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static DiffuseMaterial *Create(const TextureParameterDictionary &parameters,
                                   const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // DiffuseMaterial Private Data
    FloatTextureHandle displacement;
    SpectrumTextureHandle reflectance;
    FloatTextureHandle sigma;
};

// ConductorMaterial Declarations
class alignas(8) ConductorMaterial : public MaterialBase {
  public:
    // ConductorMaterial Public Methods
    ConductorMaterial(SpectrumTextureHandle eta, SpectrumTextureHandle k,
                      FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                      FloatTextureHandle displacement, bool remapRoughness)
        : displacement(displacement),
          eta(eta),
          k(k),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          remapRoughness(remapRoughness) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({uRoughness, vRoughness}, {eta, k}))
            return nullptr;

        Float uRough = texEval(uRoughness, si);
        Float vRough = texEval(vRoughness, si);
        if (remapRoughness) {
            uRough = TrowbridgeReitzDistribution::RoughnessToAlpha(uRough);
            vRough = TrowbridgeReitzDistribution::RoughnessToAlpha(vRough);
        }
        FresnelHandle frMf = scratchBuffer.Alloc<FresnelConductor>(
            texEval(eta, si, lambda), texEval(k, si, lambda));
        if (uRough == 0 || vRough == 0) {
            return scratchBuffer.Alloc<BSDF>(
                si, scratchBuffer.Alloc<SpecularReflectionBxDF>(frMf));
        } else {
            MicrofacetDistributionHandle distrib =
                scratchBuffer.Alloc<TrowbridgeReitzDistribution>(uRough, vRough);
            return scratchBuffer.Alloc<BSDF>(
                si, scratchBuffer.Alloc<MicrofacetReflectionBxDF>(distrib, frMf));
        }
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static ConductorMaterial *Create(const TextureParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // ConductorMaterial Private Data
    FloatTextureHandle displacement;
    SpectrumTextureHandle eta, k;
    FloatTextureHandle uRoughness, vRoughness;
    bool remapRoughness;
};

// CoatedDiffuseMaterial Declarations
class alignas(8) CoatedDiffuseMaterial : public MaterialBase {
  public:
    // CoatedDiffuseMaterial Public Methods
    CoatedDiffuseMaterial(SpectrumTextureHandle reflectance,
                          FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                          FloatTextureHandle thickness, FloatTextureHandle eta,
                          FloatTextureHandle displacement, bool remapRoughness,
                          LayeredBxDFConfig config)
        : displacement(displacement),
          reflectance(reflectance),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          thickness(thickness),
          eta(eta),
          remapRoughness(remapRoughness),
          config(config) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({uRoughness, vRoughness, thickness, eta}, {reflectance}))
            return nullptr;

        // Initialize diffuse component of plastic material
        SampledSpectrum r = Clamp(texEval(reflectance, si, lambda), 0, 1);

        // Create microfacet distribution _distrib_ for plastic material
        Float urough = texEval(uRoughness, si);
        Float vrough = texEval(vRoughness, si);
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        MicrofacetDistributionHandle distrib =
            scratchBuffer.Alloc<TrowbridgeReitzDistribution>(urough, vrough);

        Float thick = texEval(thickness, si);
        Float e = texEval(eta, si);

        BxDFHandle lb = scratchBuffer.Alloc<CoatedDiffuseBxDF>(
            DielectricInterfaceBxDF(e, distrib), DiffuseBxDF(r, SampledSpectrum(0), 0),
            thick, SampledSpectrum(0) /* albedo */, 0 /* g */, config);
        return scratchBuffer.Alloc<BSDF>(si, lb);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static CoatedDiffuseMaterial *Create(const TextureParameterDictionary &parameters,
                                         const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // CoatedDiffuseMaterial Private Data
    FloatTextureHandle displacement;
    SpectrumTextureHandle reflectance;
    FloatTextureHandle uRoughness, vRoughness, thickness, eta;
    bool remapRoughness;
    LayeredBxDFConfig config;
};

class alignas(8) LayeredMaterial : public MaterialBase {
  public:
    LayeredMaterial(MaterialHandle top, MaterialHandle base, FloatTextureHandle thickness,
                    SpectrumTextureHandle albedo, FloatTextureHandle g,
                    FloatTextureHandle displacement, LayeredBxDFConfig config)
        : displacement(displacement),
          top(top),
          base(base),
          thickness(thickness),
          albedo(albedo),
          g(g),
          config(config) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        BSDF *topBSDF = top.GetBSDF(texEval, si, lambda, scratchBuffer);
        BSDF *bottomBSDF = base.GetBSDF(texEval, si, lambda, scratchBuffer);
        if (!topBSDF || !bottomBSDF)
            return nullptr;

        if (!texEval.CanEvaluate({thickness, g}, {albedo}))
            return nullptr;

        Float thick = texEval(thickness, si);
        SampledSpectrum a = texEval(albedo, si, lambda);
        Float gg = texEval(g, si);

        BxDFHandle layered = scratchBuffer.Alloc<GeneralLayeredBxDF>(
            topBSDF->GetBxDF(), bottomBSDF->GetBxDF(), thick, a, gg, config);
        return scratchBuffer.Alloc<BSDF>(si, layered);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static LayeredMaterial *Create(const TextureParameterDictionary &parameters,
                                   MaterialHandle top, MaterialHandle base,
                                   const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    FloatTextureHandle displacement;
    MaterialHandle top, base;
    FloatTextureHandle thickness;
    SpectrumTextureHandle albedo;
    FloatTextureHandle g;
    LayeredBxDFConfig config;
};

// SubsurfaceMaterial Declarations
class alignas(8) SubsurfaceMaterial : public MaterialBase {
  public:
    // SubsurfaceMaterial Public Methods
    SubsurfaceMaterial(Float scale, SpectrumTextureHandle sigma_a,
                       SpectrumTextureHandle sigma_s, SpectrumTextureHandle reflectance,
                       SpectrumTextureHandle mfp, Float g, Float eta,
                       FloatTextureHandle uRoughness, FloatTextureHandle vRoughness,
                       FloatTextureHandle displacement, bool remapRoughness,
                       Allocator alloc)
        : displacement(displacement),
          scale(scale),
          sigma_a(sigma_a),
          sigma_s(sigma_s),
          reflectance(reflectance),
          mfp(mfp),
          uRoughness(uRoughness),
          vRoughness(vRoughness),
          eta(eta),
          remapRoughness(remapRoughness),
          table(100, 64, alloc) {
        ComputeBeamDiffusionBSSRDF(g, eta, &table);
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({uRoughness, vRoughness}, {}))
            return nullptr;

        // Initialize BSDF for _SubsurfaceMaterial_

        Float urough = texEval(uRoughness, si), vrough = texEval(vRoughness, si);
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        MicrofacetDistributionHandle distrib =
            (urough != 0 && vrough != 0)
                ? scratchBuffer.Alloc<TrowbridgeReitzDistribution>(urough, vrough)
                : nullptr;

        // Initialize _bsdf_ for smooth or rough dielectric
        return scratchBuffer.Alloc<BSDF>(
            si, scratchBuffer.Alloc<DielectricInterfaceBxDF>(eta, distrib), eta);
    }

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSSRDFHandle GetBSSRDF(TextureEvaluator texEval, SurfaceInteraction &si,
                                        const SampledWavelengths &lambda,
                                        ScratchBuffer &scratchBuffer) const {
        SampledSpectrum sig_a, sig_s;
        if (sigma_a && sigma_s) {
            if (!texEval.CanEvaluate({}, {sigma_a, sigma_s}))
                return nullptr;

            sig_a = ClampZero(scale * texEval(sigma_a, si, lambda));
            sig_s = ClampZero(scale * texEval(sigma_s, si, lambda));
        } else {
            DCHECK(reflectance && mfp);
            if (!texEval.CanEvaluate({}, {mfp, reflectance}))
                return nullptr;

            SampledSpectrum mfree = ClampZero(scale * texEval(mfp, si, lambda));
            SampledSpectrum r = Clamp(texEval(reflectance, si, lambda), 0, 1);
            SubsurfaceFromDiffuse(table, r, mfree, &sig_a, &sig_s);
        }

        return scratchBuffer.Alloc<TabulatedBSSRDF>(si, eta, sig_a, sig_s, table);
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    PBRT_CPU_GPU bool HasSubsurfaceScattering() const { return true; }

    static SubsurfaceMaterial *Create(const TextureParameterDictionary &parameters,
                                      const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    // SubsurfaceMaterial Private Data
    FloatTextureHandle displacement;
    Float scale;
    SpectrumTextureHandle sigma_a, sigma_s, reflectance, mfp;
    FloatTextureHandle uRoughness, vRoughness;
    Float eta;
    bool remapRoughness;
    BSSRDFTable table;
};

// DiffuseTransmissionMaterial Declarations
class alignas(8) DiffuseTransmissionMaterial : public MaterialBase {
  public:
    // DiffuseTransmissionMaterial Public Methods
    DiffuseTransmissionMaterial(SpectrumTextureHandle reflectance,
                                SpectrumTextureHandle transmittance,
                                FloatTextureHandle sigma, FloatTextureHandle displacement,
                                Float scale)
        : displacement(displacement),
          reflectance(reflectance),
          transmittance(transmittance),
          sigma(sigma),
          scale(scale) {}

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        if (!texEval.CanEvaluate({sigma}, {reflectance, transmittance}))
            return nullptr;

        SampledSpectrum r = Clamp(scale * texEval(reflectance, si, lambda), 0, 1);
        SampledSpectrum t = Clamp(scale * texEval(transmittance, si, lambda), 0, 1);
        Float s = texEval(sigma, si);
        return scratchBuffer.Alloc<BSDF>(si, scratchBuffer.Alloc<DiffuseBxDF>(r, t, s));
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static DiffuseTransmissionMaterial *Create(
        const TextureParameterDictionary &parameters, const FileLoc *loc,
        Allocator alloc);

    std::string ToString() const;

  private:
    // DiffuseTransmissionMaterial Private Data
    FloatTextureHandle displacement;
    SpectrumTextureHandle reflectance, transmittance;
    FloatTextureHandle sigma;
    Float scale;
};

class alignas(8) MeasuredMaterial : public MaterialBase {
  public:
    MeasuredMaterial(const std::string &filename, FloatTextureHandle displacement,
                     Allocator alloc);

    template <typename TextureEvaluator>
    PBRT_CPU_GPU BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                               const SampledWavelengths &lambda,
                               ScratchBuffer &scratchBuffer) const {
        return scratchBuffer.Alloc<BSDF>(
            si, scratchBuffer.Alloc<MeasuredBxDF>(brdfData, lambda));
    }

    PBRT_CPU_GPU
    FloatTextureHandle GetDisplacement() const { return displacement; }

    static MeasuredMaterial *Create(const TextureParameterDictionary &parameters,
                                    const FileLoc *loc, Allocator alloc);

    std::string ToString() const;

  private:
    FloatTextureHandle displacement;
    const MeasuredBRDFData *brdfData;
};

template <typename TextureEvaluator>
inline BSDF *MaterialHandle::GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                                     const SampledWavelengths &lambda,
                                     ScratchBuffer &scratchBuffer) const {
    auto get = [&](auto ptr) { return ptr->GetBSDF(texEval, si, lambda, scratchBuffer); };
    return Apply<BSDF *>(get);
}

template <typename TextureEvaluator>
inline BSSRDFHandle MaterialHandle::GetBSSRDF(TextureEvaluator texEval,
                                              SurfaceInteraction &si,
                                              const SampledWavelengths &lambda,
                                              ScratchBuffer &scratchBuffer) const {
    auto get = [&](auto ptr) { return ptr->GetBSSRDF(texEval, si, lambda, scratchBuffer); };
    return Apply<BSSRDFHandle>(get);
}

inline bool MaterialHandle::IsTransparent() const {
    auto transp = [&](auto ptr) { return ptr->IsTransparent(); };
    return Apply<bool>(transp);
}

inline bool MaterialHandle::HasSubsurfaceScattering() const {
    auto has = [&](auto ptr) { return ptr->HasSubsurfaceScattering(); };
    return Apply<bool>(has);
}

inline FloatTextureHandle MaterialHandle::GetDisplacement() const {
    auto disp = [&](auto ptr) { return ptr->GetDisplacement(); };
    return Apply<FloatTextureHandle>(disp);
}

}  // namespace pbrt

#endif  // PBRT_MATERIALS_UBER_H
