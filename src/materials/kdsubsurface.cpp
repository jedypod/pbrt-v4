
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


// materials/kdsubsurface.cpp*
#include "materials/kdsubsurface.h"

#include "textures/constant.h"
#include "memory.h"
#include "spectrum.h"
#include "microfacet.h"
#include "texture.h"
#include "paramset.h"
#include "interaction.h"

namespace pbrt {

// KdSubsurfaceMaterial Method Definitions
void KdSubsurfaceMaterial::ComputeScatteringFunctions(
    SurfaceInteraction *si, MemoryArena &arena, TransportMode mode) const {
    // Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(*bumpMap, si);
    Spectrum R = Kr->Evaluate(*si).Clamp();
    Spectrum T = Kt->Evaluate(*si).Clamp();
    Float urough = uRoughness->Evaluate(*si);
    Float vrough = vRoughness->Evaluate(*si);

    // Initialize _bsdf_ for smooth or rough dielectric
    si->bsdf = arena.Alloc<BSDF>(*si, eta);

    if (R.IsBlack() && T.IsBlack()) return;

    bool isSpecular = urough == 0 && vrough == 0;
    if (isSpecular) {
        si->bsdf->Add(arena.Alloc<FresnelSpecular>(R, T, 1.f, eta, mode));
    } else {
        if (remapRoughness) {
            urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
            vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
        }
        MicrofacetDistribution *distrib =
            isSpecular ? nullptr
            : arena.Alloc<TrowbridgeReitzDistribution>(urough, vrough);
        if (!R.IsBlack()) {
            Fresnel *fresnel = arena.Alloc<FresnelDielectric>(1.f, eta);
            si->bsdf->Add(arena.Alloc<MicrofacetReflection>(R, distrib, fresnel));
        }
        if (!T.IsBlack()) {
            si->bsdf->Add(arena.Alloc<MicrofacetTransmission>(T, distrib, 1.f, eta, mode));
        }
    }

    Spectrum mfree = scale * mfp->Evaluate(*si).Clamp();
    Spectrum kd = Kd->Evaluate(*si).Clamp();
    Spectrum sig_a, sig_s;
    SubsurfaceFromDiffuse(table, kd, mfree, &sig_a, &sig_s);
    si->bssrdf = arena.Alloc<TabulatedBSSRDF>(*si, this, mode, eta,
                                              sig_a, sig_s, table);
}

std::shared_ptr<KdSubsurfaceMaterial> CreateKdSubsurfaceMaterial(
    const TextureParams &mp) {
    Float Kd[3] = {.5, .5, .5};
    std::shared_ptr<Texture<Spectrum>> kd =
        mp.GetSpectrumTexture("Kd", Spectrum::FromRGB(Kd));
    std::shared_ptr<Texture<Spectrum>> mfp =
        mp.GetSpectrumTexture("mfp", Spectrum(1.f));
    std::shared_ptr<Texture<Spectrum>> kr =
        mp.GetSpectrumTexture("Kr", Spectrum(1.f));
    std::shared_ptr<Texture<Spectrum>> kt =
        mp.GetSpectrumTexture("Kt", Spectrum(1.f));
    std::shared_ptr<Texture<Float>> roughu =
        mp.GetFloatTexture("uroughness", 0.f);
    std::shared_ptr<Texture<Float>> roughv =
        mp.GetFloatTexture("vroughness", 0.f);
    std::shared_ptr<Texture<Float>> bumpMap =
        mp.GetFloatTextureOrNull("bumpmap");
    Float eta = mp.FindOneFloat("eta", 1.33f);
    Float scale = mp.FindOneFloat("scale", 1.0f);
    Float g = mp.FindOneFloat("g", 0.0f);
    bool remapRoughness = mp.FindOneBool("remaproughness", true);
    return std::make_shared<KdSubsurfaceMaterial>(scale, kd, kr, kt, mfp, g,
                                                  eta, roughu, roughv, bumpMap,
                                                  remapRoughness);
}

}  // namespace pbrt
