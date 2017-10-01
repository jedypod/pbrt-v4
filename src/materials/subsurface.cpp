
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


// materials/subsurface.cpp*
#include "materials/subsurface.h"

#include "textures/constant.h"
#include "error.h"
#include "util/memory.h"
#include "microfacet.h"
#include "spectrum.h"
#include "texture.h"
#include "util/interpolation.h"
#include "paramset.h"
#include "interaction.h"

namespace pbrt {

// SubsurfaceMaterial Method Definitions
void SubsurfaceMaterial::ComputeScatteringFunctions(
    SurfaceInteraction *si, MemoryArena &arena, TransportMode mode) const {
    // Perform bump mapping with _bumpMap_, if present
    if (bumpMap) Bump(*bumpMap, si);

    // Initialize BSDF for _SubsurfaceMaterial_
    Spectrum R = Kr->Evaluate(*si).Clamp();
    Spectrum T = Kt->Evaluate(*si).Clamp();
    Float urough = uRoughness->Evaluate(*si);
    Float vrough = vRoughness->Evaluate(*si);

    // Initialize _bsdf_ for smooth or rough dielectric
    si->bsdf = arena.Alloc<BSDF>(*si, eta);

    if (!R && !T) return;

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
        if (R) {
            Fresnel *fresnel = arena.Alloc<FresnelDielectric>(1.f, eta);
            si->bsdf->Add(arena.Alloc<MicrofacetReflection>(R, distrib, fresnel));
        }
        if (T) {
            si->bsdf->Add(arena.Alloc<MicrofacetTransmission>(T, distrib, 1.f, eta, mode));
        }
    }
    Spectrum sig_a = scale * sigma_a->Evaluate(*si).Clamp();
    Spectrum sig_s = scale * sigma_s->Evaluate(*si).Clamp();
    si->bssrdf = arena.Alloc<TabulatedBSSRDF>(*si, this, mode, eta,
                                              sig_a, sig_s, table);
}

std::shared_ptr<SubsurfaceMaterial> CreateSubsurfaceMaterial(
    const TextureParams &mp, const std::shared_ptr<const ParamSet> &attributes) {
    Float sig_a_rgb[3] = {.0011f, .0024f, .014f},
          sig_s_rgb[3] = {2.55f, 3.21f, 3.77f};
    Spectrum sig_a = Spectrum::FromRGB(sig_a_rgb),
             sig_s = Spectrum::FromRGB(sig_s_rgb);
    std::string name = mp.GetOneString("name", "");
    bool found = GetMediumScatteringProperties(name, &sig_a, &sig_s);
    Float g = mp.GetOneFloat("g", 0.0f);
    if (name != "") {
        if (!found)
            Warning("Named material \"%s\" not found.  Using defaults.",
                    name.c_str());
        else
            g = 0; /* Enforce g=0 (the database specifies reduced scattering
                      coefficients) */
    }
    Float scale = mp.GetOneFloat("scale", 1.f);
    Float eta = mp.GetOneFloat("eta", 1.33f);

    std::shared_ptr<Texture<Spectrum>> sigma_a, sigma_s;
    sigma_a = mp.GetSpectrumTexture("sigma_a", sig_a);
    sigma_s = mp.GetSpectrumTexture("sigma_s", sig_s);
    std::shared_ptr<Texture<Spectrum>> Kr =
        mp.GetSpectrumTexture("Kr", Spectrum(1.f));
    std::shared_ptr<Texture<Spectrum>> Kt =
        mp.GetSpectrumTexture("Kt", Spectrum(1.f));
    std::shared_ptr<Texture<Float>> roughu =
        mp.GetFloatTexture("uroughness", 0.f);
    std::shared_ptr<Texture<Float>> roughv =
        mp.GetFloatTexture("vroughness", 0.f);
    std::shared_ptr<Texture<Float>> bumpMap =
        mp.GetFloatTextureOrNull("bumpmap");
    bool remapRoughness = mp.GetOneBool("remaproughness", true);
    return std::make_shared<SubsurfaceMaterial>(scale, Kr, Kt, sigma_a, sigma_s,
                                                g, eta, roughu, roughv, bumpMap,
                                                remapRoughness, attributes);
}

}  // namespace pbrt
