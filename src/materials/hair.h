
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

/*

See the writeup "The Implementation of a Hair Scattering Model" at
http://pbrt.org/hair.pdf for a description of the implementation here.

*/

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MATERIALS_HAIR_H
#define PBRT_MATERIALS_HAIR_H

// materials/hair.h*
#include "pbrt.h"

#include "material.h"
#include "reflection.h"

#include <array>
#include <memory>

namespace pbrt {

// HairMaterial Declarations
class HairMaterial : public Material {
  public:
    // HairMaterial Public Methods
    HairMaterial(const std::shared_ptr<Texture<Spectrum>> &sigma_a,
                 const std::shared_ptr<Texture<Spectrum>> &color,
                 const std::shared_ptr<Texture<Float>> &eumelanin,
                 const std::shared_ptr<Texture<Float>> &pheomelanin,
                 const std::shared_ptr<Texture<Float>> &eta,
                 const std::shared_ptr<Texture<Float>> &beta_m,
                 const std::shared_ptr<Texture<Float>> &beta_n,
                 const std::shared_ptr<Texture<Float>> &alpha,
                 const std::shared_ptr<const ParamSet> &attributes)
        : Material(attributes),
          sigma_a(sigma_a),
          color(color),
          eumelanin(eumelanin),
          pheomelanin(pheomelanin),
          eta(eta),
          beta_m(beta_m),
          beta_n(beta_n),
          alpha(alpha) {}
    void ComputeScatteringFunctions(SurfaceInteraction *si, MemoryArena &arena,
                                    TransportMode mode) const;

  private:
    // HairMaterial Private Data
    std::shared_ptr<Texture<Spectrum>> sigma_a, color;
    std::shared_ptr<Texture<Float>> eumelanin, pheomelanin, eta;
    std::shared_ptr<Texture<Float>> beta_m, beta_n, alpha;
};

std::shared_ptr<HairMaterial> CreateHairMaterial(
    const TextureParams &mp, const std::shared_ptr<const ParamSet> &attributes);

// HairBSDF Constants
static const int pMax = 3;
static const Float SqrtPiOver8 = 0.626657069f;

// HairBSDF Declarations
class HairBSDF : public BxDF {
  public:
    // HairBSDF Public Methods
    HairBSDF(Float h, Float eta, const Spectrum &sigma_a, Float beta_m,
             Float beta_n, Float alpha);
    Spectrum f(const Vector3f &wo, const Vector3f &wi) const;
    Spectrum Sample_f(const Vector3f &wo, Vector3f *wi, const Point2f &u,
                      Float *pdf, BxDFType *sampledType) const;
    Float Pdf(const Vector3f &wo, const Vector3f &wi) const;
    std::string ToString() const;
    static Spectrum SigmaAFromConcentration(Float ce, Float cp);
    static Spectrum SigmaAFromReflectance(const Spectrum &c, Float beta_n);

  private:
    // HairBSDF Private Methods
    std::array<Float, pMax + 1> ComputeApPdf(Float cosThetaO) const;

    // HairBSDF Private Data
    const Float h, gammaO, eta;
    const Spectrum sigma_a;
    const Float beta_m, beta_n;
    Float v[pMax + 1];
    Float s;
    Float sin2kAlpha[3], cos2kAlpha[3];
};

}  // namespace pbrt

#endif  // PBRT_MATERIALS_HAIR_H
