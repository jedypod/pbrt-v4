
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

#ifndef PBRT_CORE_BSSRDF_H
#define PBRT_CORE_BSSRDF_H

// core/bssrdf.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/util/check.h>
#include <pbrt/interaction.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// BSSRDF Declarations
struct BSSRDFSample {
    SampledSpectrum S;
    SurfaceInteraction si;
    Float pdf;
};

class BSSRDF {
  public:
    // BSSRDF Public Methods
    PBRT_HOST_DEVICE
    BSSRDF(const SurfaceInteraction &po, Float eta) : po(po), eta(eta) {}
    virtual ~BSSRDF() {}

    // BSSRDF Interface
    PBRT_HOST_DEVICE
    virtual SampledSpectrum S(const SurfaceInteraction &pi, const Vector3f &wi) = 0;
    PBRT_HOST_DEVICE
    virtual pstd::optional<BSSRDFSample> Sample_S(
        const Scene &scene, Float u1, const Point2f &u2, MaterialBuffer &materialBuffer) const = 0;

    virtual std::string ToString() const = 0;

  protected:
    // BSSRDF Protected Data
    const SurfaceInteraction &po;
    Float eta;
};

class SeparableBSSRDF : public BSSRDF {
    friend class SeparableBSSRDFAdapter;

  public:
    // SeparableBSSRDF Public Methods
    PBRT_HOST_DEVICE
    SeparableBSSRDF(const SurfaceInteraction &po, Float eta,
                    MaterialHandle material, TransportMode mode)
        : BSSRDF(po, eta),
          ns(po.shading.n),
          ss(Normalize(po.shading.dpdu)),
          ts(Cross(ns, ss)),
          material(material),
          mode(mode) {}

    PBRT_HOST_DEVICE
    SampledSpectrum S(const SurfaceInteraction &pi, const Vector3f &wi) {
        ProfilerScope pp(ProfilePhase::BSSRDFEvaluation);
        Float Ft = FrDielectric(CosTheta(po.wo), eta);
        return (1 - Ft) * Sp(pi) * Sw(wi);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sw(const Vector3f &w) const {
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        return SampledSpectrum((1 - FrDielectric(CosTheta(w), eta)) /
                               (c * Pi));
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sp(const SurfaceInteraction &pi) const {
        return Sr(Distance(po.p(), pi.p()));
    }
    PBRT_HOST_DEVICE
    pstd::optional<BSSRDFSample> Sample_S(const Scene &scene, Float u1, const Point2f &u2,
                                          MaterialBuffer &materialBuffer) const;
    PBRT_HOST_DEVICE
    pstd::optional<BSSRDFSample> Sample_Sp(const Scene &scene, Float u1, const Point2f &u2,
                                           MaterialBuffer &materialBuffer) const;
    PBRT_HOST_DEVICE
    Float Pdf_Sp(const SurfaceInteraction &pi) const;

    // SeparableBSSRDF Interface
    PBRT_HOST_DEVICE
    virtual SampledSpectrum Sr(Float d) const = 0;
    PBRT_HOST_DEVICE
    virtual Float Sample_Sr(int ch, Float u) const = 0;
    PBRT_HOST_DEVICE
    virtual Float Pdf_Sr(int ch, Float r) const = 0;

    std::string ToString() const;

  private:
    // SeparableBSSRDF Private Data
    Normal3f ns;
    Vector3f ss, ts;
    MaterialHandle material;
    TransportMode mode;
};

Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r);
Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r);
void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t);

class TabulatedBSSRDF : public SeparableBSSRDF {
  public:
    // TabulatedBSSRDF Public Methods
    PBRT_HOST_DEVICE
    TabulatedBSSRDF(const SurfaceInteraction &po, MaterialHandle material,
                    TransportMode mode, Float eta, const SampledSpectrum &sigma_a,
                    const SampledSpectrum &sigma_s, const BSSRDFTable &table)
        : SeparableBSSRDF(po, eta, material, mode), table(table) {
        sigma_t = sigma_a + sigma_s;
        rho = SafeDiv(sigma_s, sigma_t);
    }
    PBRT_HOST_DEVICE
    SampledSpectrum Sr(Float distance) const;
    PBRT_HOST_DEVICE
    Float Pdf_Sr(int ch, Float distance) const;
    PBRT_HOST_DEVICE
    Float Sample_Sr(int ch, Float sample) const;

    std::string ToString() const;

  private:
    // TabulatedBSSRDF Private Data
    const BSSRDFTable &table;
    SampledSpectrum sigma_t, rho;
};

struct BSSRDFTable {
    // BSSRDFTable Public Data
    pstd::vector<Float> rhoSamples, radiusSamples;
    pstd::vector<Float> profile;
    pstd::vector<Float> rhoEff;
    pstd::vector<Float> profileCDF;

    // BSSRDFTable Public Methods
    BSSRDFTable(int nRhoSamples, int nRadiusSamples);
    inline Float EvalProfile(int rhoIndex, int radiusIndex) const {
        CHECK(rhoIndex >= 0 && rhoIndex < rhoSamples.size());
        CHECK(radiusIndex >= 0 && radiusIndex < radiusSamples.size());
        return profile[rhoIndex * radiusSamples.size() + radiusIndex];
    }

    std::string ToString() const;
};

void SubsurfaceFromDiffuse(const BSSRDFTable &table, const SampledSpectrum &rhoEff,
                           const SampledSpectrum &mfp, SampledSpectrum *sigma_a,
                           SampledSpectrum *sigma_s);


///////////////////////////////////////////////////////////////////////////
// DisneyBSSRDF

// Implementation of the empirical BSSRDF described in "Extending the
// Disney BRDF to a BSDF with integrated subsurface scattering" (Brent
// Burley) and "Approximate Reflectance Profiles for Efficient Subsurface
// Scattering (Christensen and Burley).
class DisneyBSSRDF : public SeparableBSSRDF {
  public:
    PBRT_HOST_DEVICE
    DisneyBSSRDF(const SampledSpectrum &R, const SampledSpectrum &d,
                 const SurfaceInteraction &po, Float eta,
                 MaterialHandle material, TransportMode mode)
        // 0.2 factor comes from personal communication from Brent Burley
        // and Matt Chiang.
        : SeparableBSSRDF(po, eta, material, mode), R(R), d(0.2 * d) {}

    PBRT_HOST_DEVICE
    SampledSpectrum S(const SurfaceInteraction &pi, const Vector3f &wi);
    PBRT_HOST_DEVICE
    SampledSpectrum Sr(Float d) const;
    PBRT_HOST_DEVICE
    Float Sample_Sr(int ch, Float u) const;
    PBRT_HOST_DEVICE
    Float Pdf_Sr(int ch, Float r) const;

    std::string ToString() const;

  private:
    SampledSpectrum R, d;
};

}  // namespace pbrt

#endif  // PBRT_BSSRDF_H
