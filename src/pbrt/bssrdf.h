
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
#include <pbrt/interaction.h>
#include <pbrt/util/check.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// BSSRDF Declarations
struct BSSRDFSample {
    SampledSpectrum S;
    SurfaceInteraction si;
    Float pdf;
};

Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r);
Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r);
void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t);

struct BSSRDFTable {
    // BSSRDFTable Public Data
    pstd::vector<Float> rhoSamples, radiusSamples;
    pstd::vector<Float> profile;
    pstd::vector<Float> rhoEff;
    pstd::vector<Float> profileCDF;

    // BSSRDFTable Public Methods
    BSSRDFTable(int nRhoSamples, int nRadiusSamples, Allocator alloc);

    PBRT_HOST_DEVICE
    Float EvalProfile(int rhoIndex, int radiusIndex) const {
        CHECK(rhoIndex >= 0 && rhoIndex < rhoSamples.size());
        CHECK(radiusIndex >= 0 && radiusIndex < radiusSamples.size());
        return profile[rhoIndex * radiusSamples.size() + radiusIndex];
    }

    std::string ToString() const;
};

struct BSSRDFProbeSegment {
    Point3f p0, p1;
    Float time;
};

class TabulatedBSSRDF {
  public:
    // TabulatedBSSRDF Public Methods
    PBRT_HOST_DEVICE
    TabulatedBSSRDF(const SurfaceInteraction &po, Float eta, TransportMode mode,
                    const SampledSpectrum &sigma_a,
                    const SampledSpectrum &sigma_s, const BSSRDFTable &table)
        : po(po),
          eta(eta),
          ns(po.shading.n),
          ss(Normalize(po.shading.dpdu)),
          ts(Cross(ns, ss)),
          mode(mode),
          table(table) {
        sigma_t = sigma_a + sigma_s;
        rho = SafeDiv(sigma_s, sigma_t);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum S(const SurfaceInteraction &pi, const Vector3f &wi) {
        ProfilerScope pp(ProfilePhase::BSSRDFEvaluation);
        Float Ft = FrDielectric(CosTheta(po.wo), eta);
        return (1 - Ft) * Sp(pi) * Sw(wi);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sp(const SurfaceInteraction &pi) const {
        return Sr(Distance(po.p(), pi.p()));
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sw(const Vector3f &w) const {
        Float c = 1 - 2 * FresnelMoment1(1 / eta);
        return SampledSpectrum((1 - FrDielectric(CosTheta(w), eta)) /
                               (c * Pi));
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSSRDFProbeSegment> Sample(Float u1, const Point2f &u2) const {
        ProfilerScope pp(ProfilePhase::BSSRDFSampling);

        // Choose projection axis for BSSRDF sampling
        Vector3f vx, vy, vz;
        switch (SampleDiscrete({0.5, .25, .25}, u1, nullptr, &u1)) {
        case 0:
            vx = ss;
            vy = ts;
            vz = Vector3f(ns);
            break;
        case 1:
            // Prepare for sampling rays with respect to _ss_
            vx = ts;
            vy = Vector3f(ns);
            vz = ss;
            break;
        case 2:
            // Prepare for sampling rays with respect to _ts_
            vx = Vector3f(ns);
            vy = ss;
            vz = ts;
            break;
        default:
            LOG_FATAL("Unexpected value returned from SampleDiscrete");
        }

        // Choose spectral channel for BSSRDF sampling
        int ch = std::min<int>(u1 * NSpectrumSamples, NSpectrumSamples - 1);
        u1 = std::min(u1 * NSpectrumSamples - ch, OneMinusEpsilon);

        // Sample BSSRDF profile in polar coordinates
        Float r = Sample_Sr(ch, u2[0]);
        if (r < 0) return {};
        Float phi = 2 * Pi * u2[1];

        // Compute BSSRDF profile bounds and intersection height
        Float rMax = Sample_Sr(ch, 0.999f);
        if (r >= rMax) return {};
        Float l = 2 * std::sqrt(rMax * rMax - r * r);

        // Compute BSSRDF sampling ray segment
        Point3f pStart = po.p() + r * (vx * std::cos(phi) + vy * std::sin(phi)) - l * vz * 0.5f;
        Point3f pTarget = pStart + l * vz;
        return BSSRDFProbeSegment{pStart, pTarget, po.time};
    }

    PBRT_HOST_DEVICE
    BSSRDFSample ProbeIntersectionToSample(const SurfaceInteraction &si,
                                           MaterialBuffer &materialBuffer) const {
        BSSRDFSample bs{Sp(si), si, Pdf_Sp(si)};

        bs.si.bsdf = materialBuffer.Alloc<BSDF>(bs.si, materialBuffer.Alloc<BSSRDFAdapter>(eta, mode));
        bs.si.wo = Vector3f(bs.si.shading.n);
        return bs;
    }

    PBRT_HOST_DEVICE
    Float Pdf_Sp(const SurfaceInteraction &pi) const {
        // Express $\pti-\pto$ and $\bold{n}_i$ with respect to local coordinates at
        // $\pto$
        Vector3f d = pi.p() - po.p();
        Vector3f dLocal(Dot(ss, d), Dot(ts, d), Dot(ns, d));
        Normal3f nLocal(Dot(ss, pi.n), Dot(ts, pi.n), Dot(ns, pi.n));

        // Compute BSSRDF profile radius under projection along each axis
        Float rProj[3] = {std::sqrt(dLocal.y * dLocal.y + dLocal.z * dLocal.z),
                          std::sqrt(dLocal.z * dLocal.z + dLocal.x * dLocal.x),
                          std::sqrt(dLocal.x * dLocal.x + dLocal.y * dLocal.y)};

        // Return combined probability from all BSSRDF sampling strategies
        Float pdf = 0, axisProb[3] = {.25f, .25f, .5f};
        Float chProb = 1 / (Float)NSpectrumSamples;
        for (int axis = 0; axis < 3; ++axis)
            for (int ch = 0; ch < NSpectrumSamples; ++ch)
                pdf += Pdf_Sr(ch, rProj[axis]) * std::abs(nLocal[axis]) * chProb *
                    axisProb[axis];
        return pdf;
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sr(Float r) const {
        SampledSpectrum Sr(0.f);
        for (int ch = 0; ch < NSpectrumSamples; ++ch) {
            // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
            Float rOptical = r * sigma_t[ch];

            // Compute spline weights to interpolate BSSRDF on channel _ch_
            int rhoOffset, radiusOffset;
            Float rhoWeights[4], radiusWeights[4];
            if (!CatmullRomWeights(table.rhoSamples, rho[ch], &rhoOffset,
                                   rhoWeights) ||
                !CatmullRomWeights(table.radiusSamples, rOptical, &radiusOffset,
                                   radiusWeights))
                continue;

            // Set BSSRDF value _Sr[ch]_ using tensor spline interpolation
            Float sr = 0;
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Float weight = rhoWeights[i] * radiusWeights[j];
                    if (weight != 0)
                        sr += weight *
                            table.EvalProfile(rhoOffset + i, radiusOffset + j);
                }
            }

            // Cancel marginal PDF factor from tabulated BSSRDF profile
            if (rOptical != 0) sr /= 2 * Pi * rOptical;
            Sr[ch] = sr;
        }
        // Transform BSSRDF value into world space units
        Sr *= sigma_t * sigma_t;
        return ClampZero(Sr);
    }

    PBRT_HOST_DEVICE
    Float Pdf_Sr(int ch, Float r) const {
        // Convert $r$ into unitless optical radius $r_{\roman{optical}}$
        Float rOptical = r * sigma_t[ch];

        // Compute spline weights to interpolate BSSRDF density on channel _ch_
        int rhoOffset, radiusOffset;
        Float rhoWeights[4], radiusWeights[4];
        if (!CatmullRomWeights(table.rhoSamples, rho[ch], &rhoOffset, rhoWeights) ||
            !CatmullRomWeights(table.radiusSamples, rOptical, &radiusOffset,
                               radiusWeights))
            return 0.f;

        // Return BSSRDF profile density for channel _ch_
        Float sr = 0, rhoEff = 0;
        for (int i = 0; i < 4; ++i) {
            if (rhoWeights[i] == 0) continue;
            rhoEff += table.rhoEff[rhoOffset + i] * rhoWeights[i];
            for (int j = 0; j < 4; ++j) {
                if (radiusWeights[j] == 0) continue;
                sr += table.EvalProfile(rhoOffset + i, radiusOffset + j) *
                    rhoWeights[i] * radiusWeights[j];
            }
        }

        // Cancel marginal PDF factor from tabulated BSSRDF profile
        if (rOptical != 0) sr /= 2 * Pi * rOptical;
        return std::max<Float>(0, sr * sigma_t[ch] * sigma_t[ch] / rhoEff);
    }

    PBRT_HOST_DEVICE
    Float Sample_Sr(int ch, Float u) const {
        if (sigma_t[ch] == 0) return -1;
        return SampleCatmullRom2D(table.rhoSamples, table.radiusSamples,
                                  table.profile, table.profileCDF, rho[ch], u) /
            sigma_t[ch];
    }

    std::string ToString() const;

  private:
    // TabulatedBSSRDF Private Data
    const SurfaceInteraction &po;
    Float eta;
    Normal3f ns;
    Vector3f ss, ts;
    TransportMode mode;

    const BSSRDFTable &table;
    SampledSpectrum sigma_t, rho;
};

PBRT_HOST_DEVICE
inline void SubsurfaceFromDiffuse(const BSSRDFTable &t, const SampledSpectrum &rhoEff,
                                  const SampledSpectrum &mfp, SampledSpectrum *sigma_a,
                                  SampledSpectrum *sigma_s) {
    for (int c = 0; c < NSpectrumSamples; ++c) {
        Float rho = InvertCatmullRom(t.rhoSamples, t.rhoEff, rhoEff[c]);
        (*sigma_s)[c] = rho / mfp[c];
        (*sigma_a)[c] = (1 - rho) / mfp[c];
    }
}

inline SampledSpectrum
BSSRDFHandle::S(const SurfaceInteraction &pi, const Vector3f &wi) {
    switch (Tag()) {
    case TypeIndex<TabulatedBSSRDF>():
        return Cast<TabulatedBSSRDF>()->S(pi, wi);
    default:
        LOG_FATAL("Unhandled BSSRDF type");
        return {};
    }
}

inline pstd::optional<BSSRDFProbeSegment>
BSSRDFHandle::Sample(Float u1, const Point2f &u2) const {
    switch (Tag()) {
    case TypeIndex<TabulatedBSSRDF>():
        return Cast<TabulatedBSSRDF>()->Sample(u1, u2);
    default:
        LOG_FATAL("Unhandled BSSRDF type");
        return {};
    }
}

inline BSSRDFSample
BSSRDFHandle::ProbeIntersectionToSample(const SurfaceInteraction &si,
                                        MaterialBuffer &materialBuffer) const {
    switch (Tag()) {
    case TypeIndex<TabulatedBSSRDF>():
        return Cast<TabulatedBSSRDF>()->ProbeIntersectionToSample(si, materialBuffer);
    default:
        LOG_FATAL("Unhandled BSSRDF type");
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_BSSRDF_H
