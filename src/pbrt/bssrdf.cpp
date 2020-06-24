
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


// core/bssrdf.cpp*
#include <pbrt/bssrdf.h>

#include <pbrt/util/math.h>
#include <pbrt/media.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/scene.h>

#include <cmath>

namespace pbrt {

// BSSRDF Method Definitions
pstd::optional<BSSRDFSample> SeparableBSSRDF::Sample_S(
    const Scene &scene, Float u1, const Point2f &u2, MaterialBuffer &materialBuffer) const {
    ProfilerScope pp(ProfilePhase::BSSRDFSampling);
    pstd::optional<BSSRDFSample> bs = Sample_Sp(scene, u1, u2, materialBuffer);
    if (!bs) return bs;

    // Initialize material model at sampled surface interaction
    bs->si.bsdf =
        materialBuffer.Alloc<BSDF>(bs->si, materialBuffer.Alloc<SeparableBSSRDFAdapter>(this));
    bs->si.wo = Vector3f(bs->si.shading.n);
    return bs;
}

pstd::optional<BSSRDFSample> SeparableBSSRDF::Sample_Sp(
    const Scene &scene, Float u1, const Point2f &u2, MaterialBuffer &materialBuffer) const {
    ProfilerScope pp(ProfilePhase::BSSRDFEvaluation);
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
    Interaction base;
    base.pi =
        po.pi + r * (vx * std::cos(phi) + vy * std::sin(phi)) - l * vz * 0.5f;
    base.time = po.time;
    Point3f pTarget = base.p() + l * vz;

    // Intersect BSSRDF sampling ray against the scene geometry

    // Declare _IntersectionChain_ and linked list
    struct IntersectionChain {
        SurfaceInteraction si;
        IntersectionChain *next = nullptr;
    };
    IntersectionChain *chain = materialBuffer.Alloc<IntersectionChain>();

    // Accumulate chain of intersections along ray
    IntersectionChain *ptr = chain;
    int nFound = 0;
    while (true) {
        Ray r = base.SpawnRayTo(pTarget);
        if (r.d == Vector3f(0, 0, 0))
            break;

        pstd::optional<ShapeIntersection> si = scene.Intersect(r, 1);
        if (!si)
            break;
        ptr->si = si->intr;

        base = ptr->si;
        // Append admissible intersection to _IntersectionChain_
        if (*ptr->si.material == this->material) {
            IntersectionChain *next = materialBuffer.Alloc<IntersectionChain>();
            ptr->next = next;
            ptr = next;
            nFound++;
        }
    }

    // Randomly choose one of several intersections during BSSRDF sampling
    if (nFound == 0) return {};
    int selected = Clamp((int)(u1 * nFound), 0, nFound - 1);
    while (selected-- > 0) chain = chain->next;

    // Compute sample PDF and return the spatial BSSRDF term $\Sp$
    const SurfaceInteraction &si = chain->si;
    return BSSRDFSample{this->Sp(si), si, this->Pdf_Sp(si) / nFound};
}

Float SeparableBSSRDF::Pdf_Sp(const SurfaceInteraction &pi) const {
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

std::string SeparableBSSRDF::ToString() const {
    return StringPrintf("[ SeparableBSSRDF po: %s eta: %f ns: %s ss: %s ts: %s material: %s mode: %s ]",
                        po, eta, ns, ss, ts, material ? material.ToString().c_str() : "(nullptr)", mode);
}

std::string SeparableBSSRDFAdapter::ToString() const {
    return StringPrintf("[ SeparableBSSRDFAdapter bssrdf: %s ]", *bssrdf);
}

Float BeamDiffusionMS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r) {
    const int nSamples = 100;
    Float Ed = 0;
    // Precompute information for dipole integrand

    // Compute reduced scattering coefficients $\sigmaps, \sigmapt$ and albedo
    // $\rhop$
    Float sigmap_s = sigma_s * (1 - g);
    Float sigmap_t = sigma_a + sigmap_s;
    Float rhop = sigmap_s / sigmap_t;

    // Compute non-classical diffusion coefficient $D_\roman{G}$ using
    // Equation (15.24)
    Float D_g = (2 * sigma_a + sigmap_s) / (3 * sigmap_t * sigmap_t);

    // Compute effective transport coefficient $\sigmatr$ based on $D_\roman{G}$
    Float sigma_tr = SafeSqrt(sigma_a / D_g);

    // Determine linear extrapolation distance $\depthextrapolation$ using
    // Equation (15.28)
    Float fm1 = FresnelMoment1(eta), fm2 = FresnelMoment2(eta);
    Float ze = -2 * D_g * (1 + 3 * fm2) / (1 - 2 * fm1);

    // Determine exitance scale factors using Equations (15.31) and (15.32)
    Float cPhi = .25f * (1 - 2 * fm1), cE = .5f * (1 - 3 * fm2);
    for (int i = 0; i < nSamples; ++i) {
        // Sample real point source depth $\depthreal$
        Float zr = -std::log(1 - (i + .5f) / nSamples) / sigmap_t;

        // Evaluate dipole integrand $E_{\roman{d}}$ at $\depthreal$ and add to
        // _Ed_
        Float zv = -zr + 2 * ze;
        Float dr = std::sqrt(r * r + zr * zr), dv = std::sqrt(r * r + zv * zv);

        // Compute dipole fluence rate $\dipole(r)$ using Equation (15.27)
        Float phiD = Inv4Pi / D_g * (std::exp(-sigma_tr * dr) / dr -
                                     std::exp(-sigma_tr * dv) / dv);

        // Compute dipole vector irradiance $-\N{}\cdot\dipoleE(r)$ using
        // Equation (15.27)
        Float EDn = Inv4Pi * (zr * (1 + sigma_tr * dr) *
                                  std::exp(-sigma_tr * dr) / (dr * dr * dr) -
                              zv * (1 + sigma_tr * dv) *
                                  std::exp(-sigma_tr * dv) / (dv * dv * dv));

        // Add contribution from dipole for depth $\depthreal$ to _Ed_
        Float E = phiD * cPhi + EDn * cE;
        Float kappa = 1 - std::exp(-2 * sigmap_t * (dr + zr));
        Ed += kappa * rhop * rhop * E;
    }
    return Ed / nSamples;
}

Float BeamDiffusionSS(Float sigma_s, Float sigma_a, Float g, Float eta,
                      Float r) {
    // Compute material parameters and minimum $t$ below the critical angle
    Float sigma_t = sigma_a + sigma_s, rho = sigma_s / sigma_t;
    Float tCrit = r * SafeSqrt(eta * eta - 1);
    Float Ess = 0;
    const int nSamples = 100;
    for (int i = 0; i < nSamples; ++i) {
        // Evaluate single scattering integrand and add to _Ess_
        Float ti = tCrit - std::log(1 - (i + .5f) / nSamples) / sigma_t;

        // Determine length $d$ of connecting segment and $\cos\theta_\roman{o}$
        Float d = std::sqrt(r * r + ti * ti);
        Float cosTheta_o = ti / d;

        // Add contribution of single scattering at depth $t$
        Ess += rho * std::exp(-sigma_t * (d + tCrit)) / (d * d) *
               EvaluateHenyeyGreenstein(cosTheta_o, g) * (1 - FrDielectric(-cosTheta_o, eta)) *
               std::abs(cosTheta_o);
    }
    return Ess / nSamples;
}

void ComputeBeamDiffusionBSSRDF(Float g, Float eta, BSSRDFTable *t) {
    // Choose radius values of the diffusion profile discretization
    t->radiusSamples[0] = 0;
    t->radiusSamples[1] = 2.5e-3f;
    for (int i = 2; i < t->radiusSamples.size(); ++i)
        t->radiusSamples[i] = t->radiusSamples[i - 1] * 1.2f;

    // Choose albedo values of the diffusion profile discretization
    for (int i = 0; i < t->rhoSamples.size(); ++i)
        t->rhoSamples[i] =
            (1 - std::exp(-8 * i / (Float)(t->rhoSamples.size() - 1))) /
            (1 - std::exp(-8));
    ParallelFor(0, t->rhoSamples.size(), [&](int i) {
        // Compute the diffusion profile for the _i_th albedo sample

        // Compute scattering profile for chosen albedo $\rho$
        size_t nSamples = t->radiusSamples.size();
        for (int j = 0; j < nSamples; ++j) {
            Float rho = t->rhoSamples[i], r = t->radiusSamples[j];
            t->profile[i * nSamples + j] =
                2 * Pi * r * (BeamDiffusionSS(rho, 1 - rho, g, eta, r) +
                              BeamDiffusionMS(rho, 1 - rho, g, eta, r));
        }

        // Compute effective albedo $\rho_{\roman{eff}}$ and CDF for importance
        // sampling
        t->rhoEff[i] = IntegrateCatmullRom(
            t->radiusSamples,
            {&t->profile[i * nSamples], nSamples},
            {&t->profileCDF[i * nSamples], nSamples});
    });
}


BSSRDFTable::BSSRDFTable(int nRhoSamples, int nRadiusSamples)
    : rhoSamples(nRhoSamples),
      radiusSamples(nRadiusSamples),
      profile(nRadiusSamples * nRhoSamples),
      rhoEff(nRhoSamples),
      profileCDF(nRadiusSamples * nRhoSamples) {}

std::string BSSRDFTable::ToString() const {
    return StringPrintf("[ BSSRDFTable rhoSamples: %s radiusSamples: %s profile: %s "
                        "rhoEff: %s profileCDF: %s ]", rhoSamples, radiusSamples,
                        profile, rhoEff, profileCDF);
}

SampledSpectrum TabulatedBSSRDF::Sr(Float r) const {
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

Float TabulatedBSSRDF::Sample_Sr(int ch, Float u) const {
    if (sigma_t[ch] == 0) return -1;
    return SampleCatmullRom2D(table.rhoSamples, table.radiusSamples,
                              table.profile, table.profileCDF, rho[ch], u) /
           sigma_t[ch];
}

Float TabulatedBSSRDF::Pdf_Sr(int ch, Float r) const {
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

std::string TabulatedBSSRDF::ToString() const {
    return StringPrintf("[ TabulatedBSSRDF %s sigma_t: %s rho: %s table: %s ]",
                        ((const SeparableBSSRDF *)this)->ToString(),
                        sigma_t, rho, table);
}

void SubsurfaceFromDiffuse(const BSSRDFTable &t, const SampledSpectrum &rhoEff,
                           const SampledSpectrum &mfp, SampledSpectrum *sigma_a,
                           SampledSpectrum *sigma_s) {
    for (int c = 0; c < NSpectrumSamples; ++c) {
        Float rho = InvertCatmullRom(t.rhoSamples, t.rhoEff, rhoEff[c]);
        (*sigma_s)[c] = rho / mfp[c];
        (*sigma_a)[c] = (1 - rho) / mfp[c];
    }
}

// We need to override BSSRDF::S() so that we can have access to the full
// hit information in order to modulate based on surface normal
// orientations..
SampledSpectrum DisneyBSSRDF::S(const SurfaceInteraction &pi, const Vector3f &wi) {
    ProfilerScope pp(ProfilePhase::BSSRDFEvaluation);
    // Fade based on relative orientations of the two surface normals to
    // better handle surface cavities. (Details via personal communication
    // from Brent Burley; these details aren't published in the course
    // notes.)
    //
    // TODO: test
    // TODO: explain
    Vector3f a = Normalize(pi.p() - po.p());
    Float fade = 1;
    Vector3f n = Vector3f(po.shading.n);
    Float cosTheta = Dot(a, n);
    if (cosTheta > 0) {
        // Point on or above surface plane
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Vector3f a2 = n * sinTheta - (a - n * cosTheta) * cosTheta / sinTheta;
        fade = std::max(Float(0), Dot(pi.shading.n, a2));
    }

    Float Fo = SchlickWeight(AbsCosTheta(po.wo)),
          Fi = SchlickWeight(AbsCosTheta(wi));
    return fade * (1 - Fo / 2) * (1 - Fi / 2) * Sp(pi) / Pi;
}

// Diffusion profile from Burley 2015, eq (5).
SampledSpectrum DisneyBSSRDF::Sr(Float r) const {
    ProfilerScope pp(ProfilePhase::BSSRDFEvaluation);
    if (r < 1e-6f) r = 1e-6f;  // Avoid singularity at r == 0.
    return R * (Exp(-SampledSpectrum(r) / d) +
                Exp(-SampledSpectrum(r) / (3 * d))) /
           (8 * Pi * d * r);
}

Float DisneyBSSRDF::Sample_Sr(int ch, Float u) const {
    // The good news is that diffusion profile implemented in Sr is
    // normalized---integrating in polar coordinates, we have:
    //
    // int_0^2pi int_0^Infinity Sr(r) r dr dphi == 1.
    //
    // The CDF can be found in closed-form. It is:
    //
    // 1 - e^(-x/d) / 4 - (3 / 4) e^(-x / (3d)).
    //
    // Unfortunately, inverting the CDF requires solving a cubic, which
    // would be nice to sidestep. Therefore, following Christensen and
    // Burley's suggestion (section 6), we will sample from each of the two
    // exponential terms individually (which can be done directly) and then
    // compute an overall PDF using MIS.  There are a few details to work
    // through...
    //
    // For the first exponential term, we can find:
    // normalized PDF: e^(-r/d) / (2 Pi d r)
    // CDF: 1 - e^(-r/d)
    // sampling recipe: r = d log(1 / (1 - u))
    //
    // For the second:
    // PDF: e^(-r/(3d)) / (6 Pi d r)
    // CDF: 1 - e^(-r/(3d))
    // sampling: r = 3 d log(1 / (1 - u))
    //
    // The last question is what fraction of samples to use for each
    // technique.  The second exponential has 3x the contribution to the
    // final value as the first does, so therefore we'll take three samples
    // from that for every one sample we take from the first.
    if (u < .25f) {
        // Sample the first exponential
        u = std::min<Float>(u * 4, OneMinusEpsilon);  // renormalize to [0,1)
        return d[ch] * std::log(1 / (1 - u));
    } else {
        // Second exponenital
        u = std::min<Float>((u - .25f) / .75f, OneMinusEpsilon);  // normalize to [0,1)
        return 3 * d[ch] * std::log(1 / (1 - u));
    }
}

Float DisneyBSSRDF::Pdf_Sr(int ch, Float r) const {
    if (r < 1e-6f) r = 1e-6f;  // Avoid singularity at r == 0.

    // Weight the two individual PDFs as per the sampling frequency in
    // Sample_Sr().
    return (.25f * std::exp(-r / d[ch]) / (2 * Pi * d[ch] * r) +
            .75f * std::exp(-r / (3 * d[ch])) / (6 * Pi * d[ch] * r));
}

std::string DisneyBSSRDF::ToString() const {
    return StringPrintf("[ DisneyBSSRDF %s R: %s d: %s ]",
                        ((const SeparableBSSRDF *)this)->ToString(), R, d);
}

}  // namespace pbrt
