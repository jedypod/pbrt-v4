
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

// core/bsdf.cpp*
#include <pbrt/bsdf.h>

#include <pbrt/bssrdf.h>
#include <pbrt/interaction.h>
#include <pbrt/media.h>
#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/float.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/log.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/stats.h>

#include <unordered_map>

namespace pbrt {

std::string BSDFSample::ToString() const {
    return StringPrintf("[ BSDFSample f: %s wi: %s pdf: %s flags: %s ]",
                        f, wi, pdf, flags);
}

std::string ToString(BxDFReflTransFlags flags) {
    if (flags == BxDFReflTransFlags::Unset) return "Unset";
    std::string s;
    if (flags & BxDFReflTransFlags::Reflection) s += "Reflection,";
    if (flags & BxDFReflTransFlags::Transmission) s += "Transmission,";
    return s;
}

std::string ToString(BxDFFlags flags) {
    if (flags == BxDFFlags::Unset) return "Unset";
    std::string s;
    if (flags & BxDFFlags::Reflection) s += "Reflection,";
    if (flags & BxDFFlags::Transmission) s += "Transmission,";
    if (flags & BxDFFlags::Diffuse) s += "Diffuse,";
    if (flags & BxDFFlags::Glossy) s += "Glossy,";
    if (flags & BxDFFlags::Specular) s += "Specular,";
    return s;
}

// BxDF Method Definitions
SampledSpectrum MixBxDF::f(const Vector3f &wo, const Vector3f &wi) const {
    return Lerp(t, bxdf0.f(wo, wi), bxdf1.f(wo, wi));
}

pstd::optional<BSDFSample> MixBxDF::Sample_f(const Vector3f &wo, Float uc,
                                             const Point2f &u, BxDFReflTransFlags sampleFlags) const {
    BxDFHandle b[2] = { bxdf0, bxdf1 };
    Float w[2] = { t, Float(1 - t) };

    if (uc < t)
        uc = std::min<Float>(uc / t, OneMinusEpsilon);
    else {
        pstd::swap(b[0], b[1]);
        pstd::swap(w[0], w[1]);
        uc = std::min<Float>((uc - t) / (1 - t), OneMinusEpsilon);
    }

    pstd::optional<BSDFSample> bs = b[0].Sample_f(wo, uc, u, sampleFlags);
    if (!bs || bs->pdf == 0) return {};
    if (bs->flags & BxDFFlags::Specular)
        return bs;

    bs->f = w[0] * bs->f + w[1] * b[1].f(wo, bs->wi);
    bs->pdf = w[0] * bs->pdf + w[1] * b[1].PDF(wo, bs->wi, sampleFlags);
    return bs;
}

Float MixBxDF::PDF(const Vector3f &wo, const Vector3f &wi,
                   BxDFReflTransFlags sampleFlags) const {
    return Lerp(t, bxdf0.PDF(wo, wi, sampleFlags),
                   bxdf1.PDF(wo, wi, sampleFlags));
}

std::string MixBxDF::ToString() const {
    return StringPrintf("[ MixBxDF t: %f bxdf0: %s bxdf1: %s ]", t, bxdf0, bxdf1);
}

BxDFHandle MixBxDF::Regularize(MaterialBuffer &materialBuffer) {
    return materialBuffer.Alloc<MixBxDF>(t, bxdf0.Regularize(materialBuffer),
                                         bxdf1.Regularize(materialBuffer));
}

// BSDF Method Definitions
SampledSpectrum BSDF::SampleSpecular_f(const Vector3f &wo, Vector3f *wi,
                                       BxDFReflTransFlags sampleFlags) const {
    pstd::optional<BSDFSample> s = Sample_f(wo, 0, Point2f(0, 0), sampleFlags);
    if (!s || !s->f || s->pdf == 0) return SampledSpectrum(0);
    *wi = s->wi;
    return s->f / s->pdf;
}

std::string BSDF::ToString() const {
    return StringPrintf("[ BSDF eta: %f bxdf: %s shadingFrame: %s ng: %s ]",
                        eta, bxdf, shadingFrame, ng);
}

std::string LambertianBxDF::ToString() const {
    return StringPrintf("[ Lambertian R: %s T: %s A: %f B: %f ]", R, T, A, B);
}

template <typename TopBxDF, typename BottomBxDF>
class TopOrBottomBxDF {
public:
    TopOrBottomBxDF() = default;
    PBRT_HOST_DEVICE
    TopOrBottomBxDF &operator=(const TopBxDF *t) { top = t; bottom = nullptr; return *this; }
    PBRT_HOST_DEVICE
    TopOrBottomBxDF &operator=(const BottomBxDF *b) { bottom = b; top = nullptr; return *this; }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        return top ? top->f(wo, wi) : bottom->f(wo, wi);
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return top ? top->Sample_f(wo, uc, u, sampleFlags) :
            bottom->Sample_f(wo, uc, u, sampleFlags);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return top ? top->PDF(wo, wi, sampleFlags) : bottom->PDF(wo, wi, sampleFlags);
    }

    PBRT_HOST_DEVICE
    void FlipTransportMode() {
        if (top)
            const_cast<TopBxDF *>(top)->FlipTransportMode();
        else
            const_cast<BottomBxDF *>(bottom)->FlipTransportMode();
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const {
        return top ? top->Flags() : bottom->Flags();
    }

private:
    const TopBxDF *top = nullptr;
    const BottomBxDF *bottom = nullptr;
};

template <>
class TopOrBottomBxDF<BxDFHandle, BxDFHandle> {
public:
    TopOrBottomBxDF() = default;
    PBRT_HOST_DEVICE
    TopOrBottomBxDF &operator=(const BxDFHandle *b) { bxdf = *b; return *this; }

    PBRT_HOST_DEVICE
    SampledSpectrum f(const Vector3f &wo, const Vector3f &wi) const {
        return bxdf.f(wo, wi);
    }

    PBRT_HOST_DEVICE
    pstd::optional<BSDFSample> Sample_f(const Vector3f &wo, Float uc, const Point2f &u,
                                        BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return bxdf.Sample_f(wo, uc, u, sampleFlags);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi,
              BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return bxdf.PDF(wo, wi, sampleFlags);
    }

    PBRT_HOST_DEVICE
    void FlipTransportMode() {
        bxdf.FlipTransportMode();
    }

    PBRT_HOST_DEVICE
    BxDFFlags Flags() const {
        return bxdf.Flags();
    }

private:
    BxDFHandle bxdf;
};

template <typename TopBxDF, typename BottomBxDF>
SampledSpectrum LayeredBxDF<TopBxDF, BottomBxDF>::f(const Vector3f &woOrig, const Vector3f &wiOrig) const {
    Vector3f wo = woOrig, wi = wiOrig;
    if (config.twoSided && wo.z < 0) {
        // BIG WIN
        wo = -wo;
        wi = -wi;
    }

    bool enteredTop = wo.z > 0;
    TopOrBottomBxDF<TopBxDF, BottomBxDF> enterInterface, exitInterface;
    TopOrBottomBxDF<TopBxDF, BottomBxDF> nonExitInterface;
    if (enteredTop)
        enterInterface = &top;
    else
        enterInterface = &bottom;
    if (SameHemisphere(wo, wi) ^ enteredTop) {
        exitInterface = &bottom;
        nonExitInterface = &top;
    } else {
        exitInterface = &top;
        nonExitInterface = &bottom;
    }
    Float exitZ = (SameHemisphere(wo, wi) ^ enteredTop) ? 0 : thickness;

    SampledSpectrum f(0.);
    if (SameHemisphere(wo, wi))
        f = config.nSamples * enterInterface.f(wo, wi);

#ifdef __CUDA_ARCH__
    RNG rng(Hash(0 /* seed */, wo, wi));
#else
    RNG rng(Hash(PbrtOptions.seed, wo, wi));
#endif
    auto r = [&rng]() { return std::min<Float>(rng.Uniform<Float>(), OneMinusEpsilon); };

    for (int s = 0; s < config.nSamples; ++s) {
        Float uc = r();
        Point2f u(r(), r());
        if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
        pstd::optional<BSDFSample> wos = enterInterface.Sample_f(wo, uc, u, BxDFReflTransFlags::Transmission);
        if (!wos || wos->pdf == 0 || wos->wi.z == 0)
            continue;

        uc = r();
        u = Point2f(r(), r());
        if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
        exitInterface.FlipTransportMode();
        pstd::optional<BSDFSample> wis = exitInterface.Sample_f(wi, uc, u, BxDFReflTransFlags::Transmission);
        exitInterface.FlipTransportMode();
        if (!wis || wis->pdf == 0 || wis->wi.z == 0)
            continue;

        // State
        SampledSpectrum beta = wos->f * AbsCosTheta(wos->wi) / wos->pdf;
        SampledSpectrum betaExit = wis->f / wis->pdf;
        Vector3f w = wos->wi;
        Float z = enteredTop ? thickness : 0;
        HenyeyGreenstein phase(g);

        for (int depth = 0; depth < (config.deterministic ? 1 : config.maxDepth); ++depth) {
            VLOG(2, "beta: %s, w: %s, f: %s", beta, w, f);
            // Russian roulette
            if (depth > 3 && beta.MaxComponentValue() < .25) {
                Float q = std::max<Float>(0, 1 - beta.MaxComponentValue());
                if (r() < q)
                    break;
                beta /= 1 - q;
                VLOG(2, "After RR with q = %f, beta: %s", q, beta);
            }

            if (albedo) {
                Float sigma_t = 1;
                Float dz = SampleExponential(r(), sigma_t / AbsCosTheta(w));
                Float zp = w.z > 0 ? (z + dz) : (z - dz);

                CHECK_RARE(1e-5, z == zp);
                if (z == zp) continue;

                if (0 < zp && zp < thickness) {
                    // scattering event

#if 0
                    // TODO: cancel out and simplify: should be
                    // f *= AbsCosTheta(w) / sigma_t (!!!) -- that in turn makes the tricky cosine stuff
                    // more reasonable / palatible...
                    //beta *= Tr(dz, w) / ExponentialPDF(dz, sigma_t / AbsCosTheta(w));
                    beta *= AbsCosTheta(w) / sigma_t;

                    // Tricky cosines. Always divide here since we always
                    // include it when we leave a surface.
                    beta /= AbsCosTheta(w);
#endif

                    // NEE
                    // First strategy: use pre-sampled top direction
                    Float wt = 1;
                    if (!IsSpecular(exitInterface.Flags()))
                        wt = PowerHeuristic(1, wis->pdf, 1, phase.PDF(-w, -wis->wi));
                    Float te = Tr(zp - exitZ, wis->wi);
                    f += beta * albedo * phase.p(-w, -wis->wi) * wt * te * betaExit;

                    // Second strategy: sample phase function (NEE and next
                    // bounce, all at once...)
                    pstd::optional<PhaseFunctionSample> ps = phase.Sample_p(-w, Point2f(r(), r()));
                    if (!ps || ps->pdf == 0 || ps->wi.z == 0)
                        continue;

                    beta *= albedo * ps->p / ps->pdf;
                    w = ps->wi;
                    z = zp;

                    if (!IsSpecular(exitInterface.Flags())) {
                        // MIS, part 2
                        // Order to get the IOR^2 scaling right...
                        SampledSpectrum fExit = exitInterface.f(-w, wi);
                        if (fExit) {
                            Float tp = exitInterface.PDF(-w, wi, BxDFReflTransFlags::Transmission);
                            Float wt = PowerHeuristic(1, ps->pdf, 1, tp);
                            Float ti = Tr(zp - exitZ, ps->wi);
                            f += beta * ti * fExit * wt; // 1/pdf et al are already in beta
                        }
                    }

                    continue;
                }

                // Nothing to do for transmittance: either we make it to the
                // boundary or we don't.
                // TODO: this isn't ideal for optically thin scattering media...

                // jump to the boundary,
                z = Clamp(zp, 0, thickness);
                // and fall through to scatter from the top or bottom..
            } else {
                z = (z == thickness) ? 0 : thickness;
                beta *= Tr(thickness, w);
            }

            if (z == exitZ) {
                Float uc = r();
                Point2f u(r(), r());
                pstd::optional<BSDFSample> bs = exitInterface.Sample_f(-w, uc, u, BxDFReflTransFlags::Reflection);
                if (!bs || bs->pdf == 0 || bs->wi.z == 0)
                    break;

                beta *= bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                w = bs->wi;
            } else {
                // Non-exit interface

                // NEE
                // First strategy: use pre-sampled top direction
                if (!IsSpecular(nonExitInterface.Flags())) {
                    Float wt = 1;
                    if (!IsSpecular(exitInterface.Flags()))
                        wt = PowerHeuristic(1, wis->pdf, 1, nonExitInterface.PDF(-w, -wis->wi));
                    Float te = Tr(thickness, wis->wi);
                    f += beta * nonExitInterface.f(-w, -wis->wi) * AbsCosTheta(wis->wi) *
                        wt * te * betaExit;
                }

                // Second strategy: sample bottom BSDF (NEE and next
                // bounce, all at once...)
                Float uc = r();
                Point2f u(r(), r());
                if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
                pstd::optional<BSDFSample> bs = nonExitInterface.Sample_f(-w, uc, u, BxDFReflTransFlags::Reflection);
                if (!bs || bs->pdf == 0 || bs->wi.z == 0)
                    break;

                beta *= bs->f * AbsCosTheta(bs->wi) / bs->pdf;
                w = bs->wi;

                if (!IsSpecular(exitInterface.Flags())) {
                    // MIS, part 2
                    // Order to get the IOR^2 scaling right...
                    SampledSpectrum fExit = exitInterface.f(-w, wi);
                    if (fExit) {
                        Float wt = 1;
                        if (!IsSpecular(nonExitInterface.Flags())) {
                            Float tp = exitInterface.PDF(-w, wi, BxDFReflTransFlags::Transmission);
                            wt = PowerHeuristic(1, bs->pdf, 1, tp);
                        }
                        f += beta * Tr(thickness, bs->wi) * fExit * wt; // 1/pdf et al are already in beta
                    }
                }
            }
        }
    }

    f /= config.nSamples;

    VLOG(2, "Final f = %s", f);
    VLOG(2, "--------------------");

    return f;
}

template <typename TopBxDF, typename BottomBxDF>
pstd::optional<BSDFSample> LayeredBxDF<TopBxDF, BottomBxDF>::Sample_f(
    const Vector3f &woOrig, Float uc,
    const Point2f &u, BxDFReflTransFlags sampleFlags) const {
    CHECK(sampleFlags == BxDFReflTransFlags::All); // for now

    Vector3f wo = woOrig;
    bool flipWi = false;
    if (config.twoSided && wo.z < 0) {
        wo = -wo;
        flipWi = true;
    }

    bool enteredTop = wo.z > 0;
    pstd::optional<BSDFSample> bs = enteredTop ? top.Sample_f(wo, uc, u) :
        bottom.Sample_f(wo, uc, u);
    if (!bs || !bs->pdf) return {};
    if (bs->IsReflection()) {
        if (flipWi) bs->wi = -bs->wi;
        return bs;
    }

#ifdef __CUDA_ARCH__
    RNG rng(Hash(0 /* seed */, wo, uc, u));
#else
    RNG rng(Hash(PbrtOptions.seed, wo, uc, u));
#endif

    auto r = [&rng]() { return std::min<Float>(rng.Uniform<Float>(), OneMinusEpsilon); };

    SampledSpectrum f = bs->f * AbsCosTheta(bs->wi);
    Float pdf = bs->pdf;
    Vector3f w = bs->wi;
    Float z = enteredTop ? thickness : 0;
    HenyeyGreenstein phase(g);

    for (int depth = 0; depth < (config.deterministic ? 1 : config.maxDepth); ++depth) {
        // Russian roulette
        // TODO: would like to know eta to have Sqr(eta) in rrBeta there...
        Float rrBeta = f.MaxComponentValue() / pdf;
        // It's worth being unaggressive here: a terminated path basically
        // terminates a larger-scale geometric ray path, which is a big
        // loss, and this path tracing is pretty inexpensive...
        if (depth > 3 && rrBeta < 0.25) {
            Float q = std::max<Float>(0, 1 - rrBeta);
            if (r() < q)
                return {};
            pdf *= 1 - q;
        }
        if (w.z == 0) return {};

        if (albedo) {
            Float sigma_t = 1;
            Float dz = SampleExponential(r(), sigma_t / AbsCosTheta(w));
            Float zp = w.z > 0 ? (z + dz) : (z - dz);
            CHECK_RARE(1e-5, zp == z);
            if (zp == z) return {};

            if (0 < zp && zp < thickness) {
                // scattering event

#if 0
                // TODO: cancel out and simplify: should be
                // f *= AbsCosTheta(w) / sigma_t (!!!) -- that in turn makes the tricky cosine stuff
                // more reasonable / palatible...
                //f *= Tr(dz, w) / ExponentialPDF(dz, sigma_t / AbsCosTheta(w));
                f *= AbsCosTheta(w) / sigma_t;

                // Tricky cosines. Always divide here since we always
                // include it when we leave a surface.
                f /= AbsCosTheta(w);
#endif

                // Scatter
                pstd::optional<PhaseFunctionSample> ps = phase.Sample_p(-w, Point2f(r(), r()));
                if (!ps || ps->pdf == 0 || ps->wi.z == 0)
                    return {};
                f *= albedo * ps->p;
                pdf *= ps->pdf;
                w = ps->wi;
                z = zp;

                continue;
            }

            // Nothing to do for transmittance: either we make it to the
            // boundary or we don't.
            // TODO: this isn't ideal for optically thin scattering media...

            // jump to the boundary,
            z = Clamp(zp, 0, thickness);
            // and fall through to scatter from the top or bottom..
            if (z == 0) CHECK_LT(w.z, 0);
            else CHECK_GT(w.z, 0);
        } else {
            // Bounce back and forth between the top and bottom
            z = (z == thickness) ? 0 : thickness;
            f *= Tr(thickness, w);
        }

        TopOrBottomBxDF<TopBxDF, BottomBxDF> interface;
        if (z == 0) interface = &bottom;
        else interface = &top;

        Float uc = r();
        Point2f u(r(), r());
        if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
        pstd::optional<BSDFSample> bs = interface.Sample_f(-w, uc, u);
        if (!bs || bs->pdf == 0 || bs->wi.z == 0)
            return {};

        f *= bs->f;
        pdf *= bs->pdf;
        w = bs->wi;

        if (bs->IsTransmission()) {
            BxDFFlags flags = SameHemisphere(wo, w) ? BxDFFlags::GlossyReflection :
                BxDFFlags::GlossyTransmission;
            if (flipWi)
                w = -w;
            return BSDFSample(f, w, pdf, flags);
        }

        // The cosine isn't included when we scatter out...
        f *= AbsCosTheta(bs->wi);
    }

    return {};
}

template <typename TopBxDF, typename BottomBxDF>
Float LayeredBxDF<TopBxDF, BottomBxDF>::PDF(const Vector3f &woOrig, const Vector3f &wiOrig,
                                            BxDFReflTransFlags sampleFlags) const {
    CHECK(sampleFlags == BxDFReflTransFlags::All); // for now

    Vector3f wo = woOrig, wi = wiOrig;
    if (config.twoSided && wo.z < 0) {
        wo = -wo;
        wi = -wi;
    }

#ifdef __CUDA_ARCH__
    RNG rng(Hash(0 /* seed */, wo, wi));
#else
    RNG rng(Hash(PbrtOptions.seed, wo, wi));
#endif
    auto r = [&rng]() { return std::min<Float>(rng.Uniform<Float>(), OneMinusEpsilon); };

    bool enteredTop = wo.z > 0;
    Float pdfSum = 0;

    if (SameHemisphere(wo, wi)) {
        if (enteredTop)
            pdfSum += config.nSamples * top.PDF(wo, wi, BxDFReflTransFlags::Reflection);
        else
            pdfSum += config.nSamples * bottom.PDF(wo, wi, BxDFReflTransFlags::Reflection);
    }

    for (int s = 0; s < config.nSamples; ++s) {
        if (SameHemisphere(wo, wi)) {
            // TRT
            TopOrBottomBxDF<TopBxDF, BottomBxDF> rInterface, tInterface;
            if (enteredTop) {
                rInterface = &bottom;
                tInterface = &top;
            } else {
                rInterface = &top;
                tInterface = &bottom;
            }
            Float uc = r();
            Point2f u(r(), r());
            if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
            pstd::optional<BSDFSample> wos = tInterface.Sample_f(wo, uc, u);
            if (!wos || wos->wi.z == 0 || wos->IsReflection())
                pdfSum += tInterface.PDF(wo, wi);
            else {
                uc = r();
                u = Point2f(r(), r());
                if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
                tInterface.FlipTransportMode();
                pstd::optional<BSDFSample> wis = tInterface.Sample_f(wi, uc, u);
                tInterface.FlipTransportMode();
                if (!wis || wis->wi.z == 0 || wis->IsReflection())
                    continue;

                //if (IsSpecular(tInterface.Flags()))
                pdfSum += rInterface.PDF(-wos->wi, -wis->wi);
            }
        } else {
            TopOrBottomBxDF<TopBxDF, BottomBxDF> toInterface, tiInterface;
            if (enteredTop) {
                toInterface = &top;
                tiInterface = &bottom;
            } else {
                toInterface = &bottom;
                tiInterface = &top;
            }

            Float uc = r();
            Point2f u(r(), r());
            if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
            pstd::optional<BSDFSample> wos = toInterface.Sample_f(wo, uc, u);
            if (!wos || wos->wi.z == 0 || wos->IsReflection())
                continue;

            uc = r();
            u = Point2f(r(), r());
            if (config.deterministic) { uc = 0.5, u = Point2f(0.5, 0.5); }
            tiInterface.FlipTransportMode();
            pstd::optional<BSDFSample> wis = tiInterface.Sample_f(wi, uc, u);
            tiInterface.FlipTransportMode();
            if (!wis || wis->wi.z == 0 || wis->IsReflection())
                continue;

            if (IsSpecular(toInterface.Flags()))
                pdfSum += tiInterface.PDF(-wos->wi, wi);
            else if (IsSpecular(tiInterface.Flags()))
                pdfSum += toInterface.PDF(wo, -wis->wi);
            else
                pdfSum += (toInterface.PDF(wo, -wis->wi) +
                           tiInterface.PDF(-wos->wi, wi)) / 2;
        }
    }

    return Lerp(.9, 1 / (4 * Pi), pdfSum / config.nSamples);
}

template <typename TopBxDF, typename BottomBxDF>
std::string LayeredBxDF<TopBxDF, BottomBxDF>::ToString() const {
    return StringPrintf("[ LayeredBxDF top: %s bottom: %s thickness: %f albedo: %s g: %f ]",
                        top, bottom, thickness, albedo, g);
}

template <typename TopBxDF, typename BottomBxDF>
BxDFHandle LayeredBxDF<TopBxDF, BottomBxDF>::Regularize(MaterialBuffer &materialBuffer) {
    // Note: we lose the type specialization...
    return materialBuffer.Alloc<LayeredBxDF<BxDFHandle, BxDFHandle>>(top.Regularize(materialBuffer),
                                                                     bottom.Regularize(materialBuffer),
                                                                     thickness, albedo, g, config);
}

std::string SpecularReflectionBxDF::ToString() const {
    return StringPrintf("[ SpecularReflection fresnel: %s ]", fresnel);
}

pstd::optional<BSDFSample> SpecularTransmissionBxDF::Sample_f(const Vector3f &wo, Float uc,
                                                              const Point2f &u,
                                                              BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) return {};

    // Figure out which $\eta$ is incident and which is transmitted
    bool entering = CosTheta(wo) > 0;
    Float etap = entering ? eta : (1 / eta);

    // Compute ray direction for specular transmission
    Vector3f wi;
    bool tir = !Refract(wo, FaceForward(Normal3f(0, 0, 1), wo), etap, &wi);
    CHECK_RARE(1e-6, tir);
    if (tir)
        return {};
    FresnelDielectric fresnel(eta);
    SampledSpectrum f = (1 - fresnel.Evaluate(CosTheta(wi))) / AbsCosTheta(wi);

    // Account for non-symmetry with transmission to different medium
    if (mode == TransportMode::Radiance) f /= Sqr(etap);
    return BSDFSample(f, wi, 1, BxDFFlags::SpecularTransmission);
}

std::string SpecularTransmissionBxDF::ToString() const {
    return std::string("[ SpecularTransmission: ") +
        StringPrintf(" eta: %f", eta) +
        std::string(" mode : ") +
        (mode == TransportMode::Radiance ? std::string("Radiance")
                                         : std::string("Importance")) +
        std::string(" ]");
}

BxDFHandle SpecularTransmissionBxDF::Regularize(MaterialBuffer &materialBuffer) {
    MicrofacetDistributionHandle distrib =
        materialBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
    return materialBuffer.Alloc<MicrofacetTransmissionBxDF>(SampledSpectrum(1), distrib, eta, mode);
}

std::string FresnelConductor::ToString() const {
    return StringPrintf("[ FresnelConductor eta: %s k: %s ]", eta, k);
}

std::string FresnelDielectric::ToString() const {
    return StringPrintf("[ FrenselDielectric eta: %f opaque: %s ]",
                        eta, opaque ? "true" : "false");
}

// Microfacet Utility Functions
// MicrofacetDistribution Method Definitions

std::string TrowbridgeReitzDistribution::ToString() const {
    return StringPrintf("[ TrowbridgeReitzDistribution alpha_x: %f alpha_y: %f ]",
                        alpha_x, alpha_y);
}

MicrofacetDistributionHandle TrowbridgeReitzDistribution::Regularize(
    MaterialBuffer & materialBuffer) const {
    if (alpha_x > 0.3f && alpha_y > 0.3f)
        return this;
    return materialBuffer.Alloc<TrowbridgeReitzDistribution>(std::max<Float>(alpha_x, 0.3f),
                                                             std::max<Float>(alpha_y, 0.3f));
}

SampledSpectrum DielectricInterfaceBxDF::f(const Vector3f &wo, const Vector3f &wi) const {
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

pstd::optional<BSDFSample> DielectricInterfaceBxDF::Sample_f(
    const Vector3f &wo, Float uc, const Point2f &u,
    BxDFReflTransFlags sampleFlags) const {
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

Float DielectricInterfaceBxDF::PDF(const Vector3f &wo, const Vector3f &wi,
                                   BxDFReflTransFlags sampleFlags) const {
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

std::string DielectricInterfaceBxDF::ToString() const {
    return StringPrintf("[ DielectricInterface eta: %f distrib: %s mode: %s ]",
                        eta, distrib ? distrib.ToString().c_str() : "nullptr (specular)",
                        mode == TransportMode::Radiance ? "Radiance" : "Importance");
}

BxDFHandle DielectricInterfaceBxDF::Regularize(MaterialBuffer &materialBuffer) {
    MicrofacetDistributionHandle rd = distrib ? distrib.Regularize(materialBuffer) :
        materialBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
    return materialBuffer.Alloc<DielectricInterfaceBxDF>(eta, rd, mode);
}

SampledSpectrum ThinDielectricBxDF::f(const Vector3f &wo, const Vector3f &wi) const {
    return SampledSpectrum(0);
}

Float ThinDielectricBxDF::PDF(const Vector3f &wo, const Vector3f &wi,
                          BxDFReflTransFlags sampleFlags) const {
    return 0;
}

std::string ThinDielectricBxDF::ToString() const {
    return StringPrintf("[ ThinDielectric eta: %f mode: %s ]", eta,
                        mode == TransportMode::Radiance ? "Radiance" : "Importance");
}

BxDFHandle ThinDielectricBxDF::Regularize(MaterialBuffer &materialBuffer) {
    MicrofacetDistributionHandle distrib =
        materialBuffer.Alloc<TrowbridgeReitzDistribution>(0.3f, 0.3f);
    return materialBuffer.Alloc<DielectricInterfaceBxDF>(eta, distrib, mode);
}

std::string MicrofacetReflectionBxDF::ToString() const {
    return StringPrintf("[ MicrofacetReflection distribution: %s fresnel: %s ]",
                        distribution, fresnel);
}

BxDFHandle MicrofacetReflectionBxDF::Regularize(MaterialBuffer &materialBuffer) {
    return materialBuffer.Alloc<MicrofacetReflectionBxDF>(distribution.Regularize(materialBuffer), fresnel);
}

SampledSpectrum MicrofacetTransmissionBxDF::f(const Vector3f &wo,
                                              const Vector3f &wi) const {
    if (SameHemisphere(wo, wi)) return SampledSpectrum(0);

    Float cosTheta_o = CosTheta(wo), cosTheta_i = CosTheta(wi);
    if (cosTheta_i == 0 || cosTheta_o == 0) return SampledSpectrum(0.);

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    Float etap = CosTheta(wo) > 0 ? eta : (1 / eta);
    Vector3f wh = wo + wi * etap;
    CHECK_RARE(1e-6, LengthSquared(wh) == 0);
    if (LengthSquared(wh) == 0) return SampledSpectrum(0.);
    wh = FaceForward(Normalize(wh), Normal3f(0, 0, 1));

    FresnelDielectric fresnel(eta);
    SampledSpectrum F = fresnel.Evaluate(Dot(wo, wh));

    Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
    Float factor = (mode == TransportMode::Radiance) ? etap : 1;

    return T * (1 - F) * factor *
           std::abs(distribution.D(wh) * distribution.G(wo, wi) *
                    AbsDot(wi, wh) * AbsDot(wo, wh) /
                    (cosTheta_i * cosTheta_o * Sqr(sqrtDenom)));
}

std::string MicrofacetTransmissionBxDF::ToString() const {
    return StringPrintf("[ MicrofacetTransmission distribution: %s eta %f",
                        distribution, eta) +
        std::string(" mode : ") +
           (mode == TransportMode::Radiance ? std::string("Radiance")
                                            : std::string("Importance")) +
        std::string(" ]");
}

BxDFHandle MicrofacetTransmissionBxDF::Regularize(MaterialBuffer &materialBuffer) {
    return materialBuffer.Alloc<MicrofacetTransmissionBxDF>(
        SampledSpectrum(1.f), distribution.Regularize(materialBuffer), eta, mode);
}

pstd::optional<BSDFSample> MicrofacetTransmissionBxDF::Sample_f(const Vector3f &wo, Float uc,
                                                                const Point2f &u,
                                                                BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) return {};
    if (wo.z == 0) return {};

    Vector3f wh = distribution.Sample_wm(wo, u);
    Float etap = CosTheta(wo) > 0 ? (1 / eta) : eta;
    Vector3f wi;
    if (!Refract(wo, (Normal3f)wh, etap, &wi) || wi.z == 0) return {};

    // Evaluate BSDF
    // TODO: share fragments with f(), PDF()...
    wh = FaceForward(wh, Normal3f(0, 0, 1));
    FresnelDielectric fresnel(eta);
    SampledSpectrum F = fresnel.Evaluate(Dot(wo, wh));

    Float sqrtDenom = Dot(wo, wh) + etap * Dot(wi, wh);
    Float factor = (mode == TransportMode::Radiance) ? etap : 1;

    SampledSpectrum f = T * (1 - F) * factor *
        std::abs(distribution.D(wh) * distribution.G(wo, wi) *
                 AbsDot(wi, wh) * AbsDot(wo, wh) /
                 (AbsCosTheta(wi) * AbsCosTheta(wo) * Sqr(sqrtDenom)));

    // Compute PDF
    Float dwh_dwi = Sqr(etap) * AbsDot(wi, wh) /
        Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
    Float pdf = distribution.PDF(wo, wh) * dwh_dwi;

    return BSDFSample(f, wi, pdf, BxDFFlags::GlossyTransmission);
}

Float MicrofacetTransmissionBxDF::PDF(const Vector3f &wo, const Vector3f &wi,
                                  BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Transmission)) return 0;
    if (SameHemisphere(wo, wi)) return 0;

    // Compute $\wh$ from $\wo$ and $\wi$ for microfacet transmission
    Float etap = CosTheta(wo) > 0 ? (1 / eta) : eta;
    Vector3f wh = wo + wi * etap;
    CHECK_RARE(1e-6, LengthSquared(wh) == 0);
    if (LengthSquared(wh) == 0) return 0;
    wh = Normalize(wh);

    // Compute change of variables _dwh\_dwi_ for microfacet transmission
    Float dwh_dwi = Sqr(etap) * AbsDot(wi, wh) /
        Sqr(Dot(wo, wh) + etap * Dot(wi, wh));
    return distribution.PDF(wo, wh) * dwh_dwi;
}

std::string DisneyDiffuseLobe::ToString() const {
    return StringPrintf("[ DisneyDiffuse R: %s ]", R);
}

std::string DisneyFresnel::ToString() const {
    return StringPrintf("[ DisneyFresnel R0: %s metallic: %f eta: %f ]",
                        R0, metallic, eta);
}

std::string DisneyFakeSSLobe::ToString() const {
    return StringPrintf("[ DisneyFakeSS R: %s roughness: %f ]", R, roughness);
}

std::string DisneyRetroLobe::ToString() const {
    return StringPrintf("[ DisneyRetro R: %s roughness: %f ]", R, roughness);
}

std::string DisneySheenLobe::ToString() const {
    return StringPrintf("[ DisneySheen R: %s]", R);
}

std::string DisneyClearcoatLobe::ToString() const {
    return StringPrintf("[ DisneyClearcoat weight: %f gloss: %f ]", weight,
                        gloss);
}

std::string DisneyBxDF::ToString() const {
    return StringPrintf("[ DisneyBxDF diffuseReflection: %s fakeSS: %s retro: %s "
                        "sheen: %s clearcoat: %s glossyReflection: %s "
                        "glossyTransmission: %s diffuseTransmission: %s "
                        "subsurfaceBxDF: %s",
                        diffuseReflection ? diffuseReflection->ToString() : std::string("(nullptr)"),
                        fakeSS ? fakeSS->ToString() : std::string("(nullptr)"),
                        retro ? retro->ToString() : std::string("(nullptr)"),
                        sheen ? sheen->ToString() : std::string("(nullptr)"),
                        clearcoat ? clearcoat->ToString() : std::string("(nullptr)"),
                        glossyReflection ? glossyReflection->ToString() : std::string("(nullptr)"),
                        glossyTransmission ? glossyTransmission->ToString() : std::string("(nullptr)"),
                        diffuseTransmission ? diffuseTransmission->ToString() : std::string("(nullptr)"),
                        subsurfaceBxDF ? subsurfaceBxDF->ToString() : std::string("(nullptr)"));
}

// Hair Local Functions
PBRT_HOST_DEVICE
static Float Mp(Float cosTheta_i, Float cosTheta_o, Float sinTheta_i,
                Float sinTheta_o, Float v) {
    Float a = cosTheta_i * cosTheta_o / v;
    Float b = sinTheta_i * sinTheta_o / v;
    Float mp =
        (v <= .1)
            ? (std::exp(LogI0(a) - b - 1 / v + 0.6931f + std::log(1 / (2 * v))))
            : (std::exp(-b) * I0(a)) / (std::sinh(1 / v) * 2 * v);
    CHECK(!std::isinf(mp) && !std::isnan(mp));
    return mp;
}

PBRT_HOST_DEVICE
static pstd::array<SampledSpectrum, pMax + 1> Ap(Float cosTheta_o, Float eta, Float h,
                                                 const SampledSpectrum &T) {
    pstd::array<SampledSpectrum, pMax + 1> ap;
    // Compute $p=0$ attenuation at initial cylinder intersection
    Float cosGamma_o = SafeSqrt(1 - h * h);
    Float cosTheta = cosTheta_o * cosGamma_o;
    Float f = FrDielectric(cosTheta, eta);
    ap[0] = SampledSpectrum(f);

    // Compute $p=1$ attenuation term
    ap[1] = Sqr(1 - f) * T;

    // Compute attenuation terms up to $p=_pMax_$
    for (int p = 2; p < pMax; ++p) ap[p] = ap[p - 1] * T * f;

    // Compute attenuation term accounting for remaining orders of scattering
    ap[pMax] = ap[pMax - 1] * f * T / (1.f - T * f);
    return ap;
}

PBRT_HOST_DEVICE
inline Float Phi(int p, Float gamma_o, Float gamma_t) {
    return 2 * p * gamma_t - 2 * gamma_o + p * Pi;
}

PBRT_HOST_DEVICE
inline Float Np(Float phi, int p, Float s, Float gamma_o, Float gamma_t) {
    Float dphi = phi - Phi(p, gamma_o, gamma_t);
    // Remap _dphi_ to $[-\pi,\pi]$
    while (dphi > Pi) dphi -= 2 * Pi;
    while (dphi < -Pi) dphi += 2 * Pi;
    return TrimmedLogistic(dphi, s, -Pi, Pi);
}

// HairBxDF Method Definitions
HairBxDF::HairBxDF(Float h, Float eta, const SampledSpectrum &sigma_a, Float beta_m,
                   Float beta_n, Float alpha)
    // Hmm, it is transmissive, but we don't want integrators to apply the
    // etaScale stuff when doing RR, since a single scattering event here
    // doesn't change the ray's medium...
    : h(h),
      gamma_o(SafeASin(h)),
      eta(eta),
      sigma_a(sigma_a),
      beta_m(beta_m),
      beta_n(beta_n) {
    CHECK(h >= -1 && h <= 1);
    CHECK(beta_m >= 0 && beta_m <= 1);
    CHECK(beta_n >= 0 && beta_n <= 1);
    // Compute longitudinal variance from $\beta_m$
    static_assert(
        pMax >= 3,
        "Longitudinal variance code must be updated to handle low pMax");
    v[0] = Sqr(0.726f * beta_m + 0.812f * Sqr(beta_m) + 3.7f * Pow<20>(beta_m));
    v[1] = .25 * v[0];
    v[2] = 4 * v[0];
    for (int p = 3; p <= pMax; ++p)
        // TODO: is there anything better here?
        v[p] = v[2];

    // Compute azimuthal logistic scale factor from $\beta_n$
    s = SqrtPiOver8 *
        (0.265f * beta_n + 1.194f * Sqr(beta_n) + 5.372f * Pow<22>(beta_n));
    CHECK(!std::isnan(s));

    // Compute $\alpha$ terms for hair scales
    sin2kAlpha[0] = std::sin(Radians(alpha));
    cos2kAlpha[0] = SafeSqrt(1 - Sqr(sin2kAlpha[0]));
    for (int i = 1; i < 3; ++i) {
        sin2kAlpha[i] = 2 * cos2kAlpha[i - 1] * sin2kAlpha[i - 1];
        cos2kAlpha[i] = Sqr(cos2kAlpha[i - 1]) - Sqr(sin2kAlpha[i - 1]);
    }
}

SampledSpectrum HairBxDF::f(const Vector3f &wo, const Vector3f &wi) const {
    // Compute hair coordinate system terms related to _wo_
    Float sinTheta_o = wo.x;
    Float cosTheta_o = SafeSqrt(1 - Sqr(sinTheta_o));
    Float phi_o = std::atan2(wo.z, wo.y);

    // Compute hair coordinate system terms related to _wi_
    Float sinTheta_i = wi.x;
    Float cosTheta_i = SafeSqrt(1 - Sqr(sinTheta_i));
    Float phi_i = std::atan2(wi.z, wi.y);

    // Compute $\cos \thetat$ for refracted ray
    Float sinTheta_t = sinTheta_o / eta;
    Float cosTheta_t = SafeSqrt(1 - Sqr(sinTheta_t));

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float cosGamma_t = SafeSqrt(1 - Sqr(sinGamma_t));
    Float gamma_t = SafeASin(sinGamma_t);

    // Compute the transmittance _T_ of a single path through the cylinder
    SampledSpectrum T = Exp(-sigma_a * (2 * cosGamma_t / cosTheta_t));

    // Evaluate hair BSDF
    Float phi = phi_i - phi_o;
    pstd::array<SampledSpectrum, pMax + 1> ap = Ap(cosTheta_o, eta, h, T);
    SampledSpectrum fsum(0.);
    for (int p = 0; p < pMax; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sinThetap_o, cosThetap_o;
        if (p == 0) {
            sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
            cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
        }

        // Handle remainder of $p$ values for hair scale tilt
        else if (p == 1) {
            sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
            cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
            cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
        } else {
            sinThetap_o = sinTheta_o;
            cosThetap_o = cosTheta_o;
        }

        cosThetap_o = std::abs(cosThetap_o);
        fsum += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) * ap[p] *
                Np(phi, p, s, gamma_o, gamma_t);
    }

    // Compute contribution of remaining terms after _pMax_
    fsum += Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) * ap[pMax] /
            (2.f * Pi);
    if (AbsCosTheta(wi) > 0) fsum /= AbsCosTheta(wi);
    CHECK(!std::isinf(fsum.Average()) && !std::isnan(fsum.Average()));
    return fsum;
}

pstd::array<Float, pMax + 1> HairBxDF::ComputeApPDF(Float cosTheta_o) const {
    // Compute array of $A_p$ values for _cosTheta_o_
    Float sinTheta_o = SafeSqrt(1 - cosTheta_o * cosTheta_o);

    // Compute $\cos \thetat$ for refracted ray
    Float sinTheta_t = sinTheta_o / eta;
    Float cosTheta_t = SafeSqrt(1 - Sqr(sinTheta_t));

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float cosGamma_t = SafeSqrt(1 - Sqr(sinGamma_t));

    // Compute the transmittance _T_ of a single path through the cylinder
    SampledSpectrum T = Exp(-sigma_a * (2 * cosGamma_t / cosTheta_t));
    pstd::array<SampledSpectrum, pMax + 1> ap = Ap(cosTheta_o, eta, h, T);

    // Compute $A_p$ PDF from individual $A_p$ terms
    pstd::array<Float, pMax + 1> apPDF;
    Float sumY = 0;
    for (const SampledSpectrum &as : ap)
        sumY += as.Average();
    for (int i = 0; i <= pMax; ++i)
        apPDF[i] = ap[i].Average() / sumY;
    return apPDF;
}

pstd::optional<BSDFSample> HairBxDF::Sample_f(const Vector3f &wo, Float uc,
                                              const Point2f &u, BxDFReflTransFlags sampleFlags) const {
    // TODO: maybe handle flags, but... unclear how.

    // Compute hair coordinate system terms related to _wo_
    Float sinTheta_o = wo.x;
    Float cosTheta_o = SafeSqrt(1 - Sqr(sinTheta_o));
    Float phi_o = std::atan2(wo.z, wo.y);

    // Determine which term $p$ to sample for hair scattering
    pstd::array<Float, pMax + 1> apPDF = ComputeApPDF(cosTheta_o);
    int p = SampleDiscrete(apPDF, uc, nullptr, &uc);

    // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
    Float sinThetap_o, cosThetap_o;
    if (p == 0) {
        sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
        cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
    }
    else if (p == 1) {
        sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
        cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
    } else if (p == 2) {
        sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
        cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
    } else {
        sinThetap_o = sinTheta_o;
        cosThetap_o = cosTheta_o;
    }

    // Sample $M_p$ to compute $\thetai$
    Float cosTheta =
        1 + v[p] * std::log(std::max<Float>(u[0], 1e-5) + (1 - u[0]) * std::exp(-2 / v[p]));
    Float sinTheta = SafeSqrt(1 - Sqr(cosTheta));
    Float cosPhi = std::cos(2 * Pi * u[1]);
    Float sinTheta_i = -cosTheta * sinThetap_o + sinTheta * cosPhi * cosThetap_o;
    Float cosTheta_i = SafeSqrt(1 - Sqr(sinTheta_i));

    // Sample $N_p$ to compute $\Delta\phi$

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float gamma_t = SafeASin(sinGamma_t);
    Float dphi;
    if (p < pMax)
        dphi =
            Phi(p, gamma_o, gamma_t) + SampleTrimmedLogistic(uc, s, -Pi, Pi);
    else
        dphi = 2 * Pi * uc;

    // Compute _wi_ from sampled hair scattering angles
    Float phi_i = phi_o + dphi;
    Vector3f wi(sinTheta_i, cosTheta_i * std::cos(phi_i),
                cosTheta_i * std::sin(phi_i));

    // Compute PDF for sampled hair scattering direction _wi_
    Float pdf = 0;
    for (int p = 0; p < pMax; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sinThetap_o, cosThetap_o;
        if (p == 0) {
            sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
            cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
        }

        // Handle remainder of $p$ values for hair scale tilt
        else if (p == 1) {
            sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
            cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
            cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
        } else {
            sinThetap_o = sinTheta_o;
            cosThetap_o = cosTheta_o;
        }

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cosThetap_o = std::abs(cosThetap_o);
        pdf += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) *
            apPDF[p] * Np(dphi, p, s, gamma_o, gamma_t);
    }
    pdf += Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) *
            apPDF[pMax] * (1 / (2 * Pi));
    // if (std::abs(wi->x) < .9999) CHECK_NEAR(*pdf, PDF(wo, *wi), .01);
    return BSDFSample(f(wo, wi), wi, pdf, Flags());
}

Float HairBxDF::PDF(const Vector3f &wo, const Vector3f &wi, BxDFReflTransFlags sampleFlags) const {
    // TODO? flags...

    // Compute hair coordinate system terms related to _wo_
    Float sinTheta_o = wo.x;
    Float cosTheta_o = SafeSqrt(1 - Sqr(sinTheta_o));
    Float phi_o = std::atan2(wo.z, wo.y);

    // Compute hair coordinate system terms related to _wi_
    Float sinTheta_i = wi.x;
    Float cosTheta_i = SafeSqrt(1 - Sqr(sinTheta_i));
    Float phi_i = std::atan2(wi.z, wi.y);

    // Compute $\gammat$ for refracted ray
    Float etap = SafeSqrt(eta * eta - Sqr(sinTheta_o)) / cosTheta_o;
    Float sinGamma_t = h / etap;
    Float gamma_t = SafeASin(sinGamma_t);

    // Compute PDF for $A_p$ terms
    pstd::array<Float, pMax + 1> apPDF = ComputeApPDF(cosTheta_o);

    // Compute PDF sum for hair scattering events
    Float phi = phi_i - phi_o;
    Float pdf = 0;
    for (int p = 0; p < pMax; ++p) {
        // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
        Float sinThetap_o, cosThetap_o;
        if (p == 0) {
            sinThetap_o = sinTheta_o * cos2kAlpha[1] - cosTheta_o * sin2kAlpha[1];
            cosThetap_o = cosTheta_o * cos2kAlpha[1] + sinTheta_o * sin2kAlpha[1];
        }

        // Handle remainder of $p$ values for hair scale tilt
        else if (p == 1) {
            sinThetap_o = sinTheta_o * cos2kAlpha[0] + cosTheta_o * sin2kAlpha[0];
            cosThetap_o = cosTheta_o * cos2kAlpha[0] - sinTheta_o * sin2kAlpha[0];
        } else if (p == 2) {
            sinThetap_o = sinTheta_o * cos2kAlpha[2] + cosTheta_o * sin2kAlpha[2];
            cosThetap_o = cosTheta_o * cos2kAlpha[2] - sinTheta_o * sin2kAlpha[2];
        } else {
            sinThetap_o = sinTheta_o;
            cosThetap_o = cosTheta_o;
        }

        // Handle out-of-range $\cos \thetao$ from scale adjustment
        cosThetap_o = std::abs(cosThetap_o);
        pdf += Mp(cosTheta_i, cosThetap_o, sinTheta_i, sinThetap_o, v[p]) *
               apPDF[p] * Np(phi, p, s, gamma_o, gamma_t);
    }
    pdf += Mp(cosTheta_i, cosTheta_o, sinTheta_i, sinTheta_o, v[pMax]) *
           apPDF[pMax] * (1 / (2 * Pi));
    return pdf;
}

std::string HairBxDF::ToString() const {
    return StringPrintf("[ Hair h: %f gamma_o: %f eta: %f beta_m: %f beta_n: %f "
                        "v[0]: %f s: %f sigma_a: %s ]", h, gamma_o, eta, beta_m,
                        beta_n, v[0], s, sigma_a);
}

RGBSpectrum HairBxDF::SigmaAFromConcentration(Float ce, Float cp) {
    RGB eumelaninSigmaA(0.419f, 0.697f, 1.37f);
    RGB pheomelaninSigmaA(0.187f, 0.4f, 1.05f);
    RGB sigma_a = ce * eumelaninSigmaA + cp * pheomelaninSigmaA;
#ifdef __CUDA_ARCH__
    // FIXME
    assert(false);
    return RGBSpectrum();
#else
    return RGBSpectrum(*RGBColorSpace::sRGB, sigma_a);
#endif
}

SampledSpectrum HairBxDF::SigmaAFromReflectance(const SampledSpectrum &c, Float beta_n,
                                                const SampledWavelengths &lambda) {
    SampledSpectrum sigma_a;
    for (int i = 0; i < NSpectrumSamples; ++i)
        sigma_a[i] = Sqr(std::log(c[i]) /
                         (5.969f - 0.215f * beta_n + 2.532f * Sqr(beta_n) -
                          10.73f * Pow<3>(beta_n) + 5.574f * Pow<4>(beta_n) +
                          0.245f * Pow<5>(beta_n)));
    return sigma_a;
}

// *****************************************************************************
// Tensor file I/O
// *****************************************************************************

class Tensor {
public:
    // Data type of the tensor's fields
    enum Type {
        /* Invalid/unspecified */
        Invalid = 0,

        /* Signed and unsigned integer values */
        UInt8,  Int8,
        UInt16, Int16,
        UInt32, Int32,
        UInt64, Int64,

        /* Floating point values */
        Float16, Float32, Float64,
    };

    struct Field {
        // Data type of the tensor's fields
        Type dtype;

        // Offset in the file
        size_t offset;

        /// Specifies both rank and size along each dimension
        std::vector<size_t> shape;

        /// Pointer to the start of the tensor
        std::unique_ptr<uint8_t[]> data;
    };

    /// Load a tensor file into memory
    Tensor(const std::string &filename);

    /// Does the file contain a field of the specified name?
    bool has_field(const std::string &name) const;

    /// Return a data structure with information about the specified field
    const Field &field(const std::string &name) const;

    /// Return a human-readable summary
    std::string ToString() const;

    /// Return the total size of the tensor's data
    size_t size() const { return m_size; }

    std::string filename() const { return m_filename; }

private:
    std::unordered_map<std::string, Field> m_fields;
    std::string m_filename;
    size_t m_size;
};

static std::ostream &operator<<(std::ostream &os, Tensor::Type value) {
    switch(value) {
        case Tensor::Invalid:  os << "invalid"; break;
        case Tensor::UInt8 :   os << "uint8_t"; break;
        case Tensor::Int8:     os << "int8_t"; break;
        case Tensor::UInt16:   os << "uint16_t"; break;
        case Tensor::Int16:    os << "int16_t"; break;
        case Tensor::UInt32:   os << "uint32_t"; break;
        case Tensor::Int32:    os << "int8_t"; break;
        case Tensor::UInt64:   os << "uint64_t"; break;
        case Tensor::Int64:    os << "int64_t"; break;
        case Tensor::Float16:  os << "float16_t"; break;
        case Tensor::Float32:  os << "float32_t"; break;
        case Tensor::Float64:  os << "float64_t"; break;
        default:               os << "unkown"; break;
    }
    return os;
}

static size_t type_size(Tensor::Type value) {
    switch(value) {
        case Tensor::Invalid:  return 0; break;
        case Tensor::UInt8 :   return 1; break;
        case Tensor::Int8:     return 1; break;
        case Tensor::UInt16:   return 2; break;
        case Tensor::Int16:    return 2; break;
        case Tensor::UInt32:   return 4; break;
        case Tensor::Int32:    return 4; break;
        case Tensor::UInt64:   return 8; break;
        case Tensor::Int64:    return 8; break;
        case Tensor::Float16:  return 2; break;
        case Tensor::Float32:  return 4; break;
        case Tensor::Float64:  return 8; break;
        default:               return 0; break;
    }
}

Tensor::Tensor(const std::string &filename) : m_filename(filename) {
    // Helpful macros to limit error-handling code duplication
#ifdef ASSERT
#undef ASSERT
#endif // ASSERT

    #define ASSERT(cond, msg)                              \
        do {                                               \
            if (!(cond)) {                                 \
                fclose(file);                              \
                ErrorExit("%s: Tensor: " msg, filename);   \
            }                                              \
        } while(0)

    #define SAFE_READ(vars, size, count) \
        ASSERT(fread(vars, size, count, file) == (count), "Unable to read " #vars ".")

    FILE *file = fopen(filename.c_str(), "rb");
    if (file == NULL)
        ErrorExit("%s: unable to open file", filename);

    ASSERT(!fseek(file, 0, SEEK_END), "Unable to seek to end of file.");

    long size = ftell(file);
    ASSERT(size != -1, "Unable to tell file cursor position.");
    m_size = static_cast<size_t>(size);
    rewind(file);

    ASSERT(m_size >= 12 + 2 + 4, "Invalid tensor file: too small, truncated?");

    uint8_t header[12], version[2];
    uint32_t n_fields;
    SAFE_READ(header, sizeof(*header), 12);
    SAFE_READ(version, sizeof(*version), 2);
    SAFE_READ(&n_fields, sizeof(n_fields), 1);

    ASSERT(memcmp(header, "tensor_file", 12) == 0, "Invalid tensor file: invalid header.");
    ASSERT(version[0] == 1 && version[1] == 0, "Invalid tensor file: unknown file version.");

    for (uint32_t i = 0; i < n_fields; ++i) {
        uint8_t dtype;
        uint16_t name_length, ndim;
        uint64_t offset;

        SAFE_READ(&name_length, sizeof(name_length), 1);
        std::string name(name_length, '\0');
        SAFE_READ((char*)name.data(), 1, name_length);
        SAFE_READ(&ndim, sizeof(ndim), 1);
        SAFE_READ(&dtype, sizeof(dtype), 1);
        SAFE_READ(&offset, sizeof(offset), 1);
        ASSERT(dtype != Invalid && dtype <= Float64, "Invalid tensor file: unknown type.");

        std::vector<size_t> shape(ndim);
        size_t total_size = type_size((Type)dtype);       // no need to check here, line 43 already removes invalid types
        for (size_t j = 0; j < (size_t) ndim; ++j) {
            uint64_t size_value;
            SAFE_READ(&size_value, sizeof(size_value), 1);
            shape[j] = (size_t) size_value;
            total_size *= shape[j];
        }

        auto data = std::unique_ptr<uint8_t[]>(new uint8_t[total_size]);

        long cur_pos = ftell(file);
        ASSERT(cur_pos != -1, "Unable to tell current cursor position.");
        ASSERT(fseek(file, offset, SEEK_SET) != -1, "Unable to seek to tensor offset.");
        SAFE_READ(data.get(), 1, total_size);
        ASSERT(fseek(file, cur_pos, SEEK_SET) != -1, "Unable to seek back to current position");

        m_fields[name] =
            Field{ (Type) dtype, static_cast<size_t>(offset), shape, std::move(data) };
    }

    fclose(file);

    #undef SAFE_READ
    #undef ASSERT
}

/// Does the file contain a field of the specified name?
bool Tensor::has_field(const std::string &name) const {
    return m_fields.find(name) != m_fields.end();
}

/// Return a data structure with information about the specified field
const Tensor::Field &Tensor::field(const std::string &name) const {
    auto it = m_fields.find(name);
    CHECK(it != m_fields.end());
    return it->second;
}

/// Return a human-readable summary
std::string Tensor::ToString() const {
    std::ostringstream oss;
    oss << "Tensor[" << std::endl
        << "  filename = \"" << m_filename << "\"," << std::endl
        << "  size = " << size() << "," << std::endl
        << "  fields = {" << std::endl;

    size_t ctr = 0;
    for (const auto &it : m_fields) {
        oss << "    \"" << it.first << "\"" << " => [" << std::endl
            << "      dtype = " << it.second.dtype << "," << std::endl
            << "      offset = " << it.second.offset << "," << std::endl
            << "      shape = [";
        const auto& shape = it.second.shape;
        for (size_t j = 0; j < shape.size(); ++j) {
            oss << shape[j];
            if (j + 1 < shape.size())
                oss << ", ";
        }

        oss << "]" << std::endl;

        oss << "    ]";
        if (++ctr < m_fields.size())
            oss << ",";
        oss << std::endl;

    }

    oss << "  }" << std::endl
        << "]";

    return oss.str();
}

class MeasuredBRDFData {
public:
    // only public so allocator can call it...
    MeasuredBRDFData(Allocator alloc)
        : ndf(alloc), sigma(alloc), vndf(alloc), luminance(alloc), spectra(alloc),
          wavelengths(alloc) { }

    static MeasuredBRDFData *Create(const std::string &filename, Allocator alloc);

    std::string ToString() const {
        return StringPrintf("[ MeasuredBRDFData filename: %s ]", filename);
    }

    using Warp2D0 = Marginal2D<0>;
    using Warp2D2 = Marginal2D<2>;
    using Warp2D3 = Marginal2D<3>;

    Warp2D0 ndf;
    Warp2D0 sigma;
    Warp2D2 vndf;
    Warp2D2 luminance;
    Warp2D3 spectra;
    pstd::vector<float> wavelengths;
    bool isotropic;
    bool jacobian;
    std::string filename;
};

STAT_MEMORY_COUNTER("Memory/Measured BRDF data", measuredBRDFBytes);

MeasuredBRDFData *MeasuredBRDFData::Create(const std::string &filename,
                                           Allocator alloc) {
    Tensor tf = Tensor(filename);
    auto& theta_i = tf.field("theta_i");
    auto& phi_i = tf.field("phi_i");
    auto& ndf = tf.field("ndf");
    auto& sigma = tf.field("sigma");
    auto& vndf = tf.field("vndf");
    auto& spectra = tf.field("spectra");
    auto& luminance = tf.field("luminance");
    auto& wavelengths = tf.field("wavelengths");
    auto& description = tf.field("description");
    auto& jacobian = tf.field("jacobian");

    if (!(description.shape.size() == 1 &&
          description.dtype == Tensor::UInt8 &&

          theta_i.shape.size() == 1 &&
          theta_i.dtype == Tensor::Float32 &&

          phi_i.shape.size() == 1 &&
          phi_i.dtype == Tensor::Float32 &&

          wavelengths.shape.size() == 1 &&
          wavelengths.dtype == Tensor::Float32 &&

          ndf.shape.size() == 2 &&
          ndf.dtype == Tensor::Float32 &&

          sigma.shape.size() == 2 &&
          sigma.dtype == Tensor::Float32 &&

          vndf.shape.size() == 4 &&
          vndf.dtype == Tensor::Float32 &&
          vndf.shape[0] == phi_i.shape[0] &&
          vndf.shape[1] == theta_i.shape[0] &&

          luminance.shape.size() == 4 &&
          luminance.dtype == Tensor::Float32 &&
          luminance.shape[0] == phi_i.shape[0] &&
          luminance.shape[1] == theta_i.shape[0] &&
          luminance.shape[2] == luminance.shape[3] &&

          spectra.dtype == Tensor::Float32 &&
          spectra.shape.size() == 5 &&
          spectra.shape[0] == phi_i.shape[0] &&
          spectra.shape[1] == theta_i.shape[0] &&
          spectra.shape[2] == wavelengths.shape[0] &&
          spectra.shape[3] == spectra.shape[4] &&

          luminance.shape[2] == spectra.shape[3] &&
          luminance.shape[3] == spectra.shape[4] &&

          jacobian.shape.size() == 1 &&
          jacobian.shape[0] == 1 &&
          jacobian.dtype == Tensor::UInt8)) {
        Error("%s: invalid BRDF file structure: %s", filename, tf);
        return nullptr;
    }

    MeasuredBRDFData *brdfData = alloc.new_object<MeasuredBRDFData>(alloc);
    brdfData->filename = filename;
    brdfData->isotropic = phi_i.shape[0] <= 2;
    brdfData->jacobian  = ((uint8_t *)jacobian.data.get())[0];

    if (!brdfData->isotropic) {
        float *phi_i_data = (float *)phi_i.data.get();
        int reduction = (int)std::rint((2 * Pi) /
                                       (phi_i_data[phi_i.shape[0] - 1] - phi_i_data[0]));
        if (reduction != 1)
            ErrorExit("%s: reduction %d (!= 1) not supported", filename, reduction);
    }

    /* Construct NDF interpolant data structure */
    brdfData->ndf = Warp2D0(alloc, (float *)ndf.data.get(), ndf.shape[1], ndf.shape[0],
        { }, { }, false, false
    );

    /* Construct projected surface area interpolant data structure */
    brdfData->sigma = Warp2D0(alloc, (float *)sigma.data.get(), sigma.shape[1], sigma.shape[0],
        { }, { }, false, false
    );

    /* Construct VNDF warp data structure */
    brdfData->vndf = Warp2D2(alloc, (float *)vndf.data.get(), vndf.shape[3], vndf.shape[2],
        {{ (int)phi_i.shape[0], (int)theta_i.shape[0] }},
        {{ (const float *)phi_i.data.get(), (const float *)theta_i.data.get() }}
    );

    /* Construct Luminance warp data structure */
    brdfData->luminance = Warp2D2(alloc, (float *)luminance.data.get(), luminance.shape[3], luminance.shape[2],
        {{ (int) phi_i.shape[0], (int) theta_i.shape[0] }},
        {{ (const float *)phi_i.data.get(), (const float *)theta_i.data.get() }}
    );

    /* Copy wavelength information */
    size_t size = wavelengths.shape[0];
    brdfData->wavelengths.resize(size);
    for (size_t i = 0; i < size; ++i)
        brdfData->wavelengths[i] = ((const float *)wavelengths.data.get())[i];

    /* Construct spectral interpolant */
    brdfData->spectra = Warp2D3(alloc, (float *)spectra.data.get(), spectra.shape[4], spectra.shape[3],
        {{ (int)phi_i.shape[0], (int)theta_i.shape[0], (int)wavelengths.shape[0] }},
        {{ (const float *)phi_i.data.get(), (const float *)theta_i.data.get(),
           (const float *)wavelengths.data.get() }},
        false, false
    );

    measuredBRDFBytes += sizeof(MeasuredBRDFData) + 4 * brdfData->wavelengths.size() +
        brdfData->ndf.BytesUsed() + brdfData->sigma.BytesUsed() +
        brdfData->vndf.BytesUsed() + brdfData->luminance.BytesUsed() +
        brdfData->spectra.BytesUsed();

    return brdfData;
}

MeasuredBRDFData *MeasuredBxDF::BRDFDataFromFile(const std::string &filename,
                                                 Allocator alloc) {
    static std::map<std::string, MeasuredBRDFData *> loadedData;
    if (loadedData.find(filename) == loadedData.end())
        loadedData[filename] = MeasuredBRDFData::Create(filename, alloc);
    return loadedData[filename];
}

Float MeasuredBxDF::PDF(const Vector3f &wo, const Vector3f &wi,
                        BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection))
        return 0;
    if (!SameHemisphere(wo, wi))
        return 0;
    if (wo.z < 0)
        return PDF(-wo, -wi, sampleFlags);

    Vector3f wm = wi + wo;
    if (LengthSquared(wm) == 0)
        return 0;
    wm = Normalize(wm);

    /* Cartesian -> spherical coordinates */
    Float theta_i = SphericalTheta(wi), phi_i = std::atan2(wi.y, wi.x);
    Float theta_m = SphericalTheta(wm), phi_m = std::atan2(wm.y, wm.x);

    /* Spherical coordinates -> unit coordinate system */
    Vector2f u_wm(theta2u(theta_m),
                  phi2u(brdfData->isotropic ? (phi_m - phi_i) : phi_m));
    u_wm.y = u_wm.y - std::floor(u_wm.y);

    Float params[2] = { phi_i, theta_i };
    auto ui = brdfData->vndf.invert(u_wm, params);
    Vector2f sample = ui.p;
    Float vndfPDF = ui.pdf;

    Float pdf = brdfData->luminance.eval(sample, params);
    Float sinTheta_m = std::sqrt(Sqr(wm.x) + Sqr(wm.y));
    Float jacobian = 4.f * Dot(wi, wm) *
        std::max<Float>(2 * Sqr(Pi) * u_wm.x * sinTheta_m, 1e-6f);
    return vndfPDF * pdf / jacobian;
}

SampledSpectrum MeasuredBxDF::f(const Vector3f &wo, const Vector3f &wi) const {
    if (!SameHemisphere(wo, wi))
        return SampledSpectrum(0.);
    if (wo.z < 0)
        return f(-wo, -wi);

    Vector3f wm = wi + wo;
    if (LengthSquared(wm) == 0)
        return SampledSpectrum(0);
    wm = Normalize(wm);

    /* Cartesian -> spherical coordinates */
    Float theta_i = SphericalTheta(wi), phi_i = std::atan2(wi.y, wi.x);
    Float theta_m = SphericalTheta(wm), phi_m = std::atan2(wm.y, wm.x);

    /* Spherical coordinates -> unit coordinate system */
    Vector2f u_wi(theta2u(theta_i), phi2u(phi_i));
    Vector2f u_wm(theta2u(theta_m),
                  phi2u(brdfData->isotropic ? (phi_m - phi_i) : phi_m));
    u_wm.y = u_wm.y - std::floor(u_wm.y);

    Float params[2] = { phi_i, theta_i };
    auto ui = brdfData->vndf.invert(u_wm, params);
    Vector2f sample = ui.p;
    Float vndfPDF = ui.pdf;

    SampledSpectrum fr(0);
    for (int i = 0; i < pbrt::NSpectrumSamples; ++i) {
        Float params_fr[3] = { phi_i, theta_i, lambda[i] };
        fr[i] = brdfData->spectra.eval(sample, params_fr);
        CHECK_RARE(1e-6, fr[i] < 0);
        fr[i] = std::max<Float>(0, fr[i]);
    }

    return fr * brdfData->ndf.eval(u_wm, params) /
        (4 * brdfData->sigma.eval(u_wi, params) * AbsCosTheta(wi));
}

pstd::optional<BSDFSample> MeasuredBxDF::Sample_f(const Vector3f &wo, Float uc,
                                            const Point2f &u, BxDFReflTransFlags sampleFlags) const {
    if (!(sampleFlags & BxDFReflTransFlags::Reflection)) return {};

    if (wo.z <= 0) {
        pstd::optional<BSDFSample> sample = Sample_f(-wo, uc, u, sampleFlags);
        if (sample) sample->wi = -sample->wi;
        return sample;
    }

    Float theta_i = SphericalTheta(wo), phi_i = std::atan2(wo.y, wo.x);

    Vector2f sample = Vector2f(u.y, u.x);
    Float params[2] = { phi_i, theta_i };
    auto s = brdfData->luminance.sample(sample, params);
    sample = s.p;
    Float lumPDF = s.pdf;

    s = brdfData->vndf.sample(sample, params);
    Vector2f u_wm = s.p;
    Float ndfPDF = s.pdf;

    Float phi_m = u2phi(u_wm.y), theta_m = u2theta(u_wm.x);
    if (brdfData->isotropic)
        phi_m += phi_i;

    /* Spherical -> Cartesian coordinates */
    Float sinTheta_m = std::sin(theta_m), cosTheta_m = std::cos(theta_m);
    Vector3f wm = SphericalDirection(sinTheta_m, cosTheta_m, phi_m);

    Vector3f wi = Reflect(wo, wm);
    if (wi.z <= 0)
        return {};

    SampledSpectrum fr(0);
    for (int i = 0; i < pbrt::NSpectrumSamples; ++i) {
        Float params_fr[3] = { phi_i, theta_i, lambda[i] };
        fr[i] = brdfData->spectra.eval(sample, params_fr);
        CHECK_RARE(1e-6, fr[i] < 0);
        fr[i] = std::max<Float>(0, fr[i]);
    }

    Vector2f u_wo = Vector2f(theta2u(theta_i), phi2u(phi_i));
    fr *= brdfData->ndf.eval(u_wm, params) /
        (4 * brdfData->sigma.eval(u_wo, params) * AbsCosTheta(wi));

    Float jacobian = 4 * Dot(wo, wm) *
        std::max<Float>(2 * Sqr(Pi) * u_wm.x * sinTheta_m, 1e-6f);
    Float pdf = ndfPDF * lumPDF / jacobian;

    return BSDFSample(fr, wi, pdf, BxDFFlags::GlossyReflection);
}

std::string MeasuredBxDF::ToString() const {
    return StringPrintf("[ MeasuredBRDF brdfData: %s ]", *brdfData);
}

SampledSpectrum SeparableBSSRDFAdapter::f(const Vector3f &wo, const Vector3f &wi) const {
    SampledSpectrum f = bssrdf->Sw(wi);
    // Update BSSRDF transmission term to account for adjoint light
    // transport
    if (bssrdf->mode == TransportMode::Radiance)
        f *= bssrdf->eta * bssrdf->eta;
    return f;
}


SampledSpectrum BxDFHandle::rho(const Vector3f &wo, pstd::span<const Float> uc,
                                pstd::span<const Point2f> u2) const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->rho(wo, uc, u2);
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->rho(wo, uc, u2);

    case TypeIndex<DisneyBxDF>():
    case TypeIndex<CoatedDiffuseBxDF>():
    case TypeIndex<GeneralLayeredBxDF>():
    case TypeIndex<DielectricInterfaceBxDF>():
    case TypeIndex<ThinDielectricBxDF>():
    case TypeIndex<SpecularReflectionBxDF>():
    case TypeIndex<SpecularTransmissionBxDF>():
    case TypeIndex<HairBxDF>():
    case TypeIndex<MeasuredBxDF>():
    case TypeIndex<MicrofacetReflectionBxDF>():
    case TypeIndex<MicrofacetTransmissionBxDF>():
    case TypeIndex<SeparableBSSRDFAdapter>(): {
        if (wo.z == 0) return SampledSpectrum(0.f);
        SampledSpectrum r(0.);
        CHECK_EQ(uc.size(), u2.size());
        for (size_t i = 0; i < uc.size(); ++i) {
            // Estimate one term of $\rho_\roman{hd}$
            auto bs = Sample_f(wo, uc[i], u2[i]);
            if (bs && bs->pdf > 0) r += bs->f * AbsCosTheta(bs->wi) / bs->pdf;
        }
        return r / uc.size();
    }
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

SampledSpectrum BxDFHandle::rho(pstd::span<const Float> uc1, pstd::span<const Point2f> u1,
                                pstd::span<const Float> uc2, pstd::span<const Point2f> u2) const {
    switch (Tag()) {
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->rho(uc1, u1, uc2, u2);
    case TypeIndex<LambertianBxDF>():
    case TypeIndex<CoatedDiffuseBxDF>():
    case TypeIndex<GeneralLayeredBxDF>():
    case TypeIndex<DielectricInterfaceBxDF>():
    case TypeIndex<ThinDielectricBxDF>():
    case TypeIndex<SpecularReflectionBxDF>():
    case TypeIndex<SpecularTransmissionBxDF>():
    case TypeIndex<HairBxDF>():
    case TypeIndex<MeasuredBxDF>():
    case TypeIndex<MicrofacetReflectionBxDF>():
    case TypeIndex<MicrofacetTransmissionBxDF>():
    case TypeIndex<DisneyBxDF>():
    case TypeIndex<SeparableBSSRDFAdapter>(): {
        DCHECK_EQ(uc1.size(), u1.size());
        DCHECK_EQ(uc2.size(), u2.size());
        DCHECK_EQ(u1.size(), u2.size());
        SampledSpectrum r(0.f);
        for (size_t i = 0; i < uc1.size(); ++i) {
            // Estimate one term of $\rho_\roman{hh}$
            Vector3f wo = SampleUniformHemisphere(u1[i]);
            if (wo.z == 0) continue;
            Float pdfo = UniformHemispherePDF();
            auto bs = Sample_f(wo, uc2[i], u2[i]);
            if (bs && bs->pdf > 0)
                r += bs->f * AbsCosTheta(bs->wi) * AbsCosTheta(wo) / (pdfo * bs->pdf);
        }
        return r / (Pi * u1.size());
    }
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

std::string BxDFHandle::ToString() const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->ToString();
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->ToString();
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->ToString();
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->ToString();
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->ToString();
    case TypeIndex<SpecularReflectionBxDF>():
        return Cast<SpecularReflectionBxDF>()->ToString();
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->ToString();
    case TypeIndex<HairBxDF>():
        return Cast<HairBxDF>()->ToString();
    case TypeIndex<MeasuredBxDF>():
        return Cast<MeasuredBxDF>()->ToString();
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->ToString();
    case TypeIndex<MicrofacetReflectionBxDF>():
        return Cast<MicrofacetReflectionBxDF>()->ToString();
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->ToString();
    case TypeIndex<DisneyBxDF>():
        return Cast<DisneyBxDF>()->ToString();
    case TypeIndex<SeparableBSSRDFAdapter>():
        return Cast<SeparableBSSRDFAdapter>()->ToString();
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

BxDFHandle BxDFHandle::Regularize(MaterialBuffer &materialBuffer) {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>();
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->Regularize(materialBuffer);
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->Regularize(materialBuffer);
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->Regularize(materialBuffer);
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->Regularize(materialBuffer);
    case TypeIndex<SpecularReflectionBxDF>():
        return Cast<SpecularReflectionBxDF>()->Regularize(materialBuffer);
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->Regularize(materialBuffer);
    case TypeIndex<HairBxDF>():
        return Cast<HairBxDF>();
    case TypeIndex<MeasuredBxDF>():
        return Cast<MeasuredBxDF>();
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->Regularize(materialBuffer);
    case TypeIndex<MicrofacetReflectionBxDF>():
        return Cast<MicrofacetReflectionBxDF>()->Regularize(materialBuffer);
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->Regularize(materialBuffer);
    case TypeIndex<DisneyBxDF>():
        return Cast<DisneyBxDF>();
    case TypeIndex<SeparableBSSRDFAdapter>():
        return Cast<SeparableBSSRDFAdapter>();
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

inline void BxDFHandle::FlipTransportMode() {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        break;
    case TypeIndex<CoatedDiffuseBxDF>():
        break;
    case TypeIndex<GeneralLayeredBxDF>():
        break;
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->FlipTransportMode();
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->FlipTransportMode();
    case TypeIndex<SpecularReflectionBxDF>():
        break;
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->FlipTransportMode();
    case TypeIndex<HairBxDF>():
        break;
    case TypeIndex<MeasuredBxDF>():
        break;
    case TypeIndex<MixBxDF>():
        break;
    case TypeIndex<MicrofacetReflectionBxDF>():
        break;
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->FlipTransportMode();
    case TypeIndex<DisneyBxDF>():
        break;
    case TypeIndex<SeparableBSSRDFAdapter>():
        break;
    default:
        LOG_FATAL("Unhandled BxDF type");
    }
}

bool BxDFHandle::PDFIsApproximate() const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return false;
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->PDFIsApproximate();
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->PDFIsApproximate();
    case TypeIndex<DielectricInterfaceBxDF>():
        return false;
    case TypeIndex<ThinDielectricBxDF>():
        return false;
    case TypeIndex<SpecularReflectionBxDF>():
        return false;
    case TypeIndex<SpecularTransmissionBxDF>():
        return false;
    case TypeIndex<HairBxDF>():
        return false;
    case TypeIndex<MeasuredBxDF>():
        return false;
    case TypeIndex<MixBxDF>():
        return false;
    case TypeIndex<MicrofacetReflectionBxDF>():
        return false;
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return false;
    case TypeIndex<DisneyBxDF>():
        return false;
    case TypeIndex<SeparableBSSRDFAdapter>():
        return false;
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

BxDFFlags BxDFHandle::Flags() const {
    switch (Tag()) {
    case TypeIndex<LambertianBxDF>():
        return Cast<LambertianBxDF>()->Flags();
    case TypeIndex<CoatedDiffuseBxDF>():
        return Cast<CoatedDiffuseBxDF>()->Flags();
    case TypeIndex<GeneralLayeredBxDF>():
        return Cast<GeneralLayeredBxDF>()->Flags();
    case TypeIndex<DielectricInterfaceBxDF>():
        return Cast<DielectricInterfaceBxDF>()->Flags();
    case TypeIndex<ThinDielectricBxDF>():
        return Cast<ThinDielectricBxDF>()->Flags();
    case TypeIndex<SpecularReflectionBxDF>():
        return Cast<SpecularReflectionBxDF>()->Flags();
    case TypeIndex<SpecularTransmissionBxDF>():
        return Cast<SpecularTransmissionBxDF>()->Flags();
    case TypeIndex<HairBxDF>():
        return Cast<HairBxDF>()->Flags();
    case TypeIndex<MeasuredBxDF>():
        return Cast<MeasuredBxDF>()->Flags();
    case TypeIndex<MixBxDF>():
        return Cast<MixBxDF>()->Flags();
    case TypeIndex<MicrofacetReflectionBxDF>():
        return Cast<MicrofacetReflectionBxDF>()->Flags();
    case TypeIndex<MicrofacetTransmissionBxDF>():
        return Cast<MicrofacetTransmissionBxDF>()->Flags();
    case TypeIndex<DisneyBxDF>():
        return Cast<DisneyBxDF>()->Flags();
    case TypeIndex<SeparableBSSRDFAdapter>():
        return Cast<SeparableBSSRDFAdapter>()->Flags();
    default:
        LOG_FATAL("Unhandled BxDF type");
        return {};
    }
}

template class LayeredBxDF<DielectricInterfaceBxDF, LambertianBxDF>;
template class LayeredBxDF<BxDFHandle, BxDFHandle>;

}  // namespace pbrt
