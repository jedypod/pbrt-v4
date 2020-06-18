// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// core/bsdf.cpp*
#include <pbrt/bsdf.h>

#include <pbrt/util/spectrum.h>

namespace pbrt {

std::string BSDFSample::ToString() const {
    return StringPrintf("[ BSDFSample f: %s wi: %s pdf: %s flags: %s ]", f, wi, pdf,
                        flags);
}

// BSDF Method Definitions
SampledSpectrum BSDF::SampleSpecular_f(const Vector3f &wo, Vector3f *wi,
                                       BxDFReflTransFlags sampleFlags) const {
    pstd::optional<BSDFSample> s =
        Sample_f(wo, 0, Point2f(0, 0), TransportMode::Radiance, sampleFlags);
    if (!s || !s->f || s->pdf == 0)
        return SampledSpectrum(0);
    *wi = s->wi;
    return s->f / s->pdf;
}

std::string BSDF::ToString() const {
    return StringPrintf("[ BSDF eta: %f bxdf: %s shadingFrame: %s ng: %s ]", eta, bxdf,
                        shadingFrame, ng);
}

}  // namespace pbrt
