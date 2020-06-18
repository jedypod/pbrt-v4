// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_BSSRDF_H
#define PBRT_BASE_BSSRDF_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

class BSSRDFSample;
class BSSRDFProbeSegment;

class TabulatedBSSRDF;

class BSSRDFHandle : public TaggedPointer<TabulatedBSSRDF> {
  public:
    using TaggedPointer::TaggedPointer;

    PBRT_CPU_GPU inline SampledSpectrum S(const SurfaceInteraction &pi,
                                          const Vector3f &wi);

    PBRT_CPU_GPU inline pstd::optional<BSSRDFProbeSegment> Sample(
        Float u1, const Point2f &u2) const;

    PBRT_CPU_GPU inline BSSRDFSample ProbeIntersectionToSample(
        const SurfaceInteraction &si, ScratchBuffer &scratchBuffer) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_BSSRDF_H
