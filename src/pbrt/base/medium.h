// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_MEDIUM_H
#define PBRT_BASE_MEDIUM_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/taggedptr.h>

#include <string>
#include <vector>

namespace pbrt {

// Media Declarations
struct PhaseFunctionSample {
    Float p;
    Vector3f wi;
    Float pdf;
};

class HenyeyGreensteinPhaseFunction;

class PhaseFunctionHandle : public TaggedPointer<HenyeyGreensteinPhaseFunction> {
  public:
    using TaggedPointer::TaggedPointer;

    PBRT_CPU_GPU inline Float p(const Vector3f &wo, const Vector3f &wi) const;

    PBRT_CPU_GPU inline pstd::optional<PhaseFunctionSample> Sample_p(
        const Vector3f &wo, const Point2f &u) const;

    PBRT_CPU_GPU inline Float PDF(const Vector3f &wo, const Vector3f &wi) const;

    std::string ToString() const;
};

class HomogeneousMedium;
class GridDensityMedium;
struct NewMediumSample;

class MediumHandle : public TaggedPointer<HomogeneousMedium, GridDensityMedium> {
  public:
    using TaggedPointer::TaggedPointer;

    static MediumHandle Create(const std::string &name,
                               const ParameterDictionary &parameters,
                               const Transform &worldFromMedium, const FileLoc *loc,
                               Allocator alloc);

    PBRT_CPU_GPU inline MediumSample Sample(const Ray &ray, Float tMax, RNG &rng,
                                            const SampledWavelengths &lambda,
                                            ScratchBuffer &scratchBuffer) const;

    PBRT_CPU_GPU inline NewMediumSample SampleTmaj(
        const Ray &ray, Float tMax, Float u, const SampledWavelengths &lambda,
        ScratchBuffer *scratchBuffer /* optional */) const;

    bool IsEmissive() const;

    std::string ToString() const;
};

// MediumInterface Declarations
struct MediumInterface {
    PBRT_CPU_GPU
    MediumInterface() : inside(nullptr), outside(nullptr) {}
    // MediumInterface Public Methods
    PBRT_CPU_GPU
    MediumInterface(MediumHandle medium) : inside(medium), outside(medium) {}
    PBRT_CPU_GPU
    MediumInterface(MediumHandle inside, MediumHandle outside)
        : inside(inside), outside(outside) {}

    PBRT_CPU_GPU
    bool IsMediumTransition() const { return inside != outside; }

    MediumHandle inside, outside;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_MEDIUM_H
