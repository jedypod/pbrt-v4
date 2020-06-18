// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_LIGHTSAMPLER_H
#define PBRT_BASE_LIGHTSAMPLER_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

struct SampledLight {
    LightHandle light;
    Float pdf;

    std::string ToString() const;
};

class UniformLightSampler;
class PowerLightSampler;
class BVHLightSampler;
class ExhaustiveLightSampler;

class LightSamplerHandle : public TaggedPointer<UniformLightSampler, PowerLightSampler,
                                                BVHLightSampler, ExhaustiveLightSampler> {
  public:
    using TaggedPointer::TaggedPointer;

    static LightSamplerHandle Create(const std::string &name,
                                     pstd::span<const LightHandle> lights,
                                     Allocator alloc);

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(const Interaction &intr,
                                                            Float u) const;

    PBRT_CPU_GPU inline Float PDF(const Interaction &intr, LightHandle light) const;

    PBRT_CPU_GPU inline pstd::optional<SampledLight> Sample(Float u) const;

    PBRT_CPU_GPU inline Float PDF(LightHandle light) const;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHTSAMPLER_H
