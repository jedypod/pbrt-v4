// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_MATERIAL_H
#define PBRT_BASE_MATERIAL_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <map>
#include <string>

namespace pbrt {

// Material Declarations
class CoatedDiffuseMaterial;
class ConductorMaterial;
class DielectricMaterial;
class DiffuseMaterial;
class DiffuseTransmissionMaterial;
class HairMaterial;
class LayeredMaterial;
class MeasuredMaterial;
class SubsurfaceMaterial;
class ThinDielectricMaterial;

class MaterialHandle
    : public TaggedPointer<CoatedDiffuseMaterial, ConductorMaterial, DielectricMaterial,
                           DiffuseMaterial, DiffuseTransmissionMaterial, HairMaterial,
                           LayeredMaterial, MeasuredMaterial, SubsurfaceMaterial,
                           ThinDielectricMaterial> {
  public:
    using TaggedPointer::TaggedPointer;

    static MaterialHandle Create(
        const std::string &name, const TextureParameterDictionary &parameters,
        /*const */ std::map<std::string, MaterialHandle> &namedMaterials,
        const FileLoc *loc, Allocator alloc);

    // -> bool Matches(std::init_list<FloatTextureHandle>,
    // ...<SpectrumTextureHandle>
    // -> Float operator()(FloatTextureHandle tex, cont TextureEvalContext &ctx)
    // -> SampledSpectrum operator()(SpectrumTextureHandle tex, cont
    // TextureEvalContext &ctx, SampledWavelengths lambda)
    template <typename TextureEvaluator>
    PBRT_CPU_GPU inline BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                                      const SampledWavelengths &lambda,
                                      ScratchBuffer &scratchBuffer) const;

    template <typename TextureEvaluator>
    PBRT_CPU_GPU inline BSSRDFHandle GetBSSRDF(TextureEvaluator texEval,
                                               SurfaceInteraction &si,
                                               const SampledWavelengths &lambda,
                                               ScratchBuffer &scratchBuffer) const;

    PBRT_CPU_GPU inline FloatTextureHandle GetDisplacement() const;

    PBRT_CPU_GPU inline bool IsTransparent() const;

    PBRT_CPU_GPU inline bool HasSubsurfaceScattering() const;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_MATERIAL_H
