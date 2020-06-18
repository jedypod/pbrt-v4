// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_TEXTURE_H
#define PBRT_BASE_TEXTURE_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// Texture Declarations
class TextureEvalContext;

class FloatConstantTexture;
class FloatBilerpTexture;
class FloatCheckerboardTexture;
class FloatDotsTexture;
class FBmTexture;
class GPUFloatImageTexture;
class FloatImageTexture;
class FloatMixTexture;
class FloatPtexTexture;
class FloatScaledTexture;
class WindyTexture;
class WrinkledTexture;

class FloatTextureHandle
    : public TaggedPointer<FloatImageTexture, GPUFloatImageTexture, FloatMixTexture,
                           FloatScaledTexture, FloatConstantTexture, FloatBilerpTexture,
                           FloatCheckerboardTexture, FloatDotsTexture, FBmTexture,
                           FloatPtexTexture, WindyTexture, WrinkledTexture> {
  public:
    using TaggedPointer::TaggedPointer;

    static FloatTextureHandle Create(const std::string &name,
                                     const Transform &worldFromTexture,
                                     const TextureParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc, bool gpu);

    PBRT_CPU_GPU inline Float Evaluate(const TextureEvalContext &ctx) const;

    std::string ToString() const;
};

class SpectrumConstantTexture;
class SpectrumBilerpTexture;
class SpectrumCheckerboardTexture;
class SpectrumImageTexture;
class GPUSpectrumImageTexture;
class MarbleTexture;
class SpectrumMixTexture;
class SpectrumDotsTexture;
class SpectrumPtexTexture;
class SpectrumScaledTexture;
class UVTexture;

class SpectrumTextureHandle
    : public TaggedPointer<SpectrumImageTexture, GPUSpectrumImageTexture,
                           SpectrumMixTexture, SpectrumScaledTexture,
                           SpectrumConstantTexture, SpectrumBilerpTexture,
                           SpectrumCheckerboardTexture, MarbleTexture,
                           SpectrumDotsTexture, SpectrumPtexTexture, UVTexture> {
  public:
    using TaggedPointer::TaggedPointer;

    static SpectrumTextureHandle Create(const std::string &name,
                                        const Transform &worldFromTexture,
                                        const TextureParameterDictionary &parameters,
                                        const FileLoc *loc, Allocator alloc, bool gpu);

    PBRT_CPU_GPU inline SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                                                 const SampledWavelengths &lambda) const;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_TEXTURE_H
