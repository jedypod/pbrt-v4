// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_LIGHT_H
#define PBRT_BASE_LIGHT_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// LightType Declarations
enum class LightType : int { DeltaPosition, DeltaDirection, Area, Infinite };

enum class LightSamplingMode { WithMIS, WithoutMIS };

// Light Declarations
class PointLight;
class DistantLight;
class ProjectionLight;
class GoniometricLight;
class DiffuseAreaLight;
class UniformInfiniteLight;
class ImageInfiniteLight;
class PortalImageInfiniteLight;
class SpotLight;

class AreaLight;
class InfiniteAreaLight;

class LightHandle
    : public TaggedPointer<PointLight, DistantLight, ProjectionLight, GoniometricLight,
                           SpotLight, DiffuseAreaLight, UniformInfiniteLight,
                           ImageInfiniteLight, PortalImageInfiniteLight> {
  public:
    using TaggedPointer::TaggedPointer;

    static LightHandle Create(const std::string &name,
                              const ParameterDictionary &parameters,
                              const AnimatedTransform &worldFromLight,
                              const CameraTransform &cameraTransform,
                              MediumHandle outsideMedium, const FileLoc *loc,
                              Allocator alloc);
    static LightHandle CreateArea(const std::string &name,
                                  const ParameterDictionary &parameters,
                                  const AnimatedTransform &worldFromLight,
                                  const MediumInterface &mediumInterface,
                                  const ShapeHandle shape, const FileLoc *loc,
                                  Allocator alloc);

    // These shouldn't be called. Add these to get decent error messages
    // when they are.
    PBRT_CPU_GPU
    LightHandle(const AreaLight *) = delete;
    PBRT_CPU_GPU
    LightHandle(const InfiniteAreaLight *) = delete;

    PBRT_CPU_GPU inline pstd::optional<LightLiSample> Sample_Li(
        const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda,
        LightSamplingMode mode = LightSamplingMode::WithoutMIS) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    void Preprocess(const Bounds3f &sceneBounds);

    PBRT_CPU_GPU inline Float Pdf_Li(
        const Interaction &ref, const Vector3f &wi,
        LightSamplingMode mode = LightSamplingMode::WithoutMIS) const;

    PBRT_CPU_GPU
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;

    // Note shouldn't be called for area lights..
    PBRT_CPU_GPU
    void Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU inline LightType Type() const;

    pstd::optional<LightBounds> Bounds() const;

    std::string ToString() const;

    // AreaLights only
    PBRT_CPU_GPU inline SampledSpectrum L(const Interaction &intr, const Vector3f &w,
                                          const SampledWavelengths &lambda) const;
    PBRT_CPU_GPU
    void Pdf_Le(const Interaction &intr, Vector3f &w, Float *pdfPos, Float *pdfDir) const;

    // InfiniteAreaLights only
    PBRT_CPU_GPU inline SampledSpectrum Le(const Ray &ray,
                                           const SampledWavelengths &lambda) const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_LIGHT_H
