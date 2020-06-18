// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_CAMERA_H
#define PBRT_BASE_CAMERA_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// Camera Declarations
class CameraRay;
class CameraRayDifferential;
class CameraWiSample;

struct CameraSample;

class PerspectiveCamera;
class OrthographicCamera;
class SphericalCamera;
class RealisticCamera;

class CameraTransform {
  public:
    CameraTransform() = default;
    explicit CameraTransform(const AnimatedTransform &worldFromCamera);

    AnimatedTransform rotation;
    Transform translation;

    bool HasScale() const { return rotation.HasScale(); }

    Transform GetTransform(Float time) const {
        return translation * rotation.Interpolate(time);
    }

    Transform GetInverse(Float time) const { return Inverse(GetTransform(time)); }

    Point3f ApplyInverseTranslation(const Point3f &p) const {
        return translation.ApplyInverse(p);
    }
};

std::string ToString(const CameraTransform &ct);

class CameraHandle : public TaggedPointer<PerspectiveCamera, OrthographicCamera,
                                          SphericalCamera, RealisticCamera> {
  public:
    using TaggedPointer::TaggedPointer;

    static CameraHandle Create(const std::string &name,
                               const ParameterDictionary &parameters, MediumHandle medium,
                               const CameraTransform &worldFromCamera, FilmHandle film,
                               const FileLoc *loc, Allocator alloc);

    pstd::optional<CameraRay> PBRT_CPU_GPU inline GenerateRay(
        const CameraSample &sample, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        const CameraSample &sample, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU inline FilmHandle GetFilm() const;

    PBRT_CPU_GPU inline Float SampleTime(Float u) const;

    PBRT_CPU_GPU inline const CameraTransform &WorldFromCamera() const;

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const;

    PBRT_CPU_GPU
    void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref, const Point2f &u,
                                             const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    void ApproximatedPdxy(const SurfaceInteraction &si) const;

    void InitMetadata(ImageMetadata *metadata) const;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_CAMERA_H
