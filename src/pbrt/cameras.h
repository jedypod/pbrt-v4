// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CAMERAS_ORTHOGRAPHIC_H
#define PBRT_CAMERAS_ORTHOGRAPHIC_H

// cameras/orthographic.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/film.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/samplers.h>
#include <pbrt/util/scattering.h>

#include <memory>
#include <string>
#include <vector>

namespace pbrt {

class CameraWiSample {
  public:
    CameraWiSample() = default;
    PBRT_CPU_GPU
    CameraWiSample(const SampledSpectrum &Wi, const Vector3f &wi, Float pdf,
                   Point2f pRaster, const Interaction &pRef, const Interaction &pLens)
        : Wi(Wi), wi(wi), pdf(pdf), pRaster(pRaster), pRef(pRef), pLens(pLens) {}

    SampledSpectrum Wi;
    Vector3f wi;
    Float pdf;
    Point2f pRaster;
    Interaction pRef, pLens;
};

struct CameraRay {
    Ray ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

struct CameraRayDifferential {
    RayDifferential ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

class CameraBase {
  public:
    PBRT_CPU_GPU
    FilmHandle GetFilm() const { return film; }

    PBRT_CPU_GPU
    void ApproximatedPdxy(const SurfaceInteraction &si) const;

    PBRT_CPU_GPU inline Float SampleTime(Float u) const {
        return Lerp(u, shutterOpen, shutterClose);
    }

    PBRT_CPU_GPU inline const CameraTransform &WorldFromCamera() const {
        return worldFromCamera;
    }

    void InitMetadata(ImageMetadata *metadata) const;

  protected:
    // Camera Public Data
    CameraTransform worldFromCamera;
    Float shutterOpen, shutterClose;
    FilmHandle film;
    MediumHandle medium;

    PBRT_CPU_GPU
    Ray WorldFromCamera(const Ray &r) const { return worldFromCamera.rotation(r); }

    PBRT_CPU_GPU
    RayDifferential WorldFromCamera(const RayDifferential &r) const {
        return worldFromCamera.rotation(r);
    }

    PBRT_CPU_GPU
    Vector3f WorldFromCamera(const Vector3f &v, Float time) const {
        return worldFromCamera.rotation(v, time);
    }

    PBRT_CPU_GPU
    Point3f WorldFromCamera(const Point3f &p, Float time) const {
        return worldFromCamera.rotation(p, time);
    }

    PBRT_CPU_GPU
    Vector3f CameraFromWorld(const Vector3f &v, Float time) const {
        return worldFromCamera.rotation.ApplyInverse(v, time);
    }

    PBRT_CPU_GPU
    Point3f CameraFromWorld(const Point3f &p, Float time) const {
        return worldFromCamera.rotation.ApplyInverse(p, time);
    }

    CameraBase() = default;
    CameraBase(const CameraTransform &worldFromCamera, Float shutterOpen,
               Float shutterClose, FilmHandle film, MediumHandle medium);

    PBRT_CPU_GPU
    static pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        CameraHandle camera, const CameraSample &sample,
        const SampledWavelengths &lambda);

    std::string ToString() const;

    PBRT_CPU_GPU
    void FindMinimumDifferentials(CameraHandle camera);

    Vector3f minPosDifferentialX, minPosDifferentialY;
    Vector3f minDirDifferentialX, minDirDifferentialY;
};

class ProjectiveCamera : public CameraBase {
  public:
    // ProjectiveCamera Public Methods
    ProjectiveCamera(const CameraTransform &worldFromCamera,
                     const Transform &screenFromCamera, const Bounds2f &screenWindow,
                     Float shutterOpen, Float shutterClose, Float lensRadius,
                     Float focalDistance, FilmHandle film, MediumHandle medium);
    void InitMetadata(ImageMetadata *metadata) const;

    //  protected:
    ProjectiveCamera() = default;
    std::string BaseToString() const;

    // ProjectiveCamera Protected Data
    Transform screenFromCamera, cameraFromRaster;
    Transform rasterFromScreen, screenFromRaster;
    Float lensRadius, focalDistance;
};

// OrthographicCamera Declarations
class OrthographicCamera : public ProjectiveCamera {
  public:
    // OrthographicCamera Public Methods
    OrthographicCamera(const CameraTransform &worldFromCamera,
                       const Bounds2f &screenWindow, Float shutterOpen,
                       Float shutterClose, Float lensRadius, Float focalDistance,
                       FilmHandle film, MediumHandle medium)
        : ProjectiveCamera(worldFromCamera, Orthographic(0, 1), screenWindow, shutterOpen,
                           shutterClose, lensRadius, focalDistance, film, medium) {
        // Compute differential changes in origin for orthographic camera rays
        dxCamera = cameraFromRaster(Vector3f(1, 0, 0));
        dyCamera = cameraFromRaster(Vector3f(0, 1, 0));

        minDirDifferentialX = minDirDifferentialY = Vector3f(0, 0, 0);
        minPosDifferentialX = dxCamera;
        minPosDifferentialY = dyCamera;
        // FindMinimumDifferentials();
    }
    static OrthographicCamera *Create(const ParameterDictionary &parameters,
                                      const CameraTransform &worldFromCamera,
                                      FilmHandle film, MediumHandle medium,
                                      const FileLoc *loc, Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const;
    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        const CameraSample &sample, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for OrthographicCamera");
        return {};
    }

    PBRT_CPU_GPU
    void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Pdf_We() unimplemented for OrthographicCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref,
                                             const Point2f &sample,
                                             const SampledWavelengths &lambda) const {
        LOG_FATAL("Sample_Wi() unimplemented for OrthographicCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // OrthographicCamera Private Data
    Vector3f dxCamera, dyCamera;
};

// PerspectiveCamera Declarations
class PerspectiveCamera : public ProjectiveCamera {
  public:
    PerspectiveCamera() = default;
    // PerspectiveCamera Public Methods
    PerspectiveCamera(const CameraTransform &worldFromCamera,
                      const Bounds2f &screenWindow, Float shutterOpen, Float shutterClose,
                      Float lensRadius, Float focalDistance, Float fov, FilmHandle film,
                      MediumHandle medium);

    static PerspectiveCamera *Create(const ParameterDictionary &parameters,
                                     const CameraTransform &worldFromCamera,
                                     FilmHandle film, MediumHandle medium,
                                     const FileLoc *loc, Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        const CameraSample &sample, const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const;
    PBRT_CPU_GPU
    void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;
    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref,
                                             const Point2f &sample,
                                             const SampledWavelengths &lambda) const;

    std::string ToString() const;

  private:
    // PerspectiveCamera Private Data
    Vector3f dxCamera, dyCamera;
    Float A;
    Float cosTotalWidth;
};

// SphericalCamera Declarations
class SphericalCamera : public CameraBase {
  public:
    enum Mapping { EquiRect, EquiArea };

    // SphericalCamera Public Methods
    SphericalCamera(const CameraTransform &worldFromCamera, Float shutterOpen,
                    Float shutterClose, FilmHandle film, MediumHandle medium,
                    Mapping mapping)
        : CameraBase(worldFromCamera, shutterOpen, shutterClose, film, medium),
          mapping(mapping) {
        FindMinimumDifferentials(this);
    }

    static SphericalCamera *Create(const ParameterDictionary &parameters,
                                   const CameraTransform &worldFromCamera,
                                   FilmHandle film, MediumHandle medium,
                                   const FileLoc *loc, Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        const CameraSample &sample, const SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for SphericalCamera");
        return {};
    }

    PBRT_CPU_GPU
    void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Pdf_We() unimplemented for SphericalCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref,
                                             const Point2f &sample,
                                             const SampledWavelengths &lambda) const {
        LOG_FATAL("Sample_Wi() unimplemented for SphericalCamera");
        return {};
    }

    std::string ToString() const;

  private:
    Mapping mapping;
};

// RealisticCamera Declarations
class RealisticCamera : public CameraBase {
  public:
    // RealisticCamera Public Methods
    RealisticCamera(const CameraTransform &worldFromCamera, Float shutterOpen,
                    Float shutterClose, Float apertureDiameter, Float focusDistance,
                    Float dispersionFactor, std::vector<Float> &lensData, FilmHandle film,
                    MediumHandle medium, Allocator alloc);

    static RealisticCamera *Create(const ParameterDictionary &parameters,
                                   const CameraTransform &worldFromCamera,
                                   FilmHandle film, MediumHandle medium,
                                   const FileLoc *loc, Allocator alloc = {});

    PBRT_CPU_GPU
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(
        const CameraSample &sample, const SampledWavelengths &lambda) const {
        return CameraBase::GenerateRayDifferential(this, sample, lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const {
        LOG_FATAL("We() unimplemented for RealisticCamera");
        return {};
    }

    PBRT_CPU_GPU
    void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
        LOG_FATAL("Pdf_We() unimplemented for RealisticCamera");
    }

    PBRT_CPU_GPU
    pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref,
                                             const Point2f &sample,
                                             const SampledWavelengths &lambda) const {
        LOG_FATAL("Sample_Wi() unimplemented for RealisticCamera");
        return {};
    }

    std::string ToString() const;

  private:
    // RealisticCamera Private Declarations
    struct LensElementInterface {
        Float curvatureRadius;
        Float thickness;
        Float eta;
        Float apertureRadius;

        std::string ToString() const;
    };

    // RealisticCamera Private Data
    Float dispersionFactor;
    pstd::vector<LensElementInterface> elementInterfaces;
    pstd::vector<Bounds2f> exitPupilBounds;

    // RealisticCamera Private Methods
    PBRT_CPU_GPU
    Float LensRearZ() const { return elementInterfaces.back().thickness; }
    PBRT_CPU_GPU
    Float LensFrontZ() const {
        Float zSum = 0;
        for (const LensElementInterface &element : elementInterfaces)
            zSum += element.thickness;
        return zSum;
    }
    PBRT_CPU_GPU
    Float RearElementRadius() const { return elementInterfaces.back().apertureRadius; }

    PBRT_CPU_GPU
    bool TraceLensesFromFilm(const Ray &rCamera, Ray *rOut, Float lambda = 550) const;

    PBRT_CPU_GPU
    static bool IntersectSphericalElement(Float radius, Float zCenter, const Ray &ray,
                                          Float *t, Normal3f *n) {
        // Compute _t0_ and _t1_ for ray--element intersection
        Point3f o = ray.o - Vector3f(0, 0, zCenter);
        Float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        Float B = 2 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        Float C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        Float t0, t1;
        if (!Quadratic(A, B, C, &t0, &t1))
            return false;

        // Select intersection $t$ based on ray direction and element curvature
        bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
        *t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
        if (*t < 0)
            return false;

        // Compute surface normal of element at ray intersection point
        *n = Normal3f(Vector3f(o + *t * ray.d));
        *n = FaceForward(Normalize(*n), -ray.d);
        return true;
    }

    PBRT_CPU_GPU
    bool TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const;

    void DrawLensSystem() const;
    void DrawRayPathFromFilm(const Ray &r, bool arrow, bool toOpticalIntercept) const;
    void DrawRayPathFromScene(const Ray &r, bool arrow, bool toOpticalIntercept) const;

    static void ComputeCardinalPoints(const Ray &rIn, const Ray &rOut, Float *p,
                                      Float *f);
    void ComputeThickLensApproximation(Float pz[2], Float f[2]) const;
    Float FocusThickLens(Float focusDistance);
    Float FocusBinarySearch(Float focusDistance);
    Float FocusDistance(Float filmDist);
    Bounds2f BoundExitPupil(Float pFilmX0, Float pFilmX1) const;
    void RenderExitPupil(Float sx, Float sy, const char *filename) const;

    PBRT_CPU_GPU
    Point3f SampleExitPupil(const Point2f &pFilm, const Point2f &lensSample,
                            Float *sampleBoundsArea) const;

    void TestExitPupilBounds() const;
};

inline pstd::optional<CameraRay> CameraHandle::GenerateRay(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    auto generate = [&](auto ptr) { return ptr->GenerateRay(sample, lambda); };
    return Apply<pstd::optional<CameraRay>>(generate);
}

inline FilmHandle CameraHandle::GetFilm() const {
    auto getfilm = [&](auto ptr) { return ptr->GetFilm(); };
    return Apply<FilmHandle>(getfilm);
}

inline Float CameraHandle::SampleTime(Float u) const {
    auto sample = [&](auto ptr) { return ptr->SampleTime(u); };
    return Apply<Float>(sample);
}

inline const CameraTransform &CameraHandle::WorldFromCamera() const {
    auto wfc = [&](auto ptr) -> const CameraTransform & { return ptr->WorldFromCamera(); };
    return Apply<const CameraTransform &>(wfc);
}

}  // namespace pbrt

#endif  // PBRT_CAMERAS_H
