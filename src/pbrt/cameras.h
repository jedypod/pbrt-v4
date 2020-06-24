
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CAMERAS_ORTHOGRAPHIC_H
#define PBRT_CAMERAS_ORTHOGRAPHIC_H

// cameras/orthographic.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/film.h>
#include <pbrt/util/profile.h>

#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// OrthographicCamera Declarations
class OrthographicCamera : public ProjectiveCamera {
  public:
    // OrthographicCamera Public Methods
    OrthographicCamera(const AnimatedTransform &worldFromCamera,
                       const Bounds2f &screenWindow, Float shutterOpen,
                       Float shutterClose, Float lensRadius,
                       Float focalDistance, std::unique_ptr<Film> film,
                       const Medium *medium)
        : ProjectiveCamera(worldFromCamera, Orthographic(0, 1), screenWindow,
                           shutterOpen, shutterClose, lensRadius, focalDistance,
                           std::move(film), medium) {
        // Compute differential changes in origin for orthographic camera rays
        dxCamera = cameraFromRaster(Vector3f(1, 0, 0));
        dyCamera = cameraFromRaster(Vector3f(0, 1, 0));

        minDirDifferentialX = minDirDifferentialY = Vector3f(0, 0, 0);
        minPosDifferentialX = dxCamera;
        minPosDifferentialY = dyCamera;
        // FindMinimumDifferentials();
    }
    static OrthographicCamera *Create(const ParameterDictionary &dict,
                                      const AnimatedTransform &worldFromCamera,
                                      std::unique_ptr<Film> film,
                                      const Medium *medium, Allocator alloc = {});

    PBRT_HOST_DEVICE
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const;
    pstd::optional<CameraRayDifferential> GenerateRayDifferential(const CameraSample &sample,
                                                                  const SampledWavelengths &lambda) const;

    std::string ToString() const;

  private:
    // OrthographicCamera Private Data
    Vector3f dxCamera, dyCamera;
};

// PerspectiveCamera Declarations
class PerspectiveCamera final : public ProjectiveCamera {
  public:
    PerspectiveCamera() = default;
    // PerspectiveCamera Public Methods
    PerspectiveCamera(const AnimatedTransform &worldFromCamera,
                      const Bounds2f &screenWindow, Float shutterOpen,
                      Float shutterClose, Float lensRadius, Float focalDistance,
                      Float fov, std::unique_ptr<Film> film, const Medium *medium);

    static PerspectiveCamera *Create(const ParameterDictionary &dict,
                                     const AnimatedTransform &worldFromCamera,
                                     std::unique_ptr<Film> film,
                                     const Medium *medium, Allocator alloc = {});

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const {
        ProfilerScope prof(ProfilePhase::GenerateCameraRay);
        // Compute raster and camera sample positions
        Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
        Point3f pCamera = cameraFromRaster(pFilm);

        Ray ray(Point3f(0, 0, 0), Normalize(Vector3f(pCamera)),
                Lerp(sample.time, shutterOpen, shutterClose), medium);
        // Modify ray for depth of field
        if (lensRadius > 0) {
            // Sample point on lens
            Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);

            // Compute point on plane of focus
            Float ft = focalDistance / ray.d.z;
            Point3f pFocus = ray(ft);

            // Update ray for effect of lens
            ray.o = Point3f(pLens.x, pLens.y, 0);
            ray.d = Normalize(pFocus - ray.o);
        }
        return CameraRay{worldFromCamera(ray)};
    }

    pstd::optional<CameraRayDifferential> GenerateRayDifferential(const CameraSample &sample,
                                                                  const SampledWavelengths &lambda) const;
    SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                       Point2f *pRaster2 = nullptr) const;
    void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;
    pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref, const Point2f &sample,
                                             const SampledWavelengths &lambda) const;

    std::string ToString() const;

  private:
    // PerspectiveCamera Private Data
    Vector3f dxCamera, dyCamera;
    Float A;
    Float cosTotalWidth;
};


// SphericalCamera Declarations
class SphericalCamera : public Camera {
  public:
    enum Mapping { EquiRect, EquiArea };

    // SphericalCamera Public Methods
    SphericalCamera(const AnimatedTransform &worldFromCamera, Float shutterOpen,
                    Float shutterClose, std::unique_ptr<Film> film, const Medium *medium,
                    Mapping mapping)
        : Camera(worldFromCamera, shutterOpen, shutterClose, std::move(film), medium),
          mapping(mapping) {
        FindMinimumDifferentials();
    }

    static SphericalCamera *Create(const ParameterDictionary &dict,
                                   const AnimatedTransform &worldFromCamera,
                                   std::unique_ptr<Film> film,
                                   const Medium *medium, Allocator alloc = {});

    PBRT_HOST_DEVICE
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const;

    std::string ToString() const;

  private:
    Mapping mapping;
};


// RealisticCamera Declarations
class RealisticCamera final : public Camera {
  public:
    // RealisticCamera Public Methods
    RealisticCamera(const AnimatedTransform &worldFromCamera, Float shutterOpen,
                    Float shutterClose, Float apertureDiameter,
                    Float focusDistance, Float dispersionFactor,
                    std::vector<Float> &lensData, std::unique_ptr<Film> film,
                    const Medium *medium, Allocator alloc);

    static RealisticCamera *Create(const ParameterDictionary &dict,
                                   const AnimatedTransform &worldFromCamera,
                                   std::unique_ptr<Film> film,
                                   const Medium *medium, Allocator alloc = {});

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<CameraRay> GenerateRay(const CameraSample &sample,
                                          const SampledWavelengths &lambda) const {
        ProfilerScope prof(ProfilePhase::GenerateCameraRay);
//CO        ++totalRays;
        // Find point on film, _pFilm_, corresponding to _sample.pFilm_
        Point2f s(sample.pFilm.x / film->fullResolution.x,
                  sample.pFilm.y / film->fullResolution.y);
        Point2f pFilm2 = film->PhysicalExtent().Lerp(s);
        Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

        // Trace ray from _pFilm_ through lens system
        Float exitPupilBoundsArea;
        Point3f pRear = SampleExitPupil(Point2f(pFilm.x, pFilm.y), sample.pLens,
                                        &exitPupilBoundsArea);
        Ray rFilm(pFilm, pRear - pFilm);
        Ray ray;
        if (!TraceLensesFromFilm(rFilm, &ray, lambda[0])) {
//CO            ++vignettedRays;
            return {};
        }

        // Finish initialization of _RealisticCamera_ ray
        ray.time = Lerp(sample.time, shutterOpen, shutterClose);
        ray.medium = medium;
        ray = worldFromCamera(ray);
        ray.d = Normalize(ray.d);

        if (dispersionFactor != 0)
            lambda.TerminateSecondaryWavelengths();

        // Return weighting for _RealisticCamera_ ray
        Float cosTheta = Normalize(rFilm.d).z;
        Float cos4Theta = (cosTheta * cosTheta) * (cosTheta * cosTheta);
        Float weight = (shutterClose - shutterOpen) *
            (cos4Theta * exitPupilBoundsArea) / (LensRearZ() * LensRearZ());

        return CameraRay{ray, SampledSpectrum(weight)};
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
    PBRT_HOST_DEVICE_INLINE
    Float LensRearZ() const { return elementInterfaces.back().thickness; }
    PBRT_HOST_DEVICE_INLINE
    Float LensFrontZ() const {
        Float zSum = 0;
        for (const LensElementInterface &element : elementInterfaces)
            zSum += element.thickness;
        return zSum;
    }
    PBRT_HOST_DEVICE_INLINE
    Float RearElementRadius() const {
        return elementInterfaces.back().apertureRadius;
    }

    PBRT_HOST_DEVICE_INLINE
    bool TraceLensesFromFilm(const Ray &rCamera, Ray *rOut,
                             Float lambda = 550) const {
        Float elementZ = 0;
        // Transform _rCamera_ from camera to lens system space
        Transform LensFromCamera = Scale(1, 1, -1);
        Ray rLens = LensFromCamera(rCamera);
        for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
            const LensElementInterface &element = elementInterfaces[i];
            // Update ray from film accounting for interaction with _element_
            elementZ -= element.thickness;

            // Compute intersection of ray with lens element
            Float t;
            Normal3f n;
            bool isStop = (element.curvatureRadius == 0);
            if (isStop) {
                // The refracted ray computed in the previous lens element
                // interface may be pointed towards film plane(+z) in some
                // extreme situations; in such cases, 't' becomes negative.
                if (rLens.d.z >= 0.0) return false;
                t = (elementZ - rLens.o.z) / rLens.d.z;
            } else {
                Float radius = element.curvatureRadius;
                Float zCenter = elementZ + element.curvatureRadius;
                if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                    return false;
            }
            DCHECK_GE(t, 0);

            // Test intersection point against element aperture
            Point3f pHit = rLens(t);
            Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
            if (r2 > element.apertureRadius * element.apertureRadius) return false;
            rLens.o = pHit;

            // Update ray path for element interface interaction
            if (!isStop) {
                Vector3f w;
                Float eta_i = element.eta;
                Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                    ? elementInterfaces[i - 1].eta
                    : 1;
                if (dispersionFactor != 0) {
                    Float offset = (lambda - 550) / (550 - 400); // [-1,1] for lambda in [400,700]
                    eta_i -= offset * dispersionFactor * .02;
                    eta_t -= offset * dispersionFactor * .02;
                }
                if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, &w)) return false;
                rLens.d = w;
            }
        }
        // Transform _rLens_ from lens system space back to camera space
        if (rOut != nullptr) {
            const Transform LensToCamera = Scale(1, 1, -1);
            *rOut = LensToCamera(rLens);
        }
        return true;
    }

    PBRT_HOST_DEVICE_INLINE
    static bool IntersectSphericalElement(Float radius, Float zCenter,
                                          const Ray &ray, Float *t,
                                          Normal3f *n) {
        // Compute _t0_ and _t1_ for ray--element intersection
        Point3f o = ray.o - Vector3f(0, 0, zCenter);
        Float A = ray.d.x * ray.d.x + ray.d.y * ray.d.y + ray.d.z * ray.d.z;
        Float B = 2 * (ray.d.x * o.x + ray.d.y * o.y + ray.d.z * o.z);
        Float C = o.x * o.x + o.y * o.y + o.z * o.z - radius * radius;
        Float t0, t1;
        if (!Quadratic(A, B, C, &t0, &t1)) return false;

        // Select intersection $t$ based on ray direction and element curvature
        bool useCloserT = (ray.d.z > 0) ^ (radius < 0);
        *t = useCloserT ? std::min(t0, t1) : std::max(t0, t1);
        if (*t < 0) return false;

        // Compute surface normal of element at ray intersection point
        *n = Normal3f(Vector3f(o + *t * ray.d));
        *n = FaceForward(Normalize(*n), -ray.d);
        return true;
    }

    PBRT_HOST_DEVICE
    bool TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const;

    void DrawLensSystem() const;
    void DrawRayPathFromFilm(const Ray &r, bool arrow,
                             bool toOpticalIntercept) const;
    void DrawRayPathFromScene(const Ray &r, bool arrow,
                              bool toOpticalIntercept) const;

    static void ComputeCardinalPoints(const Ray &rIn, const Ray &rOut, Float *p,
                                      Float *f);
    void ComputeThickLensApproximation(Float pz[2], Float f[2]) const;
    Float FocusThickLens(Float focusDistance);
    Float FocusBinarySearch(Float focusDistance);
    Float FocusDistance(Float filmDist);
    Bounds2f BoundExitPupil(Float pFilmX0, Float pFilmX1) const;
    void RenderExitPupil(Float sx, Float sy, const char *filename) const;

    PBRT_HOST_DEVICE_INLINE
    Point3f SampleExitPupil(const Point2f &pFilm,
                            const Point2f &lensSample,
                            Float *sampleBoundsArea) const {
        // Find exit pupil bound for sample distance from film center
        Float rFilm = std::sqrt(pFilm.x * pFilm.x + pFilm.y * pFilm.y);
        int rIndex = rFilm / (film->diagonal / 2) * exitPupilBounds.size();
        rIndex = std::min<int>(exitPupilBounds.size() - 1, rIndex);
        Bounds2f pupilBounds = exitPupilBounds[rIndex];
        if (sampleBoundsArea != nullptr) *sampleBoundsArea = pupilBounds.Area();

        // Generate sample point inside exit pupil bound
        Point2f pLens = pupilBounds.Lerp(lensSample);

        // Return sample point rotated by angle of _pFilm_ with $+x$ axis
        Float sinTheta = (rFilm != 0) ? pFilm.y / rFilm : 0;
        Float cosTheta = (rFilm != 0) ? pFilm.x / rFilm : 1;
        return {cosTheta * pLens.x - sinTheta * pLens.y,
                sinTheta * pLens.x + cosTheta * pLens.y, LensRearZ()};
    }

    void TestExitPupilBounds() const;
};

}  // namespace pbrt

#endif  // PBRT_CAMERAS_H
