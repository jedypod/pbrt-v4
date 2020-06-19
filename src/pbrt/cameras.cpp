// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// cameras/orthographic.cpp*
#include <pbrt/cameras.h>

#include <pbrt/base/medium.h>
#include <pbrt/bsdf.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>

namespace pbrt {

std::string CameraTransform::ToString() const {
    return StringPrintf("[ CameraTransform renderFromCamera: %s worldFromRender: %s ]",
                        renderFromCamera, worldFromRender);
}

CameraTransform::CameraTransform(const AnimatedTransform &worldFromCamera) {
    Point3f pCamera = (worldFromCamera(Point3f(0, 0, 0), worldFromCamera.startTime) +
                       worldFromCamera(Point3f(0, 0, 0), worldFromCamera.endTime)) /
                      2;

    switch (Options->renderingSpace) {
    case RenderingCoordinateSystem::Camera:
        worldFromRender = worldFromCamera.startTransform;
        if (worldFromCamera.IsAnimated())
            // We could always do this in theory, but if there's a big
            // translation from the origin then the numeric error can lead
            // to something substantially far from the identity matrix we
            // expect.
            renderFromCamera = AnimatedTransform(
                 Transform(), worldFromCamera.startTime,
                 Inverse(worldFromRender) * worldFromCamera.endTransform,
                 worldFromCamera.endTime);
        else
            renderFromCamera = AnimatedTransform();
        break;
    case RenderingCoordinateSystem::CameraWorld: {
        worldFromRender = Translate(Vector3f(pCamera));
        Transform renderFromWorld = Translate(-Vector3f(pCamera));
        renderFromCamera = AnimatedTransform(
            renderFromWorld * worldFromCamera.startTransform,
            worldFromCamera.startTime,
            renderFromWorld * worldFromCamera.endTransform,
            worldFromCamera.endTime);
        break;
    }
    case RenderingCoordinateSystem::World:
        worldFromRender = Transform();
        renderFromCamera = worldFromCamera;
        break;
    default:
        LOG_FATAL("Unhandled rendering coordinate space");
    }
}

// Camera Method Definitions
pstd::optional<CameraRayDifferential> CameraHandle::GenerateRayDifferential(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    auto gen = [&](auto ptr) { return ptr->GenerateRayDifferential(sample, lambda); };
    return Apply<pstd::optional<CameraRayDifferential>>(gen);
}

void CameraHandle::ApproximatedPdxy(const SurfaceInteraction &si) const {
    auto approx = [&](auto ptr) { return ptr->ApproximatedPdxy(si); };
    return Apply<void>(approx);
}

SampledSpectrum CameraHandle::We(const Ray &ray, const SampledWavelengths &lambda,
                                 Point2f *pRaster2) const {
    auto we = [&](auto ptr) { return ptr->We(ray, lambda, pRaster2); };
    return Apply<SampledSpectrum>(we);
}

void CameraHandle::Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->Pdf_We(ray, pdfPos, pdfDir); };
    return Apply<void>(pdf);
}

pstd::optional<CameraWiSample> CameraHandle::Sample_Wi(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda) const {
    auto sample = [&](auto ptr) { return ptr->Sample_Wi(ref, u, lambda); };
    return Apply<pstd::optional<CameraWiSample>>(sample);
}

void CameraHandle::InitMetadata(ImageMetadata *metadata) const {
    auto init = [&](auto ptr) { return ptr->InitMetadata(metadata); };
    return ApplyCPU<void>(init);
}

std::string CameraHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return ApplyCPU<std::string>(ts);
}

CameraBase::CameraBase(const CameraTransform &cameraTransform, Float shutterOpen,
                       Float shutterClose, FilmHandle film, MediumHandle medium)
    : cameraTransform(cameraTransform),
      shutterOpen(shutterOpen),
      shutterClose(shutterClose),
      film(film),
      medium(medium) {
    if (cameraTransform.CameraFromRenderHasScale())
        Warning("Scaling detected in world-to-camera transformation!\n"
                "The system has numerous assumptions, implicit and explicit,\n"
                "that this transform will have no scale factors in it.\n"
                "Proceed at your own risk; your image may have errors or\n"
                "the system may crash as a result of this.");
}

pstd::optional<CameraRayDifferential> CameraBase::GenerateRayDifferential(
    CameraHandle camera, const CameraSample &sample, const SampledWavelengths &lambda) {
    pstd::optional<CameraRay> cr = camera.GenerateRay(sample, lambda);
    if (!cr)
        return {};
    RayDifferential rd(cr->ray);

    // Find camera ray after shifting a fraction of a pixel in the $x$ direction
    pstd::optional<CameraRay> rx;
    for (Float eps : {.05, -.05}) {
        CameraSample sshift = sample;
        sshift.pFilm.x += eps;
        rx = camera.GenerateRay(sshift, lambda);
        if (!rx)
            continue;
        rd.rxOrigin = rd.o + (rx->ray.o - rd.o) / eps;
        rd.rxDirection = rd.d + (rx->ray.d - rd.d) / eps;
        break;
    }
    if (!rx)
        return {};

    // Find camera ray after shifting a fraction of a pixel in the $y$ direction
    pstd::optional<CameraRay> ry;
    for (Float eps : {.05, -.05}) {
        CameraSample sshift = sample;
        sshift.pFilm.y += eps;
        ry = camera.GenerateRay(sshift, lambda);
        if (!ry)
            continue;
        rd.ryOrigin = rd.o + (ry->ray.o - rd.o) / eps;
        rd.ryDirection = rd.d + (ry->ray.d - rd.d) / eps;
        break;
    }
    if (!ry)
        return {};

    rd.hasDifferentials = true;
    return CameraRayDifferential{rd, cr->weight};
}

void CameraBase::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld =
        cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();
}

void CameraBase::FindMinimumDifferentials(CameraHandle camera) {
    minPosDifferentialX = minPosDifferentialY = minDirDifferentialX =
        minDirDifferentialY = Vector3f(Infinity, Infinity, Infinity);

    CameraSample sample;
    sample.pLens = Point2f(0.5, 0.5);
    sample.time = 0.5;
    SampledWavelengths lambda = SampledWavelengths::SampleImportance(0.5);

    int n = 512;
    for (int i = 0; i < n; ++i) {
        sample.pFilm.x = Float(i) / (n - 1) * film.FullResolution().x;
        sample.pFilm.y = Float(i) / (n - 1) * film.FullResolution().y;

        pstd::optional<CameraRayDifferential> crd =
            camera.GenerateRayDifferential(sample, lambda);
        if (!crd)
            continue;

        RayDifferential &ray = crd->ray;
        Vector3f dox = CameraFromRender(ray.rxOrigin - ray.o, ray.time);
        if (Length(dox) < Length(minPosDifferentialX))
            minPosDifferentialX = dox;
        Vector3f doy = CameraFromRender(ray.ryOrigin - ray.o, ray.time);
        if (Length(doy) < Length(minPosDifferentialY))
            minPosDifferentialY = doy;

        ray.d = Normalize(ray.d);
        ray.rxDirection = Normalize(ray.rxDirection);
        ray.ryDirection = Normalize(ray.ryDirection);

        Frame f = Frame::FromZ(ray.d);
        Vector3f df = f.ToLocal(ray.d);  // should be (0, 0, 1);
        Vector3f dxf = Normalize(f.ToLocal(ray.rxDirection));
        Vector3f dyf = Normalize(f.ToLocal(ray.ryDirection));

        if (Length(dxf - df) < Length(minDirDifferentialX))
            minDirDifferentialX = dxf - df;
        if (Length(dyf - df) < Length(minDirDifferentialY))
            minDirDifferentialY = dyf - df;
    }

    LOG_VERBOSE("Camera min pos differentials: %s, %s", minPosDifferentialX,
                minPosDifferentialY);
    LOG_VERBOSE("Camera min dir differentials: %s, %s", minDirDifferentialX,
                minDirDifferentialY);
}

void CameraBase::ApproximatedPdxy(const SurfaceInteraction &si) const {
    Point3f pc = CameraFromRender(si.p(), si.time);
    Float dist = Distance(pc, Point3f(0, 0, 0));

    Frame f = Frame::FromZ(si.n);
    // ray plane:
    // (0,0,0) + minPosDifferential + ((0,0,1) + minDirDifferantial)) * t = (x,
    // x, dist)
    Float tx = (dist - minPosDifferentialX.z) / (1 + minDirDifferentialX.z);
    // 0.5 factor to sharpen them up slightly (could be / should be based
    // on spp?)
    si.dpdx = .5f * f.FromLocal(minPosDifferentialX + tx * minDirDifferentialX);
    Float ty = (dist - minPosDifferentialY.z) / (1 + minDirDifferentialY.z);
    si.dpdy = .5f * f.FromLocal(minPosDifferentialY + ty * minDirDifferentialY);
}

std::string CameraBase::ToString() const {
    return StringPrintf("cameraTransform: %s shutterOpen: %f shutterClose: %f film: %s "
                        "medium: %s minPosDifferentialX: %s minPosDifferentialY: %s "
                        "minDirDifferentialX: %s minDirDifferentialY: %s ",
                        cameraTransform, shutterOpen, shutterClose, film,
                        medium ? medium.ToString().c_str() : "(nullptr)",
                        minPosDifferentialX, minPosDifferentialY, minDirDifferentialX,
                        minDirDifferentialY);
}

std::string CameraSample::ToString() const {
    return StringPrintf("[ pFilm: %s pLens: %s time: %f weight: %f ]", pFilm, pLens, time,
                        weight);
}

ProjectiveCamera::ProjectiveCamera(const CameraTransform &cameraTransform,
                                   const Transform &screenFromCamera,
                                   const Bounds2f &screenWindow, Float shutterOpen,
                                   Float shutterClose, Float lensRadius,
                                   Float focalDistance, FilmHandle film,
                                   MediumHandle medium)
    : CameraBase(cameraTransform, shutterOpen, shutterClose, film, medium),
      screenFromCamera(screenFromCamera),
      lensRadius(lensRadius),
      focalDistance(focalDistance) {
    // Compute projective camera transformations

    // Compute projective camera screen transformations
    rasterFromScreen = Scale(film.FullResolution().x, film.FullResolution().y, 1) *
                       Scale(1 / (screenWindow.pMax.x - screenWindow.pMin.x),
                             1 / (screenWindow.pMin.y - screenWindow.pMax.y), 1) *
                       Translate(Vector3f(-screenWindow.pMin.x, -screenWindow.pMax.y, 0));
    screenFromRaster = Inverse(rasterFromScreen);
    cameraFromRaster = Inverse(screenFromCamera) * screenFromRaster;
}

void ProjectiveCamera::InitMetadata(ImageMetadata *metadata) const {
    metadata->cameraFromWorld =
        cameraTransform.CameraFromWorld(shutterOpen).GetMatrix();

    // TODO: double check this
    Transform NDCFromWorld = Translate(Vector3f(0.5, 0.5, 0.5)) * Scale(0.5, 0.5, 0.5) *
                             screenFromCamera * *metadata->cameraFromWorld;
    metadata->NDCFromWorld = NDCFromWorld.GetMatrix();

    CameraBase::InitMetadata(metadata);
}

std::string ProjectiveCamera::BaseToString() const {
    return CameraBase::ToString() +
           StringPrintf("screenFromCamera: %s cameraFromRaster: %s "
                        "rasterFromScreen: %s screenFromRaster: %s "
                        "lensRadius: %f focalDistance: %f",
                        screenFromCamera, cameraFromRaster, rasterFromScreen,
                        screenFromRaster, lensRadius, focalDistance);
}

CameraHandle CameraHandle::Create(const std::string &name,
                                  const ParameterDictionary &dict, MediumHandle medium,
                                  const CameraTransform &cameraTransform, FilmHandle film,
                                  const FileLoc *loc, Allocator alloc) {
    CameraHandle camera;
    if (name == "perspective")
        camera =
            PerspectiveCamera::Create(dict, cameraTransform, film, medium, loc, alloc);
    else if (name == "orthographic")
        camera =
            OrthographicCamera::Create(dict, cameraTransform, film, medium, loc, alloc);
    else if (name == "realistic")
        camera = RealisticCamera::Create(dict, cameraTransform, film, medium, loc, alloc);
    else if (name == "spherical")
        camera = SphericalCamera::Create(dict, cameraTransform, film, medium, loc, alloc);
    else
        ErrorExit(loc, "%s: camera type unknown.", name);

    if (!camera)
        ErrorExit(loc, "%s: unable to create camera.", name);

    dict.ReportUnused();
    return camera;
}

// OrthographicCamera Definitions
pstd::optional<CameraRay> OrthographicCamera::GenerateRay(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);
    Ray ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);
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
    return CameraRay{RenderFromCamera(ray)};
}

pstd::optional<CameraRayDifferential> OrthographicCamera::GenerateRayDifferential(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    // Compute main orthographic viewing ray

    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);
    RayDifferential ray(pCamera, Vector3f(0, 0, 1), SampleTime(sample.time), medium);

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

    // Compute ray differentials for _OrthographicCamera_
    if (lensRadius > 0) {
        // Compute _OrthographicCamera_ ray differentials accounting for lens

        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);
        Float ft = focalDistance / ray.d.z;

        Point3f pFocus = pCamera + dxCamera + (ft * Vector3f(0, 0, 1));
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        pFocus = pCamera + dyCamera + (ft * Vector3f(0, 0, 1));
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);
    } else {
        ray.rxOrigin = ray.o + dxCamera;
        ray.ryOrigin = ray.o + dyCamera;
        ray.rxDirection = ray.ryDirection = ray.d;
    }
    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

std::string OrthographicCamera::ToString() const {
    return StringPrintf("[ OrthographicCamera %s dxCamera: %s dyCamera: %s ]",
                        BaseToString(), dxCamera, dyCamera);
}

OrthographicCamera *OrthographicCamera::Create(const ParameterDictionary &dict,
                                               const CameraTransform &cameraTransform,
                                               FilmHandle film, MediumHandle medium,
                                               const FileLoc *loc, Allocator alloc) {
    // Extract common camera parameters from _ParameterDictionary_
    Float shutteropen = dict.GetOneFloat("shutteropen", 0.f);
    Float shutterclose = dict.GetOneFloat("shutterclose", 1.f);
    if (shutterclose < shutteropen) {
        Warning(loc, "Shutter close time %f < shutter open %f.  Swapping them.",
                shutterclose, shutteropen);
        pstd::swap(shutterclose, shutteropen);
    }
    Float lensradius = dict.GetOneFloat("lensradius", 0.f);
    Float focaldistance = dict.GetOneFloat("focaldistance", 1e6f);
    Float frame =
        dict.GetOneFloat("frameaspectratio",
                         Float(film.FullResolution().x) / Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = dict.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() == 4) {
            screen.pMin.x = sw[0];
            screen.pMax.x = sw[1];
            screen.pMin.y = sw[2];
            screen.pMax.y = sw[3];
        } else
            Error("\"screenwindow\" should have four values");
    }
    return alloc.new_object<OrthographicCamera>(cameraTransform, screen, shutteropen,
                                                shutterclose, lensradius, focaldistance,
                                                film, medium);
}

// PerspectiveCamera Method Definitions
PerspectiveCamera::PerspectiveCamera(const CameraTransform &cameraTransform,
                                     const Bounds2f &screenWindow, Float shutterOpen,
                                     Float shutterClose, Float lensRadius,
                                     Float focalDistance, Float fov, FilmHandle film,
                                     MediumHandle medium)
    : ProjectiveCamera(cameraTransform, Perspective(fov, 1e-2f, 1000.f), screenWindow,
                       shutterOpen, shutterClose, lensRadius, focalDistance, film,
                       medium) {
    // Compute differential changes in origin for perspective camera rays
    dxCamera = (cameraFromRaster(Point3f(1, 0, 0)) - cameraFromRaster(Point3f(0, 0, 0)));
    dyCamera = (cameraFromRaster(Point3f(0, 1, 0)) - cameraFromRaster(Point3f(0, 0, 0)));

    Point3f pCornerRaster =
        Point3f(-film.GetFilter().Radius().x, -film.GetFilter().Radius().y, 0.f);
    Vector3f wCornerCamera = Normalize(Vector3f(cameraFromRaster(pCornerRaster)));
    cosTotalWidth = wCornerCamera.z;
    DCHECK_LT(.9999 * cosTotalWidth, std::cos(Radians(fov / 2)));

    // Compute image plane bounds at $z=1$ for _PerspectiveCamera_
    Point2i res = film.FullResolution();
    Point3f pMin = cameraFromRaster(Point3f(0, 0, 0));
    Point3f pMax = cameraFromRaster(Point3f(res.x, res.y, 0));
    pMin /= pMin.z;
    pMax /= pMax.z;
    A = std::abs((pMax.x - pMin.x) * (pMax.y - pMin.y));

    FindMinimumDifferentials(this);
}

pstd::optional<CameraRay> PerspectiveCamera::GenerateRay(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);

    Ray ray(Point3f(0, 0, 0), Normalize(Vector3f(pCamera)), SampleTime(sample.time),
            medium);
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
    return CameraRay{RenderFromCamera(ray)};
}

pstd::optional<CameraRayDifferential> PerspectiveCamera::GenerateRayDifferential(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    // Compute raster and camera sample positions
    Point3f pFilm = Point3f(sample.pFilm.x, sample.pFilm.y, 0);
    Point3f pCamera = cameraFromRaster(pFilm);
    Vector3f dir = Normalize(Vector3f(pCamera.x, pCamera.y, pCamera.z));
    RayDifferential ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
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

    // Compute offset rays for _PerspectiveCamera_ ray differentials
    if (lensRadius > 0) {
        // Compute _PerspectiveCamera_ ray differentials accounting for lens

        // Sample point on lens
        Point2f pLens = lensRadius * SampleUniformDiskConcentric(sample.pLens);
        Vector3f dx = Normalize(Vector3f(pCamera + dxCamera));
        Float ft = focalDistance / dx.z;
        Point3f pFocus = Point3f(0, 0, 0) + (ft * dx);
        ray.rxOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.rxDirection = Normalize(pFocus - ray.rxOrigin);

        Vector3f dy = Normalize(Vector3f(pCamera + dyCamera));
        ft = focalDistance / dy.z;
        pFocus = Point3f(0, 0, 0) + (ft * dy);
        ray.ryOrigin = Point3f(pLens.x, pLens.y, 0);
        ray.ryDirection = Normalize(pFocus - ray.ryOrigin);
    } else {
        ray.rxOrigin = ray.ryOrigin = ray.o;
        ray.rxDirection = Normalize(Vector3f(pCamera) + dxCamera);
        ray.ryDirection = Normalize(Vector3f(pCamera) + dyCamera);
    }
    ray.hasDifferentials = true;
    return CameraRayDifferential{RenderFromCamera(ray)};
}

SampledSpectrum PerspectiveCamera::We(const Ray &ray, const SampledWavelengths &lambda,
                                      Point2f *pRaster2) const {
    // XXX Interpolate camera matrix and check if $\w{}$ is forward-facing
    Float cosTheta = Dot(ray.d, RenderFromCamera(Vector3f(0, 0, 1), ray.time));
    if (cosTheta <= cosTotalWidth)
        return SampledSpectrum(0.);

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray((lensRadius > 0 ? focalDistance : 1) / cosTheta);
    Point3f pCamera = CameraFromRender(pFocus, ray.time);
    Point3f pRaster = cameraFromRaster.ApplyInverse(pCamera);

    // Return raster position if requested
    if (pRaster2 != nullptr)
        *pRaster2 = Point2f(pRaster.x, pRaster.y);

    // Return zero importance for out of bounds points
    Bounds2f sampleBounds = film.SampleBounds();
    if (!Inside(Point2f(pRaster.x, pRaster.y), sampleBounds))
        return SampledSpectrum(0.);

    // Compute lens area of perspective camera
    Float lensArea = lensRadius != 0 ? (Pi * lensRadius * lensRadius) : 1;

    // Return importance for point on image plane
    return SampledSpectrum(1 / (A * lensArea * Pow<4>(cosTheta)));
}

void PerspectiveCamera::Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    // Interpolate camera matrix and fail if $\w{}$ is not forward-facing
    Float cosTheta = Dot(ray.d, RenderFromCamera(Vector3f(0, 0, 1), ray.time));
    if (cosTheta <= cosTotalWidth) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Map ray $(\p{}, \w{})$ onto the raster grid
    Point3f pFocus = ray((lensRadius > 0 ? focalDistance : 1) / cosTheta);
    Point3f pCamera = CameraFromRender(pFocus, ray.time);
    Point3f pRaster = cameraFromRaster.ApplyInverse(pCamera);

    // Return zero probability for out of bounds points
    Bounds2f sampleBounds = film.SampleBounds();
    if (!Inside(Point2f(pRaster.x, pRaster.y), sampleBounds)) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    // Compute lens area of perspective camera
    Float lensArea = lensRadius != 0 ? (Pi * lensRadius * lensRadius) : 1;
    *pdfPos = 1 / lensArea;
    *pdfDir = 1 / (A * Pow<3>(cosTheta));
}

pstd::optional<CameraWiSample> PerspectiveCamera::Sample_Wi(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda) const {
    // Uniformly sample a lens interaction _lensIntr_
    Point2f pLens = lensRadius * SampleUniformDiskConcentric(u);
    Point3f pLensRender = RenderFromCamera(Point3f(pLens.x, pLens.y, 0), ref.time);
    Normal3f n = Normal3f(RenderFromCamera(Vector3f(0, 0, 1), ref.time));
    Interaction lensIntr(pLensRender, n, ref.time, medium);

    // Populate arguments and compute the importance value
    Vector3f wi = lensIntr.p() - ref.p();
    Float dist = Length(wi);
    wi /= dist;

    // Compute PDF for importance arriving at _ref_

    // Compute lens area of perspective camera
    Float lensArea = lensRadius != 0 ? (Pi * lensRadius * lensRadius) : 1;
    Float pdf = (dist * dist) / (AbsDot(lensIntr.n, wi) * lensArea);
    Point2f pRaster;
    SampledSpectrum Wi = We(lensIntr.SpawnRay(-wi), lambda, &pRaster);
    if (!Wi)
        return {};

    return CameraWiSample(Wi, wi, pdf, pRaster, ref, lensIntr);
}

std::string PerspectiveCamera::ToString() const {
    return StringPrintf("[ PerspectiveCamera %s dxCamera: %s dyCamera: %s A: "
                        "%f cosTotalWidth: %f ]",
                        BaseToString(), dxCamera, dyCamera, A, cosTotalWidth);
}

PerspectiveCamera *PerspectiveCamera::Create(const ParameterDictionary &dict,
                                             const CameraTransform &cameraTransform,
                                             FilmHandle film, MediumHandle medium,
                                             const FileLoc *loc, Allocator alloc) {
    // Extract common camera parameters from _ParameterDictionary_
    Float shutteropen = dict.GetOneFloat("shutteropen", 0.f);
    Float shutterclose = dict.GetOneFloat("shutterclose", 1.f);
    if (shutterclose < shutteropen) {
        Warning(loc, "Shutter close time %f < shutter open %f.  Swapping them.",
                shutterclose, shutteropen);
        pstd::swap(shutterclose, shutteropen);
    }
    Float lensradius = dict.GetOneFloat("lensradius", 0.f);
    Float focaldistance = dict.GetOneFloat("focaldistance", 1e6);
    Float frame =
        dict.GetOneFloat("frameaspectratio",
                         Float(film.FullResolution().x) / Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = dict.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() == 4) {
            screen.pMin.x = sw[0];
            screen.pMax.x = sw[1];
            screen.pMin.y = sw[2];
            screen.pMax.y = sw[3];
        } else
            Error(loc, "\"screenwindow\" should have four values");
    }
    Float fov = dict.GetOneFloat("fov", 90.);
    return alloc.new_object<PerspectiveCamera>(cameraTransform, screen, shutteropen,
                                               shutterclose, lensradius, focaldistance,
                                               fov, film, medium);
}

// SphericalCamera Method Definitions
pstd::optional<CameraRay> SphericalCamera::GenerateRay(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    // Compute spherical camera ray direction
    Vector3f dir;
    if (mapping == EquiRect) {
        Float theta = Pi * sample.pFilm.y / film.FullResolution().y;
        Float phi = 2 * Pi * sample.pFilm.x / film.FullResolution().x;
        dir = SphericalDirection(std::sin(theta), std::cos(theta), phi);
    } else {
        Point2f uv(sample.pFilm.x / film.FullResolution().x,
                   sample.pFilm.y / film.FullResolution().y);
        uv = WrapEquiAreaSquare(uv);
        dir = EquiAreaSquareToSphere(uv);
    }
    pstd::swap(dir.y, dir.z);

    Ray ray(Point3f(0, 0, 0), dir, SampleTime(sample.time), medium);
    return CameraRay{RenderFromCamera(ray)};
}

SphericalCamera *SphericalCamera::Create(const ParameterDictionary &dict,
                                         const CameraTransform &cameraTransform,
                                         FilmHandle film, MediumHandle medium,
                                         const FileLoc *loc, Allocator alloc) {
    // Extract common camera parameters from _ParameterDictionary_
    Float shutteropen = dict.GetOneFloat("shutteropen", 0.f);
    Float shutterclose = dict.GetOneFloat("shutterclose", 1.f);
    if (shutterclose < shutteropen) {
        Warning(loc, "Shutter close time %f < shutter open %f.  Swapping them.",
                shutterclose, shutteropen);
        pstd::swap(shutterclose, shutteropen);
    }
    Float lensradius = dict.GetOneFloat("lensradius", 0.f);
    Float focaldistance = dict.GetOneFloat("focaldistance", 1e30f);
    Float frame =
        dict.GetOneFloat("frameaspectratio",
                         Float(film.FullResolution().x) / Float(film.FullResolution().y));
    Bounds2f screen;
    if (frame > 1.f) {
        screen.pMin.x = -frame;
        screen.pMax.x = frame;
        screen.pMin.y = -1.f;
        screen.pMax.y = 1.f;
    } else {
        screen.pMin.x = -1.f;
        screen.pMax.x = 1.f;
        screen.pMin.y = -1.f / frame;
        screen.pMax.y = 1.f / frame;
    }
    std::vector<Float> sw = dict.GetFloatArray("screenwindow");
    if (!sw.empty()) {
        if (sw.size() == 4) {
            screen.pMin.x = sw[0];
            screen.pMax.x = sw[1];
            screen.pMin.y = sw[2];
            screen.pMax.y = sw[3];
        } else
            Error(loc, "\"screenwindow\" should have four values");
    }
    (void)lensradius;     // don't need this
    (void)focaldistance;  // don't need this

    std::string m = dict.GetOneString("mapping", "equiarea");
    Mapping mapping;
    if (m == "equiarea")
        mapping = EquiArea;
    else if (m == "equirect")
        mapping = EquiRect;
    else
        ErrorExit(loc,
                  "%s: unknown mapping for spherical camera. (Must be "
                  "\"equiarea\" or \"equirect\".)",
                  m);

    return alloc.new_object<SphericalCamera>(cameraTransform, shutteropen, shutterclose,
                                             film, medium, mapping);
}

std::string SphericalCamera::ToString() const {
    return StringPrintf("[ SphericalCamera %s mapping: %s ]", CameraBase::ToString(),
                        mapping == EquiRect ? "EquiRect" : "EquiArea");
}

STAT_PERCENT("Camera/Rays vignetted by lens system", vignettedRays, totalRays);

// RealisticCamera Method Definitions
std::string RealisticCamera::LensElementInterface::ToString() const {
    return StringPrintf("[ LensElementInterface curvatureRadius: %f thickness: %f "
                        "eta: %f apertureRadius: %f ]",
                        curvatureRadius, thickness, eta, apertureRadius);
}

RealisticCamera::RealisticCamera(const CameraTransform &cameraTransform,
                                 Float shutterOpen, Float shutterClose,
                                 Float setApertureDiameter, Float focusDistance,
                                 Float dispersionFactor, std::vector<Float> &lensData,
                                 Float scale, FilmHandle film, MediumHandle medium,
                                 pstd::optional<Image> apertureImage, Allocator alloc)
    : CameraBase(cameraTransform, shutterOpen, shutterClose, film, medium),
      scale(scale),
      dispersionFactor(dispersionFactor),
      elementInterfaces(alloc),
      exitPupilBounds(alloc),
      apertureImage(std::move(apertureImage)) {
    for (int i = 0; i < (int)lensData.size(); i += 4) {
        Float curvatureRadius = scale * lensData[i];
        Float thickness = scale * lensData[i + 1];
        Float eta  = lensData[i + 2];
        Float apertureDiameter  = scale * lensData[i + 3];

        if (curvatureRadius == 0) {
            // It's the aperture stop
            if (setApertureDiameter > apertureDiameter) {
                Warning("Specified aperture diameter %f is greater than maximum "
                        "possible %f.  Clamping it.",
                        setApertureDiameter, apertureDiameter);
            } else {
                apertureDiameter = setApertureDiameter;
            }
        }
        elementInterfaces.push_back(LensElementInterface(
            {curvatureRadius * (Float).001, thickness * (Float).001, eta,
             apertureDiameter * Float(.001) / 2}));
    }

    // Compute lens--film distance for given focus distance
    Float fb = FocusBinarySearch(focusDistance);
    LOG_VERBOSE("Binary search focus: %f -> %f\n", fb, FocusDistance(fb));
    elementInterfaces.back().thickness = FocusThickLens(focusDistance);
    LOG_VERBOSE("Thick lens focus: %f -> %f\n", elementInterfaces.back().thickness,
                FocusDistance(elementInterfaces.back().thickness));

    // Compute exit pupil bounds at sampled points on the film
    int nSamples = 64;
    exitPupilBounds.resize(nSamples);
    ParallelFor(0, nSamples, [&](int i) {
        Float r0 = (Float)i / nSamples * FilmDiagonal() / 2;
        Float r1 = (Float)(i + 1) / nSamples * FilmDiagonal() / 2;
        exitPupilBounds[i] = BoundExitPupil(r0, r1);
    });

    FindMinimumDifferentials(this);

#if 0
    //Point2f s(.5, .5);   // middle
    Point2f s(.25, .5);   // edge
    Point2f pFilm2 = film.PhysicalExtent().Lerp(s);
    Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

    int count = 16;
    Float start = .5 / count;
    Float end = 1 - start;
    Float delta = .999 / count;
    for (Float lens = start; lens <= end; lens += delta) {
        Float rearRadius = RearElementRadius();
        Point3f pRear(Lerp(lens, -rearRadius, rearRadius),
                      0,
                      LensRearZ());
        Ray ray(pFilm, pRear - pFilm);
        DrawRayPathFromFilm(ray, false, false);
        printf("%c\n", lens + delta < end ? ',' : ' ');
    }
    exit(0);
#endif
}

pstd::optional<CameraRay> RealisticCamera::GenerateRay(
    const CameraSample &sample, const SampledWavelengths &lambda) const {
    // CO        ++totalRays;
    // Find point on film, _pFilm_, corresponding to _sample.pFilm_
    // Compute Film's physical extent
    Float aspect = (Float)film.FullResolution().y / (Float)film.FullResolution().x;
    Float diagonal = FilmDiagonal();
    Float x = std::sqrt(diagonal * diagonal / (1 + aspect * aspect));
    Float y = aspect * x;
    Bounds2f physicalExtent(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));

    Point2f s(sample.pFilm.x / film.FullResolution().x,
              sample.pFilm.y / film.FullResolution().y);
    Point2f pFilm2 = physicalExtent.Lerp(s);
    Point3f pFilm(-pFilm2.x, pFilm2.y, 0);

    // Trace ray from _pFilm_ through lens system
    Float exitPupilBoundsArea;
    Point3f pRear =
        SampleExitPupil(Point2f(pFilm.x, pFilm.y), sample.pLens, &exitPupilBoundsArea);
    Ray rFilm(pFilm, pRear - pFilm);
    Ray ray;
    Float weight = TraceLensesFromFilm(rFilm, &ray, lambda[0]);
    if (weight == 0) {
        // CO            ++vignettedRays;
        return {};
    }

    // Finish initialization of _RealisticCamera_ ray
    ray.time = SampleTime(sample.time);
    ray.medium = medium;
    ray = RenderFromCamera(ray);
    ray.d = Normalize(ray.d);

    if (dispersionFactor != 0)
        lambda.TerminateSecondaryWavelengths();

    // Return weighting for _RealisticCamera_ ray
    Float cosTheta = Normalize(rFilm.d).z;
    Float cos4Theta = (cosTheta * cosTheta) * (cosTheta * cosTheta);
    weight *= (shutterClose - shutterOpen) * (cos4Theta * exitPupilBoundsArea) /
        (LensRearZ() * LensRearZ());

    return CameraRay{ray, SampledSpectrum(weight)};
}

Point3f RealisticCamera::SampleExitPupil(const Point2f &pFilm, const Point2f &lensSample,
                                         Float *sampleBoundsArea) const {
    // Find exit pupil bound for sample distance from film center
    Float rFilm = std::sqrt(pFilm.x * pFilm.x + pFilm.y * pFilm.y);
    int rIndex = rFilm / (FilmDiagonal() / 2) * exitPupilBounds.size();
    rIndex = std::min<int>(exitPupilBounds.size() - 1, rIndex);
    Bounds2f pupilBounds = exitPupilBounds[rIndex];
    if (sampleBoundsArea != nullptr)
        *sampleBoundsArea = pupilBounds.Area();

    // Generate sample point inside exit pupil bound
    Point2f pLens = pupilBounds.Lerp(lensSample);

    // Return sample point rotated by angle of _pFilm_ with $+x$ axis
    Float sinTheta = (rFilm != 0) ? pFilm.y / rFilm : 0;
    Float cosTheta = (rFilm != 0) ? pFilm.x / rFilm : 1;
    return {cosTheta * pLens.x - sinTheta * pLens.y,
            sinTheta * pLens.x + cosTheta * pLens.y, LensRearZ()};
}

Float RealisticCamera::TraceLensesFromFilm(const Ray &rCamera, Ray *rOut,
                                           Float lambda) const {
    Float elementZ = 0;
    Float weight = 1;

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
            if (rLens.d.z >= 0.0)
                return false;
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
        if (isStop && apertureImage) {
            Point2f uv((pHit.x / element.apertureRadius + 1) / 2,
                       (pHit.y / element.apertureRadius + 1) / 2);
            uv.y = 1 - uv.y;
            weight = apertureImage->BilerpChannel(uv, 0, WrapMode::Black);
            if (weight == 0)
                return 0;
        } else {
            Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
            if (r2 > element.apertureRadius * element.apertureRadius)
                return 0;
        }
        rLens.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f w;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (dispersionFactor != 0) {
                Float offset =
                    (lambda - 550) / (550 - 400);  // [-1,1] for lambda in [400,700]
                eta_i -= offset * dispersionFactor * .02;
                eta_t -= offset * dispersionFactor * .02;
            }
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, &w))
                return 0;
            rLens.d = w;
        }
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut != nullptr) {
        const Transform LensToCamera = Scale(1, 1, -1);
        *rOut = LensToCamera(rLens);
    }
    return weight;
}

bool RealisticCamera::TraceLensesFromScene(const Ray &rCamera, Ray *rOut) const {
    Float elementZ = -LensFrontZ();
    // Transform _rCamera_ from camera to lens system space
    const Transform LensFromCamera = Scale(1, 1, -1);
    Ray rLens = LensFromCamera(rCamera);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        bool isStop = (element.curvatureRadius == 0);
        if (isStop)
            t = (elementZ - rLens.o.z) / rLens.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, rLens, &t, &n))
                return false;
        }
        CHECK_GE(t, 0);

        // Test intersection point against element aperture
        // Don't worry about the aperture image here.
        Point3f pHit = rLens(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        if (r2 > element.apertureRadius * element.apertureRadius)
            return false;
        rLens.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0)
                              ? 1
                              : elementInterfaces[i - 1].eta;
            Float eta_t = (elementInterfaces[i].eta != 0) ? elementInterfaces[i].eta : 1;
            if (!Refract(Normalize(-rLens.d), n, eta_t / eta_i, &wt))
                return false;
            rLens.d = wt;
        }
        elementZ += element.thickness;
    }
    // Transform _rLens_ from lens system space back to camera space
    if (rOut != nullptr) {
        const Transform LensToCamera = Scale(1, 1, -1);
        *rOut = LensToCamera(rLens);
    }
    return true;
}

void RealisticCamera::DrawLensSystem() const {
    Float sumz = -LensFrontZ();
    Float z = sumz;
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        Float r = element.curvatureRadius;
        if (r == 0) {
            // stop
            printf("{Thick, Line[{{%f, %f}, {%f, %f}}], ", z, element.apertureRadius, z,
                   2 * element.apertureRadius);
            printf("Line[{{%f, %f}, {%f, %f}}]}, ", z, -element.apertureRadius, z,
                   -2 * element.apertureRadius);
        } else {
            Float theta = std::abs(SafeASin(element.apertureRadius / r));
            if (r > 0) {
                // convex as seen from front of lens
                Float t0 = Pi - theta;
                Float t1 = Pi + theta;
                printf("Circle[{%f, 0}, %f, {%f, %f}], ", z + r, r, t0, t1);
            } else {
                // concave as seen from front of lens
                Float t0 = -theta;
                Float t1 = theta;
                printf("Circle[{%f, 0}, %f, {%f, %f}], ", z + r, -r, t0, t1);
            }
            if (element.eta != 0 && element.eta != 1) {
                // connect top/bottom to next element
                CHECK_LT(i + 1, elementInterfaces.size());
                Float nextApertureRadius = elementInterfaces[i + 1].apertureRadius;
                Float h = std::max(element.apertureRadius, nextApertureRadius);
                Float hlow = std::min(element.apertureRadius, nextApertureRadius);

                Float zp0, zp1;
                if (r > 0) {
                    zp0 = z + element.curvatureRadius -
                          element.apertureRadius / std::tan(theta);
                } else {
                    zp0 = z + element.curvatureRadius +
                          element.apertureRadius / std::tan(theta);
                }

                Float nextCurvatureRadius = elementInterfaces[i + 1].curvatureRadius;
                Float nextTheta =
                    std::abs(SafeASin(nextApertureRadius / nextCurvatureRadius));
                if (nextCurvatureRadius > 0) {
                    zp1 = z + element.thickness + nextCurvatureRadius -
                          nextApertureRadius / std::tan(nextTheta);
                } else {
                    zp1 = z + element.thickness + nextCurvatureRadius +
                          nextApertureRadius / std::tan(nextTheta);
                }

                // Connect tops
                printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp1, h);
                printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp1, -h);

                // vertical lines when needed to close up the element profile
                if (element.apertureRadius < nextApertureRadius) {
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, h, zp0, hlow);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp0, -h, zp0, -hlow);
                } else if (element.apertureRadius > nextApertureRadius) {
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, h, zp1, hlow);
                    printf("Line[{{%f, %f}, {%f, %f}}], ", zp1, -h, zp1, -hlow);
                }
            }
        }
        z += element.thickness;
    }

    // 24mm height for 35mm film
    printf("Line[{{0, -.012}, {0, .012}}], ");
    // optical axis
    printf("Line[{{0, 0}, {%f, 0}}] ", 1.2f * sumz);
}

void RealisticCamera::DrawRayPathFromFilm(const Ray &r, bool arrow,
                                          bool toOpticalIntercept) const {
    Float elementZ = 0;
    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    printf("{ ");
    if (TraceLensesFromFilm(r, nullptr) == 0) {
        printf("Dashed, RGBColor[.8, .5, .5]");
    } else
        printf("RGBColor[.5, .5, .8]");

    for (int i = elementInterfaces.size() - 1; i >= 0; --i) {
        const LensElementInterface &element = elementInterfaces[i];
        elementZ -= element.thickness;
        bool isStop = (element.curvatureRadius == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                goto done;
        }
        CHECK_GE(t, 0);

        printf(", Line[{{%f, %f}, {%f, %f}}]", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius * element.apertureRadius;
        if (r2 > apertureRadius2)
            goto done;
        ray.o = pHit;

        // Update ray path for element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = element.eta;
            Float eta_t = (i > 0 && elementInterfaces[i - 1].eta != 0)
                              ? elementInterfaces[i - 1].eta
                              : 1;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, &wt))
                goto done;
            ray.d = wt;
        }
    }

    ray.d = Normalize(ray.d);
    {
        Float ta = std::abs(elementZ / 4);
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf(", Point[{%f, %f}]", ray(ta).z, ray(ta).x);
        }
        printf(", %s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);

        // overdraw the optical axis if needed...
        if (toOpticalIntercept)
            printf(", Line[{{%f, 0}, {%f, 0}}]", ray.o.z, ray(ta).z * 1.05f);
    }

done:
    printf("}");
}

void RealisticCamera::DrawRayPathFromScene(const Ray &r, bool arrow,
                                           bool toOpticalIntercept) const {
    Float elementZ = LensFrontZ() * -1;

    // Transform _ray_ from camera to lens system space
    static const Transform LensFromCamera = Scale(1, 1, -1);
    Ray ray = LensFromCamera(r);
    for (size_t i = 0; i < elementInterfaces.size(); ++i) {
        const LensElementInterface &element = elementInterfaces[i];
        bool isStop = (element.curvatureRadius == 0);
        // Compute intersection of ray with lens element
        Float t;
        Normal3f n;
        if (isStop)
            t = -(ray.o.z - elementZ) / ray.d.z;
        else {
            Float radius = element.curvatureRadius;
            Float zCenter = elementZ + element.curvatureRadius;
            if (!IntersectSphericalElement(radius, zCenter, ray, &t, &n))
                return;
        }
        CHECK_GE(t, 0.f);

        printf("Line[{{%f, %f}, {%f, %f}}],", ray.o.z, ray.o.x, ray(t).z, ray(t).x);

        // Test intersection point against element aperture
        Point3f pHit = ray(t);
        Float r2 = pHit.x * pHit.x + pHit.y * pHit.y;
        Float apertureRadius2 = element.apertureRadius * element.apertureRadius;
        if (r2 > apertureRadius2)
            return;
        ray.o = pHit;

        // Update ray path for from-scene element interface interaction
        if (!isStop) {
            Vector3f wt;
            Float eta_i = (i == 0 || elementInterfaces[i - 1].eta == 0.f)
                              ? 1.f
                              : elementInterfaces[i - 1].eta;
            Float eta_t =
                (elementInterfaces[i].eta != 0.f) ? elementInterfaces[i].eta : 1.f;
            if (!Refract(Normalize(-ray.d), n, eta_t / eta_i, &wt))
                return;
            ray.d = wt;
        }
        elementZ += element.thickness;
    }

    // go to the film plane by default
    {
        Float ta = -ray.o.z / ray.d.z;
        if (toOpticalIntercept) {
            ta = -ray.o.x / ray.d.x;
            printf("Point[{%f, %f}], ", ray(ta).z, ray(ta).x);
        }
        printf("%s[{{%f, %f}, {%f, %f}}]", arrow ? "Arrow" : "Line", ray.o.z, ray.o.x,
               ray(ta).z, ray(ta).x);
    }
}

void RealisticCamera::ComputeCardinalPoints(const Ray &rIn, const Ray &rOut, Float *pz,
                                            Float *fz) {
    Float tf = -rOut.o.x / rOut.d.x;
    *fz = -rOut(tf).z;
    Float tp = (rIn.o.x - rOut.o.x) / rOut.d.x;
    *pz = -rOut(tp).z;
}

void RealisticCamera::ComputeThickLensApproximation(Float pz[2], Float fz[2]) const {
    // Find height $x$ from optical axis for parallel rays
    Float x = .001 * FilmDiagonal();

    // Compute cardinal points for film side of lens system
    Ray rScene(Point3f(x, 0, LensFrontZ() + 1), Vector3f(0, 0, -1));
    Ray rFilm;
    if (!TraceLensesFromScene(rScene, &rFilm))
        ErrorExit("Unable to trace ray from scene to film for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rScene, rFilm, &pz[0], &fz[0]);

    // Compute cardinal points for scene side of lens system
    rFilm = Ray(Point3f(x, 0, LensRearZ() - 1), Vector3f(0, 0, 1));
    if (TraceLensesFromFilm(rFilm, &rScene) == 0)
        ErrorExit("Unable to trace ray from film to scene for thick lens "
                  "approximation. Is aperture stop extremely small?");
    ComputeCardinalPoints(rFilm, rScene, &pz[1], &fz[1]);
}

Float RealisticCamera::FocusThickLens(Float focusDistance) {
    Float pz[2], fz[2];
    ComputeThickLensApproximation(pz, fz);
    LOG_VERBOSE("Cardinal points: p' = %f f' = %f, p = %f f = %f.\n", pz[0], fz[0], pz[1],
                fz[1]);
    LOG_VERBOSE("Effective focal length %f\n", fz[0] - pz[0]);

    // Compute translation of lens, _delta_, to focus at _focusDistance_
    Float f = fz[0] - pz[0];
    Float z = -focusDistance;
    Float c = (pz[1] - z - pz[0]) * (pz[1] - z - 4 * f - pz[0]);
    if (c <= 0)
        ErrorExit("Coefficient must be positive. It looks focusDistance %f "
                  " is too short for a given lenses configuration",
                  focusDistance);
    Float delta = 0.5f * (pz[1] - z + pz[0] - std::sqrt(c));
    return elementInterfaces.back().thickness + delta;
}

Float RealisticCamera::FocusBinarySearch(Float focusDistance) {
    Float filmDistanceLower, filmDistanceUpper;
    // Find _filmDistanceLower_, _filmDistanceUpper_ that bound focus distance
    filmDistanceLower = filmDistanceUpper = FocusThickLens(focusDistance);
    while (FocusDistance(filmDistanceLower) > focusDistance)
        filmDistanceLower *= 1.005f;
    while (FocusDistance(filmDistanceUpper) < focusDistance)
        filmDistanceUpper /= 1.005f;

    // Do binary search on film distances to focus
    for (int i = 0; i < 20; ++i) {
        Float fmid = 0.5f * (filmDistanceLower + filmDistanceUpper);
        Float midFocus = FocusDistance(fmid);
        if (midFocus < focusDistance)
            filmDistanceLower = fmid;
        else
            filmDistanceUpper = fmid;
    }
    return 0.5f * (filmDistanceLower + filmDistanceUpper);
}

Float RealisticCamera::FocusDistance(Float filmDistance) {
    // Find offset ray from film center through lens
    Bounds2f bounds = BoundExitPupil(0, .001 * FilmDiagonal());

    const pstd::array<Float, 3> scaleFactors = {0.1f, 0.01f, 0.001f};
    Float lu = 0.0f;

    Ray ray;

    // Try some different and decreasing scaling factor to find focus ray
    // more quickly when `aperturediameter` is too small.
    // (e.g. 2 [mm] for `aperturediameter` with wide.22mm.dat),
    bool foundFocusRay = false;
    for (Float scale : scaleFactors) {
        lu = scale * bounds.pMax[0];
        if (TraceLensesFromFilm(Ray(Point3f(0, 0, LensRearZ() - filmDistance),
                                    Vector3f(lu, 0, filmDistance)),
                                &ray)) {
            foundFocusRay = true;
            break;
        }
    }

    if (!foundFocusRay) {
        Error("Focus ray at lens pos(%f,0) didn't make it through the lenses "
              "with film distance %f?!??\n",
              lu, filmDistance);
        return Infinity;
    }

    // Compute distance _zFocus_ where ray intersects the principal axis
    Float tFocus = -ray.o.x / ray.d.x;
    Float zFocus = ray(tFocus).z;
    if (zFocus < 0)
        zFocus = Infinity;
    return zFocus;
}

Bounds2f RealisticCamera::BoundExitPupil(Float pFilmX0, Float pFilmX1) const {
    Bounds2f pupilBounds;
    // Sample a collection of points on the rear lens to find exit pupil
    const int nSamples = 1024 * 1024;
    int nExitingRays = 0;

    // Compute bounding box of projection of rear element on sampling plane
    Float rearRadius = RearElementRadius();
    Bounds2f projRearBounds(Point2f(-1.5f * rearRadius, -1.5f * rearRadius),
                            Point2f(1.5f * rearRadius, 1.5f * rearRadius));
    for (int i = 0; i < nSamples; ++i) {
        // Find location of sample points on $x$ segment and rear lens element
        Point3f pFilm(Lerp((i + 0.5f) / nSamples, pFilmX0, pFilmX1), 0, 0);
        Float u[2] = {RadicalInverse(0, i), RadicalInverse(1, i)};
        Point3f pRear(Lerp(u[0], projRearBounds.pMin.x, projRearBounds.pMax.x),
                      Lerp(u[1], projRearBounds.pMin.y, projRearBounds.pMax.y),
                      LensRearZ());

        // Expand pupil bounds if ray makes it through the lens system
        if (Inside(Point2f(pRear.x, pRear.y), pupilBounds) ||
            TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), nullptr)) {
            pupilBounds = Union(pupilBounds, Point2f(pRear.x, pRear.y));
            ++nExitingRays;
        }
    }

    // Return entire element bounds if no rays made it through the lens system
    if (nExitingRays == 0) {
        LOG_VERBOSE("Unable to find exit pupil in x = [%f,%f] on film.", pFilmX0,
                    pFilmX1);
        return projRearBounds;
    }

    // Expand bounds to account for sample spacing
    pupilBounds =
        Expand(pupilBounds, 2 * Length(projRearBounds.Diagonal()) / std::sqrt(nSamples));
    return pupilBounds;
}

void RealisticCamera::RenderExitPupil(Float sx, Float sy, const char *filename) const {
    Point3f pFilm(sx, sy, 0);

    const int nSamples = 2048;
    Image image(PixelFormat::Float, {nSamples, nSamples}, {"Y"});

    for (int y = 0; y < nSamples; ++y) {
        Float fy = (Float)y / (Float)(nSamples - 1);
        Float ly = Lerp(fy, -RearElementRadius(), RearElementRadius());
        for (int x = 0; x < nSamples; ++x) {
            Float fx = (Float)x / (Float)(nSamples - 1);
            Float lx = Lerp(fx, -RearElementRadius(), RearElementRadius());

            Point3f pRear(lx, ly, LensRearZ());

            if (lx * lx + ly * ly > RearElementRadius() * RearElementRadius())
                image.SetChannel({x, y}, 0, 1.);
            else if (TraceLensesFromFilm(Ray(pFilm, pRear - pFilm), nullptr))
                image.SetChannel({x, y}, 0, 0.5);
            else
                image.SetChannel({x, y}, 0, 0.);
        }
    }

    image.Write(filename);
}

void RealisticCamera::TestExitPupilBounds() const {
    Float filmDiagonal = FilmDiagonal();

    static RNG rng;

    Float u = rng.Uniform<Float>();
    Point3f pFilm(u * filmDiagonal / 2, 0, 0);

    Float r = pFilm.x / (filmDiagonal / 2);
    int pupilIndex = std::min<int>(exitPupilBounds.size() - 1,
                                   std::floor(r * (exitPupilBounds.size() - 1)));
    Bounds2f pupilBounds = exitPupilBounds[pupilIndex];
    if (pupilIndex + 1 < (int)exitPupilBounds.size())
        pupilBounds = Union(pupilBounds, exitPupilBounds[pupilIndex + 1]);

    // Now, randomly pick points on the aperture and see if any are outside
    // of pupil bounds...
    for (int i = 0; i < 1000; ++i) {
        Point2f u2{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point2f pd = SampleUniformDiskConcentric(u2);
        pd *= RearElementRadius();

        Ray testRay(pFilm, Point3f(pd.x, pd.y, 0.f) - pFilm);
        Ray testOut;
        if (!TraceLensesFromFilm(testRay, &testOut))
            continue;

        if (!Inside(pd, pupilBounds)) {
            fprintf(stderr,
                    "Aha! (%f,%f) went through, but outside bounds (%f,%f) - "
                    "(%f,%f)\n",
                    pd.x, pd.y, pupilBounds.pMin[0], pupilBounds.pMin[1],
                    pupilBounds.pMax[0], pupilBounds.pMax[1]);
            RenderExitPupil(
                (Float)pupilIndex / exitPupilBounds.size() * filmDiagonal / 2.f, 0.f,
                "low.exr");
            RenderExitPupil(
                (Float)(pupilIndex + 1) / exitPupilBounds.size() * filmDiagonal / 2.f,
                0.f, "high.exr");
            RenderExitPupil(pFilm.x, 0.f, "mid.exr");
            exit(0);
        }
    }
    fprintf(stderr, ".");
}

std::string RealisticCamera::ToString() const {
    return StringPrintf("[ RealisticCamera %s dispersionFactor: %f "
                        "elementInterfaces: %s exitPupilBounds: %s ]",
                        CameraBase::ToString(), dispersionFactor, elementInterfaces,
                        exitPupilBounds);
}

RealisticCamera *RealisticCamera::Create(const ParameterDictionary &dict,
                                         const CameraTransform &cameraTransform,
                                         FilmHandle film, MediumHandle medium,
                                         const FileLoc *loc, Allocator alloc) {
    Float shutteropen = dict.GetOneFloat("shutteropen", 0.f);
    Float shutterclose = dict.GetOneFloat("shutterclose", 1.f);
    if (shutterclose < shutteropen) {
        Warning(loc, "Shutter close time %f < shutter open %f.  Swapping them.",
                shutterclose, shutteropen);
        pstd::swap(shutterclose, shutteropen);
    }

    // Realistic camera-specific parameters
    std::string lensFile = ResolveFilename(dict.GetOneString("lensfile", ""));
    Float apertureDiameter = dict.GetOneFloat("aperturediameter", 1.0);
    Float focusDistance = dict.GetOneFloat("focusdistance", 10.0);
    Float dispersionFactor = dict.GetOneFloat("dispersionfactor", 0.);
    Float scale = dict.GetOneFloat("scale", 1.f);

    if (lensFile.empty()) {
        Error(loc, "No lens description file supplied!");
        return nullptr;
    }
    // Load element data from lens description file
    pstd::optional<std::vector<Float>> lensData = ReadFloatFile(lensFile);
    if (!lensData) {
        Error(loc, "Error reading lens specification file \"%s\".", lensFile);
        return nullptr;
    }
    if (lensData->size() % 4 != 0) {
        Error(loc,
              "%s: excess values in lens specification file; "
              "must be multiple-of-four values, read %d.",
              lensFile, (int)lensData->size());
        return nullptr;
    }

    int builtinRes = 256;
    auto rasterize = [&](pstd::span<const Point2f> vert) {
        Image image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                    nullptr, alloc);

        for (int y = 0; y < image.Resolution().y; ++y)
            for (int x = 0; x < image.Resolution().x; ++x) {
                Point2f p(-1 + 2 * (x + 0.5f) / image.Resolution().x,
                           -1 + 2 * (y + 0.5f) / image.Resolution().y);
                int windingNumber = 0;
                // Test against edges
                for (int i = 0; i < vert.size(); ++i) {
                    int i1 = (i + 1) % vert.size();
                    Float e = (p[0] - vert[i][0]) * (vert[i1][1] - vert[i][1]) -
                        (p[1] - vert[i][1]) * (vert[i1][0] - vert[i][0]);
                    if (vert[i].y <= p.y) {
                        if (vert[i1].y > p.y && e > 0)
                            ++windingNumber;
                    } else if (vert[i1].y <= p.y && e < 0)
                        --windingNumber;
                }

                image.SetChannel({x, y}, 0, windingNumber == 0 ? 0.f : 1.f);
            }

        return image;
    };

    std::string apertureName = ResolveFilename(dict.GetOneString("aperture", ""));
    pstd::optional<Image> apertureImage;
    if (!apertureName.empty()) {
        // built-in diaphragm shapes
        if (apertureName == "gaussian") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage->Resolution().y; ++y)
                for (int x = 0; x < apertureImage->Resolution().x; ++x) {
                    Point2f uv(-1 + 2 * (x + 0.5f) / apertureImage->Resolution().x,
                               -1 + 2 * (y + 0.5f) / apertureImage->Resolution().y);
                    Float r2 = Sqr(uv.x) + Sqr(uv.y);
                    Float sigma2 = 1;
                    Float v = std::max<Float>(0, std::exp(-r2 / sigma2) - std::exp(-1 / sigma2));
                    apertureImage->SetChannel({x, y}, 0, v);
                }
        } else if (apertureName == "square") {
            apertureImage = Image(PixelFormat::Float, {builtinRes, builtinRes}, {"Y"},
                                  nullptr, alloc);
            for (int y = 0; y < apertureImage->Resolution().y; ++y)
                for (int x = 0; x < apertureImage->Resolution().x; ++x)
                    apertureImage->SetChannel({x, y}, 0, 1.f);
        } else if (apertureName == "pentagon") {
            // https://mathworld.wolfram.com/RegularPentagon.html
            Float c1 = (std::sqrt(5.f) - 1) / 4;
            Float c2 = (std::sqrt(5.f) + 1) / 4;
            Float s1 = std::sqrt(10.f + 2.f * std::sqrt(5.f)) / 4;
            Float s2 = std::sqrt(10.f - 2.f * std::sqrt(5.f)) / 4;
            // Vertices in CW order.
            Point2f vert[5] = { Point2f(0, 1), {s1, c1}, {s2, -c2}, {-s2, -c2}, {-s1, c1} };
            // Scale down slightly
            for (int i = 0; i < 5; ++i)
                vert[i] *= .8f;
            apertureImage = rasterize(vert);
        } else if (apertureName == "star") {
            // 5-sided. Vertices are two pentagons--inner and outer radius
            pstd::array<Point2f, 10> vert;
            for (int i = 0; i < 10; ++i) {
                // inner radius: https://math.stackexchange.com/a/2136996
                Float r = (i & 1) ? 1.f :
                    (std::cos(Radians(72.f)) / std::cos(Radians(36.f)));
                vert[i] = Point2f(r * std::cos(Pi * i / 5.f),
                                  r * std::sin(Pi * i / 5.f));
            }
            std::reverse(vert.begin(), vert.end());
            apertureImage = rasterize(vert);
        } else {
            auto im = Image::Read(apertureName, alloc);
            if (im) {
                apertureImage = std::move(im->image);
                if (apertureImage->NChannels() > 1) {
                    pstd::optional<ImageChannelDesc> rgbDesc =
                        apertureImage->GetChannelDesc({"R", "G", "B"});
                    if (!rgbDesc)
                        ErrorExit("%s: didn't find R, G, B channels to average for "
                                  "aperture image.", apertureName);

                    Image mono(PixelFormat::Float, apertureImage->Resolution(), {"Y"},
                               nullptr, alloc);
                    for (int y = 0; y < mono.Resolution().y; ++y)
                        for (int x = 0; x < mono.Resolution().x; ++x) {
                            Float avg = apertureImage->GetChannels({x, y}, *rgbDesc).Average();
                            mono.SetChannel({x, y}, 0, avg);
                        }

                    apertureImage = std::move(mono);
                }
            }
        }

        if (apertureImage) {
            // Normalize it so that brightness matches a circular aperture
            Float sum = 0;
            for (int y = 0; y < apertureImage->Resolution().y; ++y)
                for (int x = 0; x < apertureImage->Resolution().x; ++x)
                    sum += apertureImage->GetChannel({x, y}, 0);
            Float avg = sum / (apertureImage->Resolution().x *
                               apertureImage->Resolution().y);

            Float scale = (Pi / 4) / avg;
            for (int y = 0; y < apertureImage->Resolution().y; ++y)
                for (int x = 0; x < apertureImage->Resolution().x; ++x)
                    apertureImage->SetChannel({x, y}, 0,
                                              apertureImage->GetChannel({x, y}, 0) * scale);
        }
    }

    return alloc.new_object<RealisticCamera>(
        cameraTransform, shutteropen, shutterclose, apertureDiameter, focusDistance,
        dispersionFactor, *lensData, scale, film, medium, std::move(apertureImage),
        alloc);
}

}  // namespace pbrt
