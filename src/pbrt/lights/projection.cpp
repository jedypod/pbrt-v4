
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

// lights/projection.cpp*
#include <pbrt/lights/projection.h>
#include <pbrt/core/sampling.h>
#include <pbrt/core/paramset.h>
#include <pbrt/core/reflection.h>
#include <pbrt/util/fileutil.h>
#include <pbrt/util/stats.h>

namespace pbrt {

// ProjectionLight Method Definitions
ProjectionLight::ProjectionLight(const Transform &LightToWorld,
                                 const MediumInterface &mediumInterface,
                                 const Spectrum &I, Image im,
                                 Float fov, const std::shared_ptr<const ParamSet> &attributes)
    : Light((int)LightFlags::DeltaPosition, LightToWorld, mediumInterface,
            attributes),
      image(std::move(im)),
      pLight(LightToWorld(Point3f(0, 0, 0))),
      I(I) {
    // Initialize _ProjectionLight_ projection matrix
    Float aspect = Float(image.resolution.x) / Float(image.resolution.y);
    if (aspect > 1)
        screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
    else
        screenBounds =
            Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
    hither = 1e-3f;
    yon = 1e30f;
    lightToScreen = Perspective(fov, hither, yon);
    screenToLight = Inverse(lightToScreen);

    // Compute cosine of cone surrounding projection directions
    Float opposite = std::tan(Radians(fov) / 2.f);
    // Area of the image on projection plane.
    A = 4 * opposite * opposite * (aspect > 1 ? aspect : 1 / aspect);
    Float tanDiag = opposite * std::sqrt(1 + 1 / (aspect * aspect));
    cosTotalWidth = std::cos(std::atan(tanDiag));

    auto dwdA = [&](Point2f p) {
        Point2f ps = screenBounds.Lerp(p);
        Vector3f w = Vector3f(screenToLight(Point3f(ps.x, ps.y, 0)));
        w = Normalize(w);
        return Pow<3>(w.z);
    };
    distrib = image.ComputeSamplingDistribution(dwdA, 1, Norm::L2);
}

Spectrum ProjectionLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                    Vector3f *wi, Float *pdf,
                                    VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    *wi = Normalize(pLight - ref.p);
    *pdf = 1;
    *vis =
        VisibilityTester(ref, Interaction(pLight, ref.time, mediumInterface));
    return Projection(-*wi) / DistanceSquared(pLight, ref.p);
}

Spectrum ProjectionLight::Projection(const Vector3f &w) const {
    Vector3f wl = WorldToLight(w);
    // Discard directions behind projection light
    if (wl.z < hither) return 0;

    // Project point onto projection plane and compute light
    Point3f p = lightToScreen(Point3f(wl.x, wl.y, wl.z));
    if (!Inside(Point2f(p.x, p.y), screenBounds)) return 0.f;
    Point2f st = Point2f(screenBounds.Offset(Point2f(p.x, p.y)));
    return I * image.BilerpSpectrum(st, SpectrumType::Illuminant);
}

Spectrum ProjectionLight::Phi() const {
    Spectrum sum(0.f);
    for (int v = 0; v < image.resolution.y; ++v)
        for (int u = 0; u < image.resolution.x; ++u) {
            Point2f ps = screenBounds.Lerp({(u + .5f) / image.resolution.x,
                                            (v + .5f) / image.resolution.y});
            Vector3f w = Vector3f(screenToLight(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(w.z);
            sum += image.GetSpectrum({u, v}, SpectrumType::Illuminant) * dwdA;
        }

    return I * A * sum / (image.resolution.x * image.resolution.y);
}

Float ProjectionLight::Pdf_Li(const Interaction &, const Vector3f &) const {
    return 0.f;
}

Spectrum ProjectionLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                    Float time, Ray *ray, Normal3f *nLight,
                                    Float *pdfPos, Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);

    Float pdf;
    Point2f uv = distrib.SampleContinuous(u1, &pdf);
    if (pdf == 0) {
        *pdfPos = *pdfDir = 0;
        return 0;
    }

    Point2f ps = screenBounds.Lerp(uv);
    Vector3f w = Vector3f(screenToLight(Point3f(ps.x, ps.y, 0)));
    w = Normalize(w);

    *ray = Ray(pLight, LightToWorld(w), Infinity, time, mediumInterface.inside);
    *nLight = (Normal3f)ray->d;
    *pdfPos = 1;
    CHECK_GT(w.z, 0);
    *pdfDir = pdf / (A * Pow<3>(w.z));

    return I * image.BilerpSpectrum(uv, SpectrumType::Illuminant);
}

void ProjectionLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                             Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    *pdfPos = 0;

    Vector3f w = Normalize(WorldToLight(ray.d));
    if (w.z < hither) {
        *pdfDir = 0;
        return;
    }
    Point3f ps = lightToScreen(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds)) {
        *pdfDir = 0;
        return;
    }
    Point2f st = Point2f(screenBounds.Offset(Point2f(ps.x, ps.y)));
    *pdfDir = distrib.Pdf(st) / (A * Pow<3>(w.z));
}

std::shared_ptr<ProjectionLight> CreateProjectionLight(
    const Transform &light2world, const Medium *medium,
    const ParamSet &paramSet, const std::shared_ptr<const ParamSet> &attributes) {
    Spectrum I = paramSet.GetOneSpectrum("I", Spectrum(1.0));
    Spectrum sc = paramSet.GetOneSpectrum("scale", Spectrum(1.0));
    Float fov = paramSet.GetOneFloat("fov", 45.);

    std::string texname = AbsolutePath(ResolveFilename(paramSet.GetOneString("mapname", "")));
    absl::optional<Image> image;
    if (texname != "")
        image = Image::Read(texname);
    if (!image) {
        std::vector<Float> one = {(Float)1};
        image = Image(std::move(one), PixelFormat::Y32, {1, 1});
    }

    return std::make_shared<ProjectionLight>(light2world, medium, I * sc,
                                             std::move(*image), fov, attributes);
}

}  // namespace pbrt
