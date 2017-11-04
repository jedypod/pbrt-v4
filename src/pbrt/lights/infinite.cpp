
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

// lights/infinite.cpp*
#include <pbrt/lights/infinite.h>
#include <pbrt/core/sampling.h>
#include <pbrt/core/paramset.h>
#include <pbrt/util/fileutil.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/stats.h>

namespace pbrt {

// InfiniteAreaLight Method Definitions
InfiniteAreaLight::InfiniteAreaLight(const Transform &LightToWorld,
                                     const Spectrum &L,
                                     Image im,
                                     const std::shared_ptr<const ParamSet> &attributes)
    : Light((int)LightFlags::Infinite, LightToWorld, MediumInterface(),
            attributes),
      image(std::move(im)),
      wrapMode(WrapMode::Repeat, WrapMode::Clamp),
      Lscale(L) {
    // Initialize sampling PDFs for infinite area light
    auto dwdA = [&](Point2f p) {
        // TODO: fixme? Do something slightly less accurate but more efficient?
        // Can take advantage of knowing that p[1] in [0,1].
        return std::sin(Pi * p[1]);
    };
    distribution = image.ComputeSamplingDistribution(dwdA, 1, Norm::L2, wrapMode);
}

Spectrum InfiniteAreaLight::Phi() const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    Spectrum sumL(0.);

    int width = image.resolution.x, height = image.resolution.y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u)
            sumL += sinTheta *
                image.GetSpectrum({u, v}, SpectrumType::Illuminant, wrapMode);
    }
    // Integrating over spherical coordinates, so have a scale of Pi for
    // theta and 2 Pi for phi.  Then one more for Pi r^2 for the area of
    // the disk receiving illumination...
    return 2 * Pi * Pi * Pi * worldRadius * worldRadius * Lscale * sumL /
        (width * height);
}

Spectrum InfiniteAreaLight::Le(const RayDifferential &ray) const {
    Vector3f w = Normalize(WorldToLight(ray.d));
    Point2f st(SphericalPhi(w) * Inv2Pi, SphericalTheta(w) * InvPi);
    return Lscale * image.BilerpSpectrum(st, SpectrumType::Illuminant, wrapMode);
}

Spectrum InfiniteAreaLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                      Vector3f *wi, Float *pdf,
                                      VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPdf;
    Point2f uv = distribution.SampleContinuous(u, &mapPdf);
    if (mapPdf == 0) return Spectrum(0.f);

    // Convert infinite light sample point to direction
    Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    *wi =
        LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));

    // Compute PDF for sampled infinite light direction
    *pdf = mapPdf / (2 * Pi * Pi * sinTheta);
    if (sinTheta == 0) *pdf = 0;

    // Return radiance value for infinite light direction
    *vis = VisibilityTester(ref, Interaction(ref.p + *wi * (2 * worldRadius),
                                             ref.time, mediumInterface));
    return Lscale * image.BilerpSpectrum(uv, SpectrumType::Illuminant, wrapMode);
}

Float InfiniteAreaLight::Pdf_Li(const Interaction &, const Vector3f &w) const {
    ProfilePhase _(Prof::LightPdf);
    Vector3f wi = WorldToLight(w);
    Float theta = SphericalTheta(wi), phi = SphericalPhi(wi);
    Float sinTheta = std::sin(theta);
    if (sinTheta == 0) return 0;
    return distribution.Pdf(Point2f(phi * Inv2Pi, theta * InvPi)) /
           (2 * Pi * Pi * sinTheta);
}

Spectrum InfiniteAreaLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                      Float time, Ray *ray, Normal3f *nLight,
                                      Float *pdfPos, Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);
    // Compute direction for infinite light sample ray
    Point2f u = u1;

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPdf;
    Point2f uv = distribution.SampleContinuous(u, &mapPdf);
    if (mapPdf == 0) return Spectrum(0.f);
    Float theta = uv[1] * Pi, phi = uv[0] * 2.f * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    Vector3f d =
        -LightToWorld(Vector3f(sinTheta * cosPhi, sinTheta * sinPhi, cosTheta));
    *nLight = (Normal3f)d;

    // Compute origin for infinite light sample ray
    Vector3f v1, v2;
    CoordinateSystem(-d, &v1, &v2);
    Point2f cd = ConcentricSampleDisk(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    *ray = Ray(pDisk + worldRadius * -d, d, Infinity, time);

    // Compute _InfiniteAreaLight_ ray PDFs
    *pdfDir = sinTheta == 0 ? 0 : mapPdf / (2 * Pi * Pi * sinTheta);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    return Lscale * image.BilerpSpectrum(uv, SpectrumType::Illuminant, wrapMode);
}

void InfiniteAreaLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                               Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    Vector3f d = -WorldToLight(ray.d);
    Float theta = SphericalTheta(d), phi = SphericalPhi(d);
    Point2f uv(phi * Inv2Pi, theta * InvPi);
    Float mapPdf = distribution.Pdf(uv);
    *pdfDir = mapPdf / (2 * Pi * Pi * std::sin(theta));
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

std::shared_ptr<InfiniteAreaLight> CreateInfiniteLight(
    const Transform &light2world, const ParamSet &paramSet,
    const std::shared_ptr<const ParamSet> &attributes) {
    Spectrum L = paramSet.GetOneSpectrum("L", Spectrum(1.0));
    Spectrum sc = paramSet.GetOneSpectrum("scale", Spectrum(1.0));
    std::string texmap = AbsolutePath(ResolveFilename(paramSet.GetOneString("mapname", "")));

    absl::optional<Image> image;
    if (texmap != "")
        image = Image::Read(texmap);
    if (!image) {
        std::vector<Float> one = {(Float)1};
        image = Image(std::move(one), PixelFormat::Y32, {1, 1});
    }

    return std::make_shared<InfiniteAreaLight>(light2world, L * sc,
                                               std::move(*image), attributes);
}

}  // namespace pbrt
