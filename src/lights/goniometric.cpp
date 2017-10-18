
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

// lights/goniometric.cpp*
#include "lights/goniometric.h"
#include "paramset.h"
#include "sampling.h"
#include "util/stats.h"

namespace pbrt {

// GonioPhotometricLight Method Definitions
GonioPhotometricLight::GonioPhotometricLight(const Transform &LightToWorld,
    const MediumInterface &mediumInterface, const Spectrum &I,
    Image im, const std::shared_ptr<const ParamSet> &attributes)
    : Light((int)LightFlags::DeltaPosition, LightToWorld, mediumInterface,
            attributes),
          pLight(LightToWorld(Point3f(0, 0, 0))),
          I(I),
          image(std::move(im)),
          wrapMode(WrapMode::Repeat, WrapMode::Clamp) {
    auto dwdA = [&](Point2f p) {
        // TODO: improve efficiency?
        return std::sin(Pi * p[1]);
    };
    distrib = image.ComputeSamplingDistribution(dwdA, 1, Norm::L2, wrapMode);
}

Spectrum GonioPhotometricLight::Sample_Li(const Interaction &ref,
                                          const Point2f &u, Vector3f *wi,
                                          Float *pdf,
                                          VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    *wi = Normalize(pLight - ref.p);
    *pdf = 1.f;
    *vis =
        VisibilityTester(ref, Interaction(pLight, ref.time, mediumInterface));
    return Scale(-*wi) / DistanceSquared(pLight, ref.p);
}

Spectrum GonioPhotometricLight::Phi() const {
    // integrate over speherical coordinates [0,Pi], [0,2pi]
    Spectrum sumL(0.);
    int width = image.resolution.x, height = image.resolution.y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u) {
            sumL += sinTheta * I *
                image.GetSpectrum({u, v}, SpectrumType::Illuminant, wrapMode);
        }
    }
    return 2 * Pi * Pi * sumL / (width * height);
}

Float GonioPhotometricLight::Pdf_Li(const Interaction &,
                                    const Vector3f &) const {
    return 0.f;
}

Spectrum GonioPhotometricLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                          Float time, Ray *ray,
                                          Normal3f *nLight, Float *pdfPos,
                                          Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);

    Float pdf;
    Point2f uv = distrib.SampleContinuous(u1, &pdf);
    Float theta = uv[1] * Pi, phi = uv[0] * 2 * Pi;
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
    // swap y and z
    Vector3f w =
        LightToWorld(Vector3f(sinTheta * cosPhi, cosTheta, sinTheta * sinPhi));
    *pdfDir = sinTheta == 0 ? 0 : pdf / (2 * Pi * Pi * sinTheta);

    *ray = Ray(pLight, w, Infinity, time, mediumInterface.inside);
    *nLight = (Normal3f)ray->d;
    *pdfPos = 1.f;
    return Scale(ray->d);
}

void GonioPhotometricLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                                   Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    *pdfPos = 0.f;

    Vector3f w = Normalize(WorldToLight(ray.d));
    std::swap(w.y, w.z);
    Float theta = SphericalTheta(w), phi = SphericalPhi(w);
    Point2f uv(phi * Inv2Pi, theta * InvPi);
    *pdfDir = distrib.Pdf(uv) / (2 * Pi * Pi * std::sin(theta));
}

std::shared_ptr<GonioPhotometricLight> CreateGoniometricLight(
    const Transform &light2world, const Medium *medium,
    const ParamSet &paramSet, const std::shared_ptr<const ParamSet> &attributes) {
    Spectrum I = paramSet.GetOneSpectrum("I", Spectrum(1.0));
    Spectrum sc = paramSet.GetOneSpectrum("scale", Spectrum(1.0));

    std::string texname = paramSet.GetOneFilename("mapname", "");
    absl::optional<Image> image;
    if (texname != "")
        image = Image::Read(texname);
    if (!image) {
        std::vector<Float> one = {(Float)1};
        image = Image(std::move(one), PixelFormat::Y32, {1, 1});
    }

    return std::make_shared<GonioPhotometricLight>(light2world, medium, I * sc,
                                                   std::move(*image), attributes);
}

}  // namespace pbrt
