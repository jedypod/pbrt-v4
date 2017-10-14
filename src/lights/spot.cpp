
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

// lights/spot.cpp*
#include "lights/spot.h"
#include "paramset.h"
#include "reflection.h"
#include "sampling.h"
#include "util/mathutil.h"
#include "util/stats.h"

namespace pbrt {

// SpotLight Method Definitions
SpotLight::SpotLight(const Transform &LightToWorld,
                     const MediumInterface &mediumInterface, const Spectrum &I,
                     Float totalWidth, Float falloffStart,
                     const std::shared_ptr<const ParamSet> &attributes)
    : Light((int)LightFlags::DeltaPosition, LightToWorld, mediumInterface,
            attributes),
      pLight(LightToWorld(Point3f(0, 0, 0))),
      I(I),
      cosFalloffEnd(std::cos(Radians(totalWidth))),
      cosFalloffStart(std::cos(Radians(falloffStart))) {
    CHECK_LE(falloffStart, totalWidth);

    // Compute coefficients for CDF polynomial
    // cdf[t] = c0 + c1 Cos[t] + c2 Cos[t]^2 + ...
    c[0] = (1280 * Pow<3>(cosFalloffEnd) * Pow<2>(cosFalloffStart) -
            40 * (17 + 2 * (-1 + 2 * Pow<2>(cosFalloffEnd))) *
                Pow<3>(cosFalloffStart) +
            640 * cosFalloffEnd * Pow<4>(cosFalloffStart) -
            88 * Pow<5>(cosFalloffStart) -
            5 * cosFalloffStart *
                (-74 + 240 * Pow<2>(cosFalloffEnd) +
                 16 * (-1 + 2 * Pow<2>(-1 + 2 * Pow<2>(cosFalloffEnd))) +
                 56 * (-1 + 2 * Pow<2>(cosFalloffEnd)) *
                     (-1 + 2 * Pow<2>(cosFalloffStart)) +
                 2 * Pow<2>(-1 + 2 * Pow<2>(cosFalloffStart)))) /
           (128. * Pow<5>(cosFalloffEnd - cosFalloffStart));
    c[1] = 5 * Pow<4>(cosFalloffEnd) / Pow<5>(cosFalloffEnd - cosFalloffStart);
    c[2] =
        -10 * Pow<3>(cosFalloffEnd) / Pow<5>(cosFalloffEnd - cosFalloffStart);
    c[3] = 10 * Pow<2>(cosFalloffEnd) / Pow<5>(cosFalloffEnd - cosFalloffStart);
    c[4] = -5 * cosFalloffEnd / Pow<5>(cosFalloffEnd - cosFalloffStart);
    c[5] = 1 / Pow<5>(cosFalloffEnd - cosFalloffStart);
}

Spectrum SpotLight::Sample_Li(const Interaction &ref, const Point2f &u,
                              Vector3f *wi, Float *pdf,
                              VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    *wi = Normalize(pLight - ref.p);
    *pdf = 1.f;
    *vis =
        VisibilityTester(ref, Interaction(pLight, ref.time, mediumInterface));
    return I * Falloff(-*wi) / DistanceSquared(pLight, ref.p);
}

Float SpotLight::Falloff(const Vector3f &w) const {
    Vector3f wl = Normalize(WorldToLight(w));
    Float cosTheta = wl.z;
    if (cosTheta < cosFalloffEnd) return 0;
    if (cosTheta >= cosFalloffStart) return 1;
    // Compute falloff inside spotlight cone
    Float delta =
        (cosTheta - cosFalloffEnd) / (cosFalloffStart - cosFalloffEnd);
    return Pow<4>(delta);
}

Spectrum SpotLight::Phi() const {
    // int_0^start sin theta dtheta = 1 - cosFalloffStart
    // int_start^end [(cos theta - cos end) / (cos start - cos end)]^4
    //      sin theta dtheta = (cos start - cos end) / 5 (!!!!!)
    return I * 2 * Pi *
           ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 5);
}

Float SpotLight::Pdf_Li(const Interaction &, const Vector3f &) const {
    return 0.f;
}

Spectrum SpotLight::Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                              Ray *ray, Normal3f *nLight, Float *pdfPos,
                              Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);
    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 5};
    Float sectionPdf;
    Vector3f w;
    int section = SampleDiscrete(p, u2[0], &sectionPdf);
    if (section == 0) {
        // Sample center cone
        w = UniformSampleCone(u1, cosFalloffStart);
        *pdfDir = sectionPdf * UniformConePdf(cosFalloffStart);
    } else {
        DCHECK_EQ(1, section);
        // Sample the falloff region
        // See notes/sample-spotlight.nb

        // Want to solve u1[0] = cdf[t] for Cos[t], or
        // CDF(theta) - u1[0] = 0.
        auto CDFZero = [=](Float cosTheta) -> std::pair<Float, Float> {
            return std::make_pair(-u1[0] +
                c[0] + c[1] * cosTheta + c[2] * Pow<2>(cosTheta) +
                    c[3] * Pow<3>(cosTheta) + c[4] * Pow<4>(cosTheta) +
                    c[5] * Pow<5>(cosTheta),
                c[1] + 2 * c[2] * cosTheta + 3 * c[3] * Pow<2>(cosTheta) +
                    4 * c[4] * Pow<3>(cosTheta) + 5 * c[5] * Pow<4>(cosTheta));
        };

        Float cosTheta =
            NewtonBisection(cosFalloffEnd, cosFalloffStart, CDFZero);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Float phi = u1[1] * 2 * Pi;
        w = SphericalDirection(sinTheta, cosTheta, phi);
        *pdfDir = sectionPdf * 5 * Pow<4>(cosTheta - cosFalloffEnd) /
                  (2 * Pi * Pow<5>(cosFalloffStart - cosFalloffEnd));
    }

    *ray = Ray(pLight, LightToWorld(w), Infinity, time, mediumInterface.inside);
    *nLight = (Normal3f)ray->d;
    *pdfPos = 1;
    return I * Falloff(ray->d);
}

void SpotLight::Pdf_Le(const Ray &ray, const Normal3f &, Float *pdfPos,
                       Float *pdfDir) const {
    ProfilePhase _(Prof::LightPdf);
    *pdfPos = 0;

    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 5};

    Float cosTheta = CosTheta(WorldToLight(ray.d));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePdf(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = 5 * Pow<4>(cosTheta - cosFalloffEnd) /
                  (2 * Pi * Pow<5>(cosFalloffStart - cosFalloffEnd)) *
                  (p[1] / (p[0] + p[1]));
}

std::shared_ptr<SpotLight> CreateSpotLight(
    const Transform &l2w, const Medium *medium, const ParamSet &paramSet,
    const std::shared_ptr<const ParamSet> &attributes) {
    Spectrum I = paramSet.GetOneSpectrum("I", Spectrum(1.0));
    Spectrum sc = paramSet.GetOneSpectrum("scale", Spectrum(1.0));
    Float coneangle = paramSet.GetOneFloat("coneangle", 30.);
    Float conedelta = paramSet.GetOneFloat("conedeltaangle", 5.);
    // Compute spotlight world to light transformation
    Point3f from = paramSet.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = paramSet.GetOnePoint3f("to", Point3f(0, 0, 1));
    Vector3f dir = Normalize(to - from);
    Vector3f du, dv;
    CoordinateSystem(dir, &du, &dv);
    Transform dirToZ =
        Transform(Matrix4x4(du.x, du.y, du.z, 0., dv.x, dv.y, dv.z, 0., dir.x,
                            dir.y, dir.z, 0., 0, 0, 0, 1.));
    Transform light2world =
        l2w * Translate(Vector3f(from.x, from.y, from.z)) * Inverse(dirToZ);
    return std::make_shared<SpotLight>(light2world, medium, I * sc, coneangle,
                                       coneangle - conedelta, attributes);
}

}  // namespace pbrt
