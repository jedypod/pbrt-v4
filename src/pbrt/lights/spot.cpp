
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
#include <pbrt/lights/spot.h>
#include <pbrt/core/paramset.h>
#include <pbrt/core/reflection.h>
#include <pbrt/core/sampling.h>
#include <pbrt/util/math.h>
#include <pbrt/util/stats.h>

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
}

Spectrum SpotLight::Sample_Li(const Interaction &ref, const Point2f &u,
                              Vector3f *wi, Float *pdf,
                              VisibilityTester *vis) const {
    ProfilePhase _(Prof::LightSample);
    *wi = Normalize(pLight - ref.p);
    *pdf = 1;
    *vis =
        VisibilityTester(ref, Interaction(pLight, ref.time, mediumInterface));
    return I * Falloff(-*wi) / DistanceSquared(pLight, ref.p);
}

Float SpotLight::Falloff(const Vector3f &w) const {
    Vector3f wl = Normalize(WorldToLight(w));
    Float cosTheta = wl.z;  // CosTheta(wl);
    if (cosTheta >= cosFalloffStart) return 1;
    // Compute falloff inside spotlight cone
    return Smoothstep(cosTheta, cosFalloffEnd, cosFalloffStart);
}

Spectrum SpotLight::Phi() const {
    // int_0^start sin theta dtheta = 1 - cosFalloffStart
    // See notes/sample-spotlight.nb for the falloff part:
    // int_start^end smoothstep(cost, end, start) sin theta dtheta =
    //  (cosStart - cosEnd) / 2
    return I * 2 * Pi * ((1 - cosFalloffStart) +
                         (cosFalloffStart - cosFalloffEnd) / 2);
}

Float SpotLight::Pdf_Li(const Interaction &, const Vector3f &) const {
    return 0.f;
}

Spectrum SpotLight::Sample_Le(const Point2f &u1, const Point2f &u2, Float time,
                              Ray *ray, Normal3f *nLight, Float *pdfPos,
                              Float *pdfDir) const {
    ProfilePhase _(Prof::LightSample);
    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    Float sectionPdf;
    Vector3f w;
    int section = SampleDiscrete(p, u2[0], &sectionPdf);
    if (section == 0) {
        // Sample center cone
        w = UniformSampleCone(u1, cosFalloffStart);
        *pdfDir = sectionPdf * UniformConePdf(cosFalloffStart);
    } else {
        DCHECK_EQ(1, section);

        Float cosTheta = SampleSmoothstep(u1[0], cosFalloffEnd,
                                          cosFalloffStart);
        CHECK(cosTheta >= cosFalloffEnd && cosTheta <= cosFalloffStart);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Float phi = u1[1] * 2 * Pi;
        w = SphericalDirection(sinTheta, cosTheta, phi);
        *pdfDir = sectionPdf * SmoothstepPdf(cosTheta, cosFalloffEnd,
                                             cosFalloffStart) / (2 * Pi);
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
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};

    Float cosTheta = CosTheta(WorldToLight(ray.d));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePdf(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = SmoothstepPdf(cosTheta, cosFalloffEnd, cosFalloffStart) /
            (2 * Pi) * (p[1] / (p[0] + p[1]));
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
