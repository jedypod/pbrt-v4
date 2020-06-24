
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

// lights/point.cpp*
#include <pbrt/lights.h>

#include <pbrt/integrators.h>
#include <pbrt/paramdict.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

namespace pbrt {

STAT_COUNTER("Scene/Lights", numLights);
STAT_COUNTER("Scene/AreaLights", numAreaLights);

// Light Method Definitions
std::string ToString(LightType lf) {
    switch (lf) {
    case LightType::DeltaPosition:
        return "DeltaPosition";
    case LightType::DeltaDirection:
        return "DeltaDirection,";
    case LightType::Area:
        return "Area";
    case LightType::Infinite:
        return "Infinite";
    default:
        LOG_FATAL("Unhandled type");
        return "";
    }
}

Light::Light(LightType type, const AnimatedTransform &worldFromLight,
             const MediumInterface &mediumInterface)
    : type(type),
      mediumInterface(mediumInterface),
      worldFromLight(worldFromLight) {
    ++numLights;
}

std::string Light::BaseToString() const {
    return StringPrintf("type: %s mediumInterface: %s worldFromLight: %s",
                        type, mediumInterface, worldFromLight);
}

std::string LightBounds::ToString() const {
    return StringPrintf("[ LightBounds b: %s w: %s phi: %f theta_o: %f theta_e: %f "
                        "cosTheta_o: %f cosTheta_e: %f twoSided: %s ]",
                        b, w, phi, theta_o, theta_e, cosTheta_o, cosTheta_e, twoSided);
}

LightBounds Union(const LightBounds &a, const LightBounds &b) {
    if (a.phi == 0)
        return b;
    if (b.phi == 0)
        return a;
    DirectionCone c = Union(DirectionCone(a.w, a.cosTheta_o),
                            DirectionCone(b.w, b.cosTheta_o));
    Float theta_o = SafeACos(c.cosTheta);
    return LightBounds(Union(a.b, b.b), c.w, a.phi + b.phi, theta_o,
                       std::max(a.theta_e, b.theta_e), a.twoSided | b.twoSided);
}

bool LightLiSample::Unoccluded(const Scene &scene) const {
    return !scene.IntersectP(pRef.SpawnRayTo(pLight), 1 - ShadowEpsilon);
}

SampledSpectrum LightLiSample::Tr(const Scene &scene,
                                  const SampledWavelengths &lambda,
                                  Sampler &sampler) const {
    return pbrt::Tr(scene, lambda, sampler, pRef, pLight);
}

// PointLight Method Definitions
SampledSpectrum PointLight::Phi(const SampledWavelengths &lambda) const {
    return 4 * Pi * I.Sample(lambda);
}

pstd::optional<LightLeSample> PointLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);
    Point3f p = worldFromLight(Point3f(0, 0, 0), time);
    Ray ray(p, SampleUniformSphere(u1), time, mediumInterface.outside);
    return LightLeSample(I.Sample(lambda), ray, 1, UniformSpherePDF());
}

void PointLight::Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);
    *pdfPos = 0;
    *pdfDir = UniformSpherePDF();
}

LightBounds PointLight::Bounds() const {
    Point3f p = worldFromLight(Point3f(0, 0, 0), 0 /* TODO: time?? */);
    return LightBounds(p, Vector3f(0, 0, 1),
                       4 * Pi * I.MaxValue(),
                       Pi, Pi / 2, false);
}

std::string PointLight::ToString() const {
    return StringPrintf("[ PointLight %s I: %s ]", BaseToString(), I);
}

PointLight *PointLight::Create(
    const AnimatedTransform &worldFromLight, const Medium *medium,
    const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
    Allocator alloc) {
    SpectrumHandle I = dict.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        I = alloc.new_object<ScaledSpectrum>(sc, I);

    Point3f from = dict.GetOnePoint3f("from", Point3f(0, 0, 0));
    Transform tf = Translate(Vector3f(from.x, from.y, from.z));
    AnimatedTransform worldFromLightAnim(alloc.new_object<Transform>(*worldFromLight.startTransform * tf),
                                         worldFromLight.startTime,
                                         alloc.new_object<Transform>(*worldFromLight.endTransform * tf),
                                         worldFromLight.endTime);

    return alloc.new_object<PointLight>(worldFromLightAnim, medium, I, alloc);
}

// DistantLight Method Definitions
DistantLight::DistantLight(const AnimatedTransform &worldFromLight,
                           SpectrumHandle L, Allocator alloc)
    : Light(LightType::DeltaDirection, worldFromLight, nullptr),
      L(L, alloc) {}

SampledSpectrum DistantLight::Phi(const SampledWavelengths &lambda) const {
    return L.Sample(lambda) * Pi * worldRadius * worldRadius;
}

pstd::optional<LightLeSample> DistantLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);
    // Choose point on disk oriented toward infinite light direction
    Vector3f w = Normalize(worldFromLight(Vector3f(0, 0, 1), time));
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u1);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);

    // Set ray origin and direction for infinite light ray
    Ray ray(pDisk + worldRadius * w, -w, time);
    return LightLeSample(L.Sample(lambda), ray,
                         1 / (Pi * worldRadius * worldRadius), 1);
}

void DistantLight::Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
    *pdfDir = 0;
}

std::string DistantLight::ToString() const {
    return StringPrintf("[ DistantLight %s L: %s ]", BaseToString(), L);
}

DistantLight *DistantLight::Create(
    const AnimatedTransform &worldFromLight, const ParameterDictionary &dict,
    const RGBColorSpace *colorSpace, Allocator alloc) {
    SpectrumHandle L = dict.GetOneSpectrum("L", &colorSpace->illuminant,
                                           SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        L = alloc.new_object<ScaledSpectrum>(sc, L);

    Point3f from = dict.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = dict.GetOnePoint3f("to", Point3f(0, 0, 1));

    Vector3f w = Normalize(from - to);
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Float m[4][4] = { v1.x, v2.x, w.x, 0,
                      v1.y, v2.y, w.y, 0,
                      v1.z, v2.z, w.z, 0,
                      0, 0, 0, 1 };
    Transform t(m);
    AnimatedTransform worldFromLightAnim(alloc.new_object<Transform>(*worldFromLight.startTransform * t),
                                         worldFromLight.startTime,
                                         alloc.new_object<Transform>(*worldFromLight.endTransform * t),
                                         worldFromLight.endTime);

    return alloc.new_object<DistantLight>(worldFromLightAnim, L, alloc);
}

STAT_MEMORY_COUNTER("Memory/Light image and distributions", imageBytes);

// ProjectionLight Method Definitions
ProjectionLight::ProjectionLight(const AnimatedTransform &worldFromLight,
                                 const MediumInterface &mediumInterface,
                                 Image im, const RGBColorSpace *imageColorSpace,
                                 Float scale, Float fov, Allocator alloc)
    : Light(LightType::DeltaPosition, worldFromLight, mediumInterface),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      distrib(alloc) {
    // Initialize _ProjectionLight_ projection matrix
    Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
    if (aspect > 1)
        screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
    else
        screenBounds =
            Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
    hither = 1e-3f;
    Float yon = 1e30f;
    ScreenFromLight = Perspective(fov, hither, yon);
    LightFromScreen = Inverse(ScreenFromLight);

    // Compute cosine of cone surrounding projection directions
    Float opposite = std::tan(Radians(fov) / 2.f);
    // Area of the image on projection plane.
    A = 4 * opposite * opposite * (aspect > 1 ? aspect : 1 / aspect);

    Point3f pCorner(screenBounds.pMax.x, screenBounds.pMax.y, 0);
    Vector3f wCorner = Normalize(Vector3f(LightFromScreen(pCorner)));
    cosTotalWidth = wCorner.z;

    pstd::optional<ImageChannelDesc> channelDesc = image.GetChannelDesc({ "R", "G", "B" });
    if (!channelDesc)
        ErrorExit("Image used for ProjectionLight doesn't have R, G, B channels.");
    CHECK_EQ(3, channelDesc->size());
    CHECK(channelDesc->IsIdentity());

    auto dwdA = [&](Point2f ps) {
        Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));
        w = Normalize(w);
        return Pow<3>(w.z);
    };
    Array2D<Float> d = image.ComputeSamplingDistribution(dwdA, *channelDesc,
                                                         1, screenBounds, Norm::L2);
    distrib = Distribution2D(d, screenBounds);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

pstd::optional<LightLiSample> ProjectionLight::Sample_Li(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda) const {
    ProfilerScope _(ProfilePhase::LightSample);
    Point3f p = worldFromLight(Point3f(0, 0, 0), ref.time);
    Vector3f wi = Normalize(p - ref.p());
    Vector3f wl = worldFromLight.ApplyInverse(-wi, ref.time);
    return LightLiSample(this, Projection(wl, lambda) / DistanceSquared(p, ref.p()), wi, 1,
                         ref, Interaction(p, ref.time, &mediumInterface));
}

// Takes wl already in light coordinate system!
SampledSpectrum ProjectionLight::Projection(const Vector3f &wl,
                                            const SampledWavelengths &lambda) const {
    // Discard directions behind projection light
    if (wl.z < hither) return SampledSpectrum(0.);

    // Project point onto projection plane and compute light
    Point3f ps = ScreenFromLight(Point3f(wl.x, wl.y, wl.z));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds))
        return SampledSpectrum(0.f);
    Point2f st = Point2f(screenBounds.Offset(Point2f(ps.x, ps.y)));

    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.BilerpChannel(st, c);

    return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
}

SampledSpectrum ProjectionLight::Phi(const SampledWavelengths &lambda) const {
    SampledSpectrum sum(0.f);
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u) {
            Point2f ps = screenBounds.Lerp({(u + .5f) / image.Resolution().x,
                                            (v + .5f) / image.Resolution().y});
            Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(w.z);

            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({u, v}, c);

            SampledSpectrum L = RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);

            sum += L * dwdA;
        }

    return scale * A * sum / (image.Resolution().x * image.Resolution().y);
}

Float ProjectionLight::Pdf_Li(const Interaction &, const Vector3f &) const {
    return 0.f;
}

pstd::optional<LightLeSample> ProjectionLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);

    Float pdf;
    Point2f ps = distrib.SampleContinuous(u1, &pdf);
    if (pdf == 0)
        return {};

    Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));

    Ray ray = worldFromLight(Ray(Point3f(0, 0, 0), Normalize(w), time,
                                 mediumInterface.outside));
    Float cosTheta = CosTheta(Normalize(w));
    CHECK_GT(cosTheta, 0);
    Float pdfDir = pdf * screenBounds.Area() / (A * Pow<3>(cosTheta));

    Point2f pi = Point2f(screenBounds.Offset(ps));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.BilerpChannel(pi, c);

    SampledSpectrum L = scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);

    return LightLeSample(L, ray, 1, pdfDir);
}

void ProjectionLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);
    *pdfPos = 0;

    Vector3f w = Normalize(worldFromLight.ApplyInverse(ray.d, ray.time));
    if (w.z < hither) {
        *pdfDir = 0;
        return;
    }
    Point3f ps = ScreenFromLight(Point3f(w));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds)) {
        *pdfDir = 0;
        return;
    }
    *pdfDir = distrib.ContinuousPDF(Point2f(ps.x, ps.y)) * screenBounds.Area() /
        (A * Pow<3>(w.z));
}

LightBounds ProjectionLight::Bounds() const {
#if 0
    // Along the lines of Phi()
    Float sum = 0;
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u) {
            Point2f ps = screenBounds.Lerp({(u + .5f) / image.Resolution().x,
                                            (v + .5f) / image.Resolution().y});
            Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));
            w = Normalize(w);
            Float dwdA = Pow<3>(w.z);
            sum += image.GetChannels({u, v}, rgbChannelDesc).MaxValue() * dwdA;
        }
    Float phi = scale * A * sum / (image.Resolution().x * image.Resolution().y);
#else
    // See comment in SpotLight::Bounds()
    Float sum = 0;
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u)
            sum += std::max({image.GetChannel({u, v}, 0),
                             image.GetChannel({u, v}, 1),
                             image.GetChannel({u, v}, 2)});
    Float phi = scale * sum / (image.Resolution().x * image.Resolution().y);
#endif
    Point3f p = worldFromLight(Point3f(0, 0, 0), 0.);  /* TODO: handle animation */
    Vector3f w = Normalize(worldFromLight(Vector3f(0, 0, 1), 0.));
    return LightBounds(p, w, phi, 0.f, std::acos(cosTotalWidth), false);
}

std::string ProjectionLight::ToString() const {
    return StringPrintf("[ ProjectionLight %s scale: %f A: %f cosTotalWidth: %f ]",
                        BaseToString(), scale, A, cosTotalWidth);
}

ProjectionLight *ProjectionLight::Create(
    const AnimatedTransform &worldFromLight, const Medium *medium,
    const ParameterDictionary &dict, Allocator alloc) {
    Float scale = dict.GetOneFloat("scale", 1);
    Float fov = dict.GetOneFloat("fov", 90.);

    std::string texname = ResolveFilename(dict.GetOneString("imagefile", ""));
    if (texname.empty())
        ErrorExit("Must provide \"imagefile\" to \"projection\" light source");

    pstd::optional<ImageAndMetadata> imageAndMetadata = Image::Read(texname, alloc);
    if (!imageAndMetadata)
        return nullptr;
    const RGBColorSpace *colorSpace = imageAndMetadata->metadata.GetColorSpace();

    pstd::optional<ImageChannelDesc> channelDesc =
        imageAndMetadata->image.GetChannelDesc({ "R", "G", "B" });
    if (!channelDesc)
        ErrorExit("Image provided to \"projection\" light must have R, G, and B channels.");
    Image image = imageAndMetadata->image.SelectChannels(*channelDesc, alloc);

    Transform flip = Scale(1, -1, 1);
    AnimatedTransform worldFromLightFlipY(alloc.new_object<Transform>(*worldFromLight.startTransform * flip),
                                          worldFromLight.startTime,
                                          alloc.new_object<Transform>(*worldFromLight.endTransform * flip),
                                          worldFromLight.endTime);

    return alloc.new_object<ProjectionLight>(worldFromLightFlipY, medium,
                                             std::move(image), colorSpace,
                                             scale, fov, alloc);
}

// GoniometricLight Method Definitions
GoniometricLight::GoniometricLight(const AnimatedTransform &worldFromLight,
    const MediumInterface &mediumInterface, SpectrumHandle I,
    Image im, const RGBColorSpace *imageColorSpace, Allocator alloc)
    : Light(LightType::DeltaPosition, worldFromLight, mediumInterface),
      I(I, alloc),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      wrapMode(WrapMode::Repeat, WrapMode::Clamp),
      distrib(alloc) {
    auto dwdA = [](Point2f p) {
        // TODO: improve efficiency?
        return std::sin(p[1]);
    };

    CHECK_EQ(1, image.NChannels());

    Bounds2f domain(Point2f(0, 0), Point2f(2 * Pi, Pi));
    Array2D<Float> d =
        image.ComputeSamplingDistribution(dwdA, image.AllChannelsDesc(),
                                          1, domain, Norm::L2, wrapMode);
    distrib = Distribution2D(d, domain);
    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

pstd::optional<LightLiSample> GoniometricLight::Sample_Li(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda) const {
    ProfilerScope _(ProfilePhase::LightSample);
    Point3f p = worldFromLight(Point3f(0, 0, 0), ref.time);
    Vector3f wi = Normalize(p - ref.p());
    SampledSpectrum L = Scale(worldFromLight.ApplyInverse(-wi, ref.time), lambda) /
        DistanceSquared(p, ref.p());
    return LightLiSample(this, L, wi, 1, ref, Interaction(p, ref.time, &mediumInterface));
}

SampledSpectrum GoniometricLight::Phi(const SampledWavelengths &lambda) const {
    // integrate over speherical coordinates [0,Pi], [0,2pi]
    Float sumY = 0;
    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u)
            sumY += sinTheta * image.GetChannels({u, v}, wrapMode).Average();
    }
    return I.Sample(lambda) * 2 * Pi * Pi * sumY / (width * height);
}

Float GoniometricLight::Pdf_Li(const Interaction &, const Vector3f &) const {
    return 0.f;
}

pstd::optional<LightLeSample> GoniometricLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);

    Float pdf;
    Point2f uv = distrib.SampleContinuous(u1, &pdf);
    Float theta = uv[1], phi = uv[0];
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Vector3f wl = SphericalDirection(sinTheta, cosTheta, phi);
    Float pdfDir = sinTheta == 0 ? 0 : pdf / sinTheta;

    Ray ray = worldFromLight(Ray(Point3f(0, 0, 0), wl, time,
                                 mediumInterface.inside));
    return LightLeSample(Scale(wl, lambda), ray, 1, pdfDir);
}

void GoniometricLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);
    *pdfPos = 0.f;

    Vector3f wl = Normalize(worldFromLight.ApplyInverse(ray.d, ray.time));
    Float theta = SphericalTheta(wl), phi = SphericalPhi(wl);
    *pdfDir = distrib.ContinuousPDF(Point2f(phi, theta)) / std::sin(theta);
}

LightBounds GoniometricLight::Bounds() const {
    // Like Phi() method, but compute the weighted max component value of
    // the image map.
    Float weightedMaxImageSum = 0;
    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u)
            weightedMaxImageSum += sinTheta * image.GetChannels({u, v}, wrapMode).MaxValue();
    }
    Float phi = I.MaxValue() * 2 * Pi * Pi * weightedMaxImageSum /
        (width * height);

    Point3f p = worldFromLight(Point3f(0, 0, 0), 0 /* TODO: time?? */);
    // Bound it as an isotropic point light.
    return LightBounds(p, Vector3f(0, 0, 1), phi, Pi, Pi / 2, false);
}

std::string GoniometricLight::ToString() const {
    return StringPrintf("[ GoniometricLight %s I: %s ]", BaseToString(), I);
}

GoniometricLight *GoniometricLight::Create(
    const AnimatedTransform &worldFromLight, const Medium *medium,
    const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
    Allocator alloc) {
    SpectrumHandle I = dict.GetOneSpectrum("I", &colorSpace->illuminant,
                                            SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        I = alloc.new_object<ScaledSpectrum>(sc, I);

    std::string texname = ResolveFilename(dict.GetOneString("imagefile", ""));
    pstd::optional<ImageAndMetadata> imageAndMetadata;
    if (!texname.empty())
        imageAndMetadata = Image::Read(texname, alloc);
    if (imageAndMetadata) {
        pstd::optional<ImageChannelDesc> rgbDesc = imageAndMetadata->image.GetChannelDesc({"R", "G", "B"});
        pstd::optional<ImageChannelDesc> yDesc = imageAndMetadata->image.GetChannelDesc({"Y"});
        if (!rgbDesc && !yDesc)
            ErrorExit("%s: has neither \"R\", \"G\", and \"B\" or \"Y\" channels.", texname);
        if (rgbDesc && yDesc)
            ErrorExit("%s: has both \"R\", \"G\", and \"B\" or \"Y\" channels.", texname);
        if (rgbDesc) {
            Image image(imageAndMetadata->image.Format(),
                        imageAndMetadata->image.Resolution(), {"Y"},
                        imageAndMetadata->image.Encoding(), alloc);
            for (int y = 0; y < image.Resolution().y; ++y)
                for (int x = 0; x < image.Resolution().x; ++x)
                    image.SetChannel({x, y}, 0, imageAndMetadata->image.GetChannels({x, y}, *rgbDesc).Average());
            imageAndMetadata->image = std::move(image);
        }
    } else {
        pstd::vector<float> one(alloc);
        one.push_back(1.f);
        imageAndMetadata->image = Image(std::move(one), {1, 1}, { "Y" });
    }

    const RGBColorSpace *imageColorSpace = imageAndMetadata->metadata.GetColorSpace();

    const Float swapYZ[4][4] = { 1, 0, 0, 0,
                                 0, 0, 1, 0,
                                 0, 1, 0, 0,
                                 0, 0, 0, 1 };
    Transform t(swapYZ);
    AnimatedTransform worldFromLightAnim(alloc.new_object<Transform>(*worldFromLight.startTransform * t),
                                         worldFromLight.startTime,
                                         alloc.new_object<Transform>(*worldFromLight.endTransform * t),
                                         worldFromLight.endTime);

    return alloc.new_object<GoniometricLight>(worldFromLightAnim, medium, I,
                                              std::move(imageAndMetadata->image),
                                              imageColorSpace, alloc);
}

// DiffuseAreaLight Method Definitions
DiffuseAreaLight::DiffuseAreaLight(const AnimatedTransform &worldFromLight,
                                   const MediumInterface &mediumInterface,
                                   SpectrumHandle Le, Float scale, const ShapeHandle shape,
                                   pstd::optional<Image> im, const RGBColorSpace *imageColorSpace,
                                   bool twoSided, Allocator alloc)
    : Light(LightType::Area, worldFromLight, mediumInterface),
      Lemit(Le, alloc),
      scale(scale),
      shape(shape),
      twoSided(twoSided),
      area(shape.Area()),
      imageColorSpace(imageColorSpace),
      image(std::move(im)) {
    ++numAreaLights;

    if (image) {
        pstd::optional<ImageChannelDesc> desc = image->GetChannelDesc({ "R", "G", "B" });
        if (!desc)
            ErrorExit("Image used for DiffuseAreaLight doesn't have R, G, B channels.");
        CHECK_EQ(3, desc->size());
        CHECK(desc->IsIdentity());
        CHECK(imageColorSpace != nullptr);
    } else {
        CHECK(Le);
    }

    // Warn if light has transformation with non-uniform scale, though not
    // for Triangles, since this doesn't matter for them.
    // FIXME: is this still true with animated transformations?
    if (worldFromLight.HasScale() && !shape.Is<Triangle>() && !shape.Is<BilinearPatch>())
        Warning(
            "Scaling detected in world to light transformation! "
            "The system has numerous assumptions, implicit and explicit, "
            "that this transform will have no scale factors in it. "
            "Proceed at your own risk; your image may have errors.");
}

SampledSpectrum DiffuseAreaLight::Phi(const SampledWavelengths &lambda) const {
    SampledSpectrum phi(0);
    if (image) {
        // Assume no distortion in the mapping, FWIW...
        for (int y = 0; y < image->Resolution().y; ++y)
            for (int x = 0; x < image->Resolution().x; ++x) {
                RGB rgb;
                for (int c = 0; c < 3; ++c)
                    rgb[c] = image->GetChannel({x, y}, c);
                phi += RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
            }
        phi /= image->Resolution().x * image->Resolution().y;
    }
    else
        phi = Lemit.Sample(lambda);

    return phi * (twoSided ? 2 : 1) * scale * area * Pi;
}

pstd::optional<LightLeSample> DiffuseAreaLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);
    // Sample a point on the area light's _Shape_, _pShape_
    Float pdfDir;
    pstd::optional<ShapeSample> ss = shape.Sample(u1);
    if (!ss) return {};
    if (worldFromLight.IsAnimated())
        ss->intr = worldFromLight(ss->intr);
    ss->intr.time = time;
    ss->intr.mediumInterface = &mediumInterface;

    // Sample a cosine-weighted outgoing direction _w_ for area light
    Vector3f w;
    if (twoSided) {
        Point2f u = u2;
        // Choose a side to sample and then remap u[0] to [0,1] before
        // applying cosine-weighted hemisphere sampling for the chosen side.
        if (u[0] < .5) {
            u[0] = std::min(u[0] * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u);
        } else {
            u[0] = std::min((u[0] - .5f) * 2, OneMinusEpsilon);
            w = SampleCosineHemisphere(u);
            w.z *= -1;
        }
        pdfDir = 0.5f * CosineHemispherePDF(std::abs(w.z));
    } else {
        w = SampleCosineHemisphere(u2);
        pdfDir = CosineHemispherePDF(w.z);
    }

    if (pdfDir == 0) return {};

    Frame nFrame = Frame::FromZ(ss->intr.n);
    w = nFrame.FromLocal(w);
    return LightLeSample(L(ss->intr, w, lambda), ss->intr.SpawnRay(w), ss->intr,
                         ss->pdf, pdfDir);
}

void DiffuseAreaLight::Pdf_Le(const Interaction &intr, Vector3f &w,
                              Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);
    CHECK_NE(intr.n, Normal3f(0, 0, 0));
    if (worldFromLight.IsAnimated()) {
        Interaction lightIntr = worldFromLight.ApplyInverse(intr);
        *pdfPos = shape.PDF(lightIntr);
        *pdfDir = twoSided ? (.5 * CosineHemispherePDF(AbsDot(lightIntr.n, w)))
                           : CosineHemispherePDF(Dot(lightIntr.n, w));
    } else {
        *pdfPos = shape.PDF(intr);
        *pdfDir = twoSided ? (.5 * CosineHemispherePDF(AbsDot(intr.n, w)))
                           : CosineHemispherePDF(Dot(intr.n, w));
    }
}

LightBounds DiffuseAreaLight::Bounds() const {
    Float phi = 0;
    if (image) {
        // Assume no distortion in the mapping, FWIW...
        for (int y = 0; y < image->Resolution().y; ++y)
            for (int x = 0; x < image->Resolution().x; ++x)
                for (int c = 0; c < 3; ++c)
                    phi += image->GetChannel({x, y}, c);
        phi /= 3 * image->Resolution().x * image->Resolution().y;
    }
    else
        phi = Lemit.MaxValue();

    phi *= (twoSided ? 2 : 1) * area * Pi;

    // TODO: for animated shapes, we probably need to worry about
    // worldFromLight as in Sample_Li().
    DirectionCone nb = shape.NormalBounds();
    return LightBounds(shape.WorldBound(), nb.w, phi, SafeACos(nb.cosTheta),
                       Pi / 2, twoSided);
}

std::string DiffuseAreaLight::ToString() const {
    return StringPrintf("[ DiffuseAreaLight %s Lemit: %s scale: %f shape: %s "
                        "twoSided: %s area: %f image: %s ]",
                        BaseToString(), Lemit,
                        scale, shape, twoSided ? "true" : "false", area, image);
}

DiffuseAreaLight *DiffuseAreaLight::Create(
    const AnimatedTransform &worldFromLight, const Medium *medium,
    const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
    Allocator alloc, const ShapeHandle shape) {
    SpectrumHandle L = dict.GetOneSpectrum("L", nullptr,
                                            SpectrumType::General, alloc);
    Float scale = dict.GetOneFloat("scale", 1);
    bool twoSided = dict.GetOneBool("twosided", false);

    std::string filename = ResolveFilename(dict.GetOneString("imagefile", ""));
    pstd::optional<Image> image;
    const RGBColorSpace *imageColorSpace = nullptr;
    if (!filename.empty()) {
        if (L != nullptr)
            ErrorExit("Both \"L\" and \"imagefile\" specified for DiffuseAreaLight.");
        auto im = Image::Read(filename, alloc);
        CHECK(im);

        pstd::optional<ImageChannelDesc> channelDesc =
            im->image.GetChannelDesc({ "R", "G", "B" });
        if (!channelDesc)
            ErrorExit("%s: Image provided to \"diffuse\" area light must have R, G, and B channels.",
                      filename);
        image = im->image.SelectChannels(*channelDesc, alloc);

        imageColorSpace = im->metadata.GetColorSpace();
    } else if (L == nullptr)
        L = &colorSpace->illuminant;

    return alloc.new_object<DiffuseAreaLight>(worldFromLight, medium, L, scale,
                                              shape, std::move(image), imageColorSpace,
                                              twoSided, alloc);
}

// UniformInfiniteLight Method Definitions
UniformInfiniteLight::UniformInfiniteLight(const AnimatedTransform &worldFromLight,
                                           SpectrumHandle L, Allocator alloc)
    : Light(LightType::Infinite, worldFromLight, MediumInterface()),
      L(L, alloc) { }

SampledSpectrum UniformInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // TODO: is there another Pi or so for the hemisphere?
    // pi r^2 for disk
    // 2pi for cosine-weighted sphere
    return 2 * Pi * Pi * worldRadius * worldRadius * L.Sample(lambda);
}

SampledSpectrum UniformInfiniteLight::Le(const Ray &ray,
                                         const SampledWavelengths &lambda) const {
    return L.Sample(lambda);
}

pstd::optional<LightLiSample> UniformInfiniteLight::Sample_Li(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda) const {
    ProfilerScope _(ProfilePhase::LightSample);

    Vector3f wi = SampleUniformSphere(u);
    Float pdf = UniformSpherePDF();
    return LightLiSample(this, L.Sample(lambda), wi, pdf, ref,
                         Interaction(ref.p() + wi * (2 * worldRadius),
                                     ref.time, &mediumInterface));
}

Float UniformInfiniteLight::Pdf_Li(const Interaction &ref, const Vector3f &w) const {
    ProfilerScope _(ProfilePhase::LightPDF);

    return UniformSpherePDF();
}

pstd::optional<LightLeSample> UniformInfiniteLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);

    Vector3f w = SampleUniformSphere(u1);

    Vector3f v1, v2;
    CoordinateSystem(-w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    Ray ray(pDisk + worldRadius * -w, w, time);

    Float pdfPos = 1 / (Pi * worldRadius * worldRadius);
    Float pdfDir = UniformSpherePDF();

    return LightLeSample(L.Sample(lambda), ray, pdfPos, pdfDir);
}

void UniformInfiniteLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);

    *pdfDir = UniformSpherePDF();
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

std::string UniformInfiniteLight::ToString() const {
    return StringPrintf("[ UniformInfiniteLight %s L: %s ]", BaseToString(), L);
}

// ImageInfiniteLight Method Definitions
ImageInfiniteLight::ImageInfiniteLight(const AnimatedTransform &worldFromLight,
                                       Image im, const RGBColorSpace *imageColorSpace, Float scale,
                                       const std::string &imageFile, Allocator alloc)
    : Light(LightType::Infinite, worldFromLight, MediumInterface()),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      imageFile(imageFile),
      wrapMode(WrapMode::OctahedralSphere, WrapMode::OctahedralSphere),
      distribution(alloc) {
    // Initialize sampling PDFs for infinite area light
    pstd::optional<ImageChannelDesc> channelDesc = image.GetChannelDesc({ "R", "G", "B" });
    if (!channelDesc)
        ErrorExit("%s: image used for ImageInfiniteLight doesn't have R, G, B channels.",
                  imageFile);
    CHECK_EQ(3, channelDesc->size());
    CHECK(channelDesc->IsIdentity());

    if (image.Resolution().x != image.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely this is an "
                  "equirect environment map.", imageFile, image.Resolution().x,
                  image.Resolution().y);

    Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
    Array2D<Float> d = image.ComputeSamplingDistribution(image.AllChannelsDesc(), 1,
                                                         domain, Norm::L2, wrapMode);
    distribution = Distribution2D(d, domain, alloc);
#if 0
    const char *base = getenv("BASE");
    int xs = atoi(getenv("XS")), ys = atoi(getenv("YS"));
    {
    Image im(PixelFormat::U256, {256,256}, { "R", "G", "B" }, ColorEncoding::sRGB);
    for (int y = 0; y < ys; ++y)
        for (int x = 0; x < xs; ++x) {
            int tile = x + xs * y;
            Float rgb[3] = { RadicalInverse(0, tile), RadicalInverse(1, tile),
                             RadicalInverse(2, tile) };
            for (int i = 0; i < 256/ys; ++i)
                for (int j = 0; j < 256/xs; ++j)
                    im.SetChannels({xs*x+j, ys*y+i}, {rgb[0], rgb[1], rgb[2]});
        }
    im.Write(StringPrintf("original-strata-%d-%d.png", xs, ys));
    }

    WarpedStrataVisualization(distribution, xs, ys).Write(StringPrintf("%s-2d-%d-%d.exr", base, xs, ys));
    Hierarchical2DWarp hw(d, domain);
    WarpedStrataVisualization(hw, xs, ys).Write(StringPrintf("%s-hier-%d-%d.exr", base, xs, ys));
    ErrorExit("wrote yo viz");
#endif

}

SampledSpectrum ImageInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    SampledSpectrum sumL(0.);

    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        for (int u = 0; u < width; ++u) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({u, v}, c, wrapMode);
            sumL += RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
        }
    }
    // Integrating over the sphere, so 4pi for that.  Then one more for Pi
    // r^2 for the area of the disk receiving illumination...
    return 4 * Pi * Pi * worldRadius * worldRadius *
        scale * sumL / (width * height);
}

Float ImageInfiniteLight::Pdf_Li(const Interaction &ref, const Vector3f &w) const {
    ProfilerScope _(ProfilePhase::LightPDF);

    Vector3f wl = worldFromLight.ApplyInverse(w, ref.time);
    return distribution.ContinuousPDF(EquiAreaSphereToSquare(wl)) / (4 * Pi);
}

pstd::optional<LightLeSample> ImageInfiniteLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);

    // Compute direction for infinite light sample ray

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPDF;
    Point2f uv = distribution.SampleContinuous(u1, &mapPDF);
    Vector3f wl = EquiAreaSquareToSphere(uv);
    Vector3f w = -worldFromLight(wl, time);

    // Compute origin for infinite light sample ray
    Vector3f v1, v2;
    CoordinateSystem(-w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = worldCenter + worldRadius * (cd.x * v1 + cd.y * v2);
    Ray ray(pDisk + worldRadius * -w, w, time);

    // Compute _ImageInfiniteLight_ ray PDFs
    Float pdfDir = mapPDF / (4 * Pi);
    Float pdfPos = 1 / (Pi * worldRadius * worldRadius);
    SampledSpectrum L = scale * bilerpSpectrum(uv, lambda);
    return LightLeSample(L, ray, pdfPos, pdfDir);
}

void ImageInfiniteLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);

    Vector3f wl = -worldFromLight.ApplyInverse(ray.d, ray.time);
    Float mapPDF = distribution.ContinuousPDF(EquiAreaSphereToSquare(wl));
    *pdfDir = mapPDF / (4 * Pi);
    *pdfPos = 1 / (Pi * worldRadius * worldRadius);
}

std::string ImageInfiniteLight::ToString() const {
    return StringPrintf("[ ImageInfiniteLight %s imagefile:%s scale: %f ]",
                        BaseToString(), imageFile, scale);
}

// SpotLight Method Definitions
SpotLight::SpotLight(const AnimatedTransform &worldFromLight,
                     const MediumInterface &mediumInterface,
                     SpectrumHandle I, Float totalWidth, Float falloffStart,
                     Allocator alloc)
    : Light(LightType::DeltaPosition, worldFromLight, mediumInterface),
      I(I, alloc),
      cosFalloffEnd(std::cos(Radians(totalWidth))),
      cosFalloffStart(std::cos(Radians(falloffStart))) {
    CHECK_LE(falloffStart, totalWidth);
}

pstd::optional<LightLiSample> SpotLight::Sample_Li(const Interaction &ref, const Point2f &u,
                                                   const SampledWavelengths &lambda) const {
    ProfilerScope _(ProfilePhase::LightSample);
    Point3f p = worldFromLight(Point3f(0, 0, 0), ref.time);
    Vector3f wi = Normalize(p - ref.p());
    Vector3f wl = Normalize(worldFromLight.ApplyInverse(-wi, ref.time));
    SampledSpectrum L = I.Sample(lambda) * Falloff(wl) / DistanceSquared(p, ref.p());
    if (!L) return {};
    return LightLiSample(this, L, wi, 1, ref, Interaction(p, ref.time, &mediumInterface));
}

Float SpotLight::Falloff(const Vector3f &wl) const {
    Float cosTheta = CosTheta(wl);
    if (cosTheta >= cosFalloffStart) return 1;
    // Compute falloff inside spotlight cone
    return SmoothStep(cosTheta, cosFalloffEnd, cosFalloffStart);
}

SampledSpectrum SpotLight::Phi(const SampledWavelengths &lambda) const {
    // int_0^start sin theta dtheta = 1 - cosFalloffStart
    // See notes/sample-spotlight.nb for the falloff part:
    // int_start^end smoothstep(cost, end, start) sin theta dtheta =
    //  (cosStart - cosEnd) / 2
    return I.Sample(lambda) * 2 * Pi * ((1 - cosFalloffStart) +
                                         (cosFalloffStart - cosFalloffEnd) / 2);
}

Float SpotLight::Pdf_Li(const Interaction &, const Vector3f &) const {
    return 0.f;
}

pstd::optional<LightLeSample> SpotLight::Sample_Le(
   const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
   Float time) const {
    ProfilerScope _(ProfilePhase::LightSample);
    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};
    Float sectionPDF;
    Vector3f wl;
    int section = SampleDiscrete(p, u2[0], &sectionPDF);
    Float pdfDir;
    if (section == 0) {
        // Sample center cone
        wl = SampleUniformCone(u1, cosFalloffStart);
        pdfDir = sectionPDF * UniformConePDF(cosFalloffStart);
    } else {
        DCHECK_EQ(1, section);

        Float cosTheta = SampleSmoothStep(u1[0], cosFalloffEnd,
                                          cosFalloffStart);
        CHECK(cosTheta >= cosFalloffEnd && cosTheta <= cosFalloffStart);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Float phi = u1[1] * 2 * Pi;
        wl = SphericalDirection(sinTheta, cosTheta, phi);
        pdfDir = sectionPDF * SmoothStepPDF(cosTheta, cosFalloffEnd,
                                            cosFalloffStart) / (2 * Pi);
    }

    Ray ray = worldFromLight(Ray(Point3f(0, 0, 0), wl, time,
                                 mediumInterface.outside));
    return LightLeSample(I.Sample(lambda) * Falloff(wl), ray, 1, pdfDir);
}

void SpotLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    ProfilerScope _(ProfilePhase::LightPDF);
    *pdfPos = 0;

    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};

    Float cosTheta = CosTheta(worldFromLight.ApplyInverse(ray.d, ray.time));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePDF(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) /
            (2 * Pi) * (p[1] / (p[0] + p[1]));
}

LightBounds SpotLight::Bounds() const {
    Point3f p = worldFromLight(Point3f(0, 0, 0), 0 /* TODO: time? */);
    Vector3f w = Normalize(worldFromLight(Vector3f(0, 0, 1), 0 /* TODO: time */));
    // As in Phi()
#if 0
    Float phi = I.MaxValue() * 2 * Pi * ((1 - cosFalloffStart) +
                                          (cosFalloffStart - cosFalloffEnd) / 2);
#else
    // cf. room-subsurf-from-kd.pbrt test: we sorta kinda actually want to
    // compute power as if it was an isotropic light source; the
    // LightBounds geometric terms give zero importance outside the spot
    // light's cone, so inside the cone, it doesn't matter if the overall
    // power is low; it's more accurate to effectively treat it as a point
    // light source.
    Float phi = I.MaxValue() * 4 * Pi;
#endif

    return LightBounds(p, w, phi, 0.f, std::acos(cosFalloffEnd), false);
}

std::string SpotLight::ToString() const {
    return StringPrintf("[ SpotLight %s I: %s cosFalloffStart: %f cosFalloffEnd: %f ]",
                        BaseToString(), I, cosFalloffStart, cosFalloffEnd);
}

SpotLight *SpotLight::Create(
    const AnimatedTransform &worldFromLight, const Medium *medium,
    const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
    Allocator alloc) {
    SpectrumHandle I = dict.GetOneSpectrum("I", &colorSpace->illuminant,
                                           SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        I = alloc.new_object<ScaledSpectrum>(sc, I);

    Float coneangle = dict.GetOneFloat("coneangle", 30.);
    Float conedelta = dict.GetOneFloat("conedeltaangle", 5.);
    // Compute spotlight world to light transformation
    Point3f from = dict.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = dict.GetOnePoint3f("to", Point3f(0, 0, 1));

    Transform dirToZ = (Transform)Frame::FromZ(Normalize(to - from));
    Transform t = Translate(Vector3f(from.x, from.y, from.z)) * Inverse(dirToZ);
    AnimatedTransform worldFromLightAnim(alloc.new_object<Transform>(*worldFromLight.startTransform * t),
                                         worldFromLight.startTime,
                                         alloc.new_object<Transform>(*worldFromLight.endTransform * t),
                                         worldFromLight.endTime);

    return alloc.new_object<SpotLight>(worldFromLightAnim, medium, I, coneangle,
                                       coneangle - conedelta, alloc);
}

SampledSpectrum LightHandle::Phi(const SampledWavelengths &lambda) const {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Phi(lambda);
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->Phi(lambda);
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Phi(lambda);
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Phi(lambda);
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Phi(lambda);
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Phi(lambda);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Phi(lambda);
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Phi(lambda);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

void LightHandle::Preprocess(const Bounds3f &worldBounds) {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Preprocess(worldBounds);
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->Preprocess(worldBounds);
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Preprocess(worldBounds);
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Preprocess(worldBounds);
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Preprocess(worldBounds);
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Preprocess(worldBounds);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Preprocess(worldBounds);
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Preprocess(worldBounds);
    default:
        LOG_FATAL("Unhandled light type");
    }
}

pstd::optional<LightLeSample> LightHandle::Sample_Le(const Point2f &u1, const Point2f &u2,
                                                     const SampledWavelengths &lambda,
                                                     Float time) const {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Sample_Le(u1, u2, lambda, time);
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Sample_Le(u1, u2, lambda, time);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

void LightHandle::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    // Note shouldn't be called for area lights..
    CHECK(!(((const Light *)ptr())->type != LightType::Area));

    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Pdf_Le(ray, pdfPos, pdfDir);
    default:
        LOG_FATAL("Unhandled light type");
    }
}

pstd::optional<LightBounds> LightHandle::Bounds() const {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Bounds();
    case TypeIndex<DistantLight>():
        return {};
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Bounds();
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Bounds();
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Bounds();
    case TypeIndex<UniformInfiniteLight>():
        return {};
    case TypeIndex<ImageInfiniteLight>():
        return {};
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Bounds();
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

std::string LightHandle::ToString() const {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->ToString();
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->ToString();
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->ToString();
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->ToString();
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->ToString();
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->ToString();
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->ToString();
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->ToString();
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

void LightHandle::Pdf_Le(const Interaction &intr, Vector3f &w, Float *pdfPos,
                         Float *pdfDir) const {
    // AreaLights only
    CHECK(((const Light *)ptr())->type == LightType::Area);

    switch (Tag()) {
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Pdf_Le(intr, w, pdfPos, pdfDir);
    default:
        LOG_FATAL("Unhandled light type");
    }
}

LightHandle LightHandle::Create(const std::string &name,
                                const ParameterDictionary &dict,
                                const AnimatedTransform &worldFromLight,
                                const Medium *outsideMedium, FileLoc loc,
                                Allocator alloc) {
    LightHandle light = nullptr;
    if (name == "point")
        light = PointLight::Create(worldFromLight, outsideMedium, dict,
                                   dict.ColorSpace(), alloc);
    else if (name == "spot")
        light = SpotLight::Create(worldFromLight, outsideMedium, dict,
                                  dict.ColorSpace(), alloc);
    else if (name == "goniometric")
        light = GoniometricLight::Create(worldFromLight, outsideMedium,
                                         dict, dict.ColorSpace(), alloc);
    else if (name == "projection")
        light = ProjectionLight::Create(worldFromLight, outsideMedium,
                                        dict, alloc);
    else if (name == "distant")
        light = DistantLight::Create(worldFromLight, dict, dict.ColorSpace(),
                                     alloc);
    else if (name == "infinite") {
        const RGBColorSpace *colorSpace = dict.ColorSpace();
        std::vector<SpectrumHandle > L = dict.GetSpectrumArray("L", SpectrumType::General, alloc);
        Float scale = dict.GetOneFloat("scale", 1);

        std::string filename = ResolveFilename(dict.GetOneString("imagefile", ""));

        if (L.empty() && filename.empty())
            // Default: color space's std illuminant
            return alloc.new_object<UniformInfiniteLight>(worldFromLight, &colorSpace->illuminant,
                                                          alloc);

        if (!L.empty()) {
            if (!filename.empty())
                ErrorExit(&loc, "Can't specify both emission \"L\" and \"imagefile\" with InfiniteAreaLight");

            if (scale != 1) {
                SpectrumHandle Ls = alloc.new_object<ScaledSpectrum>(scale, L[0]);
                return alloc.new_object<UniformInfiniteLight>(worldFromLight, Ls, alloc);
            }
            return alloc.new_object<UniformInfiniteLight>(worldFromLight, L[0], alloc);
        } else {
            pstd::optional<ImageAndMetadata> imageAndMetadata =
                Image::Read(filename, alloc);
            if (!imageAndMetadata)
                return nullptr;
            const RGBColorSpace *colorSpace = imageAndMetadata->metadata.GetColorSpace();

            pstd::optional<ImageChannelDesc> channelDesc =
                imageAndMetadata->image.GetChannelDesc({ "R", "G", "B" });
            if (!channelDesc)
                ErrorExit(&loc, "%s: image provided to \"infinite\" light must have R, G, and B channels.",
                          filename);
            Image image = imageAndMetadata->image.SelectChannels(*channelDesc, alloc);

            return alloc.new_object<ImageInfiniteLight>(worldFromLight, std::move(image),
                                                        colorSpace, scale, filename, alloc);
        }
    }
    else
        ErrorExit(&loc, "%s: light type unknown.", name);

    if (!light)
        ErrorExit(&loc, "%s: unable to create light.", name);

    dict.ReportUnused();
    return light;
}

LightHandle LightHandle::CreateArea(const std::string &name,
                                    const ParameterDictionary &dict,
                                    const AnimatedTransform &worldFromLight,
                                    const MediumInterface &mediumInterface,
                                    const ShapeHandle shape, FileLoc loc,
                                    Allocator alloc) {
    LightHandle area = nullptr;
    if (name == "diffuse")
        area = DiffuseAreaLight::Create(worldFromLight, mediumInterface.outside,
                                        dict, dict.ColorSpace(), alloc, shape);
    else
        ErrorExit(&loc, "%s: area light type unknown.", name);

    if (!area)
        ErrorExit(&loc, "%s: unable to create area light.", name);

    dict.ReportUnused();
    return area;
}

}  // namespace pbrt
