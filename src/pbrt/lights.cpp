// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// lights/point.cpp*
#include <pbrt/lights.h>

#include <pbrt/cameras.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
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

LightBase::LightBase(LightType type, const AnimatedTransform &worldFromLight,
                     const MediumInterface &mediumInterface)
    : type(type), mediumInterface(mediumInterface), worldFromLight(worldFromLight) {
    ++numLights;
}

std::string LightBase::BaseToString() const {
    return StringPrintf("type: %s mediumInterface: %s worldFromLight: %s", type,
                        mediumInterface, worldFromLight);
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
    DirectionCone c =
        Union(DirectionCone(a.w, a.cosTheta_o), DirectionCone(b.w, b.cosTheta_o));
    Float theta_o = SafeACos(c.cosTheta);
    return LightBounds(Union(a.b, b.b), c.w, a.phi + b.phi, theta_o,
                       std::max(a.theta_e, b.theta_e), a.twoSided | b.twoSided);
}

// PointLight Method Definitions
SampledSpectrum PointLight::Phi(const SampledWavelengths &lambda) const {
    return 4 * Pi * I.Sample(lambda);
}

pstd::optional<LightLeSample> PointLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                                    const SampledWavelengths &lambda,
                                                    Float time) const {
    Point3f p = worldFromLight(Point3f(0, 0, 0), time);
    Ray ray(p, SampleUniformSphere(u1), time, mediumInterface.outside);
    return LightLeSample(I.Sample(lambda), ray, 1, UniformSpherePDF());
}

void PointLight::Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;
    *pdfDir = UniformSpherePDF();
}

LightBounds PointLight::Bounds() const {
    Point3f p = worldFromLight(Point3f(0, 0, 0), 0 /* TODO: time?? */);
    return LightBounds(p, Vector3f(0, 0, 1), 4 * Pi * I.MaxValue(), Pi, Pi / 2, false);
}

std::string PointLight::ToString() const {
    return StringPrintf("[ PointLight %s I: %s ]", BaseToString(), I);
}

PointLight *PointLight::Create(const AnimatedTransform &worldFromLight,
                               MediumHandle medium, const ParameterDictionary &dict,
                               const RGBColorSpace *colorSpace, const FileLoc *loc,
                               Allocator alloc) {
    SpectrumHandle I =
        dict.GetOneSpectrum("I", &colorSpace->illuminant, SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        I = alloc.new_object<ScaledSpectrum>(sc, I);

    Point3f from = dict.GetOnePoint3f("from", Point3f(0, 0, 0));
    Transform tf = Translate(Vector3f(from.x, from.y, from.z));
    AnimatedTransform worldFromLightAnim(
        worldFromLight.startTransform * tf, worldFromLight.startTime,
        worldFromLight.endTransform * tf, worldFromLight.endTime);

    return alloc.new_object<PointLight>(worldFromLightAnim, medium, I, alloc);
}

// DistantLight Method Definitions
DistantLight::DistantLight(const AnimatedTransform &worldFromLight, SpectrumHandle Lemit,
                           Allocator alloc)
    : LightBase(LightType::DeltaDirection, worldFromLight, MediumInterface()),
      Lemit(Lemit) {}

SampledSpectrum DistantLight::Phi(const SampledWavelengths &lambda) const {
    return Lemit.Sample(lambda) * Pi * sceneRadius * sceneRadius;
}

pstd::optional<LightLeSample> DistantLight::Sample_Le(const Point2f &u1,
                                                      const Point2f &u2,
                                                      const SampledWavelengths &lambda,
                                                      Float time) const {
    // Choose point on disk oriented toward infinite light direction
    Vector3f w = Normalize(worldFromLight(Vector3f(0, 0, 1), time));
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u1);
    Point3f pDisk = sceneCenter + sceneRadius * (cd.x * v1 + cd.y * v2);

    // Set ray origin and direction for infinite light ray
    Ray ray(pDisk + sceneRadius * w, -w, time);
    return LightLeSample(Lemit.Sample(lambda), ray,
                         1 / (Pi * sceneRadius * sceneRadius), 1);
}

void DistantLight::Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 1 / (Pi * sceneRadius * sceneRadius);
    *pdfDir = 0;
}

std::string DistantLight::ToString() const {
    return StringPrintf("[ DistantLight %s Lemit: %s ]", BaseToString(), Lemit);
}

DistantLight *DistantLight::Create(const AnimatedTransform &worldFromLight,
                                   const ParameterDictionary &dict,
                                   const RGBColorSpace *colorSpace, const FileLoc *loc,
                                   Allocator alloc) {
    SpectrumHandle L =
        dict.GetOneSpectrum("L", &colorSpace->illuminant, SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        L = alloc.new_object<ScaledSpectrum>(sc, L);

    Point3f from = dict.GetOnePoint3f("from", Point3f(0, 0, 0));
    Point3f to = dict.GetOnePoint3f("to", Point3f(0, 0, 1));

    Vector3f w = Normalize(from - to);
    Vector3f v1, v2;
    CoordinateSystem(w, &v1, &v2);
    Float m[4][4] = {v1.x, v2.x, w.x, 0, v1.y, v2.y, w.y, 0,
                     v1.z, v2.z, w.z, 0, 0,    0,    0,   1};
    Transform t(m);
    AnimatedTransform worldFromLightAnim(
        worldFromLight.startTransform * t, worldFromLight.startTime,
        worldFromLight.endTransform * t, worldFromLight.endTime);

    return alloc.new_object<DistantLight>(worldFromLightAnim, L, alloc);
}

STAT_MEMORY_COUNTER("Memory/Light image and distributions", imageBytes);

// ProjectionLight Method Definitions
ProjectionLight::ProjectionLight(const AnimatedTransform &worldFromLight,
                                 const MediumInterface &mediumInterface, Image im,
                                 const RGBColorSpace *imageColorSpace, Float scale,
                                 Float fov, Allocator alloc)
    : LightBase(LightType::DeltaPosition, worldFromLight, mediumInterface),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      distrib(alloc) {
    // Initialize _ProjectionLight_ projection matrix
    Float aspect = Float(image.Resolution().x) / Float(image.Resolution().y);
    if (aspect > 1)
        screenBounds = Bounds2f(Point2f(-aspect, -1), Point2f(aspect, 1));
    else
        screenBounds = Bounds2f(Point2f(-1, -1 / aspect), Point2f(1, 1 / aspect));
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

    pstd::optional<ImageChannelDesc> channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("Image used for ProjectionLight doesn't have R, G, B channels.");
    CHECK_EQ(3, channelDesc->size());
    CHECK(channelDesc->IsIdentity());

    auto dwdA = [&](const Point2f &p) {
        Vector3f w = Vector3f(LightFromScreen(Point3f(p.x, p.y, 0)));
        w = Normalize(w);
        return Pow<3>(w.z);
    };
    Array2D<Float> d = image.GetSamplingDistribution(dwdA, screenBounds);
    distrib = PiecewiseConstant2D(d, screenBounds);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

pstd::optional<LightLiSample> ProjectionLight::Sample_Li(const Interaction &ref,
                                                         const Point2f &u,
                                                         const SampledWavelengths &lambda,
                                                         LightSamplingMode mode) const {
    Point3f p = worldFromLight(Point3f(0, 0, 0), ref.time);
    Vector3f wi = Normalize(p - ref.p());
    Vector3f wl = worldFromLight.ApplyInverse(-wi, ref.time);
    return LightLiSample(this, Projection(wl, lambda) / DistanceSquared(p, ref.p()), wi,
                         1, ref, Interaction(p, ref.time, &mediumInterface));
}

// Takes wl already in light coordinate system!
SampledSpectrum ProjectionLight::Projection(const Vector3f &wl,
                                            const SampledWavelengths &lambda) const {
    // Discard directions behind projection light
    if (wl.z < hither)
        return SampledSpectrum(0.);

    // Project point onto projection plane and compute light
    Point3f ps = ScreenFromLight(Point3f(wl.x, wl.y, wl.z));
    if (!Inside(Point2f(ps.x, ps.y), screenBounds))
        return SampledSpectrum(0.f);

    Point2f st = Point2f(screenBounds.Offset(Point2f(ps.x, ps.y)));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(st, c);

    return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
}

SampledSpectrum ProjectionLight::Phi(const SampledWavelengths &lambda) const {
    SampledSpectrum sum(0.f);
    for (int v = 0; v < image.Resolution().y; ++v)
        for (int u = 0; u < image.Resolution().x; ++u) {
            Point2f ps = screenBounds.Lerp(
                {(u + .5f) / image.Resolution().x, (v + .5f) / image.Resolution().y});
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

Float ProjectionLight::Pdf_Li(const Interaction &, const Vector3f &,
                              LightSamplingMode mode) const {
    return 0.f;
}

pstd::optional<LightLeSample> ProjectionLight::Sample_Le(const Point2f &u1,
                                                         const Point2f &u2,
                                                         const SampledWavelengths &lambda,
                                                         Float time) const {
    Float pdf;
    Point2f ps = distrib.Sample(u1, &pdf);
    if (pdf == 0)
        return {};

    Vector3f w = Vector3f(LightFromScreen(Point3f(ps.x, ps.y, 0)));

    Ray ray = worldFromLight(
        Ray(Point3f(0, 0, 0), Normalize(w), time, mediumInterface.outside));
    Float cosTheta = CosTheta(Normalize(w));
    CHECK_GT(cosTheta, 0);
    Float pdfDir = pdf * screenBounds.Area() / (A * Pow<3>(cosTheta));

    Point2f p = Point2f(screenBounds.Offset(ps));
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(p, c);

    SampledSpectrum L = scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);

    return LightLeSample(L, ray, 1, pdfDir);
}

void ProjectionLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
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
    *pdfDir = distrib.PDF(Point2f(ps.x, ps.y)) * screenBounds.Area() / (A * Pow<3>(w.z));
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
            sum += std::max({image.GetChannel({u, v}, 0), image.GetChannel({u, v}, 1),
                             image.GetChannel({u, v}, 2)});
    Float phi = scale * sum / (image.Resolution().x * image.Resolution().y);
#endif
    Point3f p = worldFromLight(Point3f(0, 0, 0), 0.); /* TODO: handle animation */
    Vector3f w = Normalize(worldFromLight(Vector3f(0, 0, 1), 0.));
    return LightBounds(p, w, phi, 0.f, std::acos(cosTotalWidth), false);
}

std::string ProjectionLight::ToString() const {
    return StringPrintf("[ ProjectionLight %s scale: %f A: %f cosTotalWidth: %f ]",
                        BaseToString(), scale, A, cosTotalWidth);
}

ProjectionLight *ProjectionLight::Create(const AnimatedTransform &worldFromLight,
                                         MediumHandle medium,
                                         const ParameterDictionary &dict,
                                         const FileLoc *loc, Allocator alloc) {
    Float scale = dict.GetOneFloat("scale", 1);
    Float fov = dict.GetOneFloat("fov", 90.);

    std::string texname = ResolveFilename(dict.GetOneString("imagefile", ""));
    if (texname.empty())
        ErrorExit(loc, "Must provide \"imagefile\" to \"projection\" light source");

    pstd::optional<ImageAndMetadata> imageAndMetadata = Image::Read(texname, alloc);
    if (!imageAndMetadata)
        return nullptr;
    const RGBColorSpace *colorSpace = imageAndMetadata->metadata.GetColorSpace();

    pstd::optional<ImageChannelDesc> channelDesc =
        imageAndMetadata->image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit(loc, "Image provided to \"projection\" light must have R, G, "
                       "and B channels.");
    Image image = imageAndMetadata->image.SelectChannels(*channelDesc, alloc);

    Transform flip = Scale(1, -1, 1);
    AnimatedTransform worldFromLightFlipY(
        worldFromLight.startTransform * flip, worldFromLight.startTime,
        worldFromLight.endTransform * flip, worldFromLight.endTime);

    return alloc.new_object<ProjectionLight>(
        worldFromLightFlipY, medium, std::move(image), colorSpace, scale, fov, alloc);
}

// GoniometricLight Method Definitions
GoniometricLight::GoniometricLight(const AnimatedTransform &worldFromLight,
                                   const MediumInterface &mediumInterface,
                                   SpectrumHandle I, Image im,
                                   const RGBColorSpace *imageColorSpace, Allocator alloc)
    : LightBase(LightType::DeltaPosition, worldFromLight, mediumInterface),
      I(I),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      wrapMode(WrapMode::Repeat, WrapMode::Clamp),
      distrib(alloc) {
    CHECK_EQ(1, image.NChannels());

    Bounds2f domain(Point2f(0, 0), Point2f(2 * Pi, Pi));
    auto dpdA = [](const Point2f &p) { return std::sin(p.y); };
    Array2D<Float> d = image.GetSamplingDistribution(dpdA, domain);
    distrib = PiecewiseConstant2D(d, domain);

    imageBytes += image.BytesUsed() + distrib.BytesUsed();
}

pstd::optional<LightLiSample> GoniometricLight::Sample_Li(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda,
    LightSamplingMode mode) const {
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

Float GoniometricLight::Pdf_Li(const Interaction &, const Vector3f &,
                               LightSamplingMode mode) const {
    return 0.f;
}

pstd::optional<LightLeSample> GoniometricLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    Float pdf;
    Point2f uv = distrib.Sample(u1, &pdf);
    Float theta = uv[1], phi = uv[0];
    Float cosTheta = std::cos(theta), sinTheta = std::sin(theta);
    Vector3f wl = SphericalDirection(sinTheta, cosTheta, phi);
    Float pdfDir = sinTheta == 0 ? 0 : pdf / sinTheta;

    Ray ray = worldFromLight(Ray(Point3f(0, 0, 0), wl, time, mediumInterface.inside));
    return LightLeSample(Scale(wl, lambda), ray, 1, pdfDir);
}

void GoniometricLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0.f;

    Vector3f wl = Normalize(worldFromLight.ApplyInverse(ray.d, ray.time));
    Float theta = SphericalTheta(wl), phi = SphericalPhi(wl);
    *pdfDir = distrib.PDF(Point2f(phi, theta)) / std::sin(theta);
}

LightBounds GoniometricLight::Bounds() const {
    // Like Phi() method, but compute the weighted max component value of
    // the image map.
    Float weightedMaxImageSum = 0;
    int width = image.Resolution().x, height = image.Resolution().y;
    for (int v = 0; v < height; ++v) {
        Float sinTheta = std::sin(Pi * Float(v + .5f) / Float(height));
        for (int u = 0; u < width; ++u)
            weightedMaxImageSum +=
                sinTheta * image.GetChannels({u, v}, wrapMode).MaxValue();
    }
    Float phi = I.MaxValue() * 2 * Pi * Pi * weightedMaxImageSum / (width * height);

    Point3f p = worldFromLight(Point3f(0, 0, 0), 0 /* TODO: time?? */);
    // Bound it as an isotropic point light.
    return LightBounds(p, Vector3f(0, 0, 1), phi, Pi, Pi / 2, false);
}

std::string GoniometricLight::ToString() const {
    return StringPrintf("[ GoniometricLight %s I: %s ]", BaseToString(), I);
}

GoniometricLight *GoniometricLight::Create(const AnimatedTransform &worldFromLight,
                                           MediumHandle medium,
                                           const ParameterDictionary &dict,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc) {
    SpectrumHandle I =
        dict.GetOneSpectrum("I", &colorSpace->illuminant, SpectrumType::General, alloc);
    Float sc = dict.GetOneFloat("scale", 1);
    if (sc != 1)
        I = alloc.new_object<ScaledSpectrum>(sc, I);

    Image image(alloc);
    const RGBColorSpace *imageColorSpace = nullptr;

    std::string texname = ResolveFilename(dict.GetOneString("imagefile", ""));
    if (!texname.empty()) {
        pstd::optional<ImageAndMetadata> imageAndMetadata = Image::Read(texname, alloc);
        if (imageAndMetadata) {
            pstd::optional<ImageChannelDesc> rgbDesc =
                imageAndMetadata->image.GetChannelDesc({"R", "G", "B"});
            pstd::optional<ImageChannelDesc> yDesc =
                imageAndMetadata->image.GetChannelDesc({"Y"});

            imageColorSpace = imageAndMetadata->metadata.GetColorSpace();

            if (rgbDesc) {
                if (yDesc)
                    ErrorExit("%s: has both \"R\", \"G\", and \"B\" or \"Y\" "
                              "channels.",
                              texname);
                image = Image(imageAndMetadata->image.Format(),
                              imageAndMetadata->image.Resolution(), {"Y"},
                              imageAndMetadata->image.Encoding(), alloc);
                for (int y = 0; y < image.Resolution().y; ++y)
                    for (int x = 0; x < image.Resolution().x; ++x)
                        image.SetChannel(
                            {x, y}, 0,
                            imageAndMetadata->image.GetChannels({x, y}, *rgbDesc)
                                .Average());
            } else if (yDesc)
                image = imageAndMetadata->image;
            else
                ErrorExit(loc,
                          "%s: has neither \"R\", \"G\", and \"B\" or \"Y\" "
                          "channels.",
                          texname);
        } else {
            pstd::vector<float> one = {1.f};
            image = Image(std::move(one), {1, 1}, {"Y"});
        }
    }

    const Float swapYZ[4][4] = {1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1};
    Transform t(swapYZ);
    AnimatedTransform worldFromLightAnim(
        worldFromLight.startTransform * t, worldFromLight.startTime,
        worldFromLight.endTransform * t, worldFromLight.endTime);

    return alloc.new_object<GoniometricLight>(worldFromLightAnim, medium, I,
                                              std::move(image), imageColorSpace, alloc);
}

// DiffuseAreaLight Method Definitions
DiffuseAreaLight::DiffuseAreaLight(const AnimatedTransform &worldFromLight,
                                   const MediumInterface &mediumInterface,
                                   SpectrumHandle Le, Float scale,
                                   const ShapeHandle shape, pstd::optional<Image> im,
                                   const RGBColorSpace *imageColorSpace, bool twoSided,
                                   Allocator alloc)
    : LightBase(LightType::Area, worldFromLight, mediumInterface),
      Lemit(Le),
      scale(scale),
      shape(shape),
      twoSided(twoSided),
      area(shape.Area()),
      imageColorSpace(imageColorSpace),
      image(std::move(im)) {
    ++numAreaLights;

    if (image) {
        pstd::optional<ImageChannelDesc> desc = image->GetChannelDesc({"R", "G", "B"});
        if (!desc)
            ErrorExit("Image used for DiffuseAreaLight doesn't have R, G, B "
                      "channels.");
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
        Warning("Scaling detected in world to light transformation! "
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
    } else
        phi = Lemit.Sample(lambda);

    return phi * (twoSided ? 2 : 1) * scale * area * Pi;
}

pstd::optional<LightLeSample> DiffuseAreaLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    // Sample a point on the area light's _Shape_, _pShape_
    Float pdfDir;
    pstd::optional<ShapeSample> ss = shape.Sample(u1);
    if (!ss)
        return {};
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

    if (pdfDir == 0)
        return {};

    Frame nFrame = Frame::FromZ(ss->intr.n);
    w = nFrame.FromLocal(w);
    return LightLeSample(L(ss->intr, w, lambda), ss->intr.SpawnRay(w), ss->intr, ss->pdf,
                         pdfDir);
}

void DiffuseAreaLight::Pdf_Le(const Interaction &intr, Vector3f &w, Float *pdfPos,
                              Float *pdfDir) const {
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
    } else
        phi = Lemit.MaxValue();

    phi *= scale * (twoSided ? 2 : 1) * area * Pi;

    // TODO: for animated shapes, we probably need to worry about
    // worldFromLight as in Sample_Li().
    DirectionCone nb = shape.NormalBounds();
    return LightBounds(shape.Bounds(), nb.w, phi, SafeACos(nb.cosTheta), Pi / 2,
                       twoSided);
}

std::string DiffuseAreaLight::ToString() const {
    return StringPrintf("[ DiffuseAreaLight %s Lemit: %s scale: %f shape: %s "
                        "twoSided: %s area: %f image: %s ]",
                        BaseToString(), Lemit, scale, shape, twoSided ? "true" : "false",
                        area, image);
}

DiffuseAreaLight *DiffuseAreaLight::Create(const AnimatedTransform &worldFromLight,
                                           MediumHandle medium,
                                           const ParameterDictionary &dict,
                                           const RGBColorSpace *colorSpace,
                                           const FileLoc *loc, Allocator alloc,
                                           const ShapeHandle shape) {
    SpectrumHandle L = dict.GetOneSpectrum("L", nullptr, SpectrumType::General, alloc);
    Float scale = dict.GetOneFloat("scale", 1);
    bool twoSided = dict.GetOneBool("twosided", false);

    std::string filename = ResolveFilename(dict.GetOneString("imagefile", ""));
    pstd::optional<Image> image;
    const RGBColorSpace *imageColorSpace = nullptr;
    if (!filename.empty()) {
        if (L != nullptr)
            ErrorExit(loc,
                      "Both \"L\" and \"imagefile\" specified for DiffuseAreaLight.");
        auto im = Image::Read(filename, alloc);
        CHECK(im);

        pstd::optional<ImageChannelDesc> channelDesc =
            im->image.GetChannelDesc({"R", "G", "B"});
        if (!channelDesc)
            ErrorExit(loc,
                      "%s: Image provided to \"diffuse\" area light must have "
                      "R, G, and B channels.",
                      filename);
        image = im->image.SelectChannels(*channelDesc, alloc);

        imageColorSpace = im->metadata.GetColorSpace();
    } else if (L == nullptr)
        L = &colorSpace->illuminant;

    return alloc.new_object<DiffuseAreaLight>(worldFromLight, medium, L, scale, shape,
                                              std::move(image), imageColorSpace, twoSided,
                                              alloc);
}

// UniformInfiniteLight Method Definitions
UniformInfiniteLight::UniformInfiniteLight(const AnimatedTransform &worldFromLight,
                                           SpectrumHandle Lemit, Allocator alloc)
    : LightBase(LightType::Infinite, worldFromLight, MediumInterface()), Lemit(Lemit) {}

SampledSpectrum UniformInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // TODO: is there another Pi or so for the hemisphere?
    // pi r^2 for disk
    // 2pi for cosine-weighted sphere
    return 2 * Pi * Pi * Sqr(sceneRadius) * Lemit.Sample(lambda);
}

SampledSpectrum UniformInfiniteLight::Le(const Ray &ray,
                                         const SampledWavelengths &lambda) const {
    return Lemit.Sample(lambda);
}

pstd::optional<LightLiSample> UniformInfiniteLight::Sample_Li(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda,
    LightSamplingMode mode) const {
    Vector3f wi = SampleUniformSphere(u);
    Float pdf = UniformSpherePDF();
    return LightLiSample(
        this, Lemit.Sample(lambda), wi, pdf, ref,
        Interaction(ref.p() + wi * (2 * sceneRadius), ref.time, &mediumInterface));
}

Float UniformInfiniteLight::Pdf_Li(const Interaction &ref, const Vector3f &w,
                                   LightSamplingMode mode) const {
    return UniformSpherePDF();
}

pstd::optional<LightLeSample> UniformInfiniteLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    Vector3f w = SampleUniformSphere(u1);

    Vector3f v1, v2;
    CoordinateSystem(-w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * (cd.x * v1 + cd.y * v2);
    Ray ray(pDisk + sceneRadius * -w, w, time);

    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
    Float pdfDir = UniformSpherePDF();

    return LightLeSample(Lemit.Sample(lambda), ray, pdfPos, pdfDir);
}

void UniformInfiniteLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfDir = UniformSpherePDF();
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string UniformInfiniteLight::ToString() const {
    return StringPrintf("[ UniformInfiniteLight %s Lemit: %s ]", BaseToString(), Lemit);
}

// ImageInfiniteLight Method Definitions
ImageInfiniteLight::ImageInfiniteLight(const AnimatedTransform &worldFromLight, Image im,
                                       const RGBColorSpace *imageColorSpace, Float scale,
                                       const std::string &imageFile, Allocator alloc)
    : LightBase(LightType::Infinite, worldFromLight, MediumInterface()),
      image(std::move(im)),
      imageColorSpace(imageColorSpace),
      scale(scale),
      imageFile(imageFile),
      wrapMode(WrapMode::OctahedralSphere, WrapMode::OctahedralSphere),
      distribution(alloc),
      compensatedDistribution(alloc) {
    // Initialize sampling PDFs for infinite area light
    pstd::optional<ImageChannelDesc> channelDesc = image.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for ImageInfiniteLight doesn't have R, G, B "
                  "channels.",
                  imageFile);
    CHECK_EQ(3, channelDesc->size());
    CHECK(channelDesc->IsIdentity());

    if (image.Resolution().x != image.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an "
                  "equirect environment map.",
                  imageFile, image.Resolution().x, image.Resolution().y);

    Array2D<Float> d = image.GetSamplingDistribution();
    Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
    distribution = PiecewiseConstant2D(d, domain, alloc);

    // MIS compentasion
    Float average = std::accumulate(d.begin(), d.end(), 0.) / d.size();
    for (Float &v : d)
        v = std::max<Float>(v - average, std::min<Float>(.001f * average, v));
    compensatedDistribution = PiecewiseConstant2D(d, domain, alloc);

#if 0
    const char *base = getenv("BASE");
    int xs = atoi(getenv("XS")), ys = atoi(getenv("YS"));
    {
    Image im(PixelFormat::U256, {256,256}, { "R", "G", "B" }, ColorEncodingHandle::sRGB);
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
    return 4 * Pi * Pi * Sqr(sceneRadius) * scale * sumL / (width * height);
}

Float ImageInfiniteLight::Pdf_Li(const Interaction &ref, const Vector3f &w,
                                 LightSamplingMode mode) const {
    Vector3f wl = worldFromLight.ApplyInverse(w, ref.time);
    Float pdf = (mode == LightSamplingMode::WithMIS)
                    ? compensatedDistribution.PDF(EquiAreaSphereToSquare(wl))
                    : distribution.PDF(EquiAreaSphereToSquare(wl));
    return pdf / (4 * Pi);
}

pstd::optional<LightLeSample> ImageInfiniteLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    // Compute direction for infinite light sample ray

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPDF;
    Point2f uv = distribution.Sample(u1, &mapPDF);
    Vector3f wl = EquiAreaSquareToSphere(uv);
    Vector3f w = -worldFromLight(wl, time);

    // Compute origin for infinite light sample ray
    Vector3f v1, v2;
    CoordinateSystem(-w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * (cd.x * v1 + cd.y * v2);
    Ray ray(pDisk + sceneRadius * -w, w, time);

    // Compute _ImageInfiniteLight_ ray PDFs
    Float pdfDir = mapPDF / (4 * Pi);
    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
    SampledSpectrum L = scale * lookupSpectrum(uv, lambda);
    return LightLeSample(L, ray, pdfPos, pdfDir);
}

void ImageInfiniteLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    Vector3f wl = -worldFromLight.ApplyInverse(ray.d, ray.time);
    Float mapPDF = distribution.PDF(EquiAreaSphereToSquare(wl));
    *pdfDir = mapPDF / (4 * Pi);
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
}

std::string ImageInfiniteLight::ToString() const {
    return StringPrintf("[ ImageInfiniteLight %s imagefile:%s scale: %f ]",
                        BaseToString(), imageFile, scale);
}

PortalImageInfiniteLight::PortalImageInfiniteLight(
    const AnimatedTransform &worldFromLight, Image equiAreaImage,
    const RGBColorSpace *imageColorSpace, Float scale, const std::string &imageFile,
    std::vector<Point3f> p, Allocator alloc)
    : LightBase(LightType::Infinite, worldFromLight, MediumInterface()),
      image(alloc),
      imageColorSpace(imageColorSpace),
      scale(scale),
      imageFile(imageFile),
      distribution(alloc) {
    // Initialize sampling PDFs for infinite area light
    pstd::optional<ImageChannelDesc> channelDesc =
        equiAreaImage.GetChannelDesc({"R", "G", "B"});
    if (!channelDesc)
        ErrorExit("%s: image used for PortalImageInfiniteLight doesn't have R, "
                  "G, B channels.",
                  imageFile);
    CHECK_EQ(3, channelDesc->size());
    CHECK(channelDesc->IsIdentity());

    if (worldFromLight.IsAnimated())
        ErrorExit("Animated world-from-light transform is not supported with portal "
                  "infinite lights.");

    if (equiAreaImage.Resolution().x != equiAreaImage.Resolution().y)
        ErrorExit("%s: image resolution (%d, %d) is non-square. It's unlikely "
                  "this is an "
                  "equirect environment map.",
                  imageFile, equiAreaImage.Resolution().x, equiAreaImage.Resolution().y);

    if (p.size() != 4)
        ErrorExit("Expected 4 vertices for infinite light portal but given %d", p.size());
    for (int i = 0; i < 4; ++i)
        portal[i] = p[i];

    // Make sure the portal is a planar quad...
    // TODO: there's probably a more elegant way to do this.
    Vector3f p01 = Normalize(portal[1] - portal[0]);
    Vector3f p12 = Normalize(portal[2] - portal[1]);
    Vector3f p32 = Normalize(portal[2] - portal[3]);
    Vector3f p03 = Normalize(portal[3] - portal[0]);
    // Do opposite edges have the same direction?
    if (std::abs(Dot(p01, p32) - 1) > .001 || std::abs(Dot(p12, p03) - 1) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");
    // Sides perpendicular?
    if (std::abs(Dot(p01, p12)) > .001 || std::abs(Dot(p12, p32)) > .001 ||
        std::abs(Dot(p32, p03)) > .001 || std::abs(Dot(p03, p01)) > .001)
        Error("Infinite light portal isn't a planar quadrilateral");

    portalFrame = Frame::FromXY(p01, p03);

    // Resample the latlong map into rectified coordinates
    image = Image(PixelFormat::Float, equiAreaImage.Resolution(), {"R", "G", "B"},
                  equiAreaImage.Encoding(), alloc);
    ParallelFor(0, image.Resolution().y, [&](int y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            // [0,1]^2 image coordinates
            Point2f st((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);

            Vector3f w = ImageToWorld(st);

            w = Normalize(worldFromLight.ApplyInverse(w, 0 /* time */));

            WrapMode2D equiAreaWrap(WrapMode::OctahedralSphere,
                                    WrapMode::OctahedralSphere);
            Point2f stEqui = EquiAreaSphereToSquare(w);
            for (int c = 0; c < 3; ++c)
                image.SetChannel({x, y}, c,
                                 equiAreaImage.BilerpChannel(stEqui, c, equiAreaWrap));
        }
    });

    // Initialize sampling PDFs for infinite area light
    auto duvdw = [&](const Point2f &p) {
        Float duv_dw;
        (void)ImageToWorld(p, &duv_dw);
        return duv_dw;
    };
    Array2D<Float> d = image.GetSamplingDistribution(duvdw);
    distribution = SATPiecewiseConstant2D(d, alloc);
}

SampledSpectrum PortalImageInfiniteLight::Phi(const SampledWavelengths &lambda) const {
    // We're really computing fluence, then converting to power, for what
    // that's worth..
    SampledSpectrum sumL(0.);

    for (int y = 0; y < image.Resolution().y; ++y) {
        for (int x = 0; x < image.Resolution().x; ++x) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image.GetChannel({x, y}, c);

            Point2f st((x + 0.5f) / image.Resolution().x,
                       (y + 0.5f) / image.Resolution().y);
            Float duv_dw;
            (void)ImageToWorld(st, &duv_dw);

            sumL += RGBSpectrum(*imageColorSpace, rgb).Sample(lambda) / duv_dw;
        }
    }

    return scale * Area() * sumL / (image.Resolution().x * image.Resolution().y);
}

SampledSpectrum PortalImageInfiniteLight::Le(const Ray &ray,
                                             const SampledWavelengths &lambda) const {
    // Ignore world to light...
    Vector3f w = Normalize(ray.d);
    Point2f st = WorldToImage(w);

    if (!Inside(st, ImageBounds(ray.o)))
        return SampledSpectrum(0.f);

    return ImageLookup(st, lambda);
}

SampledSpectrum PortalImageInfiniteLight::ImageLookup(
    const Point2f &st, const SampledWavelengths &lambda) const {
    RGB rgb;
    for (int c = 0; c < 3; ++c)
        rgb[c] = image.LookupNearestChannel(st, c);

    return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
}

pstd::optional<LightLiSample> PortalImageInfiniteLight::Sample_Li(
    const Interaction &ref, const Point2f &u, const SampledWavelengths &lambda,
    LightSamplingMode mode) const {
    Bounds2f b = ImageBounds(ref.p());

    // Find $(u,v)$ sample coordinates in infinite light texture
    Float mapPDF;
    Point2f uv = distribution.Sample(u, b, &mapPDF);
    if (mapPDF == 0)
        return {};

    // Convert infinite light sample point to direction
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Vector3f wi = ImageToWorld(uv, &duv_dw);
    if (duv_dw == 0)
        return {};

    // Compute PDF for sampled infinite light direction
    Float pdf = mapPDF / duv_dw;
    CHECK(!std::isinf(pdf));

    SampledSpectrum L = ImageLookup(uv, lambda);

    return LightLiSample(
        this, L, wi, pdf, ref,
        Interaction(ref.p() + wi * (2 * sceneRadius), ref.time, &mediumInterface));
}

Float PortalImageInfiniteLight::Pdf_Li(const Interaction &ref, const Vector3f &w,
                                       LightSamplingMode mode) const {
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Point2f st = WorldToImage(w, &duv_dw);
    if (duv_dw == 0)
        return 0;

    Bounds2f b = ImageBounds(ref.p());
    Float pdf = distribution.PDF(st, b);
    return pdf / duv_dw;
}

pstd::optional<LightLeSample> PortalImageInfiniteLight::Sample_Le(
    const Point2f &u1, const Point2f &u2, const SampledWavelengths &lambda,
    Float time) const {
    Float mapPDF;
    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    Point2f uv = distribution.Sample(u1, b, &mapPDF);
    if (mapPDF == 0)
        return {};

    // Convert infinite light sample point to direction
    // Note: ignore WorldToLight since we already folded it in when we
    // resampled...
    Float duv_dw;
    Vector3f w = -ImageToWorld(uv, &duv_dw);
    if (duv_dw == 0)
        return {};

    // Compute PDF for sampled infinite light direction
    Float pdfDir = mapPDF / duv_dw;

#if 0
    // Just sample within the portal.
    // This works with the light path integrator, but not BDPT :-(
    Point3f p = portal[0] + u2[0] * (portal[1] - portal[0]) +
        u2[1] * (portal[3] - portal[0]);
    // Compute _PortalImageInfiniteLight_ ray PDFs
    Ray ray(p, w, time);

    // Cosine to account for projected area of portal w.r.t. ray direction.
    Normal3f n = Normal3f(portalFrame.z);
    Float pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    Vector3f v1, v2;
    CoordinateSystem(-w, &v1, &v2);
    Point2f cd = SampleUniformDiskConcentric(u2);
    Point3f pDisk = sceneCenter + sceneRadius * (cd.x * v1 + cd.y * v2);
    Ray ray(pDisk + sceneRadius * -w, w, time);

    Float pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    SampledSpectrum L = ImageLookup(uv, lambda);

    return LightLeSample(L, ray, pdfPos, pdfDir);
}

void PortalImageInfiniteLight::Pdf_Le(const Ray &ray, Float *pdfPos,
                                      Float *pdfDir) const {
    // TODO: negate here or???
    Vector3f w = -Normalize(ray.d);
    Float duv_dw;
    Point2f st = WorldToImage(w, &duv_dw);

    if (duv_dw == 0) {
        *pdfPos = *pdfDir = 0;
        return;
    }

    Bounds2f b(Point2f(0, 0), Point2f(1, 1));
    Float pdf = distribution.PDF(st, b);

#if 0
    Normal3f n = Normal3f(portalFrame.z);
    *pdfPos = 1 / (Area() * AbsDot(n, w));
#else
    *pdfPos = 1 / (Pi * Sqr(sceneRadius));
#endif

    *pdfDir = pdf / duv_dw;
}

std::string PortalImageInfiniteLight::ToString() const {
    return StringPrintf("[ PortalImageInfiniteLight %s imagefile:%s scale: %f portal: %s "
                        " portalFrame: %s ]",
                        BaseToString(), imageFile, scale, portal, portalFrame);
}

// SpotLight Method Definitions
SpotLight::SpotLight(const AnimatedTransform &worldFromLight,
                     const MediumInterface &mediumInterface, SpectrumHandle I,
                     Float totalWidth, Float falloffStart, Allocator alloc)
    : LightBase(LightType::DeltaPosition, worldFromLight, mediumInterface),
      I(I),
      cosFalloffEnd(std::cos(Radians(totalWidth))),
      cosFalloffStart(std::cos(Radians(falloffStart))) {
    CHECK_LE(falloffStart, totalWidth);
}

pstd::optional<LightLiSample> SpotLight::Sample_Li(const Interaction &ref,
                                                   const Point2f &u,
                                                   const SampledWavelengths &lambda,
                                                   LightSamplingMode mode) const {
    Point3f p = worldFromLight(Point3f(0, 0, 0), ref.time);
    Vector3f wi = Normalize(p - ref.p());
    Vector3f wl = Normalize(worldFromLight.ApplyInverse(-wi, ref.time));
    SampledSpectrum L = I.Sample(lambda) * Falloff(wl) / DistanceSquared(p, ref.p());
    if (!L)
        return {};
    return LightLiSample(this, L, wi, 1, ref, Interaction(p, ref.time, &mediumInterface));
}

Float SpotLight::Falloff(const Vector3f &wl) const {
    Float cosTheta = CosTheta(wl);
    if (cosTheta >= cosFalloffStart)
        return 1;
    // Compute falloff inside spotlight cone
    return SmoothStep(cosTheta, cosFalloffEnd, cosFalloffStart);
}

SampledSpectrum SpotLight::Phi(const SampledWavelengths &lambda) const {
    // int_0^start sin theta dtheta = 1 - cosFalloffStart
    // See notes/sample-spotlight.nb for the falloff part:
    // int_start^end smoothstep(cost, end, start) sin theta dtheta =
    //  (cosStart - cosEnd) / 2
    return I.Sample(lambda) * 2 * Pi *
           ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2);
}

Float SpotLight::Pdf_Li(const Interaction &, const Vector3f &,
                        LightSamplingMode mode) const {
    return 0.f;
}

pstd::optional<LightLeSample> SpotLight::Sample_Le(const Point2f &u1, const Point2f &u2,
                                                   const SampledWavelengths &lambda,
                                                   Float time) const {
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

        Float cosTheta = SampleSmoothStep(u1[0], cosFalloffEnd, cosFalloffStart);
        CHECK(cosTheta >= cosFalloffEnd && cosTheta <= cosFalloffStart);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Float phi = u1[1] * 2 * Pi;
        wl = SphericalDirection(sinTheta, cosTheta, phi);
        pdfDir = sectionPDF * SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) /
                 (2 * Pi);
    }

    Ray ray = worldFromLight(Ray(Point3f(0, 0, 0), wl, time, mediumInterface.outside));
    return LightLeSample(I.Sample(lambda) * Falloff(wl), ray, 1, pdfDir);
}

void SpotLight::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    *pdfPos = 0;

    // Unnormalized probabilities of sampling each part.
    Float p[2] = {1 - cosFalloffStart, (cosFalloffStart - cosFalloffEnd) / 2};

    Float cosTheta = CosTheta(worldFromLight.ApplyInverse(ray.d, ray.time));
    if (cosTheta >= cosFalloffStart)
        *pdfDir = UniformConePDF(cosFalloffStart) * p[0] / (p[0] + p[1]);
    else
        *pdfDir = SmoothStepPDF(cosTheta, cosFalloffEnd, cosFalloffStart) / (2 * Pi) *
                  (p[1] / (p[0] + p[1]));
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

SpotLight *SpotLight::Create(const AnimatedTransform &worldFromLight, MediumHandle medium,
                             const ParameterDictionary &dict,
                             const RGBColorSpace *colorSpace, const FileLoc *loc,
                             Allocator alloc) {
    SpectrumHandle I =
        dict.GetOneSpectrum("I", &colorSpace->illuminant, SpectrumType::General, alloc);
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
    AnimatedTransform worldFromLightAnim(
        worldFromLight.startTransform * t, worldFromLight.startTime,
        worldFromLight.endTransform * t, worldFromLight.endTime);

    return alloc.new_object<SpotLight>(worldFromLightAnim, medium, I, coneangle,
                                       coneangle - conedelta, alloc);
}

SampledSpectrum LightHandle::Phi(const SampledWavelengths &lambda) const {
    auto phi = [&](auto ptr) { return ptr->Phi(lambda); };
    return ApplyCPU<SampledSpectrum>(phi);
}

void LightHandle::Preprocess(const Bounds3f &sceneBounds) {
    auto preprocess = [&](auto ptr) { return ptr->Preprocess(sceneBounds); };
    return ApplyCPU<void>(preprocess);
}

pstd::optional<LightLeSample> LightHandle::Sample_Le(const Point2f &u1, const Point2f &u2,
                                                     const SampledWavelengths &lambda,
                                                     Float time) const {
    auto sample = [&](auto ptr) { return ptr->Sample_Le(u1, u2, lambda, time); };
    return Apply<pstd::optional<LightLeSample>>(sample);
}

void LightHandle::Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->Pdf_Le(ray, pdfPos, pdfDir); };
    return Apply<void>(pdf);
}

pstd::optional<LightBounds> LightHandle::Bounds() const {
    auto bounds = [](auto ptr) { return ptr->Bounds(); };
    return ApplyCPU<pstd::optional<LightBounds>>(bounds);
}

std::string LightHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto str = [](auto ptr) { return ptr->ToString(); };
    return ApplyCPU<std::string>(str);
}

void LightHandle::Pdf_Le(const Interaction &intr, Vector3f &w, Float *pdfPos,
                         Float *pdfDir) const {
    auto pdf = [&](auto ptr) { return ptr->Pdf_Le(intr, w, pdfPos, pdfDir); };
    return Apply<void>(pdf);
}

LightHandle LightHandle::Create(const std::string &name, const ParameterDictionary &dict,
                                const AnimatedTransform &worldFromLight,
                                const CameraTransform &cameraTransform,
                                MediumHandle outsideMedium, const FileLoc *loc,
                                Allocator alloc) {
    LightHandle light = nullptr;
    if (name == "point")
        light = PointLight::Create(worldFromLight, outsideMedium, dict, dict.ColorSpace(),
                                   loc, alloc);
    else if (name == "spot")
        light = SpotLight::Create(worldFromLight, outsideMedium, dict, dict.ColorSpace(),
                                  loc, alloc);
    else if (name == "goniometric")
        light = GoniometricLight::Create(worldFromLight, outsideMedium, dict,
                                         dict.ColorSpace(), loc, alloc);
    else if (name == "projection")
        light = ProjectionLight::Create(worldFromLight, outsideMedium, dict, loc, alloc);
    else if (name == "distant")
        light = DistantLight::Create(worldFromLight, dict, dict.ColorSpace(), loc, alloc);
    else if (name == "infinite") {
        const RGBColorSpace *colorSpace = dict.ColorSpace();
        std::vector<SpectrumHandle> L =
            dict.GetSpectrumArray("L", SpectrumType::General, alloc);
        Float scale = dict.GetOneFloat("scale", 1);
        std::vector<Point3f> portal = dict.GetPoint3fArray("portal");
        std::string filename = ResolveFilename(dict.GetOneString("imagefile", ""));

        if (L.empty() && filename.empty())
            // Default: color space's std illuminant
            light = alloc.new_object<UniformInfiniteLight>(
                worldFromLight, &colorSpace->illuminant, alloc);
        else if (!L.empty()) {
            if (!filename.empty())
                ErrorExit(loc, "Can't specify both emission \"L\" and "
                               "\"imagefile\" with InfiniteAreaLight");

            if (!portal.empty())
                ErrorExit(loc, "Portals are not supported for InfiniteAreaLights "
                               "without \"imagefile\".");
            if (scale != 1) {
                SpectrumHandle Ls = alloc.new_object<ScaledSpectrum>(scale, L[0]);
                light = alloc.new_object<UniformInfiniteLight>(worldFromLight, Ls, alloc);
            } else
                light =
                    alloc.new_object<UniformInfiniteLight>(worldFromLight, L[0], alloc);
        } else {
            pstd::optional<ImageAndMetadata> imageAndMetadata =
                Image::Read(filename, alloc);
            if (imageAndMetadata) {
                const RGBColorSpace *colorSpace =
                    imageAndMetadata->metadata.GetColorSpace();

                pstd::optional<ImageChannelDesc> channelDesc =
                    imageAndMetadata->image.GetChannelDesc({"R", "G", "B"});
                if (!channelDesc)
                    ErrorExit(loc,
                              "%s: image provided to \"infinite\" light must "
                              "have R, G, and B channels.",
                              filename);
                Image image = imageAndMetadata->image.SelectChannels(*channelDesc, alloc);

                if (!portal.empty()) {
                    for (Point3f &p : portal)
                        p = cameraTransform.RenderFromWorld(p);

                    light = alloc.new_object<PortalImageInfiniteLight>(
                        worldFromLight, std::move(image), colorSpace, scale, filename,
                        portal, alloc);
                } else
                    light = alloc.new_object<ImageInfiniteLight>(
                        worldFromLight, std::move(image), colorSpace, scale, filename,
                        alloc);
            }
        }
    } else
        ErrorExit(loc, "%s: light type unknown.", name);

    if (!light)
        ErrorExit(loc, "%s: unable to create light.", name);

    dict.ReportUnused();
    return light;
}

LightHandle LightHandle::CreateArea(const std::string &name,
                                    const ParameterDictionary &dict,
                                    const AnimatedTransform &worldFromLight,
                                    const MediumInterface &mediumInterface,
                                    const ShapeHandle shape, const FileLoc *loc,
                                    Allocator alloc) {
    LightHandle area = nullptr;
    if (name == "diffuse")
        area = DiffuseAreaLight::Create(worldFromLight, mediumInterface.outside, dict,
                                        dict.ColorSpace(), loc, alloc, shape);
    else
        ErrorExit(loc, "%s: area light type unknown.", name);

    if (!area)
        ErrorExit(loc, "%s: unable to create area light.", name);

    dict.ReportUnused();
    return area;
}

}  // namespace pbrt
