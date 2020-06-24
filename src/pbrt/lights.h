
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

#ifndef PBRT_LIGHTS_POINT_H
#define PBRT_LIGHTS_POINT_H

// lights/point.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/primitive.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/transform.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <memory>

namespace pbrt {

// PointLight Declarations
class PointLight : public Light {
  public:
    // PointLight Public Methods
    PointLight(const AnimatedTransform &worldFromLight,
               const MediumInterface &mediumInterface,
               SpectrumHandle I, Allocator alloc)
        : Light(LightType::DeltaPosition, worldFromLight, mediumInterface),
          I(I, alloc) {}

    static PointLight *Create(
        const AnimatedTransform &worldFromLight, const Medium *medium,
        const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
        Allocator alloc);

    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const {
        ProfilerScope _(ProfilePhase::LightSample);
        Point3f p = worldFromLight(Point3f(0, 0, 0), ref.time);
        Vector3f wi = Normalize(p - ref.p());
        return LightLiSample(this, I.Sample(lambda) / DistanceSquared(p, ref.p()),
                             wi, 1, ref, Interaction(p, ref.time, &mediumInterface));
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;
    void Preprocess(const Bounds3f &worldBounds) { }

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const {
        return 0;
    }

    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    LightBounds Bounds() const;

    std::string ToString() const;

  private:
    // PointLight Private Data
    DenselySampledSpectrum I;
};

// DistantLight Declarations
class DistantLight : public Light {
  public:
    // DistantLight Public Methods
    DistantLight(const AnimatedTransform &worldFromLight,
                 SpectrumHandle L, Allocator alloc);

    static DistantLight *Create(
        const AnimatedTransform &worldFromLight, const ParameterDictionary &dict,
        const RGBColorSpace *colorSpace, Allocator alloc);

    void Preprocess(const Bounds3f &worldBounds) {
        worldBounds.BoundingSphere(&worldCenter, &worldRadius);
    }
    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const {
        ProfilerScope _(ProfilePhase::LightSample);
        Vector3f wi = Normalize(worldFromLight(Vector3f(0, 0, 1), ref.time));
        Point3f pOutside = ref.p() + wi * (2 * worldRadius);
        return LightLiSample(this, L.Sample(lambda), wi, 1, ref,
                             Interaction(pOutside, ref.time, &mediumInterface));
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const {
        return 0;
    }

    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda, Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    std::string ToString() const;

  private:
    // DistantLight Private Data
    DenselySampledSpectrum L;
    Point3f worldCenter;
    Float worldRadius;
};

// ProjectionLight Declarations
class ProjectionLight : public Light {
  public:
    // ProjectionLight Public Methods
    ProjectionLight(const AnimatedTransform &worldFromLight,
                    const MediumInterface &medium,
                    Image image, const RGBColorSpace *colorSpace,
                    Float scale, Float fov, Allocator alloc);

    static ProjectionLight *Create(
        const AnimatedTransform &worldFromLight, const Medium *medium,
        const ParameterDictionary &dict, Allocator alloc);

    void Preprocess(const Bounds3f &worldBounds) { }

    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE
    SampledSpectrum Projection(const Vector3f &w,
                               const SampledWavelengths &lambda) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;
    LightBounds Bounds() const;

    std::string ToString() const;

  private:
    // ProjectionLight Private Data
    Image image;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    Transform ScreenFromLight, LightFromScreen;
    Float hither;
    Bounds2f screenBounds;
    Float cosTotalWidth;
    Float A;
    Distribution2D distrib;
};


// GoniometricLight Declarations
class GoniometricLight : public Light {
  public:
    // GoniometricLight Public Methods
    GoniometricLight(const AnimatedTransform &worldFromLight,
                     const MediumInterface &mediumInterface,
                     SpectrumHandle I, Image image, const RGBColorSpace *imageColorSpace,
                     Allocator alloc);

    static GoniometricLight *Create(
        const AnimatedTransform &worldFromLight, const Medium *medium,
        const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
        Allocator alloc);

    void Preprocess(const Bounds3f &worldBounds) { }

    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Scale(Vector3f wl, const SampledWavelengths &lambda) const {
        Float theta = SphericalTheta(wl), phi = SphericalPhi(wl);
        Point2f st(phi * Inv2Pi, theta * InvPi);
        return I.Sample(lambda) * image.BilerpChannel(st, 0, wrapMode);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    LightBounds Bounds() const;
    std::string ToString() const;

  private:
    // GoniometricLight Private Data
    DenselySampledSpectrum I;
    Image image;
    const RGBColorSpace *imageColorSpace;
    WrapMode2D wrapMode;
    Distribution2D distrib;
};


// DiffuseAreaLight Declarations
class DiffuseAreaLight : public Light {
  public:
    // DiffuseAreaLight Public Methods
    DiffuseAreaLight(const AnimatedTransform &worldFromLight,
                     const MediumInterface &mediumInterface,
                     SpectrumHandle Le, Float scale, const ShapeHandle shape,
                     pstd::optional<Image> image, const RGBColorSpace *imageColorSpace,
                     bool twoSided, Allocator alloc);

    static DiffuseAreaLight *Create(
        const AnimatedTransform &worldFromLight, const Medium *medium,
        const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
        Allocator alloc, const ShapeHandle shape);

    void Preprocess(const Bounds3f &worldBounds) { }

    PBRT_HOST_DEVICE
    SampledSpectrum L(const Interaction &intr, const Vector3f &w,
                      const SampledWavelengths &lambda) const {
        if (!twoSided && Dot(intr.n, w) < 0)
            return SampledSpectrum(0.f);

        if (image) {
            RGB rgb;
            for (int c = 0; c < 3; ++c)
                rgb[c] = image->BilerpChannel(intr.uv, c);
            return scale * RGBSpectrum(*imageColorSpace, rgb).Sample(lambda);
        } else
            return scale * Lemit.Sample(lambda);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const {
        ProfilerScope _(ProfilePhase::LightSample);
        pstd::optional<ShapeSample> ss;
        Float pdf;
        if (worldFromLight.IsAnimated()) {
            // Will need to discuss this thoroughly in the book.
            // TODO: is there some nice abstraction we could add or
            // some way to make this cleaner?
            //
            // The issue is that animated shapes are given an identity
            // transform and for them world to shape is handled via
            // AnimatedPrimitive. This is important for performance, since it
            // means we can build a BVH over collections shapes with the
            // animated transformations and in turn, only interpolate the
            // matrix and transform the ray to object space once, rather than
            // once per tested shape.
            //
            // However, this means that adjustments need to be made to calls to
            // Shape::Sample and Shape::Pdf, which expect the given reference
            // point to be in world space. However, as far as animated Shapes
            // are concerned, world space == their object space, since they
            // were given the identity for their transform. So we need to
            // transform the reference point to the shape's object space,
            // sample, and then transform the returned interaction back to
            // world space. Yuck.
            Interaction refLight = worldFromLight.ApplyInverse(ref);
            ss = shape.Sample(refLight, u);
            if (!ss) return {};
            ss->intr = worldFromLight(ss->intr);
        } else {
            ss = shape.Sample(ref, u);
            if (!ss) return {};
        }
        ss->intr.time = ref.time;
        ss->intr.mediumInterface = &mediumInterface;

        if (ss->pdf == 0 || LengthSquared(ss->intr.p() - ref.p()) == 0)
            return {};

        Vector3f wi = Normalize(ss->intr.p() - ref.p());
        SampledSpectrum Le = L(ss->intr, -wi, lambda);
        if (!Le) return {};

        return LightLiSample(this, Le, wi, ss->pdf, ref, ss->intr);
    }

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &ref, const Vector3f &wi) const {
        ProfilerScope _(ProfilePhase::LightPDF);
        if (worldFromLight.IsAnimated())
            return shape.PDF(worldFromLight.ApplyInverse(ref), wi);
        else
            return shape.PDF(ref, wi);
    }

    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Interaction &, Vector3f &w, Float *pdfPos,
                Float *pdfDir) const;

    LightBounds Bounds() const;

    std::string ToString() const;

  protected:
    // DiffuseAreaLight Protected Data
    DenselySampledSpectrum Lemit;
    Float scale;
    ShapeHandle shape;
    bool twoSided;
    Float area;
    const RGBColorSpace *imageColorSpace;
    pstd::optional<Image> image;
};


// UniformInfiniteLight Declarations
class UniformInfiniteLight : public Light {
  public:
    // UniformInfiniteLight Public Methods
    UniformInfiniteLight(const AnimatedTransform &worldFromLight,
                         SpectrumHandle L, Allocator alloc);

    void Preprocess(const Bounds3f &worldBounds) {
        worldBounds.BoundingSphere(&worldCenter, &worldRadius);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda, Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;
    std::string ToString() const;

  private:
    // UniformInfiniteLight Private Data
    DenselySampledSpectrum L;
    Point3f worldCenter;
    Float worldRadius;
};

class ImageInfiniteLight : public Light {
  public:
    // ImageInfiniteLight Public Methods
    ImageInfiniteLight(const AnimatedTransform &worldFromLight,
                       Image image, const RGBColorSpace *imageColorSpace, Float scale,
                       const std::string &imageFile, Allocator alloc);
    void Preprocess(const Bounds3f &worldBounds) {
        worldBounds.BoundingSphere(&worldCenter, &worldRadius);
    }

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const {
        Vector3f wl = Normalize(worldFromLight.ApplyInverse(ray.d, ray.time));
        Point2f st = EquiAreaSphereToSquare(wl);
        return scale * bilerpSpectrum(st, lambda);
    }

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const {
        ProfilerScope _(ProfilePhase::LightSample);

        // Find $(u,v)$ sample coordinates in infinite light texture
        Float mapPDF;
        Point2f uv = distribution.SampleContinuous(u, &mapPDF);
        if (mapPDF == 0) return {};

        // Convert infinite light sample point to direction
        Vector3f wl = EquiAreaSquareToSphere(uv);
        Vector3f wi = worldFromLight(wl, ref.time);

        // Compute PDF for sampled infinite light direction
        Float pdf = mapPDF / (4 * Pi);

        // Return radiance value for infinite light direction
        SampledSpectrum L = bilerpSpectrum(uv, lambda);

        L *= scale;

        return LightLiSample(this, L, wi, pdf, ref,
                             Interaction(ref.p() + wi * (2 * worldRadius),
                                         ref.time, &mediumInterface));
    }

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum bilerpSpectrum(Point2f st, const SampledWavelengths &lambda) const {
        RGB rgb;
        for (int c = 0; c < 3; ++c)
            rgb[c] = image.BilerpChannel(st, c, wrapMode);
        RGBSpectrum rgbs(*imageColorSpace, rgb);
        return rgbs.Sample(lambda);
    }

    // ImageInfiniteLight Private Data
    std::string imageFile;
    Image image;
    const RGBColorSpace *imageColorSpace;
    Float scale;
    WrapMode2D wrapMode;
    Point3f worldCenter;
    Float worldRadius;
    Distribution2D distribution;
};

// SpotLight Declarations
class SpotLight : public Light {
  public:
    // SpotLight Public Methods
    SpotLight(const AnimatedTransform &worldFromLight, const MediumInterface &m,
              SpectrumHandle I, Float totalWidth, Float falloffStart,
              Allocator alloc);

    static SpotLight *Create(
        const AnimatedTransform &worldFromLight, const Medium *medium,
        const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
        Allocator alloc);

    void Preprocess(const Bounds3f &worldBounds) { }

    PBRT_HOST_DEVICE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE
    Float Falloff(const Vector3f &w) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE
    Float Pdf_Li(const Interaction &, const Vector3f &) const;
    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &, Float *pdfPos, Float *pdfDir) const;

    LightBounds Bounds() const;

    std::string ToString() const;

  private:
    // SpotLight Private Data
    DenselySampledSpectrum I;
    Float cosFalloffStart, cosFalloffEnd;
};

inline pstd::optional<LightLiSample>
LightHandle::Sample_Li(const Interaction &ref, const Point2f &u,
                       const SampledWavelengths &lambda) const {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Sample_Li(ref, u, lambda);
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Sample_Li(ref, u, lambda);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

inline Float
LightHandle::Pdf_Li(const Interaction &ref, const Vector3f &wi) const {
    switch (Tag()) {
    case TypeIndex<PointLight>():
        return Cast<PointLight>()->Pdf_Li(ref, wi);
    case TypeIndex<DistantLight>():
        return Cast<DistantLight>()->Pdf_Li(ref, wi);
    case TypeIndex<ProjectionLight>():
        return Cast<ProjectionLight>()->Pdf_Li(ref, wi);
    case TypeIndex<GoniometricLight>():
        return Cast<GoniometricLight>()->Pdf_Li(ref, wi);
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->Pdf_Li(ref, wi);
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Pdf_Li(ref, wi);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Pdf_Li(ref, wi);
    case TypeIndex<SpotLight>():
        return Cast<SpotLight>()->Pdf_Li(ref, wi);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

inline SampledSpectrum
LightHandle::L(const Interaction &intr, const Vector3f &w,
               const SampledWavelengths &lambda) const {
    CHECK(((const Light *)ptr())->type == LightType::Area);
    switch (Tag()) {
    case TypeIndex<DiffuseAreaLight>():
        return Cast<DiffuseAreaLight>()->L(intr, w, lambda);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

inline SampledSpectrum
LightHandle::Le(const Ray &ray, const SampledWavelengths &lambda) const {
    CHECK(((const Light *)ptr())->type == LightType::Infinite);
    switch (Tag()) {
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Le(ray, lambda);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Le(ray, lambda);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_LIGHTS_SPOT_H
