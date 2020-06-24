
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

class LightLiSample {
public:
    LightLiSample() = default;
    PBRT_HOST_DEVICE_INLINE
    LightLiSample(const Light *light, const SampledSpectrum &L, const Vector3f &wi,
                  Float pdf, const Interaction &pRef, const Interaction &pLight)
        : L(L), wi(wi), pdf(pdf), light(light), pRef(pRef), pLight(pLight) {}

    bool Unoccluded(const Scene &scene) const;
    SampledSpectrum Tr(const Scene &scene, const SampledWavelengths &lambda,
                       SamplerHandle sampler) const;

    const Light *light;
    SampledSpectrum L;
    Vector3f wi;
    Float pdf;
    Interaction pRef, pLight;
};

class LightLeSample {
public:
    LightLeSample() = default;
    PBRT_HOST_DEVICE_INLINE
    LightLeSample(const SampledSpectrum &L, const Ray &ray, Float pdfPos,
                  Float pdfDir)
        : L(L), ray(ray), pdfPos(pdfPos), pdfDir(pdfDir) {}
    PBRT_HOST_DEVICE_INLINE
    LightLeSample(const SampledSpectrum &L, const Ray &ray, const Interaction &intr,
                  Float pdfPos, Float pdfDir)
        : L(L), ray(ray), intr(intr), pdfPos(pdfPos), pdfDir(pdfDir) {
        CHECK(this->intr->n != Normal3f(0, 0, 0));
    }

    PBRT_HOST_DEVICE_INLINE
    Float AbsCosTheta(const Vector3f &w) const { return intr ? AbsDot(w, intr->n) : 1; }

    SampledSpectrum L;
    Ray ray;
    pstd::optional<Interaction> intr;
    Float pdfPos, pdfDir;
};

struct LightBounds {
    LightBounds() = default;
    LightBounds(const Bounds3f &b, const Vector3f &w, Float phi, Float theta_o,
                Float theta_e, bool twoSided)
      : b(b), w(Normalize(w)), phi(phi), theta_o(theta_o), theta_e(theta_e),
        cosTheta_o(std::cos(theta_o)), cosTheta_e(std::cos(theta_e)), twoSided(twoSided) {}
    LightBounds(const Point3f &p, const Vector3f &w, Float phi, Float theta_o,
                Float theta_e, bool twoSided)
      : b(p, p), w(Normalize(w)), phi(phi), theta_o(theta_o), theta_e(theta_e),
        cosTheta_o(std::cos(theta_o)), cosTheta_e(std::cos(theta_e)), twoSided(twoSided) {}

    // baseline: 38s in importance
    // acos hack: 34s
    // theta_u 0 if far away: 23s (!)
    PBRT_HOST_DEVICE
    Float Importance(const Interaction &intr) const {
        //ProfilerScope _(ProfilePhase::LightDistribImportance);

        Point3f pc = (b.pMin + b.pMax) / 2;
        Float d2 = DistanceSquared(intr.p(), pc);
        // Don't let d2 get too small if p is inside the bounds.
        d2 = std::max(d2, Length(b.Diagonal()) / 2);

        Vector3f wi = Normalize(intr.p() - pc);

#if 0
        Float cosTheta = Dot(w, wi);
        Float theta = SafeACos(cosTheta);
        if (twoSided && theta > Pi / 2) {
            theta = std::max<Float>(0, Pi - theta);
            cosTheta = std::abs(cosTheta);
        }
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);

        Float cosTheta_u = BoundSubtendedDirections(b, intr.p).cosTheta;
        Float theta_u = SafeACos(cosTheta_u);

        Float thetap = std::max<Float>(0, theta - theta_o - theta_u);

        if (thetap >= theta_e)
            return 0;

        Float imp = phi * std::cos(thetap) / d2;
        CHECK_GE(imp, -1e-3);

        if (intr.n != Normal3f(0,0,0)) {
            Float cosTheta_i = AbsDot(wi, intr.n);
            Float theta_i = SafeACos(cosTheta_i);
            Float thetap_i = std::max<Float>(theta_i - theta_u, 0);
            imp *= std::cos(thetap_i);
        }
#else
        Float cosTheta = Dot(w, wi);
        if (twoSided)
            cosTheta = std::abs(cosTheta);
        // FIXME? unstable when cosTheta \approx 1
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);

        // cos(max(0, a-b))
        auto cosSubClamped = [](Float sinThetaA, Float cosThetaA,
                                Float sinThetaB, Float cosThetaB) -> Float {
                                 if (cosThetaA > cosThetaB)
                                     // Handle the max(0, ...)
                                     return 1;
                                 return cosThetaA * cosThetaB + sinThetaA * sinThetaB;
                             };
        // sin(max(0, a-b))
        auto sinSubClamped = [](Float sinThetaA, Float cosThetaA,
                                Float sinThetaB, Float cosThetaB) -> Float {
                                 if (cosThetaA > cosThetaB)
                                     // Handle the max(0, ...)
                                     return 0;
                                 return sinThetaA * cosThetaB - cosThetaA * sinThetaB;
                             };

        Float cosTheta_u = BoundSubtendedDirections(b, intr.p()).cosTheta;
        Float sinTheta_u = SafeSqrt(1 - cosTheta_u * cosTheta_u);

        // Open issue: for a tri light that's axis aligned, we'd like to have
        // very low to zero importance for points in its plane. This doesn't
        // quite work out due to subtracting out the bounds' subtended angle.

        // cos(theta_p). Compute in two steps
        Float cosTheta_x = cosSubClamped(sinTheta, cosTheta,
                                         SafeSqrt(1 - cosTheta_o * cosTheta_o),
                                         cosTheta_o);
        Float sinTheta_x = sinSubClamped(sinTheta, cosTheta,
                                         SafeSqrt(1 - cosTheta_o * cosTheta_o),
                                         cosTheta_o);
        Float cosTheta_p = cosSubClamped(sinTheta_x, cosTheta_x,
                                         sinTheta_u, cosTheta_u);

        if (cosTheta_p <= cosTheta_e)
            return 0;

        Float imp = phi * cosTheta_p / d2;
        DCHECK_GE(imp, -1e-3);

        if (intr.n != Normal3f(0, 0, 0)) {
            // cos(thetap_i) = cos(max(0, theta_i - theta_u))
            // cos (a-b) = cos a cos b + sin a sin b
            Float cosTheta_i = AbsDot(wi, intr.n);
            Float sinTheta_i = SafeSqrt(1 - cosTheta_i * cosTheta_i);
            Float cosThetap_i = cosSubClamped(sinTheta_i, cosTheta_i,
                                              sinTheta_u, cosTheta_u);
            imp *= cosThetap_i;
        }
#endif

        return std::max<Float>(imp, 0);
    }



    std::string ToString() const;

    Bounds3f b;  // TODO: rename to |bounds|?
    Vector3f w;
    Float phi = 0;
    Float theta_o = 0, theta_e = 0;
    Float cosTheta_o = 1, cosTheta_e = 1;
    bool twoSided = false;
};

LightBounds Union(const LightBounds &a, const LightBounds &b);

// PointLight Declarations
class PointLight : public Light {
  public:
    // PointLight Public Methods
    PointLight(const AnimatedTransform &worldFromLight,
               const MediumInterface &mediumInterface,
               SpectrumHandle I, Allocator alloc)
        : Light(LightType::DeltaPosition, worldFromLight, mediumInterface),
          I(I) {}

    static PointLight *Create(
        const AnimatedTransform &worldFromLight, const Medium *medium,
        const ParameterDictionary &dict, const RGBColorSpace *colorSpace,
        const FileLoc *loc, Allocator alloc);

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
    SpectrumHandle I;
};

// DistantLight Declarations
class DistantLight : public Light {
  public:
    // DistantLight Public Methods
    DistantLight(const AnimatedTransform &worldFromLight,
                 SpectrumHandle L, Allocator alloc);

    static DistantLight *Create(
        const AnimatedTransform &worldFromLight, const ParameterDictionary &dict,
        const RGBColorSpace *colorSpace, const FileLoc *loc, Allocator alloc);

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
    SpectrumHandle L;
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
        const ParameterDictionary &dict, const FileLoc *loc, Allocator alloc);

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
        const FileLoc *loc, Allocator alloc);

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
    SpectrumHandle I;
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
        const FileLoc *loc, Allocator alloc, const ShapeHandle shape);

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
    SpectrumHandle Lemit;
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
    SpectrumHandle L;
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
        const FileLoc *loc, Allocator alloc);

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
    SpectrumHandle I;
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
    switch (Tag()) {
    case TypeIndex<UniformInfiniteLight>():
        return Cast<UniformInfiniteLight>()->Le(ray, lambda);
    case TypeIndex<ImageInfiniteLight>():
        return Cast<ImageInfiniteLight>()->Le(ray, lambda);
    case TypeIndex<PointLight>():
    case TypeIndex<DistantLight>():
    case TypeIndex<ProjectionLight>():
    case TypeIndex<GoniometricLight>():
    case TypeIndex<DiffuseAreaLight>():
    case TypeIndex<SpotLight>():
        return SampledSpectrum(0.f);
    default:
        LOG_FATAL("Unhandled light type");
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_LIGHTS_SPOT_H
