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

#ifndef PBRT_BASE_H
#define PBRT_BASE_H

#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/bsdf.h>
#include <pbrt/interaction.h>
#include <pbrt/options.h>
#include <pbrt/ray.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/array2d.h>
#include <pbrt/transform.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/buffercache.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// Primitive Declarations
class Primitive {
  public:
    // Primitive Interface
    virtual ~Primitive();
    virtual Bounds3f WorldBound() const = 0;
    virtual pstd::optional<ShapeIntersection> Intersect(const Ray &r, Float tMax = Infinity) const = 0;
    virtual bool IntersectP(const Ray &r, Float tMax = Infinity) const = 0;
};

// Filter Declarations
struct FilterSample {
    Point2f p;
    Float weight;
};

class Filter {
  public:
    // Filter Interface
    virtual ~Filter();
    Filter(const Vector2f &radius)
        : radius(radius) {}

    static Filter *Create(const std::string &name,
                          const ParameterDictionary &dict,
                          const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE
    virtual Float Evaluate(const Point2f &p) const = 0;
    PBRT_HOST_DEVICE
    virtual FilterSample Sample(const Point2f &u) const = 0;
    PBRT_HOST_DEVICE
    virtual Float Integral() const;

    virtual std::string ToString() const = 0;

    // Filter Public Data
    const Vector2f radius;
};

class FilterSampler {
  public:
    FilterSampler(const Filter *filter, int freq = 64, Allocator alloc = {});

    PBRT_HOST_DEVICE_INLINE
    FilterSample Sample(const Point2f &u) const {
        Point2f p = distrib.SampleContinuous(u);
        Point2f p01 = Point2f(domain.Offset(p));
        Point2i pi(Clamp(p01.x * values.xSize() + 0.5f, 0, values.xSize() - 1),
                   Clamp(p01.y * values.ySize() + 0.5f, 0, values.ySize() - 1));
        return { p, values[pi] < 0 ? -1.f : 1.f };
    }

    std::string ToString() const;

 private:
    Bounds2f domain;
    Array2D<Float> values;
    Distribution2D distrib;
};

// Film Declarations
// Note: AOVs only really make sense for PathIntegrator...
// volpath: medium interactions...
// bdpt, lightpath: Ld is partially computed via splats...
// simplepath, whitted: want to keep it simple
// sppm: 
struct VisibleSurface {
    VisibleSurface() = default;
    VisibleSurface(const SurfaceInteraction &si, const Camera &camera,
                   const SampledWavelengths &lambda);

    std::string ToString() const;

    Point3f p;
    Float dzdx = 0, dzdy = 0; // x/y: raster space, z: camera space
    Vector3f woWorld;
    Normal3f n, ns;
    Float time = 0;
    SampledSpectrum Le, Ld;
    BSDF *bsdf = nullptr;

private:
    Vector3f dpdx, dpdy; // world(ish) space
};

class Film {
  public:
    // Film Public Methods
    virtual ~Film();

    PBRT_HOST_DEVICE
    Bounds2f SampleBounds() const {
        return Bounds2f(
            Point2f(pixelBounds.pMin) - filter->radius + Vector2f(0.5f, 0.5f),
            Point2f(pixelBounds.pMax) + filter->radius - Vector2f(0.5f, 0.5f));
    }

    PBRT_HOST_DEVICE
    Bounds2f PhysicalExtent() const {
        Float aspect = (Float)fullResolution.y / (Float)fullResolution.x;
        Float x = std::sqrt(diagonal * diagonal / (1 + aspect * aspect));
        Float y = aspect * x;
        return Bounds2f(Point2f(-x / 2, -y / 2), Point2f(x / 2, y / 2));
    }

    PBRT_HOST_DEVICE
    virtual SampledWavelengths SampleWavelengths(Float u) const = 0;

    PBRT_HOST_DEVICE
    virtual void AddSample(const Point2i &pFilm, SampledSpectrum L,
                           const SampledWavelengths &lambda,
                           const pstd::optional<VisibleSurface> &visibleSurface,
                           Float weight) = 0;
    PBRT_HOST_DEVICE
    virtual void AddSplat(const Point2f &p, SampledSpectrum v,
                          const SampledWavelengths &lambda) = 0;

    virtual void WriteImage(ImageMetadata metadata, Float splatScale = 1) = 0;
    virtual Image GetImage(ImageMetadata *metadata, Float splatScale = 1) = 0;

    virtual std::string ToString() const = 0;

    // Film Public Data
    Point2i fullResolution;
    Float diagonal;
    pstd::unique_ptr<Filter> filter;
    std::string filename;
    Bounds2i pixelBounds;

 protected:
    Film(const Point2i &resolution, const Bounds2i &pixelBounds,
         pstd::unique_ptr<Filter> filt, Float diagonal,
         const std::string &filename);
    std::string BaseToString() const;
};

// Camera Declarations
class CameraWiSample {
public:
    CameraWiSample() = default;
    CameraWiSample(const SampledSpectrum &Wi, const Vector3f &wi, Float pdf,
                   Point2f pRaster, const Interaction &pRef, const Interaction &pLens)
        : Wi(Wi), wi(wi), pdf(pdf), pRaster(pRaster), pRef(pRef), pLens(pLens) {}

    bool Unoccluded(const Scene &scene) const;
    SampledSpectrum Tr(const Scene &scene, const SampledWavelengths &lambda,
                       Sampler &sampler) const;

    SampledSpectrum Wi;
    Vector3f wi;
    Float pdf;
    Point2f pRaster;
    Interaction pRef, pLens;
};

struct CameraRay {
    Ray ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

struct CameraRayDifferential {
    RayDifferential ray;
    SampledSpectrum weight = SampledSpectrum(1);
};

class Camera {
  public:
    // Camera Interface
    Camera(const AnimatedTransform &worldFromCamera, Float shutterOpen,
           Float shutterClose, std::unique_ptr<Film> film, const Medium *medium);
    virtual ~Camera();

    virtual pstd::optional<CameraRay>
    PBRT_HOST_DEVICE
    GenerateRay(const CameraSample &sample, const SampledWavelengths &lambda) const = 0;
    virtual pstd::optional<CameraRayDifferential> GenerateRayDifferential(const CameraSample &sample,
                                                                          const SampledWavelengths &lambda) const;
    virtual SampledSpectrum We(const Ray &ray, const SampledWavelengths &lambda,
                               Point2f *pRaster2 = nullptr) const;
    virtual void Pdf_We(const Ray &ray, Float *pdfPos, Float *pdfDir) const;
    virtual pstd::optional<CameraWiSample> Sample_Wi(const Interaction &ref, const Point2f &u,
                                                     const SampledWavelengths &lambda) const;

    virtual void InitMetadata(ImageMetadata *metadata) const;
    virtual std::string ToString() const = 0;

    void ApproximatedPdxy(const SurfaceInteraction &si) const;

    // Camera Public Data
    AnimatedTransform worldFromCamera;
    Float shutterOpen, shutterClose;
    pstd::unique_ptr<Film> film;
    const Medium *medium;

  protected:
    Camera() = default;
    std::string BaseToString() const;

    void FindMinimumDifferentials();
    Vector3f minPosDifferentialX, minPosDifferentialY;
    Vector3f minDirDifferentialX, minDirDifferentialY;
};

struct CameraSample {
    Point2f pFilm;
    Point2f pLens;
    Float time = 0;
    Float weight = 1;

    std::string ToString() const;
};

class ProjectiveCamera : public Camera {
  public:
    // ProjectiveCamera Public Methods
    ProjectiveCamera(const AnimatedTransform &worldFromCamera,
                     const Transform &screenFromCamera,
                     const Bounds2f &screenWindow, Float shutterOpen,
                     Float shutterClose, Float lensRadius, Float focalDistance,
                     std::unique_ptr<Film> f, const Medium *medium);
    void InitMetadata(ImageMetadata *metadata) const;

    //  protected:
    ProjectiveCamera() = default;
    std::string BaseToString() const;

    // ProjectiveCamera Protected Data
    Transform screenFromCamera, cameraFromRaster;
    Transform rasterFromScreen, screenFromRaster;
    Float lensRadius, focalDistance;
};


// Shape Declarations
struct ShapeSample {
    Interaction intr;
    Float pdf;

    std::string ToString() const;
};

struct ShapeIntersection {
    SurfaceInteraction intr;
    Float tHit;

    std::string ToString() const;
};

class Triangle;
class BilinearPatch;
class Curve;
class Sphere;
class Cylinder;
class Disk;

class ShapeHandle : public TaggedPointer<Triangle, BilinearPatch, Curve, Sphere,
                                         Cylinder, Disk> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    ShapeHandle(TaggedPointer<Triangle, BilinearPatch, Curve, Sphere, Cylinder,
                              Disk> sp)
        : TaggedPointer(sp) { }

    static pstd::vector<ShapeHandle> Create(
        const std::string &name, const Transform *worldFromObject,
        const Transform *objectFromWorld, bool reverseOrientation,
        const ParameterDictionary &dict, Allocator alloc, FileLoc loc);

    PBRT_HOST_DEVICE_INLINE
    Bounds3f WorldBound() const;

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax = Infinity) const;
    PBRT_HOST_DEVICE_INLINE
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_HOST_DEVICE_INLINE
    Float Area() const;

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE_INLINE
    Float PDF(const Interaction &) const;

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<ShapeSample> Sample(const Interaction &ref, const Point2f &u) const;
    PBRT_HOST_DEVICE_INLINE
    Float PDF(const Interaction &ref, const Vector3f &wi) const;

    PBRT_HOST_DEVICE_INLINE
    Float SolidAngle(const Point3f &p, int nSamples = 512) const;
    PBRT_HOST_DEVICE_INLINE
    DirectionCone NormalBounds() const;

    PBRT_HOST_DEVICE_INLINE
    bool OrientationIsReversed() const;
    PBRT_HOST_DEVICE_INLINE
    bool TransformSwapsHandedness() const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    Float SampledSolidAngle(const Point3f &p, int nSamples = 512) const;

    static void InitBufferCaches(Allocator alloc);
    static void FreeBufferCaches();

  private:
    friend class TriangleMesh;
    friend class BilinearPatchMesh;

    static BufferCache<int> *indexBufferCache;
    static BufferCache<Point3f> *pBufferCache;
    static BufferCache<Normal3f> *nBufferCache;
    static BufferCache<Point2f> *uvBufferCache;
    static BufferCache<Vector3f> *sBufferCache;
    static BufferCache<int> *faceIndexBufferCache;
};


// Texture Declarations
class TextureEvalContext;

class FloatConstantTexture;
class FloatBilerpTexture;
class FloatCheckerboardTexture;
class FloatDotsTexture;
class FBmTexture;
class GPUFloatImageTexture;
class FloatImageTexture;
class FloatMixTexture;
class FloatPtexTexture;
class FloatScaledTexture;
class WindyTexture;
class WrinkledTexture;

class FloatTextureHandle : public TaggedPointer<FloatImageTexture, GPUFloatImageTexture,
                                                FloatMixTexture, FloatScaledTexture, FloatConstantTexture,
                                                FloatBilerpTexture, FloatCheckerboardTexture,
                                                FloatDotsTexture, FBmTexture, FloatPtexTexture,
                                                WindyTexture, WrinkledTexture> {
public:
    using TaggedPointer::TaggedPointer;
    FloatTextureHandle(TaggedPointer<FloatImageTexture, GPUFloatImageTexture,
                                     FloatMixTexture, FloatScaledTexture, FloatConstantTexture,
                                     FloatBilerpTexture, FloatCheckerboardTexture,
                                     FloatDotsTexture, FBmTexture, FloatPtexTexture,
                                     WindyTexture, WrinkledTexture> tp)
        : TaggedPointer(tp) { }


    static FloatTextureHandle Create(const std::string &name,
                                     const Transform &worldFromTexture,
                                     const TextureParameterDictionary &dict,
                                     Allocator alloc, FileLoc loc, bool gpu);

    PBRT_HOST_DEVICE_INLINE
    Float Evaluate(const TextureEvalContext &ctx) const;
    PBRT_HOST_DEVICE
    Float EvaluateRare(const TextureEvalContext &ctx) const;

    std::string ToString() const;
};


class SpectrumConstantTexture;
class SpectrumBilerpTexture;
class SpectrumCheckerboardTexture;
class SpectrumImageTexture;
class GPUSpectrumImageTexture;
class MarbleTexture;
class SpectrumMixTexture;
class SpectrumDotsTexture;
class SpectrumPtexTexture;
class SpectrumScaledTexture;
class UVTexture;

class SpectrumTextureHandle :
        public TaggedPointer<SpectrumImageTexture, GPUSpectrumImageTexture,
                             SpectrumMixTexture, SpectrumScaledTexture, SpectrumConstantTexture,
                             SpectrumBilerpTexture, SpectrumCheckerboardTexture,
                             MarbleTexture, SpectrumDotsTexture,
                             SpectrumPtexTexture, UVTexture> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE_INLINE
    SpectrumTextureHandle(TaggedPointer<SpectrumImageTexture, GPUSpectrumImageTexture,
                                        SpectrumMixTexture, SpectrumScaledTexture, SpectrumConstantTexture,
                                        SpectrumBilerpTexture, SpectrumCheckerboardTexture,
                                        MarbleTexture, SpectrumDotsTexture,
                                        SpectrumPtexTexture, UVTexture> tp)
        : TaggedPointer(tp) { }

    static SpectrumTextureHandle Create(const std::string &name,
                                        const Transform &worldFromTexture,
                                        const TextureParameterDictionary &dict,
                                        Allocator alloc, FileLoc loc, bool gpu);

    // This is defined in textures.h. That's kind of weird...
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE
    SampledSpectrum EvaluateRare(const TextureEvalContext &ctx,
                                 const SampledWavelengths &lambda) const;

    std::string ToString() const;
};


// Material Declarations
class CoatedDiffuseMaterial;
class ConductorMaterial;
class DielectricMaterial;
class DiffuseMaterial;
class DiffuseTransmissionMaterial;
class DisneyMaterial;
class HairMaterial;
class LayeredMaterial;
class MeasuredMaterial;
class MixMaterial;
class SubsurfaceMaterial;
class ThinDielectricMaterial;

class MaterialHandle : public TaggedPointer<CoatedDiffuseMaterial, ConductorMaterial,
                                            DielectricMaterial, DiffuseMaterial,
                                            DiffuseTransmissionMaterial, DisneyMaterial,
                                            HairMaterial, LayeredMaterial, MeasuredMaterial,
                                            MixMaterial, SubsurfaceMaterial,
                                            ThinDielectricMaterial> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE_INLINE
    MaterialHandle(TaggedPointer<CoatedDiffuseMaterial, ConductorMaterial,
                                 DielectricMaterial, DiffuseMaterial,
                                 DiffuseTransmissionMaterial, DisneyMaterial,
                                 HairMaterial, LayeredMaterial, MeasuredMaterial,
                                 MixMaterial, SubsurfaceMaterial,
                                 ThinDielectricMaterial> tp)
        : TaggedPointer(tp) { }

    static MaterialHandle Create(const std::string &name, const TextureParameterDictionary &dict,
                                 /*const */std::map<std::string, MaterialHandle> &namedMaterials,
                                 Allocator alloc, FileLoc loc);

    // -> bool Matches(std::init_list<FloatTextureHandle>, ...<SpectrumTextureHandle>
    // -> Float operator()(FloatTextureHandle tex, cont TextureEvalContext &ctx)
    // -> SampledSpectrum operator()(SpectrumTextureHandle tex, cont TextureEvalContext &ctx, SampledWavelengths lambda)
    template <typename TextureEvaluator>
    PBRT_HOST_DEVICE_INLINE
    BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                  const SampledWavelengths &lambda,
                  MaterialBuffer &materialBuffer, TransportMode mode) const;

    PBRT_HOST_DEVICE_INLINE
    BSSRDF *GetBSSRDF(SurfaceInteraction &si, const SampledWavelengths &lambda,
                      MaterialBuffer &materialBuffer, TransportMode mode) const;

    PBRT_HOST_DEVICE_INLINE
    FloatTextureHandle GetDisplacement() const;

    PBRT_HOST_DEVICE_INLINE
    bool IsTransparent() const;

    std::string ToString() const;
};

// Sampler Declarations
class Sampler {
  public:
    // Sampler Interface
    PBRT_HOST_DEVICE
    Sampler(int samplesPerPixel);
    PBRT_HOST_DEVICE
    virtual ~Sampler();

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int sampleIndex);

    PBRT_HOST_DEVICE
    Float Get1D();
    PBRT_HOST_DEVICE
    Point2f Get2D();

    PBRT_HOST_DEVICE
    CameraSample GetCameraSample(const Filter &filter);

    PBRT_HOST_DEVICE_INLINE
    int GetDiscrete1D(int n) {
        return std::min<int>(Get1D() * n, n - 1);
    }

    // Sampler Public Data
    const int samplesPerPixel;

    virtual std::unique_ptr<Sampler> Clone() = 0;

    virtual std::string ToString() const = 0;

  protected:
    std::string BaseToString() const;

    class SamplerState {
      public:
        SamplerState() = default;
        PBRT_HOST_DEVICE_INLINE
        SamplerState(const Point2i &p, int sampleIndex)
            : p(p),
              sampleIndex(sampleIndex),
              dimension(0) { }

        Point2i p{std::numeric_limits<int>::lowest(),
                  std::numeric_limits<int>::lowest()};
        int sampleIndex = 0;
        int dimension = 0;

        std::string ToString() const;

        // Based on p and dimension, *not* sampleIndex.
        PBRT_HOST_DEVICE_INLINE
        uint64_t Hash() const {
            return MixBits(((uint64_t)p.x << 48) ^ ((uint64_t)p.y << 32) ^
                           ((uint64_t)dimension << 16)
#ifndef __CUDA_ARCH__
                           ^ PbrtOptions.seed
#endif
                           );
        }
    };

    // Methods that Sampler implementations must provide.
    PBRT_HOST_DEVICE
    virtual void ImplStartPixelSample(const Point2i &p, int sampleIndex);
    PBRT_HOST_DEVICE
    virtual Float ImplGet1D(const SamplerState &s) = 0;
    PBRT_HOST_DEVICE
    virtual Point2f ImplGet2D(const SamplerState &s) = 0;

  private:
    // Sampler Private Data
    SamplerState state;
};

// Integrator Declarations
class Integrator {
  public:
    // Integrator Interface
    virtual ~Integrator();
    virtual void Render() = 0;
    virtual std::string ToString() const = 0;

 protected:
    Integrator(const Scene &scene) : scene(scene) {}
    const Scene &scene;
};

// Media Declarations
struct PhaseFunctionSample {
    Float p;
    Vector3f wi;
    Float pdf;
};

class PhaseFunction {
  public:
    // PhaseFunction Interface
    PBRT_HOST_DEVICE
    virtual ~PhaseFunction();

    PBRT_HOST_DEVICE
    virtual Float p(const Vector3f &wo, const Vector3f &wi) const = 0;

    PBRT_HOST_DEVICE
    virtual pstd::optional<PhaseFunctionSample> Sample_p(
        const Vector3f &wo, const Point2f &u) const = 0;

    PBRT_HOST_DEVICE
    virtual Float PDF(const Vector3f &wo, const Vector3f &wi) const = 0;

    virtual std::string ToString() const = 0;
};

// Medium Declarations
class Medium {
  public:
    // Medium Interface
    virtual ~Medium() {}
    virtual SampledSpectrum Tr(const Ray &ray, Float tMax, const SampledWavelengths &lambda,
                               Sampler &sampler) const = 0;
    virtual SampledSpectrum Sample(const Ray &ray, Float tMax, Sampler &sampler,
                                   const SampledWavelengths &lambda,
                                   MemoryArena &arena,
                                   MediumInteraction *mi) const = 0;
    virtual std::string ToString() const = 0;
};

// MediumInterface Declarations
struct MediumInterface {
    MediumInterface() : inside(nullptr), outside(nullptr) {}
    // MediumInterface Public Methods
    MediumInterface(const Medium *medium) : inside(medium), outside(medium) {}
    MediumInterface(const Medium *inside, const Medium *outside)
        : inside(inside), outside(outside) {}
    bool IsMediumTransition() const { return inside != outside; }
    const Medium *inside, *outside;

    std::string ToString() const;
};


// LightType Declarations
enum class LightType : int {
    DeltaPosition = 1,
    DeltaDirection = 2,
    Area = 4,
    Infinite = 8
};

std::string ToString(LightType type);

PBRT_HOST_DEVICE_INLINE
bool IsDeltaLight(LightType type) {
    return (type == LightType::DeltaPosition ||
            type == LightType::DeltaDirection);
}

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

// Light Declarations
class LightLiSample {
public:
    LightLiSample() = default;
    PBRT_HOST_DEVICE_INLINE
    LightLiSample(const Light *light, const SampledSpectrum &L, const Vector3f &wi,
                  Float pdf, const Interaction &pRef, const Interaction &pLight)
        : L(L), wi(wi), pdf(pdf), light(light), pRef(pRef), pLight(pLight) {}

    bool Unoccluded(const Scene &scene) const;
    SampledSpectrum Tr(const Scene &scene, const SampledWavelengths &lambda,
                       Sampler &sampler) const;

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

class PointLight;
class DistantLight;
class ProjectionLight;
class GoniometricLight;
class DiffuseAreaLight;
class UniformInfiniteLight;
class ImageInfiniteLight;
class SpotLight;

class AreaLight;
class InfiniteAreaLight;

class LightHandle : public TaggedPointer<PointLight, DistantLight, ProjectionLight,
                                         GoniometricLight, SpotLight, DiffuseAreaLight,
                                         UniformInfiniteLight, ImageInfiniteLight> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    LightHandle(TaggedPointer<PointLight, DistantLight, ProjectionLight,
                              GoniometricLight, SpotLight, DiffuseAreaLight,
                              UniformInfiniteLight, ImageInfiniteLight> tp)
        : TaggedPointer(tp) { }

    static LightHandle Create(const std::string &name,
                              const ParameterDictionary &dict,
                              const AnimatedTransform &worldFromLight,
                              const Medium *outsideMedium, FileLoc loc,
                              Allocator alloc);
    static LightHandle CreateArea(const std::string &name,
                                  const ParameterDictionary &dict,
                                  const AnimatedTransform &worldFromLight,
                                  const MediumInterface &mediumInterface,
                                  const ShapeHandle shape, FileLoc loc,
                                  Allocator alloc);

    // These shouldn't be called. Add these to get decent error messages
    // when they are.
    PBRT_HOST_DEVICE
    LightHandle(const AreaLight *) = delete;
    PBRT_HOST_DEVICE
    LightHandle(const InfiniteAreaLight *) = delete;

    // All of the types inherit from Light to pick up a few data members...
    // TODO: this is probably too subtle an interface.
    PBRT_HOST_DEVICE
    Light *operator->() { return (Light *)ptr(); }
    PBRT_HOST_DEVICE
    const Light *operator->() const { return (const Light *)ptr(); }

    PBRT_HOST_DEVICE_INLINE
    pstd::optional<LightLiSample> Sample_Li(const Interaction &ref, const Point2f &u,
                                            const SampledWavelengths &lambda) const;

    SampledSpectrum Phi(const SampledWavelengths &lambda) const;

    void Preprocess(const Bounds3f &worldBounds);

    PBRT_HOST_DEVICE_INLINE
    Float Pdf_Li(const Interaction &ref, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    pstd::optional<LightLeSample> Sample_Le(const Point2f &u1, const Point2f &u2,
                                            const SampledWavelengths &lambda,
                                            Float time) const;

    // Note shouldn't be called for area lights..
    PBRT_HOST_DEVICE
    void Pdf_Le(const Ray &ray, Float *pdfPos, Float *pdfDir) const;

    pstd::optional<LightBounds> Bounds() const;

    std::string ToString() const;

    // AreaLights only
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum L(const Interaction &intr, const Vector3f &w,
                      const SampledWavelengths &lambda) const;
    PBRT_HOST_DEVICE
    void Pdf_Le(const Interaction &intr, Vector3f &w, Float *pdfPos,
                Float *pdfDir) const;

    // InfiniteAreaLights only
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Le(const Ray &ray, const SampledWavelengths &lambda) const;
};

struct LightHandleHash {
    PBRT_HOST_DEVICE
    size_t operator()(LightHandle lightHandle) const {
        return Hash(lightHandle.ptr());
    }
};

class Light {
  public:
    Light(LightType flags, const AnimatedTransform &worldFromLight,
          const MediumInterface &mediumInterface);

    // Light Public Data
    LightType type;
    MediumInterface mediumInterface;

  protected:
    std::string BaseToString() const;

    // Light Protected Data
    AnimatedTransform worldFromLight;
};

}  // namespace pbrt

#endif // PBRT_BASE_H
