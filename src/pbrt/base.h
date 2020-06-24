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

#include <pbrt/options.h>
#include <pbrt/ray.h>
#include <pbrt/transform.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/array2d.h>
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

class BoxFilter;
class GaussianFilter;
class MitchellFilter;
class LanczosSincFilter;
class TriangleFilter;

class FilterHandle : public TaggedPointer<BoxFilter, GaussianFilter, MitchellFilter,
                                          LanczosSincFilter, TriangleFilter> {
public:
    using TaggedPointer::TaggedPointer;
    FilterHandle(TaggedPointer<BoxFilter, GaussianFilter, MitchellFilter,
                               LanczosSincFilter, TriangleFilter> tp)
        : TaggedPointer(tp) { }

    static FilterHandle Create(const std::string &name,
                               const ParameterDictionary &dict,
                               const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const;
    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE
    Vector2f Radius() const;
    PBRT_HOST_DEVICE
    Float Integral() const;

    std::string ToString() const;
};

class VisibleSurface;

class Film {
  public:
    // Film Public Methods
    virtual ~Film();

    static Film *Create(const std::string &name, const ParameterDictionary &dict,
                        const FileLoc *loc, FilterHandle filter, Allocator alloc);

    PBRT_HOST_DEVICE
    Bounds2f SampleBounds() const;

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
    FilterHandle filter;
    std::string filename;
    Bounds2i pixelBounds;

 protected:
    Film(const Point2i &resolution, const Bounds2i &pixelBounds,
         FilterHandle filter, Float diagonal, const std::string &filename);

    std::string BaseToString() const;
};

// Camera Declarations
class CameraRay;
class CameraRayDifferential;
class CameraWiSample;

struct CameraSample {
    Point2f pFilm;
    Point2f pLens;
    Float time = 0;
    Float weight = 1;

    std::string ToString() const;
};

class Camera {
  public:
    // Camera Interface
    Camera(const AnimatedTransform &worldFromCamera, Float shutterOpen,
           Float shutterClose, Film *film, const Medium *medium);
    virtual ~Camera();

    static Camera *Create(const std::string &name, const ParameterDictionary &dict,
                          const Medium *medium, const AnimatedTransform &worldFromCamera,
                          Film *film, const FileLoc *loc,
                          Allocator alloc);

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

    PBRT_HOST_DEVICE
    void ApproximatedPdxy(const SurfaceInteraction &si) const;

    // Camera Public Data
    AnimatedTransform worldFromCamera;
    Float shutterOpen, shutterClose;
    Film  *film;
    const Medium *medium;

  protected:
    Camera() = default;
    std::string BaseToString() const;

    void FindMinimumDifferentials();
    Vector3f minPosDifferentialX, minPosDifferentialY;
    Vector3f minDirDifferentialX, minDirDifferentialY;
};

// Shape Declarations
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
        const ParameterDictionary &dict, const FileLoc *loc, Allocator alloc);

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
                                     const FileLoc *loc, Allocator alloc, bool gpu);

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
                                        const FileLoc *loc, Allocator alloc, bool gpu);

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
                                 const FileLoc *loc, Allocator alloc);

    // -> bool Matches(std::init_list<FloatTextureHandle>, ...<SpectrumTextureHandle>
    // -> Float operator()(FloatTextureHandle tex, cont TextureEvalContext &ctx)
    // -> SampledSpectrum operator()(SpectrumTextureHandle tex, cont TextureEvalContext &ctx, SampledWavelengths lambda)
    template <typename TextureEvaluator>
    PBRT_HOST_DEVICE_INLINE
    BSDF *GetBSDF(TextureEvaluator texEval, SurfaceInteraction &si,
                  const SampledWavelengths &lambda,
                  MaterialBuffer &materialBuffer, TransportMode mode) const;

    template <typename TextureEvaluator>
    PBRT_HOST_DEVICE_INLINE
    BSSRDFHandle GetBSSRDF(TextureEvaluator texEval, SurfaceInteraction &si,
                           const SampledWavelengths &lambda,
                           MaterialBuffer &materialBuffer, TransportMode mode) const;

    PBRT_HOST_DEVICE_INLINE
    FloatTextureHandle GetDisplacement() const;

    PBRT_HOST_DEVICE_INLINE
    bool IsTransparent() const;

    PBRT_HOST_DEVICE_INLINE
    bool HasSubsurfaceScattering() const;

    std::string ToString() const;
};

class BSSRDFSample;
class BSSRDFProbeSegment;

class TabulatedBSSRDF;

class BSSRDFHandle : public TaggedPointer<TabulatedBSSRDF> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE
    BSSRDFHandle(TaggedPointer<TabulatedBSSRDF> tp)
        : TaggedPointer(tp) { }

    PBRT_HOST_DEVICE
    SampledSpectrum S(const SurfaceInteraction &pi, const Vector3f &wi);

    PBRT_HOST_DEVICE
    pstd::optional<BSSRDFProbeSegment> Sample(Float u1, const Point2f &u2) const;

    PBRT_HOST_DEVICE
    BSSRDFSample ProbeIntersectionToSample(const SurfaceInteraction &si,
                                           MaterialBuffer &materialBuffer) const;
};

// Sampler Declarations
class HaltonSampler;
class PaddedSobolSampler;
class PMJ02BNSampler;
class RandomSampler;
class SobolSampler;
class StratifiedSampler;
class MLTSampler;
class DebugMLTSampler;

class SamplerHandle : public TaggedPointer<HaltonSampler, PaddedSobolSampler, PMJ02BNSampler,
                                           RandomSampler, SobolSampler, StratifiedSampler,
                                           MLTSampler, DebugMLTSampler> {
public:
    using TaggedPointer::TaggedPointer;
    SamplerHandle(TaggedPointer<HaltonSampler, PaddedSobolSampler, PMJ02BNSampler,
                                RandomSampler, SobolSampler, StratifiedSampler,
                                MLTSampler, DebugMLTSampler> tp)
        : TaggedPointer(tp) { }

    static SamplerHandle Create(const std::string &name, const ParameterDictionary &dict,
                                const Point2i &fullResolution, const FileLoc *loc,
                                Allocator alloc);

    PBRT_HOST_DEVICE
    int SamplesPerPixel() const;

    PBRT_HOST_DEVICE
    void StartPixelSample(const Point2i &p, int sampleIndex);

    PBRT_HOST_DEVICE
    Float Get1D();

    PBRT_HOST_DEVICE
    Point2f Get2D();

    PBRT_HOST_DEVICE
    CameraSample GetCameraSample(const Point2i &pPixel, FilterHandle filter);

    PBRT_HOST_DEVICE
    int GetDiscrete1D(int n) {
        return std::min<int>(Get1D() * n, n - 1);
    }

    std::vector<SamplerHandle> Clone(int n, Allocator alloc);

    std::string ToString() const;
};

// Integrator Declarations
class Integrator {
  public:
    // Integrator Interface
    virtual ~Integrator();

    static std::unique_ptr<Integrator> Create(
        const std::string &name, const ParameterDictionary &dict,
        const Scene &scene, std::unique_ptr<Camera> camera,
        SamplerHandle sampler,
        const RGBColorSpace *colorSpace, const FileLoc *loc);

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

    static Medium *Create(const std::string &name,
                          const ParameterDictionary &dict,
                          const Transform &worldFromMedium, const FileLoc *loc,
                          Allocator alloc);

    virtual SampledSpectrum Tr(const Ray &ray, Float tMax, const SampledWavelengths &lambda,
                               SamplerHandle sampler) const = 0;
    virtual SampledSpectrum Sample(const Ray &ray, Float tMax, SamplerHandle sampler,
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
    DeltaPosition,
    DeltaDirection,
    Area,
    Infinite
};

std::string ToString(LightType type);

PBRT_HOST_DEVICE_INLINE
bool IsDeltaLight(LightType type) {
    return (type == LightType::DeltaPosition ||
            type == LightType::DeltaDirection);
}

// Light Declarations
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
                              const Medium *outsideMedium, const FileLoc *loc,
                              Allocator alloc);
    static LightHandle CreateArea(const std::string &name,
                                  const ParameterDictionary &dict,
                                  const AnimatedTransform &worldFromLight,
                                  const MediumInterface &mediumInterface,
                                  const ShapeHandle shape, const FileLoc *loc,
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
