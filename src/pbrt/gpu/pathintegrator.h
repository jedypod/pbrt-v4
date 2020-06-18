// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_GPU_H
#define PBRT_GPU_H

#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/workqueue.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

class ParsedScene;

void GPUInit();
void GPURender(ParsedScene &scene);

struct PathState {
    SampledSpectrum L;
    SampledSpectrum beta;
    Float filterSampleWeight;
    Float etaScale;
    bool anyNonSpecularBounces;

    pstd::optional<SampledLight> sampledLight;

    Float scatteringPDF;
    BxDFFlags bsdfFlags;
};

class IndexBase {
  public:
    PBRT_CPU_GPU
    explicit operator int() const { return index; }
    PBRT_CPU_GPU
    explicit operator size_t() const { return index; }

    PBRT_CPU_GPU
    bool operator>=(int v) const { return index >= v; }

  protected:
    PBRT_CPU_GPU
    IndexBase(int index) : index(index) {}

    int index = -1;
};

class PathRayIndex : public IndexBase {
  public:
    PathRayIndex() = default;
    PBRT_CPU_GPU
    explicit PathRayIndex(int index) : IndexBase(index) {}
};

class ShadowRayIndex : public IndexBase {
  public:
    ShadowRayIndex() = default;
    PBRT_CPU_GPU
    explicit ShadowRayIndex(int index) : IndexBase(index) {}
};

class SSRayIndex : public IndexBase {
  public:
    SSRayIndex() = default;
    PBRT_CPU_GPU
    explicit SSRayIndex(int index) : IndexBase(index) {}
};

class PixelIndex : public IndexBase {
  public:
    PixelIndex() = default;
    PBRT_CPU_GPU
    explicit PixelIndex(int index) : IndexBase(index) {}
};

class BxDFEvalIndex : public IndexBase {
  public:
    BxDFEvalIndex() = default;
    PBRT_CPU_GPU
    explicit BxDFEvalIndex(int index) : IndexBase(index) {}
};

class MediumEvalIndex : public IndexBase {
  public:
    MediumEvalIndex() = default;
    PBRT_CPU_GPU
    explicit MediumEvalIndex(int index) : IndexBase(index) {}
};

template <typename Index = size_t>
class Point3fSOA {
  public:
    Point3fSOA(Allocator alloc, size_t n = 0) : pts(alloc, n) {}

    PBRT_CPU_GPU
    Point3f operator[](Index offset) const {
        return {pts.at<0>(size_t(offset)), pts.at<1>(size_t(offset)),
                pts.at<2>(size_t(offset))};
    }

    class Point3fRef {
      public:
        PBRT_CPU_GPU
        Point3fRef(Float *x, Float *y, Float *z) : x(x), y(y), z(z) {}

        PBRT_CPU_GPU
        operator Point3f() const { return {*x, *y, *z}; }
        PBRT_CPU_GPU
        void operator=(const Point3f &p) {
            *x = p.x;
            *y = p.y;
            *z = p.z;
        }

      private:
        Float *x, *y, *z;
    };

    PBRT_CPU_GPU
    Point3fRef operator[](Index offset) {
        return Point3fRef(pts.ptr<0>(size_t(offset)), pts.ptr<1>(size_t(offset)),
                          pts.ptr<2>(size_t(offset)));
    }

  private:
    AoSoA<Float, Float, Float> pts;
};

template <typename Index = size_t>
class Vector3fSOA {
  public:
    Vector3fSOA(Allocator alloc, size_t n = 0) : v(alloc, n) {}

    PBRT_CPU_GPU
    Vector3f operator[](Index offset) const {
        return {v.at<0>(size_t(offset)), v.at<1>(size_t(offset)),
                v.at<2>(size_t(offset))};
    }

    class Vector3fRef {
      public:
        PBRT_CPU_GPU
        Vector3fRef(Float *x, Float *y, Float *z) : x(x), y(y), z(z) {}

        PBRT_CPU_GPU
        operator Vector3f() const { return {*x, *y, *z}; }
        PBRT_CPU_GPU
        void operator=(const Vector3f &p) {
            *x = p.x;
            *y = p.y;
            *z = p.z;
        }

      private:
        Float *x, *y, *z;
    };

    PBRT_CPU_GPU
    Vector3fRef operator[](Index offset) {
        return Vector3fRef(v.ptr<0>(size_t(offset)), v.ptr<1>(size_t(offset)),
                           v.ptr<2>(size_t(offset)));
    }

  private:
    AoSoA<Float, Float, Float> v;
};

template <typename Index>
class SampledSpectrumSOA {
  public:
    // FIXME
    static_assert(NSpectrumSamples == 4,
                  "SampledSpectrumSOA has hard-coded NSpectrumSamples");

    // CO    SampledSpectrumSOA() = default;
    SampledSpectrumSOA(Allocator alloc, size_t n = 0) : v(alloc, n) {}

    PBRT_CPU_GPU
    SampledSpectrum operator[](Index offset) const {
        return SampledSpectrum({v.at<0>(size_t(offset)), v.at<1>(size_t(offset)),
                                v.at<2>(size_t(offset)), v.at<3>(size_t(offset))});
    }

    class SampledSpectrumRef {
      public:
        PBRT_CPU_GPU
        SampledSpectrumRef(pstd::array<Float *, NSpectrumSamples> ptrs) : ptrs(ptrs) {}

        PBRT_CPU_GPU
        operator SampledSpectrum() const {
            return SampledSpectrum({*ptrs[0], *ptrs[1], *ptrs[2], *ptrs[3]});
        }
        PBRT_CPU_GPU
        explicit operator bool() const {
            return (bool)SampledSpectrum({*ptrs[0], *ptrs[1], *ptrs[2], *ptrs[3]});
        }
        PBRT_CPU_GPU
        void operator=(const SampledSpectrum &s) {
            for (int i = 0; i < NSpectrumSamples; ++i)
                *ptrs[i] = s[i];
        }

      private:
        pstd::array<Float *, NSpectrumSamples> ptrs;
    };

    PBRT_CPU_GPU
    SampledSpectrumRef operator[](Index offset) {
        return SampledSpectrumRef({v.ptr<0>(size_t(offset)), v.ptr<1>(size_t(offset)),
                                   v.ptr<2>(size_t(offset)), v.ptr<3>(size_t(offset))});
    }

  private:
    AoSoA<float, float, float, float> v;
};

// FIXME
template <typename RayIndex>
class RayQueue;
class VisibleSurface;
class UniformInfiniteLight;
class ImageInfiniteLight;
class GPUAccel;
template <typename T, typename Float>
class WeightedReservoirSampler;

enum class InteractionType { Surface, Medium, None };

class GPUPathIntegrator {
  public:
    GPUPathIntegrator(Allocator alloc, const ParsedScene &scene);

    void Render(ImageMetadata *metadata);

    void GenerateCameraRays(int pixelSample);
    void TraceShadowRays();
    void SampleMediumInteraction(int depth);
    void HandleMediumTransitions(int depth);
    void SampleSubsurface(int depth);
    template <typename TextureEvaluator>
    void EvaluateMaterial(TextureEvaluator texEval, int depth);

    void SampleLight(int depth);
    void SampleDirect(int depth);
    template <typename BxDF>
    void SampleDirect(int depth);

    void SampleIndirect(int depth);
    template <typename BxDF>
    void SampleIndirect(int depth, bool overrideRay = false);

    void UpdateFilm();
    void InitializeVisibleSurface();
    template <typename BxDF>
    void InitializeVisibleSurface();

    FilterHandle filter;
    FilmHandle film;
    SamplerHandle sampler;
    CameraHandle camera;
    Vector2i resolution;
    LightHandle envLight;
    pstd::vector<LightHandle> allLights;
    LightSamplerHandle lightSampler = nullptr;

    int maxDepth;
    bool regularize;
    int pixelsPerPass, scanlinesPerPass;

    bool initializeVisibleSurface;
    bool haveSubsurface;
    bool haveMedia;

    GPUAccel *accel = nullptr;

    TypedIndexSpan<PathState, PixelIndex> pathStates;
    // These are worth keeping independent of PathState for perf benefit.
    TypedIndexSpan<Point2i, PixelIndex> pPixels;
    TypedIndexSpan<SampledWavelengths, PixelIndex> lambdas;
    TypedIndexSpan<SurfaceInteraction, PixelIndex> intersections[2];
    TypedIndexSpan<MediumInteraction, PixelIndex> mediumInteractions[2];
    TypedIndexSpan<InteractionType, PathRayIndex> interactionType;
    TypedIndexSpan<SamplerHandle, PixelIndex> samplers;
    TypedIndexSpan<RNG *, PixelIndex> rngs;
    SampledSpectrumSOA<PixelIndex> cameraRayWeights;
    TypedIndexSpan<pstd::optional<VisibleSurface>, PixelIndex> visibleSurfaces;
    TypedIndexSpan<ScratchBuffer, PixelIndex> scratchBuffers;

    RayQueue<PathRayIndex> *pathRayQueue;
    int *numActiveRays;
    TypedIndexSpan<PixelIndex, PathRayIndex> rayIndexToPixelIndex[2];
    TypedIndexSpan<PathRayIndex, PixelIndex> pixelIndexToRayIndex;

    RayQueue<ShadowRayIndex> *shadowRayQueue;
    TypedIndexSpan<PixelIndex, ShadowRayIndex> shadowRayIndexToPixelIndex;
    SampledSpectrumSOA<ShadowRayIndex> shadowRayLd;
    TypedIndexSpan<SampledSpectrum, ShadowRayIndex> shadowTr;

    RayQueue<SSRayIndex> *randomHitRayQueue;
    TypedIndexSpan<MaterialHandle, SSRayIndex> subsurfaceMaterials;
    TypedIndexSpan<PathRayIndex, SSRayIndex> subsurfaceRayIndexToPathRayIndex;
    TypedIndexSpan<WeightedReservoirSampler<SurfaceInteraction>, SSRayIndex>
        subsurfaceReservoirSamplers;

    MultiWorkQueue<BxDFHandle::NumTags(), PathRayIndex, BxDFEvalIndex> *bxdfEvalQueues;
    WorkQueue<PathRayIndex, MediumEvalIndex> *mediumSampleQueue = nullptr;
    WorkQueue<PathRayIndex, MediumEvalIndex> *mediumEvalQueue = nullptr;
};

}  // namespace pbrt

#endif  // PBRT_GPU_H
