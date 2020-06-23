// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_MEDIA_HOMOGENEOUS_H
#define PBRT_MEDIA_HOMOGENEOUS_H

// media/homogeneous.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/interaction.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/octree.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>

#include <memory>
#include <vector>

namespace pbrt {

bool GetMediumScatteringProperties(const std::string &name, SpectrumHandle *sigma_a,
                                   SpectrumHandle *sigma_s, Allocator alloc);

// Media Inline Functions

// HenyeyGreenstein Declarations
class alignas(8) HenyeyGreensteinPhaseFunction {
  public:
    // HenyeyGreenstein Public Methods
    PBRT_CPU_GPU
    HenyeyGreensteinPhaseFunction(Float g) : g(g) {}

    PBRT_CPU_GPU
    Float p(const Vector3f &wo, const Vector3f &wi) const {
        return EvaluateHenyeyGreenstein(Dot(wo, wi), g);
    }

    PBRT_CPU_GPU
    pstd::optional<PhaseFunctionSample> Sample_p(const Vector3f &wo,
                                                 const Point2f &u) const {
        Float pdf;
        Vector3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};  // sampling is exact.
    }

    PBRT_CPU_GPU
    Float PDF(const Vector3f &wo, const Vector3f &wi) const { return p(wo, wi); }

    std::string ToString() const;

  private:
    Float g;
};

struct MediumSample {
    SampledSpectrum Tr;  // Actually Tr/pdf .... FIXME/revisit
    pstd::optional<MediumInteraction> intr;
};

struct NewMediumSample {
    NewMediumSample() : Tmaj(1) { }
    explicit NewMediumSample(const SampledSpectrum &Tmaj) : Tmaj(Tmaj) {}
    NewMediumSample(const MediumInteraction &intr, Float t, const SampledSpectrum &Tmaj)
        : intr(intr), t(t), Tmaj(Tmaj) {}

    pstd::optional<MediumInteraction> intr;
    Float t;
    SampledSpectrum Tmaj;

    std::string ToString() const;
};

// HomogeneousMedium Declarations
class alignas(8) HomogeneousMedium {
  public:
    // HomogeneousMedium Public Methods
    HomogeneousMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, SpectrumHandle Le,
                      Float g, Allocator alloc)
        : sigma_a_spec(sigma_a, alloc), sigma_s_spec(sigma_s, alloc), Le_spec(Le, alloc),
          phase(g), g(g) {}

    static HomogeneousMedium *Create(const ParameterDictionary &parameters,
                                     const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU
    NewMediumSample SampleTmaj(const Ray &ray, Float tMax, Float u,
                               const SampledWavelengths &lambda,
                               ScratchBuffer *scratchBuffer) const;

    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
    HenyeyGreensteinPhaseFunction phase;
    Float g;
};

// GridDensityMedium Declarations
class alignas(8) GridDensityMedium {
  public:
    // GridDensityMedium Public Methods
    GridDensityMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, SpectrumHandle Le,
                      Float g, const Transform &worldFromMedium,
                      pstd::optional<SampledGrid<Float>> densityGrid,
                      pstd::optional<SampledGrid<RGB>> rgbDensityGrid,
                      const RGBColorSpace *colorSpace, SampledGrid<Float> LeScaleGrid,
                      Allocator alloc);

    static GridDensityMedium *Create(const ParameterDictionary &parameters,
                                     const Transform &worldFromMedium, const FileLoc *loc,
                                     Allocator alloc);

    PBRT_CPU_GPU
    NewMediumSample SampleTmaj(const Ray &ray, Float tMax, Float u,
                               const SampledWavelengths &lambda,
                               ScratchBuffer *scratchBuffer) const;

    bool IsEmissive() const { return Le_spec.MaxValue() > 0; }

    std::string ToString() const;

  private:
    struct OctreeNode {
        Float maxDensity;  //  unused for interior nodes
        pstd::array<OctreeNode *, 8> *children = nullptr;

        PBRT_CPU_GPU
        OctreeNode *&child(int n) {
            DCHECK(children);
            return (*children)[n];
        }

        PBRT_CPU_GPU
        const OctreeNode *child(int n) const {
            return children ? (*children)[n] : nullptr;
        }
    };

    void buildOctree(OctreeNode *node, Allocator alloc, const Bounds3f &bounds, int depth);
    void simplifyOctree(OctreeNode *node, const Bounds3f &bounds, Float SE,
                        Float sigma_t);

    PBRT_CPU_GPU
    SampledSpectrum Le(const Point3f &p, const SampledWavelengths &lambda) const {
        return Le_spec.Sample(lambda) * LeScaleGrid.Lookup(p);
    }

    // GridDensityMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec, Le_spec;
    HenyeyGreensteinPhaseFunction phase;
    Float g;
    Transform mediumFromWorld, worldFromMedium;

    pstd::optional<SampledGrid<Float>> densityGrid;
    pstd::optional<SampledGrid<RGB>> rgbDensityGrid;
    const RGBColorSpace *colorSpace;

    SampledGrid<Float> LeScaleGrid;
    OctreeNode densityOctree;
    pstd::pmr::monotonic_buffer_resource treeBufferResource;
};

inline Float PhaseFunctionHandle::p(const Vector3f &wo, const Vector3f &wi) const {
    auto p = [&](auto ptr) { return ptr->p(wo, wi); };
    return Apply<Float>(p);
}

inline pstd::optional<PhaseFunctionSample> PhaseFunctionHandle::Sample_p(
    const Vector3f &wo, const Point2f &u) const {
    auto sample = [&](auto ptr) { return ptr->Sample_p(wo, u); };
    return Apply<pstd::optional<PhaseFunctionSample>>(sample);
}

inline Float PhaseFunctionHandle::PDF(const Vector3f &wo, const Vector3f &wi) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(wo, wi); };
    return Apply<Float>(pdf);
}

inline NewMediumSample MediumHandle::SampleTmaj(const Ray &ray, Float tMax, Float u,
                                                const SampledWavelengths &lambda,
                                                ScratchBuffer *scratchBuffer) const {
    auto sampletn = [&](auto ptr) { return ptr->SampleTmaj(ray, tMax, u, lambda,
                                                           scratchBuffer); };
    return Apply<NewMediumSample>(sampletn);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_GRID_H
