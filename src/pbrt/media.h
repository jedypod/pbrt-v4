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
#include <pbrt/util/colorspace.h>
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
    MediumSample() : Tmaj(1) { }

    PBRT_CPU_GPU
    explicit MediumSample(const SampledSpectrum &Tmaj) : Tmaj(Tmaj) {}

    PBRT_CPU_GPU
    MediumSample(const MediumInteraction &intr, Float t, const SampledSpectrum &Tmaj)
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
    MediumSample Sample_Tmaj(const Ray &ray, Float tMax, Float u,
                             const SampledWavelengths &lambda,
                             ScratchBuffer *scratchBuffer) const {
        // So t corresponds to distance...
        tMax *= Length(ray.d);
        Ray rayp(ray.o, Normalize(ray.d));

        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        SampledSpectrum sigma_maj = sigma_t;

        Float t = SampleExponential(u, sigma_maj[0]);
        if (t >= tMax)
            return MediumSample(FastExp(-tMax * sigma_maj));

        SampledSpectrum Tmaj = FastExp(-t * sigma_maj);
        SampledSpectrum Le = Le_spec.Sample(lambda);
        MediumInteraction intr(rayp(t), -rayp.d, ray.time, sigma_a, sigma_s, sigma_maj,
                               Le, this, &phase);

        return MediumSample(intr, t, Tmaj);
    }

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
    MediumSample Sample_Tmaj(const Ray &rWorld, Float raytMax, Float u,
                             const SampledWavelengths &lambda,
                             ScratchBuffer *scratchBuffer) const {
        raytMax *= Length(rWorld.d);
        Ray ray = mediumFromWorld(Ray(rWorld.o, Normalize(rWorld.d)), &raytMax);
        // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium bounds
        const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
        Float tMin, tMax;
        if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return {};

        DCHECK_LE(tMax, raytMax);

        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;

        MediumSample mediumSample;

        TraverseOctree(&densityOctree, ray.o, ray.d, raytMax,
            [&](const OctreeNode &node, Float t0, Float t1) {
                if (node.maxDensity == 0)
                    // Empty--skip it!
                    return OctreeTraversal::Continue;

                SampledSpectrum sigma_maj(sigma_t * node.maxDensity);

                // At what u value do we hit the the cell exit point?
                Float uEnd = InvertExponentialSample(t1 - t0, sigma_maj[0]);
                if (u >= uEnd) {
                    // exit this cell w/o a scattering event
                    u = (u - uEnd) / (1 - uEnd);  // remap to [0,1)
                    return OctreeTraversal::Continue;
                }

                Float t = t0 + SampleExponential(u, sigma_maj[0]);
                CHECK_RARE(1e-5, t > t1);

                if (t >= tMax) {
                    // Nothing before the geom intersection; get out of here
                    mediumSample = MediumSample(FastExp(-sigma_maj * (tMax - t0)));
                    return OctreeTraversal::Abort;
                }

                // Scattering event (of some sort)
                Point3f p = ray(t);
                SampledSpectrum Tmaj = FastExp(-sigma_maj * (t - t0));

                if (densityGrid) {
                    Float density = densityGrid->Lookup(p);
                    sigma_a *= density;
                    sigma_s *= density;
                } else {
                    RGB density = rgbDensityGrid->Lookup(p);
                    CHECK_LE(density.r, node.maxDensity);
                    CHECK_LE(density.g, node.maxDensity);
                    CHECK_LE(density.b, node.maxDensity);
                    SampledSpectrum spec =
#ifdef PBRT_IS_GPU_CODE
                        RGBSpectrum(*RGBColorSpace_sRGB, density).Sample(lambda);
#else
                        RGBSpectrum(*RGBColorSpace::sRGB, density).Sample(lambda);
#endif
                    sigma_a *= spec;
                    sigma_s *= spec;
                }

                MediumInteraction intr(worldFromMedium(p), -Normalize(rWorld.d), rWorld.time,
                                       sigma_a, sigma_s, sigma_maj, Le(p, lambda), this,
                                       &phase);
                mediumSample = MediumSample(intr, t, Tmaj);
                return OctreeTraversal::Abort;
            });

        return mediumSample;
    }

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

inline MediumSample MediumHandle::Sample_Tmaj(const Ray &ray, Float tMax, Float u,
                                                const SampledWavelengths &lambda,
                                                ScratchBuffer *scratchBuffer) const {
    auto sampletn = [&](auto ptr) { return ptr->Sample_Tmaj(ray, tMax, u, lambda,
                                                           scratchBuffer); };
    return Apply<MediumSample>(sampletn);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_GRID_H
