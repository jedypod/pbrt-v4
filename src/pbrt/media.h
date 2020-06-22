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
    MediumInteraction intr;
    Float t;
    SampledSpectrum sigma_a, sigma_s;
    Float sigma_maj, Tmaj;
    SampledSpectrum Le;

    PBRT_CPU_GPU
    SampledSpectrum sigma_n() const {
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        SampledSpectrum sigma_n = SampledSpectrum(sigma_maj) - sigma_t;
        CHECK_RARE(1e-5, sigma_n.MinComponentValue() < 0);
        return ClampZero(sigma_n);
    }

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
    SampledSpectrum Tr(const Ray &ray, Float tMax, const SampledWavelengths &lambda,
                       RNG &rng) const {
        SampledSpectrum sigma_t = sigma_a_spec.Sample(lambda) +
            sigma_s_spec.Sample(lambda);
        return Exp(-sigma_t * std::min(tMax * Length(ray.d), MaxFloat));
    }

    PBRT_CPU_GPU
    MediumSample Sample(const Ray &ray, Float tMax, RNG &rng,
                        const SampledWavelengths &lambda,
                        ScratchBuffer &scratchBuffer) const {
        MediumSample mediumSample;

        // Sample a channel and distance along the ray
        SampledSpectrum sigma_t = sigma_a_spec.Sample(lambda) +
            sigma_s_spec.Sample(lambda);
        int channel = rng.Uniform<int>(NSpectrumSamples);
        if (sigma_t[channel] == 0) {
            mediumSample.Tr = SampledSpectrum(1);
            return mediumSample;
        }
        Float dist = -std::log(1 - rng.Uniform<Float>()) / sigma_t[channel];
        Float t = std::min(dist / Length(ray.d), tMax);
        bool sampledMedium = t < tMax;
        if (sampledMedium)
            mediumSample.intr =
                MediumInteraction(ray(t), -ray.d, ray.time, this,
                                  scratchBuffer.Alloc<HenyeyGreensteinPhaseFunction>(g));

        // Compute the transmittance and sampling density
        SampledSpectrum Tr = Exp(-sigma_t * std::min(t, MaxFloat) * Length(ray.d));

        // Return weighting factor for scattering from homogeneous medium
        SampledSpectrum density = sampledMedium ? (sigma_t * Tr) : Tr;
        Float pdf = density.Average();
        if (pdf == 0) {
            CHECK(!Tr);
            pdf = 1;
        }

        mediumSample.Tr =
            sampledMedium ? (Tr * sigma_s_spec.Sample(lambda) / pdf) : (Tr / pdf);
        return mediumSample;
    }

    PBRT_CPU_GPU
    pstd::optional<NewMediumSample> SampleTmaj(const Ray &ray, Float tMax, Float u,
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
                      SampledGrid<Float> densityGrid, SampledGrid<Float> LeScaleGrid,
                      Allocator alloc);

    static GridDensityMedium *Create(const ParameterDictionary &parameters,
                                     const Transform &worldFromMedium, const FileLoc *loc,
                                     Allocator alloc);

    PBRT_CPU_GPU
    MediumSample Sample(const Ray &rWorld, Float raytMax, RNG &rng,
                        const SampledWavelengths &lambda,
                        ScratchBuffer &scratchBuffer) const {
        // CO    ++nSampleCalls;

        raytMax *= Length(rWorld.d);
        Ray ray = mediumFromWorld(Ray(rWorld.o, Normalize(rWorld.d)), &raytMax);
        // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium
        // bounds
        const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
        Float tMin, tMax;
        if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return MediumSample{SampledSpectrum(1.f)};

        // Run delta-tracking iterations to sample a medium interaction
        bool foundInteraction = false;
        // For now...
        Float sigma_t = sigma_a_spec.MaxValue() + sigma_s_spec.MaxValue();
        MediumSample mediumSample;

        TraverseOctree(
            &densityOctree, ray.o, ray.d, raytMax,
            [&](const OctreeNode &node, Float t, Float t1) {
                if (node.maxDensity == 0)
                    // Empty--skip it!
                    return OctreeTraversal::Continue;

                DCHECK_RARE(1e-5, densityGrid.Lookup(ray((t + t1) / 2)) > node.maxDensity);
                while (true) {
                    // CO                ++nSampleSteps;
                    t +=
                        -std::log(1 - rng.Uniform<Float>()) / (sigma_t * node.maxDensity);

                    if (t >= t1)
                        // exited this cell w/o a scattering event
                        return OctreeTraversal::Continue;

                    if (t >= tMax)
                        // Nothing before the geom intersection; get out of here
                        return OctreeTraversal::Abort;

                    if (densityGrid.Lookup(ray(t)) > rng.Uniform<Float>() * node.maxDensity) {
                        // Populate _mi_ with medium interaction information and
                        // return
                        PhaseFunctionHandle phase =
                            scratchBuffer.Alloc<HenyeyGreensteinPhaseFunction>(g);
                        mediumSample.intr = MediumInteraction(rWorld(t), -rWorld.d,
                                                              rWorld.time, this, phase);
                        foundInteraction = true;
                        return OctreeTraversal::Abort;
                    }
                }
            });

        mediumSample.Tr = foundInteraction ? sigma_s_spec.Sample(lambda) / sigma_t
                                           : SampledSpectrum(1.f);
        return mediumSample;
    }

    PBRT_CPU_GPU
    SampledSpectrum Tr(const Ray &rWorld, Float raytMax, const SampledWavelengths &lambda,
                       RNG &rng) const {
        // CO    ++nTrCalls;

        raytMax *= Length(rWorld.d);
        Ray ray = mediumFromWorld(Ray(rWorld.o, Normalize(rWorld.d)), &raytMax);
        // Compute $[\tmin, \tmax]$ interval of _ray_'s overlap with medium
        // bounds
        const Bounds3f b(Point3f(0, 0, 0), Point3f(1, 1, 1));
        Float tMin, tMax;
        if (!b.IntersectP(ray.o, ray.d, raytMax, &tMin, &tMax))
            return SampledSpectrum(1.f);

        SampledSpectrum sigma_a = sigma_a_spec.Sample(lambda);
        SampledSpectrum sigma_s = sigma_s_spec.Sample(lambda);
        SampledSpectrum sigma_t = sigma_a + sigma_s;

        // Perform ratio tracking to estimate the transmittance value
        SampledSpectrum Tr(1.f);
        TraverseOctree(&densityOctree, ray.o, ray.d, raytMax,
                       [&](const OctreeNode &node, Float t, Float t1) {
                           if (node.maxDensity == 0)
                               // Empty--skip it!
                               return OctreeTraversal::Continue;

                           DCHECK_GE(t1, 0);

                           CHECK_GE(t, .999 * tMin);

                           // ratio tracking
                           Float sigma_bar = node.maxDensity * sigma_t.MaxComponentValue();
                           DCHECK_GE(sigma_bar, 0);
                           if (sigma_bar == 0)
                               return OctreeTraversal::Continue;

                           while (true) {
                               // CO                ++nTrSteps;
                               t += -std::log(1 - rng.Uniform<Float>()) / sigma_bar;
                               if (t >= t1)
                                   // exited node; keep going
                                   return OctreeTraversal::Continue;

                               if (t >= tMax)
                                   // past hit point. stop
                                   return OctreeTraversal::Abort;

                               Float density = densityGrid.Lookup(ray(t));
                               CHECK_RARE(1e-9, density < 0);
                               density = std::max<Float>(density, 0);

                               // FIXME: if sigma_bar isn't a majorant, then is this clamp
                               // wrong???
                               Tr *= 1 - Clamp(density * sigma_t / sigma_bar, 0, 1);
                               CHECK(Tr.MaxComponentValue() != Infinity);
                               CHECK(!Tr.HasNaNs());
                               Float Tr_max = Tr.MaxComponentValue();
                               if (Tr_max < 1) {
                                   Float q = 1 - Tr_max;
                                   if (rng.Uniform<Float>() < q) {
                                       Tr = SampledSpectrum(0.f);
                                       return OctreeTraversal::Abort;
                                   }
                                   Tr /= 1 - q;
                                   CHECK(Tr.MaxComponentValue() != Infinity);
                                   CHECK(!Tr.HasNaNs());
                               }
                           }
                       });

        CHECK(Tr.MaxComponentValue() != Infinity);
        CHECK(!Tr.HasNaNs());
        return Tr;
    }

    PBRT_CPU_GPU
    pstd::optional<NewMediumSample> SampleTmaj(const Ray &ray, Float tMax, Float u,
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

    SampledGrid<Float> densityGrid, LeScaleGrid;
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

inline SampledSpectrum MediumHandle::Tr(const Ray &ray, Float tMax,
                                        const SampledWavelengths &lambda,
                                        RNG &rng) const {
    auto tr = [&](auto ptr) { return ptr->Tr(ray, tMax, lambda, rng); };
    return Apply<SampledSpectrum>(tr);
}

inline MediumSample MediumHandle::Sample(const Ray &ray, Float tMax, RNG &rng,
                                         const SampledWavelengths &lambda,
                                         ScratchBuffer &scratchBuffer) const {
    auto sample = [&](auto ptr) { return ptr->Sample(ray, tMax, rng, lambda, scratchBuffer); };
    return Apply<MediumSample>(sample);
}

inline pstd::optional<NewMediumSample> MediumHandle::SampleTmaj(
    const Ray &ray, Float tMax, Float u, const SampledWavelengths &lambda,
    ScratchBuffer *scratchBuffer) const {
    auto sampletn = [&](auto ptr) { return ptr->SampleTmaj(ray, tMax, u, lambda,
                                                           scratchBuffer); };
    return Apply<pstd::optional<NewMediumSample>>(sampletn);
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_GRID_H
