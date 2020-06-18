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
    Float sigma_nt, Tn;

    PBRT_CPU_GPU
    SampledSpectrum sigma_n() const {
        SampledSpectrum sigma_t = sigma_a + sigma_s;
        SampledSpectrum sigma_n = SampledSpectrum(sigma_nt) - sigma_t;
        CHECK_RARE(1e-5, sigma_n.MinComponentValue() < 0);
        return ClampZero(sigma_n);
    }

    std::string ToString() const;
};

// HomogeneousMedium Declarations
class alignas(8) HomogeneousMedium {
  public:
    // HomogeneousMedium Public Methods
    HomogeneousMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, Float g,
                      Allocator alloc)
        : sigma_a_spec(sigma_a, alloc), sigma_s_spec(sigma_s, alloc), phase(g), g(g) {}

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
    pstd::optional<NewMediumSample> SampleTn(const Ray &ray, Float tMax, Float u,
                                             const SampledWavelengths &lambda,
                                             ScratchBuffer *scratchBuffer) const;

    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    HenyeyGreensteinPhaseFunction phase;
    Float g;
};

// GridDensityMedium Declarations
class alignas(8) GridDensityMedium {
  public:
    // GridDensityMedium Public Methods
    GridDensityMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, Float g, int nx,
                      int ny, int nz, const Transform &worldFromMedium,
                      std::vector<Float> density, Allocator alloc);

    static GridDensityMedium *Create(const ParameterDictionary &parameters,
                                     const Transform &worldFromMedium, const FileLoc *loc,
                                     Allocator alloc);

    PBRT_CPU_GPU
    Float D(const Point3i &p) const {
        Bounds3i sampleBounds(Point3i(0, 0, 0), Point3i(nx, ny, nz));
        if (!InsideExclusive(p, sampleBounds))
            return 0;
        return density[(p.z * ny + p.y) * nx + p.x];
    }

    PBRT_CPU_GPU
    Float Density(const Point3f &p) const {
        // Compute voxel coordinates and offsets for _p_
        Point3f pSamples(p.x * nx - .5f, p.y * ny - .5f, p.z * nz - .5f);
        Point3i pi = (Point3i)Floor(pSamples);
        Vector3f d = pSamples - (Point3f)pi;

        // Trilinearly interpolate density values to compute local density
        Float d00 = Lerp(d.x, D(pi), D(pi + Vector3i(1, 0, 0)));
        Float d10 = Lerp(d.x, D(pi + Vector3i(0, 1, 0)), D(pi + Vector3i(1, 1, 0)));
        Float d01 = Lerp(d.x, D(pi + Vector3i(0, 0, 1)), D(pi + Vector3i(1, 0, 1)));
        Float d11 = Lerp(d.x, D(pi + Vector3i(0, 1, 1)), D(pi + Vector3i(1, 1, 1)));
        Float d0 = Lerp(d.y, d00, d10);
        Float d1 = Lerp(d.y, d01, d11);
        return Lerp(d.z, d0, d1);
    }

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

                DCHECK_RARE(1e-5, Density(ray((t + t1) / 2)) > node.maxDensity);
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

                    if (Density(ray(t)) > rng.Uniform<Float>() * node.maxDensity) {
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

                           // Residual tracking. First, account for the constant part.
                           if (node.minDensity > 0) {
                               Float dt = std::min(t1, tMax) - t;
                               Tr *= Exp(-dt * node.minDensity * sigma_t);
                               CHECK(Tr.MaxComponentValue() != Infinity);
                               CHECK(!Tr.HasNaNs());
                           }

                           // Now do ratio tracking through the residual volume.
                           Float sigma_bar = (node.maxDensity - node.minDensity) *
                                             sigma_t.MaxComponentValue();
                           DCHECK_GE(sigma_bar, 0);
                           if (sigma_bar == 0)
                               // There's no residual; go on to the next octree node.
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

                               Float density = Density(ray(t)) - node.minDensity;
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
    pstd::optional<NewMediumSample> SampleTn(const Ray &ray, Float tMax, Float u,
                                             const SampledWavelengths &lambda,
                                             ScratchBuffer *scratchBuffer) const;

    std::string ToString() const;

  private:
    struct OctreeNode {
        Float minDensity, maxDensity;  //  unused for interior nodes
        pstd::array<OctreeNode *, 8> *children = nullptr;
        Bounds3f bounds;

        PBRT_CPU_GPU
        OctreeNode *&child(int n) {
            CHECK(children);
            return (*children)[n];
        }

        PBRT_CPU_GPU
        const OctreeNode *child(int n) const {
            return children ? (*children)[n] : nullptr;
        }
    };

    void buildOctree(OctreeNode *node, Allocator alloc, const Bounds3f &bounds,
                     int depth);
    void simplifyOctree(OctreeNode *node, const Bounds3f &bounds, Float SE,
                        Float sigma_t);

    // GridDensityMedium Private Data
    DenselySampledSpectrum sigma_a_spec, sigma_s_spec;
    HenyeyGreensteinPhaseFunction phase;
    Float g;
    int nx, ny, nz;
    Transform mediumFromWorld, worldFromMedium;
    pstd::vector<Float> density;
    OctreeNode densityOctree;
    pstd::pmr::monotonic_buffer_resource treeBufferResource;
};

inline Float PhaseFunctionHandle::p(const Vector3f &wo, const Vector3f &wi) const {
    switch (Tag()) {
    case TypeIndex<HenyeyGreensteinPhaseFunction>():
        return Cast<HenyeyGreensteinPhaseFunction>()->p(wo, wi);
    default:
        LOG_FATAL("Unhandled phase function");
        return {};
    }
}

inline pstd::optional<PhaseFunctionSample> PhaseFunctionHandle::Sample_p(
    const Vector3f &wo, const Point2f &u) const {
    switch (Tag()) {
    case TypeIndex<HenyeyGreensteinPhaseFunction>():
        return Cast<HenyeyGreensteinPhaseFunction>()->Sample_p(wo, u);
    default:
        LOG_FATAL("Unhandled phase function");
        return {};
    }
}

inline Float PhaseFunctionHandle::PDF(const Vector3f &wo, const Vector3f &wi) const {
    switch (Tag()) {
    case TypeIndex<HenyeyGreensteinPhaseFunction>():
        return Cast<HenyeyGreensteinPhaseFunction>()->PDF(wo, wi);
    default:
        LOG_FATAL("Unhandled phase function");
        return {};
    }
}

inline SampledSpectrum MediumHandle::Tr(const Ray &ray, Float tMax,
                                        const SampledWavelengths &lambda,
                                        RNG &rng) const {
    switch (Tag()) {
    case TypeIndex<HomogeneousMedium>():
        return Cast<HomogeneousMedium>()->Tr(ray, tMax, lambda, rng);
    case TypeIndex<GridDensityMedium>():
        return Cast<GridDensityMedium>()->Tr(ray, tMax, lambda, rng);
    default:
        LOG_FATAL("Unhandled medium");
        return {};
    }
}

inline MediumSample MediumHandle::Sample(const Ray &ray, Float tMax, RNG &rng,
                                         const SampledWavelengths &lambda,
                                         ScratchBuffer &scratchBuffer) const {
    switch (Tag()) {
    case TypeIndex<HomogeneousMedium>():
        return Cast<HomogeneousMedium>()->Sample(ray, tMax, rng, lambda, scratchBuffer);
    case TypeIndex<GridDensityMedium>():
        return Cast<GridDensityMedium>()->Sample(ray, tMax, rng, lambda, scratchBuffer);
    default:
        LOG_FATAL("Unhandled medium");
        return {};
    }
}

inline pstd::optional<NewMediumSample> MediumHandle::SampleTn(
    const Ray &ray, Float tMax, Float u, const SampledWavelengths &lambda,
    ScratchBuffer *scratchBuffer) const {
    switch (Tag()) {
    case TypeIndex<HomogeneousMedium>():
        return Cast<HomogeneousMedium>()->SampleTn(ray, tMax, u, lambda, scratchBuffer);
    case TypeIndex<GridDensityMedium>():
        return Cast<GridDensityMedium>()->SampleTn(ray, tMax, u, lambda, scratchBuffer);
    default:
        LOG_FATAL("Unhandled medium");
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_MEDIA_GRID_H