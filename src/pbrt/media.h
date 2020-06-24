
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

#ifndef PBRT_MEDIA_HOMOGENEOUS_H
#define PBRT_MEDIA_HOMOGENEOUS_H

// media/homogeneous.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/transform.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/scattering.h>

#include <memory>
#include <vector>

namespace pbrt {

bool GetMediumScatteringProperties(const std::string &name, SpectrumHandle *sigma_a,
                                   SpectrumHandle *sigma_s, Allocator alloc);

// Media Inline Functions

// HenyeyGreenstein Declarations
class HenyeyGreenstein final : public PhaseFunction {
  public:
    // HenyeyGreenstein Public Methods
    PBRT_HOST_DEVICE
    HenyeyGreenstein(Float g) : g(g) {}

    PBRT_HOST_DEVICE
    Float p(const Vector3f &wo, const Vector3f &wi) const {
        ProfilerScope _(ProfilePhase::PhaseFuncEvaluation);
        return EvaluateHenyeyGreenstein(Dot(wo, wi), g);
    }

    PBRT_HOST_DEVICE
    pstd::optional<PhaseFunctionSample> Sample_p(const Vector3f &wo,
                                                 const Point2f &u) const {
        ProfilerScope _(ProfilePhase::PhaseFuncSampling);
        Float pdf;
        Vector3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};  // sampling is exact.
    }

    PBRT_HOST_DEVICE
    Float PDF(const Vector3f &wo, const Vector3f &wi) const {
        return p(wo, wi);
    }

    std::string ToString() const;

  private:
    Float g;
};

// HomogeneousMedium Declarations
class HomogeneousMedium : public Medium {
  public:
    // HomogeneousMedium Public Methods
    HomogeneousMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, Float g)
        : sigma_a(sigma_a),
          sigma_s(sigma_s),
          g(g) {}
    static HomogeneousMedium *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                     Allocator alloc);

    SampledSpectrum Tr(const Ray &ray, Float tMax, const SampledWavelengths &lambda,
                       SamplerHandle sampler) const;
    SampledSpectrum Sample(const Ray &ray, Float tMax, SamplerHandle sampler,
                           const SampledWavelengths &lambda,
                           MemoryArena &arena,
                           MediumInteraction *mi) const;
    std::string ToString() const;

  private:
    // HomogeneousMedium Private Data
    SpectrumHandle sigma_a, sigma_s;
    Float g;
};


// GridDensityMedium Declarations
class GridDensityMedium : public Medium {
  public:
    // GridDensityMedium Public Methods
    GridDensityMedium(SpectrumHandle sigma_a, SpectrumHandle sigma_s, Float g,
                      int nx, int ny, int nz, const Transform &worldFromMedium,
                      std::vector<Float> density, Allocator alloc);
    static GridDensityMedium *Create(const ParameterDictionary &dict,
                                     const Transform &worldFromMedium, const FileLoc *loc,
                                     Allocator alloc);

    Float Density(const Point3f &p) const;
    Float D(const Point3i &p) const {
        Bounds3i sampleBounds(Point3i(0, 0, 0), Point3i(nx, ny, nz));
        if (!InsideExclusive(p, sampleBounds)) return 0;
        return density[(p.z * ny + p.y) * nx + p.x];
    }
    SampledSpectrum Sample(const Ray &ray, Float tMax, SamplerHandle sampler,
                           const SampledWavelengths &lambda,
                           MemoryArena &arena,
                           MediumInteraction *mi) const;
    SampledSpectrum Tr(const Ray &ray, Float tMax, const SampledWavelengths &lambda,
                       SamplerHandle sampler) const;

    std::string ToString() const;

  private:
    struct OctreeNode {
        Float minDensity, maxDensity;    //  unused for interior nodes
        pstd::array<OctreeNode *, 8> *children = nullptr;
        Bounds3f bounds;

        OctreeNode *&child(int n) { CHECK(children); return (*children)[n]; }
        const OctreeNode *child(int n) const { return children ? (*children)[n] : nullptr; }
    };

    void buildOctree(OctreeNode *node, Allocator alloc,
                     const Bounds3f &bounds, int depth);
    void simplifyOctree(OctreeNode *node, const Bounds3f &bounds, Float SE,
                        Float sigma_t);

    // GridDensityMedium Private Data
    SpectrumHandle sigma_a_spec, sigma_s_spec;
    Float g;
    int nx, ny, nz;
    Transform mediumFromWorld, worldFromMedium;
    std::vector<Float> density;
    OctreeNode densityOctree;
};

}  // namespace pbrt

#endif  // PBRT_MEDIA_GRID_H
