
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

#ifndef PBRT_UTIL_PROFILE_H
#define PBRT_UTIL_PROFILE_H

// util/profile.h*
#include <pbrt/pbrt.h>

#include <stdio.h>
#include <string>

namespace pbrt {

enum class ProfilePhase {
    SceneConstruction,
    ParsingAndGenScene,
    CPUSceneConstruction,
    ShapeConstruction,
    LightConstruction,
    AccelConstruction,
    BVHInitialBound,
    BVHFindBestSplit,
    PLYLoading,
    TextureLoading,
    MIPMapCreation,

    IntegratorRender,
    RayIntegratorLi,
    SPPMCameraPass,
    SPPMGridConstruction,
    SPPMPhotonPass,
    SPPMStatsUpdate,
    BDPTGenerateSubpath,
    BDPTConnectSubpaths,
    RISCameraPass,
    RISResamplePass,
    RISFinalPass,
    DirectLighting,
    LightDistribLookup,
    LightDistribSpinWait,
    LightDistribCreation,
    BSDFEvaluation,
    BSDFSampling,
    BSDFPdf,
    BSSRDFEvaluation,
    BSSRDFSampling,
    PhaseFuncEvaluation,
    PhaseFuncSampling,
    AccelIntersect,
    AccelIntersectP,
    LightSample,
    LightPDF,
    MediumSample,
    MediumTr,
    TriIntersect,
    TriIntersectP,
    CurveIntersect,
    CurveIntersectP,
    ShapeIntersect,
    ShapeIntersectP,
    GetBSDF,
    GetBSSRDF,
    GenerateCameraRay,
    MergeFilmTile,
    SplatFilm,
    AddFilmSample,
    StartPixelSample,
    GetSample,
    TexFiltBasic,
    TexFiltEWA,
    TexFiltPtex,
    RGBToSpectrum,
    NProfCategories
};

static_assert((int)ProfilePhase::NProfCategories <= 64,
              "No more than 64 profiling categories may be defined.");

inline uint64_t ProfilePhaseToBits(ProfilePhase p) { return 1ull << (int)p; }
ProfilePhase BitsToProfilePhase(uint64_t b);

static const char *ProfNames[] = {
    "Scene creation",
    "Parsing and GeneralScene creation",
    "CPU-specialized scene creation",
    "Shape creation",
    "Light creation",
    "Acceleration structure creation",
    "BVH initial bound",
    "BVH find best split",
    "PLY file loading",
    "Texture loading",
    "MIP map generation",

    "Integrator::Render()",
    "RayIntegrator::Li()",
    "SPPM camera pass",
    "SPPM grid construction",
    "SPPM photon pass",
    "SPPM photon statistics update",
    "BDPT subpath generation",
    "BDPT subpath connections",
    "RIS camera pass",
    "RIS resampling pass",
    "RIS final pass",
    "Direct lighting",
    "LightSampler lookup",
    "LightSampler spin wait",
    "LightSampler creation",
    "BSDF::f()",
    "BSDF::Sample_f()",
    "BSDF::PDF()",
    "BSSRDF::f()",
    "BSSRDF::Sample_f()",
    "PhaseFunction::p()",
    "PhaseFunction::Sample_p()",
    "Accelerator::Intersect()",
    "Accelerator::IntersectP()",
    "Light::Sample_*()",
    "Light::PDF()",
    "Medium::Sample()",
    "Medium::Tr()",
    "Triangle::Intersect()",
    "Triangle::IntersectP()",
    "Curve::Intersect()",
    "Curve::IntersectP()",
    "Other Shape::Intersect()",
    "Other Shape::IntersectP()",
    "Material::GetBSDF()",
    "Material::GetBSSRDF()",
    "Camera::GenerateRay[Differential]()",
    "Film::MergeTile()",
    "Film::AddSplat()",
    "Film::AddSample()",
    "Sampler::StartPixelSample()",
    "Sampler::GetSample[12]D()",
    "MIPMap::Lookup() (basic)",
    "MIPMap::Lookup() (EWA)",
    "Ptex lookup",
    "RGB to Spectrum",
};

static_assert((int)ProfilePhase::NProfCategories == PBRT_ARRAYSIZE(ProfNames),
              "ProfNames[] array and ProfilePhase enumerant have different "
              "numbers of entries!");

extern thread_local uint64_t ProfilerState;
inline uint64_t CurrentProfilerState() { return ProfilerState; }

class ProfilerScope {
  public:
    // ProfilerScope Public Methods
    PBRT_HOST_DEVICE
    ProfilerScope(ProfilePhase p) {
#ifndef __CUDA_ARCH__
        categoryBit = ProfilePhaseToBits(p);
        reset = (ProfilerState & categoryBit) == 0;
        ProfilerState |= categoryBit;
#endif // __CUDA_ARCH__
    }
    PBRT_HOST_DEVICE
    ~ProfilerScope() {
#ifndef __CUDA_ARCH__
        if (reset) ProfilerState &= ~categoryBit;
#endif // __CUDA_ARCH__
    }
    ProfilerScope(const ProfilerScope &) = delete;
    ProfilerScope &operator=(const ProfilerScope &) = delete;

    std::string ToString() const;

  private:
#ifndef __CUDA_ARCH__
    // ProfilerScope Private Data
    bool reset;
    uint64_t categoryBit;
#endif // __CUDA_ARCH__
};

void InitProfiler();
void SuspendProfiler();
void ResumeProfiler();
void ProfilerWorkerThreadInit();
void ReportProfilerResults(FILE *dest);
void ClearProfiler();
void CleanupProfiler();

}  // namespace pbrt

#endif  // PBRT_UTIL_PROFILE_H
