// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_GPU_DIRECT_H
#define PBRT_GPU_DIRECT_H

#include <pbrt/pbrt.h>

#include <pbrt/bsdf.h>
#include <pbrt/bxdfs.h>
#include <pbrt/gpu/accel.h>  // rayqueue
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/samplers.h>
#include <pbrt/util/check.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

template <typename BxDF>
inline void GPUPathIntegrator::SampleDirect(int depth) {
    int bxdfTag = BxDFHandle::TypeIndex<BxDF>();
    std::string description =
        StringPrintf("Sample direct - %s", BxDFTraits<BxDF>::name());

    GPUParallelFor(
        description.c_str(), pixelsPerPass, [=] __device__(BxDFEvalIndex bsdfEvalIndex) {
            // IMPORTANT: must not read rayo/rayd in this kernel...
            if (bsdfEvalIndex >= bxdfEvalQueues->Size(bxdfTag))
                return;

            PathRayIndex rayIndex = bxdfEvalQueues->Get(bxdfTag, bsdfEvalIndex);
            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            const SampledWavelengths &lambda = lambdas[pixelIndex];
            PathState *pathState = &pathStates[pixelIndex];

            SampledSpectrum beta = pathState->beta;
            if (!beta)
                return;

            const pstd::optional<SampledLight> &sampledLight = pathState->sampledLight;
            if (!sampledLight)
                return;

            LightHandle light = sampledLight->light;
            Float lightChoicePDF = sampledLight->pdf;

            SamplerHandle sampler = samplers[pixelIndex];
            Point2f u = sampler.Get2D();

            pstd::optional<LightLiSample> ls =
                light.Sample_Li(intersection, u, lambda, LightSamplingMode::WithMIS);
            if (!ls || ls->pdf == 0 || !ls->L)
                return;

            BSDF *bsdf = intersection.bsdf;
            Vector3f wo = intersection.wo;
            SampledSpectrum f = bsdf->f<BxDF>(wo, ls->wi);
            if (!f)
                return;

            beta *= f * AbsDot(ls->wi, intersection.shading.n);

            Ray ray = intersection.SpawnRayTo(ls->pLight);
            ShadowRayIndex shadowRayIndex = shadowRayQueue->Add(ray, 1 - ShadowEpsilon);
            shadowRayIndexToPixelIndex[shadowRayIndex] = pixelIndex;

            Float lightPDF = ls->pdf * lightChoicePDF;
            // This causes pdfUni to be zero for the shadow ray, so that
            // part of MIS just becomes a no-op.
            Float bsdfPDF = IsDeltaLight(light.Type()) ? 0.f :
                bsdf->PDF<BxDF>(wo, ls->wi);

            shadowRayLd[shadowRayIndex] = beta * ls->L;
            shadowRayPDFLight[shadowRayIndex] = pathState->pdfUni * lightPDF;
            shadowRayPDFUni[shadowRayIndex] = pathState->pdfUni * bsdfPDF;
        });
}

}  // namespace pbrt

#endif  // PBRT_GPU_DIRECT_H
