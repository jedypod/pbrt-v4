// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_GPU_INDIRECT_H
#define PBRT_GPU_INDIRECT_H

#include <pbrt/pbrt.h>

#include <pbrt/bsdf.h>
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
inline void GPUPathIntegrator::SampleIndirect(int depth, bool overrideRay) {
    int bxdfTag = BxDFHandle::TypeIndex<BxDF>();
    std::string description =
        StringPrintf("Sample indirect - %s", BxDFTraits<BxDF>::name());

    GPUParallelFor(
        description.c_str(), pixelsPerPass, [=] __device__(BxDFEvalIndex bsdfEvalIndex) {
            // IMPORTANT: must not read rayo/rayd in this kernel...
            if (bsdfEvalIndex >= bxdfEvalQueues->Size(bxdfTag))
                return;

            PathRayIndex rayIndex = bxdfEvalQueues->Get(bxdfTag, bsdfEvalIndex);
            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            PathState *pathState = &pathStates[pixelIndex];

            SampledSpectrum beta = pathState->beta;
            if (!beta)
                return;

            BSDF *bsdf = intersection.bsdf;
            Vector3f wo = intersection.wo;

            SamplerHandle sampler = samplers[pixelIndex];
            Point2f u = sampler.Get2D();
            Float uc = sampler.Get1D();

            pstd::optional<BSDFSample> bs = bsdf->Sample_f<BxDF>(wo, uc, u);
            if (!bs || !bs->f || bs->pdf == 0) {
                pathState->beta = SampledSpectrum(0);
                return;
            }
            beta *= bs->f * (AbsDot(intersection.shading.n, bs->wi) / bs->pdf);
            DCHECK(!beta.HasNaNs());

            pathState->scatteringPDF =
                bsdf->SampledPDFIsProportional() ? bsdf->PDF(wo, bs->wi) : bs->pdf;
            pathState->bsdfFlags = bs->flags;

            if (regularize && !bs->IsSpecular())
                pathState->anyNonSpecularBounces = true;

            if (bs->IsTransmission())
                // Update the term that tracks radiance scaling for refraction.
                pathState->etaScale *= Sqr(bsdf->eta);

            // russian roulette
            SampledSpectrum rrBeta = beta * pathState->etaScale;
            if (rrBeta.MaxComponentValue() < 1 && depth > 3) {
                Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                if (sampler.Get1D() < q) {
                    pathState->beta = SampledSpectrum(0);
                    return;
                }
                beta /= 1 - q;
                DCHECK(!beta.HasNaNs());
            }

            Ray ray = intersection.SpawnRay(bs->wi);
            if (overrideRay) {
                PathRayIndex newRayIndex = pixelIndexToRayIndex[pixelIndex];
                pathRayQueue->SetRay(newRayIndex, ray, 1e20f);
            } else {
                PathRayIndex newRayIndex = pathRayQueue->Add(ray, 1e20f);
                pixelIndexToRayIndex[pixelIndex] = newRayIndex;
                rayIndexToPixelIndex[(depth & 1) ^ 1][newRayIndex] = pixelIndex;
            }

            pathState->beta = beta;
        });
}

}  // namespace pbrt

#endif  // PBRT_GPU_INDIRECT_H
