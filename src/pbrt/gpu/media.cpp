// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/gpu/pathintegrator.h>

#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/media.h>

namespace pbrt {

void GPUPathIntegrator::SampleMediumInteraction(int depth) {
    GPUDo("Reset medium queues", mediumSampleQueue->Reset(); mediumEvalQueue->Reset(););

    GPUParallelFor("Initialize mediumSampleQueue", pixelsPerPass,
                   [=] __device__(PathRayIndex rayIndex) {
                       if (rayIndex >= *numActiveRays)
                           return;

                       Ray ray = pathRayQueue->GetRay(rayIndex);
                       if (ray.medium)
                           mediumSampleQueue->Add(rayIndex);
                   });

    GPUParallelFor("Sample medium interaction", pixelsPerPass,
                   [=] __device__(MediumEvalIndex mediumIndex) {
                       if (mediumIndex >= mediumSampleQueue->Size())
                           return;

                       PathRayIndex rayIndex = mediumSampleQueue->Get(mediumIndex);
                       Float tMax;
                       Ray ray = pathRayQueue->GetRay(rayIndex, &tMax);

                       PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
                       RNG &rng = *rngs[pixelIndex];
                       const SampledWavelengths &lambda = lambdas[pixelIndex];
                       ScratchBuffer &scratchBuffer = scratchBuffers[pixelIndex];
                       PathState &pathState = pathStates[pixelIndex];
                       SampledSpectrum &beta = pathState.beta;
                       SampledSpectrum &pdfUni = pathState.pdfUni;
                       SampledSpectrum &pdfNEE = pathState.pdfNEE;
                       SampledSpectrum &L = pathState.L;

                       while (true) {
                           // Note: mostly cut and pasted from VolPathIntegrator.
                           // Share fragments??
                           Float u = rng.Uniform<Float>();
                           MediumSample mediumSample =
                               ray.medium.Sample_Tmaj(ray, tMax, u, lambda, &scratchBuffer);

                           // Handle an interaction with a medium or a surface
                           if (!mediumSample.intr) {
                               // FIXME: review this, esp the pdf...
                               beta *= mediumSample.Tmaj;
                               pdfUni *= mediumSample.Tmaj;
                               break;
                           }

                           const MediumInteraction &intr = *mediumSample.intr;
                           const SampledSpectrum &sigma_a = intr.sigma_a;
                           const SampledSpectrum &sigma_s = intr.sigma_s;
                           const SampledSpectrum &Tmaj = mediumSample.Tmaj;

                           if (depth < maxDepth)
                               // FIXME: beta?
                               L += intr.Le * sigma_a / intr.sigma_maj[0];

                           Float pAbsorb = sigma_a[0] / intr.sigma_maj[0];
                           Float pScatter = sigma_s[0] / intr.sigma_maj[0];
                           Float pNull = std::max<Float>(0, 1 - pAbsorb - pScatter);

                           Float um = rng.Uniform<Float>();
                           int mode = SampleDiscrete({pAbsorb, pScatter, pNull}, um);

                           if (mode == 0) {
                               // absorption; done
                               pathState.beta = SampledSpectrum(0.f);
                               return;
                           } else if (mode == 1) {
#if 0
                               if (depth >= maxDepth)
                                   return L;
#endif
                               // scatter
                               beta *= Tmaj * sigma_s;
                               pdfUni *= Tmaj * sigma_s;

                               interactionType[depth & 1][rayIndex] = InteractionType::Medium;
                               mediumInteractions[depth & 1][pixelIndex] = intr;
                               mediumEvalQueue->Add(rayIndex);
                               return;
                           } else {
                               // null scatter
                               SampledSpectrum sigma_n = intr.sigma_n();

                               beta *= Tmaj * sigma_n;
                               pdfUni *= Tmaj * sigma_n;
                               pdfNEE *= Tmaj * intr.sigma_maj;

                               tMax -= mediumSample.t;
                               ray = intr.SpawnRay(ray.d);
                           }

                           // Avoid overflow...
                           if (beta.MaxComponentValue() > 0x1p24f ||
                               pdfUni.MaxComponentValue() > 0x1p24f ||
                               pdfNEE.MaxComponentValue() > 0x1p24f) {
                               beta *= 1.f / 0x1p24f;
                               pdfUni *= 1.f / 0x1p24f;
                               pdfNEE *= 1.f / 0x1p24f;
                           }
                       }
                   });
}

void GPUPathIntegrator::HandleMediumTransitions(int depth) {
    // Handle null-material intersections
    GPUParallelFor(
        "Handle medium transitions", pixelsPerPass,
        [=] __device__(PathRayIndex rayIndex) {
            if (rayIndex >= *numActiveRays)
                return;

            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            if (interactionType[depth & 1][rayIndex] == InteractionType::Surface &&
                intersection.material == nullptr) {
                // Note: use SurfaceInteraction::SkipIntersection()
                // if we switch to ray differentials...
                Vector3f w = -intersection.wo;
                Ray ray(intersection.OffsetRayOrigin(w), w, 0.f /*time*/,
                        intersection.GetMedium(w));

                // Note: do *not* set pathState.bsdfPDF or
                // bsdfFlags; those should carry through unchanged
                // from the last actual scattering event!

                PathRayIndex newRayIndex = pathRayQueue->Add(ray, 1e20f);
                pixelIndexToRayIndex[pixelIndex] = newRayIndex;
                rayIndexToPixelIndex[(depth & 1) ^ 1][newRayIndex] = pixelIndex;
            }
        });
}

}  // namespace pbrt
