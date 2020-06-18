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

                       MediumSample mediumSample =
                           ray.medium.Sample(ray, tMax, rng, lambda, scratchBuffer);

                       PathState &pathState = pathStates[pixelIndex];
                       pathState.beta *= mediumSample.Tr;

                       if (mediumSample.intr) {
                           interactionType[rayIndex] = InteractionType::Medium;
                           mediumInteractions[depth & 1][pixelIndex] = *mediumSample.intr;
                           mediumEvalQueue->Add(rayIndex);
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
            if (interactionType[rayIndex] == InteractionType::Surface &&
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
