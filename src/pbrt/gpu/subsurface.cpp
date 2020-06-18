// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/pbrt.h>

#include <pbrt/bssrdf.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/direct.h>
#include <pbrt/gpu/indirect.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/interaction.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/samplers.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

namespace pbrt {

void GPUPathIntegrator::SampleSubsurface(int depth) {
    // Subsurface scattering
    GPUDo("Reset randomHitRayQueue", randomHitRayQueue->Reset(););

    GPUParallelFor(
        "Sample subsurface scattering", pixelsPerPass,
        [=] __device__(PathRayIndex rayIndex) {
            if (rayIndex >= *numActiveRays)
                return;

            if (interactionType[rayIndex] != InteractionType::Surface)
                return;

            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            PathState &pathState = pathStates[pixelIndex];

            if (!pathState.beta || !intersection.material || !intersection.bsdf ||
                !intersection.bssrdf || !IsTransmissive(pathState.bsdfFlags))
                return;

            SamplerHandle sampler = samplers[pixelIndex];
            pstd::optional<BSSRDFProbeSegment> probeSeg =
                intersection.bssrdf.Sample(sampler.Get1D(), sampler.Get2D());
            if (!probeSeg)
                return;

            // Enqueue ray
            Ray ray(probeSeg->p0, probeSeg->p1 - probeSeg->p0);
            SSRayIndex ssRayIndex = randomHitRayQueue->Add(ray, 1.f);
            subsurfaceMaterials[ssRayIndex] = intersection.material;
            subsurfaceRayIndexToPathRayIndex[ssRayIndex] = rayIndex;

            subsurfaceReservoirSamplers[ssRayIndex].Reset();
            uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
            subsurfaceReservoirSamplers[ssRayIndex].Seed(seed);
        });

    auto events =
        accel->IntersectOneRandom(randomHitRayQueue, pixelsPerPass, subsurfaceMaterials,
                                  subsurfaceReservoirSamplers);
    struct IsectRandomHack {};
    GetGPUKernelStats<IsectRandomHack>("Tracing subsurface scattering probe rays")
        .launchEvents.push_back(events);

    GPUParallelFor(
        "Incorporate subsurface S factor", pixelsPerPass,
        [=] __device__(SSRayIndex ssRayIndex) {
            if (ssRayIndex >= randomHitRayQueue->Size())
                return;

            PathRayIndex rayIndex =
                subsurfaceRayIndexToPathRayIndex[ssRayIndex];  // incident ray
            CHECK(interactionType[rayIndex] == InteractionType::Surface);

            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            PathState &pathState = pathStates[pixelIndex];
            if (!pathState.beta)
                return;

            WeightedReservoirSampler<SurfaceInteraction> &interactionSampler =
                subsurfaceReservoirSamplers[ssRayIndex];

            if (!interactionSampler.HasSample())
                return;

            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            ScratchBuffer &scratchBuffer = scratchBuffers[pixelIndex];
            BSSRDFSample bssrdfSample = intersection.bssrdf.ProbeIntersectionToSample(
                interactionSampler.GetSample(), scratchBuffer);

            if (!bssrdfSample.S || bssrdfSample.pdf == 0) {
                pathState.beta = SampledSpectrum(0.);
                return;
            }

            pathState.beta *=
                bssrdfSample.S * interactionSampler.WeightSum() / bssrdfSample.pdf;
            DCHECK(!pathState.beta.HasNaNs());

            if (regularize)
                pathState.anyNonSpecularBounces = true;

            intersection.pi = bssrdfSample.si.pi;
            intersection.dpdu = bssrdfSample.si.dpdu;
            intersection.dpdv = bssrdfSample.si.dpdv;
            intersection.dndu = bssrdfSample.si.dndu;
            intersection.dndv = bssrdfSample.si.dndv;
            intersection.n = bssrdfSample.si.n;
            intersection.uv = bssrdfSample.si.uv;
            intersection.wo = bssrdfSample.si.wo;
            intersection.shading = bssrdfSample.si.shading;
            intersection.bsdf = bssrdfSample.si.bsdf;  // important!

            bxdfEvalQueues->Add(intersection.bsdf->GetBxDF().Tag(), rayIndex);
        });

    GPUParallelFor(
        "Sample light after SSS", pixelsPerPass, [=] __device__(SSRayIndex ssRayIndex) {
            if (ssRayIndex >= randomHitRayQueue->Size())
                return;

            PathRayIndex rayIndex =
                subsurfaceRayIndexToPathRayIndex[ssRayIndex];  // incident ray
            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            PathState &pathState = pathStates[pixelIndex];
            if (!pathState.beta)
                return;

            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            SamplerHandle sampler = samplers[pixelIndex];

            pathState.sampledLight = lightSampler.Sample(intersection, sampler.Get1D());
        });

    SampleDirect<BSSRDFAdapter>(depth);

    TraceShadowRays();

    SampleIndirect<BSSRDFAdapter>(depth, true);
}

}  // namespace pbrt
