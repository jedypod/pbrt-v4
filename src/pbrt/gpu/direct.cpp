// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/gpu/direct.h>

#include <pbrt/lightsamplers.h>
#include <pbrt/media.h>

namespace pbrt {

static void SampleDirectRecursive(TypePack<> types, GPUPathIntegrator *integrator,
                                  int depth) {
    // done!
}

template <typename... Types>
static void SampleDirectRecursive(TypePack<Types...> types, GPUPathIntegrator *integrator,
                                  int depth) {
    using BxDF = typename GetFirst<TypePack<Types...>>::type;
    integrator->SampleDirect<BxDF>(depth);

    SampleDirectRecursive(typename RemoveFirst<TypePack<Types...>>::type(), integrator,
                          depth);
}

void GPUPathIntegrator::SampleDirect(int depth) {
    SampleDirectRecursive(typename BxDFHandle::Types(), this, depth);

    if (haveMedia) {
        GPUParallelFor(
            "Sample direct - phase function", pixelsPerPass,
            [=] __device__(MediumEvalIndex mediumEvalIndex) {
                if (mediumEvalIndex >= mediumEvalQueue->Size())
                    return;

                PathRayIndex rayIndex = mediumEvalQueue->Get(mediumEvalIndex);
                PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
                PathState &pathState = pathStates[pixelIndex];

                SampledSpectrum beta = pathState.beta;
                if (!beta)
                    return;

                const pstd::optional<SampledLight> &sampledLight = pathState.sampledLight;
                if (!sampledLight)
                    return;

                LightHandle light = sampledLight->light;
                Float lightChoicePDF = sampledLight->pdf;
                MediumInteraction &intr = mediumInteractions[depth & 1][pixelIndex];
                const SampledWavelengths &lambda = lambdas[pixelIndex];

                SamplerHandle sampler = samplers[pixelIndex];
                Point2f u = sampler.Get2D();

                pstd::optional<LightLiSample> ls =
                    light.Sample_Li(intr, u, lambda, LightSamplingMode::WithMIS);
                if (!ls || ls->pdf == 0 || !ls->L)
                    return;

                Float p = intr.phase.p(intr.wo, ls->wi);
                if (p == 0)
                    return;

                Float lightPDF = ls->pdf * lightChoicePDF;
                Float weight = 1;
                if (!IsDeltaLight(light.Type())) {
                    Float phasePDF = intr.phase.PDF(intr.wo, ls->wi);
                    weight = PowerHeuristic(1, lightPDF, 1, phasePDF);
                }

                Ray ray = intr.SpawnRayTo(ls->pLight);
                ShadowRayIndex shadowRayIndex =
                    shadowRayQueue->Add(ray, 1 - ShadowEpsilon);
                shadowRayIndexToPixelIndex[shadowRayIndex] = pixelIndex;

                SampledSpectrum Ld = beta * ls->L * p * weight / lightPDF;
                DCHECK(!Ld.HasNaNs());
                shadowRayLd[shadowRayIndex] = Ld;
            });
    }
}

void GPUPathIntegrator::SampleLight(int depth) {
    GPUParallelFor(
        "Choose Light to Sample", pixelsPerPass, [=] __device__(PathRayIndex rayIndex) {
            if (rayIndex >= *numActiveRays ||
                interactionType[rayIndex] == InteractionType::None)
                return;

            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            PathState &pathState = pathStates[pixelIndex];
            if (!pathState.beta)
                return;

            Interaction *intr = nullptr;
            if (interactionType[rayIndex] == InteractionType::Surface) {
                SurfaceInteraction &surfaceIntr = intersections[depth & 1][pixelIndex];
                if (surfaceIntr.material == nullptr ||
                    (surfaceIntr.bsdf->IsSpecular() &&
                     !surfaceIntr.bsdf->IsNonSpecular())) {  // TODO: is the second check
                                                             // needed?
                    pathState.sampledLight.reset();
                    return;
                } else
                    intr = &surfaceIntr;
            } else
                intr = &mediumInteractions[depth & 1][pixelIndex];

            SamplerHandle sampler = samplers[pixelIndex];
            pathState.sampledLight = lightSampler.Sample(*intr, sampler.Get1D());
        });
}

}  // namespace pbrt
