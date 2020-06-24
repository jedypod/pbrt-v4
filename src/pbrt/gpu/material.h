// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/pbrt.h>

#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/samplers.h>
#include <pbrt/textures.h>
#include <pbrt/util/check.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

template <typename TextureEvaluator>
void GPUPathIntegrator::EvaluateMaterial(TextureEvaluator texEval, int depth) {
    GPUParallelFor(
        "Bump and Material::GetBSDF/GetBSSRDF", pixelsPerPass,
        [=] __device__(PathRayIndex rayIndex) {
            if (rayIndex >= *numActiveRays)
                return;

            if (interactionType[depth & 1][rayIndex] != InteractionType::Surface)
                return;

            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            PathState &pathState = pathStates[pixelIndex];
            if (!pathState.beta)
                return;

            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            if (!intersection.material)
                // medium transition
                return;

            if (intersection.bsdf)
                // already successfully evaluated
                return;

            FloatTextureHandle displacement = intersection.material.GetDisplacement();
            if (displacement) {
                if (!texEval.CanEvaluate({displacement}, {}))
                    return;

                Vector3f dpdu, dpdv;
                Bump(texEval, displacement, intersection, &dpdu, &dpdv);
                intersection.SetShadingGeometry(Normal3f(Normalize(Cross(dpdu, dpdv))),
                                                dpdu, dpdv, Normal3f(0, 0, 0),
                                                Normal3f(0, 0, 0), false);
            }

            const SampledWavelengths &lambda = lambdas[pixelIndex];
            ScratchBuffer &scratchBuffer = scratchBuffers[pixelIndex];

            intersection.bsdf = intersection.material.GetBSDF(texEval, intersection,
                                                              lambda, scratchBuffer);
            if (!intersection.bsdf)
                return;

            if (regularize && pathState.anyNonSpecularBounces)
                intersection.bsdf->Regularize(scratchBuffer);

            if (GetOptions().forceDiffuse) {
                SamplerHandle sampler = samplers[pixelIndex];
                SampledSpectrum r = intersection.bsdf->rho(
                    intersection.wo, {sampler.Get1D()}, {sampler.Get2D()});
                BxDFHandle bxdf =
                    scratchBuffer.Alloc<DiffuseBxDF>(r, SampledSpectrum(0.), 0.);
                intersection.bsdf =
                    scratchBuffer.Alloc<BSDF>(intersection, bxdf, intersection.bsdf->eta);
            }

            bxdfEvalQueues->Add(intersection.bsdf->GetBxDF().Tag(), rayIndex);

            intersection.bssrdf = intersection.material.GetBSSRDF(texEval, intersection,
                                                                  lambda, scratchBuffer);
        });
}

}  // namespace pbrt
