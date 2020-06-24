// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/gpu/indirect.h>

#include <pbrt/media.h>

namespace pbrt {

static void SampleIndirectRecursive(TypePack<> types, GPUPathIntegrator *integrator,
                                    int depth) {
    // done!
}

template <typename... Types>
static void SampleIndirectRecursive(TypePack<Types...> types,
                                    GPUPathIntegrator *integrator, int depth) {
    using BxDF = typename GetFirst<TypePack<Types...>>::type;
    integrator->SampleIndirect<BxDF>(depth);

    SampleIndirectRecursive(typename RemoveFirst<TypePack<Types...>>::type(), integrator,
                            depth);
}

void GPUPathIntegrator::SampleIndirect(int depth) {
    SampleIndirectRecursive(typename BxDFHandle::Types(), this, depth);

    if (haveMedia) {
        GPUParallelFor(
            "Sample indirect - phase function", pixelsPerPass,
            [=] __device__(MediumEvalIndex mediumEvalIndex) {
                if (mediumEvalIndex >= mediumEvalQueue->Size())
                    return;

                PathRayIndex rayIndex = mediumEvalQueue->Get(mediumEvalIndex);
                PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
                PathState &pathState = pathStates[pixelIndex];

                SampledSpectrum &beta = pathState.beta;
                if (!beta)
                    return;

                SampledSpectrum &pdfUni = pathState.pdfUni;
                SampledSpectrum &pdfNEE = pathState.pdfNEE;
                SamplerHandle sampler = samplers[pixelIndex];
                MediumInteraction &intr = mediumInteractions[depth & 1][pixelIndex];

                pstd::optional<PhaseFunctionSample> ps =
                    intr.phase.Sample_p(intr.wo, sampler.Get2D());
                if (!ps || ps->pdf == 0) {
                    beta = SampledSpectrum(0);
                    return;
                }

                beta *= ps->p;
                pdfNEE = pdfUni;
                pdfUni *= ps->pdf;

                pathState.bsdfFlags = BxDFFlags::Unset;
                if (regularize)
                    pathState.anyNonSpecularBounces =
                        true;  // assume |g| isn't basically 1...

                // russian roulette
                SampledSpectrum rrBeta = beta * pathState.etaScale;
                if (rrBeta.MaxComponentValue() < 1 && depth > 3) {
                    Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
                    if (sampler.Get1D() < q) {
                        beta = SampledSpectrum(0);
                        return;
                    }
                    pdfUni *= 1 - q;
                    pdfNEE *= 1 - q;
                }

                Ray ray = intr.SpawnRay(ps->wi);
                PathRayIndex newRayIndex = pathRayQueue->Add(ray, 1e20f);
                pixelIndexToRayIndex[pixelIndex] = newRayIndex;
                rayIndexToPixelIndex[(depth & 1) ^ 1][newRayIndex] = pixelIndex;
            });
    }
}

}  // namespace pbrt
