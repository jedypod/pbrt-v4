// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/pbrt.h>

#include <pbrt/cameras.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/options.h>
#include <pbrt/samplers.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

namespace pbrt {

void GPUPathIntegrator::GenerateCameraRays(int pixelSample) {
    GPUParallelFor(
        "Generate Camera rays", pixelsPerPass, [=] __device__(PixelIndex pixelIndex) {
            Point2i pPixel = pPixels[pixelIndex];
            if (!InsideExclusive(pPixel, camera.GetFilm().PixelBounds()))
                return;

            SamplerHandle sampler = samplers[pixelIndex];

            CameraSample cameraSample = sampler.GetCameraSample(pPixel, filter);

            Float lu = RadicalInverse(1, pixelSample) + BlueNoise(47, pPixel.x, pPixel.y);
            if (lu >= 1)
                lu -= 1;
            if (GetOptions().disableWavelengthJitter)
                lu = 0.5f;
            SampledWavelengths lambda = SampledWavelengths::SampleImportance(lu);
            lambdas[pixelIndex] = lambda;

            pstd::optional<CameraRay> cr = camera.GenerateRay(cameraSample, lambda);
            if (!cr) {
                cameraRayWeights[PixelIndex(pixelIndex)] = SampledSpectrum(0);
                return;
            }
            cameraRayWeights[pixelIndex] = cr->weight;

            PathState &pathState = pathStates[pixelIndex];
            pathState.filterSampleWeight = cameraSample.weight;

            PathRayIndex rayIndex = pathRayQueue->Add(cr->ray, 1e20f);
            rayIndexToPixelIndex[0][rayIndex] = pixelIndex;
        });
}

}  // namespace pbrt
