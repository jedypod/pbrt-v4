// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/pbrt.h>

#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/pathintegrator.h>
#include <pbrt/interaction.h>

namespace pbrt {

static void InitializeVisibleSurfaceRecursive(TypePack<> types,
                                              GPUPathIntegrator *integrator) {}

template <typename... Types>
static void InitializeVisibleSurfaceRecursive(TypePack<Types...> types,
                                              GPUPathIntegrator *integrator) {
    using BxDF = typename GetFirst<TypePack<Types...>>::type;
    integrator->InitializeVisibleSurface<BxDF>();

    InitializeVisibleSurfaceRecursive(typename RemoveFirst<TypePack<Types...>>::type(),
                                      integrator);
}

template <typename BxDF>
void GPUPathIntegrator::InitializeVisibleSurface() {
    int bxdfTag = BxDFHandle::TypeIndex<BxDF>();
    std::string description =
        StringPrintf("InitializeVisibleSurface - %s", BxDFTraits<BxDF>::name());

    // Set the pixel's VisibleSurface for the first visible point.
    GPUParallelFor(
        description.c_str(), pixelsPerPass, [=] __device__(BxDFEvalIndex bsdfEvalIndex) {
            // IMPORTANT: must not read rayo/rayd in this kernel...
            if (bsdfEvalIndex >= bxdfEvalQueues->Size(bxdfTag))
                return;

            PathRayIndex rayIndex = bxdfEvalQueues->Get(bxdfTag, bsdfEvalIndex);
            int depth = 0;
            PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
            SurfaceInteraction &intersection = intersections[depth & 1][pixelIndex];
            const SampledWavelengths &lambda = lambdas[pixelIndex];
            PathState *pathState = &pathStates[pixelIndex];

            if (intersection.material)
                visibleSurfaces[pixelIndex] =
                    VisibleSurface(intersection, camera.WorldFromCamera(), lambda);
        });
}

void GPUPathIntegrator::InitializeVisibleSurface() {
    InitializeVisibleSurfaceRecursive(typename BxDFHandle::Types(), this);
}

void GPUPathIntegrator::UpdateFilm() {
    // Update film
    GPUParallelFor("Update Film", pixelsPerPass, [=] __device__(PixelIndex pixelIndex) {
        Point2i pPixel = pPixels[pixelIndex];
        if (!InsideExclusive(pPixel, film.PixelBounds()))
            return;

        const SampledWavelengths &lambda = lambdas[pixelIndex];
        const PathState &pathState = pathStates[pixelIndex];

        SampledSpectrum Lw = pathState.L * cameraRayWeights[pixelIndex];
        const pstd::optional<VisibleSurface> &visibleSurface =
            !visibleSurfaces.empty() ? visibleSurfaces[pixelIndex]
                                     : pstd::optional<VisibleSurface>();

        // NOTE: assumes that no more than one thread is
        // working on each pixel.
        film.AddSample(pPixel, Lw, lambda, visibleSurface, pathState.filterSampleWeight);
    });
}

}  // namespace pbrt
