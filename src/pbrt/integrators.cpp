
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */


// integrators/whitted.cpp*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/integrators.h>
#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/string.h>

namespace pbrt {

STAT_COUNTER("Integrator/Camera rays traced", nCameraRays);

// Integrator Method Definitions
Integrator::~Integrator() {}


// ImageTileIntegrator Method Definitions
void ImageTileIntegrator::Render() {
    if (PbrtOptions.debugStart.has_value()) {
        pstd::optional<std::vector<int>> c =
            SplitStringToInts(*PbrtOptions.debugStart, ',');
        if (!c)
            ErrorExit("Didn't find integer values after --debugstart: %s",
                      *PbrtOptions.debugStart);
        if (c->size() != 3)
            ErrorExit("Didn't find three integer values after --debugstart: %s",
                      *PbrtOptions.debugStart);

        Point2i pPixel((*c)[0], (*c)[1]);
        int sampleIndex = (*c)[2];

        MemoryArena arena;
        MaterialBuffer materialBuffer(16384);
        SamplerHandle tileSampler = initialSampler.Clone(1, Allocator())[0];
        tileSampler.StartPixelSample(pPixel, sampleIndex);

        EvaluatePixelSample(pPixel, sampleIndex, tileSampler, arena, materialBuffer);

        return;
    }

    thread_local Point2i threadPixel;
    thread_local int threadSampleIndex;
    CheckCallbackScope _([&]() {
            return StringPrintf("Rendering failed at pixel (%d, %d) sample %d. Debug with "
                                "\"--debugstart %d,%d,%d\"\n",
                                threadPixel.x, threadPixel.y, threadSampleIndex,
                                threadPixel.x, threadPixel.y, threadSampleIndex);
        });

    // Render image tiles in parallel

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i pixelBounds = camera->film->pixelBounds;
    StatsSetImageResolution(pixelBounds);
    StatsSetPixelStatsBaseName(RemoveExtension(camera->film->filename).c_str());

    Vector2i pixelExtent = pixelBounds.Diagonal();

    int spp = initialSampler.SamplesPerPixel();
    ProgressReporter reporter(int64_t(spp) * pixelBounds.Area(), "Rendering");
    int startWave = 0, endWave = 1, waveDelta = 1;
    std::vector<MemoryArena> arenas(MaxThreadIndex());
    std::vector<MaterialBuffer> materialBuffers(MaxThreadIndex());
    for (auto &m : materialBuffers)
        m = MaterialBuffer(16384);
    std::vector<SamplerHandle> samplers =
        initialSampler.Clone(MaxThreadIndex(), Allocator());

    pstd::optional<Image> referenceImage;
    FILE *mseOutFile = nullptr;
    if (PbrtOptions.mseReferenceImage) {
        auto mse = Image::Read(*PbrtOptions.mseReferenceImage);
        if (!mse)
            ErrorExit("%s: unable to read MSE image", *PbrtOptions.mseReferenceImage);
        referenceImage = mse->image;

        Bounds2i msePixelBounds = mse->metadata.pixelBounds ? *mse->metadata.pixelBounds :
            Bounds2i(Point2i(0, 0), referenceImage->Resolution());
        if (!Inside(pixelBounds, msePixelBounds))
            ErrorExit("Output image pixel bounds %s aren't inside the MSE image's pixel bounds %s.",
                      pixelBounds, msePixelBounds);

        // Transform the pixelBounds of the image we're rendering to the
        // coordinate system with msePixelBounds.pMin at the origin, which
        // in turn gives us the section of the MSE image to crop. (This is
        // complicated by the fact that Image doesn't support pixel
        // bounds...)
        Bounds2i cropBounds(Point2i(pixelBounds.pMin - msePixelBounds.pMin),
                            Point2i(pixelBounds.pMax - msePixelBounds.pMin));
        *referenceImage = referenceImage->Crop(cropBounds);
        CHECK_EQ(referenceImage->Resolution(), Point2i(pixelExtent));

        mseOutFile = fopen(PbrtOptions.mseReferenceOutput->c_str(), "w");
        if (!mseOutFile)
            ErrorExit("%s: %s", *PbrtOptions.mseReferenceOutput, strerror(errno));
    }

    while (startWave < spp) {
        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            // Render section of image corresponding to _tile_

            // Allocate _MemoryArena_ for tile
            MemoryArena &arena = arenas[ThreadIndex];
            MaterialBuffer &materialBuffer = materialBuffers[ThreadIndex];

            VLOG(1, "Starting image tile %s startWave %d, endWave %d", tileBounds,
                 startWave, endWave);

            // Loop over pixels in tile to render them
            for (Point2i pPixel : tileBounds) {
                StatsReportPixelStart(pPixel);
                threadPixel = pPixel;

                for (int sampleIndex = startWave; sampleIndex < endWave; ++sampleIndex) {
                    threadSampleIndex = sampleIndex;

                    // Get sampler instance for tile
                    SamplerHandle &sampler = samplers[ThreadIndex];
                    sampler.StartPixelSample(pPixel, sampleIndex);

                    EvaluatePixelSample(pPixel, sampleIndex, sampler, arena,
                                        materialBuffer);

                    // Free _MemoryArena_ memory from computing image sample
                    // value
                    arena.Reset();
                    materialBuffer.Reset();
                }
                StatsReportPixelEnd(pPixel);
            }
            VLOG(1, "Finished image tile %s", tileBounds);

            reporter.Update((endWave - startWave) * tileBounds.Area());
        });

        startWave = endWave;
        endWave = std::min(spp, endWave + waveDelta);
        if (!referenceImage)
            waveDelta = std::min(2 * waveDelta, 64);

        LOG_VERBOSE("Writing image with spp = %d", startWave);
        ImageMetadata metadata;
        metadata.renderTimeSeconds = reporter.ElapsedSeconds();
        metadata.samplesPerPixel = startWave;
        if (referenceImage) {
            ImageMetadata filmMetadata;
            Image filmImage = camera->film->GetImage(&filmMetadata, 1.f / startWave);

            ImageChannelValues mse = filmImage.MSE(filmImage.AllChannelsDesc(), *referenceImage);
            fprintf(mseOutFile, "%d, %.9g\n", startWave, mse.Average());

            metadata.MSE = mse.Average();

            fflush(mseOutFile);
        }

        camera->InitMetadata(&metadata);
        camera->film->WriteImage(metadata, 1.0f / startWave);

    }

    if (mseOutFile) fclose(mseOutFile);
    reporter.Done();
    LOG_VERBOSE("Rendering finished");
}

void RayIntegrator::EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                                        SamplerHandle sampler, MemoryArena &arena,
                                        MaterialBuffer &materialBuffer) {
    // Initialize _CameraSample_ for current sample
    CameraSample cameraSample = sampler.GetCameraSample(pPixel, camera->film->filter);

    // Sample wavelengths for the ray
    Float lu = RadicalInverse(1, sampleIndex) + BlueNoise(47, pPixel.x, pPixel.y);
    if (lu >= 1) lu -= 1;
    if (PbrtOptions.disableWavelengthJitter)
        lu = 0.5;
    SampledWavelengths lambda = camera->film->SampleWavelengths(lu);

    // Generate camera ray for current sample
    pstd::optional<CameraRayDifferential> cameraRay =
        camera->GenerateRayDifferential(cameraSample, lambda);
    SampledSpectrum L(0.);
    pstd::optional<VisibleSurface> visibleSurface;
    if (cameraRay) {
        Float rayDiffScale =
            std::max<Float>(.125, 1 / std::sqrt((Float)sampler.SamplesPerPixel()));
        if (!PbrtOptions.disablePixelJitter)
            cameraRay->ray.ScaleDifferentials(rayDiffScale);
        ++nCameraRays;

        // Evaluate radiance along camera ray
        L = cameraRay->weight * Li(cameraRay->ray, lambda, sampler, arena, materialBuffer,
                                   &visibleSurface);

        // Issue warning if unexpected radiance value returned
        if (L.HasNaNs()) {
            LOG_ERROR("Not-a-number radiance value returned for pixel (%d, %d), sample %d. "
                      "Setting to black.", pPixel.x, pPixel.y, sampleIndex);
            L = SampledSpectrum(0.f);
        } else if (std::isinf(L.y(lambda))) {
            LOG_ERROR("Infinite radiance value returned for pixel (%d, %d), sample %d. "
                      "Setting to black.", pPixel.x, pPixel.y, sampleIndex);
            L = SampledSpectrum(0.f);
        }
        if (cameraRay)
            VLOG(2, "Camera sample: %s -> ray %s -> L = %s, visibleSurface %s",
                 cameraSample, cameraRay->ray, L,
                 (visibleSurface ? visibleSurface->ToString() : "(none)"));
        else
            VLOG(2, "Camera sample: %s -> no ray generated", cameraSample);
    }

    // Add camera ray's contribution to image
    camera->film->AddSample(pPixel, L, lambda, visibleSurface, cameraSample.weight);
}

// Integrator Utility Functions
SampledSpectrum Tr(const Scene &scene, const SampledWavelengths &lambda,
                   SamplerHandle sampler, const Interaction &p0, const Interaction &p1) {
    Ray ray = p0.SpawnRayTo(p1);
    SampledSpectrum Tr(1.f);
    if (LengthSquared(ray.d) == 0) return Tr;

    while (true) {
        pstd::optional<ShapeIntersection> si = scene.Intersect(ray, 1 - ShadowEpsilon);
        // Handle opaque surface along ray's path
        if (si && si->intr.material)
            return SampledSpectrum(0.0f);

        // Update transmittance for current ray segment
        if (ray.medium != nullptr)
            Tr *= ray.medium->Tr(ray, si ? si->tHit : 1 - ShadowEpsilon, lambda, sampler);

        // Generate next ray segment or return final transmittance
        if (!si) break;
        ray = si->intr.SpawnRayTo(p1);
    }
    VLOG(2, "Tr from %s to %s = %s", p0.pi, p1.pi, Tr);
    return Tr;
}

SampledSpectrum LdSampleLights(const SurfaceInteraction &intr, const Scene &scene,
                               const SampledWavelengths &lambda,
                               SamplerHandle sampler, MemoryArena &arena,
                               const LightSampler &lightSampler,
                               bool handleMedia) {
    ProfilerScope _(ProfilePhase::DirectLighting);

    SampledLightVector lights = lightSampler.Sample(intr, sampler.Get1D(), arena);

    Point2f uLight = sampler.Get2D();
    SampledSpectrum Ld(0.f);

    for (const SampledLight &sampledLight : lights) {
        LightHandle light = sampledLight.light;
        DCHECK(light != nullptr && sampledLight.pdf > 0);

        // Sample light source with multiple importance sampling
        pstd::optional<LightLiSample> ls = light.Sample_Li(intr, uLight, lambda);
        if (!ls || !ls->L || ls->pdf == 0)
            continue;

        // Evaluate BSDF for light sampling strategy
        Vector3f wo = intr.wo, wi = ls->wi;
        SampledSpectrum f = intr.bsdf->f(wo, wi) * AbsDot(wi, intr.shading.n);
        if (!f)
            continue;

        SampledSpectrum Li = ls->L;
        if (handleMedia) {
            Li *= ls->Tr(scene, lambda, sampler);
            if (!Li)
                continue;
        } else if (!ls->Unoccluded(scene))
            continue;

        // Add light's contribution to reflected radiance
        Float lightPDF = sampledLight.pdf * ls->pdf;
        if (IsDeltaLight(light->type))
            Ld += f * Li / lightPDF;
        else {
            Float bsdfPDF = intr.bsdf->PDF(wo, wi);
            CHECK_RARE(1e-6, intr.bsdf->PDFIsApproximate() == false && bsdfPDF == 0);
            Float weight = PowerHeuristic(1, lightPDF, 1, bsdfPDF);
            Ld += f * Li * weight / lightPDF;
            VLOG(2, "light u %s, pt %s -> bsdf %s, Li %s wi %s weight %f pdf %f", uLight,
                 intr.p(), f, Li, wi, weight, lightPDF);
        }
    }

    return Ld;
}

SampledSpectrum LdSampleLights(const MediumInteraction &intr, const Scene &scene,
                               const SampledWavelengths &lambda,
                               SamplerHandle sampler, MemoryArena &arena,
                               const LightSampler &lightSampler) {
    ProfilerScope _(ProfilePhase::DirectLighting);

    SampledLightVector lights = lightSampler.Sample(intr, sampler.Get1D(), arena);

    Point2f uLight = sampler.Get2D();
    SampledSpectrum Ld(0.f);

    for (const SampledLight &sampledLight : lights) {
        LightHandle light = sampledLight.light;
        CHECK(light != nullptr && sampledLight.pdf != 0);

        // Sample light source with multiple importance sampling
        pstd::optional<LightLiSample> ls = light.Sample_Li(intr, uLight, lambda);
        if (!ls || ls->pdf == 0 || !ls->L)
            continue;

        // Evaluate phase function for light sampling strategy
        Float p = intr.phase->p(intr.wo, ls->wi);
        SampledSpectrum f = SampledSpectrum(p);
        Float phasePDF = p;
        if (!f || phasePDF == 0)
            continue;

        // Compute effect of visibility for light source sample
        SampledSpectrum Li = ls->L * ls->Tr(scene, lambda, sampler);
        if (!Li)
            continue;

        // Add light's contribution to reflected radiance
        Float lightPDF = sampledLight.pdf * ls->pdf;
        if (IsDeltaLight(light->type))
            Ld += f * Li / lightPDF;
        else {
            Float weight = PowerHeuristic(1, lightPDF, 1, phasePDF);
            Ld += f * Li * weight / lightPDF;
        }
    }

    return Ld;
}

SampledSpectrum LdSampleLightsAndBSDF(const SurfaceInteraction &intr, const Scene &scene,
                                      const SampledWavelengths &lambda,
                                      SamplerHandle sampler, MemoryArena &arena,
                                      const LightSampler &lightSampler) {
    SampledSpectrum Ld = LdSampleLights(intr, scene, lambda, sampler,
                                        arena, lightSampler);

    ProfilerScope _(ProfilePhase::DirectLighting);

    Float uScattering = sampler.Get1D();
    pstd::optional<BSDFSample> bs =
        intr.bsdf->Sample_f(intr.wo, uScattering, sampler.Get2D());
    if (!bs || !bs->f || bs->pdf == 0)
        return Ld;

    Vector3f wi = bs->wi;
    SampledSpectrum f = bs->f * AbsDot(wi, intr.shading.n);

    Ray ray = intr.SpawnRay(wi);
    pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
    if (si) {
        SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
        if (Le) {
            if (bs->IsSpecular())
                Ld += f * Le / bs->pdf;
            else {
                // Compute MIS pdf...
                LightHandle areaLight(si->intr.areaLight);
                Float lightPDF = lightSampler.PDF(intr, areaLight) *
                    areaLight.Pdf_Li(intr, wi);
                Float bsdfPDF = intr.bsdf->PDFIsApproximate() ?
                    intr.bsdf->PDF(intr.wo, wi) : bs->pdf;
                Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                Ld += f * Le * weight / bs->pdf;
            }
        }
    } else {
        for (const auto &light : scene.infiniteLights) {
            SampledSpectrum Le = light.Le(ray, lambda);
            if (bs->IsSpecular())
                Ld += f * Le / bs->pdf;
            else {
                // Compute MIS pdf...
                Float lightPDF = lightSampler.PDF(intr, light) *
                    light.Pdf_Li(intr, wi);
                Float bsdfPDF = intr.bsdf->PDFIsApproximate() ?
                    intr.bsdf->PDF(intr.wo, wi) : bs->pdf;
                Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                Ld += f * Le * weight / bs->pdf;
            }
        }
    }
    return Ld;
}

// WhittedIntegrator Method Definitions
SampledSpectrum WhittedIntegrator::Li(RayDifferential ray,
                                      const SampledWavelengths &lambda,
                                      SamplerHandle sampler,
                                      MemoryArena &arena,
                                      MaterialBuffer &materialBuffer,
                                      pstd::optional<VisibleSurface> *visibleSurface) const {
    return WhittedLi(ray, lambda, sampler, arena, materialBuffer, 0);
}

SampledSpectrum WhittedIntegrator::WhittedLi(RayDifferential ray,
                                             const SampledWavelengths &lambda,
                                             SamplerHandle sampler,
                                             MemoryArena &arena,
                                             MaterialBuffer &materialBuffer,
                                             int depth) const {
    SampledSpectrum L(0.);
    // Find closest ray intersection or return background radiance
    pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
    if (!si) {
        for (const auto &light : scene.lights) L += light.Le(ray, lambda);
        return L;
    }

    // Compute emitted and reflected light at ray intersection point

    // Initialize common variables for Whitted integrator
    SurfaceInteraction &isect = si->intr;
    const Normal3f &ns = isect.shading.n;
    Vector3f wo = isect.wo;

    // Compute scattering functions for surface interaction
    isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler);
    const BSDF *bsdf = isect.bsdf;
    if (bsdf == nullptr) {
        isect.SkipIntersection(&ray, si->tHit);
        return WhittedLi(ray, lambda, sampler, arena, materialBuffer, depth);
    }

    // Compute emitted light if ray hit an area light source
    L += isect.Le(wo, lambda);

    // Add contribution of each light source
    for (const auto &light : scene.lights) {
        pstd::optional<LightLiSample> ls =
            light.Sample_Li(isect, sampler.Get2D(), lambda);
        if (!ls) continue;
        SampledSpectrum f = bsdf->f(wo, ls->wi);
        if (f && ls->Unoccluded(scene))
            L += f * ls->L * AbsDot(ls->wi, ns) / ls->pdf;
    }
    if (depth + 1 < maxDepth) {
        // Trace rays for specular reflection and refraction
        for (BxDFReflTransFlags flags : { BxDFReflTransFlags::Reflection,
                                          BxDFReflTransFlags::Transmission }) {
            Vector3f wi;
            SampledSpectrum f = bsdf->SampleSpecular_f(wo, &wi, flags);
            if (f) {
                RayDifferential r = isect.SpawnRay(ray, wi, BxDFFlags(flags) | BxDFFlags::Specular);
                L += f * WhittedLi(r, lambda, sampler, arena, materialBuffer,
                                   depth + 1) * AbsDot(wi, ns);
            }
        }
    }
    return L;
}

std::string WhittedIntegrator::ToString() const {
    return StringPrintf("[ WhittedIntegrator maxDepth: %d ]", maxDepth);
}

std::unique_ptr<WhittedIntegrator> WhittedIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    return std::make_unique<WhittedIntegrator>(maxDepth, scene, std::move(camera),
                                               std::move(sampler));
}


// SimplePathIntegrator Method Definitions
SimplePathIntegrator::SimplePathIntegrator(int maxDepth, bool sampleLights, bool sampleBSDF,
                                           const Scene &scene,
                                           std::unique_ptr<const Camera> c,
                                           SamplerHandle sampler)
    : RayIntegrator(scene, std::move(c), std::move(sampler)),
      maxDepth(maxDepth),
      sampleLights(sampleLights),
      sampleBSDF(sampleBSDF),
      lightSampler(scene.lights, Allocator()) {
}

SampledSpectrum SimplePathIntegrator::Li(RayDifferential ray,
                                         const SampledWavelengths &lambda,
                                         SamplerHandle sampler,
                                         MemoryArena &arena,
                                         MaterialBuffer &materialBuffer,
                                         pstd::optional<VisibleSurface> *visibleSurface) const {
    ProfilerScope _(ProfilePhase::RayIntegratorLi);
    SampledSpectrum L(0.f), beta(1.f);
    bool specularBounce = true;

    for (int depth = 0; beta; ++depth) {
        // Intersect _ray_ with scene and store intersection in _isect_
        pstd::optional<ShapeIntersection> si = scene.Intersect(ray);

        if (!si) {
            if (!sampleLights || specularBounce)
                for (const auto &light : scene.infiniteLights)
                    L += beta * light.Le(ray, lambda);
            break;
        }

        if (!sampleLights || specularBounce)
            L += beta * si->intr.Le(-ray.d, lambda);

        if (depth == maxDepth)
            break;

        // Compute scattering functions and skip over medium boundaries
        SurfaceInteraction &isect = si->intr;
        isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler);
        BSDF *bsdf = isect.bsdf;
        Vector3f wo = -ray.d;
        if (bsdf == nullptr) {
            isect.SkipIntersection(&ray, si->tHit);
            depth--;
            continue;
        }

        if (sampleLights) {
            Float lightPDF;
            LightHandle light = lightSampler.Sample(sampler.Get1D(), &lightPDF);

            Point2f uLight = sampler.Get2D();
            pstd::optional<LightLiSample> ls = light.Sample_Li(isect, uLight, lambda);
            if (ls && ls->L && ls->pdf > 0) {
                // Evaluate BSDF for light sampling strategy
                Vector3f wi = ls->wi;
                SampledSpectrum f = isect.bsdf->f(wo, wi) * AbsDot(wi, isect.shading.n);
                if (f && ls->Unoccluded(scene))
                    L += beta * f * ls->L / (lightPDF * ls->pdf);
            }
        }

        if (sampleBSDF) {
            // Sample BSDF to get new path direction
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = bsdf->Sample_f(wo, u, sampler.Get2D());
            if (!bs) break;
            beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
            specularBounce = bs->IsSpecular();
            ray = isect.SpawnRay(bs->wi);
        } else {
            Float pdf;
            Vector3f wi;
            if (bsdf->HasReflection() && bsdf->HasTransmission()) {
                wi = SampleUniformSphere(sampler.Get2D());
                pdf = UniformSpherePDF();
            } else {
                wi = SampleUniformHemisphere(sampler.Get2D());
                pdf = UniformHemispherePDF();
                if (bsdf->HasReflection() && Dot(wo, isect.n) * Dot(wi, isect.n) < 0)
                    wi = -wi;
                else if (bsdf->HasTransmission() && Dot(wo, isect.n) * Dot(wi, isect.n) > 0)
                    wi = -wi;
            }
            beta *= bsdf->f(wo, wi) * AbsDot(wi, isect.shading.n) / pdf;
            specularBounce = false;
            ray = isect.SpawnRay(wi);
        }
        if (sampleLights && !specularBounce && depth + 1 == maxDepth)
            break;

        CHECK_GE(beta.y(lambda), 0.f);
        DCHECK(!std::isinf(beta.y(lambda)));
    }

    return L;
}

std::string SimplePathIntegrator::ToString() const {
    return StringPrintf("[ SimplePathIntegrator maxDepth: %d sampleLights: %s "
                        "sampleBSDF: %s ]", maxDepth, sampleLights, sampleBSDF);
}

std::unique_ptr<SimplePathIntegrator> SimplePathIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    bool sampleLights = dict.GetOneBool("samplelights", true);
    bool sampleBSDF = dict.GetOneBool("samplebsdf", true);
    return std::make_unique<SimplePathIntegrator>(maxDepth, sampleLights, sampleBSDF,
                                                  scene, std::move(camera),
                                                  std::move(sampler));
}

// LightPathIntegrator Method Definitions
LightPathIntegrator::LightPathIntegrator(int maxDepth, const Scene &scene,
                                         std::unique_ptr<const Camera> c,
                                         SamplerHandle sampler)
    : ImageTileIntegrator(scene, std::move(c), std::move(sampler)),
      maxDepth(maxDepth) {
    lightSampler = std::make_unique<PowerLightSampler>(scene.lights, Allocator());
}

void LightPathIntegrator::EvaluatePixelSample(const Point2i &pPixel, int sampleIndex,
                                              SamplerHandle sampler, MemoryArena &arena,
                                              MaterialBuffer &materialBuffer) {
    // Eat the first two samples since they're "special"...
    (void)sampler.Get2D();

    // Sample wavelengths for the ray
    Float lu = RadicalInverse(1, sampleIndex) + BlueNoise(47, pPixel.x, pPixel.y);
    if (lu >= 1) lu -= 1;
    if (PbrtOptions.disableWavelengthJitter)
        lu = 0.5;
    SampledWavelengths lambda = camera->film->SampleWavelengths(lu);

    // Sample a light
    Float lightPDF;
    LightHandle light = lightSampler->Sample(sampler.Get1D(), &lightPDF);
    if (!light || lightPDF == 0)
        return;

    Float time = Lerp(sampler.Get1D(), camera->shutterOpen, camera->shutterClose);
    pstd::optional<LightLeSample> les = light.Sample_Le(sampler.Get2D(), sampler.Get2D(),
                                                         lambda, time);
    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L)
        return;
    RayDifferential ray(les->ray);
    SampledSpectrum beta = les->L * les->AbsCosTheta(ray.d) /
        (lightPDF * les->pdfPos * les->pdfDir);

    // Is the light sample directly visible?
    if (les->intr) {
        pstd::optional<CameraWiSample> cs =
            camera->Sample_Wi(*les->intr, sampler.Get2D(), lambda);
        if (cs && cs->pdf != 0) {
            Float pdf = light.Pdf_Li(cs->pLens, cs->wi);
            if (pdf > 0) {
                SampledSpectrum Le = light.L(*les->intr, cs->wi, lambda);
                if (Le && cs->Unoccluded(scene)) {
                    SampledSpectrum L = Le * les->AbsCosTheta(cs->wi) * cs->Wi /
                        (lightPDF * pdf * cs->pdf);
                    camera->film->AddSplat(cs->pRaster, L, lambda);
                }
            }
        }
    }

    for (int depth = 0; depth < maxDepth && beta; ++depth) {
        pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
        if (!si)
            break;

        // Compute scattering functions for _mode_ and skip over medium
        // boundaries
        SurfaceInteraction &isect = si->intr;
        isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler,
                                         TransportMode::Importance);
        if (isect.bsdf == nullptr) {
            isect.SkipIntersection(&ray, si->tHit);
            --depth;
            continue;
        }
        Vector3f wo = isect.wo;

        // Try to splat into the film
        pstd::optional<CameraWiSample> cs =
            camera->Sample_Wi(isect, sampler.Get2D(), lambda);
        if (cs && cs->pdf != 0) {
            SampledSpectrum L = beta * isect.bsdf->f(wo, cs->wi) *
                AbsDot(cs->wi, isect.shading.n) * cs->Wi / cs->pdf;
            if (L && cs->Unoccluded(scene))
                camera->film->AddSplat(cs->pRaster, L, lambda);
        }

        // Sample the BSDF...
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = isect.bsdf->Sample_f(wo, u, sampler.Get2D());
        if (!bs || bs->pdf == 0)
            break;

        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        ray = isect.SpawnRay(ray, bs->wi, bs->flags);
    }
}

std::string LightPathIntegrator::ToString() const {
    return StringPrintf("[ LightPathIntegrator maxDepth: %d lightSampler: %s ]",
                        maxDepth, lightSampler);
}

std::unique_ptr<LightPathIntegrator> LightPathIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    return std::make_unique<LightPathIntegrator>(maxDepth, scene, std::move(camera),
                                                  std::move(sampler));
}


STAT_PERCENT("Integrator/Zero-radiance paths", zeroRadiancePaths, totalPaths);
STAT_PERCENT("Integrator/Regularized BSDFs", regularizedBSDFs, totalBSDFs);
STAT_INT_DISTRIBUTION("Integrator/Path length", pathLength);

// PathIntegrator Method Definitions
PathIntegrator::PathIntegrator(int maxDepth, const Scene &scene,
                               std::unique_ptr<const Camera> c,
                               SamplerHandle sampler,
                               Float rrThreshold,
                               const std::string &lightSampleStrategy,
                               bool regularize)
    : RayIntegrator(scene, std::move(c), std::move(sampler)),
      maxDepth(maxDepth),
      rrThreshold(rrThreshold),
      lightSampler(LightSampler::Create(lightSampleStrategy, scene.lights, Allocator())),
      regularize(regularize) { }

SampledSpectrum PathIntegrator::Li(RayDifferential ray,
                                   const SampledWavelengths &lambda,
                                   SamplerHandle sampler,
                                   MemoryArena &arena,
                                   MaterialBuffer &materialBuffer,
                                   pstd::optional<VisibleSurface> *visibleSurface) const {
    ProfilerScope p(ProfilePhase::RayIntegratorLi);
    SampledSpectrum L(0.f), beta(1.f);
    bool specularBounce = false, anyNonSpecularBounces = false;
    int depth;
    Float etaScale = 1;

    Float bsdfPDF;
    SurfaceInteraction prevIntr;

    for (depth = 0;; ++depth) {
        // Find next path vertex and accumulate contribution
        VLOG(2, "Path tracer depth %d, current L = %s, beta = %s", depth, L, beta);

        // Intersect _ray_ with scene and store intersection in _isect_
        pstd::optional<ShapeIntersection> si = scene.Intersect(ray);

        // Add emitted light at path vertex or from the environment
        if (si) {
            SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
            if (Le) {
                if (depth == 0 || specularBounce) {
                    L += beta * Le;
                    if (depth == 1)
                        (*visibleSurface)->Ld += beta * Le;
                }
                else {
                    // Compute MIS pdf...
                    LightHandle areaLight(si->intr.areaLight);
                    Float lightPDF = lightSampler->PDF(prevIntr, areaLight) *
                        areaLight.Pdf_Li(prevIntr, ray.d);
                    Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                    // beta already includes 1 / bsdf pdf.
                    L += beta * weight * Le;
                    if (depth == 1)
                        (*visibleSurface)->Ld += beta * weight * Le;
                }
                VLOG(2, "Added Le -> L = %s", L);
            }
        } else {
            for (const auto &light : scene.infiniteLights) {
                SampledSpectrum Le = light.Le(ray, lambda);
                if (depth == 0 || specularBounce)
                    L += beta * Le;
                else {
                    // Compute MIS pdf...
                    Float lightPDF = lightSampler->PDF(prevIntr, light) *
                        light.Pdf_Li(prevIntr, ray.d);
                    Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                    // beta already includes 1 / bsdf pdf.
                    L += beta * weight * Le;
                    if (depth == 1)
                        (*visibleSurface)->Ld += beta * weight * Le;
                }
            }
            VLOG(2, "Added infinite area lights -> L = %s", L);

            break;
        }

        // Terminate path if ray escaped or _maxDepth_ was reached
        if (depth >= maxDepth) break;

        // Compute scattering functions and skip over medium boundaries
        SurfaceInteraction &isect = si->intr;
        isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler);
        BSDF *bsdf = isect.bsdf;
        if (bsdf == nullptr) {
            isect.SkipIntersection(&ray, si->tHit);
            depth--;
            continue;
        }
        prevIntr = si->intr;
        if (depth == 0)
            *visibleSurface = VisibleSurface(si->intr, *camera, lambda);

        if (regularize && anyNonSpecularBounces) {
            ++regularizedBSDFs;
            bsdf->Regularize(materialBuffer);
        }
        ++totalBSDFs;

        // Sample illumination from lights to find path contribution.
        // (But skip this for perfectly specular BSDFs.)
        if (bsdf->IsNonSpecular()) {
            ++totalPaths;
            SampledSpectrum Ld = LdSampleLights(isect, scene, lambda, sampler, arena,
                                                *lightSampler);
            if (depth == 0 && visibleSurface != nullptr)
                (*visibleSurface)->Ld = beta * Ld;
            VLOG(2, "Sampled direct lighting Ld = %s", Ld);
            if (!Ld) ++zeroRadiancePaths;
            CHECK_GE(Ld.y(lambda), 0.f);
            L += beta * Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d;
        Float u = sampler.Get1D();
        pstd::optional<BSDFSample> bs = bsdf->Sample_f(wo, u, sampler.Get2D());
        if (!bs) break;
        VLOG(2, "Sampled BSDF, f = %s, pdf = %f", bs->f, bs->pdf);
        beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
        bsdfPDF = bsdf->PDFIsApproximate() ? bsdf->PDF(wo, bs->wi) : bs->pdf;

        VLOG(2, "Updated beta = %s", beta);
        CHECK_GE(beta.y(lambda), 0.f);
        DCHECK(!std::isinf(beta.y(lambda)));
        specularBounce = bs->IsSpecular();
        anyNonSpecularBounces |= !bs->IsSpecular();
        if (bs->IsTransmission())
            // Update the term that tracks radiance scaling for refraction.
            etaScale *= Sqr(bsdf->eta);
        ray = isect.SpawnRay(ray, bs->wi, bs->flags);

        // Account for subsurface scattering, if applicable
        if (isect.bssrdf && bs->IsTransmission()) {
            // Importance sample the BSSRDF
            pstd::optional<BSSRDFProbeSegment> probeSeg =
                isect.bssrdf.Sample(sampler.Get1D(), sampler.Get2D());
            if (!probeSeg)
                break;

            uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
            WeightedReservoirSampler<SurfaceInteraction, Float> interactionSampler(seed);

            // Intersect BSSRDF sampling ray against the scene geometry
            Interaction base(probeSeg->p0, probeSeg->time, (const Medium *)nullptr);
            while (true) {
                Ray r = base.SpawnRayTo(probeSeg->p1);
                if (r.d == Vector3f(0, 0, 0))
                    break;

                pstd::optional<ShapeIntersection> si = scene.Intersect(r, 1);
                if (!si)
                    break;

                base = si->intr;
                if (si->intr.material == isect.material)
                    interactionSampler.Add(si->intr, 1.f);
            }

            if (!interactionSampler.HasSample())
                break;

            BSSRDFSample bssrdfSample =
                isect.bssrdf.ProbeIntersectionToSample(interactionSampler.GetSample(),
                                                       materialBuffer);
            if (!bssrdfSample.S || bssrdfSample.pdf == 0)
                break;
            beta *= bssrdfSample.S * interactionSampler.WeightSum() / bssrdfSample.pdf;

            const SurfaceInteraction &pi = bssrdfSample.si;
            anyNonSpecularBounces = true;
            if (regularize) {
                ++regularizedBSDFs;
                pi.bsdf->Regularize(materialBuffer);
            } else
                ++totalBSDFs;

            // Account for the direct subsurface scattering component
            // FIXME: if isect.bsdf is non-specular need to trace a ray for the BSDF
            // sampled direct lighting component...
            SampledSpectrum Ld = LdSampleLights(pi, scene, lambda, sampler, arena,
                                                *lightSampler);
            L += beta * Ld;

            // Account for the indirect subsurface scattering component
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = pi.bsdf->Sample_f(pi.wo, u, sampler.Get2D());
            if (!bs) break;
            beta *= bs->f * AbsDot(bs->wi, pi.shading.n) / bs->pdf;
            bsdfPDF = pi.bsdf->PDFIsApproximate() ? pi.bsdf->PDF(wo, bs->wi) : bs->pdf;
            DCHECK(!std::isinf(beta.y(lambda)));
            specularBounce = bs->IsSpecular();
            ray = RayDifferential(pi.SpawnRay(bs->wi));
        }

        // Possibly terminate the path with Russian roulette.
        // Factor out radiance scaling due to refraction in rrBeta.
        SampledSpectrum rrBeta = beta * etaScale;
        VLOG(2, "etaScale %f -> rrBeta %s", etaScale, rrBeta);
        if (rrBeta.MaxComponentValue() < rrThreshold && depth > 3) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            DCHECK(!std::isinf(beta.y(lambda)));
        }
    }
    ReportValue(pathLength, depth);
    return L;
}

std::string PathIntegrator::ToString() const {
    return StringPrintf("[ PathIntegrator maxDepth: %d rrThreshold: %f "
                        "lightSampler: %s regularize: %s ]", maxDepth,
                        rrThreshold, lightSampler, regularize);
}

std::unique_ptr<PathIntegrator> PathIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    Float rrThreshold = dict.GetOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        dict.GetOneString("lightsamplestrategy", "bvh");
    bool regularize = dict.GetOneBool("regularize", true);
    return std::make_unique<PathIntegrator>(maxDepth, scene, std::move(camera),
                                            std::move(sampler), rrThreshold,
                                            lightStrategy, regularize);
}


STAT_COUNTER("Integrator/Volume interactions", volumeInteractions);
STAT_COUNTER("Integrator/Surface interactions", surfaceInteractions);

// VolPathIntegrator Method Definitions
SampledSpectrum VolPathIntegrator::Li(RayDifferential ray,
                                      const SampledWavelengths &lambda,
                                      SamplerHandle sampler,
                                      MemoryArena &arena,
                                      MaterialBuffer &materialBuffer,
                                      pstd::optional<VisibleSurface> *visibleSurface) const {
    ProfilerScope p(ProfilePhase::RayIntegratorLi);
    SampledSpectrum L(0.f), beta(1.f);
    bool specularBounce = false, anyNonSpecularBounces = false;
    int depth;
    Float scatterPDF;
    Float etaScale = 1;
    Interaction prevIntr;

    for (depth = 0;; ++depth) {
        VLOG(2, "Path tracer depth %d, current L = %s, beta = %s", depth, L, beta);

        // Intersect _ray_ with scene and store intersection in _isect_
        pstd::optional<ShapeIntersection> si = scene.Intersect(ray);

        // Sample the participating medium, if present
        Float tMax = si ? si->tHit : Infinity;
        MediumInteraction mi;
        if (ray.medium) {
            beta *= ray.medium->Sample(ray, tMax, sampler, lambda, arena, &mi);
            VLOG(2, "After medium sample, beta = %s", beta);
        }
        if (!beta) break;

        // Handle an interaction with a medium or a surface
        if (mi.IsValid()) {
            // Terminate path if ray escaped or _maxDepth_ was reached
            if (depth >= maxDepth) break;

            ++volumeInteractions;
            // Handle scattering at point in medium for volumetric path tracer
            SampledSpectrum Ld = LdSampleLights(mi, scene, lambda, sampler, arena,
                                                *lightSampler);
            L += beta * Ld;

            pstd::optional<PhaseFunctionSample> ps =
                mi.phase->Sample_p(-ray.d, sampler.Get2D());
            if (!ps || ps->pdf == 0)
                break;
            VLOG(2, "Sampled phase function p = %s, wi = %s", ps->p, ps->wi);
            beta *= ps->p / ps->pdf;
            scatterPDF = ps->pdf;
            prevIntr = mi;

            ray = mi.SpawnRay(ps->wi);
            specularBounce = false;
            anyNonSpecularBounces = true;
        } else {
            ++surfaceInteractions;
            // Handle scattering at point on surface for volumetric path tracer

            // Add emitted light at path vertex or from the environment
            if (si) {
                SampledSpectrum Le = si->intr.Le(-ray.d, lambda);
                if (Le) {
                    // TODO/think: is it correct to include the transmittance from
                    // sampling the medium here? i.e., sometimes we don't get a chance
                    // to hit the light due to a medium interaction, so are we double-counting
                    // in a sense?  (But then Medium::Sample() only returns an event for
                    // scattering, not absorption, so.... ???)
                    if (depth == 0 || specularBounce)
                        L += beta * Le;
                    else {
                        // Compute MIS pdf...
                        LightHandle areaLight(si->intr.areaLight);
                        Float lightPDF = lightSampler->PDF(prevIntr, areaLight) *
                            areaLight.Pdf_Li(prevIntr, ray.d);
                        Float weight = PowerHeuristic(1, scatterPDF, 1, lightPDF);
                        // beta already includes 1 / bsdf pdf.
                        L += beta * weight * Le;
                    }
                }
            } else {
                for (const auto &light : scene.infiniteLights) {
                    SampledSpectrum Le = light.Le(ray, lambda);
                    if (depth == 0 || specularBounce)
                        L += beta * Le;
                    else {
                        // Compute MIS pdf...
                        Float lightPDF = lightSampler->PDF(prevIntr, light) *
                            light.Pdf_Li(prevIntr, ray.d);
                        Float weight = PowerHeuristic(1, scatterPDF, 1, lightPDF);
                        // beta already includes 1 / bsdf pdf.
                        L += beta * weight * Le;
                    }
                }
                break;
            }

            // Terminate path if ray escaped or _maxDepth_ was reached
            if (depth >= maxDepth) break;

            // Compute scattering functions and skip over medium boundaries
            SurfaceInteraction &isect = si->intr;
            isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler);
            if (isect.bsdf == nullptr) {
                isect.SkipIntersection(&ray, si->tHit);
                depth--;
                continue;
            }
            prevIntr = isect;

            if (regularize && anyNonSpecularBounces) {
                ++regularizedBSDFs;
                isect.bsdf->Regularize(materialBuffer);
            }
            ++totalBSDFs;

            // Sample illumination from lights to find attenuated path
            // contribution
            if (isect.bsdf->IsNonSpecular()) {
                SampledSpectrum Ld = LdSampleLights(isect, scene, lambda, sampler,
                                                    arena, *lightSampler, true);
                L += beta * Ld;
                VLOG(2, "Sampled direct lighting Ld = %s -> L = %s", Ld, L);
            }

            // Sample BSDF to get new path direction
            Vector3f wo = -ray.d;
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs =
                isect.bsdf->Sample_f(wo, u, sampler.Get2D());
            if (!bs) break;
            beta *= bs->f * AbsDot(bs->wi, isect.shading.n) /  bs->pdf;
            VLOG(2, "Sampled BSDF, f = %s, pdf = %f -> beta = %s", bs->f, bs->pdf, beta);
            scatterPDF = isect.bsdf->PDFIsApproximate() ? isect.bsdf->PDF(wo, bs->wi) : bs->pdf;

            DCHECK(std::isinf(beta.y(lambda)) == false);
            specularBounce = bs->IsSpecular();
            anyNonSpecularBounces |= !bs->IsSpecular();
            if (bs->IsTransmission())
                // Update the term that tracks radiance scaling for refraction.
                etaScale *= Sqr(isect.bsdf->eta);
            ray = isect.SpawnRay(ray, bs->wi, bs->flags);

            // Account for attenuated subsurface scattering, if applicable
            if (isect.bssrdf && bs->IsTransmission()) {
                // Importance sample the BSSRDF
                // Importance sample the BSSRDF
                pstd::optional<BSSRDFProbeSegment> probeSeg =
                    isect.bssrdf.Sample(sampler.Get1D(), sampler.Get2D());
                if (!probeSeg)
                    break;

                uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
                WeightedReservoirSampler<SurfaceInteraction, Float> interactionSampler(seed);

                // Intersect BSSRDF sampling ray against the scene geometry
                Interaction base(probeSeg->p0, probeSeg->time, (const Medium *)nullptr);
                while (true) {
                    Ray r = base.SpawnRayTo(probeSeg->p1);
                    if (r.d == Vector3f(0, 0, 0))
                        break;

                    pstd::optional<ShapeIntersection> si = scene.Intersect(r, 1);
                    if (!si)
                        break;

                    base = si->intr;
                    if (si->intr.material == isect.material)
                        interactionSampler.Add(si->intr, 1.f);
                }

                if (!interactionSampler.HasSample())
                    break;

                BSSRDFSample bssrdfSample =
                    isect.bssrdf.ProbeIntersectionToSample(interactionSampler.GetSample(),
                                                           materialBuffer);
                if (!bssrdfSample.S || bssrdfSample.pdf == 0)
                    break;
                beta *= bssrdfSample.S * interactionSampler.WeightSum() / bssrdfSample.pdf;

                const SurfaceInteraction &pi = bssrdfSample.si;
                anyNonSpecularBounces = true;
                if (regularize) {
                    ++regularizedBSDFs;
                    pi.bsdf->Regularize(materialBuffer);
                } else
                    ++totalBSDFs;
                prevIntr = pi;

                // Account for the attenuated direct subsurface scattering
                // component
                SampledSpectrum Ld = LdSampleLights(pi, scene, lambda, sampler, arena,
                                                    *lightSampler);
                L += beta * Ld;

                // Account for the indirect subsurface scattering component
                Float u = sampler.Get1D();
                pstd::optional<BSDFSample> bs = pi.bsdf->Sample_f(pi.wo, u, sampler.Get2D());
                if (!bs) break;
                beta *= bs->f * AbsDot(bs->wi, pi.shading.n) / bs->pdf;
                scatterPDF = bs->pdf;

                DCHECK(!std::isinf(beta.y(lambda)));
                specularBounce = bs->IsSpecular();
                ray = RayDifferential(pi.SpawnRay(bs->wi));
            }
        }

        // Possibly terminate the path with Russian roulette
        // Factor out radiance scaling due to refraction in rrBeta.
        SampledSpectrum rrBeta = beta * etaScale;
        VLOG(2, "etaScale %f -> rrBeta %f", etaScale, rrBeta);
        if (rrBeta.MaxComponentValue() < rrThreshold && depth > 3) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) break;
            beta /= 1 - q;
            VLOG(2, "RR didn't happen, beta = %f", beta);
            DCHECK(std::isinf(beta.y(lambda)) == false);
        }
    }
    ReportValue(pathLength, depth);
    return L;
}

std::string VolPathIntegrator::ToString() const {
    return StringPrintf("[ VolPathIntegrator maxDepth: %d rrThreshold: %f "
                        "lightSampler: %s regularize: %s ]", maxDepth,
                        rrThreshold, lightSampler, regularize);
}

std::unique_ptr<VolPathIntegrator> VolPathIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    Float rrThreshold = dict.GetOneFloat("rrthreshold", 1.);
    std::string lightStrategy =
        dict.GetOneString("lightsamplestrategy", "bvh");
    bool regularize = dict.GetOneBool("regularize", true);
    return std::make_unique<VolPathIntegrator>(maxDepth, scene, std::move(camera),
                                               std::move(sampler),
                                               rrThreshold, lightStrategy, regularize);
}


// AOIntegrator Method Definitions
AOIntegrator::AOIntegrator(bool cosSample, Float maxDist, const Scene &scene,
                           std::unique_ptr<const Camera> c,
                           SamplerHandle sampler, SpectrumHandle illuminant)
    : RayIntegrator(scene, std::move(c), std::move(sampler)),
      cosSample(cosSample), maxDist(maxDist), illuminant(illuminant) {}

SampledSpectrum AOIntegrator::Li(
    RayDifferential ray, const SampledWavelengths &lambda,
    SamplerHandle sampler, MemoryArena &arena,
    MaterialBuffer &materialBuffer,
    pstd::optional<VisibleSurface> *visibleSurface) const {
    ProfilerScope p(ProfilePhase::RayIntegratorLi);
    SampledSpectrum L(0.f);

    // Intersect _ray_ with scene and store intersection in _isect_
    pstd::optional<ShapeIntersection> si;
 retry:
    si = scene.Intersect(ray);
    if (si) {
        SurfaceInteraction &isect = si->intr;
        isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler);
        if (isect.bsdf == nullptr) {
            isect.SkipIntersection(&ray, si->tHit);
            goto retry;
        }

        // Compute coordinate frame based on true geometry, not shading
        // geometry.
        Normal3f n = FaceForward(isect.n, -ray.d);
        Vector3f s = Normalize(isect.dpdu);
        Vector3f t = Cross(isect.n, s);

        Vector3f wi;
        Float pdf;
        Point2f u = sampler.Get2D();
        if (cosSample) {
            wi = SampleCosineHemisphere(u);
            pdf = CosineHemispherePDF(std::abs(wi.z));
        } else {
            wi = SampleUniformHemisphere(u);
            pdf = UniformHemispherePDF();
        }
        if (pdf == 0)
            return SampledSpectrum(0.);

        Frame f = Frame::FromZ(n);
        wi = f.FromLocal(wi);

        // Divide by pi so that fully visible is one.
        Ray r = isect.SpawnRay(wi);
        if (!scene.IntersectP(r, maxDist))
            return illuminant.Sample(lambda) * SampledSpectrum(Dot(wi, n) / (Pi * pdf));
    }
    return SampledSpectrum(0.);
}

std::string AOIntegrator::ToString() const {
    return StringPrintf("[ AOIntegrator cosSample: %s maxDist: %f illuminant: %s ]",
                        cosSample, maxDist, illuminant);
}

std::unique_ptr<AOIntegrator> AOIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene, SpectrumHandle illuminant,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    bool cosSample = dict.GetOneBool("cossample", true);
    Float maxDist = dict.GetOneFloat("maxdistance", Infinity);
    return std::make_unique<AOIntegrator>(cosSample, maxDist, scene, std::move(camera),
                                          std::move(sampler), illuminant);
}

// EndpointInteraction Declarations
struct EndpointInteraction : Interaction {
    union {
        const Camera *camera;
        LightHandle light;
    };
    // EndpointInteraction Public Methods
    EndpointInteraction() : Interaction(), light(nullptr) {}
    EndpointInteraction(const Interaction &it, const Camera *camera)
        : Interaction(it), camera(camera) {}
    EndpointInteraction(const Camera *camera, const Ray &ray)
        : Interaction(ray.o, ray.time, ray.medium), camera(camera) {}
    EndpointInteraction(LightHandle light, const Ray &r, const Interaction &intr)
        : Interaction(intr), light(light) {
    }
    EndpointInteraction(LightHandle light, const Ray &r)
        : Interaction(r.o, r.time, r.medium), light(light) {
    }
    EndpointInteraction(const Interaction &it, LightHandle light)
        : Interaction(it), light(light) {
    }
    EndpointInteraction(const Ray &ray)
        : Interaction(ray(1), Normal3f(-ray.d), ray.time, ray.medium),
          light(nullptr) {
    }

    EndpointInteraction(const EndpointInteraction &ei)
        : Interaction(ei), camera(ei.camera) {
        static_assert(sizeof(LightHandle) == sizeof(Camera *),
                      "Expect both union members have same size");
    }
};

// BDPT Helper Definitions
enum class VertexType { Camera, Light, Surface, Medium };
struct Vertex;
template <typename Type>
class ScopedAssignment {
  public:
    // ScopedAssignment Public Methods
    ScopedAssignment(Type *target = nullptr, Type value = Type())
        : target(target) {
        if (target) {
            backup = *target;
            *target = value;
        }
    }
    ~ScopedAssignment() {
        if (target) *target = backup;
    }
    ScopedAssignment(const ScopedAssignment &) = delete;
    ScopedAssignment &operator=(const ScopedAssignment &) = delete;
    ScopedAssignment &operator=(ScopedAssignment &&other) {
        if (target) *target = backup;
        target = other.target;
        backup = other.backup;
        other.target = nullptr;
        return *this;
    }

  private:
    Type *target, backup;
};

inline Float InfiniteLightDensity(
    const Scene &scene, const FixedLightSampler &lightSampler,
    const Vector3f &w) {
    Float pdf = 0;
    for (const auto &light : scene.infiniteLights)
        pdf += light.Pdf_Li(Interaction(), -w) * lightSampler.PDF(light);
    return pdf;
}

struct Vertex {
    // Vertex Public Data
    VertexType type;
    SampledSpectrum beta;
    union {
        EndpointInteraction ei;
        MediumInteraction mi;
        SurfaceInteraction si;
    };
    bool delta = false;
    Float pdfFwd = 0, pdfRev = 0;

    // Vertex Public Methods
    Vertex() : ei() {}
    Vertex(VertexType type, const EndpointInteraction &ei,
           const SampledSpectrum &beta)
        : type(type), beta(beta), ei(ei) {}
    Vertex(const SurfaceInteraction &si, const SampledSpectrum &beta)
        : type(VertexType::Surface), beta(beta), si(si) {}

    // Need to define these two to make compilers happy with the non-POD
    // objects in the anonymous union above.
    Vertex(const Vertex &v) { memcpy(this, &v, sizeof(Vertex)); }
    Vertex &operator=(const Vertex &v) {
        memcpy(this, &v, sizeof(Vertex));
        return *this;
    }

    static inline Vertex CreateCamera(const Camera *camera, const Ray &ray,
                                      const SampledSpectrum &beta);
    static inline Vertex CreateCamera(const Camera *camera,
                                      const Interaction &it,
                                      const SampledSpectrum &beta);
    static inline Vertex CreateLight(LightHandle light, const Ray &ray,
                                     const SampledSpectrum &Le,
                                     Float pdf);
    static inline Vertex CreateLight(LightHandle light, const Ray &ray,
                                     const Interaction &intr,
                                     const SampledSpectrum &Le, Float pdf);
    static inline Vertex CreateLight(const EndpointInteraction &ei,
                                     const SampledSpectrum &beta, Float pdf);
    static inline Vertex CreateMedium(const MediumInteraction &mi,
                                      const SampledSpectrum &beta, Float pdf,
                                      const Vertex &prev);
    static inline Vertex CreateSurface(const SurfaceInteraction &si,
                                       const SampledSpectrum &beta, Float pdf,
                                       const Vertex &prev);
    Vertex(const MediumInteraction &mi, const SampledSpectrum &beta)
        : type(VertexType::Medium), beta(beta), mi(mi) {}
    const Interaction &GetInteraction() const {
        switch (type) {
        case VertexType::Medium:
            return mi;
        case VertexType::Surface:
            return si;
        default:
            return ei;
        }
    }
    Point3f p() const { return GetInteraction().p(); }
    Float time() const { return GetInteraction().time; }
    const Normal3f &ng() const { return GetInteraction().n; }
    const Normal3f &ns() const {
        if (type == VertexType::Surface)
            return si.shading.n;
        else
            return GetInteraction().n;
    }
    bool IsOnSurface() const { return ng() != Normal3f(); }
    SampledSpectrum f(const Vertex &next, TransportMode mode) const {
        Vector3f wi = next.p() - p();
        if (LengthSquared(wi) == 0) return SampledSpectrum(0.);
        wi = Normalize(wi);
        switch (type) {
        case VertexType::Surface:
            return si.bsdf->f(si.wo, wi);
        case VertexType::Medium:
            return SampledSpectrum(mi.phase->p(mi.wo, wi));
        default:
            LOG_FATAL("Vertex::f(): Unimplemented");
            return SampledSpectrum(0.f);
        }
    }
    bool IsConnectible() const {
        switch (type) {
        case VertexType::Medium:
            return true;
        case VertexType::Light:
            return ei.light->type != LightType::DeltaDirection;
        case VertexType::Camera:
            return true;
        case VertexType::Surface:
            return si.bsdf->IsNonSpecular();
        }
        LOG_FATAL("Unhandled vertex type in IsConnectable()");
    }
    bool IsLight() const {
        return type == VertexType::Light ||
               (type == VertexType::Surface && si.areaLight);
    }
    bool IsDeltaLight() const {
        return type == VertexType::Light && ei.light &&
               pbrt::IsDeltaLight(ei.light->type);
    }
    bool IsInfiniteLight() const {
        return type == VertexType::Light &&
               (!ei.light || ei.light->type == LightType::Infinite ||
                ei.light->type == LightType::DeltaDirection);
    }
    SampledSpectrum Le(const Scene &scene, const Vertex &v,
                       const SampledWavelengths &lambda) const {
        if (!IsLight()) return SampledSpectrum(0.f);
        Vector3f w = v.p() - p();
        if (LengthSquared(w) == 0) return SampledSpectrum(0.);
        w = Normalize(w);
        if (IsInfiniteLight()) {
            // Return emitted radiance for infinite light sources
            SampledSpectrum Le(0.f);
            for (const auto &light : scene.infiniteLights)
                Le += light.Le(Ray(p(), -w), lambda);
            return Le;
        } else {
            return si.areaLight ? si.areaLight.L(si, w, lambda) : SampledSpectrum(0.);
        }
    }

    std::string ToString() const {
        std::string s = std::string("[ Vertex type: ");
        switch (type) {
        case VertexType::Camera:
            s += "camera";
            break;
        case VertexType::Light:
            s += "light";
            break;
        case VertexType::Surface:
            s += "surface";
            break;
        case VertexType::Medium:
            s += "medium";
            break;
        }
        s += StringPrintf(" connectible: %s p: %s ng: %s pdfFwd: %f pdfRev: %f beta: %s",
                          IsConnectible(), p(), ng(), pdfFwd, pdfRev, beta);
        switch (type) {
        case VertexType::Camera:
            // TODO
            break;
        case VertexType::Light:
            // TODO
            break;
        case VertexType::Surface:
            s += std::string("\n  bsdf: ") + si.bsdf->ToString();
            break;
        case VertexType::Medium:
            s += std::string("\n  phase: ") + mi.phase->ToString();
            break;
        }
        s += std::string(" ]");
        return s;
    }
    Float ConvertDensity(Float pdf, const Vertex &next) const {
        // Return solid angle density if _next_ is an infinite area light
        if (next.IsInfiniteLight()) return pdf;
        Vector3f w = next.p() - p();
        if (LengthSquared(w) == 0) return 0;
        Float invDist2 = 1 / LengthSquared(w);
        if (next.IsOnSurface())
            pdf *= AbsDot(next.ng(), w * std::sqrt(invDist2));
        return pdf * invDist2;
    }
    Float PDF(const Scene &scene, const Vertex *prev,
              const Vertex &next) const {
        if (type == VertexType::Light) return PdfLight(scene, next);
        // Compute directions to preceding and next vertex
        Vector3f wn = next.p() - p();
        if (LengthSquared(wn) == 0) return 0;
        wn = Normalize(wn);
        Vector3f wp;
        if (prev) {
            wp = prev->p() - p();
            if (LengthSquared(wp) == 0) return 0;
            wp = Normalize(wp);
        } else
            CHECK(type == VertexType::Camera);

        // Compute directional density depending on the vertex types
        Float pdf = 0, unused;
        if (type == VertexType::Camera)
            ei.camera->Pdf_We(ei.SpawnRay(wn), &unused, &pdf);
        else if (type == VertexType::Surface)
            pdf = si.bsdf->PDF(wp, wn);
        else if (type == VertexType::Medium)
            pdf = mi.phase->p(wp, wn);
        else
            LOG_FATAL("Vertex::PDF(): Unimplemented");

        // Return probability per unit area at vertex _next_
        return ConvertDensity(pdf, next);
    }
    Float PdfLight(const Scene &scene, const Vertex &v) const {
        Vector3f w = v.p() - p();
        Float invDist2 = 1 / LengthSquared(w);
        w *= std::sqrt(invDist2);
        Float pdf;
        if (IsInfiniteLight()) {
            // Compute planar sampling density for infinite light sources
            Point3f worldCenter;
            Float worldRadius;
            scene.WorldBound().BoundingSphere(&worldCenter, &worldRadius);
            pdf = 1 / (Pi * worldRadius * worldRadius);
        } else if (IsOnSurface()) {
            if (type == VertexType::Light)
                CHECK(ei.light.Is<DiffuseAreaLight>()); // since that's all we've got currently...

            LightHandle light = (type == VertexType::Light) ? ei.light : si.areaLight;
            Float pdfPos, pdfDir;
            light.Pdf_Le(ei, w, &pdfPos, &pdfDir);
            pdf = pdfDir * invDist2;
        } else {
            // Get pointer _light_ to the light source at the vertex
            CHECK(type == VertexType::Light);
            CHECK(ei.light != nullptr);
            LightHandle light = ei.light;

            // Compute sampling density for non-infinite light sources
            Float pdfPos, pdfDir;
            light.Pdf_Le(Ray(p(), w, time()), &pdfPos, &pdfDir);
            pdf = pdfDir * invDist2;
        }
        if (v.IsOnSurface()) pdf *= AbsDot(v.ng(), w);
        return pdf;
    }
    Float PdfLightOrigin(const Scene &scene, const Vertex &v,
                         const FixedLightSampler &lightSampler) {
        Vector3f w = v.p() - p();
        if (LengthSquared(w) == 0) return 0.;
        w = Normalize(w);
        if (IsInfiniteLight()) {
            // Return solid angle density for infinite light sources
            return InfiniteLightDensity(scene, lightSampler, w);
        } else if (IsOnSurface()) {
            if (type == VertexType::Light)
                CHECK(ei.light.Is<DiffuseAreaLight>()); // since that's all we've got currently...

            LightHandle light = (type == VertexType::Light) ? ei.light : si.areaLight;
            Float pdfChoice = lightSampler.PDF(light);
            Float pdfPos, pdfDir;
            light.Pdf_Le(ei, w, &pdfPos, &pdfDir);
            return pdfPos * pdfChoice;
        } else {
            // Return solid angle density for non-infinite light sources
            Float pdfPos, pdfDir;

            // Get pointer _light_ to the light source at the vertex
            CHECK(IsLight());
            LightHandle light = type == VertexType::Light
                                     ? ei.light
                                     : si.areaLight;
            CHECK(light != nullptr);

            // Compute the discrete probability of sampling _light_, _pdfChoice_
            Float pdfChoice = lightSampler.PDF(light);
            light.Pdf_Le(Ray(p(), w, time()), &pdfPos, &pdfDir);
            return pdfPos * pdfChoice;
        }
    }
};

int GenerateCameraSubpath(const RayDifferential &ray, const Scene &scene,
                          const SampledWavelengths &lambda,
                          SamplerHandle sampler, MemoryArena &arena,
                          MaterialBuffer &materialBuffer, int maxDepth,
                          const Camera &camera, Vertex *path, bool regularize);

int GenerateLightSubpath(
    const Scene &scene, const SampledWavelengths &lambda, SamplerHandle sampler,
    const Camera &camera, MemoryArena &arena, MaterialBuffer &materialBuffer,
    int maxDepth, Float time,
    const FixedLightSampler &lightSampler, Vertex *path, bool regularize);

SampledSpectrum ConnectBDPT(
    const Scene &scene, const SampledWavelengths &lambda,
    Vertex *lightVertices, Vertex *cameraVertices, int s,
    int t, const FixedLightSampler &lightSampler,
    const Camera &camera, SamplerHandle sampler, pstd::optional<Point2f> *pRaster,
    Float *misWeight = nullptr);

// Vertex Inline Method Definitions
inline Vertex Vertex::CreateCamera(const Camera *camera, const Ray &ray,
                                   const SampledSpectrum &beta) {
    return Vertex(VertexType::Camera, EndpointInteraction(camera, ray), beta);
}

inline Vertex Vertex::CreateCamera(const Camera *camera, const Interaction &it,
                                   const SampledSpectrum &beta) {
    return Vertex(VertexType::Camera, EndpointInteraction(it, camera), beta);
}

inline Vertex Vertex::CreateLight(LightHandle light, const Ray &ray,
                                  const SampledSpectrum &Le, Float pdf) {
    Vertex v(VertexType::Light, EndpointInteraction(light, ray), Le);
    v.pdfFwd = pdf;
    return v;
}

inline Vertex Vertex::CreateLight(LightHandle light, const Ray &ray,
                                  const Interaction &intr,
                                  const SampledSpectrum &Le, Float pdf) {
    Vertex v(VertexType::Light, EndpointInteraction(light, ray, intr), Le);
    v.pdfFwd = pdf;
    return v;
}

inline Vertex Vertex::CreateSurface(const SurfaceInteraction &si,
                                    const SampledSpectrum &beta, Float pdf,
                                    const Vertex &prev) {
    Vertex v(si, beta);
    v.pdfFwd = prev.ConvertDensity(pdf, v);
    return v;
}

inline Vertex Vertex::CreateMedium(const MediumInteraction &mi,
                                   const SampledSpectrum &beta, Float pdf,
                                   const Vertex &prev) {
    Vertex v(mi, beta);
    v.pdfFwd = prev.ConvertDensity(pdf, v);
    return v;
}

inline Vertex Vertex::CreateLight(const EndpointInteraction &ei,
                                  const SampledSpectrum &beta, Float pdf) {
    Vertex v(VertexType::Light, ei, beta);
    v.pdfFwd = pdf;
    return v;
}

// BDPT Forward Declarations
int RandomWalk(const Scene &scene, const SampledWavelengths &lambda,
               RayDifferential ray, SamplerHandle sampler, const Camera &camera,
               MemoryArena &arena, MaterialBuffer &materialBuffer,
               SampledSpectrum beta, Float pdf,
               int maxDepth, TransportMode mode, Vertex *path,
               bool regularize);

// BDPT Utility Functions
int GenerateCameraSubpath(const RayDifferential &ray, const Scene &scene,
                          const SampledWavelengths &lambda,
                          SamplerHandle sampler, MemoryArena &arena,
                          MaterialBuffer &materialBuffer, int maxDepth,
                          const Camera &camera, Vertex *path, bool regularize) {
    if (maxDepth == 0) return 0;
    ProfilerScope _(ProfilePhase::BDPTGenerateSubpath);
    SampledSpectrum beta(1.f);

    // Generate first vertex on camera subpath and start random walk
    Float pdfPos, pdfDir;
    path[0] = Vertex::CreateCamera(&camera, ray, beta);
    camera.Pdf_We(ray, &pdfPos, &pdfDir);
    VLOG(2, "Starting camera subpath. Ray: %s, beta %s, pdfPos %f, pdfDir %f",
         ray, beta, pdfPos, pdfDir);
    return RandomWalk(scene, lambda, ray, sampler, camera, arena, materialBuffer, beta, pdfDir,
                      maxDepth - 1, TransportMode::Radiance, path + 1,
                      regularize) +
           1;
}

int GenerateLightSubpath(
    const Scene &scene, const SampledWavelengths &lambda, SamplerHandle sampler,
    const Camera &camera, MemoryArena &arena, MaterialBuffer &materialBuffer,
    int maxDepth, Float time,
    const FixedLightSampler &lightSampler, Vertex *path, bool regularize) {
    if (maxDepth == 0) return 0;
    ProfilerScope _(ProfilePhase::BDPTGenerateSubpath);
    // Sample initial ray for light subpath
    Float lightPDF;
    LightHandle light = lightSampler.Sample(sampler.Get1D(), &lightPDF);
    if (lightPDF == 0) return 0;

    pstd::optional<LightLeSample> les = light.Sample_Le(sampler.Get2D(), sampler.Get2D(),
                                                         lambda, time);
    if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L) return 0;
    RayDifferential ray(les->ray);

    // Generate first vertex on light subpath and start random walk
    path[0] = les->intr ?
        Vertex::CreateLight(light, ray, *les->intr, les->L, les->pdfPos * lightPDF) :
        Vertex::CreateLight(light, ray, les->L, les->pdfPos * lightPDF);
    SampledSpectrum beta = les->L * les->AbsCosTheta(ray.d) /
        (lightPDF * les->pdfPos * les->pdfDir);
    VLOG(2, "Starting light subpath. Ray: %s, Le %s, beta %s, pdfPos %f, pdfDir %f",
         ray, les->L, beta, les->pdfPos, les->pdfDir);
    int nVertices =
        RandomWalk(scene, lambda, ray, sampler, camera, arena, materialBuffer, beta, les->pdfDir,
                   maxDepth - 1, TransportMode::Importance, path + 1,
                   regularize);

    // Correct subpath sampling densities for infinite area lights
    if (path[0].IsInfiniteLight()) {
        // Set spatial density of _path[1]_ for infinite area light
        if (nVertices > 0) {
            path[1].pdfFwd = les->pdfPos;
            if (path[1].IsOnSurface())
                path[1].pdfFwd *= AbsDot(ray.d, path[1].ng());
        }

        // Set spatial density of _path[0]_ for infinite area light
        path[0].pdfFwd =
            InfiniteLightDensity(scene, lightSampler, ray.d);
    }
    return nVertices + 1;
}

int RandomWalk(const Scene &scene, const SampledWavelengths &lambda,
               RayDifferential ray, SamplerHandle sampler, const Camera &camera,
               MemoryArena &arena, MaterialBuffer &materialBuffer,
               SampledSpectrum beta, Float pdf,
               int maxDepth, TransportMode mode, Vertex *path,
               bool regularize) {
    if (maxDepth == 0) return 0;
    int bounces = 0;
    bool anyNonSpecularBounces = false;
    // Declare variables for forward and reverse probability densities
    Float pdfFwd = pdf, pdfRev = 0;
    while (true) {
        // Attempt to create the next subpath vertex in _path_
        MediumInteraction mi;

        VLOG(2, "Random walk. Bounces %d, beta %s, pdfFwd %f, pdfRef %f",
             bounces, beta, pdfFwd, pdfRev);
        // Trace a ray and sample the medium, if any
        pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
        if (ray.medium) beta *= ray.medium->Sample(ray, si ? si->tHit : Infinity,
                                                   sampler, lambda, arena, &mi);
        if (!beta) break;
        Vertex &vertex = path[bounces], &prev = path[bounces - 1];
        if (mi.IsValid()) {
            // Record medium interaction in _path_ and compute forward density
            vertex = Vertex::CreateMedium(mi, beta, pdfFwd, prev);
            if (++bounces >= maxDepth) break;

            // Sample direction and compute reverse density at preceding vertex
            pstd::optional<PhaseFunctionSample> ps =
                mi.phase->Sample_p(-ray.d, sampler.Get2D());
            if (!ps || ps->pdf == 0)
                break;
            pdfFwd = pdfRev = ps->pdf;
            beta *= ps->p / pdfFwd;
            ray = mi.SpawnRay(ps->wi);
            anyNonSpecularBounces = true;
        } else {
            // Handle surface interaction for path generation
            if (!si) {
                // Capture escaped rays when tracing from the camera
                if (mode == TransportMode::Radiance) {
                    vertex = Vertex::CreateLight(EndpointInteraction(ray), beta,
                                                 pdfFwd);
                    ++bounces;
                }
                break;
            }

            // Compute scattering functions for _mode_ and skip over medium
            // boundaries
            SurfaceInteraction &isect = si->intr;
            isect.ComputeScatteringFunctions(ray, lambda, camera, materialBuffer, sampler, mode);
            if (isect.bsdf == nullptr) {
                isect.SkipIntersection(&ray, si->tHit);
                continue;
            }

            if (regularize && anyNonSpecularBounces) {
                ++regularizedBSDFs;
                isect.bsdf->Regularize(materialBuffer);
            }
            ++totalBSDFs;

            // Initialize _vertex_ with surface intersection information
            vertex = Vertex::CreateSurface(isect, beta, pdfFwd, prev);
            if (++bounces >= maxDepth) break;

            // Sample BSDF at current vertex and compute reverse probability
            Vector3f wo = isect.wo;
            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = isect.bsdf->Sample_f(wo, u, sampler.Get2D());
            if (!bs) break;
            pdfFwd = bs->pdf;
            anyNonSpecularBounces |= !bs->IsSpecular();
            VLOG(2, "Random walk sampled dir %s, f: %s, pdfFwd %f", bs->wi, bs->f, pdfFwd);
            beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
            VLOG(2, "Random walk beta now %s", beta);
            pdfRev = isect.bsdf->PDF(bs->wi, wo);
            if (bs->IsSpecular()) {
                vertex.delta = true;
                pdfRev = pdfFwd = 0;
            }
            VLOG(2, "Random walk beta after shading normal correction %s", beta);
            ray = isect.SpawnRay(ray, bs->wi, bs->flags);
        }

        // Compute reverse area density at preceding vertex
        prev.pdfRev = vertex.ConvertDensity(pdfRev, prev);
    }
    return bounces;
}

SampledSpectrum G(const Scene &scene, SamplerHandle sampler, const Vertex &v0,
                  const Vertex &v1, const SampledWavelengths &lambda) {
    Vector3f d = v0.p() - v1.p();
    Float g = 1 / LengthSquared(d);
    d *= std::sqrt(g);
    if (v0.IsOnSurface()) g *= AbsDot(v0.ns(), d);
    if (v1.IsOnSurface()) g *= AbsDot(v1.ns(), d);
    return g * Tr(scene, lambda, sampler, v0.GetInteraction(),
                  v1.GetInteraction());
}

Float MISWeight(const Scene &scene, Vertex *lightVertices,
                Vertex *cameraVertices, Vertex &sampled, int s, int t,
                const FixedLightSampler &lightSampler) {
    if (s + t == 2) return 1;
    Float sumRi = 0;
    // Define helper function _remap0_ that deals with Dirac delta functions
    auto remap0 = [](Float f) -> Float { return f != 0 ? f : 1; };

    // Temporarily update vertex properties for current strategy

    // Look up connection vertices and their predecessors
    Vertex *qs = s > 0 ? &lightVertices[s - 1] : nullptr,
           *pt = t > 0 ? &cameraVertices[t - 1] : nullptr,
           *qsMinus = s > 1 ? &lightVertices[s - 2] : nullptr,
           *ptMinus = t > 1 ? &cameraVertices[t - 2] : nullptr;

    // Update sampled vertex for $s=1$ or $t=1$ strategy
    ScopedAssignment<Vertex> a1;
    if (s == 1)
        a1 = {qs, sampled};
    else if (t == 1)
        a1 = {pt, sampled};

    // Mark connection vertices as non-degenerate
    ScopedAssignment<bool> a2, a3;
    if (pt != nullptr) a2 = {&pt->delta, false};
    if (qs != nullptr) a3 = {&qs->delta, false};

    // Update reverse density of vertex $\pt{}_{t-1}$
    ScopedAssignment<Float> a4;
    if (pt != nullptr)
        a4 = {&pt->pdfRev, s > 0 ? qs->PDF(scene, qsMinus, *pt)
                                 : pt->PdfLightOrigin(scene, *ptMinus, lightSampler)};

    // Update reverse density of vertex $\pt{}_{t-2}$
    ScopedAssignment<Float> a5;
    if (ptMinus != nullptr)
        a5 = {&ptMinus->pdfRev, s > 0 ? pt->PDF(scene, qs, *ptMinus)
                                      : pt->PdfLight(scene, *ptMinus)};

    // Update reverse density of vertices $\pq{}_{s-1}$ and $\pq{}_{s-2}$
    ScopedAssignment<Float> a6;
    if (qs != nullptr) a6 = {&qs->pdfRev, pt->PDF(scene, ptMinus, *qs)};
    ScopedAssignment<Float> a7;
    if (qsMinus != nullptr) a7 = {&qsMinus->pdfRev, qs->PDF(scene, pt, *qsMinus)};

    // Consider hypothetical connection strategies along the camera subpath
    Float ri = 1;
    for (int i = t - 1; i > 0; --i) {
        ri *=
            remap0(cameraVertices[i].pdfRev) / remap0(cameraVertices[i].pdfFwd);
        if (!cameraVertices[i].delta && !cameraVertices[i - 1].delta)
            sumRi += ri;
    }

    // Consider hypothetical connection strategies along the light subpath
    ri = 1;
    for (int i = s - 1; i >= 0; --i) {
        ri *= remap0(lightVertices[i].pdfRev) / remap0(lightVertices[i].pdfFwd);
        bool deltaLightvertex = i > 0 ? lightVertices[i - 1].delta
                                      : lightVertices[0].IsDeltaLight();
        if (!lightVertices[i].delta && !deltaLightvertex) sumRi += ri;
    }
    return 1 / (1 + sumRi);
}

// BDPT Method Definitions
inline int BufferIndex(int s, int t) {
    int above = s + t - 2;
    return s + above * (5 + above) / 2;
}

void BDPTIntegrator::Render() {
    lightSampler = std::make_unique<PowerLightSampler>(scene.lights, Allocator());

    // Allocate buffers for debug visualization
    if (visualizeStrategies || visualizeWeights) {
        const int bufferCount = (1 + maxDepth) * (6 + maxDepth) / 2;
        weightFilms.resize(bufferCount);
        for (int depth = 0; depth <= maxDepth; ++depth) {
            for (int s = 0; s <= depth + 2; ++s) {
                int t = depth + 2 - s;
                if (t == 0 || (s == 1 && t == 1)) continue;

                std::string filename =
                    StringPrintf("bdpt_d%02i_s%02i_t%02i.exr", depth, s, t);

                weightFilms[BufferIndex(s, t)] = std::make_unique<RGBFilm>(
                    camera->film->fullResolution,
                    Bounds2i(Point2i(0, 0), camera->film->fullResolution),
                    new BoxFilter,  // FIXME: leak
                    camera->film->diagonal * 1000, filename, 1.f, RGBColorSpace::sRGB);
            }
        }
    }

    RayIntegrator::Render();

    // Write buffers for debug visualization
    if (visualizeStrategies || visualizeWeights) {
        const Float invSampleCount = 1.0f / initialSampler.SamplesPerPixel();
        for (size_t i = 0; i < weightFilms.size(); ++i) {
            ImageMetadata metadata;
            if (weightFilms[i]) weightFilms[i]->WriteImage(metadata,
                                                           invSampleCount);
        }
        weightFilms.clear();
    }
}

SampledSpectrum BDPTIntegrator::Li(RayDifferential ray,
                                   const SampledWavelengths &lambda,
                                   SamplerHandle sampler,
                                   MemoryArena &arena,
                                   MaterialBuffer &materialBuffer,
                                   pstd::optional<VisibleSurface> *visibleSurface) const {
    // Trace the camera subpath
    Vertex *cameraVertices = arena.Alloc<Vertex[]>(maxDepth + 2);
    int nCamera = GenerateCameraSubpath(ray, scene, lambda, sampler, arena, materialBuffer,
                                        maxDepth + 2, *camera,
                                        cameraVertices, regularize);

    // Now trace the light subpath
    Vertex *lightVertices = arena.Alloc<Vertex[]>(maxDepth + 1);
    int nLight = GenerateLightSubpath(scene, lambda, sampler, *camera, arena, materialBuffer,
                                      maxDepth + 1,
                                      cameraVertices[0].time(), *lightSampler, lightVertices,
                                      regularize);

    // Execute all BDPT connection strategies
    SampledSpectrum L(0.f);
    for (int t = 1; t <= nCamera; ++t) {
        for (int s = 0; s <= nLight; ++s) {
            int depth = t + s - 2;
            if ((s == 1 && t == 1) || depth < 0 ||
                depth > maxDepth)
                continue;
            // Execute the $(s, t)$ connection strategy and
            // update _L_
            pstd::optional<Point2f> pFilmNew;
            Float misWeight = 0.f;
            SampledSpectrum Lpath = ConnectBDPT(scene, lambda, lightVertices, cameraVertices, s, t,
                                                *lightSampler, *camera, sampler,
                                                &pFilmNew, &misWeight);
            VLOG(2, "Connect bdpt s: %d, t: %d, Lpath: %s, misWeight: %f", s, t,
                 Lpath, misWeight);
            if (visualizeStrategies || visualizeWeights) {
                SampledSpectrum value;
                if (visualizeStrategies)
                    value =
                        misWeight == 0 ? SampledSpectrum(0.) :
                        Lpath / misWeight;
                if (visualizeWeights) value = Lpath;
                CHECK(pFilmNew.has_value());
                weightFilms[BufferIndex(s, t)]->AddSplat(*pFilmNew, value, lambda);
            }
            if (t != 1)
                L += Lpath;
            else if (Lpath) {
                CHECK(pFilmNew.has_value());
                camera->film->AddSplat(*pFilmNew, Lpath, lambda);
            }
        }
    }

    return L;
}


SampledSpectrum ConnectBDPT(
    const Scene &scene, const SampledWavelengths &lambda,
    Vertex *lightVertices, Vertex *cameraVertices, int s,
    int t, const FixedLightSampler &lightSampler,
    const Camera &camera, SamplerHandle sampler, pstd::optional<Point2f> *pRaster,
    Float *misWeightPtr) {
    ProfilerScope _(ProfilePhase::BDPTConnectSubpaths);
    SampledSpectrum L(0.f);
    // Ignore invalid connections related to infinite area lights
    if (t > 1 && s != 0 && cameraVertices[t - 1].type == VertexType::Light)
        return SampledSpectrum(0.f);

    // Perform connection and write contribution to _L_
    Vertex sampled;
    if (s == 0) {
        // Interpret the camera subpath as a complete path
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsLight()) L = pt.Le(scene, cameraVertices[t - 2], lambda) * pt.beta;
        DCHECK(!L.HasNaNs());
    } else if (t == 1) {
        // Sample a point on the camera and connect it to the light subpath
        const Vertex &qs = lightVertices[s - 1];
        if (qs.IsConnectible()) {
            pstd::optional<CameraWiSample> cs =
                camera.Sample_Wi(qs.GetInteraction(), sampler.Get2D(), lambda);
            if (cs) {
                *pRaster = cs->pRaster;
                // Initialize dynamically sampled vertex and _L_ for $t=1$ case
                sampled = Vertex::CreateCamera(&camera, cs->pLens, cs->Wi / cs->pdf);
                L = qs.beta * qs.f(sampled, TransportMode::Importance) * sampled.beta;
                if (qs.IsOnSurface()) L *= AbsDot(cs->wi, qs.ns());
                DCHECK(!L.HasNaNs());
                // Only check visibility after we know that the path would
                // make a non-zero contribution.
                if (L) L *= cs->Tr(scene, lambda, sampler);
            }
        }
    } else if (s == 1) {
        // Sample a point on a light and connect it to the camera subpath
        const Vertex &pt = cameraVertices[t - 1];
        if (pt.IsConnectible()) {
            Float lightPDF;
            LightHandle light = lightSampler.Sample(sampler.Get1D(), &lightPDF);

            pstd::optional<LightLiSample> lightWeight = light.Sample_Li(
                pt.GetInteraction(), sampler.Get2D(), lambda);
            if (lightWeight) {
                EndpointInteraction ei(lightWeight->pLight, light);
                sampled =
                    Vertex::CreateLight(ei, lightWeight->L / (lightWeight->pdf * lightPDF), 0);
                sampled.pdfFwd =
                    sampled.PdfLightOrigin(scene, pt, lightSampler);
                L = pt.beta * pt.f(sampled, TransportMode::Radiance) * sampled.beta;
                if (pt.IsOnSurface()) L *= AbsDot(lightWeight->wi, pt.ns());
                // Only check visibility if the path would carry radiance.
                if (L) L *= lightWeight->Tr(scene, lambda, sampler);
            }
        }
    } else {
        // Handle all other bidirectional connection cases
        const Vertex &qs = lightVertices[s - 1], &pt = cameraVertices[t - 1];
        if (qs.IsConnectible() && pt.IsConnectible()) {
            L = qs.beta * qs.f(pt, TransportMode::Importance) * pt.f(qs, TransportMode::Radiance) * pt.beta;
            VLOG(2, "General connect s: %d, t: %d, qs: %s, pt: %s, qs.f(pt): %s, pt.f(qs): %s, "
                 "G: %s, dist^2: %f", s, t, qs, pt, qs.f(pt, TransportMode::Importance),
                 pt.f(qs, TransportMode::Radiance), G(scene, sampler, qs, pt, lambda),
                 DistanceSquared(qs.p(), pt.p()));

            if (L) L *= G(scene, sampler, qs, pt, lambda);
        }
    }

    ++totalPaths;
    if (!L) ++zeroRadiancePaths;
    ReportValue(pathLength, s + t - 2);

    // Compute MIS weight for connection strategy
    Float misWeight =
        L ? MISWeight(scene, lightVertices, cameraVertices, sampled, s, t, lightSampler) : 0.f;
    VLOG(2,"MIS weight for (s,t) = (%d, %d) connection: %f", s, t, misWeight);
    DCHECK(!std::isnan(misWeight));
    L *= misWeight;
    if (misWeightPtr != nullptr) *misWeightPtr = misWeight;
    return L;
}

std::string BDPTIntegrator::ToString() const {
    return StringPrintf("[ BDPTIntegrator maxDepth: %d visualizeStrategies: %s "
                        "visualizeWeights: %s lightSampleStrategy: %s regularize: %s "
                        "lightSampler: %s ]", maxDepth, visualizeStrategies,
                        visualizeWeights, lightSampleStrategy, regularize, lightSampler);
}

std::unique_ptr<BDPTIntegrator> BDPTIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, SamplerHandle sampler, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    bool visualizeStrategies = dict.GetOneBool("visualizestrategies", false);
    bool visualizeWeights = dict.GetOneBool("visualizeweights", false);

    if ((visualizeStrategies || visualizeWeights) && maxDepth > 5) {
        Warning(loc,
            "visualizestrategies/visualizeweights was enabled, limiting "
            "maxdepth to 5");
        maxDepth = 5;
    }

    std::string lightStrategy = dict.GetOneString("lightsamplestrategy", "power");
    bool regularize = dict.GetOneBool("regularize", true);
    return std::make_unique<BDPTIntegrator>(
        scene, std::move(camera), std::move(sampler), maxDepth, visualizeStrategies,
        visualizeWeights, lightStrategy, regularize);
}


STAT_PERCENT("Integrator/Acceptance rate", acceptedMutations, totalMutations);

// MLTIntegrator Constants
const int MLTIntegrator::cameraStreamIndex = 0;
const int MLTIntegrator::lightStreamIndex = 1;
const int MLTIntegrator::connectionStreamIndex = 2;
const int MLTIntegrator::nSampleStreams = 3;

// MLT Method Definitions
SampledSpectrum MLTIntegrator::L(MemoryArena &arena, MaterialBuffer &materialBuffer,
                                 MLTSampler &sampler, int depth,
                                 Point2f *pRaster, SampledWavelengths *lambda) {
    sampler.StartStream(cameraStreamIndex);
    // Determine the number of available strategies and pick a specific one
    int s, t, nStrategies;
    if (depth == 0) {
        nStrategies = 1;
        s = 0;
        t = 2;
    } else {
        nStrategies = depth + 2;
        s = std::min<int>(sampler.Get1D() * nStrategies, nStrategies - 1);
        t = nStrategies - s;
    }

    if (PbrtOptions.disableWavelengthJitter)
        *lambda = camera->film->SampleWavelengths(0.5);
    else
        *lambda = camera->film->SampleWavelengths(sampler.Get1D());

    // Generate a camera subpath with exactly _t_ vertices
    Vertex *cameraVertices = arena.Alloc<Vertex[]>(t);
    Bounds2f sampleBounds = camera->film->SampleBounds();
    *pRaster = sampleBounds.Lerp(sampler.Get2D());
    CameraSample cameraSample;
    cameraSample.pFilm = *pRaster;
    cameraSample.time = sampler.Get1D();
    cameraSample.pLens = sampler.Get2D();
    pstd::optional<CameraRayDifferential> crd =
        camera->GenerateRayDifferential(cameraSample, *lambda);
    if (!crd || !crd->weight)
        return SampledSpectrum(0.f);
    Float rayDiffScale =
        std::max<Float>(.125, 1 / std::sqrt((Float)sampler.SamplesPerPixel()));
    crd->ray.ScaleDifferentials(rayDiffScale);

    if (GenerateCameraSubpath(crd->ray, scene, *lambda, &sampler, arena, materialBuffer,
                              t, *camera,
                              cameraVertices, regularize) != t)
        return SampledSpectrum(0.f);

    // Generate a light subpath with exactly _s_ vertices
    sampler.StartStream(lightStreamIndex);
    Vertex *lightVertices = arena.Alloc<Vertex[]>(s);
    if (GenerateLightSubpath(scene, *lambda, &sampler, *camera, arena, materialBuffer, s,
                             cameraVertices[0].time(),
                             *lightSampler, lightVertices, regularize) != s)
        return SampledSpectrum(0.f);

    // Execute connection strategy and return the radiance estimate
    sampler.StartStream(connectionStreamIndex);
    pstd::optional<Point2f> pRasterNew;
    SampledSpectrum L = ConnectBDPT(scene, *lambda, lightVertices, cameraVertices, s, t,
                                    *lightSampler, *camera, &sampler, &pRasterNew) *
        nStrategies;
    if (pRasterNew.has_value())
        *pRaster = *pRasterNew;
    return L;
}

void MLTIntegrator::Render() {
    lightSampler = std::make_unique<PowerLightSampler>(scene.lights, Allocator());

    StatsSetImageResolution(camera->film->pixelBounds);
    StatsSetPixelStatsBaseName(RemoveExtension(camera->film->filename).c_str());

    if (PbrtOptions.debugStart.has_value()) {
        std::vector<std::string> c = SplitString(*PbrtOptions.debugStart, ',');
        if (c.empty())
            ErrorExit("Didn't find comma-separated values after --debugstart: %s",
                      *PbrtOptions.debugStart);

        int depth;
        if (!Atoi(c[0], &depth))
            ErrorExit("Unable to decode first --debugstart value: %s", c[0]);

        pstd::span<const std::string> span = pstd::MakeSpan(c);
        span.remove_prefix(1);
        DebugMLTSampler sampler = DebugMLTSampler::Create(span, nSampleStreams);

        Point2f pRaster;
        SampledWavelengths lambda;
        MemoryArena arena;
        MaterialBuffer materialBuffer(16384);
        (void)L(arena, materialBuffer, sampler, depth, &pRaster, &lambda);
        return;
    }

    thread_local MLTSampler *threadSampler = nullptr;
    thread_local int threadDepth;
    CheckCallbackScope _([&]() -> std::string {
            return StringPrintf("Rendering failed. Debug with --debugstart %d,%s\"\n",
                                threadDepth, threadSampler->DumpState());
        });

    // Generate bootstrap samples and compute normalization constant $b$
    Timer timer;
    int nBootstrapSamples = nBootstrap * (maxDepth + 1);
    std::vector<Float> bootstrapWeights(nBootstrapSamples, 0);
    if (!scene.lights.empty()) {
        std::vector<MemoryArena> bootstrapThreadArenas(MaxThreadIndex());
        std::vector<MaterialBuffer> bootstrapMaterialBuffers(MaxThreadIndex());
        for (auto &m : bootstrapMaterialBuffers)
            m = MaterialBuffer(16384);

        ParallelFor(0, nBootstrap, [&](int64_t start, int64_t end) {
            MemoryArena &arena = bootstrapThreadArenas[ThreadIndex];
            MaterialBuffer &materialBuffer = bootstrapMaterialBuffers[ThreadIndex];
            for (int64_t i = start; i < end; ++i) {
                // Generate _i_th bootstrap sample
                for (int depth = 0; depth <= maxDepth; ++depth) {
                    int rngIndex = i * (maxDepth + 1) + depth;
                    MLTSampler sampler(mutationsPerPixel, rngIndex, sigma,
                                       largeStepProbability, nSampleStreams);
                    threadSampler = &sampler;
                    threadDepth = depth;

                    Point2f pRaster;
                    SampledWavelengths lambda;
                    bootstrapWeights[rngIndex] =
                        L(arena, materialBuffer, sampler, depth, &pRaster, &lambda).Average();

                    arena.Reset();
                    materialBuffer.Reset();
                }
            }
        }, "Generating bootstrap paths");
    }
    Distribution1D bootstrap(bootstrapWeights);
    Float b = bootstrap.funcInt * (maxDepth + 1);

    // Run _nChains_ Markov chains in parallel
    Film &film = *camera->film;
    int64_t nTotalMutations =
        (int64_t)mutationsPerPixel * (int64_t)film.SampleBounds().Area();
    if (!scene.lights.empty()) {
        std::vector<MemoryArena> threadArenas(MaxThreadIndex());
        std::vector<MaterialBuffer> threadMaterialBuffers(MaxThreadIndex());
        for (auto &m : threadMaterialBuffers)
            m = MaterialBuffer(16384);
        ParallelFor(0, nChains, [&](int i) {
            int64_t nChainMutations =
                std::min((i + 1) * nTotalMutations / nChains, nTotalMutations) -
                i * nTotalMutations / nChains;
            // Follow {i}th Markov chain for _nChainMutations_
            MemoryArena &arena = threadArenas[ThreadIndex];
            MaterialBuffer &materialBuffer = threadMaterialBuffers[ThreadIndex];

            // Select initial state from the set of bootstrap samples
            RNG rng(i);
            int bootstrapIndex = bootstrap.SampleDiscrete(rng.Uniform<Float>());
            int depth = bootstrapIndex % (maxDepth + 1);
            threadDepth = depth;

            // Initialize local variables for selected state
            MLTSampler sampler(mutationsPerPixel, bootstrapIndex, sigma,
                               largeStepProbability, nSampleStreams);
            threadSampler = &sampler;

            Point2f pCurrent;
            SampledWavelengths lambdaCurrent;
            SampledSpectrum LCurrent =
                L(arena, materialBuffer, sampler, depth, &pCurrent, &lambdaCurrent);

            // Run the Markov chain for _nChainMutations_ steps
            for (int64_t j = 0; j < nChainMutations; ++j) {
                StatsReportPixelStart(Point2i(pCurrent));
                sampler.StartIteration();
                Point2f pProposed;
                SampledWavelengths lambdaProposed;
                SampledSpectrum LProposed =
                    L(arena, materialBuffer, sampler, depth, &pProposed, &lambdaProposed);
                // Compute acceptance probability for proposed sample
                Float accept = std::min<Float>(1, LProposed.Average() / LCurrent.Average());

                // Splat both current and proposed samples to _film_
                if (accept > 0)
                    film.AddSplat(pProposed,
                                  LProposed * accept / LProposed.Average(),
                                  lambdaProposed);
                film.AddSplat(pCurrent,
                              LCurrent * (1 - accept) / LCurrent.Average(),
                              lambdaCurrent);

                // Accept or reject the proposal
                if (rng.Uniform<Float>() < accept) {
                    StatsReportPixelEnd(Point2i(pCurrent));
                    StatsReportPixelStart(Point2i(pProposed));
                    pCurrent = pProposed;
                    LCurrent = LProposed;
                    lambdaCurrent = lambdaProposed;
                    sampler.Accept();
                    ++acceptedMutations;
                } else
                    sampler.Reject();
                ++totalMutations;
                arena.Reset();
                materialBuffer.Reset();

                StatsReportPixelEnd(Point2i(pCurrent));
            }
        }, "Rendering");
    }

    // Store final image computed with MLT
    ImageMetadata metadata;
    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    camera->InitMetadata(&metadata);
    camera->film->WriteImage(metadata, b / mutationsPerPixel);
}

std::string MLTIntegrator::ToString() const {
    return StringPrintf("[ MLTIntegrator camera: %s maxDepth: %d nBootstrap: %d "
                        "nChains: %d mutationsPerPixel: %d sigma: %f "
                        "largeStepProbability: %f lightSampler: %s regularize: %s ]",
                        camera, maxDepth, nBootstrap, nChains, mutationsPerPixel,
                        sigma, largeStepProbability, lightSampler, regularize);
}

std::unique_ptr<MLTIntegrator> MLTIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera, const FileLoc *loc) {
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    int nBootstrap = dict.GetOneInt("bootstrapsamples", 100000);
    int64_t nChains = dict.GetOneInt("chains", 1000);
    int mutationsPerPixel = dict.GetOneInt("mutationsperpixel", 100);
    Float largeStepProbability =
        dict.GetOneFloat("largestepprobability", 0.3f);
    Float sigma = dict.GetOneFloat("sigma", .01f);
    if (PbrtOptions.quickRender) {
        mutationsPerPixel = std::max(1, mutationsPerPixel / 16);
        nBootstrap = std::max(1, nBootstrap / 16);
    }
    bool regularize = dict.GetOneBool("regularize", true);
    return std::make_unique<MLTIntegrator>(scene, std::move(camera), maxDepth, nBootstrap,
                                           nChains, mutationsPerPixel, sigma,
                                           largeStepProbability, regularize);
}

STAT_RATIO(
    "Stochastic Progressive Photon Mapping/Visible points checked per photon "
    "intersection",
    visiblePointsChecked, totalPhotonSurfaceInteractions);
STAT_COUNTER("Stochastic Progressive Photon Mapping/Photon paths followed",
             photonPaths);
STAT_INT_DISTRIBUTION(
    "Stochastic Progressive Photon Mapping/Grid cells per visible point",
    gridCellsPerVisiblePoint);
STAT_MEMORY_COUNTER("Memory/SPPM Pixels", pixelMemoryBytes);
STAT_MEMORY_COUNTER("Memory/SPPM BSDF and Grid Memory", sppmMemoryArenaBytes);

// SPPM Local Definitions
struct SPPMPixel {
    // SPPMPixel Public Methods
    SPPMPixel() : M(0) {}

    // SPPMPixel Public Data
    Float radius = 0;
    RGB Ld;
    struct VisiblePoint {
        // VisiblePoint Public Methods
        VisiblePoint() = default;
        VisiblePoint(const Point3f &p, const Vector3f &wo, const BSDF *bsdf,
                     const SampledSpectrum &beta)
            : p(p), wo(wo), bsdf(bsdf), beta(beta) {}
        Point3f p;
        Vector3f wo;
        const BSDF *bsdf = nullptr;
        SampledSpectrum beta;
    } vp;
    AtomicFloat Phi[NSpectrumSamples];
    std::atomic<int> M;
    Float N = 0;
    RGB tau;
};

struct SPPMPixelListNode {
    SPPMPixel *pixel;
    SPPMPixelListNode *next;
};

static bool ToGrid(const Point3f &p, const Bounds3f &bounds,
                   const int gridRes[3], Point3i *pi) {
    bool inBounds = true;
    Vector3f pg = bounds.Offset(p);
    for (int i = 0; i < 3; ++i) {
        (*pi)[i] = (int)(gridRes[i] * pg[i]);
        inBounds &= ((*pi)[i] >= 0 && (*pi)[i] < gridRes[i]);
        (*pi)[i] = Clamp((*pi)[i], 0, gridRes[i] - 1);
    }
    return inBounds;
}

inline unsigned int hash(const Point3i &p, int hashSize) {
    return Hash(p.x, p.y, p.z) % hashSize;
}

// SPPM Method Definitions
void SPPMIntegrator::Render() {
    StatsSetImageResolution(camera->film->pixelBounds);
    StatsSetPixelStatsBaseName(RemoveExtension(camera->film->filename).c_str());

    std::unique_ptr<pstd::vector<DigitPermutation>> digitPermutations(
        ComputeRadicalInversePermutations(digitPermutationsSeed));

    ProfilerScope p(ProfilePhase::IntegratorRender);
    // Initialize _pixelBounds_ and _pixels_ array for SPPM
    Bounds2i pixelBounds = camera->film->pixelBounds;
    CHECK(!pixelBounds.IsEmpty());
    int nPixels = pixelBounds.Area();
    Array2D<SPPMPixel> pixels(pixelBounds);
    for (SPPMPixel & p : pixels)
        p.radius = initialSearchRadius;
    const Float invSqrtSPP = 1.f / std::sqrt(nIterations);
    pixelMemoryBytes += pixels.size() * sizeof(SPPMPixel);

    // Compute _lightSampler_ for sampling lights proportional to power
    BVHLightSampler directLightSampler(scene.lights, Allocator());
    PowerLightSampler shootLightSampler(scene.lights, Allocator());

    // Perform _nIterations_ of SPPM integration
    HaltonSampler sampler(nIterations, camera->film->fullResolution);

    // Compute number of tiles to use for SPPM camera pass
    Vector2i pixelExtent = pixelBounds.Diagonal();
    ProgressReporter progress(2 * nIterations, "Rendering");
    std::vector<MemoryArena> perThreadArenas(MaxThreadIndex());
    std::vector<MaterialBuffer> perThreadMaterialBuffers;
    for (int i = 0; i < MaxThreadIndex(); ++i)
        // TODO: size this
        perThreadMaterialBuffers.emplace_back(nPixels * 1024);
    std::vector<SamplerHandle> tileSamplers = sampler.Clone(MaxThreadIndex(), Allocator());

    for (int iter = 0; iter < nIterations; ++iter) {
        SampledWavelengths lambda =
            PbrtOptions.disableWavelengthJitter ? camera->film->SampleWavelengths(0.5) :
            camera->film->SampleWavelengths(RadicalInverse(1, iter));

        // Generate SPPM visible points
        {
            ProfilerScope _(ProfilePhase::SPPMCameraPass);
            ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
                MemoryArena &arena = perThreadArenas[ThreadIndex];
                MaterialBuffer &materialBuffer = perThreadMaterialBuffers[ThreadIndex];

                // Follow camera paths for _tile_ in image for SPPM
                SamplerHandle &tileSampler = tileSamplers[ThreadIndex];

                for (Point2i pPixel : tileBounds) {
                    // Prepare _tileSampler_ for _pPixel_
                    tileSampler.StartPixelSample(pPixel, iter);

                    // Generate camera ray for pixel for SPPM
                    CameraSample cameraSample = tileSampler.GetCameraSample(pPixel, camera->film->filter);

                    pstd::optional<CameraRayDifferential> crd =
                        camera->GenerateRayDifferential(cameraSample, lambda);
                    if (!crd || !crd->weight) continue;

                    SampledSpectrum beta = crd->weight;
                    RayDifferential &ray = crd->ray;
                    if (!PbrtOptions.disablePixelJitter)
                        ray.ScaleDifferentials(invSqrtSPP);

                    // Follow camera ray path until a visible point is created

                    // Get _SPPMPixel_ for _pPixel_
                    SPPMPixel &pixel = pixels[pPixel];
                    Float etaScale = 1;
                    bool specularBounce = false, anyNonSpecularBounces = false;
                    for (int depth = 0; depth < maxDepth; ++depth) {
                        ++totalPhotonSurfaceInteractions;
                        pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
                        if (!si) {
                            if (depth == 0) {
                                // Accumulate light contributions for ray with no
                                // intersection
                                for (const auto &light : scene.infiniteLights) {
                                    RGB rgb = (beta * light.Le(ray, lambda)).ToRGB(lambda, *colorSpace);
                                    pixel.Ld += rgb;
                                }
                            }
                            break;
                        }

                        // Process SPPM camera ray intersection

                        // Compute BSDF at SPPM camera ray intersection
                        SurfaceInteraction &isect = si->intr;
                        isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, &sampler);
                        if (isect.bsdf == nullptr) {
                            isect.SkipIntersection(&ray, si->tHit);
                            --depth;
                            continue;
                        }

                        if (regularize && anyNonSpecularBounces) {
                            ++regularizedBSDFs;
                            isect.bsdf->Regularize(materialBuffer);
                        }
                        ++totalBSDFs;

                        const BSDF &bsdf = *isect.bsdf;

                        // Accumulate direct illumination at SPPM camera ray
                        // intersection
                        Vector3f wo = -ray.d;
                        if (depth == 0 || specularBounce) {
                            RGB rgb = (beta * isect.Le(wo, lambda)).ToRGB(lambda, *colorSpace);
                            pixel.Ld += rgb;
                        }

                        SampledSpectrum Ld =
                            LdSampleLightsAndBSDF(isect, scene, lambda, tileSampler, arena,
                                                  directLightSampler);
                        pixel.Ld += (beta * Ld).ToRGB(lambda, *colorSpace);

                        // Possibly create visible point and end camera path
                        if (bsdf.IsDiffuse() || (bsdf.IsGlossy() && depth == maxDepth - 1)) {
                            pixel.vp = {isect.p(), wo, &bsdf, beta};
                            break;
                        }

                        // Spawn ray from SPPM camera path vertex
                        if (depth < maxDepth - 1) {
                            Float u = tileSampler.Get1D();
                            pstd::optional<BSDFSample> bs =
                                bsdf.Sample_f(wo, u, tileSampler.Get2D());
                            if (!bs) break;
                            specularBounce = bs->IsSpecular();
                            anyNonSpecularBounces |= !bs->IsSpecular();
                            if (bs->IsTransmission())
                                etaScale *= Sqr(isect.bsdf->eta);

                            beta *= bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;
                            SampledSpectrum rrBeta = beta * etaScale;
                            if (rrBeta.MaxComponentValue() < 1) {
                                Float q =
                                    std::max<Float>(.05f, 1 - rrBeta.MaxComponentValue());
                                if (tileSampler.Get1D() < q) break;
                                beta /= 1 - q;
                            }
                            ray = isect.SpawnRay(ray, bs->wi, bs->flags);
                        }
                    }
                }
            });
        }
        progress.Update();

        // Create grid of all SPPM visible points
        int gridRes[3];
        Bounds3f gridBounds;
        // Allocate grid for SPPM visible points
        const int hashSize = NextPrime(nPixels);
        std::vector<std::atomic<SPPMPixelListNode *>> grid(hashSize);
        {
            ProfilerScope _(ProfilePhase::SPPMGridConstruction);

            // Compute grid bounds for SPPM visible points
            Float maxRadius = 0.;
            for (const SPPMPixel & pixel : pixels) {
                if (!pixel.vp.beta) continue;
                Bounds3f vpBound = Expand(Bounds3f(pixel.vp.p), pixel.radius);
                gridBounds = Union(gridBounds, vpBound);
                maxRadius = std::max(maxRadius, pixel.radius);
            }

            // Compute resolution of SPPM grid in each dimension
            Vector3f diag = gridBounds.Diagonal();
            Float maxDiag = MaxComponentValue(diag);
            int baseGridRes = (int)(maxDiag / maxRadius);
            for (int i = 0; i < 3; ++i)
                gridRes[i] = std::max<int>(baseGridRes * diag[i] / maxDiag, 1);

            // Add visible points to SPPM grid
            ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
                MemoryArena &arena = perThreadArenas[ThreadIndex];
                for (Point2i pPixel : tileBounds) {
                    SPPMPixel &pixel = pixels[pPixel];
                    if (pixel.vp.beta) {
                        // Add pixel's visible point to applicable grid cells
                        Float radius = pixel.radius;
                        Point3i pMin, pMax;
                        ToGrid(pixel.vp.p - Vector3f(radius, radius, radius),
                               gridBounds, gridRes, &pMin);
                        ToGrid(pixel.vp.p + Vector3f(radius, radius, radius),
                               gridBounds, gridRes, &pMax);
                        for (int z = pMin.z; z <= pMax.z; ++z)
                            for (int y = pMin.y; y <= pMax.y; ++y)
                                for (int x = pMin.x; x <= pMax.x; ++x) {
                                    // Add visible point to grid cell $(x, y, z)$
                                    int h = hash(Point3i(x, y, z), hashSize);
                                    SPPMPixelListNode *node =
                                        arena.Alloc<SPPMPixelListNode>();
                                    node->pixel = &pixel;

                                    // Atomically add _node_ to the start of
                                    // _grid[h]_'s linked list
                                    node->next = grid[h];
                                    while (!grid[h].compare_exchange_weak(
                                               node->next, node))
                                        ;
                                }
                        ReportValue(gridCellsPerVisiblePoint,
                                    (1 + pMax.x - pMin.x) * (1 + pMax.y - pMin.y) *
                                    (1 + pMax.z - pMin.z));
                    }
                }
            });
        }

        // Trace photons and accumulate contributions
        {
            ProfilerScope _(ProfilePhase::SPPMPhotonPass);
            std::vector<MaterialBuffer> photonShootMaterialBuffers(MaxThreadIndex());
            for (auto &m : photonShootMaterialBuffers)
                m = MaterialBuffer(16384);

            ParallelFor(0, photonsPerIteration, [&](int64_t start, int64_t end) {
                MaterialBuffer &materialBuffer = photonShootMaterialBuffers[ThreadIndex];

                for (int64_t photonIndex = start; photonIndex < end; ++photonIndex) {
                // Follow photon path for _photonIndex_
                uint64_t haltonIndex =
                    (uint64_t)iter * (uint64_t)photonsPerIteration +
                    photonIndex;
                int haltonDim = 0;
                auto Sample1D = [&]() {
                    Float u = ScrambledRadicalInverse(haltonDim, haltonIndex,
                                                      (*digitPermutations)[haltonDim]);
                    ++haltonDim;
                    return u;
                };
                auto Sample2D = [&]() {
                    Point2f u(ScrambledRadicalInverse(haltonDim, haltonIndex,
                                                      (*digitPermutations)[haltonDim]),
                              ScrambledRadicalInverse(haltonDim + 1, haltonIndex,
                                                      (*digitPermutations)[haltonDim + 1]));
                    haltonDim += 2;
                    return u;
                };

                // Choose light to shoot photon from
                Float lightPDF;
                Float lightSample = Sample1D();
                LightHandle light = shootLightSampler.Sample(lightSample, &lightPDF);
                if (lightPDF == 0) continue;

                // Compute sample values for photon ray leaving light source
                Point2f uLight0 = Sample2D();
                Point2f uLight1 = Sample2D();
                Float uLightTime = Lerp(Sample1D(), camera->shutterOpen, camera->shutterClose);

                // Generate _photonRay_ from light source and initialize _beta_
                pstd::optional<LightLeSample> les =
                    light.Sample_Le(uLight0, uLight1, lambda, uLightTime);
                if (!les || les->pdfPos == 0 || les->pdfDir == 0 || !les->L) continue;
                RayDifferential photonRay = RayDifferential(les->ray);
                SampledSpectrum beta = (les->AbsCosTheta(photonRay.d) * les->L) /
                                (lightPDF * les->pdfPos * les->pdfDir);
                if (!beta) continue;

                // Follow photon path through scene and record intersections
                SurfaceInteraction isect;
                for (int depth = 0; depth < maxDepth; ++depth) {
                    pstd::optional<ShapeIntersection> si = scene.Intersect(photonRay);
                    if (!si) break;
                    SurfaceInteraction &isect = si->intr;
                    ++totalPhotonSurfaceInteractions;
                    if (depth > 0) {
                        // Add photon contribution to nearby visible points
                        Point3i photonGridIndex;
                        if (ToGrid(isect.p(), gridBounds, gridRes,
                                   &photonGridIndex)) {
                            int h = hash(photonGridIndex, hashSize);
                            // Add photon contribution to visible points in
                            // _grid[h]_
                            for (SPPMPixelListNode *node =
                                     grid[h].load(std::memory_order_relaxed);
                                 node != nullptr; node = node->next) {
                                ++visiblePointsChecked;
                                SPPMPixel &pixel = *node->pixel;
                                Float radius = pixel.radius;
                                if (DistanceSquared(pixel.vp.p, isect.p()) >
                                    radius * radius)
                                    continue;
                                // Update _pixel_ $\Phi$ and $M$ for nearby
                                // photon
                                Vector3f wi = -photonRay.d;
                                SampledSpectrum Phi =
                                    beta * pixel.vp.bsdf->f(pixel.vp.wo, wi);
                                for (int i = 0; i < NSpectrumSamples; ++i)
                                    pixel.Phi[i].Add(Phi[i]);
                                ++pixel.M;
                            }
                        }
                    }
                    // Sample new photon ray direction

                    // Compute BSDF at photon intersection point
                    isect.ComputeScatteringFunctions(photonRay, lambda, *camera, materialBuffer,
                                                     &sampler, TransportMode::Importance);
                    if (isect.bsdf == nullptr) {
                        isect.SkipIntersection(&photonRay, si->tHit);
                        --depth;
                        continue;
                    }
                    const BSDF &photonBSDF = *isect.bsdf;

                    // Sample BSDF _fr_ and direction _wi_ for reflected photon
                    Vector3f wo = -photonRay.d;

                    // Generate _bsdfSample_ for outgoing photon sample
                    Float bsdfSample = Sample1D();
                    Point2f bsdfSample2 = Sample2D();
                    pstd::optional<BSDFSample> bs =
                        photonBSDF.Sample_f(wo, bsdfSample, bsdfSample2);

                    if (!bs) break;
                    SampledSpectrum bnew =
                        beta * bs->f * AbsDot(bs->wi, isect.shading.n) / bs->pdf;

                    // Possibly terminate photon path with Russian roulette
                    Float q = std::max<Float>(0, 1 - (bnew.MaxComponentValue() /
                                                      beta.MaxComponentValue()));
                    if (Sample1D() < q) break;
                    beta = bnew / (1 - q);
                    photonRay = RayDifferential(isect.SpawnRay(bs->wi));
                }
                materialBuffer.Reset();
                }
            });

            for (MemoryArena &arena : perThreadArenas)
                arena.Reset();
            for (MaterialBuffer &materialBuffer : perThreadMaterialBuffers)
                materialBuffer.Reset();

            progress.Update();
            photonPaths += photonsPerIteration;
        }

        // Update pixel values from this pass's photons
        {
            ProfilerScope _(ProfilePhase::SPPMStatsUpdate);
            ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
                SPPMPixel &p = pixels[pPixel];
                int M = p.M.load();
                if (M > 0) {
                    // Update pixel photon count, search radius, and $\tau$ from
                    // photons
                    Float gamma = (Float)2 / (Float)3;
                    Float Nnew = p.N + gamma * M;
                    Float Rnew = p.radius * std::sqrt(Nnew / (p.N + M));
                    SampledSpectrum Phi;
                    for (int j = 0; j < NSpectrumSamples; ++j)
                        Phi[j] = p.Phi[j];
                    RGB rgb = (p.vp.beta * Phi).ToRGB(lambda, *colorSpace);
                    p.tau = (p.tau + rgb) * (Rnew * Rnew) / (p.radius * p.radius);
                    p.N = Nnew;
                    p.radius = Rnew;
                    p.M = 0;
                    for (int j = 0; j < NSpectrumSamples; ++j)
                        p.Phi[j] = (Float)0;
                }
                // Reset _VisiblePoint_ in pixel
                p.vp.beta = SampledSpectrum(0.);
                p.vp.bsdf = nullptr;
            });
        }

        // Periodically store SPPM image in film and write image
        if (iter + 1 == nIterations ||
            (iter + 1 <= 64 && IsPowerOf2(iter + 1)) ||
            ((iter + 1) % 64 == 0)) {
            uint64_t Np = (uint64_t)(iter + 1) * (uint64_t)photonsPerIteration;
            Image rgbImage(PixelFormat::Float, Point2i(pixelBounds.Diagonal()),
                           { "R", "G", "B" });

            ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
                // Compute radiance _L_ for SPPM pixel _pixel_
                const SPPMPixel &pixel = pixels[pPixel];
                RGB L = pixel.Ld / (iter + 1);
                L += pixel.tau / (Np * Pi * pixel.radius * pixel.radius);
                Point2i pImage = Point2i(pPixel - pixelBounds.pMin);
                rgbImage.SetChannels(pImage, {L.r, L.g, L.b});
            });

            ImageMetadata metadata;
            metadata.renderTimeSeconds = progress.ElapsedSeconds();
            metadata.samplesPerPixel = iter + 1;
            metadata.pixelBounds = pixelBounds;
            metadata.fullResolution = camera->film->fullResolution;
            metadata.colorSpace = colorSpace;
            camera->InitMetadata(&metadata);
            rgbImage.Write(camera->film->filename, metadata);

            // Write SPPM radius image, if requested
            if (getenv("SPPM_RADIUS") != nullptr) {
                Image rimg(PixelFormat::Float, Point2i(pixelBounds.Diagonal()), { "Radius" });
                Float minrad = 1e30f, maxrad = 0;
                for (const SPPMPixel &p : pixels) {
                    minrad = std::min(minrad, p.radius);
                    maxrad = std::max(maxrad, p.radius);
                }
                fprintf(stderr,
                        "iterations: %d (%.2f s) radius range: %f - %f\n",
                        iter + 1, progress.ElapsedSeconds(), minrad, maxrad);
                int offset = 0;
                for (Point2i pPixel : pixelBounds) {
                    const SPPMPixel &p = pixels[pPixel];
                    Float v = 1.f - (p.radius - minrad) / (maxrad - minrad);
                    Point2i pImage = Point2i(pPixel - pixelBounds.pMin);
                    rimg.SetChannel(pImage, 0, v);
                }
                ImageMetadata metadata;
                metadata.pixelBounds = pixelBounds;
                metadata.fullResolution = camera->film->fullResolution;
                rimg.Write("sppm_radius.png", metadata);
            }
        }
    }

    sppmMemoryArenaBytes += std::accumulate(perThreadArenas.begin(), perThreadArenas.end(),
                                            size_t(0), [&](size_t v, const MemoryArena &arena) {
                                                           return v + arena.BytesAllocated();
                                                       });
    progress.Done();
}

std::string SPPMIntegrator::ToString() const {
    return StringPrintf("[ SPPMIntegrator camera: %s initialSearchRadius: %f "
                        "nIterations: %d maxDepth: %d photonsPerIteration: %d "
                        "regularize: %s colorSpace: %s digitPermutations:(elided) ]",
                        camera, initialSearchRadius, nIterations, maxDepth,
                        photonsPerIteration, regularize, *colorSpace);
}

std::unique_ptr<SPPMIntegrator> SPPMIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    const RGBColorSpace *colorSpace, std::unique_ptr<const Camera> camera, const FileLoc *loc) {
    int nIterations = dict.GetOneInt("iterations", 64);
    int maxDepth = dict.GetOneInt("maxdepth", 5);
    int photonsPerIter = dict.GetOneInt("photonsperiteration", -1);
    Float radius = dict.GetOneFloat("radius", 1.f);
    if (PbrtOptions.quickRender) nIterations = std::max(1, nIterations / 16);
    bool regularize = dict.GetOneBool("regularize", true);
    int seed = dict.GetOneInt("seed", 0);
    return std::make_unique<SPPMIntegrator>(scene, std::move(camera), nIterations, photonsPerIter,
                                            maxDepth, radius, regularize, seed, colorSpace);
}


struct RISLightSample {
    LightHandle light = nullptr;
    Point2f u = {-1.f, -1.f};
    Float g = 0.f;

    std::string ToString() const {
        return StringPrintf("[ RISLightSample light:%s u: %s g: %f ]",
                            light, u, g);
    }
};

struct PixelSample {
    PixelSample() = default;
    PixelSample(Point2f pFilm, const SurfaceInteraction &intr, int N, MemoryArena &arena, RNG &rng)
        : pFilm(pFilm), intr(intr) {
        reservoirs = arena.Alloc<WeightedReservoirSampler<RISLightSample>[]>(N);
        resampledReservoirs = arena.Alloc<WeightedReservoirSampler<RISLightSample>[]>(N);

        for (int i = 0; i < N; ++i) {
            reservoirs[i] =
                WeightedReservoirSampler<RISLightSample>(rng.Uniform<uint64_t>());
            resampledReservoirs[i] =
                WeightedReservoirSampler<RISLightSample>(rng.Uniform<uint64_t>());
        }
    }

    Point2f pFilm;
    pstd::optional<SurfaceInteraction> intr;
    WeightedReservoirSampler<RISLightSample> *reservoirs = nullptr, *resampledReservoirs = nullptr;
};

struct Pixel {
    RNG rng;
    PixelSample *pixelSamples = nullptr;
    uint8_t *stratumToSampleIndex = nullptr, *sampleIndexToStratum = nullptr;
};

RISIntegrator::RISIntegrator(int N, int M, int nSpatio, bool useMIS,
                             LightMISStrategy lightMISStrategy,
                             const std::string &lightSampleStrategy,
                             const Scene &scene,
                             std::unique_ptr<const Camera> c,
                             int spp)
    : Integrator(scene),
      N(N),
      M(M),
      rootM(std::sqrt(M)),
      nSpatioResamples(nSpatio),
      useMIS(useMIS),
      lightMISStrategy(lightMISStrategy),
      camera(std::move(c)),
      spp(spp),
      lightSampleStrategy(lightSampleStrategy) {
    CHECK(IsPowerOf4(spp));
    CHECK_EQ(M, Sqr(rootM));
    CHECK_LE(spp, 255);   // uint8_t in Pixel...

    if (nSpatioResamples > 0 && (nSpatioResamples & 1)) {
        ++nSpatioResamples;
        Warning("Must have an even number spatial resmapling passes. Adding one -> %d.",
                nSpatioResamples);
    }
}

STAT_MEMORY_COUNTER("Memory/RIS hit points", risIntrBytesUsed);
STAT_MEMORY_COUNTER("Memory/RIS BSDFs", risBSDFBytesUsed);
STAT_MEMORY_COUNTER("Memory/RIS reservoirs", risReservoirBytesUsed);

void RISIntegrator::Render() {
    std::unique_ptr<LightSampler> lightSampler(
        LightSampler::Create(lightSampleStrategy, scene.lights, Allocator()));

    if (PbrtOptions.debugStart.has_value()) {
        // It's tricky...
        LOG_FATAL("--debugstart not supported for RISIntegrator");
    }

    // Compute number of tiles, _nTiles_, to use for parallel rendering
    Bounds2i pixelBounds = camera->film->pixelBounds;
    Vector2i pixelExtent = pixelBounds.Diagonal();

    CHECK(IsPowerOf4(spp));
    int rootSpp = std::sqrt(spp);
    Array2D<Pixel> pixels(pixelBounds);

    auto sampleLambda = [this](Point2i pPixel, int sampleIndex) {
        Float lu = RadicalInverse(1, sampleIndex) + BlueNoise(47, pPixel.x, pPixel.y);
        if (lu >= 1) lu -= 1;
        if (PbrtOptions.disableWavelengthJitter)
            lu = 0.5;
        return camera->film->SampleWavelengths(lu);
    };

    Float rayDiffScale = std::max<Float>(.125, 1 / std::sqrt((Float)spp));

    // Allocate data structures
    MemoryArena pixelArena;
    uint64_t seed = 0;
    for (Point2i pPixel : pixelBounds) {
        pixels[pPixel].rng.SetSequence(seed++);
        pixels[pPixel].pixelSamples = pixelArena.Alloc<PixelSample[]>(spp);
        pixels[pPixel].stratumToSampleIndex = pixelArena.Alloc<uint8_t[]>(spp);
        pixels[pPixel].sampleIndexToStratum = pixelArena.Alloc<uint8_t[]>(spp);
    }
    risIntrBytesUsed += pixelBounds.Area() * sizeof(Pixel) + pixelArena.BytesAllocated();

    // Camera pass
    std::vector<SamplerHandle> threadSamplers(MaxThreadIndex());
    for (auto &sampler : threadSamplers)
        sampler = new PMJ02BNSampler(spp);
    std::vector<MaterialBuffer> materialBuffers;
    for (int i = 0; i < MaxThreadIndex(); ++i)
        // TODO: sizing
        materialBuffers.emplace_back(spp * pixelBounds.Area() * 1024);
    std::vector<MemoryArena> reservoirArenas(MaxThreadIndex());
    std::vector<MemoryArena> lightSampleArenas(MaxThreadIndex());
    Timer timer;
    ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
        ProfilerScope _(ProfilePhase::RISCameraPass);
        // Render section of image corresponding to _tile_

        MaterialBuffer &materialBuffer = materialBuffers[ThreadIndex];
        MemoryArena &reservoirArena = reservoirArenas[ThreadIndex];
        MemoryArena &lightSampleArena = lightSampleArenas[ThreadIndex];
        SamplerHandle &sampler = threadSamplers[ThreadIndex];

        for (Point2i pPixel : tileBounds) {
            RNG &rng = pixels[pPixel].rng;

            for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
                sampler.StartPixelSample(pPixel, sampleIndex);

                CameraSample cameraSample = sampler.GetCameraSample(pPixel, camera->film->filter);
                CHECK_EQ(1.f, cameraSample.weight); // for now; make filtering to final image easy.

                Float uLight = sampler.Get1D();
                int stratumIndex = std::min(int(uLight * spp), spp - 1);
                pixels[pPixel].stratumToSampleIndex[stratumIndex] = sampleIndex;
                pixels[pPixel].sampleIndexToStratum[sampleIndex] = stratumIndex;

                // Sample wavelengths for ray
                SampledWavelengths lambda = sampleLambda(pPixel, stratumIndex);

                // Generate camera ray for current sample
                pstd::optional<CameraRayDifferential> cameraRay =
                    camera->GenerateRayDifferential(cameraSample, lambda);

                if (!PbrtOptions.disablePixelJitter)
                    cameraRay->ray.ScaleDifferentials(rayDiffScale);

                RayDifferential &ray = cameraRay->ray;
                pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
                if (!si) {
                    SampledSpectrum L(0.f);
                    for (const auto &light : scene.infiniteLights)
                        L += light.Le(ray, lambda);
                    camera->film->AddSample(pPixel, L, lambda, {}, cameraSample.weight);
                    continue;
                }

                // Compute BSDF
                SurfaceInteraction &isect = si->intr;
                isect.ComputeScatteringFunctions(ray, lambda, *camera, materialBuffer, sampler);
                CHECK(isect.bsdf != nullptr); /* TODO handle this */

                CHECK(pixels[pPixel].pixelSamples[sampleIndex].intr.has_value() == false);
                pixels[pPixel].pixelSamples[sampleIndex] =
                    PixelSample(cameraSample.pFilm, isect, N, reservoirArena,
                                pixels[pPixel].rng);

                WeightedReservoirSampler<RISLightSample> *reservoirs =
                    pixels[pPixel].pixelSamples[sampleIndex].reservoirs;

                Point2f u = sampler.Get2D();

                for (int i = 0; i < M; ++i) {
                    // Take light samples.
                    // Use the light sampler
#if 1
                    Float uu = (stratumIndex + rng.Uniform<Float>()) / spp;
                    SampledLightVector slv =
                        lightSampler->Sample(isect, uu, lightSampleArena);
                    if (slv.empty())
                        continue;
                    if (slv.size() > 1) {
                        pstd::swap(slv[0], slv[rng.Uniform<int>(slv.size())]);
                        slv[0].pdf /= slv.size();
                    }

                    LightHandle light = slv[0].light;
                    Float lightChoicePDF = slv[0].pdf;
                    auto cp = [](Float v) { return v >= 1 ? v - 1 : v; };
                    Point2f ul(cp(u.x + Float(i % rootM) / rootM),
                               cp(u.y + Float(i / rootM) / rootM));
#else
                    // Paper
                    uint64_t mortonIndex = EncodeMorton2(pPixel.x, pPixel.y);
                    mortonIndex = spp * mortonIndex + sampleIndex;
                    mortonIndex = M * mortonIndex + i;
                    uint32_t lightIndex = mortonIndex % scene.lights.size();
                    LightHandle light = scene.lights[lightIndex];
                    Float lightChoicePDF = 1.f / scene.lights.size();

                    Point2f ul(rng.Uniform<Float>(), rng.Uniform<Float>());
#endif
                    pstd::optional<LightLiSample> ls = light.Sample_Li(isect, ul, lambda);
                    if (!ls || !ls->L || ls->pdf == 0)
                        continue;

                    SampledSpectrum f = isect.bsdf->f(isect.wo, ls->wi) * AbsDot(ls->wi, isect.shading.n);
                    // Include 1/ls->pdf in g rather than just the weight,
                    // so that when we reweight during resampling, we can
                    // use ls->pdf at the current point...
                    Float g = (f * ls->L).MaxComponentValue() / ls->pdf;
                    Float weight = g / lightChoicePDF;
                    // TODO? MIS
#if 0
                    // Trade-offs here??
                    for (int j = 0; j < N; ++j)
                        reservoirs[j].Add(RISLightSample{light, ul, g}, weight);
#else
                    CHECK_GE(M, N); // make sure all get at least one!
                    reservoirs[i % N].Add(RISLightSample{light, ul, g}, weight);
#endif
                }

                // shadow rays
                for (int i = 0; i < N; ++i) {
                    if (!reservoirs[i].HasSample())
                        continue;

                    RISLightSample candidate = reservoirs[i].GetSample();
                    CHECK_GT(candidate.g, 0);

                    pstd::optional<LightLiSample> ls = candidate.light.Sample_Li(isect, candidate.u, lambda);
                    CHECK(ls.has_value());
                    if (!ls->Unoccluded(scene))
                        reservoirs[i].Reset();
                }
                lightSampleArena.Reset();
            }
        }
    }, "Camera pass");

#if 0
    risBSDFBytesUsed += std::accumulate(bsdfArenas.begin(), bsdfArenas.end(), size_t(0),
                                        [&](size_t v, const MemoryArena &arena) {
                                            return v + arena.BytesAllocated();
                                        });
#endif
    risReservoirBytesUsed += std::accumulate(reservoirArenas.begin(), reservoirArenas.end(), size_t(0),
                                             [&](size_t v, const MemoryArena &arena) {
                                                 return v + arena.BytesAllocated();
                                             });

    auto compatible = [rayDiffScale](const PixelSample &ps,
                                     const PixelSample &otherSample) -> bool {
        if (!otherSample.intr)
            return false;

        if (ps.intr->bsdf->IsDiffuse() && otherSample.intr->bsdf->IsGlossy())
            return false;

        if (Dot(otherSample.intr->n, ps.intr->n) < 0.95)
            return false;

        Float z = ps.intr->p().z, zo = otherSample.intr->p().z;
        Vector2f dpf = otherSample.pFilm - ps.pFilm;
        Float zEst = z + (dpf.x * ps.intr->dpdx.z + dpf.y * ps.intr->dpdy.z) / rayDiffScale;
        Float zErr = std::abs((zo - zEst) / ((zo + zEst) * 0.5f));
        if (zErr > .01)
            return false;

        return true;
    };

    auto resample = [this,compatible](PixelSample &ps, const PixelSample &otherSample,
                                      const SampledWavelengths &lambda) {
        if (!compatible(ps, otherSample))
            return;

        // Note: this seems to work better than looping over
        // all of otherSample's reservoirs...
        for (int j = 0; j < N; ++j) {
            const WeightedReservoirSampler<RISLightSample> &src = otherSample.reservoirs[j];
            WeightedReservoirSampler<RISLightSample> &dest = ps.resampledReservoirs[j];

            if (src.HasSample()) {
                // Reweight by evaluating our own BSDF...
                Float weight = src.WeightSum() / src.GetSample().g;
                Float g = 0;
                const SurfaceInteraction &isect = *ps.intr;
                pstd::optional<LightLiSample> ls =
                    src.GetSample().light.Sample_Li(isect, src.GetSample().u, lambda);
                if (!ls || !ls->L || ls->pdf == 0)
                    weight = 0;
                else {
                    SampledSpectrum f = isect.bsdf->f(isect.wo, ls->wi) *
                        AbsDot(ls->wi, isect.shading.n);
                    g = (f * ls->L).MaxComponentValue() / ls->pdf;
                    weight *= g;
                }

                dest.Add(RISLightSample{src.GetSample().light,
                                        src.GetSample().u, g}, weight,
                    src.NSamplesConsidered());
            } else
                dest.Add(RISLightSample{}, 0, src.NSamplesConsidered());
        }
    };

    // RIS passes
    for (int pass = 0; pass < nSpatioResamples; ++pass) {
        int delta;
        if (pass == 0)
            delta = 0;
        else
            delta = (pass - 1 <= nSpatioResamples / 2) ? (1 << (pass - 1)) :
                (1 << (nSpatioResamples - 1 - (pass - 1)));
        LOG_VERBOSE("pass %d -> delta %d", pass, delta);

        ProfilerScope _(ProfilePhase::RISResamplePass);
        ParallelFor2D(pixelBounds, [&](Bounds2i tileBounds) {
            for (Point2i pPixel : tileBounds) {
                RNG &rng = pixels[pPixel].rng;

                for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
                    PixelSample &ps = pixels[pPixel].pixelSamples[sampleIndex];
                    if (!ps.intr)
                        continue;

                    int stratumIndex = pixels[pPixel].sampleIndexToStratum[sampleIndex];
                    SampledWavelengths lambda = sampleLambda(pPixel, stratumIndex);

                    for (int i = 0; i < N; ++i)
                        ps.resampledReservoirs[i].Copy(ps.reservoirs[i]);

                    if (pass == 0) {
                        // Resample from the other samples in the pixel
                        for (int otherSampleIndex = 0; otherSampleIndex < spp; ++otherSampleIndex) {
                            if (otherSampleIndex == sampleIndex)
                                continue;
                            resample(ps, pixels[pPixel].pixelSamples[otherSampleIndex], lambda);
                        }
                    } else {
                        Vector2i jitter(-1 + rng.Uniform<int>(3), -1 + rng.Uniform<int>(3));

                        // A-trous
                        for (int dy = -2 * delta; dy <= 2 * delta; dy += delta) {
                            for (int dx = -2 * delta; dx <= 2 * delta; dx += delta) {
                                Point2i pOther = pPixel + Vector2i(dx, dy) + jitter;

                                if (!InsideExclusive(pOther, pixelBounds))
                                    continue;

                                int otherSampleIndex = rng.Uniform<int>(spp);
                                //CO  int otherSampleIndex = sampleIndex;
                                //CO int stratumIndex = pixels[pPixel].sampleIndexToStratum[sampleIndex];
                                //CO int otherSampleIndex = pixels[pOther].stratumToSampleIndex[stratumIndex];

                                if (pOther == pPixel && otherSampleIndex == sampleIndex)
                                    continue;

                                const PixelSample &otherSample = pixels[pOther].pixelSamples[otherSampleIndex];
                                resample(ps, otherSample, lambda);
                            }
                        }
                    }
                }
            }
        }, "Resampling pass");
        ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
            for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
                PixelSample &ps = pixels[pPixel].pixelSamples[sampleIndex];
                if (ps.intr)
                    for (int i = 0; i < N; ++i)
                        ps.reservoirs[i].Copy(ps.resampledReservoirs[i]);
            }
        });
    }

    // Final shading pass and update film.
    {
    ProfilerScope _(ProfilePhase::RISFinalPass);
    ParallelFor2D(pixelBounds, [&](Point2i pPixel) {
        for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
            PixelSample &ps = pixels[pPixel].pixelSamples[sampleIndex];
            if (!ps.intr)
                continue;

            // Sample wavelengths for ray
            int stratumIndex = pixels[pPixel].sampleIndexToStratum[sampleIndex];
            SampledWavelengths lambda = sampleLambda(pPixel, stratumIndex);

            const SurfaceInteraction &intr = *ps.intr;
            SampledSpectrum L = intr.Le(intr.wo, lambda);

            for (int i = 0; i < N; ++i) {
                if (!ps.reservoirs[i].HasSample())
                    continue;

                RISLightSample candidate = ps.reservoirs[i].GetSample();
                CHECK_GT(candidate.g, 0);

                pstd::optional<LightLiSample> ls = candidate.light.Sample_Li(intr, candidate.u, lambda);
                // May be unset if we've resampled someone else's...
                // TODO: this should no longer be the case if we reweight during resampling...
                if (!ls || ls->pdf == 0 || !ls->Unoccluded(scene))
                    continue;

                // p(x) (source distribution): light
                // g(x) (resampling distribution): brdf * light
                // w = g / p
                // draw one sample with probability proportional to w
                // f(x) (full function) = brdf * light * visibility
                // single-sample RIS estimator: f(y) / g(y) * (1/M) * sum_i^M g(x_i)/p(x_i)
                Vector3f wo = intr.wo, wi = ls->wi;
                SampledSpectrum LL =
                    (intr.bsdf->f(wo, wi) * AbsDot(wi, intr.shading.n) * ls->L) / candidate.g *
                    (ps.reservoirs[i].WeightSum() / ps.reservoirs[i].NSamplesConsidered()) /
                    (N * ls->pdf);
                L += LL;
                CHECK(!L.HasNaNs());
            }

            // Perfect specular, just to see if it hits a light
            SamplerHandle &sampler = threadSamplers[ThreadIndex];
            sampler.StartPixelSample(pPixel, sampleIndex);
            (void)sampler.GetCameraSample(pPixel, camera->film->filter);
            (void)sampler.Get1D();
            (void)sampler.Get2D();

            Float u = sampler.Get1D();
            pstd::optional<BSDFSample> bs = intr.bsdf->Sample_f(intr.wo, u, sampler.Get2D());
            if (bs && bs->pdf > 0 && bs->IsSpecular()) {
                Ray ray = intr.SpawnRay(bs->wi);
                pstd::optional<ShapeIntersection> si = scene.Intersect(ray);
                if (si)
                    L += bs->f * AbsDot(bs->wi, intr.shading.n) * si->intr.Le(-ray.d, lambda) /
                        bs->pdf;
            }

            camera->film->AddSample(pPixel, L, lambda, {}, 1.f);
        }
    }, "Trace and filter");
    }

    LOG_VERBOSE("Writing image with spp = %d", spp);
    ImageMetadata metadata;
    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    metadata.samplesPerPixel = spp;
    camera->InitMetadata(&metadata);
    camera->film->WriteImage(metadata, 1. / spp);

    LOG_VERBOSE("Rendering finished");
}

std::string RISIntegrator::ToString() const {
    return "TODO RISIntegrator::ToString()";
}

std::unique_ptr<RISIntegrator> RISIntegrator::Create(
    const ParameterDictionary &dict, const Scene &scene,
    std::unique_ptr<const Camera> camera,
    SamplerHandle sampler, const FileLoc *loc) {
    int N = dict.GetOneInt("N", 1);
    int M = dict.GetOneInt("M", 9);
    bool mis = dict.GetOneBool("mis", true);
    std::string lightStrategy =
        dict.GetOneString("lightsamplestrategy", "bvh");

    std::string lms = dict.GetOneString("lightmisstrategy", "M");
    RISIntegrator::LightMISStrategy lightMISStrategy;
    if (lms == "M")
        lightMISStrategy = RISIntegrator::LightMISStrategy::M;
    else if (lms == "sqrtM")
        lightMISStrategy = RISIntegrator::LightMISStrategy::SqrtM;
    else if (lms == "one")
        lightMISStrategy = RISIntegrator::LightMISStrategy::One;
    else
        ErrorExit(loc, "%s: unknown light MIS strategy", lms.c_str());

    int nSpatioResamples = dict.GetOneInt("nresamples", 5);

    return std::make_unique<RISIntegrator>(N, M, nSpatioResamples, mis, lightMISStrategy,
                                           lightStrategy, scene, std::move(camera),
                                           sampler.SamplesPerPixel());
}

std::unique_ptr<Integrator> Integrator::Create(
        const std::string &name, const ParameterDictionary &dict,
        const Scene &scene, std::unique_ptr<Camera> camera,
        SamplerHandle sampler,
        const RGBColorSpace *colorSpace, const FileLoc *loc) {
    std::unique_ptr<Integrator> integrator;
    if (name == "whitted")
        integrator = WhittedIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "path")
        integrator = PathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "simplepath")
        integrator = SimplePathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "lightpath")
        integrator = LightPathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "volpath")
        integrator = VolPathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "bdpt")
        integrator = BDPTIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "mlt")
        integrator = MLTIntegrator::Create(dict, scene, std::move(camera), loc);
    else if (name == "ambientocclusion")
        integrator = AOIntegrator::Create(dict, scene, &colorSpace->illuminant,
                                          std::move(camera), std::move(sampler), loc);
    else if (name == "ris")
        integrator = RISIntegrator::Create(dict, scene, std::move(camera), std::move(sampler), loc);
    else if (name == "sppm")
        integrator = SPPMIntegrator::Create(dict, scene, colorSpace, std::move(camera), loc);
    else
        ErrorExit(loc, "%s: integrator type unknown.", name);

    if (!integrator)
        ErrorExit(loc, "%s: unable to create integrator.", name);

    dict.ReportUnused();
    return integrator;
}

}  // namespace pbrt
