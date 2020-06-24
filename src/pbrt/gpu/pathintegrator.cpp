// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/gpu/pathintegrator.h>

#include <pbrt/base/medium.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/direct.h>
#include <pbrt/gpu/indirect.h>
#include <pbrt/gpu/launch.h>
#include <pbrt/gpu/material.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/lights.h>
#include <pbrt/lightsamplers.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/samplers.h>
#include <pbrt/textures.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/taggedptr.h>

#include <cstring>
#include <iostream>
#include <map>

#if (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ == 2)
  #include <cuda/std/std/atomic>
#else
  #include <cuda/std/atomic>
#endif

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/GPU path integrator pixel state", pathIntegratorBytes);

GPUPathIntegrator::GPUPathIntegrator(Allocator alloc, const ParsedScene &scene)
    : allLights(alloc), cameraRayWeights(alloc), shadowRayLd(alloc) {
    std::map<std::string, MediumHandle> media = scene.CreateMedia(alloc);

    // Slightly conservative, but...
    haveMedia = !media.empty();
    auto findMedium = [&media](const std::string &s, const FileLoc *loc) -> MediumHandle {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        return iter->second;
    };

    haveSubsurface = false;
    for (const auto &mtl : scene.materials)
        haveSubsurface |= (mtl.name == "subsurface");
    for (const auto &mtlPair : scene.namedMaterials)
        haveSubsurface |= (mtlPair.second.name == "subsurface");

    filter = FilterHandle::Create(scene.filter.name, scene.filter.parameters,
                                  &scene.filter.loc, alloc);

    film = FilmHandle::Create(scene.film.name, scene.film.parameters, &scene.film.loc,
                              filter, alloc);
    initializeVisibleSurface = film.UsesVisibleSurface();

    sampler = SamplerHandle::Create(scene.sampler.name, scene.sampler.parameters,
                                    film.FullResolution(), &scene.sampler.loc, alloc);

    MediumHandle cameraMedium = findMedium(scene.camera.medium, &scene.camera.loc);
    camera = CameraHandle::Create(scene.camera.name, scene.camera.parameters,
                                  cameraMedium, scene.camera.cameraTransform, film,
                                  &scene.camera.loc, alloc);

    maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    {
        for (const auto &light : scene.lights) {
            MediumHandle outsideMedium = findMedium(light.medium, &light.loc);
            LightHandle l = LightHandle::Create(
                light.name, light.parameters, light.worldFromObject,
                scene.camera.cameraTransform, outsideMedium, &light.loc, alloc);
            if (l.Is<UniformInfiniteLight>() || l.Is<ImageInfiniteLight>()) {
                if (envLight)
                    Warning(&light.loc,
                            "Multiple infinite lights provided. Using this one.");
                envLight = l;
            }

            allLights.push_back(l);
        }
    }

    // Area lights...
    std::map<int, pstd::vector<LightHandle> *> shapeIndexToAreaLights;
    {
        for (size_t i = 0; i < scene.shapes.size(); ++i) {
            const auto &shape = scene.shapes[i];
            if (shape.lightIndex == -1)
                continue;
            CHECK_LT(shape.lightIndex, scene.areaLights.size());
            const auto &areaLightEntity = scene.areaLights[shape.lightIndex];
            AnimatedTransform worldFromLight(*shape.worldFromObject);

            pstd::vector<ShapeHandle> shapeHandles = ShapeHandle::Create(
                shape.name, shape.worldFromObject, shape.objectFromWorld,
                shape.reverseOrientation, shape.parameters, &shape.loc, alloc);

            if (shapeHandles.empty())
                continue;

            MediumHandle outsideMedium = findMedium(shape.outsideMedium, &shape.loc);

            pstd::vector<LightHandle> *lightsForShape =
                alloc.new_object<pstd::vector<LightHandle>>(alloc);
            for (ShapeHandle sh : shapeHandles) {
                DiffuseAreaLight *area = DiffuseAreaLight::Create(
                    worldFromLight, outsideMedium, areaLightEntity.parameters,
                    areaLightEntity.parameters.ColorSpace(), &areaLightEntity.loc, alloc,
                    sh);
                allLights.push_back(area);
                lightsForShape->push_back(area);
            }
            shapeIndexToAreaLights[i] = lightsForShape;
        }
    }

    accel = new GPUAccel(scene, alloc, nullptr /* cuda stream */, shapeIndexToAreaLights,
                         media);

    // preprocess...
    for (LightHandle light : allLights)
        light.Preprocess(accel->Bounds());

    bool haveLights = !allLights.empty();
    for (const auto &m : media)
        haveLights |= m.second.IsEmissive();
    if (!haveLights)
        ErrorExit("No light sources specified");

    std::string lightSamplerName =
        scene.integrator.parameters.GetOneString("lightsampler", "bvh");
    lightSampler = LightSamplerHandle::Create(lightSamplerName, allLights, alloc);

    regularize = scene.integrator.parameters.GetOneBool("regularize", true);

    ///////////////////////////////////////////////////////////////////////////
    // Allocate storage for all of the state...
    CUDATrackedMemoryResource *mr =
        dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
    CHECK(mr != nullptr);
    size_t startSize = mr->BytesAllocated();

    // Compute number of scanlines to render per pass.
    Vector2i resolution = film.PixelBounds().Diagonal();
    int maxSamples = 1024 * 1024;  // TODO: make configurable
    scanlinesPerPass = std::max(1, maxSamples / resolution.x);
    int nPasses = (resolution.y + scanlinesPerPass - 1) / scanlinesPerPass;
    scanlinesPerPass = (resolution.y + nPasses - 1) / nPasses;
    pixelsPerPass = resolution.x * scanlinesPerPass;
    LOG_VERBOSE("Will render image in %d passes %d scanlines per pass (%d samples)\n",
                nPasses, scanlinesPerPass, pixelsPerPass);

    Point2i *pPixel = alloc.allocate_object<Point2i>(pixelsPerPass);
    pPixels = pstd::MakeSpan(pPixel, pixelsPerPass);

    SampledWavelengths *lambda = alloc.allocate_object<SampledWavelengths>(pixelsPerPass);
    for (int i = 0; i < pixelsPerPass; ++i)
        alloc.construct(&lambda[i]);
    lambdas = pstd::MakeSpan(lambda, pixelsPerPass);

    pathRayQueue = alloc.new_object<RayQueue<PathRayIndex>>(alloc, pixelsPerPass);
    numActiveRays = alloc.allocate_object<int>();

    PixelIndex *rayToPixel0 = alloc.allocate_object<PixelIndex>(pixelsPerPass);
    PixelIndex *rayToPixel1 = alloc.allocate_object<PixelIndex>(pixelsPerPass);
    rayIndexToPixelIndex[0] = pstd::MakeSpan(rayToPixel0, pixelsPerPass);
    rayIndexToPixelIndex[1] = pstd::MakeSpan(rayToPixel1, pixelsPerPass);

    PathRayIndex *pixelToRay = alloc.allocate_object<PathRayIndex>(pixelsPerPass);
    pixelIndexToRayIndex = pstd::MakeSpan(pixelToRay, pixelsPerPass);

    shadowRayQueue = alloc.new_object<RayQueue<ShadowRayIndex>>(alloc, pixelsPerPass);

    PixelIndex *shadowToPixel = alloc.allocate_object<PixelIndex>(pixelsPerPass);
    shadowRayIndexToPixelIndex = pstd::MakeSpan(shadowToPixel, pixelsPerPass);

    if (haveSubsurface) {
        randomHitRayQueue = alloc.new_object<RayQueue<SSRayIndex>>(alloc, pixelsPerPass);

        MaterialHandle *smat = alloc.allocate_object<MaterialHandle>(pixelsPerPass);
        subsurfaceMaterials = pstd::MakeSpan(smat, pixelsPerPass);

        PathRayIndex *subsurfaceRayToRay =
            alloc.allocate_object<PathRayIndex>(pixelsPerPass);
        subsurfaceRayIndexToPathRayIndex =
            pstd::MakeSpan(subsurfaceRayToRay, pixelsPerPass);

        WeightedReservoirSampler<SurfaceInteraction> *subsurfaceWRS =
            alloc.allocate_object<WeightedReservoirSampler<SurfaceInteraction>>(
                pixelsPerPass);
        for (int i = 0; i < pixelsPerPass; ++i)
            alloc.construct(&subsurfaceWRS[i]);
        subsurfaceReservoirSamplers = pstd::MakeSpan(subsurfaceWRS, pixelsPerPass);
    }

    cameraRayWeights = SampledSpectrumSOA<PixelIndex>(alloc, pixelsPerPass);

    SurfaceInteraction *intrs[2];
    intrs[0] = alloc.allocate_object<SurfaceInteraction>(pixelsPerPass);
    intrs[1] = alloc.allocate_object<SurfaceInteraction>(pixelsPerPass);
    for (int i = 0; i < pixelsPerPass; ++i) {
        alloc.construct(&intrs[0][i]);
        alloc.construct(&intrs[1][i]);
    }
    intersections[0] = pstd::MakeSpan(intrs[0], pixelsPerPass);
    intersections[1] = pstd::MakeSpan(intrs[1], pixelsPerPass);

    if (haveMedia) {
        MediumInteraction *mediumIntrs[2];
        mediumIntrs[0] = alloc.allocate_object<MediumInteraction>(pixelsPerPass);
        mediumIntrs[1] = alloc.allocate_object<MediumInteraction>(pixelsPerPass);
        for (int i = 0; i < pixelsPerPass; ++i) {
            alloc.construct(&mediumIntrs[0][i]);
            alloc.construct(&mediumIntrs[1][i]);
        }
        mediumInteractions[0] = pstd::MakeSpan(mediumIntrs[0], pixelsPerPass);
        mediumInteractions[1] = pstd::MakeSpan(mediumIntrs[1], pixelsPerPass);
    }

    InteractionType *it = alloc.allocate_object<InteractionType>(pixelsPerPass);
    interactionType = pstd::MakeSpan(it, pixelsPerPass);

    SampledSpectrum *str = alloc.allocate_object<SampledSpectrum>(pixelsPerPass);
    shadowTr = pstd::MakeSpan(str, pixelsPerPass);

    shadowRayLd = SampledSpectrumSOA<ShadowRayIndex>(alloc, pixelsPerPass);

    PathState *pathState = alloc.allocate_object<PathState>(pixelsPerPass);
    for (int i = 0; i < pixelsPerPass; ++i)
        alloc.construct(&pathState[i]);
    pathStates = pstd::MakeSpan(pathState, pixelsPerPass);

    constexpr int scratchBufferSize = 512;
    uint8_t *scratchBufferMemory =
        alloc.allocate_object<uint8_t>(pixelsPerPass * scratchBufferSize);
    ScratchBuffer *sbs = alloc.allocate_object<ScratchBuffer>(pixelsPerPass);
    for (int i = 0; i < pixelsPerPass; ++i)
        sbs[i] =
            ScratchBuffer(scratchBufferMemory + i * scratchBufferSize, scratchBufferSize);
    scratchBuffers = pstd::MakeSpan(sbs, pixelsPerPass);

    bxdfEvalQueues = alloc.new_object<
        MultiWorkQueue<BxDFHandle::NumTags(), PathRayIndex, BxDFEvalIndex>>(
        alloc, pixelsPerPass);

    if (haveMedia) {
        mediumEvalQueue = alloc.new_object<WorkQueue<PathRayIndex, MediumEvalIndex>>(
            alloc, pixelsPerPass);
        mediumSampleQueue = alloc.new_object<WorkQueue<PathRayIndex, MediumEvalIndex>>(
            alloc, pixelsPerPass);
    }

    if (initializeVisibleSurface) {
        pstd::optional<VisibleSurface> *vs =
            alloc.allocate_object<pstd::optional<VisibleSurface>>(pixelsPerPass);
        for (int i = 0; i < pixelsPerPass; ++i)
            alloc.construct(&vs[i]);
        visibleSurfaces = pstd::MakeSpan(vs, pixelsPerPass);
    }

    SamplerHandle *sh = alloc.allocate_object<SamplerHandle>(pixelsPerPass);
    samplers =
        TypedIndexSpan<SamplerHandle, PixelIndex>(pstd::MakeSpan(sh, pixelsPerPass));

    std::vector<SamplerHandle> clonedSamplers = sampler.Clone(pixelsPerPass, alloc);
    for (int i = 0; i < pixelsPerPass; ++i)
        samplers[PixelIndex(i)] = clonedSamplers[i];

    if (haveMedia) {
        RNG **rs = alloc.allocate_object<RNG *>(pixelsPerPass);
        for (int i = 0; i < pixelsPerPass; ++i)
            rs[i] = &samplers[PixelIndex(i)].GetRNG();
        rngs = pstd::MakeSpan(rs, pixelsPerPass);
    }

    size_t endSize = mr->BytesAllocated();
    pathIntegratorBytes += endSize - startSize;
}

void GPUPathIntegrator::TraceShadowRays() {
    std::pair<cudaEvent_t, cudaEvent_t> events;
    if (haveMedia)
        events = accel->IntersectShadowTr(shadowRayQueue, pixelsPerPass, shadowTr,
                                          shadowRayIndexToPixelIndex, lambdas, rngs);
    else
        events = accel->IntersectShadow(shadowRayQueue, pixelsPerPass, shadowTr);
    struct IsectShadowHack {};
    GetGPUKernelStats<IsectShadowHack>("Path tracing shadow rays")
        .launchEvents.push_back(events);

    // Add contribution if light was visible
    GPUParallelFor(
        "Incorporate shadow ray contribution", pixelsPerPass,
        [=] PBRT_GPU(ShadowRayIndex shadowRayIndex) {
            if (shadowRayIndex >= shadowRayQueue->Size() || !shadowTr[shadowRayIndex])
                return;

            SampledSpectrum Ld =
                SampledSpectrum(shadowRayLd[shadowRayIndex]) * shadowTr[shadowRayIndex];
            PixelIndex pixelIndex = shadowRayIndexToPixelIndex[shadowRayIndex];
            PathState &pathState = pathStates[pixelIndex];
            pathState.L += Ld;
        });

    GPUDo("Reset shadowRayQueue", shadowRayQueue->Reset(););
}

void GPUPathIntegrator::Render(ImageMetadata *metadata) {
    Vector2i resolution = film.PixelBounds().Diagonal();
    int spp = sampler.SamplesPerPixel();

    RGB *displayRGB = nullptr, *displayRGBHost = nullptr;
    std::atomic<bool> exitCopyThread{false};
    std::thread copyThread;

    if (Options->displayServer) {
        CUDA_CHECK(cudaMalloc(&displayRGB, resolution.x * resolution.y * sizeof(RGB)));
        CUDA_CHECK(cudaMemset(displayRGB, 0, resolution.x * resolution.y * sizeof(RGB)));

        // Let this leak so that the lambda passed to DisplayDynamic below
        // doesn't access freed memory after Render() returns...
        displayRGBHost = new RGB[resolution.x * resolution.y];

        copyThread = std::thread([&]() {
            // Copy back to the CPU using a separate stream so that we can
            // periodically but asynchronously pick up the latest results
            // from the GPU.
            cudaStream_t memcpyStream;
            CUDA_CHECK(cudaStreamCreate(&memcpyStream));

            while (!exitCopyThread) {
                CUDA_CHECK(cudaMemcpyAsync(displayRGBHost, displayRGB,
                                           resolution.x * resolution.y * sizeof(RGB),
                                           cudaMemcpyDeviceToHost, memcpyStream));

                std::this_thread::sleep_for(std::chrono::milliseconds(50));

                CUDA_CHECK(cudaStreamSynchronize(memcpyStream));
            }

            // One more time...
            CUDA_CHECK(cudaMemcpy(displayRGBHost, displayRGB,
                                  resolution.x * resolution.y * sizeof(RGB),
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
        });

        DisplayDynamic(film.GetFilename(), {resolution.x, resolution.y}, {"R", "G", "B"},
                       [resolution, displayRGBHost](
                           Bounds2i b, pstd::span<pstd::span<Float>> displayValue) {
                           int index = 0;
                           for (Point2i p : b) {
                               RGB rgb = displayRGBHost[p.x + p.y * resolution.x];
                               displayValue[0][index] = rgb.r;
                               displayValue[1][index] = rgb.g;
                               displayValue[2][index] = rgb.b;
                               ++index;
                           }
                       });
    }

    ProgressReporter progress(spp, "Rendering", Options->quiet, true /* GPU */);

    for (int sampleIndex = 0; sampleIndex < spp; ++sampleIndex) {
        for (int y0 = 0; y0 < resolution.y; y0 += scanlinesPerPass) {
            GPUParallelFor(
                "Initialize pPixels", pixelsPerPass,
                [=] PBRT_GPU(PixelIndex pixelIndex, Point2i pMin) {
                    pPixels[pixelIndex] =
                        Point2i(pMin.x + int(pixelIndex) % resolution.x,
                                pMin.y + y0 + int(pixelIndex) / resolution.x);
                },
                film.PixelBounds().pMin);

            // Generate camera rays
            GPUDo("Reset ray queue", pathRayQueue->Reset(););

            GPUParallelFor("Reset sampler dimension", pixelsPerPass,
                           [=] PBRT_GPU(PixelIndex pixelIndex) {
                               SamplerHandle sampler = samplers[pixelIndex];
                               sampler.StartPixelSample(pPixels[pixelIndex], sampleIndex);
                           });

            GenerateCameraRays(sampleIndex);

            GPUDo("Set numActiveRays", *numActiveRays = pathRayQueue->Size(););

            GPUParallelFor("Initialize PathState", pixelsPerPass,
                           [=] PBRT_GPU(PixelIndex pixelIndex) {
                               // Initialize all of them, to be sure we have
                               // zero L for rays that didn't spawn.
                               PathState &pathState = pathStates[pixelIndex];
                               pathState.L = SampledSpectrum(0.f);
                               pathState.beta = SampledSpectrum(1.f);
                               pathState.etaScale = 1.f;
                               pathState.anyNonSpecularBounces = false;

                               if (initializeVisibleSurface)
                                   visibleSurfaces[pixelIndex].reset();
                           });

            for (int depth = 0; true; ++depth) {
                GPUParallelFor(
                    "Clear intersections", pixelsPerPass,
                    [=] PBRT_GPU(PixelIndex pixelIndex) {
                        intersections[depth & 1][pixelIndex].bsdf = nullptr;
                        intersections[depth & 1][pixelIndex].bssrdf = nullptr;
                        intersections[depth & 1][pixelIndex].material = nullptr;
                        intersections[depth & 1][pixelIndex].areaLight = nullptr;

                        ScratchBuffer &scratchBuffer = scratchBuffers[pixelIndex];
                        scratchBuffer.Reset();
                    });

                auto events = accel->IntersectClosest(
                    pathRayQueue, pixelsPerPass, intersections[depth & 1],
                    rayIndexToPixelIndex[depth & 1], interactionType);
                struct IsectHack {};
                GetGPUKernelStats<IsectHack>("Path tracing closest hit rays")
                    .launchEvents.push_back(events);

                if (haveMedia)
                    SampleMediumInteraction(depth);

                GPUParallelFor(
                    "Handle ray-found emission", pixelsPerPass,
                    [=] PBRT_GPU(PathRayIndex rayIndex) {
                        if (rayIndex >= *numActiveRays)
                            return;

                        PixelIndex pixelIndex = rayIndexToPixelIndex[depth & 1][rayIndex];
                        PathState *pathState = &pathStates[pixelIndex];
                        SampledSpectrum beta = pathState->beta;

                        if (!beta)
                            return;

                        SurfaceInteraction &intersection =
                            intersections[depth & 1][pixelIndex];
                        const SampledWavelengths &lambda = lambdas[pixelIndex];
                        Ray ray = pathRayQueue->GetRay(rayIndex);

                        // Hit something; add surface emission if there is any
                        SampledSpectrum Le(0.f);
                        LightHandle light;
                        if (interactionType[rayIndex] == InteractionType::Surface &&
                            intersection.areaLight) {
                            Vector3f wo = -ray.d;
                            Le = intersection.areaLight.L(intersection, wo, lambda);
                            light = intersection.areaLight;
                        } else if (envLight &&
                                   interactionType[rayIndex] == InteractionType::None) {
                            Le = envLight.Le(ray, lambda);
                            light = envLight;
                        }

                        if (Le) {
                            Float weight;
                            if (depth == 0 || IsSpecular(pathState->bsdfFlags))
                                weight = 1;
                            else {
                                const SurfaceInteraction &prevIntr =
                                    intersections[(depth & 1) ^ 1][pixelIndex];
                                // Compute MIS pdf...
                                Float lightChoicePDF = lightSampler.PDF(prevIntr, light);
                                Float lightPDF = lightChoicePDF *
                                                 light.Pdf_Li(prevIntr, ray.d,
                                                              LightSamplingMode::WithMIS);
                                weight = PowerHeuristic(1, pathState->scatteringPDF, 1,
                                                        lightPDF);
                            }
                            pathState->L += beta * weight * Le;
                        }
                    });

                if (depth == maxDepth)
                    break;

                GPUDo("Reset bxdfEvalQueues", bxdfEvalQueues->Reset(););

                // Basically a 10% perf win for doing this...
                EvaluateMaterial(BasicTextureEvaluator(), depth);
                EvaluateMaterial(UniversalTextureEvaluator(), depth);

                if (depth == 0 && initializeVisibleSurface)
                    InitializeVisibleSurface();

                SampleLight(depth);

                SampleDirect(depth);

                TraceShadowRays();

                GPUDo("Reset ray queue", pathRayQueue->Reset(););

                SampleIndirect(depth);

                if (haveMedia)
                    HandleMediumTransitions(depth);

                if (haveSubsurface)
                    SampleSubsurface(depth);

                // Do this only now, since subsurface needs numActiveRays
                GPUDo("Set numActiveRays", *numActiveRays = pathRayQueue->Size(););
            }

            UpdateFilm();

            if (Options->displayServer)
                GPUParallelFor("Update Display RGB Buffer", pixelsPerPass,
                               [=] PBRT_GPU(PixelIndex pixelIndex) {
                                   Point2i pPixel = pPixels[pixelIndex];
                                   if (!InsideExclusive(pPixel, film.PixelBounds()))
                                       return;

                                   Point2i p(pPixel - film.PixelBounds().pMin);
                                   displayRGB[p.x + p.y * resolution.x] =
                                       film.GetPixelRGB(pPixel);
                               });
        }

        progress.Update();
    }
    progress.Done();

    CUDA_CHECK(cudaDeviceSynchronize());

    // Wait until rendering is all done before we start to shut down the
    // display stuff..
    if (Options->displayServer) {
        exitCopyThread = true;
        copyThread.join();
    }

    metadata->samplesPerPixel = sampler.SamplesPerPixel();
    camera.InitMetadata(metadata);
}

void GPURender(ParsedScene &scene) {
    GPUPathIntegrator *integrator =
        gpuMemoryAllocator.new_object<GPUPathIntegrator>(gpuMemoryAllocator, scene);

    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));
    CUDA_CHECK(
        cudaMemAdvise(integrator, sizeof(*integrator), cudaMemAdviseSetReadMostly, 0));
    CUDA_CHECK(cudaMemAdvise(integrator, sizeof(*integrator),
                             cudaMemAdviseSetPreferredLocation, deviceIndex));

    CUDATrackedMemoryResource *mr =
        dynamic_cast<CUDATrackedMemoryResource *>(gpuMemoryAllocator.resource());
    CHECK(mr != nullptr);
    mr->PrefetchToGPU();

    ///////////////////////// Render!

    Timer timer;

    ImageMetadata metadata;
    integrator->Render(&metadata);

    LOG_VERBOSE("Total rendering time: %.3f s", timer.ElapsedSeconds());

    CUDA_CHECK(cudaProfilerStop());

    if (!Options->quiet)
        ReportKernelStats();

    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    metadata.samplesPerPixel = integrator->sampler.SamplesPerPixel();

    std::vector<GPULogItem> logs = ReadGPULogs();
    for (const auto &item : logs)
        Log(item.level, item.file, item.line, item.message);

    integrator->film.WriteImage(metadata);

    // Cleanup
    // This takes almost a second and doesn't really matter...
    // FreeBufferCaches();
}

}  // namespace pbrt
