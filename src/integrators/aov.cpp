
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

// integrators/aov.cpp*
#include "integrators/aov.h"

#include "error.h"
#include "fileutil.h"
#include "film.h"
#include "image.h"
#include "interaction.h"
#include "lightdistrib.h"
#include "lowdiscrepancy.h"
#include "parallel.h"
#include "paramset.h"
#include "progressreporter.h"
#include "reflection.h"
#include "samplers/halton.h"
#include "stats.h"
#include "scene.h"

#include <mutex>

using gtl::ArraySlice;

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/AOV images", aovImageBytes);

void AOVIntegrator::Render(const Scene &scene) {
    // Allocate FBs (Image vs e.g. vector<Point3f>?)
    Bounds2i croppedPixelBounds = camera->film->croppedPixelBounds;
    Point2i resolution = Point2i(croppedPixelBounds.Diagonal());

    Image pImage(PixelFormat::RGB32, resolution);
    Image nImage(PixelFormat::RGB32, resolution);
    Image nsImage(PixelFormat::RGB32, resolution);
    Image uvImage(PixelFormat::RGB32, resolution);
    Image albedoImage(PixelFormat::RGB32, resolution);
    Image aoImage(PixelFormat::Y32, resolution);
    Image eImage(PixelFormat::RGB32, resolution);
    Image shapeIdImage(PixelFormat::RGB32, resolution);
    Image materialIdImage(PixelFormat::RGB32, resolution);
    int totalChannels = 6 * 3 + 1 * 1;
    aovImageBytes += totalChannels * sizeof(Float) * croppedPixelBounds.Area();

    std::mutex nameToSpectrumMutex;
    std::map<std::string, Spectrum> shapeNameToSpectrum, materialNameToSpectrum;
    std::vector<std::string> shapeNameMetadata, materialNameMetadata;

    std::unique_ptr<Sampler> sampler =
        std::make_unique<HaltonSampler>(1, croppedPixelBounds);

    albedoSamples = sampler->RoundCount(albedoSamples);
    sampler->Request2DArray(albedoSamples);

    if (aoSamples > 0) {
        aoSamples = sampler->RoundCount(aoSamples);
        sampler->Request2DArray(aoSamples);
    }
    if (eSamples > 0) {
        eSamples = sampler->RoundCount(eSamples);
        sampler->Request1DArray(eSamples);
        sampler->Request2DArray(eSamples);
    }

    std::unique_ptr<LightDistribution> lightDistrib =
        std::make_unique<SpatialLightDistribution>(scene);

    const int tileSize = 16;
    ProgressReporter reporter(resolution.x * resolution.y, "Rendering");
    ParallelFor2D(croppedPixelBounds, tileSize, [&](Bounds2i tileBounds) {
        // Allocate _MemoryArena_ for tile
        MemoryArena arena;

        // Get sampler instance for tile
        std::unique_ptr<Sampler> tileSampler = sampler->Clone();

        LOG(INFO) << "Starting image tile " << tileBounds;

        // Loop over pixels in tile to render them
        for (Point2i pPixel : tileBounds) {
            if (!InsideExclusive(pPixel, pixelBounds)) continue;

            tileSampler->StartSequence(pPixel, 0);

            // Initialize _CameraSample_ for the center of the domain.
            CameraSample cameraSample;
            cameraSample.pFilm = (Point2f)pPixel + Point2f(0.5f, 0.5f);
            cameraSample.time = 0.5f;
            cameraSample.pLens = Point2f(0.5f, 0.5f);

            // Generate camera ray for current sample
            RayDifferential ray;
            Float rayWeight =
                camera->GenerateRayDifferential(cameraSample, &ray);
            if (rayWeight == 0) continue;

            auto findIntersectionAndShade = [&scene, &arena](
                RayDifferential &ray, SurfaceInteraction *isect) {
                retry:
                    bool hit = scene.Intersect(ray, isect);
                    if (hit) {
                        isect->ComputeScatteringFunctions(ray, arena);
                        if (!isect->bsdf) {
                            ray = isect->SpawnRay(ray.d);
                            goto retry;
                        }
                    }
                    return hit;
            };

            SurfaceInteraction isect;
            if (findIntersectionAndShade(ray, &isect)) {
                Point2i p = Point2i(pPixel - croppedPixelBounds.pMin);

                for (int c = 0; c < 3; ++c) {
                    pImage.SetChannel(p, c, isect.p[c]);
                    nImage.SetChannel(p, c, isect.n[c]);
                    nsImage.SetChannel(p, c, isect.shading.n[c]);
                }
                for (int c = 0; c < 2; ++c)
                    uvImage.SetChannel(p, c, isect.uv[c]);

                std::string shapeName = isect.primitive->GetAttributes() ?
                    isect.primitive->GetAttributes()->GetOneString("name", "") : "";
                Spectrum shapeIdSpectrum(0.f);
                if (shapeName != "") {
                    std::lock_guard<std::mutex> lock(nameToSpectrumMutex);
                    if (shapeNameToSpectrum.find(shapeName) ==
                        shapeNameToSpectrum.end()) {
                        int index = shapeNameToSpectrum.size() + 1;
                        std::array<Float, 3> rgb = {
                            SobolSampleFloat(index, 0),
                            SobolSampleFloat(index, 1),
                            SobolSampleFloat(index, 2) };
                        shapeNameToSpectrum[shapeName] = Spectrum::FromRGB(rgb);
                        shapeNameMetadata.push_back(
                            StringPrintf("%f %f %f %s", rgb[0], rgb[1], rgb[2], shapeName.c_str()));
                    }
                    shapeIdSpectrum = shapeNameToSpectrum[shapeName];
                }
                shapeIdImage.SetSpectrum(p, shapeIdSpectrum);

                std::string materialName = "";
                if (isect.primitive->GetMaterial() &&
                    isect.primitive->GetMaterial()->attributes)
                    materialName = isect.primitive->GetMaterial()->attributes->
                        GetOneString("name", "");
                Spectrum materialIdSpectrum(0.f);
                if (materialName != "") {
                    std::lock_guard<std::mutex> lock(nameToSpectrumMutex);
                    if (materialNameToSpectrum.find(materialName) ==
                        materialNameToSpectrum.end()) {
                        int index = materialNameToSpectrum.size() + 1;
                        std::array<Float, 3> rgb = {
                            SobolSampleFloat(index, 0),
                            SobolSampleFloat(index, 1),
                            SobolSampleFloat(index, 2) };
                        materialNameToSpectrum[materialName] = Spectrum::FromRGB(rgb);
                        materialNameMetadata.push_back(
                            StringPrintf("%f %f %f %s", rgb[0], rgb[1], rgb[2], materialName.c_str()));
                    }
                    materialIdSpectrum = materialNameToSpectrum[materialName];
                }
                materialIdImage.SetSpectrum(p, materialIdSpectrum);

                // Order--albedo, ao, e is important--must match order of
                // sample array requests.
                Spectrum albedo = isect.bsdf->rho(
                    -ray.d, tileSampler->Get2DArray(albedoSamples));
                albedoImage.SetSpectrum(p, albedo);

                Float ao = 0;
                if (aoSamples > 0) {
                    Normal3f n = Faceforward(isect.n, -ray.d);
                    Vector3f s = Normalize(isect.dpdu);
                    Vector3f t = Cross(isect.n, s);

                    ArraySlice<Point2f> u = tileSampler->Get2DArray(aoSamples);
                    for (int i = 0; i < aoSamples; ++i) {
                        Vector3f wi = CosineSampleHemisphere(u[i]);
                        Float pdf = CosineHemispherePdf(std::abs(wi.z));

                        // Transform wi from local frame to world space.
                        wi = Vector3f(s.x * wi.x + t.x * wi.y + n.x * wi.z,
                                      s.y * wi.x + t.y * wi.y + n.y * wi.z,
                                      s.z * wi.x + t.z * wi.y + n.z * wi.z);

                        // TODO/FIXME: does this handle alpha??
                        Ray ray = isect.SpawnRay(wi);
                        ray.tMax = aoMaxDist;
                        if (!scene.IntersectP(ray))
                            // since we're doing cosine-weighted sampling,
                            // the pdf and the (n . w) cancel out.
                            ao += 1.f / aoSamples;
                    }
                    aoImage.SetChannel(p, 0, ao);
                }

                if (eSamples > 0) {
                    Normal3f n = Faceforward(isect.n, -ray.d);

                    ArraySlice<Float> ul = tileSampler->Get1DArray(eSamples);
                    ArraySlice<Point2f> u = tileSampler->Get2DArray(eSamples);
                    Spectrum E(0);
                    const Distribution1D *ld = lightDistrib->Lookup(isect.p);
                    for (int i = 0; i < eSamples; ++i) {
                        Float lightPdf;
                        int lightIndex = ld->SampleDiscrete(ul[i], &lightPdf);
                        const Light *light = scene.lights[lightIndex].get();

                        Vector3f wi;
                        Float pdf;
                        VisibilityTester vis;
                        Spectrum Li =
                            light->Sample_Li(isect, u[i], &wi, &pdf, &vis);
                        if (Dot(n, wi) > 0 && !Li.IsBlack() &&
                            vis.Unoccluded(scene))
                            E += Li * Dot(n, wi) / (lightPdf * pdf * eSamples);
                    }
                    eImage.SetSpectrum(p, E);
                }
            }

            // Free _MemoryArena_ memory from computing image sample
            // value
            arena.Reset();
        }
        LOG(INFO) << "Finished image tile " << tileBounds;
        reporter.Update(tileBounds.Area());
    });
    reporter.Done();

    LOG(INFO) << "Rendering finished";

    // Write images
    auto mungeFilename = [](std::string base, const std::string &aov) {
        auto dot = base.find_last_of('.');
        CHECK_NE(dot, std::string::npos);
        return base.insert(dot, aov);
    };

    ImageMetadata metadata;
    metadata.renderTimeSeconds = reporter.ElapsedMS() / 1000.;
    metadata.pixelBounds = camera->film->croppedPixelBounds;
    metadata.fullResolution = camera->film->fullResolution;
    pImage.Write(mungeFilename(camera->film->filename, "_p"), &metadata);
    nImage.Write(mungeFilename(camera->film->filename, "_n"), &metadata);
    nsImage.Write(mungeFilename(camera->film->filename, "_ns"), &metadata);
    uvImage.Write(mungeFilename(camera->film->filename, "_uv"), &metadata);
    albedoImage.Write(mungeFilename(camera->film->filename, "_albedo"), &metadata);

    ImageMetadata shapeImageMetadata = metadata;
    shapeImageMetadata.stringVectors["shapeIds"] = shapeNameMetadata;
    shapeIdImage.Write(mungeFilename(camera->film->filename, "_shapeid"),
                       &shapeImageMetadata);

    ImageMetadata materialImageMetadata = metadata;
    materialImageMetadata.stringVectors["materialIds"] = materialNameMetadata;
    materialIdImage.Write(mungeFilename(camera->film->filename, "_materialid"),
                          &materialImageMetadata);

    if (aoSamples > 0)
        aoImage.Write(mungeFilename(camera->film->filename, "_ao"), &metadata);
    if (eSamples > 0) eImage.Write(mungeFilename(camera->film->filename, "_E"),
                                   &metadata);
}

std::unique_ptr<AOVIntegrator> CreateAOVIntegrator(
    const ParamSet &params, std::shared_ptr<const Camera> camera) {
    int albedoSamples = params.GetOneInt("abledosamples", 32);
    int aoSamples = params.GetOneInt("aosamples", 512);
    Float aoMaxDist = params.GetOneFloat("aomaxdist", Infinity);
    int eSamples = params.GetOneInt("esamples", 512);

    if (!HasExtension(camera->film->filename, "exr") &&
        !HasExtension(camera->film->filename, "pfm")) {
        Error(
            "%s: image format doesn't support floating-point values; use "
            "EXR or PFM.",
            camera->film->filename.c_str());
        return nullptr;
    }

    gtl::ArraySlice<int> pb = params.GetIntArray("pixelbounds");
    Bounds2i pixelBounds = camera->film->GetSampleBounds();
    if (!pb.empty()) {
        if (pb.size() != 4)
            Error("Expected four values for \"pixelbounds\" parameter. Got %d.",
                  (int)pb.size());
        else {
            pixelBounds = Intersect(pixelBounds,
                                    Bounds2i{{pb[0], pb[2]}, {pb[1], pb[3]}});
            if (pixelBounds.Empty())
                Error("Degenerate \"pixelbounds\" specified.");
        }
    }

    return std::make_unique<AOVIntegrator>(camera, pixelBounds, albedoSamples,
                                           aoSamples, aoMaxDist, eSamples);
}

}  // namespace pbrt
