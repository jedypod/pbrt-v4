
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


// film.cpp*
#include <pbrt/film.h>

#include <pbrt/bsdf.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Film pixels", filmPixelMemory);

// Film Method Definitions
RGBFilm::RGBFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
                 pstd::unique_ptr<Filter> filt, Float diagonal,
                 const std::string &filename, Float scale,
                 const RGBColorSpace *colorSpace,
                 Float maxSampleLuminance, bool writeFP16, bool saveVariance,
                 Allocator allocator)
    : Film(resolution, pixelBounds, std::move(filt), diagonal, filename),
      pixels(pixelBounds, allocator),
      scale(scale),
      colorSpace(colorSpace),
      maxSampleLuminance(maxSampleLuminance),
      writeFP16(writeFP16),
      saveVariance(saveVariance) {
    // Allocate film image storage
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
}

SampledWavelengths RGBFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleImportance(u);
}

void RGBFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                       const SampledWavelengths &lambda) {
    ProfilerScope pp(ProfilePhase::SplatFilm);

    XYZ xyz = v.ToXYZ(lambda, colorSpace->X, colorSpace->Y, colorSpace->Z);
    Float y = xyz.Y;
    CHECK(!v.HasNaNs());
    CHECK(!std::isinf(y));

    if (y > maxSampleLuminance) {
        v *= maxSampleLuminance / y;
        xyz = v.ToXYZ(lambda, colorSpace->X, colorSpace->Y, colorSpace->Z);
    }
    RGB rgb = colorSpace->ToRGB(xyz);

    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - filter->radius)),
                         Point2i(Floor(pDiscrete + filter->radius)) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);
    for (Point2i pi : splatBounds) {
        Float wt = filter->Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];
            for (int i = 0; i < 3; ++i) pixel.splatRGB[i].Add(wt * rgb[i]);
        }
    }
}

void RGBFilm::WriteImage(ImageMetadata metadata, Float splatScale) {
    Image image = GetImage(&metadata, splatScale);
    LOG_VERBOSE("Writing image %s with bounds %s", filename, pixelBounds);
    image.Write(filename, metadata);
}

Image RGBFilm::GetImage(ImageMetadata *metadata, Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG_VERBOSE("Converting image to RGB and computing final weighted pixel values");
    PixelFormat format = writeFP16 ? PixelFormat::Half : PixelFormat::Float;
    std::vector<std::string> channels = { "R", "G", "B" };
    if (saveVariance) channels.push_back("Variance");
    Image image(format, Point2i(pixelBounds.Diagonal()), channels);

    Float filterIntegral = filter->Integral();

    for (Point2i p : pixelBounds) {
        Pixel &pixel = pixels[p];
        RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum;
        if (weightSum != 0)
            rgb /= weightSum;

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.splatRGB[c] / filterIntegral;

        // Scale pixel value by _scale_
        rgb *= scale;

        Point2i pOffset(p.x - pixelBounds.pMin.x,
                        p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, {rgb[0], rgb[1], rgb[2]});
        if (saveVariance)
            image.SetChannel(pOffset, 3, pixel.varianceEstimator.Variance());
    }

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;

    Float varianceSum = 0;
    for (Point2i p : pixelBounds) {
        const Pixel &pixel = pixels[p];
        varianceSum += Float(pixel.varianceEstimator.Variance());
    }
    metadata->estimatedVariance = varianceSum / pixelBounds.Area();

    return image;
}

std::string RGBFilm::ToString() const {
    return StringPrintf("[ RGBFilm %s scale: %f colorSpace: %s maxSampleLuminance: %f "
                        "writeFP16: %s saveVariance: %s ]", BaseToString(), scale,
                        *colorSpace, maxSampleLuminance, writeFP16, saveVariance);
}

RGBFilm *RGBFilm::Create(const ParameterDictionary &dict,
                         pstd::unique_ptr<Filter> filter,
                         const RGBColorSpace *colorSpace,
                         Allocator alloc) {
    std::string filename = dict.GetOneString("filename", "");
    if (PbrtOptions.imageFile) {
        if (!filename.empty())
            Warning(
                "Output filename supplied on command line, \"%s\" will override "
                "filename provided in scene description file, \"%s\".",
                *PbrtOptions.imageFile, filename);
        filename = *PbrtOptions.imageFile;
    } else if (filename.empty())
        filename = "pbrt.exr";

    Point2i fullResolution(dict.GetOneInt("xresolution", 1280),
                           dict.GetOneInt("yresolution", 720));
    if (PbrtOptions.quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    Bounds2i pixelBounds(Point2i(0, 0), fullResolution);
    std::vector<int> pb = dict.GetIntArray("pixelbounds");
    if (PbrtOptions.pixelBounds) {
        Bounds2i newBounds = *PbrtOptions.pixelBounds;
        if (Intersect(newBounds, pixelBounds) != newBounds)
            Warning("Supplied pixel bounds extend beyond image resolution. Clamping.");
        pixelBounds = Intersect(newBounds, pixelBounds);

        if (!pb.empty())
            Warning("Both pixel bounds and crop window were specified. Using the "
                    "crop window.");
    }
    else if (!pb.empty()) {
        if (pb.size() != 4)
            Error("%d values supplied for \"pixelbounds\". Expected 4.",
                  int(pb.size()));
        else {
            Bounds2i newBounds = Bounds2i({pb[0], pb[2]}, {pb[1], pb[3]});
            if (Intersect(newBounds, pixelBounds) != newBounds)
                Warning("Supplied pixel bounds extend beyond image resolution. Clamping.");
            pixelBounds = Intersect(newBounds, pixelBounds);
        }
    }

    std::vector<Float> cr = dict.GetFloatArray("cropwindow");
    if (PbrtOptions.cropWindow) {
        Bounds2f crop = *PbrtOptions.cropWindow;
        // Compute film image bounds
        pixelBounds =
            Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                             std::ceil(fullResolution.y * crop.pMin.y)),
                     Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                             std::ceil(fullResolution.y * crop.pMax.y)));

        if (!cr.empty())
            Warning("Crop window supplied on command line will override "
                    "crop window specified with Film.");
        if (!pb.empty())
            Warning("Both pixel bounds and crop window were specified. Using the "
                    "crop window.");
    } else if (!cr.empty()) {
        if (cr.size() == 4) {
            if (!pb.empty())
                Warning("Both pixel bounds and crop window were specified. Using the "
                        "crop window.");

            Bounds2f crop;
            crop.pMin.x = Clamp(std::min(cr[0], cr[1]), 0.f, 1.f);
            crop.pMax.x = Clamp(std::max(cr[0], cr[1]), 0.f, 1.f);
            crop.pMin.y = Clamp(std::min(cr[2], cr[3]), 0.f, 1.f);
            crop.pMax.y = Clamp(std::max(cr[2], cr[3]), 0.f, 1.f);

            // Compute film image bounds
            pixelBounds =
                Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                 std::ceil(fullResolution.y * crop.pMin.y)),
                         Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                 std::ceil(fullResolution.y * crop.pMax.y)));
        }
        else
            Error("%d values supplied for \"cropwindow\". Expected 4.",
                  (int)cr.size());
    }

    if (pixelBounds.IsEmpty())
        ErrorExit("Degenerate pixel bounds provided to film: %s.", pixelBounds);

    Float scale = dict.GetOneFloat("scale", 1.);
    Float diagonal = dict.GetOneFloat("diagonal", 35.);
    Float maxSampleLuminance = dict.GetOneFloat("maxsampleluminance",
                                                Infinity);
    bool writeFP16 = dict.GetOneBool("savefp16", true);
    bool saveVariance = dict.GetOneBool("savevariance", false);
    return alloc.new_object<RGBFilm>(fullResolution, pixelBounds, std::move(filter),
                                     diagonal, filename, scale, colorSpace,
                                     maxSampleLuminance, writeFP16, saveVariance, alloc);
}

// Film Method Definitions
AOVFilm::AOVFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
                 pstd::unique_ptr<Filter> filt, Float diagonal,
                 const std::string &filename,
                 const RGBColorSpace *colorSpace,
                 Float maxSampleLuminance, bool writeFP16)
    : Film(resolution, pixelBounds, std::move(filt), diagonal, filename),
      pixels(pixelBounds),
      colorSpace(colorSpace),
      maxSampleLuminance(maxSampleLuminance),
      writeFP16(writeFP16) {
    // Allocate film image storage
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
}

SampledWavelengths AOVFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleImportance(u);
}

void AOVFilm::AddSample(const Point2i &pFilm, SampledSpectrum L,
                        const SampledWavelengths &lambda,
                        const pstd::optional<VisibleSurface> &visibleSurface, Float weight) {
    ProfilerScope _(ProfilePhase::AddFilmSample);

    SampledSpectrum Le(0.), Ld(0.), Li(0.);
    if (visibleSurface) {
        Le = visibleSurface->Le;
        Ld = visibleSurface->Ld;
        Li = L - Ld - Le;
        // Clamp Li and Ld individually. (Allow separate thresholds?)
        if (Ld.y(lambda, colorSpace->Y) > maxSampleLuminance)
            Ld *= maxSampleLuminance / Ld.y(lambda, colorSpace->Y);
        if (Li.y(lambda, colorSpace->Y) > maxSampleLuminance)
            Li *= maxSampleLuminance / Li.y(lambda, colorSpace->Y);
        L = Le + Ld + Li;
    } else {
        if (L.y(lambda, colorSpace->Y) > maxSampleLuminance)
            L *= maxSampleLuminance / L.y(lambda, colorSpace->Y);
        Li = L;
    }

    Le *= weight;
    Ld *= weight;
    Li *= weight;

    CHECK(InsideExclusive(pFilm, pixelBounds));

    auto addRGB = [this, &lambda](const SampledSpectrum &s,
                                  double rgbOut[3]) {
                      RGB rgb = s.ToRGB(lambda, *colorSpace);
                      for (int c = 0; c < 3; ++c) rgbOut[c] += rgb[c];
                  };

    Pixel &p = pixels[pFilm];
    if (visibleSurface) {
        // Update variance estimates.
        p.LdVarianceEstimator.Add(Ld.y(lambda, colorSpace->Y));
        p.LiVarianceEstimator.Add(Li.y(lambda, colorSpace->Y));

        p.pSum += weight * visibleSurface->p;

        p.nSum += weight * visibleSurface->n;
        p.nsSum += weight * visibleSurface->ns;

        p.dzdxSum += weight * visibleSurface->dzdx;
        p.dzdySum += weight * visibleSurface->dzdy;

        addRGB(Le, p.LeSum);
        addRGB(Ld, p.LdSum);

#ifndef __CUDA_ARCH__
        // FIXME
        if (visibleSurface->bsdf) {
            constexpr int nRhoSamples = 16;
            Point2f rhoSamples[nRhoSamples];
            Float rhoCSamples[nRhoSamples];
            CranleyPattersonRotator rot[3] = { CranleyPattersonRotator(BlueNoise(0, pFilm.x, pFilm.y)),
                                               CranleyPattersonRotator(BlueNoise(1, pFilm.x, pFilm.y)),
                                               CranleyPattersonRotator(BlueNoise(2, pFilm.x, pFilm.y)) };
            for (int i = 0; i < nRhoSamples; ++i) {
                // FIXME: rho() always discards the first sample since rhoSamples[0][0].x == 0,
                // since that's how Sobol' rolls and then the blue noise texture has zero for
                // the first entry....
                rhoCSamples[i] = SobolSampleFloat(i, 0, rot[0]);
                rhoSamples[i] = Point2f(SobolSampleFloat(i, 1, rot[1]),
                                        SobolSampleFloat(i, 2, rot[2]));
            }

            SampledSpectrum albedo = visibleSurface->bsdf->rho(visibleSurface->woWorld,
                                                         rhoCSamples, rhoSamples) *
                colorSpace->illuminant.Sample(lambda);
            albedo *= weight;
            addRGB(albedo, p.albedoSum);
        }
#endif

#if 0
        if (visibleSurface->materialAttributes) {
            std::string name = visibleSurface->materialAttributes->GetOneString("name", "");
            if (!name.empty()) {
                auto iter = staticMaterialIdMap.find(name);
                if (iter != staticMaterialIdMap.end())
                    p.materialRGB = iter->second;
                else {
                    std::lock_guard<std::mutex> lock(materialIdMapLock);
                    iter = dynamicMaterialIdMap.find(name);
                    if (iter != dynamicMaterialIdMap.end())
                        p.materialRGB = iter->second;
                    else {
                        // reserve 0 for unknown
                        int n = dynamicMaterialIdMap.size() + 1;
                        RGB rgb(RadicalInverse(0, n), RadicalInverse(1, n),
                                RadicalInverse(2, n));
                        dynamicMaterialIdMap[name] = rgb;
                        p.materialRGB = rgb;
                    }
                }
            }
        }
#endif // !__CUDA_ARCH__
    }

    addRGB(L, p.LSum);
    p.weightSum += weight;
}

void AOVFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                       const SampledWavelengths &lambda) {
    ProfilerScope pp(ProfilePhase::SplatFilm);

    // NOTE: same code as RGBFilm::AddSplat()...

    XYZ xyz = v.ToXYZ(lambda, colorSpace->X, colorSpace->Y, colorSpace->Z);
    Float y = xyz.Y;
    CHECK(!v.HasNaNs());
    CHECK(!std::isinf(y));

    if (y > maxSampleLuminance) {
        v *= maxSampleLuminance / y;
        xyz = v.ToXYZ(lambda, colorSpace->X, colorSpace->Y, colorSpace->Z);
    }
    RGB rgb = colorSpace->ToRGB(xyz);

    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - filter->radius)),
                         Point2i(Floor(pDiscrete + filter->radius)) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);
    for (Point2i pi : splatBounds) {
        Float wt = filter->Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];
            for (int i = 0; i < 3; ++i) pixel.splatRGB[i].Add(wt * rgb[i]);
        }
    }
}

void AOVFilm::WriteImage(ImageMetadata metadata, Float splatScale) {
    Image image = GetImage(&metadata, splatScale);
    LOG_VERBOSE("Writing image %s with bounds %s", filename, pixelBounds);
    image.Write(filename, metadata);
    staticMaterialIdMap = dynamicMaterialIdMap;
}

Image AOVFilm::GetImage(ImageMetadata *metadata, Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG_VERBOSE("Converting image to RGB and computing final weighted pixel values");
    PixelFormat format = writeFP16 ? PixelFormat::Half : PixelFormat::Float;
    Image image(format, Point2i(pixelBounds.Diagonal()),
                { "R", "G", "B", "Le.R", "Le.G", "Le.B", "Ld.R", "Ld.G", "Ld.B",
                  "Li.R", "Li.G", "Li.B", "Albedo.R", "Albedo.G", "Albedo.B",
                  "Px", "Py", "Pz", "dzdx", "dzdy", "Nx", "Ny", "Nz", "Nsx", "Nsy", "Nsz",
                  "materialId.R", "materialId.G", "materialId.B",
                  "LdVariance", "LdRelativeVariance", "LiVariance", "LiRelativeVariance" });

    ImageChannelDesc rgbDesc = *image.GetChannelDesc({ "R", "G", "B" });
    ImageChannelDesc LeDesc = *image.GetChannelDesc({ "Le.R", "Le.G", "Le.B" });
    ImageChannelDesc LdDesc = *image.GetChannelDesc({ "Ld.R", "Ld.G", "Ld.B" });
    ImageChannelDesc LiDesc = *image.GetChannelDesc({ "Li.R", "Li.G", "Li.B" });
    ImageChannelDesc pDesc = *image.GetChannelDesc({ "Px", "Py", "Pz" });
    ImageChannelDesc dzDesc = *image.GetChannelDesc({ "dzdx", "dzdy" });
    ImageChannelDesc nDesc = *image.GetChannelDesc({ "Nx", "Ny", "Nz"});
    ImageChannelDesc nsDesc = *image.GetChannelDesc({ "Nsx", "Nsy", "Nsz"});
    ImageChannelDesc matIdDesc = *image.GetChannelDesc({ "materialId.R", "materialId.G", "materialId.B" });
    ImageChannelDesc albedoRgbDesc = *image.GetChannelDesc({ "Albedo.R", "Albedo.G", "Albedo.B"});
    ImageChannelDesc ldVarianceDesc = *image.GetChannelDesc({ "LdVariance", "LdRelativeVariance" });
    ImageChannelDesc liVarianceDesc = *image.GetChannelDesc({ "LiVariance", "LiRelativeVariance" });

    Float filterIntegral = filter->Integral();

    ParallelFor2D(pixelBounds, [&](Point2i p) {
        Pixel &pixel = pixels[p];
        RGB Lrgb(pixel.LSum[0], pixel.LSum[1], pixel.LSum[2]);
        RGB Lergb(pixel.LeSum[0], pixel.LeSum[1], pixel.LeSum[2]);
        RGB Ldrgb(pixel.LdSum[0], pixel.LdSum[1], pixel.LdSum[2]);
        RGB albedoRgb(pixel.albedoSum[0], pixel.albedoSum[1], pixel.albedoSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum;
        Point3f pt = pixel.pSum;
        Float dzdx = pixel.dzdxSum, dzdy = pixel.dzdySum;
        if (weightSum != 0) {
            Lrgb /= weightSum;
            Lergb /= weightSum;
            Ldrgb /= weightSum;
            albedoRgb /= weightSum;
            pt /= weightSum;
            dzdx /= weightSum;
            dzdy /= weightSum;
        }
        RGB Lirgb = Lrgb - Lergb - Ldrgb;

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            Lrgb[c] += splatScale * pixel.splatRGB[c] / filterIntegral;

        Point2i pOffset(p.x - pixelBounds.pMin.x,
                        p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, rgbDesc, {Lrgb[0], Lrgb[1], Lrgb[2]});
        image.SetChannels(pOffset, LeDesc, {Lergb[0], Lergb[1], Lergb[2]});
        image.SetChannels(pOffset, LdDesc, {Ldrgb[0], Ldrgb[1], Ldrgb[2]});
        image.SetChannels(pOffset, LiDesc, {Lirgb[0], Lirgb[1], Lirgb[2]});
        image.SetChannels(pOffset, albedoRgbDesc, {albedoRgb[0], albedoRgb[1], albedoRgb[2]});

        Normal3f n = LengthSquared(pixel.nSum) > 0 ? Normalize(pixel.nSum) : Normal3f(0,0,0);
        Normal3f ns = LengthSquared(pixel.nsSum) > 0 ? Normalize(pixel.nsSum) : Normal3f(0,0,0);
        image.SetChannels(pOffset, pDesc, { pt.x, pt.y, pt.z });
        image.SetChannels(pOffset, dzDesc, { std::abs(dzdx), std::abs(dzdy) });
        image.SetChannels(pOffset, nDesc, { n.x, n.y, n.z });
        image.SetChannels(pOffset, nsDesc, { ns.x, ns.y, ns.z });
        image.SetChannels(pOffset, matIdDesc, { pixel.materialRGB.r, pixel.materialRGB.g,
                                                pixel.materialRGB.b });

        image.SetChannels(pOffset, ldVarianceDesc, { pixel.LdVarianceEstimator.Variance(),
                                                     pixel.LdVarianceEstimator.RelativeVariance() });
        image.SetChannels(pOffset, liVarianceDesc, { pixel.LiVarianceEstimator.Variance(),
                                                     pixel.LiVarianceEstimator.RelativeVariance() });
    });

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;

    Float varianceSum = 0;
    for (Point2i p : pixelBounds) {
        const Pixel &pixel = pixels[p];
        varianceSum += Float(pixel.LdVarianceEstimator.Variance() +
                             pixel.LiVarianceEstimator.Variance());
    }
    metadata->estimatedVariance = varianceSum / pixelBounds.Area();

    std::vector<std::string> materialRGBs{"unknown 0 0 0"};
    for (auto material : dynamicMaterialIdMap)
        materialRGBs.push_back(StringPrintf("%s %f %f %f", material.first,
                                            material.second.r, material.second.g,
                                            material.second.b));
    metadata->stringVectors["materials"] = materialRGBs;

    return image;
}

std::string AOVFilm::ToString() const {
    return StringPrintf("[ AOVFilm %s colorSpace: %s maxSampleLuminance: %f "
                        "writeFP16: %s ]", BaseToString(),
                        *colorSpace, maxSampleLuminance, writeFP16);
}

std::unique_ptr<Film> AOVFilm::Create(const ParameterDictionary &dict,
                                      pstd::unique_ptr<Filter> filter,
                                      const RGBColorSpace *colorSpace) {
    std::string filename = dict.GetOneString("filename", "");
    if (PbrtOptions.imageFile) {
        if (!filename.empty())
            Warning(
                "Output filename supplied on command line, \"%s\" will override "
                "filename provided in scene description file, \"%s\".",
                *PbrtOptions.imageFile, filename);
        filename = *PbrtOptions.imageFile;
    } else if (filename.empty())
        filename = "pbrt.exr";

    Point2i fullResolution(dict.GetOneInt("xresolution", 1280),
                           dict.GetOneInt("yresolution", 720));
    if (PbrtOptions.quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    Bounds2i pixelBounds(Point2i(0, 0), fullResolution);
    std::vector<int> pb = dict.GetIntArray("pixelbounds");
    if (PbrtOptions.pixelBounds) {
        Bounds2i newBounds = *PbrtOptions.pixelBounds;
        if (Intersect(newBounds, pixelBounds) != newBounds)
            Warning("Supplied pixel bounds extend beyond image resolution. Clamping.");
        pixelBounds = Intersect(newBounds, pixelBounds);

        if (!pb.empty())
            Warning("Both pixel bounds and crop window were specified. Using the "
                    "crop window.");
    }
    else if (!pb.empty()) {
        if (pb.size() != 4)
            Error("%d values supplied for \"pixelbounds\". Expected 4.",
                  int(pb.size()));
        else {
            Bounds2i newBounds = Bounds2i({pb[0], pb[2]}, {pb[1], pb[3]});
            if (Intersect(newBounds, pixelBounds) != newBounds)
                Warning("Supplied pixel bounds extend beyond image resolution. Clamping.");
            pixelBounds = Intersect(newBounds, pixelBounds);
        }
    }

    std::vector<Float> cr = dict.GetFloatArray("cropwindow");
    if (PbrtOptions.cropWindow) {
        Bounds2f crop = *PbrtOptions.cropWindow;
        // Compute film image bounds
        pixelBounds =
            Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                             std::ceil(fullResolution.y * crop.pMin.y)),
                     Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                             std::ceil(fullResolution.y * crop.pMax.y)));

        if (!cr.empty())
            Warning("Crop window supplied on command line will override "
                    "crop window specified with Film.");
        if (!pb.empty())
            Warning("Both pixel bounds and crop window were specified. Using the "
                    "crop window.");
    } else if (!cr.empty()) {
        if (cr.size() == 4) {
            if (!pb.empty())
            Warning("Both pixel bounds and crop window were specified. Using the "
                    "crop window.");

            Bounds2f crop;
            crop.pMin.x = Clamp(std::min(cr[0], cr[1]), 0.f, 1.f);
            crop.pMax.x = Clamp(std::max(cr[0], cr[1]), 0.f, 1.f);
            crop.pMin.y = Clamp(std::min(cr[2], cr[3]), 0.f, 1.f);
            crop.pMax.y = Clamp(std::max(cr[2], cr[3]), 0.f, 1.f);

            // Compute film image bounds
            pixelBounds =
                Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                 std::ceil(fullResolution.y * crop.pMin.y)),
                         Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                 std::ceil(fullResolution.y * crop.pMax.y)));
        }
        else
            Error("%d values supplied for \"cropwindow\". Expected 4.",
                  (int)cr.size());
    }

    if (pixelBounds.IsEmpty())
        ErrorExit("Degenerate pixel bounds provided to film: %s.", pixelBounds);

    Float diagonal = dict.GetOneFloat("diagonal", 35.);
    Float maxSampleLuminance = dict.GetOneFloat("maxsampleluminance",
                                                  Infinity);
    bool writeFP16 = dict.GetOneBool("savefp16", true);
    return std::make_unique<AOVFilm>(fullResolution, pixelBounds, std::move(filter),
                                     diagonal, filename, colorSpace,
                                     maxSampleLuminance, writeFP16);
}

}  // namespace pbrt
