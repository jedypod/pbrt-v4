// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// film.cpp*
#include <pbrt/film.h>

#include <pbrt/bsdf.h>
#include <pbrt/cameras.h>
#include <pbrt/filters.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/error.h>
#include <pbrt/util/image.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/transform.h>

namespace pbrt {

void FilmHandle::AddSplat(const Point2f &p, SampledSpectrum v,
                          const SampledWavelengths &lambda) {
    auto splat = [&](auto ptr) { return ptr->AddSplat(p, v, lambda); };
    return Apply<void>(splat);
}

void FilmHandle::WriteImage(ImageMetadata metadata, Float splatScale) {
    auto write = [&](auto ptr) { return ptr->WriteImage(metadata, splatScale); };
    return ApplyCPU<void>(write);
}

Image FilmHandle::GetImage(ImageMetadata *metadata, Float splatScale) {
    auto get = [&](auto ptr) { return ptr->GetImage(metadata, splatScale); };
    return ApplyCPU<Image>(get);
}

std::string FilmHandle::ToString() const {
    if (ptr() == nullptr)
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return ApplyCPU<std::string>(ts);
}

std::string FilmHandle::GetFilename() const {
    auto get = [&](auto ptr) { return ptr->GetFilename(); };
    return ApplyCPU<std::string>(get);
}

FilmBase::FilmBase(const Point2i &resolution, const Bounds2i &pixelBounds,
                   FilterHandle filter, Float diagonal, const std::string &filename)
    : fullResolution(resolution),
      diagonal(diagonal * .001),
      filter(filter),
      filename(filename),
      pixelBounds(pixelBounds) {
    CHECK(!pixelBounds.IsEmpty());
    CHECK_GE(pixelBounds.pMin.x, 0);
    CHECK_LE(pixelBounds.pMax.x, resolution.x);
    CHECK_GE(pixelBounds.pMin.y, 0);
    CHECK_LE(pixelBounds.pMax.y, resolution.y);
    LOG_VERBOSE("Created film with full resolution %s, pixelBounds %s", resolution,
                pixelBounds);
}

Bounds2f FilmBase::SampleBounds() const {
    return Bounds2f(Point2f(pixelBounds.pMin) - filter.Radius() + Vector2f(0.5f, 0.5f),
                    Point2f(pixelBounds.pMax) + filter.Radius() - Vector2f(0.5f, 0.5f));
}

std::string FilmBase::BaseToString() const {
    return StringPrintf("fullResolution: %s diagonal: %f filter: %s filename: %s "
                        "pixelBounds: %s",
                        fullResolution, diagonal, filter, filename, pixelBounds);
}

VisibleSurface::VisibleSurface(const SurfaceInteraction &si,
                               const CameraTransform &cameraTransform,
                               const SampledWavelengths &lambda) {
    Transform cameraFromRender = cameraTransform.CameraFromRender(si.time);

    time = si.time;
    p = cameraFromRender(si.p());
    n = cameraFromRender(si.n);
    Vector3f wo = cameraFromRender(si.wo);
    n = FaceForward(n, wo);
    ns = cameraFromRender(si.shading.n);
    ns = FaceForward(ns, wo);
    dzdx = cameraFromRender(si.dpdx).z;
    dzdy = cameraFromRender(si.dpdy).z;

    if (si.bsdf) {
        constexpr int nRhoSamples = 16;
        SampledSpectrum rho(0.f);
        for (int i = 0; i < nRhoSamples; ++i) {
            // Start at 2nd sample since (0,0) is generally not a useful
            // one...
            Float uc = RadicalInverse(0, i + 1);
            Point2f u(RadicalInverse(1, i + 1), RadicalInverse(2, i + 1));

            // Estimate one term of $\rho_\roman{hd}$
            auto bs = si.bsdf->Sample_f(si.wo, uc, u);
            if (bs && bs->pdf > 0)
                // Use si.shading.n rather than use ns, since that's now in
                // the wrong coordinate space!
                rho += bs->f * AbsDot(bs->wi, si.shading.n) / bs->pdf;
        }
        albedo = rho / nRhoSamples;
    }
}

std::string VisibleSurface::ToString() const {
    return StringPrintf("[ VisibleSurface p: %s n: %s ns: %s dzdx: %f dzdy: %f "
                        "time: %f albedo: %s dpdx: %s dpdy: %s ]",
                        p, n, ns, dzdx, dzdy, time, albedo, dpdx, dpdy);
}

STAT_MEMORY_COUNTER("Memory/Film pixels", filmPixelMemory);

// Film Method Definitions
RGBFilm::RGBFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
                 FilterHandle filter, Float diagonal, const std::string &filename,
                 Float scale, const RGBColorSpace *colorSpace, Float maxSampleLuminance,
                 bool writeFP16, bool saveVariance, Allocator allocator)
    : FilmBase(resolution, pixelBounds, filter, diagonal, filename),
      pixels(pixelBounds, allocator),
      scale(scale),
      colorSpace(colorSpace),
      maxSampleLuminance(maxSampleLuminance),
      writeFP16(writeFP16),
      saveVariance(saveVariance) {
    filterIntegral = filter.Integral();

    CHECK(!pixelBounds.IsEmpty());
    CHECK(colorSpace != nullptr);
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
}

SampledWavelengths RGBFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleImportance(u);
}

void RGBFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                       const SampledWavelengths &lambda) {
    XYZ xyz = v.ToXYZ(lambda);
    Float y = xyz.Y;
    CHECK(!v.HasNaNs());
    CHECK(!std::isinf(y));

    if (y > maxSampleLuminance) {
        v *= maxSampleLuminance / y;
        xyz = v.ToXYZ(lambda);
    }
    RGB rgb = colorSpace->ToRGB(xyz);

    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - filter.Radius())),
                         Point2i(Floor(pDiscrete + filter.Radius())) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);
    for (Point2i pi : splatBounds) {
        Float wt = filter.Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];
            for (int i = 0; i < 3; ++i)
                pixel.splatRGB[i].Add(wt * rgb[i]);
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
    std::vector<std::string> channels = {"R", "G", "B"};
    if (saveVariance)
        channels.push_back("Variance");
    Image image(format, Point2i(pixelBounds.Diagonal()), channels);

    ParallelFor2D(pixelBounds, [&](Point2i p) {
        RGB rgb = GetPixelRGB(p, splatScale);

        Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, {rgb[0], rgb[1], rgb[2]});
        if (saveVariance)
            image.SetChannel(pOffset, 3, pixels[p].varianceEstimator.Variance());
    });

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;

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
                        "writeFP16: %s saveVariance: %s ]",
                        BaseToString(), scale, *colorSpace, maxSampleLuminance, writeFP16,
                        saveVariance);
}

RGBFilm *RGBFilm::Create(const ParameterDictionary &dict, FilterHandle filter,
                         const RGBColorSpace *colorSpace, const FileLoc *loc,
                         Allocator alloc) {
    std::string filename = dict.GetOneString("filename", "");
    if (Options->imageFile) {
        if (!filename.empty())
            Warning(loc,
                    "Output filename supplied on command line, \"%s\" will "
                    "override "
                    "filename provided in scene description file, \"%s\".",
                    *Options->imageFile, filename);
        filename = *Options->imageFile;
    } else if (filename.empty())
        filename = "pbrt.exr";

    Point2i fullResolution(dict.GetOneInt("xresolution", 1280),
                           dict.GetOneInt("yresolution", 720));
    if (Options->quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    Bounds2i pixelBounds(Point2i(0, 0), fullResolution);
    std::vector<int> pb = dict.GetIntArray("pixelbounds");
    if (Options->pixelBounds) {
        Bounds2i newBounds = *Options->pixelBounds;
        if (Intersect(newBounds, pixelBounds) != newBounds)
            Warning(loc, "Supplied pixel bounds extend beyond image "
                         "resolution. Clamping.");
        pixelBounds = Intersect(newBounds, pixelBounds);

        if (!pb.empty())
            Warning(loc, "Both pixel bounds and crop window were specified. Using the "
                         "crop window.");
    } else if (!pb.empty()) {
        if (pb.size() != 4)
            Error(loc, "%d values supplied for \"pixelbounds\". Expected 4.",
                  int(pb.size()));
        else {
            Bounds2i newBounds = Bounds2i({pb[0], pb[2]}, {pb[1], pb[3]});
            if (Intersect(newBounds, pixelBounds) != newBounds)
                Warning(loc, "Supplied pixel bounds extend beyond image "
                             "resolution. Clamping.");
            pixelBounds = Intersect(newBounds, pixelBounds);
        }
    }

    std::vector<Float> cr = dict.GetFloatArray("cropwindow");
    if (Options->cropWindow) {
        Bounds2f crop = *Options->cropWindow;
        // Compute film image bounds
        pixelBounds = Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                       std::ceil(fullResolution.y * crop.pMin.y)),
                               Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                       std::ceil(fullResolution.y * crop.pMax.y)));

        if (!cr.empty())
            Warning(loc, "Crop window supplied on command line will override "
                         "crop window specified with Film.");
        if (Options->pixelBounds || !pb.empty())
            Warning(loc, "Both pixel bounds and crop window were specified. Using the "
                         "crop window.");
    } else if (!cr.empty()) {
        if (Options->pixelBounds)
            Warning(loc, "Ignoring \"cropwindow\" since pixel bounds were specified "
                         "on the command line.");
        else if (cr.size() == 4) {
            if (!pb.empty())
                Warning(loc, "Both pixel bounds and crop window were "
                             "specified. Using the "
                             "crop window.");

            Bounds2f crop;
            crop.pMin.x = Clamp(std::min(cr[0], cr[1]), 0.f, 1.f);
            crop.pMax.x = Clamp(std::max(cr[0], cr[1]), 0.f, 1.f);
            crop.pMin.y = Clamp(std::min(cr[2], cr[3]), 0.f, 1.f);
            crop.pMax.y = Clamp(std::max(cr[2], cr[3]), 0.f, 1.f);

            // Compute film image bounds
            pixelBounds = Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                           std::ceil(fullResolution.y * crop.pMin.y)),
                                   Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                           std::ceil(fullResolution.y * crop.pMax.y)));
        } else
            Error(loc, "%d values supplied for \"cropwindow\". Expected 4.",
                  (int)cr.size());
    }

    if (pixelBounds.IsEmpty())
        ErrorExit(loc, "Degenerate pixel bounds provided to film: %s.", pixelBounds);

    Float scale = dict.GetOneFloat("scale", 1.);
    Float diagonal = dict.GetOneFloat("diagonal", 35.);
    Float maxSampleLuminance = dict.GetOneFloat("maxsampleluminance", Infinity);
    bool writeFP16 = dict.GetOneBool("savefp16", true);
    bool saveVariance = dict.GetOneBool("savevariance", false);
    return alloc.new_object<RGBFilm>(fullResolution, pixelBounds, filter, diagonal,
                                     filename, scale, colorSpace, maxSampleLuminance,
                                     writeFP16, saveVariance, alloc);
}

// Film Method Definitions
GBufferFilm::GBufferFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
                         FilterHandle filter, Float diagonal, const std::string &filename,
                         Float scale, const RGBColorSpace *colorSpace,
                         Float maxSampleLuminance, bool writeFP16, Allocator alloc)
    : FilmBase(resolution, pixelBounds, filter, diagonal, filename),
      pixels(pixelBounds, alloc),
      scale(scale),
      colorSpace(colorSpace),
      maxSampleLuminance(maxSampleLuminance),
      writeFP16(writeFP16) {
    // Allocate film image storage
    CHECK(!pixelBounds.IsEmpty());
    filmPixelMemory += pixelBounds.Area() * sizeof(Pixel);
}

SampledWavelengths GBufferFilm::SampleWavelengths(Float u) const {
    return SampledWavelengths::SampleImportance(u);
}

void GBufferFilm::AddSample(const Point2i &pFilm, SampledSpectrum L,
                            const SampledWavelengths &lambda,
                            const pstd::optional<VisibleSurface> &visibleSurface,
                            Float weight) {
    L *= scale;

    Float clampScale =
        (L.y(lambda) > maxSampleLuminance) ? maxSampleLuminance / L.y(lambda) : 1;
    L *= clampScale;
    L *= weight;

    DCHECK(InsideExclusive(pFilm, pixelBounds));

    auto addRGB = [this, &lambda](const SampledSpectrum &s, double rgbOut[3]) {
        RGB rgb = s.ToRGB(lambda, *colorSpace);
        for (int c = 0; c < 3; ++c)
            rgbOut[c] += rgb[c];
    };

    Pixel &p = pixels[pFilm];
    if (visibleSurface) {
        // Update variance estimates.
        p.LVarianceEstimator.Add(L.y(lambda));

        p.pSum += weight * visibleSurface->p;

        p.nSum += weight * visibleSurface->n;
        p.nsSum += weight * visibleSurface->ns;

        p.dzdxSum += weight * visibleSurface->dzdx;
        p.dzdySum += weight * visibleSurface->dzdy;

        SampledSpectrum albedo =
            visibleSurface->albedo * colorSpace->illuminant.Sample(lambda) * weight;
        addRGB(albedo, p.albedoSum);
    }

    addRGB(L, p.rgbSum);
    p.weightSum += weight;
}

void GBufferFilm::AddSplat(const Point2f &p, SampledSpectrum v,
                           const SampledWavelengths &lambda) {
    // NOTE: same code as RGBFilm::AddSplat()...

    XYZ xyz = v.ToXYZ(lambda);
    Float y = xyz.Y;
    CHECK(!v.HasNaNs());
    CHECK(!std::isinf(y));

    if (y > maxSampleLuminance) {
        v *= maxSampleLuminance / y;
        xyz = v.ToXYZ(lambda);
    }
    RGB rgb = colorSpace->ToRGB(xyz);

    Point2f pDiscrete = p + Vector2f(0.5, 0.5);
    Bounds2i splatBounds(Point2i(Floor(pDiscrete - filter.Radius())),
                         Point2i(Floor(pDiscrete + filter.Radius())) + Vector2i(1, 1));
    splatBounds = Intersect(splatBounds, pixelBounds);
    for (Point2i pi : splatBounds) {
        Float wt = filter.Evaluate(Point2f(p - pi - Vector2f(0.5, 0.5)));
        if (wt != 0) {
            Pixel &pixel = pixels[pi];
            for (int i = 0; i < 3; ++i)
                pixel.splatRGB[i].Add(wt * rgb[i]);
        }
    }
}

void GBufferFilm::WriteImage(ImageMetadata metadata, Float splatScale) {
    Image image = GetImage(&metadata, splatScale);
    LOG_VERBOSE("Writing image %s with bounds %s", filename, pixelBounds);
    image.Write(filename, metadata);
}

Image GBufferFilm::GetImage(ImageMetadata *metadata, Float splatScale) {
    // Convert image to RGB and compute final pixel values
    LOG_VERBOSE("Converting image to RGB and computing final weighted pixel values");
    PixelFormat format = writeFP16 ? PixelFormat::Half : PixelFormat::Float;
    Image image(format, Point2i(pixelBounds.Diagonal()),
                {"R",
                 "G",
                 "B",
                 "Albedo.R",
                 "Albedo.G",
                 "Albedo.B",
                 "Px",
                 "Py",
                 "Pz",
                 "dzdx",
                 "dzdy",
                 "Nx",
                 "Ny",
                 "Nz",
                 "Nsx",
                 "Nsy",
                 "Nsz",
                 "materialId.R",
                 "materialId.G",
                 "materialId.B",
                 "LVariance",
                 "LRelativeVariance"});

    ImageChannelDesc rgbDesc = *image.GetChannelDesc({"R", "G", "B"});
    ImageChannelDesc pDesc = *image.GetChannelDesc({"Px", "Py", "Pz"});
    ImageChannelDesc dzDesc = *image.GetChannelDesc({"dzdx", "dzdy"});
    ImageChannelDesc nDesc = *image.GetChannelDesc({"Nx", "Ny", "Nz"});
    ImageChannelDesc nsDesc = *image.GetChannelDesc({"Nsx", "Nsy", "Nsz"});
    ImageChannelDesc albedoRgbDesc =
        *image.GetChannelDesc({"Albedo.R", "Albedo.G", "Albedo.B"});
    ImageChannelDesc lVarianceDesc =
        *image.GetChannelDesc({"LVariance", "LRelativeVariance"});

    Float filterIntegral = filter.Integral();

    ParallelFor2D(pixelBounds, [&](Point2i p) {
        Pixel &pixel = pixels[p];
        RGB rgb(pixel.rgbSum[0], pixel.rgbSum[1], pixel.rgbSum[2]);
        RGB albedoRgb(pixel.albedoSum[0], pixel.albedoSum[1], pixel.albedoSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum;
        Point3f pt = pixel.pSum;
        Float dzdx = pixel.dzdxSum, dzdy = pixel.dzdySum;
        if (weightSum != 0) {
            rgb /= weightSum;
            albedoRgb /= weightSum;
            pt /= weightSum;
            dzdx /= weightSum;
            dzdy /= weightSum;
        }

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.splatRGB[c] / filterIntegral;

        Point2i pOffset(p.x - pixelBounds.pMin.x, p.y - pixelBounds.pMin.y);
        image.SetChannels(pOffset, rgbDesc, {rgb[0], rgb[1], rgb[2]});
        image.SetChannels(pOffset, albedoRgbDesc,
                          {albedoRgb[0], albedoRgb[1], albedoRgb[2]});

        Normal3f n =
            LengthSquared(pixel.nSum) > 0 ? Normalize(pixel.nSum) : Normal3f(0, 0, 0);
        Normal3f ns =
            LengthSquared(pixel.nsSum) > 0 ? Normalize(pixel.nsSum) : Normal3f(0, 0, 0);
        image.SetChannels(pOffset, pDesc, {pt.x, pt.y, pt.z});
        image.SetChannels(pOffset, dzDesc, {std::abs(dzdx), std::abs(dzdy)});
        image.SetChannels(pOffset, nDesc, {n.x, n.y, n.z});
        image.SetChannels(pOffset, nsDesc, {ns.x, ns.y, ns.z});
        image.SetChannels(pOffset, lVarianceDesc,
                          {pixel.LVarianceEstimator.Variance(),
                           pixel.LVarianceEstimator.RelativeVariance()});
    });

    metadata->pixelBounds = pixelBounds;
    metadata->fullResolution = fullResolution;
    metadata->colorSpace = colorSpace;

    Float varianceSum = 0;
    for (Point2i p : pixelBounds) {
        const Pixel &pixel = pixels[p];
        varianceSum += pixel.LVarianceEstimator.Variance();
    }
    metadata->estimatedVariance = varianceSum / pixelBounds.Area();

    return image;
}

std::string GBufferFilm::ToString() const {
    return StringPrintf("[ GBufferFilm %s colorSpace: %s maxSampleLuminance: %f "
                        "writeFP16: %s ]",
                        BaseToString(), *colorSpace, maxSampleLuminance, writeFP16);
}

GBufferFilm *GBufferFilm::Create(const ParameterDictionary &dict, FilterHandle filter,
                                 const RGBColorSpace *colorSpace, const FileLoc *loc,
                                 Allocator alloc) {
    std::string filename = dict.GetOneString("filename", "");
    if (Options->imageFile) {
        if (!filename.empty())
            Warning(loc,
                    "Output filename supplied on command line, \"%s\" will "
                    "override "
                    "filename provided in scene description file, \"%s\".",
                    *Options->imageFile, filename);
        filename = *Options->imageFile;
    } else if (filename.empty())
        filename = "pbrt.exr";

    Point2i fullResolution(dict.GetOneInt("xresolution", 1280),
                           dict.GetOneInt("yresolution", 720));
    if (Options->quickRender) {
        fullResolution.x = std::max(1, fullResolution.x / 4);
        fullResolution.y = std::max(1, fullResolution.y / 4);
    }

    Bounds2i pixelBounds(Point2i(0, 0), fullResolution);
    std::vector<int> pb = dict.GetIntArray("pixelbounds");
    if (Options->pixelBounds) {
        Bounds2i newBounds = *Options->pixelBounds;
        if (Intersect(newBounds, pixelBounds) != newBounds)
            Warning(loc, "Supplied pixel bounds extend beyond image "
                         "resolution. Clamping.");
        pixelBounds = Intersect(newBounds, pixelBounds);

        if (!pb.empty())
            Warning(loc, "Both pixel bounds and crop window were specified. Using the "
                         "crop window.");
    } else if (!pb.empty()) {
        if (pb.size() != 4)
            Error(loc, "%d values supplied for \"pixelbounds\". Expected 4.",
                  int(pb.size()));
        else {
            Bounds2i newBounds = Bounds2i({pb[0], pb[2]}, {pb[1], pb[3]});
            if (Intersect(newBounds, pixelBounds) != newBounds)
                Warning(loc, "Supplied pixel bounds extend beyond image "
                             "resolution. Clamping.");
            pixelBounds = Intersect(newBounds, pixelBounds);
        }
    }

    std::vector<Float> cr = dict.GetFloatArray("cropwindow");
    if (Options->cropWindow) {
        Bounds2f crop = *Options->cropWindow;
        // Compute film image bounds
        pixelBounds = Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                       std::ceil(fullResolution.y * crop.pMin.y)),
                               Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                       std::ceil(fullResolution.y * crop.pMax.y)));

        if (!cr.empty())
            Warning(loc, "Crop window supplied on command line will override "
                         "crop window specified with Film.");
        if (Options->pixelBounds || !pb.empty())
            Warning(loc, "Both pixel bounds and crop window were specified. Using the "
                         "crop window.");
    } else if (!cr.empty()) {
        if (Options->pixelBounds)
            Warning(loc, "Ignoring \"cropwindow\" since pixel bounds were specified "
                         "on the command line.");
        else if (cr.size() == 4) {
            if (!pb.empty())
                Warning(loc, "Both pixel bounds and crop window were "
                             "specified. Using the "
                             "crop window.");

            Bounds2f crop;
            crop.pMin.x = Clamp(std::min(cr[0], cr[1]), 0.f, 1.f);
            crop.pMax.x = Clamp(std::max(cr[0], cr[1]), 0.f, 1.f);
            crop.pMin.y = Clamp(std::min(cr[2], cr[3]), 0.f, 1.f);
            crop.pMax.y = Clamp(std::max(cr[2], cr[3]), 0.f, 1.f);

            // Compute film image bounds
            pixelBounds = Bounds2i(Point2i(std::ceil(fullResolution.x * crop.pMin.x),
                                           std::ceil(fullResolution.y * crop.pMin.y)),
                                   Point2i(std::ceil(fullResolution.x * crop.pMax.x),
                                           std::ceil(fullResolution.y * crop.pMax.y)));
        } else
            Error(loc, "%d values supplied for \"cropwindow\". Expected 4.",
                  (int)cr.size());
    }

    if (pixelBounds.IsEmpty())
        ErrorExit(loc, "Degenerate pixel bounds provided to film: %s.", pixelBounds);

    Float diagonal = dict.GetOneFloat("diagonal", 35.);
    Float maxSampleLuminance = dict.GetOneFloat("maxsampleluminance", Infinity);
    Float scale = dict.GetOneFloat("scale", 1.);
    bool writeFP16 = dict.GetOneBool("savefp16", true);
    return alloc.new_object<GBufferFilm>(fullResolution, pixelBounds, filter, diagonal,
                                         filename, scale, colorSpace, maxSampleLuminance,
                                         writeFP16, alloc);
}

FilmHandle FilmHandle::Create(const std::string &name, const ParameterDictionary &dict,
                              const FileLoc *loc, FilterHandle filter, Allocator alloc) {
    FilmHandle film;
    if (name == "rgb")
        film = RGBFilm::Create(dict, filter, dict.ColorSpace(), loc, alloc);
    else if (name == "gbuffer")
        film = GBufferFilm::Create(dict, filter, dict.ColorSpace(), loc, alloc);
    else
        ErrorExit(loc, "%s: film type unknown.", name);

    if (!film)
        ErrorExit(loc, "%s: unable to create film.", name);

    dict.ReportUnused();
    return film;
}

}  // namespace pbrt
