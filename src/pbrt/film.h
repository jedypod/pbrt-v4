// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_FILM_RGB_H
#define PBRT_FILM_RGB_H

// film.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/bsdf.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <atomic>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace pbrt {

// Film Declarations
// Note: AOVs only really make sense for PathIntegrator...
// volpath: medium interactions...
// bdpt, lightpath: Ld is partially computed via splats...
// simplepath, whitted: want to keep it simple
// sppm: ...
struct VisibleSurface {
    VisibleSurface() = default;
    PBRT_CPU_GPU
    VisibleSurface(const SurfaceInteraction &si, const CameraTransform &worldFromCamera,
                   const SampledWavelengths &lambda);

    std::string ToString() const;

    Point3f p;
    Float dzdx = 0, dzdy = 0;  // x/y: raster space, z: camera space
    Normal3f n, ns;
    Float time = 0;
    SampledSpectrum albedo;
    Vector3f dpdx, dpdy;  // world(ish) space
};

class FilmBase {
  public:
    PBRT_CPU_GPU
    Bounds2f SampleBounds() const;

    PBRT_CPU_GPU
    FilterHandle GetFilter() const { return filter; }

    PBRT_CPU_GPU
    Point2i FullResolution() const { return fullResolution; }

    PBRT_CPU_GPU
    Float Diagonal() const { return diagonal; }

    PBRT_CPU_GPU
    Bounds2i PixelBounds() const { return pixelBounds; }

    std::string GetFilename() const { return filename; }

  protected:
    Point2i fullResolution;
    Float diagonal;
    FilterHandle filter;
    std::string filename;
    Bounds2i pixelBounds;

    FilmBase(const Point2i &resolution, const Bounds2i &pixelBounds, FilterHandle filter,
             Float diagonal, const std::string &filename);

    std::string BaseToString() const;
};

class RGBFilm : public FilmBase {
  public:
    RGBFilm() = default;
    RGBFilm(const Point2i &resolution, const Bounds2i &pixelBounds, FilterHandle filter,
            Float diagonal, const std::string &filename, Float scale,
            const RGBColorSpace *colorSpace, Float maxSampleLuminance = Infinity,
            bool writeFP16 = true, bool saveVariance = false, Allocator allocator = {});

    static RGBFilm *Create(const ParameterDictionary &parameters, FilterHandle filter,
                           const RGBColorSpace *colorSpace, const FileLoc *loc,
                           Allocator alloc);

    PBRT_CPU_GPU
    SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_CPU_GPU
    void AddSample(const Point2i &pFilm, SampledSpectrum L,
                   const SampledWavelengths &lambda,
                   const pstd::optional<VisibleSurface> &visibleSurface, Float weight) {
        if (L.y(lambda) > maxSampleLuminance)
            L *= maxSampleLuminance / L.y(lambda);

        DCHECK(InsideExclusive(pFilm, pixelBounds));

        // Update variance estimate.
        pixels[pFilm].varianceEstimator.Add(L.Average());

        // Update pixel values with filtered sample contribution
        Pixel &pixel = pixels[pFilm];
        RGB rgb = L.ToRGB(lambda, *colorSpace);
        for (int c = 0; c < 3; ++c)
            pixel.rgbSum[c] += weight * rgb[c];
        pixel.weightSum += weight;
    }

    PBRT_CPU_GPU
    void AddSplat(const Point2f &p, SampledSpectrum v, const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    bool UsesVisibleSurface() const { return false; }

    PBRT_CPU_GPU
    RGB GetPixelRGB(const Point2i &p, Float splatScale = 1) const {
        const Pixel &pixel = pixels[p];
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

        return rgb;
    }

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;

  private:
    // RGBFilm Private Data
    struct Pixel {
        Pixel() = default;
        double rgbSum[3] = {0., 0., 0.};
        double weightSum = 0.;
        AtomicDouble splatRGB[3];
        VarianceEstimator<Float> varianceEstimator;
    };
    Array2D<Pixel> pixels;
    Float scale;
    const RGBColorSpace *colorSpace;
    Float maxSampleLuminance;
    bool writeFP16, saveVariance;
    Float filterIntegral;
};

class GBufferFilm : public FilmBase {
  public:
    GBufferFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
                FilterHandle filter, Float diagonal, const std::string &filename,
                Float scale, const RGBColorSpace *colorSpace,
                Float maxSampleLuminance = Infinity, bool writeFP16 = true,
                Allocator alloc = {});

    static GBufferFilm *Create(const ParameterDictionary &parameters, FilterHandle filter,
                               const RGBColorSpace *colorSpace, const FileLoc *loc,
                               Allocator alloc);

    PBRT_CPU_GPU
    SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_CPU_GPU
    void AddSample(const Point2i &pFilm, SampledSpectrum L,
                   const SampledWavelengths &lambda,
                   const pstd::optional<VisibleSurface> &visibleSurface, Float weight);

    PBRT_CPU_GPU
    void AddSplat(const Point2f &p, SampledSpectrum v, const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    bool UsesVisibleSurface() const { return true; }

    PBRT_CPU_GPU
    RGB GetPixelRGB(const Point2i &p, Float splatScale = 1) const {
        const Pixel &pixel = pixels[p];
        RGB rgb(pixel.LSum[0], pixel.LSum[1], pixel.LSum[2]);

        // Normalize pixel with weight sum
        Float weightSum = pixel.weightSum;
        if (weightSum != 0)
            rgb /= weightSum;

        // Add splat value at pixel
        for (int c = 0; c < 3; ++c)
            rgb[c] += splatScale * pixel.splatRGB[c] / filterIntegral;

        // Scale pixel value by _scale_
        rgb *= scale;

        return rgb;
    }

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;

  private:
    // GBufferFilm Private Data
    struct Pixel {
        Pixel() = default;
        double LSum[3] = {0., 0., 0.};
        double weightSum = 0.;
        AtomicDouble splatRGB[3];
        Point3f pSum;
        Float dzdxSum = 0, dzdySum = 0;
        Normal3f nSum, nsSum;
        double albedoSum[3] = {0., 0., 0.};
        VarianceEstimator<Float> LVarianceEstimator;
        RGB materialRGB;
    };
    Array2D<Pixel> pixels;
    Float scale;
    const RGBColorSpace *colorSpace;
    Float maxSampleLuminance;
    bool writeFP16;
    Float filterIntegral;
};

PBRT_CPU_GPU
inline SampledWavelengths FilmHandle::SampleWavelengths(Float u) const {
    auto sample = [&](auto ptr) { return ptr->SampleWavelengths(u); };
    return Apply<SampledWavelengths>(sample);
}

PBRT_CPU_GPU
inline Bounds2f FilmHandle::SampleBounds() const {
    auto sb = [&](auto ptr) { return ptr->SampleBounds(); };
    return Apply<Bounds2f>(sb);
}

PBRT_CPU_GPU
inline Bounds2i FilmHandle::PixelBounds() const {
    auto pb = [&](auto ptr) { return ptr->PixelBounds(); };
    return Apply<Bounds2i>(pb);
}

PBRT_CPU_GPU
inline Point2i FilmHandle::FullResolution() const {
    auto fr = [&](auto ptr) { return ptr->FullResolution(); };
    return Apply<Point2i>(fr);
}

PBRT_CPU_GPU
inline Float FilmHandle::Diagonal() const {
    auto diag = [&](auto ptr) { return ptr->Diagonal(); };
    return Apply<Float>(diag);
}

PBRT_CPU_GPU
inline FilterHandle FilmHandle::GetFilter() const {
    auto filter = [&](auto ptr) { return ptr->GetFilter(); };
    return Apply<FilterHandle>(filter);
}

PBRT_CPU_GPU
inline bool FilmHandle::UsesVisibleSurface() const {
    auto uses = [&](auto ptr) { return ptr->UsesVisibleSurface(); };
    return Apply<bool>(uses);
}

PBRT_CPU_GPU
inline RGB FilmHandle::GetPixelRGB(const Point2i &p, Float splatScale) const {
    auto get = [&](auto ptr) { return ptr->GetPixelRGB(p, splatScale); };
    return Apply<RGB>(get);
}

PBRT_CPU_GPU
inline void FilmHandle::AddSample(const Point2i &pFilm, SampledSpectrum L,
                                  const SampledWavelengths &lambda,
                                  const pstd::optional<VisibleSurface> &visibleSurface,
                                  Float weight) {
    auto add = [&](auto ptr) { return ptr->AddSample(pFilm, L, lambda, visibleSurface, weight); };
    return Apply<void>(add);
}

}  // namespace pbrt

#endif  // PBRT_FILM_AOV_H
