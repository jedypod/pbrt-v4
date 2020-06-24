
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_FILM_RGB_H
#define PBRT_FILM_RGB_H

// film.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/film.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <map>
#include <mutex>
#include <string>

namespace pbrt {

// Film Declarations
// Note: AOVs only really make sense for PathIntegrator...
// volpath: medium interactions...
// bdpt, lightpath: Ld is partially computed via splats...
// simplepath, whitted: want to keep it simple
// sppm: ...
struct VisibleSurface {
    VisibleSurface() = default;
    PBRT_HOST_DEVICE
    VisibleSurface(const SurfaceInteraction &si, const Camera &camera,
                   const SampledWavelengths &lambda);

    std::string ToString() const;

    Point3f p;
    Float dzdx = 0, dzdy = 0; // x/y: raster space, z: camera space
    Normal3f n, ns;
    Float time = 0;
    SampledSpectrum Le, Ld, albedo;
    BSDF *bsdf = nullptr;

private:
    Vector3f dpdx, dpdy; // world(ish) space
};

class RGBFilm final : public Film {
  public:
    RGBFilm() = default;
    RGBFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
            FilterHandle filter, Float diagonal,
            const std::string &filename, Float scale,
            const RGBColorSpace *colorSpace,
            Float maxSampleLuminance = Infinity,
            bool writeFP16 = true, bool saveVariance = false,
            Allocator allocator = {});

    static RGBFilm *Create(const ParameterDictionary &dict,
                           FilterHandle filter,
                           const RGBColorSpace *colorSpace,
                           const FileLoc *loc, Allocator alloc);

    PBRT_HOST_DEVICE
    SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_HOST_DEVICE_INLINE
    void AddSample(const Point2i &pFilm, SampledSpectrum L,
                   const SampledWavelengths &lambda,
                   const pstd::optional<VisibleSurface> &visibleSurface,
                   Float weight) {
        ProfilerScope _(ProfilePhase::AddFilmSample);
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

    PBRT_HOST_DEVICE
    void AddSplat(const Point2f &p, SampledSpectrum v,
                  const SampledWavelengths &lambda);
    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;

  private:
    // RGBFilm Private Data
    struct Pixel {
        Pixel() = default;
        double rgbSum[3] = { 0., 0., 0. };
        double weightSum = 0.;
        AtomicDouble splatRGB[3];
        VarianceEstimator<Float> varianceEstimator;
    };
    Array2D<Pixel> pixels;
    Float scale;
    const RGBColorSpace *colorSpace;
    Float maxSampleLuminance;
    bool writeFP16, saveVariance;
};

class AOVFilm final : public Film {
  public:
    AOVFilm(const Point2i &resolution, const Bounds2i &pixelBounds,
            FilterHandle filter, Float diagonal,
            const std::string &filename, const RGBColorSpace *colorSpace,
            Float maxSampleLuminance = Infinity,
            bool writeFP16 = true,
            Allocator alloc = {});

    static AOVFilm *Create(const ParameterDictionary &dict, FilterHandle filter,
                           const RGBColorSpace *colorSpace, const FileLoc *loc,
                           Allocator alloc);

    PBRT_HOST_DEVICE
    SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_HOST_DEVICE
    void AddSample(const Point2i &pFilm, SampledSpectrum L,
                   const SampledWavelengths &lambda,
                   const pstd::optional<VisibleSurface> &visibleSurface,
                   Float weight);

    PBRT_HOST_DEVICE
    void AddSplat(const Point2f &p, SampledSpectrum v,
                  const SampledWavelengths &lambda);

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);
    std::string ToString() const;

  private:
    // AOVFilm Private Data
    struct Pixel {
        Pixel() = default;
        double LSum[3] = { 0., 0., 0. };
        double weightSum = 0.;
        AtomicDouble splatRGB[3];
        Point3f pSum;
        Float dzdxSum = 0, dzdySum = 0;
        Normal3f nSum, nsSum;
        double albedoSum[3] = { 0., 0., 0. };
        double LeSum[3] = { 0., 0., 0. };
        double LdSum[3] = { 0., 0., 0. };
        VarianceEstimator<Float> LdVarianceEstimator, LiVarianceEstimator;
        RGB materialRGB;
    };
    std::mutex materialIdMapLock;
    std::map<std::string, RGB> staticMaterialIdMap, dynamicMaterialIdMap;
    Array2D<Pixel> pixels;
    const RGBColorSpace *colorSpace;
    Float maxSampleLuminance;
    bool writeFP16;
};

}  // namespace pbrt

#endif  // PBRT_FILM_AOV_H
