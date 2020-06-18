// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_FILM_H
#define PBRT_BASE_FILM_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/base/filter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

class VisibleSurface;
class RGBFilm;
class GBufferFilm;

class FilmHandle : public TaggedPointer<RGBFilm, GBufferFilm> {
  public:
    using TaggedPointer::TaggedPointer;

    static FilmHandle Create(const std::string &name,
                             const ParameterDictionary &parameters, const FileLoc *loc,
                             FilterHandle filter, Allocator alloc);

    PBRT_CPU_GPU inline SampledWavelengths SampleWavelengths(Float u) const;

    PBRT_CPU_GPU inline void AddSample(
        const Point2i &pFilm, SampledSpectrum L, const SampledWavelengths &lambda,
        const pstd::optional<VisibleSurface> &visibleSurface, Float weight);

    PBRT_CPU_GPU inline Bounds2f SampleBounds() const;

    PBRT_CPU_GPU inline FilterHandle GetFilter() const;

    PBRT_CPU_GPU inline Point2i FullResolution() const;

    PBRT_CPU_GPU inline Float Diagonal() const;

    PBRT_CPU_GPU inline Bounds2i PixelBounds() const;

    PBRT_CPU_GPU
    void AddSplat(const Point2f &p, SampledSpectrum v, const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    bool UsesVisibleSurface() const;

    PBRT_CPU_GPU
    RGB GetPixelRGB(const Point2i &p, Float splatScale = 1) const;

    std::string GetFilename() const;

    void WriteImage(ImageMetadata metadata, Float splatScale = 1);
    Image GetImage(ImageMetadata *metadata, Float splatScale = 1);

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_FILM_H
