// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_MIPMAP_H
#define PBRT_UTIL_MIPMAP_H

// core/mipmap.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/image.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <memory>
#include <string>
#include <vector>

namespace pbrt {

enum class FilterFunction { Point, Bilinear, Trilinear, EWA };

inline pstd::optional<FilterFunction> ParseFilter(const std::string &f) {
    if (f == "ewa" || f == "EWA")
        return FilterFunction::EWA;
    else if (f == "trilinear")
        return FilterFunction::Trilinear;
    else if (f == "bilinear")
        return FilterFunction::Bilinear;
    else if (f == "point")
        return FilterFunction::Point;
    else
        return {};
}

std::string ToString(FilterFunction f);

struct MIPMapFilterOptions {
    FilterFunction filter = FilterFunction::EWA;
    Float maxAnisotropy = 8.f;

    std::string ToString() const;
};

class MIPMap {
  public:
    MIPMap(Image image, const RGBColorSpace *colorSpace, WrapMode wrapMode,
           Allocator alloc, const MIPMapFilterOptions &options);
    static std::unique_ptr<MIPMap> CreateFromFile(const std::string &filename,
                                                  const MIPMapFilterOptions &options,
                                                  WrapMode wrapMode,
                                                  ColorEncodingHandle encoding,
                                                  Allocator alloc);

    template <typename T>
    T Lookup(const Point2f &st, Float width = 0.f) const;
    template <typename T>
    T Lookup(const Point2f &st, Vector2f dstdx, Vector2f dstdy) const;

    Point2i LevelResolution(int level) const {
        CHECK(level >= 0 && level < pyramid.size());
        return pyramid[level].Resolution();
    }
    int Levels() const { return int(pyramid.size()); }

    const RGBColorSpace *GetRGBColorSpace() const { return colorSpace; }

    std::string ToString() const;

  private:
    template <typename T>
    T Texel(int level, Point2i st) const;
    template <typename T>
    T Bilerp(int level, Point2f st) const;
    template <typename T>
    T EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const;

    pstd::vector<Image> pyramid;
    const RGBColorSpace *colorSpace;
    WrapMode wrapMode;
    MIPMapFilterOptions options;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MIPMAP_H
