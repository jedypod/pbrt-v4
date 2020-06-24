
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
        return  FilterFunction::EWA;
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
    MIPMap(Image image, const RGBColorSpace *colorSpace,
           WrapMode wrapMode, Allocator alloc, const MIPMapFilterOptions &options);
    static std::unique_ptr<MIPMap> CreateFromFile(
        const std::string &filename, const MIPMapFilterOptions &options,
        WrapMode wrapMode, const ColorEncoding *encoding, Allocator alloc);
    ~MIPMap();

    template <typename T>
    PBRT_HOST_DEVICE
    T Lookup(const Point2f &st, Float width = 0.f) const;
    template <typename T>
    PBRT_HOST_DEVICE
    T Lookup(const Point2f &st, Vector2f dstdx, Vector2f dstdy) const;

    PBRT_HOST_DEVICE_INLINE
    Point2i LevelResolution(int level) const {
        CHECK(level >= 0 && level < pyramid.size());
        return pyramid[level].Resolution();
    }
    PBRT_HOST_DEVICE_INLINE
    int Levels() const { return int(pyramid.size()); }

    PBRT_HOST_DEVICE_INLINE
    const RGBColorSpace *GetRGBColorSpace() const {
        return colorSpace;
    }

    std::string ToString() const;

  private:
    template <typename T>
    PBRT_HOST_DEVICE
    T Texel(int level, Point2i st) const;
    template <typename T>
    PBRT_HOST_DEVICE
    T Bilerp(int level, Point2f st) const;
    template <typename T>
    PBRT_HOST_DEVICE
    T EWA(int level, Point2f st, Vector2f dst0, Vector2f dst1) const;

    pstd::vector<Image> pyramid;
    const RGBColorSpace *colorSpace;
    WrapMode wrapMode;
    MIPMapFilterOptions options;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MIPMAP_H
