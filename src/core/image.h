
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

#ifndef PBRT_CORE_IMAGE_H
#define PBRT_CORE_IMAGE_H

// core/image.h*
#include "pbrt.h"

#include "spectrum.h"
#include "util/bounds.h"
#include "util/geometry.h"
#include "util/half.h"
#include "util/transform.h"
#include <glog/logging.h>
#include <absl/types/span.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <experimental/optional>
#include <map>
#include <vector>

namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// PixelFormat

// TODO: Y8 -> G8 (or GREY8?)
enum class PixelFormat { SY8, Y8, RGB8, SRGB8, Y16, RGB16, Y32, RGB32 };

inline bool Is8Bit(PixelFormat format) {
    return (format == PixelFormat::SY8 || format == PixelFormat::Y8 ||
            format == PixelFormat::SRGB8 || format == PixelFormat::RGB8);
}

inline bool Is16Bit(PixelFormat format) {
    return (format == PixelFormat::Y16 || format == PixelFormat::RGB16);
}

inline bool Is32Bit(PixelFormat format) {
    return (format == PixelFormat::Y32 || format == PixelFormat::RGB32);
}

inline int nChannels(PixelFormat format) {
    switch (format) {
    case PixelFormat::SY8:
    case PixelFormat::Y8:
    case PixelFormat::Y16:
    case PixelFormat::Y32:
        return 1;
    case PixelFormat::RGB8:
    case PixelFormat::SRGB8:
    case PixelFormat::RGB16:
    case PixelFormat::RGB32:
        return 3;
    }
}

inline const char *FormatName(PixelFormat format) {
    switch (format) {
    case PixelFormat::SY8: return "SY8";
    case PixelFormat::Y8: return "Y8";
    case PixelFormat::Y16: return "Y16";
    case PixelFormat::Y32: return "Y32";
    case PixelFormat::RGB8: return "RGB8";
    case PixelFormat::SRGB8: return "SRGB8";
    case PixelFormat::RGB16: return "RGB16";
    case PixelFormat::RGB32: return "RGB32";
    default: return nullptr;
    }
}

inline int TexelBytes(PixelFormat format) {
    switch (format) {
    case PixelFormat::SY8:
    case PixelFormat::Y8:
        return 1;
    case PixelFormat::RGB8:
    case PixelFormat::SRGB8:
        return 3;
    case PixelFormat::Y16:
        return 2;
    case PixelFormat::RGB16:
        return 6;
    case PixelFormat::Y32:
        return 4;
    case PixelFormat::RGB32:
        return 12;
    default:
        LOG(ERROR) << "Unhandled PixelFormat in TexelBytes()";
    }
}

template <typename T>
static T ConvertTexel(const void *ptr, PixelFormat format) {
    T::unimplemented_function;
}

template <>
Spectrum ConvertTexel(const void *ptr, PixelFormat format);

template <>
Float ConvertTexel(const void *ptr, PixelFormat format) {
    if (nChannels(format) != 1)
        return ConvertTexel<Spectrum>(ptr, format).Average();
    if (ptr == nullptr) return 0;

    // TODO: are those pointer casts ok or not? ok if char I think, not
    // sure about uint8_t, strictly speaking...
    switch (format) {
    case PixelFormat::SY8:
        return SRGB8ToLinear(*((uint8_t *)ptr));
    case PixelFormat::Y8:
        return Float(*((uint8_t *)ptr)) / 255.f;
    case PixelFormat::Y16:
        return float(*(Half *)ptr);
    case PixelFormat::Y32:
        return Float(*((float *)ptr));
    default:
        LOG(FATAL) << "Unhandled PixelFormat";
    }
}

template <>
Spectrum ConvertTexel(const void *ptr, PixelFormat format) {
    if (nChannels(format) == 1)
        return Spectrum(ConvertTexel<Float>(ptr, format));
    if (ptr == nullptr) return Spectrum(0);

    CHECK_EQ(3, nChannels(format));
    Float rgb[3];
    for (int c = 0; c < 3; ++c) {
        switch (format) {
        case PixelFormat::SRGB8:
            rgb[c] = SRGB8ToLinear(((uint8_t *)ptr)[c]);
            break;
        case PixelFormat::RGB8:
            rgb[c] = Float(((uint8_t *)ptr)[c]) / 255.f;
            break;
        case PixelFormat::RGB16:
            rgb[c] = float(((Half *)ptr)[c]);
            break;
        case PixelFormat::RGB32:
            rgb[c] = Float(((float *)ptr)[c]);
            break;
        default:
            LOG(FATAL) << "Unhandled pixelformat";
        }
    }

    // TODO: pass through illuminant/reflectance enum? (Or nix this whole
    // idea)...
    return Spectrum::FromRGB(rgb);
}

///////////////////////////////////////////////////////////////////////////
// WrapMode

enum class WrapMode { Repeat, Black, Clamp };

inline bool ParseWrapMode(const char *w, WrapMode *wrapMode) {
    if (!strcmp(w, "clamp")) {
        *wrapMode = WrapMode::Clamp;
        return true;
    } else if (!strcmp(w, "repeat")) {
        *wrapMode = WrapMode::Repeat;
        return true;
    } else if (!strcmp(w, "black")) {
        *wrapMode = WrapMode::Black;
        return true;
    }
    return false;
}

inline const char *WrapModeString(WrapMode mode) {
    switch (mode) {
    case WrapMode::Clamp:
        return "clamp";
    case WrapMode::Repeat:
        return "repeat";
    case WrapMode::Black:
        return "black";
    default:
        LOG(FATAL) << "Unhandled wrap mode";
        return nullptr;
    }
}

bool RemapPixelCoords(Point2i *p, Point2i resolution, WrapMode wrapMode);

///////////////////////////////////////////////////////////////////////////
// ImageMetadata

struct ImageMetadata {
    // These may or may not be present in the metadata of an Image.
    std::experimental::optional<Float> renderTimeSeconds;
    std::experimental::optional<Matrix4x4> worldToCamera, worldToNDC;
    std::experimental::optional<Bounds2i> pixelBounds;
    std::experimental::optional<Point2i> fullResolution;
    std::map<std::string, std::vector<std::string>> stringVectors;
};

///////////////////////////////////////////////////////////////////////////
// Image

// Important: coordinate system for our images has (0,0) at the upper left
// corner.

class Image {
  public:
    Image() : format(PixelFormat::Y8), resolution(0, 0) {}
    Image(std::vector<uint8_t> p8, PixelFormat format, Point2i resolution);
    Image(std::vector<Half> p16, PixelFormat format, Point2i resolution);
    Image(std::vector<float> p32, PixelFormat format, Point2i resolution);
    Image(PixelFormat format, Point2i resolution);

    // TODO: make gamma option more flexible: sRGB vs provided gamma
    // exponent...
    static std::experimental::optional<Image> Read(
        const std::string &filename, ImageMetadata *metadata = nullptr,
        bool gamma = true);
    bool Write(const std::string &name,
               const ImageMetadata *metadata = nullptr) const;

    Image ConvertToFormat(PixelFormat format) const;

    // TODO? provide an iterator to iterate over all pixels and channels?

    Float GetChannel(Point2i p, int c,
                     WrapMode wrapMode = WrapMode::Clamp) const;
    Float GetY(Point2i p, WrapMode wrapMode = WrapMode::Clamp) const;
    Spectrum GetSpectrum(Point2i p,
                         SpectrumType spectrumType = SpectrumType::Reflectance,
                         WrapMode wrapMode = WrapMode::Clamp) const;

    // FIXME: could be / should be const...
    void CopyRectOut(const Bounds2i &extent,
                     absl::Span<Float> buf,
                     WrapMode wrapMode = WrapMode::Clamp);
    void CopyRectIn(const Bounds2i &extent,
                    absl::Span<const Float> buf);

    Float BilerpChannel(Point2f p, int c,
                        WrapMode wrapMode = WrapMode::Clamp) const;
    Float BilerpY(Point2f p, WrapMode wrapMode = WrapMode::Clamp) const;
    Spectrum BilerpSpectrum(
        Point2f p, SpectrumType spectrumType = SpectrumType::Reflectance,
        WrapMode wrapMode = WrapMode::Clamp) const;

    void SetChannel(Point2i p, int c, Float value);
    void SetY(Point2i p, Float value);
    void SetSpectrum(Point2i p, const Spectrum &value);

    Image FloatResize(Point2i newResolution, WrapMode wrap) const;
    void FlipY();
    static std::vector<Image> GenerateMIPMap(Image image, WrapMode wrapMode);

    int nChannels() const { return pbrt::nChannels(format); }
    size_t BytesUsed() const {
        return p8.size() + 2 * p16.size() + 4 * p32.size();
    }

    PixelFormat format;
    Point2i resolution;

    size_t PixelOffset(Point2i p, int c = 0) const {
        DCHECK(c >= 0 && c < nChannels());
        DCHECK(InsideExclusive(p, Bounds2i({0, 0}, resolution)));
        return nChannels() * (p.y * resolution.x + p.x) + c;
    }
    const void *RawPointer(Point2i p) const {
        if (Is8Bit(format)) return p8.data() + PixelOffset(p);
        if (Is16Bit(format))
            return p16.data() + PixelOffset(p);
        else {
            CHECK(Is32Bit(format));
            return p32.data() + PixelOffset(p);
        }
    }
    void *RawPointer(Point2i p) {
        return const_cast<void *>(((const Image *)this)->RawPointer(p));
    }

  private:
    std::array<Float, 3> GetRGB(Point2i p, WrapMode wrapMode) const;

    bool WriteEXR(const std::string &name, const ImageMetadata *metadata) const;
    bool WritePFM(const std::string &name, const ImageMetadata *metadata) const;
    bool WritePNG(const std::string &name, const ImageMetadata *metadata) const;

    template <typename F> void ForExtent1(const Bounds2i &extent, WrapMode wrapMode, F op) {
        CHECK_LT(extent.pMin.x, extent.pMax.x);
        CHECK_LT(extent.pMin.y, extent.pMax.y);
        CHECK_EQ(nChannels(), 1);

        int nu = extent.pMax[0] - extent.pMin[0];
        if (Intersect(extent, Bounds2i({0, 0}, resolution)) == extent) {
            // All in bounds
            for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
                int offset = PixelOffset({extent.pMin[0], v}, 0);
                for (int u = 0; u < nu; ++u)
                    op(offset++);
            }
        } else {
            for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
                for (int u = 0; u < nu; ++u) {
                    Point2i p(extent.pMin[0] + u, v);
                    // FIXME: this will return false on Black wrap mode
                    CHECK(RemapPixelCoords(&p, resolution, wrapMode));
                    int offset = PixelOffset(p, 0);
                    op(offset++);
                }
            }
        }
    }
    template <typename F> void ForExtent3(const Bounds2i &extent, WrapMode wrapMode,
                                          F op) {
        CHECK_LT(extent.pMin.x, extent.pMax.x);
        CHECK_LT(extent.pMin.y, extent.pMax.y);
        CHECK_EQ(nChannels(), 3);

        int nu = extent.pMax[0] - extent.pMin[0];
        if (Intersect(extent, Bounds2i({0, 0}, resolution)) == extent) {
            // All in bounds
            for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
                int offset = PixelOffset({extent.pMin[0], v}, 0);
                for (int u = 0; u < nu; ++u)
                    for (int c = 0; c < 3; ++c)
                        op(offset++);
            }
        } else {
            for (int v = extent.pMin[1]; v < extent.pMax[1]; ++v) {
                for (int u = 0; u < nu; ++u) {
                    Point2i p(extent.pMin[0] + u, v);
                    // FIXME: this will return false on Black wrap mode
                    CHECK(RemapPixelCoords(&p, resolution, wrapMode));
                    int offset = PixelOffset(p, 0);
                    for (int c = 0; c < 3; ++c)
                        op(offset++);
                }
            }
        }
    }

    std::vector<uint8_t> p8;
    std::vector<Half> p16;
    std::vector<float> p32;
};

}  // namespace pbrt

#endif  // PBRT_CORE_IMAGE_H
