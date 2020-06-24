
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

#ifndef PBRT_UTIL_IMAGE_H
#define PBRT_UTIL_IMAGE_H

// util/image.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/array2d.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#if defined(PBRT_HAVE_OPTIX) && defined(__CUDA_ARCH__)
  #include <cuda_fp16.h>
#endif

#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <vector>

namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// PixelFormat

enum class PixelFormat { U256, Half, Float };

PBRT_HOST_DEVICE_INLINE
bool Is8Bit(PixelFormat format) {
    return format == PixelFormat::U256;
}

PBRT_HOST_DEVICE_INLINE
bool Is16Bit(PixelFormat format) {
    return format == PixelFormat::Half;
}

PBRT_HOST_DEVICE_INLINE
bool Is32Bit(PixelFormat format) {
    return format == PixelFormat::Float;
}

std::string ToString(PixelFormat format);

    PBRT_HOST_DEVICE
int TexelBytes(PixelFormat format);

///////////////////////////////////////////////////////////////////////////
// WrapMode

enum class WrapMode { Repeat, Black, Clamp, OctahedralSphere };

struct WrapMode2D {
    PBRT_HOST_DEVICE
    WrapMode2D(WrapMode w) : wrap{w, w} {}
    PBRT_HOST_DEVICE
    WrapMode2D(WrapMode x, WrapMode y) : wrap{x, y} {}

    pstd::array<WrapMode, 2> wrap;
};

inline pstd::optional<WrapMode> ParseWrapMode(const char *w) {
    if (!strcmp(w, "clamp"))
        return WrapMode::Clamp;
    else if (!strcmp(w, "repeat"))
        return WrapMode::Repeat;
    else if (!strcmp(w, "black"))
        return WrapMode::Black;
    else if (!strcmp(w, "octahedralsphere"))
        return WrapMode::OctahedralSphere;
    else
        return {};
}

inline std::string ToString(WrapMode mode) {
    switch (mode) {
    case WrapMode::Clamp:
        return "clamp";
    case WrapMode::Repeat:
        return "repeat";
    case WrapMode::Black:
        return "black";
    case WrapMode::OctahedralSphere:
        return "octahedralsphere";
    default:
        LOG_FATAL("Unhandled wrap mode");
        return nullptr;
    }
}

PBRT_HOST_DEVICE_INLINE
bool RemapPixelCoords(Point2i *pp, Point2i resolution, WrapMode2D wrapMode) {
    Point2i &p = *pp;

    if (wrapMode.wrap[0] == WrapMode::OctahedralSphere) {
        CHECK(wrapMode.wrap[1] == WrapMode::OctahedralSphere);
        if (p[0] < 0) {
            p[0] = -p[0];     // mirror across u = 0
            p[1] = resolution[1] - 1 - p[1];  // mirror across v = 0.5
        } else if (p[0] >= resolution[0]) {
            p[0] = 2 * resolution[0] - 1 - p[0];  // mirror across u = 1
            p[1] = resolution[1] - 1 - p[1];  // mirror across v = 0.5
        }

        if (p[1] < 0) {
            p[0] = resolution[0] - 1 - p[0];  // mirror across u = 0.5
            p[1] = -p[1];     // mirror across v = 0;
        } else if (p[1] >= resolution[1]) {
            p[0] = resolution[0] - 1 - p[0];  // mirror across u = 0.5
            p[1] = 2 * resolution[1] - 1 - p[1];  // mirror across v = 1
        }

        // Bleh: things don't go as expected for 1x1 images.
        if (resolution[0] == 1) p[0] = 0;
        if (resolution[1] == 1) p[1] = 0;

        return true;
    }

    for (int c = 0; c < 2; ++c) {
        if (p[c] >= 0 && p[c] < resolution[c])
            // in bounds
            continue;

        switch (wrapMode.wrap[c]) {
        case WrapMode::Repeat:
            p[c] = Mod(p[c], resolution[c]);
            break;
        case WrapMode::Clamp:
            p[c] = Clamp(p[c], 0, resolution[c] - 1);
            break;
        case WrapMode::Black:
            return false;
        default:
            LOG_FATAL("Unhandled WrapMode mode");
        }
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////
// ImageMetadata

struct ImageMetadata {
    // These may or may not be present in the metadata of an Image.
    pstd::optional<float> renderTimeSeconds;
    pstd::optional<SquareMatrix<4>> cameraFromWorld, NDCFromWorld;
    pstd::optional<Bounds2i> pixelBounds;
    pstd::optional<Point2i> fullResolution;
    pstd::optional<int> samplesPerPixel;
    pstd::optional<float> estimatedVariance, MSE;
    pstd::optional<const RGBColorSpace *> colorSpace;
    std::map<std::string, std::vector<std::string>> stringVectors;

    const RGBColorSpace *GetColorSpace() const;
    std::string ToString() const;
};

struct ImageAndMetadata;

struct ImageChannelDesc {
    size_t size() const { return offset.size(); }
    bool IsIdentity() const {
        for (size_t i = 0; i < offset.size(); ++i)
            if (offset[i] != i)
                return false;
        return true;
    }
    std::string ToString() const;

    InlinedVector<int, 4> offset;
};

struct ImageChannelValues : public InlinedVector<Float, 4> {
    //ImageChannelValues() = default;
    explicit ImageChannelValues(size_t sz, Float v = {}) : InlinedVector<Float, 4>(sz, v) {}

    operator Float() const {
        CHECK_EQ(1, size());
        return (*this)[0];
    }
    operator pstd::array<Float, 3>() const {
        CHECK_EQ(3, size());
        return { (*this)[0], (*this)[1], (*this)[2] };
    }

    Float MaxValue() const {
        Float m = (*this)[0];
        for (int i = 1; i < size(); ++i)
            m = std::max(m, (*this)[i]);
        return m;
    }
    Float Average() const {
        Float sum = 0;
        for (int i = 0; i < size(); ++i)
            sum += (*this)[i];
        return sum / size();
    }

    std::string ToString() const;
};

///////////////////////////////////////////////////////////////////////////
// Image

// Important: coordinate system for our images has (0,0) at the upper left
// corner.

class Image {
  public:
    Image(Allocator alloc = {}) : p8(alloc), p16(alloc), p32(alloc),
                                  format(PixelFormat::U256), resolution(0, 0) {}
    Image(pstd::vector<uint8_t> p8, Point2i resolution,
          pstd::span<const std::string> channels,
          const ColorEncoding *encoding);
    Image(pstd::vector<Half> p16, Point2i resolution,
          pstd::span<const std::string> channels);
    Image(pstd::vector<float> p32, Point2i resolution,
          pstd::span<const std::string> channels);
    Image(PixelFormat format, Point2i resolution,
          pstd::span<const std::string> channels,
          const ColorEncoding *encoding = nullptr, Allocator alloc = {});

    static pstd::optional<ImageAndMetadata>
    Read(const std::string &filename, Allocator alloc = {},
         const ColorEncoding *encoding = nullptr);

    bool Write(const std::string &name,
               const ImageMetadata &metadata = {}) const;

    Image ConvertToFormat(PixelFormat format,
                          const ColorEncoding *encoding = nullptr) const;

    // TODO? provide an iterator to iterate over all pixels and channels?

    ImageChannelDesc AllChannelsDesc() const {
        ImageChannelDesc desc;
        desc.offset.resize(NChannels());
        for (int i = 0; i < NChannels(); ++i)
            desc.offset[i] = i;
        return desc;
    }
    pstd::optional<ImageChannelDesc> GetChannelDesc(pstd::span<const std::string> channels) const;
    Image SelectChannels(const ImageChannelDesc &desc, Allocator alloc = {}) const;
    Image Crop(const Bounds2i &bounds, Allocator alloc = {}) const;

    PBRT_HOST_DEVICE_INLINE
    Float GetChannel(Point2i p, int c,
                     WrapMode2D wrapMode = WrapMode::Clamp) const {
        if (!RemapPixelCoords(&p, resolution, wrapMode)) return 0;

        switch (format) {
        case PixelFormat::U256: {
#ifdef __CUDA_ARCH__
            // Assume sRGB...
            return SRGB8ToLinear(p8[PixelOffset(p) + c]);
#else
            Float r;
            encoding->ToLinear({&p8[PixelOffset(p) + c], 1}, {&r, 1});
            return r;
#endif
        }
        case PixelFormat::Half:
#ifdef __CUDA_ARCH__
            return __ushort_as_half(p16[PixelOffset(p) + c].Bits());
#else
            return Float(p16[PixelOffset(p) + c]);
#endif
        case PixelFormat::Float:
            return p32[PixelOffset(p) + c];
        default:
            LOG_FATAL("Unhandled PixelFormat");
            return 0;
        }
    }

    ImageChannelValues GetChannels(Point2i p, WrapMode2D wrapMode = WrapMode::Clamp) const;
    ImageChannelValues GetChannels(Point2i p, const ImageChannelDesc &desc,
                                   WrapMode2D wrapMode = WrapMode::Clamp) const;

    // FIXME: could be / should be const...
    void CopyRectOut(const Bounds2i &extent, pstd::span<float> buf,
                     WrapMode2D wrapMode = WrapMode::Clamp);
    void CopyRectIn(const Bounds2i &extent, pstd::span<const float> buf);

    PBRT_HOST_DEVICE_INLINE
    Float BilerpChannel(Point2f p, int c,
                        WrapMode2D wrapMode = WrapMode::Clamp) const {
        Float x = p[0] * resolution.x - 0.5f, y = p[1] * resolution.y - 0.5f;
        int xi = std::floor(x), yi = std::floor(y);
        Float dx = x - xi, dy = y - yi;
        pstd::array<Float, 4> v = { GetChannel({xi, yi}, c, wrapMode),
                                    GetChannel({xi + 1, yi}, c, wrapMode),
                                    GetChannel({xi, yi + 1}, c, wrapMode),
                                    GetChannel({xi + 1, yi + 1}, c, wrapMode) };
        return pbrt::Bilerp({dx, dy}, v);
    }

    ImageChannelValues Bilerp(Point2f p, WrapMode2D wrapMode = WrapMode::Clamp) const;
    ImageChannelValues Bilerp(Point2f p, const ImageChannelDesc &desc,
                              WrapMode2D wrapMode = WrapMode::Clamp) const;

    PBRT_HOST_DEVICE_INLINE
    void SetChannel(Point2i p, int c, Float value) {
        //CHECK(!std::isnan(value));
        if (std::isnan(value)) {
#ifndef __CUDA_ARCH__
            LOG_ERROR("NaN at pixel %d,%d comp %d", p.x, p.y, c);
#endif
            value = 0;
        }

        switch (format) {
        case PixelFormat::U256:
#ifdef __CUDA_ARCH__
            p8[PixelOffset(p) + c] = LinearToSRGB8(value);
#else
            encoding->FromLinear({&value, 1}, {&p8[PixelOffset(p) + c], 1});
#endif
            break;
        case PixelFormat::Half:
#ifdef __CUDA_ARCH__
            p16[PixelOffset(p) + c] = Half::FromBits(__half_as_ushort(__float2half(value)));
#else
            p16[PixelOffset(p) + c] = Half(value);
#endif
            break;
        case PixelFormat::Float:
            p32[PixelOffset(p) + c] = value;
            break;
        default:
            LOG_FATAL("Unhandled PixelFormat in Image::SetChannel()");
        }
    }

    void SetChannels(Point2i p, const ImageChannelValues &values);
    void SetChannels(Point2i p, pstd::span<const Float> values);
    void SetChannels(Point2i p, const ImageChannelDesc &desc,
                     pstd::span<const Float> values);

    Image FloatResize(Point2i newResolution, WrapMode2D wrap) const;
    void FlipY();
    static pstd::vector<Image> GenerateMIPMap(Image image, WrapMode2D wrapMode,
                                              Allocator alloc = {});

    PBRT_HOST_DEVICE_INLINE
    PixelFormat Format() const { return format; }
    PBRT_HOST_DEVICE_INLINE
    Point2i Resolution() const { return resolution; }
    PBRT_HOST_DEVICE_INLINE
    int NChannels() const { return channelNames.size(); }

    std::vector<std::string> ChannelNames() const;
    std::vector<std::string> ChannelNames(const ImageChannelDesc &) const;
    const ColorEncoding *Encoding() const { return encoding; }

    PBRT_HOST_DEVICE_INLINE
    size_t BytesUsed() const {
        return p8.size() + 2 * p16.size() + 4 * p32.size();
    }

    ImageChannelValues Average(const ImageChannelDesc &desc) const;
    ImageChannelValues L1Error(const ImageChannelDesc &desc, const Image &ref,
                             Image *errorImage = nullptr) const;
    ImageChannelValues MSE(const ImageChannelDesc &desc, const Image &ref,
                           Image *mseImage = nullptr) const;
    ImageChannelValues MRSE(const ImageChannelDesc &desc, const Image &ref,
                            Image *mrseImage = nullptr) const;

    PBRT_HOST_DEVICE_INLINE
    size_t PixelOffset(Point2i p) const {
        DCHECK(InsideExclusive(p, Bounds2i({0, 0}, resolution)));
        return NChannels() * (p.y * resolution.x + p.x);
    }
    PBRT_HOST_DEVICE_INLINE
    const void *RawPointer(Point2i p) const {
        if (Is8Bit(format)) return p8.data() + PixelOffset(p);
        if (Is16Bit(format))
            return p16.data() + PixelOffset(p);
        else {
            CHECK(Is32Bit(format));
            return p32.data() + PixelOffset(p);
        }
    }
    PBRT_HOST_DEVICE_INLINE
    void *RawPointer(Point2i p) {
        return const_cast<void *>(((const Image *)this)->RawPointer(p));
    }

    Image GaussianFilter(const ImageChannelDesc &desc, int halfWidth, Float sigma) const;
    Image JointBilateralFilter(const ImageChannelDesc &toFilter,
                               int halfWidth, const Float xySigma[2],
                               const ImageChannelDesc &joint,
                               const ImageChannelValues &jointSigma) const;

    Array2D<Float> ComputeSamplingDistribution(
        std::function<Float(Point2f)> dxdA, const ImageChannelDesc &desc, int resScale = 1,
        Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1)),
        Norm norm = Norm::LInfinity,
        WrapMode2D wrap = WrapMode::Clamp);

    Array2D<Float> ComputeSamplingDistribution(
        const ImageChannelDesc &desc, int resScale = 1,
        Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1)),
        Norm norm = Norm::LInfinity,
        WrapMode2D wrap = WrapMode::Clamp) {
        return ComputeSamplingDistribution([](Point2f) { return Float(1); },
                                           desc, resScale, domain, norm, wrap);
    }

    std::string ToString() const;

  private:
    PixelFormat format;
    Point2i resolution;
    InlinedVector<std::string, 4> channelNames;
    // Note: encoding is only used for 8-bit pixel formats. Everything else
    // is expected to be linear already.
    const ColorEncoding *encoding = nullptr;

    bool WriteEXR(const std::string &name, const ImageMetadata &metadata) const;
    bool WritePFM(const std::string &name, const ImageMetadata &metadata) const;
    bool WritePNG(const std::string &name, const ImageMetadata &metadata) const;

    pstd::vector<uint8_t> p8;
    pstd::vector<Half> p16;
    pstd::vector<float> p32;
};

struct ImageAndMetadata {
    Image image;
    ImageMetadata metadata;
};

class SummedAreaTable {
public:
    explicit SummedAreaTable(const Array2D<Float> &values) {
        sum = integrate({values.xSize(), values.ySize()},
                         [&values](int x, int y) { return values(x, y); });
        sum2 = integrate({values.xSize(), values.ySize()},
                          [&values](int x, int y) { return values(x, y) * values(x, y); });
    }

    double Sum(const Bounds2i &extent) const {
        return (lookup(sum, extent.pMax.x, extent.pMax.y) +
                lookup(sum, extent.pMin.x, extent.pMin.y) -
                lookup(sum, extent.pMin.x, extent.pMax.y) -
                lookup(sum, extent.pMax.x, extent.pMin.y));
    }
    double Average(const Bounds2i &extent) const {
        return Sum(extent) / extent.Area();
    }
    double Variance(const Bounds2i &extent) const {
        double s = Sum(extent);
        double s2 = (lookup(sum2, extent.pMax.x, extent.pMax.y) +
                     lookup(sum2, extent.pMin.x, extent.pMin.y) -
                     lookup(sum2, extent.pMin.x, extent.pMax.y) -
                     lookup(sum2, extent.pMax.x, extent.pMin.y));
        // https://en.wikipedia.org/wiki/Summed-area_table
        Float n = extent.Area();
        return (1 / n) * std::max<Float>(0, s2 - s * s / n);
    }

    std::string ToString() const;

private:
    template <typename F> Array2D<double> integrate(Point2i res, F f) {
        Array2D<double> result(res.x, res.y);

        result(0, 0) = f(0, 0);

        // sum across first scanline
        for (int x = 1; x < result.xSize(); ++x)
            result(x, 0) = f(x, 0) + result(x - 1, 0);

        // sum up first column
        for (int y = 1; y < result.ySize(); ++y)
            result(0, y) = f(0, y) + result(0, y - 1);

        // and all the rest of it
        for (int y = 1; y < result.ySize(); ++y)
            for (int x = 1; x < result.xSize(); ++x)
                result(x, y) = (f(x, y) + result(x - 1, y) +
                                result(x, y - 1) - result(x - 1, y - 1));

        return result;
    }

    static double lookup(const Array2D<double> &s, int x, int y) {
        if (--x < 0) return 0;
        if (--y < 0) return 0;
        return s(x, y);
    }

    Array2D<double> sum, sum2;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_IMAGE_H
