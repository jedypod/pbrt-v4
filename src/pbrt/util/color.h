// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SPECTRUM_RGB_H
#define PBRT_SPECTRUM_RGB_H

// spectrum/rgb.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>

namespace pbrt {

class RGB {
  public:
    RGB() = default;
    PBRT_CPU_GPU
    RGB(Float r, Float g, Float b) : r(r), g(g), b(b) {}

    PBRT_CPU_GPU
    RGB &operator+=(const RGB &s) {
        r += s.r;
        g += s.g;
        b += s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator+(const RGB &s) const {
        RGB ret = *this;
        return ret += s;
    }

    PBRT_CPU_GPU
    RGB &operator-=(const RGB &s) {
        r -= s.r;
        g -= s.g;
        b -= s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator-(const RGB &s) const {
        RGB ret = *this;
        return ret -= s;
    }
    PBRT_CPU_GPU
    friend RGB operator-(Float a, const RGB &s) { return {a - s.r, a - s.g, a - s.b}; }

    PBRT_CPU_GPU
    RGB &operator*=(const RGB &s) {
        r *= s.r;
        g *= s.g;
        b *= s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator*(const RGB &s) const {
        RGB ret = *this;
        return ret *= s;
    }
    PBRT_CPU_GPU
    RGB operator*(Float a) const {
        DCHECK(!std::isnan(a));
        return {a * r, a * g, a * b};
    }
    PBRT_CPU_GPU
    RGB &operator*=(Float a) {
        DCHECK(!std::isnan(a));
        r *= a;
        g *= a;
        b *= a;
        return *this;
    }
    PBRT_CPU_GPU
    friend RGB operator*(Float a, const RGB &s) { return s * a; }

    PBRT_CPU_GPU
    RGB &operator/=(const RGB &s) {
        r /= s.r;
        g /= s.g;
        b /= s.b;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator/(const RGB &s) const {
        RGB ret = *this;
        return ret /= s;
    }
    PBRT_CPU_GPU
    RGB &operator/=(Float a) {
        DCHECK(!std::isnan(a));
        DCHECK_NE(a, 0);
        r /= a;
        g /= a;
        b /= a;
        return *this;
    }
    PBRT_CPU_GPU
    RGB operator/(Float a) const {
        RGB ret = *this;
        return ret /= a;
    }

    PBRT_CPU_GPU
    RGB operator-() const { return {-r, -g, -b}; }

    PBRT_CPU_GPU
    Float Average() const { return (r + g + b) / 3; }

    PBRT_CPU_GPU
    bool operator==(const RGB &s) const { return r == s.r && g == s.g && b == s.b; }
    PBRT_CPU_GPU
    bool operator!=(const RGB &s) const { return r != s.r || g != s.g || b != s.b; }
    PBRT_CPU_GPU
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return r;
        else if (c == 1)
            return g;
        return b;
    }
    PBRT_CPU_GPU
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return r;
        else if (c == 1)
            return g;
        return b;
    }

    std::string ToString() const;

    Float r = 0, g = 0, b = 0;
};

template <typename U, typename V>
PBRT_CPU_GPU inline RGB Clamp(const RGB &rgb, U min, V max) {
    return RGB(pbrt::Clamp(rgb.r, min, max), pbrt::Clamp(rgb.g, min, max),
               pbrt::Clamp(rgb.b, min, max));
}

PBRT_CPU_GPU
inline RGB ClampZero(const RGB &rgb) {
    return RGB(std::max<Float>(0, rgb.r), std::max<Float>(0, rgb.g),
               std::max<Float>(0, rgb.b));
}

PBRT_CPU_GPU
inline RGB Lerp(Float t, const RGB &s1, const RGB &s2) {
    return (1 - t) * s1 + t * s2;
}

class XYZ {
  public:
    XYZ() = default;
    PBRT_CPU_GPU
    XYZ(Float X, Float Y, Float Z) : X(X), Y(Y), Z(Z) {}
    PBRT_CPU_GPU
    static XYZ FromxyY(Float x, Float y, Float Y = 1) {
        if (y == 0)
            return XYZ(0, 0, 0);
        return XYZ(x * Y / y, Y, (1 - x - y) * Y / y);
    }

    PBRT_CPU_GPU
    XYZ &operator+=(const XYZ &s) {
        X += s.X;
        Y += s.Y;
        Z += s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator+(const XYZ &s) const {
        XYZ ret = *this;
        return ret += s;
    }

    PBRT_CPU_GPU
    XYZ &operator-=(const XYZ &s) {
        X -= s.X;
        Y -= s.Y;
        Z -= s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator-(const XYZ &s) const {
        XYZ ret = *this;
        return ret -= s;
    }
    PBRT_CPU_GPU
    friend XYZ operator-(Float a, const XYZ &s) { return {a - s.X, a - s.Y, a - s.Z}; }

    PBRT_CPU_GPU
    XYZ &operator*=(const XYZ &s) {
        X *= s.X;
        Y *= s.Y;
        Z *= s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator*(const XYZ &s) const {
        XYZ ret = *this;
        return ret *= s;
    }
    PBRT_CPU_GPU
    XYZ operator*(Float a) const {
        DCHECK(!std::isnan(a));
        return {a * X, a * Y, a * Z};
    }
    PBRT_CPU_GPU
    XYZ &operator*=(Float a) {
        DCHECK(!std::isnan(a));
        X *= a;
        Y *= a;
        Z *= a;
        return *this;
    }

    PBRT_CPU_GPU
    XYZ &operator/=(const XYZ &s) {
        X /= s.X;
        Y /= s.Y;
        Z /= s.Z;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator/(const XYZ &s) const {
        XYZ ret = *this;
        return ret /= s;
    }
    PBRT_CPU_GPU
    XYZ &operator/=(Float a) {
        DCHECK(!std::isnan(a));
        DCHECK_NE(a, 0);
        X /= a;
        Y /= a;
        Z /= a;
        return *this;
    }
    PBRT_CPU_GPU
    XYZ operator/(Float a) const {
        XYZ ret = *this;
        return ret /= a;
    }

    PBRT_CPU_GPU
    XYZ operator-() const { return {-X, -Y, -Z}; }

    PBRT_CPU_GPU
    Float Average() const { return (X + Y + Z) / 3; }

    PBRT_CPU_GPU
    bool operator==(const XYZ &s) const { return X == s.X && Y == s.Y && Z == s.Z; }
    PBRT_CPU_GPU
    bool operator!=(const XYZ &s) const { return X != s.X || Y != s.Y || Z != s.Z; }
    PBRT_CPU_GPU
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return X;
        else if (c == 1)
            return Y;
        return Z;
    }
    PBRT_CPU_GPU
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0)
            return X;
        else if (c == 1)
            return Y;
        return Z;
    }

    std::string ToString() const;

    Float X = 0, Y = 0, Z = 0;
};

PBRT_CPU_GPU
inline XYZ operator*(Float a, const XYZ &s) {
    return s * a;
}

template <typename U, typename V>
PBRT_CPU_GPU inline XYZ Clamp(const XYZ &xyz, U min, V max) {
    return XYZ(pbrt::Clamp(xyz.X, min, max), pbrt::Clamp(xyz.Y, min, max),
               pbrt::Clamp(xyz.Z, min, max));
}

PBRT_CPU_GPU
inline XYZ ClampZero(const XYZ &xyz) {
    return XYZ(std::max<Float>(0, xyz.X), std::max<Float>(0, xyz.Y),
               std::max<Float>(0, xyz.Z));
}

PBRT_CPU_GPU
inline XYZ Lerp(Float t, const XYZ &s1, const XYZ &s2) {
    return (1 - t) * s1 + t * s2;
}

class RGBSigmoidPolynomial {
  public:
    RGBSigmoidPolynomial() = default;
    PBRT_CPU_GPU
    RGBSigmoidPolynomial(Float c0, Float c1, Float c2) : c0(c0), c1(c1), c2(c2) {}

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        // c2 + c1 * lambda + c0 * lambda^2
        Float v = EvaluatePolynomial(lambda, c2, c1, c0);
        if (std::isinf(v))
            return v > 0 ? 1 : 0;
        return S(v);
    }

    PBRT_CPU_GPU
    Float MaxValue() const {
        // if c0 > 0, then the extremum is a minimum
        if (c0 < 0) {
            Float lambda = -c1 / (2 * c0);
            if (lambda >= 360 && lambda <= 830)
                return std::max({(*this)(lambda), (*this)(360), (*this)(830)});
        }
        return std::max((*this)(360), (*this)(830));
    }

    std::string ToString() const;

  private:
    PBRT_CPU_GPU
    static Float S(Float x) { return .5 + x / (2 * std::sqrt(1 + x * x)); };

    Float c0, c1, c2;
};

class RGBToSpectrumTable {
  public:
    RGBToSpectrumTable(int res, const float *scale, const float *data)
        : res(res), scale(scale), data(data) {}

    PBRT_CPU_GPU
    RGBSigmoidPolynomial operator()(const RGB &rgb) const;

    static void Init(Allocator alloc);

    static const RGBToSpectrumTable *sRGB;
    static const RGBToSpectrumTable *Rec2020;
    static const RGBToSpectrumTable *ACES2065_1;

    std::string ToString() const;

  private:
    int res = 0;
    const float *scale = nullptr, *data = nullptr;
};

class LinearColorEncoding;
class sRGBColorEncoding;
class GammaColorEncoding;

class ColorEncodingHandle
    : public TaggedPointer<LinearColorEncoding, sRGBColorEncoding, GammaColorEncoding> {
  public:
    using TaggedPointer::TaggedPointer;
    ColorEncodingHandle(
        TaggedPointer<LinearColorEncoding, sRGBColorEncoding, GammaColorEncoding> tp)
        : TaggedPointer(tp) {}

    PBRT_CPU_GPU inline void ToLinear(pstd::span<const uint8_t> vin,
                                      pstd::span<Float> vout) const;

    PBRT_CPU_GPU inline Float ToFloatLinear(Float v) const;

    PBRT_CPU_GPU inline void FromLinear(pstd::span<const Float> vin,
                                        pstd::span<uint8_t> vout) const;

    std::string ToString() const;

    static const ColorEncodingHandle Linear;
    static const ColorEncodingHandle sRGB;

    static const ColorEncodingHandle Get(const std::string &name);
};

class LinearColorEncoding {
  public:
    PBRT_CPU_GPU
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = vin[i] / 255.f;
    }

    PBRT_CPU_GPU
    Float ToFloatLinear(Float v) const { return v; }

    PBRT_CPU_GPU
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = uint8_t(Clamp(vin[i] * 255.f + 0.5f, 0, 255));
    }

    std::string ToString() const { return "[ LinearColorEncoding ]"; }
};

class sRGBColorEncoding {
  public:
    PBRT_CPU_GPU
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const;
    PBRT_CPU_GPU
    Float ToFloatLinear(Float v) const;
    PBRT_CPU_GPU
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const;

    std::string ToString() const { return "[ sRGBColorEncoding ]"; }
};

class GammaColorEncoding {
  public:
    PBRT_CPU_GPU
    GammaColorEncoding(Float gamma);

    PBRT_CPU_GPU
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const;
    PBRT_CPU_GPU
    Float ToFloatLinear(Float v) const;
    PBRT_CPU_GPU
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const;

    std::string ToString() const;

  private:
    Float gamma;
    pstd::array<Float, 256> applyLUT;
    pstd::array<Float, 1024> inverseLUT;
};

inline void ColorEncodingHandle::ToLinear(pstd::span<const uint8_t> vin,
                                          pstd::span<Float> vout) const {
    auto tolin = [&](auto ptr) { return ptr->ToLinear(vin, vout); };
    return Apply<void>(tolin);
}

inline Float ColorEncodingHandle::ToFloatLinear(Float v) const {
    auto tfl = [&](auto ptr) { return ptr->ToFloatLinear(v); };
    return Apply<Float>(tfl);
}

inline void ColorEncodingHandle::FromLinear(pstd::span<const Float> vin,
                                            pstd::span<uint8_t> vout) const {
    auto fl = [&](auto ptr) { return ptr->FromLinear(vin, vout); };
    return Apply<void>(fl);
}

PBRT_CPU_GPU
inline Float LinearToSRGBFull(Float value) {
    if (value <= 0.0031308f)
        return 12.92f * value;
    return 1.055f * std::pow(value, (Float)(1.f / 2.4f)) - 0.055f;
}

struct PiecewiseLinearSegment {
    Float base, slope;
};

// Piecewise linear fit to LinearToSRGBFull() (via Mathematica).
// Table size 1024 gave avg error: 7.36217e-07, max: 0.000284649
// 512 gave avg: 1.76644e-06, max: 0.000490334
// 256 gave avg: 5.68012e-06, max: 0.00116351
// 128 gave avg: 2.90114e-05, max: 0.00502084
// 256 seemed like a reasonable trade-off.

extern PBRT_CONST PiecewiseLinearSegment LinearToSRGBPiecewise[];
constexpr int LinearToSRGBPiecewiseSize = 256;

PBRT_CPU_GPU
inline Float LinearToSRGB(Float value) {
    int index = int(value * LinearToSRGBPiecewiseSize);
    if (index < 0)
        return 0;
    if (index >= LinearToSRGBPiecewiseSize)
        return 1;
    return LinearToSRGBPiecewise[index].base + value * LinearToSRGBPiecewise[index].slope;
}

PBRT_CPU_GPU
inline uint8_t LinearToSRGB8(Float value, Float dither = 0) {
    if (value <= 0)
        return 0;
    if (value >= 1)
        return 255;
    return Clamp(255.f * LinearToSRGB(value) + dither, 0, 255);
}

PBRT_CPU_GPU
inline Float SRGBToLinear(Float value) {
    if (value <= 0.04045f)
        return value * (1 / 12.92f);
    return std::pow((value + 0.055f) * (1 / 1.055f), (Float)2.4f);
}

extern PBRT_CONST Float SRGBToLinearLUT[256];

PBRT_CPU_GPU
inline Float SRGB8ToLinear(uint8_t value) {
    return SRGBToLinearLUT[value];
}

}  // namespace pbrt

#endif  // PBRT_COLOR_H
