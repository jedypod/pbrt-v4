
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

#ifndef PBRT_SPECTRUM_RGB_H
#define PBRT_SPECTRUM_RGB_H

// spectrum/rgb.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <cmath>
#include <map>
#include <memory>
#include <string>

namespace pbrt {

class RGB {
  public:
    RGB() = default;
    PBRT_HOST_DEVICE_INLINE
    RGB(Float r, Float g, Float b)
        : r(r), g(g), b(b) {}

    PBRT_HOST_DEVICE_INLINE
    RGB &operator+=(const RGB &s) {
        r += s.r;
        g += s.g;
        b += s.b;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB operator+(const RGB &s) const {
        RGB ret = *this;
        return ret += s;
    }

    PBRT_HOST_DEVICE_INLINE
    RGB &operator-=(const RGB &s) {
        r -= s.r;
        g -= s.g;
        b -= s.b;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB operator-(const RGB &s) const {
        RGB ret = *this;
        return ret -= s;
    }
    PBRT_HOST_DEVICE_INLINE
    friend RGB operator-(Float a, const RGB &s) {
        return {a - s.r, a - s.g, a - s.b};
    }

    PBRT_HOST_DEVICE_INLINE
    RGB &operator*=(const RGB &s) {
        r *= s.r;
        g *= s.g;
        b *= s.b;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB operator*(const RGB &s) const {
        RGB ret = *this;
        return ret *= s;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB operator*(Float a) const {
        DCHECK(!std::isnan(a));
        return {a * r, a * g, a * b};
    }
    PBRT_HOST_DEVICE_INLINE
    RGB &operator*=(Float a) {
        DCHECK(!std::isnan(a));
        r *= a;
        g *= a;
        b *= a;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    friend RGB operator*(Float a, const RGB &s) {
        return s * a;
    }

    PBRT_HOST_DEVICE_INLINE
    RGB &operator/=(const RGB &s) {
        r /= s.r;
        g /= s.g;
        b /= s.b;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB operator/(const RGB &s) const {
        RGB ret = *this;
        return ret /= s;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB &operator/=(Float a) {
        DCHECK(!std::isnan(a));
        DCHECK_NE(a, 0);
        r /= a;
        g /= a;
        b /= a;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    RGB operator/(Float a) const {
        RGB ret = *this;
        return ret /= a;
    }

    PBRT_HOST_DEVICE_INLINE
    RGB operator-() const { return {-r, -g, -b}; }

    PBRT_HOST_DEVICE_INLINE
    Float Average() const { return (r + g + b) / 3; }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const RGB &s) const {
        return r == s.r && g == s.g && b == s.b;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const RGB &s) const {
        return r != s.r || g != s.g || b != s.b;
    }
    PBRT_HOST_DEVICE_INLINE
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0) return r;
        else if (c == 1) return g;
        return b;
    }
    PBRT_HOST_DEVICE_INLINE
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0) return r;
        else if (c == 1) return g;
        return b;
    }

    std::string ToString() const;

    Float r = 0, g = 0, b = 0;
};

template <typename U, typename V>
PBRT_HOST_DEVICE_INLINE
RGB Clamp(const RGB &rgb, U min, V max) {
    return RGB(pbrt::Clamp(rgb.r, min, max),
               pbrt::Clamp(rgb.g, min, max),
               pbrt::Clamp(rgb.b, min, max));
}

PBRT_HOST_DEVICE_INLINE
RGB ClampZero(const RGB &rgb) {
    return RGB(std::max<Float>(0, rgb.r), std::max<Float>(0, rgb.g), std::max<Float>(0, rgb.b));
}

PBRT_HOST_DEVICE_INLINE
RGB Lerp(Float t, const RGB &s1, const RGB &s2) {
    return (1 - t) * s1 + t * s2;
}


class XYZ {
  public:
    XYZ() = default;
    PBRT_HOST_DEVICE_INLINE
    XYZ(Float X, Float Y, Float Z)
        : X(X), Y(Y), Z(Z) {}
    PBRT_HOST_DEVICE_INLINE
    static XYZ FromxyY(Float x, Float y, Float Y = 1) {
        if (y == 0) return XYZ(0, 0, 0);
        return XYZ(x * Y / y, Y, (1 - x - y) * Y / y);
    }

    PBRT_HOST_DEVICE_INLINE
    XYZ &operator+=(const XYZ &s) {
        X += s.X;
        Y += s.Y;
        Z += s.Z;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ operator+(const XYZ &s) const {
        XYZ ret = *this;
        return ret += s;
    }

    PBRT_HOST_DEVICE_INLINE
    XYZ &operator-=(const XYZ &s) {
        X -= s.X;
        Y -= s.Y;
        Z -= s.Z;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ operator-(const XYZ &s) const {
        XYZ ret = *this;
        return ret -= s;
    }
    PBRT_HOST_DEVICE_INLINE
    friend XYZ operator-(Float a, const XYZ &s) {
        return {a - s.X, a - s.Y, a - s.Z};
    }

    PBRT_HOST_DEVICE_INLINE
    XYZ &operator*=(const XYZ &s) {
        X *= s.X;
        Y *= s.Y;
        Z *= s.Z;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ operator*(const XYZ &s) const {
        XYZ ret = *this;
        return ret *= s;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ operator*(Float a) const {
        DCHECK(!std::isnan(a));
        return {a * X, a * Y, a * Z};
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ &operator*=(Float a) {
        DCHECK(!std::isnan(a));
        X *= a;
        Y *= a;
        Z *= a;
        return *this;
    }

    PBRT_HOST_DEVICE_INLINE
    XYZ &operator/=(const XYZ &s) {
        X /= s.X;
        Y /= s.Y;
        Z /= s.Z;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ operator/(const XYZ &s) const {
        XYZ ret = *this;
        return ret /= s;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ &operator/=(Float a) {
        DCHECK(!std::isnan(a));
        DCHECK_NE(a, 0);
        X /= a;
        Y /= a;
        Z /= a;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    XYZ operator/(Float a) const {
        XYZ ret = *this;
        return ret /= a;
    }

    PBRT_HOST_DEVICE_INLINE
    XYZ operator-() const { return {-X, -Y, -Z}; }

    PBRT_HOST_DEVICE_INLINE
    Float Average() const { return (X + Y + Z) / 3; }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const XYZ &s) const {
        return X == s.X && Y == s.Y && Z == s.Z;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const XYZ &s) const {
        return X != s.X || Y != s.Y || Z != s.Z;
    }
    PBRT_HOST_DEVICE_INLINE
    Float operator[](int c) const {
        DCHECK(c >= 0 && c < 3);
        if (c == 0) return X;
        else if (c == 1) return Y;
        return Z;
    }
    PBRT_HOST_DEVICE_INLINE
    Float &operator[](int c) {
        DCHECK(c >= 0 && c < 3);
        if (c == 0) return X;
        else if (c == 1) return Y;
        return Z;
    }

    std::string ToString() const;

    Float X = 0, Y = 0, Z = 0;
};

PBRT_HOST_DEVICE_INLINE
XYZ operator*(Float a, const XYZ &s) {
    return s * a;
}

template <typename U, typename V>
PBRT_HOST_DEVICE_INLINE
XYZ Clamp(const XYZ &xyz, U min, V max) {
    return XYZ(pbrt::Clamp(xyz.X, min, max),
               pbrt::Clamp(xyz.Y, min, max),
               pbrt::Clamp(xyz.Z, min, max));
}

PBRT_HOST_DEVICE_INLINE
XYZ ClampZero(const XYZ &xyz) {
    return XYZ(std::max<Float>(0, xyz.X), std::max<Float>(0, xyz.Y), std::max<Float>(0, xyz.Z));
}

PBRT_HOST_DEVICE_INLINE
XYZ Lerp(Float t, const XYZ &s1, const XYZ &s2) {
    return (1 - t) * s1 + t * s2;
}

class RGBSigmoidPolynomial {
public:
    RGBSigmoidPolynomial() = default;
    PBRT_HOST_DEVICE_INLINE
    RGBSigmoidPolynomial(Float c0, Float c1, Float c2)
        : c0(c0), c1(c1), c2(c2) { }

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        // c2 + c1 * lambda + c0 * lambda^2
        Float v = EvaluatePolynomial(lambda, c2, c1, c0);
        if (std::isinf(v))
            return v > 0 ? 1 : 0;
        return S(v);
    }

    PBRT_HOST_DEVICE_INLINE
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
    PBRT_HOST_DEVICE_INLINE
    static Float S(Float x) { return .5 + x / (2 * std::sqrt(1 + x * x)); };

    Float c0, c1, c2;
};

class RGBToSpectrumTable {
public:
    RGBToSpectrumTable(int res, const float *scale, const float *data)
        : res(res), scale(scale), data(data) {}

    PBRT_HOST_DEVICE
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


class ColorEncoding {
 public:
    PBRT_HOST_DEVICE
    virtual ~ColorEncoding();
    PBRT_HOST_DEVICE
    virtual void ToLinear(pstd::span<const uint8_t> vin,
                          pstd::span<Float> vout) const = 0;
    PBRT_HOST_DEVICE
    virtual Float ToFloatLinear(Float v) const = 0;
    PBRT_HOST_DEVICE
    virtual void FromLinear(pstd::span<const Float> vin,
                            pstd::span<uint8_t> vout) const = 0;

    virtual std::string ToString() const = 0;

    static const ColorEncoding *Linear;
    static const ColorEncoding *sRGB;

    static const ColorEncoding *Get(const std::string &name);
};

class LinearColorEncoding : public ColorEncoding {
 public:
    PBRT_HOST_DEVICE_INLINE
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = vin[i] / 255.f;
    }

    PBRT_HOST_DEVICE_INLINE
    Float ToFloatLinear(Float v) const { return v; }

    PBRT_HOST_DEVICE_INLINE
    void FromLinear(pstd::span<const Float> vin,
                    pstd::span<uint8_t> vout) const {
        DCHECK_EQ(vin.size(), vout.size());
        for (size_t i = 0; i < vin.size(); ++i)
            vout[i] = uint8_t(Clamp(vin[i] * 255.f + 0.5f, 0, 255));
    }

    std::string ToString() const { return "[ LinearColorEncoding ]"; }
};

class sRGBColorEncoding : public ColorEncoding {
 public:
    PBRT_HOST_DEVICE
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const;
    PBRT_HOST_DEVICE
    Float ToFloatLinear(Float v) const;
    PBRT_HOST_DEVICE
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const;

    std::string ToString() const { return "[ sRGBColorEncoding ]"; }
};

class GammaColorEncoding : public ColorEncoding {
 public:
    PBRT_HOST_DEVICE
    GammaColorEncoding(Float gamma);

    PBRT_HOST_DEVICE
    void ToLinear(pstd::span<const uint8_t> vin, pstd::span<Float> vout) const;
    PBRT_HOST_DEVICE
    Float ToFloatLinear(Float v) const;
    PBRT_HOST_DEVICE
    void FromLinear(pstd::span<const Float> vin, pstd::span<uint8_t> vout) const;

    std::string ToString() const;

 private:
    Float gamma;
    pstd::array<Float, 256> applyLUT;
    pstd::array<Float, 1024> inverseLUT;
};

PBRT_HOST_DEVICE_INLINE
Float LinearToSRGBFull(Float value) {
    if (value <= 0.0031308f) return 12.92f * value;
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

PBRT_HOST_DEVICE_INLINE
Float LinearToSRGB(Float value) {
    int index = int(value * LinearToSRGBPiecewiseSize);
    if (index < 0) return 0;
    if (index >= LinearToSRGBPiecewiseSize) return 1;
    return LinearToSRGBPiecewise[index].base + value * LinearToSRGBPiecewise[index].slope;
}

PBRT_HOST_DEVICE_INLINE
uint8_t LinearToSRGB8(Float value, Float dither = 0) {
    if (value <= 0) return 0;
    if (value >= 1) return 255;
    return Clamp(255.f * LinearToSRGB(value) + dither, 0, 255);
}

PBRT_HOST_DEVICE_INLINE
Float SRGBToLinear(Float value) {
    if (value <= 0.04045f) return value * (1 / 12.92f);
    return std::pow((value + 0.055f) * (1 / 1.055f), (Float)2.4f);
}

extern PBRT_CONST Float SRGBToLinearLUT[256];

PBRT_HOST_DEVICE_INLINE
Float SRGB8ToLinear(uint8_t value) {
    return SRGBToLinearLUT[value];
}

}  // namespace pbrt

#endif  // PBRT_COLOR_H
