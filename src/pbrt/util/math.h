
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

#ifndef PBRT_UTIL_MATH_H
#define PBRT_UTIL_MATH_H

// util/math.h*
#include <pbrt/core/pbrt.h>

#include <absl/types/span.h>
#include <glog/logging.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#ifdef PBRT_IS_MSVC
#include <intrin.h>
#endif  // PBRT_IS_MSVC

namespace pbrt {

// Global Constants
static constexpr Float MaxFloat = std::numeric_limits<Float>::max();
static constexpr Float Infinity = std::numeric_limits<Float>::infinity();
static constexpr Float MachineEpsilon =
    std::numeric_limits<Float>::epsilon() * 0.5;
static constexpr Float ShadowEpsilon = 0.0001f;
static constexpr Float Pi = 3.14159265358979323846;
static constexpr Float InvPi = 0.31830988618379067154;
static constexpr Float Inv2Pi = 0.15915494309189533577;
static constexpr Float Inv4Pi = 0.07957747154594766788;
static constexpr Float PiOver2 = 1.57079632679489661923;
static constexpr Float PiOver4 = 0.78539816339744830961;
static constexpr Float Sqrt2 = 1.41421356237309504880;

// Global Inline Functions
inline uint32_t FloatToBits(float f) {
    uint32_t ui;
    std::memcpy(&ui, &f, sizeof(float));
    return ui;
}

inline float BitsToFloat(uint32_t ui) {
    float f;
    std::memcpy(&f, &ui, sizeof(uint32_t));
    return f;
}

inline uint64_t FloatToBits(double f) {
    uint64_t ui;
    std::memcpy(&ui, &f, sizeof(double));
    return ui;
}

inline double BitsToFloat(uint64_t ui) {
    double f;
    std::memcpy(&f, &ui, sizeof(uint64_t));
    return f;
}

inline float NextFloatUp(float v) {
    // Handle infinity and negative zero for _NextFloatUp()_
    if (std::isinf(v) && v > 0.) return v;
    if (v == -0.f) v = 0.f;

    // Advance _v_ to next higher float
    uint32_t ui = FloatToBits(v);
    if (v >= 0)
        ++ui;
    else
        --ui;
    return BitsToFloat(ui);
}

inline float NextFloatDown(float v) {
    // Handle infinity and positive zero for _NextFloatDown()_
    if (std::isinf(v) && v < 0.) return v;
    if (v == 0.f) v = -0.f;
    uint32_t ui = FloatToBits(v);
    if (v > 0)
        --ui;
    else
        ++ui;
    return BitsToFloat(ui);
}

inline double NextFloatUp(double v, int delta = 1) {
    if (std::isinf(v) && v > 0.) return v;
    if (v == -0.f) v = 0.f;
    uint64_t ui = FloatToBits(v);
    if (v >= 0.)
        ui += delta;
    else
        ui -= delta;
    return BitsToFloat(ui);
}

inline double NextFloatDown(double v, int delta = 1) {
    if (std::isinf(v) && v < 0.) return v;
    if (v == 0.f) v = -0.f;
    uint64_t ui = FloatToBits(v);
    if (v > 0.)
        ui -= delta;
    else
        ui += delta;
    return BitsToFloat(ui);
}

inline constexpr Float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

template <typename T, typename U, typename V>
inline constexpr T Clamp(T val, U low, V high) {
    if (val < low)
        return low;
    else if (val > high)
        return high;
    else
        return val;
}

template <typename T>
inline T Mod(T a, T b) {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

template <>
inline Float Mod(Float a, Float b) {
    return std::fmod(a, b);
}

inline Float Radians(Float deg) { return (Pi / 180) * deg; }

inline Float Degrees(Float rad) { return (180 / Pi) * rad; }

inline Float Log2(Float x) {
    const Float invLog2 = 1.442695040888963387004650940071;
    return std::log(x) * invLog2;
}

inline int Log2Int(uint32_t v) {
#if defined(PBRT_IS_MSVC)
    unsigned long lz = 0;
    if (_BitScanReverse(&lz, v)) return lz;
    return 0;
#else
    return 31 - __builtin_clz(v);
#endif
}

inline int Log2Int(int32_t v) { return Log2Int((uint32_t)v); }

inline int Log2Int(uint64_t v) {
#if defined(PBRT_IS_MSVC)
    unsigned long lz = 0;
#if defined(_WIN64)
    _BitScanReverse64(&lz, v);
#else
    if (_BitScanReverse(&lz, v >> 32))
        lz += 32;
    else
        _BitScanReverse(&lz, v & 0xffffffff);
#endif  // _WIN64
    return lz;
#else  // PBRT_IS_MSVC
    return 63 - __builtin_clzll(v);
#endif
}

inline int Log2Int(int64_t v) { return Log2Int((uint64_t)v); }

template <typename T>
inline constexpr bool IsPowerOf2(T v) {
    return v && !(v & (v - 1));
}

inline constexpr int32_t RoundUpPow2(int32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return v + 1;
}

inline constexpr int64_t RoundUpPow2(int64_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

template <typename Predicate>
size_t FindInterval(size_t size, const Predicate &pred) {
    size_t first = 0, len = size;
    while (len > 0) {
        size_t half = len >> 1, middle = first + half;
        // Bisect range based on value of _pred_ at _middle_
        if (pred(middle)) {
            first = middle + 1;
            len -= half + 1;
        } else
            len = half;
    }
    return Clamp(first - 1, 0, size - 2);
}

inline constexpr Float Lerp(Float t, Float v1, Float v2) {
    return (1 - t) * v1 + t * v2;
}

// (0,0): v[0], (1, 0): v[1], (0, 1): v[2], (1, 1): v[3]
inline Float
Bilerp(std::array<Float, 2> p, absl::Span<const Float> v) {
    return ((1 - p[0]) * (1 - p[1]) * v[0] +
                 p[0]  * (1 - p[1]) * v[1] +
            (1 - p[0]) *      p[1]  * v[2] +
                 p[0]  *      p[1]  * v[3]);
}

template <typename T>
inline constexpr T Sqr(T v) { return v * v; }

inline float SafeASin(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}

inline double SafeASin(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::asin(Clamp(x, -1, 1));
}

inline float SafeACos(float x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

inline double SafeACos(double x) {
    DCHECK(x >= -1.0001 && x <= 1.0001);
    return std::acos(Clamp(x, -1, 1));
}

inline float SafeSqrt(float x) {
    DCHECK_GE(x, -1e-4f);  // not too negative
    return std::sqrt(std::max(0.f, x));
}

inline double SafeSqrt(double x) {
    DCHECK_GE(x, -1e-4);  // not too negative
    return std::sqrt(std::max(0., x));
}

// Would be nice to allow Float to be a template type here, but it's tricky:
// https://stackoverflow.com/questions/5101516/why-function-template-cannot-be-partially-specialized
template <int n>
inline constexpr float Pow(float v) {
    static_assert(n > 0, "Power can't be negative");
    float n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
inline constexpr float Pow<1>(float v) {
    return v;
}

template <>
inline constexpr float Pow<0>(float v) {
    return 1;
}

template <int n>
inline constexpr double Pow(double v) {
    static_assert(n > 0, "Power can't be negative");
    double n2 = Pow<n / 2>(v);
    return n2 * n2 * Pow<n & 1>(v);
}

template <>
inline constexpr double Pow<1>(double v) {
    return v;
}

template <>
inline constexpr double Pow<0>(double v) {
    return 1;
}

inline bool Quadratic(Float a, Float b, Float c, Float *t0, Float *t1) {
    // Find quadratic discriminant
    double discrim = (double)b * (double)b - 4 * (double)a * (double)c;
    if (discrim < 0) return false;
    double rootDiscrim = std::sqrt(discrim);

    // Compute quadratic _t_ values
    double q;
    if (b < 0)
        q = -.5 * (b - rootDiscrim);
    else
        q = -.5 * (b + rootDiscrim);
    *t0 = q / a;
    *t1 = c / q;
    if (*t0 > *t1) std::swap(*t0, *t1);
    return true;
}

inline Float ErfInv(Float x) {
    Float w, p;
    x = Clamp(x, -.99999f, .99999f);
    w = -std::log((1 - x) * (1 + x));
    if (w < 5) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = std::sqrt(w) - 3;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

inline Float Erf(Float x) {
    // constants
    Float a1 = 0.254829592f;
    Float a2 = -0.284496736f;
    Float a3 = 1.421413741f;
    Float a4 = -1.453152027f;
    Float a5 = 1.061405429f;
    Float p = 0.3275911f;

    // Save the sign of x
    int sign = 1;
    if (x < 0) sign = -1;
    x = std::abs(x);

    // A&S formula 7.1.26
    Float t = 1 / (1 + p * x);
    Float y =
        1 -
        (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return sign * y;
}

// TODO: move to image.h?

inline Float LinearToSRGBFull(Float value) {
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

extern const PiecewiseLinearSegment LinearToSRGBPiecewise[];
constexpr int LinearToSRGBPiecewiseSize = 256;

inline Float LinearToSRGB(Float value) {
    int index = int(value * LinearToSRGBPiecewiseSize);
    if (index < 0) return 0;
    if (index >= LinearToSRGBPiecewiseSize) return 1;
    return LinearToSRGBPiecewise[index].base + value * LinearToSRGBPiecewise[index].slope;
}

inline uint8_t LinearToSRGB8(Float value) {
    return Clamp(255.f * LinearToSRGB(value) + 0.5f, 0, 255);
}

inline Float SRGBToLinear(Float value) {
    if (value <= 0.04045f) return value * (1 / 12.92f);
    return std::pow((value + 0.055f) * (1 / 1.055f), (Float)2.4f);
}

extern const Float SRGBToLinearLUT[256];

inline Float SRGB8ToLinear(uint8_t value) {
    return SRGBToLinearLUT[value];
}

inline Float Sinc(Float x) {
    // http://www.plunk.org/~hatch/rightway.php
    // The power series expansion of sin(x)/x is 1 - x^2/3! + x^4/5! - ...
    // So if x is small and 1 - x^2/3! rounds to 1, then sin(x)/x will
    // also round to one. Making the test a bit more conservative, as below,
    // works as well.
    if (1 + Pow<2>(Pi * x) == 1)
        return 1;
    return std::sin(Pi * x) / (Pi * x);
}

inline Float WindowedSinc(Float x, Float radius, Float tau) {
    if (std::abs(x) > radius) return 0;
    Float lanczos = Sinc(x / tau);
    return Sinc(x) * lanczos;
}

// Solve f[x] == 0 over [x0, x1]. f should return a std::pair<FloatType,
// FloatType>[f(x), f'(x)]. Only enabled for float and double.
template <typename FloatType, typename Func>
FloatType NewtonBisection(
    FloatType x0, FloatType x1, Func f, Float xEps = 1e-6f, Float fEps = 1e-6f,
    typename std::enable_if<std::is_floating_point<FloatType>::value>::type * =
        nullptr) {
    CHECK_LT(x0, x1);
    Float fx0 = f(x0).first, fx1 = f(x1).first;
    if (std::abs(fx0) < fEps) return x0;
    if (std::abs(fx1) < fEps) return x1;
    bool startIsNegative = fx0 < 0;
    // Implicit line equation: (y-y0)/(y1-y0) = (x-x0)/(x1-x0).
    // Solve for y = 0 to find x for starting point.
    FloatType xMid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0);

    while (true) {
        if (!(x0 < xMid && xMid < x1))
            // Fall back to bisection.
            xMid = (x0 + x1) / 2;

        Float fx0 = f(x0).first, fx1 = f(x1).first;
        std::pair<FloatType, FloatType> fxMid = f(xMid);
        DCHECK(!std::isnan(fxMid.first));

        if (startIsNegative == fxMid.first < 0)
            x0 = xMid;
        else
            x1 = xMid;

        if ((x1 - x0) < xEps || std::abs(fxMid.first) < fEps) return xMid;

        // Try a Newton step.
        xMid -= fxMid.first / fxMid.second;
    }
}

// If an integral type is used for x0 and x1, assume Float.
template <typename NotFloatType, typename Func>
Float NewtonBisection(
    NotFloatType x0, NotFloatType x1, Func f, Float xEps = 1e-6f,
    Float fEps = 1e-6f,
    typename std::enable_if<std::is_integral<NotFloatType>::value>::type * =
        nullptr) {
    return NewtonBisection(Float(x0), Float(x1), f, xEps, fEps);
}

template <typename Float, typename C>
inline constexpr Float EvaluatePolynomial(Float t, C c) {
    return c;
}

template <typename Float, typename C, typename ...Args>
inline constexpr Float EvaluatePolynomial(Float t, C c, Args... cRemaining) {
    return c + t * EvaluatePolynomial(t, cRemaining...);
}

inline Float Smoothstep(Float x, Float a, Float b) {
    CHECK_LT(a, b);
    Float t = Clamp((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

inline Float I0(Float x) {
    Float val = 0;
    Float x2i = 1;
    int64_t ifact = 1;
    int i4 = 1;
    // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
    for (int i = 0; i < 10; ++i) {
        if (i > 1) ifact *= i;
        val += x2i / (i4 * Sqr(ifact));
        x2i *= x * x;
        i4 *= 4;
    }
    return val;
}

inline Float LogI0(Float x) {
    if (x > 12)
        return x + 0.5 * (-std::log(2 * Pi) + std::log(1 / x) + 1 / (8 * x));
    else
        return std::log(I0(x));
}

inline Float Logistic(Float x, Float s) {
    x = std::abs(x);
    return std::exp(-x / s) / (s * Sqr(1 + std::exp(-x / s)));
}

inline Float LogisticCDF(Float x, Float s) {
    return 1 / (1 + std::exp(-x / s));
}

inline Float TrimmedLogistic(Float x, Float s, Float a, Float b) {
    CHECK_LT(a, b);
    return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

template <typename Float = double>
class KahanSum {
  public:
    KahanSum() = default;
    explicit KahanSum(Float v) : sum(v) { }

    KahanSum &operator=(Float v) { sum = v; c = 0; return *this; }
    KahanSum &operator+=(Float v) {
        Float delta = v - c;
        Float newSum = sum + delta;
        c = (newSum - sum) - delta;
        sum = newSum;
        return *this;
    }

    explicit operator Float() const { return sum; }

  private:
    Float sum = 0., c = 0.;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MATH_H
