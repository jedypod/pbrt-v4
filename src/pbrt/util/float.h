
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

#ifndef PBRT_UTIL_FLOAT_H
#define PBRT_UTIL_FLOAT_H

// util/float.h*
#include <pbrt/pbrt.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

namespace pbrt {

#ifdef __CUDA_ARCH__

#define DoubleOneMinusEpsilon 0x1.fffffffffffffp-1
#define FloatOneMinusEpsilon float(0x1.fffffep-1)

#ifdef PBRT_FLOAT_IS_DOUBLE
#define OneMinusEpsilon  DoubleOneMinusEpsilon
#else
#define OneMinusEpsilon FloatOneMinusEpsilon
#endif

#define MaxFloat std::numeric_limits<Float>::max()
#define Infinity std::numeric_limits<Float>::infinity()
#define MachineEpsilon std::numeric_limits<Float>::epsilon() * 0.5f

#else

#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
static const double DoubleOneMinusEpsilon = 0.99999999999999989;
static const float FloatOneMinusEpsilon = 0.99999994;
#else
static const double DoubleOneMinusEpsilon = 0x1.fffffffffffffp-1;
static const float FloatOneMinusEpsilon = 0x1.fffffep-1;
#endif

#ifdef PBRT_FLOAT_IS_DOUBLE
static const double OneMinusEpsilon = DoubleOneMinusEpsilon;
#else
static const float OneMinusEpsilon = FloatOneMinusEpsilon;
#endif

static constexpr Float MaxFloat = std::numeric_limits<Float>::max();
static constexpr Float Infinity = std::numeric_limits<Float>::infinity();
static constexpr Float MachineEpsilon =
    std::numeric_limits<Float>::epsilon() * 0.5;

#endif // __CUDA_ARCH__

PBRT_HOST_DEVICE_INLINE
uint32_t FloatToBits(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    uint32_t ui;
    std::memcpy(&ui, &f, sizeof(float));
    return ui;
#endif
}

PBRT_HOST_DEVICE_INLINE
float BitsToFloat(uint32_t ui) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(ui);
#else
    float f;
    std::memcpy(&f, &ui, sizeof(uint32_t));
    return f;
#endif
}

PBRT_HOST_DEVICE_INLINE
uint64_t FloatToBits(double f) {
#ifdef __CUDA_ARCH__
    return __double_as_longlong(f);
#else
    uint64_t ui;
    std::memcpy(&ui, &f, sizeof(double));
    return ui;
#endif
}

PBRT_HOST_DEVICE_INLINE
double BitsToFloat(uint64_t ui) {
#ifdef __CUDA_ARCH__
    return __longlong_as_double(ui);
#else
    double f;
    std::memcpy(&f, &ui, sizeof(uint64_t));
    return f;
#endif
}

PBRT_HOST_DEVICE_INLINE
float NextFloatUp(float v, int delta = 1) {
    // Handle infinity and negative zero for _NextFloatUp()_
    if (std::isinf(v) && v > 0.) return v;
    if (v == -0.f) v = 0.f;

    // Advance _v_ to next higher float
    uint32_t ui = FloatToBits(v);
    if (v >= 0)
        ui += delta;
    else
        ui -= delta;
    return BitsToFloat(ui);
}

PBRT_HOST_DEVICE_INLINE
float NextFloatDown(float v, int delta = 1) {
    // Handle infinity and positive zero for _NextFloatDown()_
    if (std::isinf(v) && v < 0.) return v;
    if (v == 0.f) v = -0.f;
    uint32_t ui = FloatToBits(v);
    if (v > 0)
        ui -= delta;
    else
        ui += delta;
    return BitsToFloat(ui);
}

PBRT_HOST_DEVICE_INLINE
double NextFloatUp(double v, int delta = 1) {
    if (std::isinf(v) && v > 0.) return v;
    if (v == -0.f) v = 0.f;
    uint64_t ui = FloatToBits(v);
    if (v >= 0.)
        ui += delta;
    else
        ui -= delta;
    return BitsToFloat(ui);
}

PBRT_HOST_DEVICE_INLINE
double NextFloatDown(double v, int delta = 1) {
    if (std::isinf(v) && v < 0.) return v;
    if (v == 0.f) v = -0.f;
    uint64_t ui = FloatToBits(v);
    if (v > 0.)
        ui -= delta;
    else
        ui += delta;
    return BitsToFloat(ui);
}

PBRT_HOST_DEVICE_INLINE
int Exponent(float v) {
    return (FloatToBits(v) >> 23) - 127;
}

PBRT_HOST_DEVICE_INLINE
int Significand(float v) {
    return FloatToBits(v) & ((1 << 23) - 1);
}

PBRT_HOST_DEVICE_INLINE
int Exponent(double d) {
    return (FloatToBits(d) >> 52) - 1023;
}

PBRT_HOST_DEVICE_INLINE
uint64_t Significand(double d) {
    return FloatToBits(d) & ((1ull << 52) - 1);
}

PBRT_HOST_DEVICE_INLINE
uint32_t SignBit(float v) {
    return FloatToBits(v) & 0x80000000;
}

PBRT_HOST_DEVICE_INLINE
uint64_t SignBit(double v) {
    return FloatToBits(v) & 0x8000000000000000;
}

// Return a float with |a|'s magnitude, but negated if |b| is negative.
PBRT_HOST_DEVICE_INLINE
float FlipSign(float a, float b) {
    return BitsToFloat(FloatToBits(a) ^ SignBit(b));
}

PBRT_HOST_DEVICE_INLINE
double FlipSign(double a, double b) {
    return BitsToFloat(FloatToBits(a) ^ SignBit(b));
}

PBRT_HOST_DEVICE_INLINE
constexpr Float gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

static const int HalfExponentMask = 0b0111110000000000;
static const int HalfSignificandMask = 0b1111111111;
static const int HalfNegativeZero = 0b1000000000000000;
static const int HalfPositiveZero = 0;
// Exponent all 1s, significand zero
static const int HalfNegativeInfinity = 0b1111110000000000;
static const int HalfPositiveInfinity = 0b0111110000000000;

namespace {

// TODO: support for non-AVX systems, check CPUID stuff, etc..

// https://gist.github.com/rygorous/2156668
union FP32 {
    uint32_t u;
    float f;
    struct {
        unsigned int Mantissa : 23;
        unsigned int Exponent : 8;
        unsigned int Sign : 1;
    };
};

union FP16 {
    uint16_t u;
    struct {
        unsigned int Mantissa : 10;
        unsigned int Exponent : 5;
        unsigned int Sign : 1;
    };
};

} // namespace

class Half {
 public:
    Half() = default;
    Half(const Half &) = default;
    Half &operator=(const Half &) = default;

    PBRT_HOST_DEVICE_INLINE
    static Half FromBits(uint16_t v) { return Half(v); }

    explicit Half(float ff) {
        // Rounding ties to nearest even instead of towards +inf
        FP32 f;
        f.f = ff;
        FP32 f32infty = {255 << 23};
        FP32 f16max = {(127 + 16) << 23};
        FP32 denorm_magic = {((127 - 15) + (23 - 10) + 1) << 23};
        unsigned int sign_mask = 0x80000000u;
        FP16 o = {0};

        unsigned int sign = f.u & sign_mask;
        f.u ^= sign;

        // NOTE all the integer compares in this function can be safely
        // compiled into signed compares since all operands are below
        // 0x80000000. Important if you want fast straight SSE2 code
        // (since there's no unsigned PCMPGTD).

        if (f.u >= f16max.u)  // result is Inf or NaN (all exponent bits set)
            o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
        else  { // (De)normalized number or zero
            if (f.u < (113 << 23))  { // resulting FP16 is subnormal or zero
                // use a magic value to align our 10 mantissa bits at the bottom of
                // the float. as long as FP addition is round-to-nearest-even this
                // just works.
                f.f += denorm_magic.f;

                // and one integer subtract of the bias later, we have our final
                // float!
                o.u = f.u - denorm_magic.u;
            } else {
                unsigned int mant_odd = (f.u >> 13) & 1;  // resulting mantissa is odd

                // update exponent, rounding bias part 1
                f.u += (uint32_t(15 - 127) << 23) + 0xfff;
                // rounding bias part 2
                f.u += mant_odd;
                // take the bits!
                o.u = f.u >> 13;
            }
        }

        o.u |= sign >> 16;
        h = o.u;
    }
    explicit Half(double d) : Half(float(d)) {}

    explicit operator float() const {
        FP16 h;
        h.u = this->h;
        static const FP32 magic = {113 << 23};
        static const unsigned int shifted_exp = 0x7c00 << 13;  // exponent mask after shift
        FP32 o;

        o.u = (h.u & 0x7fff) << 13;    // exponent/mantissa bits
        unsigned int exp = shifted_exp & o.u;  // just the exponent
        o.u += (127 - 15) << 23;       // exponent adjust

        // handle exponent special cases
        if (exp == shifted_exp)       // Inf/NaN?
            o.u += (128 - 16) << 23;  // extra exp adjust
        else if (exp == 0) {            // Zero/Denormal?
            o.u += 1 << 23;  // extra exp adjust
            o.f -= magic.f;  // renormalize
        }

        o.u |= (h.u & 0x8000) << 16;  // sign bit
        return o.f;
    }
    explicit operator double() const {
        return (float)(*this);
    }

    bool operator==(const Half &v) const {
        if (Bits() == v.Bits()) return true;
        return ((Bits() == HalfNegativeZero && v.Bits() == HalfPositiveZero) ||
                (Bits() == HalfPositiveZero && v.Bits() == HalfNegativeZero));
    }
    bool operator!=(const Half &v) const {
        return !(*this == v);
    }

    Half operator-() const { return FromBits(h ^ (1 << 15)); }

    PBRT_HOST_DEVICE_INLINE
    uint16_t Bits() const { return h; }

    int Sign() { return (h >> 15) ? -1 : 1; }
    bool IsInf() {
        return h == HalfPositiveInfinity || h == HalfNegativeInfinity;
    }
    bool IsNaN() {
        return ((h & HalfExponentMask) == HalfExponentMask &&
                (h & HalfSignificandMask) != 0);
    }
    Half NextUp() {
        if (IsInf() && Sign() == 1) return *this;

        Half up = *this;
        if (up.h == HalfNegativeZero) up.h = HalfPositiveZero;
        // Advance _v_ to next higher float
        if (up.Sign() >= 0)
            ++up.h;
        else
            --up.h;
        return up;
    }
    Half NextDown() {
        if (IsInf() && Sign() == -1) return *this;

        Half down = *this;
        if (down.h == HalfPositiveZero) down.h = HalfNegativeZero;
        if (down.Sign() >= 0)
            --down.h;
        else
            ++down.h;
        return down;
    }

    std::string ToString() const;

 private:
    PBRT_HOST_DEVICE_INLINE
    explicit Half(uint16_t h) : h(h) { }
    uint16_t h;
};

}  // namespace pbrt

#endif  // PBRT_FLOAT_H
