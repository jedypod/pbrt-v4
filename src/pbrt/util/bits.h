
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

#ifndef PBRT_UTIL_BITS_H
#define PBRT_UTIL_BITS_H

// util/bits.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>

#include <cstdint>

#ifdef PBRT_HAS_INTRIN_H
#include <intrin.h>
#endif  // PBRT_HAS_INTRIN_H

namespace pbrt {

PBRT_HOST_DEVICE_INLINE
uint32_t ReverseBits32(uint32_t n) {
#ifdef __CUDA_ARCH__
    return __brev(n);
#else
    n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
    n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
    return n;
#endif
}

PBRT_HOST_DEVICE_INLINE
uint64_t ReverseBits64(uint64_t n) {
#ifdef __CUDA_ARCH__
    return __brevll(n);
#else
    uint64_t n0 = ReverseBits32((uint32_t)n);
    uint64_t n1 = ReverseBits32((uint32_t)(n >> 32));
    return (n0 << 32) | n1;
#endif
}

PBRT_HOST_DEVICE_INLINE
uint32_t GrayCode(uint32_t v) { return (v >> 1) ^ v; }

PBRT_HOST_DEVICE_INLINE
int CountTrailingZeros(uint32_t v) {
#ifdef __CUDA_ARCH__
    return __ffs(v);
#elif defined(PBRT_HAS_INTRIN_H)
    unsigned long index;
    if (_BitScanForward(&index, v))
        return index;
    else
        return 32;
#else
    return __builtin_ctz(v);
#endif
}

PBRT_HOST_DEVICE_INLINE
int CountTrailingZeros(uint64_t v) {
#ifdef __CUDA_ARCH__
    return __ffsll(v);
#elif defined(PBRT_HAS_INTRIN_H)
    unsigned long index;
    if (_BitScanForward64(&index, v))
        return index;
    else
        return 64;
#else
    return __builtin_ctzl(v);
#endif
}

PBRT_HOST_DEVICE_INLINE
int PopCount(uint32_t v) {
#ifdef __CUDA_ARCH__
    return __popc(v);
#elif defined(PBRT_HAS_INTRIN_H)
    return __popcnt(v);
#else
    return __builtin_popcount(v);
#endif
}

PBRT_HOST_DEVICE_INLINE
int PopCount(uint64_t v) {
#ifdef __CUDA_ARCH__
    return __popcll(v);
#elif defined(PBRT_HAS_INTRIN_H)
    return __popcnt64(v);
#else
    return __builtin_popcountll(v);
#endif
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
// "Insert" a 0 bit after each of the 16 low bits of x
PBRT_HOST_DEVICE_INLINE
uint32_t LeftShift2(uint32_t x) {
    x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

PBRT_HOST_DEVICE_INLINE
uint32_t EncodeMorton2(uint16_t x, uint16_t y) {
    return (LeftShift2(y) << 1) | LeftShift2(x);
}

PBRT_HOST_DEVICE_INLINE
uint32_t LeftShift3(uint32_t x) {
    DCHECK_LE(x, (1u << 10));
    if (x == (1 << 10)) --x;
    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x << 8)) & 0b00000011000000001111000000001111;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x << 4)) & 0b00000011000011000011000011000011;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x << 2)) & 0b00001001001001001001001001001001;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    return x;
}

PBRT_HOST_DEVICE_INLINE
uint32_t EncodeMorton3(float x, float y, float z) {
    DCHECK_GE(x, 0);
    DCHECK_GE(y, 0);
    DCHECK_GE(z, 0);
    return (LeftShift3(z) << 2) | (LeftShift3(y) << 1) | LeftShift3(x);
}

PBRT_HOST_DEVICE_INLINE
uint32_t Compact1By1(uint32_t x) {
    // TODO: as of Haswell, the PEXT instruction could do all this in a
    // single instruction.
    // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x &= 0x55555555;
    // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 1)) & 0x33333333;
    // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 2)) & 0x0f0f0f0f;
    // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 4)) & 0x00ff00ff;
    // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x >> 8)) & 0x0000ffff;
    return x;
}

PBRT_HOST_DEVICE_INLINE
void DecodeMorton2(uint32_t v, uint16_t *x, uint16_t *y) {
    *x = Compact1By1(v);
    *y = Compact1By1(v >> 1);
}

PBRT_HOST_DEVICE_INLINE
uint32_t Compact1By2(uint32_t x) {
    x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >>  2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >>  4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >>  8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    return x;
}

// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
PBRT_HOST_DEVICE_INLINE
uint64_t MixBits(uint64_t v) {
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_BITS_H
