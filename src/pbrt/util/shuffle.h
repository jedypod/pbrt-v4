// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_SHUFFLE_H
#define PBRT_UTIL_SHUFFLE_H

// sampling/shuffle.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>

#include <algorithm>
#include <cstdint>

namespace pbrt {

// Returns the random permutation of the i'th out of n elements,
// using the given seed |p|.
PBRT_CPU_GPU
inline int PermutationElement(uint32_t i, uint32_t l, uint32_t p) {
    uint32_t w = l - 1;
    w |= w >> 1;
    w |= w >> 2;
    w |= w >> 4;
    w |= w >> 8;
    w |= w >> 16;
    do {
        i ^= p;
        i *= 0xe170893d;
        i ^= p >> 16;
        i ^= (i & w) >> 4;
        i ^= p >> 8;
        i *= 0x0929eb3f;
        i ^= p >> 23;
        i ^= (i & w) >> 1;
        i *= 1 | p >> 27;
        i *= 0x6935fa69;
        i ^= (i & w) >> 11;
        i *= 0x74dcb303;
        i ^= (i & w) >> 2;
        i *= 0x9e501cc3;
        i ^= (i & w) >> 2;
        i *= 0xc860a3df;
        i &= w;
        i ^= i >> 5;
    } while (i >= l);
    return (i + p) % l;
}

template <typename T>
PBRT_CPU_GPU inline void Shuffle(pstd::span<T> values, RNG &rng) {
    for (size_t i = 0; i < values.size(); ++i) {
        size_t other = i + rng.Uniform<uint32_t>(values.size() - i);
        pstd::swap(values[i], values[other]);
    }
}

}  // namespace pbrt

#endif  // PBRT_UTIL_SHUFFLE_H
