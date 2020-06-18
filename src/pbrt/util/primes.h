// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_PRIMES_H
#define PBRT_UTIL_PRIMES_H

// sampling/primes.h*

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

namespace pbrt {

static constexpr int PrimeTableSize = 1000;
extern const pstd::array<int, PrimeTableSize> Primes;
extern const pstd::array<int, PrimeTableSize> PrimeSums;

}  // namespace pbrt

#endif  // PBRT_UTIL_PRIMES_H
