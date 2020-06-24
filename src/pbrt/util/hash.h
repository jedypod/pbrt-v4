
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

#ifndef PBRT_UTIL_HASH_H
#define PBRT_UTIL_HASH_H

#include <pbrt/pbrt.h>

namespace pbrt {

// https://github.com/explosion/murmurhash/blob/master/murmurhash/MurmurHash2.cpp
PBRT_HOST_DEVICE_INLINE
uint64_t MurmurHash64A(const void *key, int len, uint64_t seed) {
    const uint64_t m = 0xc6a4a7935bd1e995ull;
    const int r = 47;

    uint64_t h = seed ^ (len * m);

    const uint64_t *data = (const uint64_t *)key;
    const uint64_t *end = data + (len/8);

    while (data != end) {
        uint64_t k = *data++;

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const unsigned char *data2 = (const unsigned char*)data;

    switch(len & 7) {
    case 7: h ^= uint64_t(data2[6]) << 48;
    case 6: h ^= uint64_t(data2[5]) << 40;
    case 5: h ^= uint64_t(data2[4]) << 32;
    case 4: h ^= uint64_t(data2[3]) << 24;
    case 3: h ^= uint64_t(data2[2]) << 16;
    case 2: h ^= uint64_t(data2[1]) << 8;
    case 1: h ^= uint64_t(data2[0]);
    h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}

PBRT_HOST_DEVICE_INLINE
uint64_t HashBuffer(const void *ptr, size_t size, uint64_t seed = 0) {
    return MurmurHash64A(ptr, size, seed);
}

template <size_t size>
PBRT_HOST_DEVICE_INLINE
uint64_t HashBuffer(const void *ptr, uint64_t seed = 0) {
    return MurmurHash64A(ptr, size, seed);
}

template <typename... Args>
PBRT_HOST_DEVICE_INLINE
uint64_t hashInternal(uint64_t hash, Args...);

template <>
PBRT_HOST_DEVICE_INLINE
uint64_t hashInternal(uint64_t hash) { return hash; }

template <typename T, typename... Args>
PBRT_HOST_DEVICE_INLINE
uint64_t hashInternal(uint64_t hash, T v, Args... args) {
    return MurmurHash64A(&v, sizeof(v), hashInternal(hash, args...));
}

template <typename... Args>
PBRT_HOST_DEVICE_INLINE
uint64_t Hash(Args... args) {
    return hashInternal(0, args...);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_HASH_H
