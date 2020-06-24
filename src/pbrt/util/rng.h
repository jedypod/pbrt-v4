
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

#ifndef PBRT_UTIL_RNG_H
#define PBRT_UTIL_RNG_H

// util/rng.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace pbrt {

// Random Number Declarations
#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

class RNG {
  public:
    // RNG Public Methods
    PBRT_HOST_DEVICE_INLINE
    RNG();
    PBRT_HOST_DEVICE_INLINE
    RNG(uint64_t sequenceIndex) { SetSequence(sequenceIndex); }
    PBRT_HOST_DEVICE_INLINE
    void SetSequence(uint64_t sequenceIndex);

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    T Uniform();

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    T Uniform(
        T b, typename std::enable_if_t<std::is_integral<T>::value> * = nullptr) {
        T threshold = (~b + 1u) % b;
        while (true) {
            T r = Uniform<T>();
            if (r >= threshold) return r % b;
        }
    }

    PBRT_HOST_DEVICE_INLINE
    void Advance(int64_t idelta) {
        uint64_t curMult = PCG32_MULT, curPlus = inc, accMult = 1u;
        uint64_t accPlus = 0u, delta = (uint64_t)idelta;
        while (delta > 0) {
            if (delta & 1) {
                accMult *= curMult;
                accPlus = accPlus * curMult + curPlus;
            }
            curPlus = (curMult + 1) * curPlus;
            curMult *= curMult;
            delta /= 2;
        }
        state = accMult * state + accPlus;
    }

    PBRT_HOST_DEVICE_INLINE
    int64_t operator-(const RNG &other) const {
        CHECK_EQ(inc, other.inc);
        uint64_t curMult = PCG32_MULT, curPlus = inc, curState = other.state;
        uint64_t theBit = 1u, distance = 0u;
        while (state != curState) {
            if ((state & theBit) != (curState & theBit)) {
                curState = curState * curMult + curPlus;
                distance |= theBit;
            }
            CHECK_EQ(state & theBit, curState & theBit);
            theBit <<= 1;
            curPlus = (curMult + 1ULL) * curPlus;
            curMult *= curMult;
        }
        return (int64_t)distance;
    }

    std::string ToString() const;

  private:
    // RNG Private Data
    uint64_t state, inc;
};

// RNG Inline Method Definitions
inline RNG::RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

template <>
inline uint32_t RNG::Uniform<uint32_t>() {
    uint64_t oldstate = state;
    state = oldstate * PCG32_MULT + inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

template <>
inline uint64_t RNG::Uniform<uint64_t>() {
    uint64_t v0 = Uniform<uint32_t>(), v1 = Uniform<uint32_t>();
    return (v0 << 32) | v1;
}

template <>
inline int32_t RNG::Uniform<int32_t>() {
    // https://stackoverflow.com/a/13208789
    uint32_t v = Uniform<uint32_t>();
    if (v <= (uint32_t)std::numeric_limits<int32_t>::max())
        // Safe to type convert directly.
        return int32_t(v);

    DCHECK_GE(v, (uint32_t)std::numeric_limits<int32_t>::min());
    return int32_t(v - std::numeric_limits<int32_t>::min()) +
        std::numeric_limits<int32_t>::min();
}

template <>
inline int64_t RNG::Uniform<int64_t>() {
    // https://stackoverflow.com/a/16408789
    uint64_t v = Uniform<uint64_t>();
    if (v <= (uint64_t)std::numeric_limits<int64_t>::max())
        // Safe to type convert directly.
        return int64_t(v);

    DCHECK_GE(v, (uint64_t)std::numeric_limits<int64_t>::min());
    return int64_t(v - std::numeric_limits<int64_t>::min()) +
        std::numeric_limits<int64_t>::min();
}

template<>
inline float RNG::Uniform<float>() {
#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
    return std::min<float>(OneMinusEpsilon,
                            Uniform<uint32_t>() * 2.3283064365386963e-10f);
#else
    return std::min<float>(OneMinusEpsilon, Uniform<uint32_t>() * 0x1p-32f);
#endif
}

template<>
inline double RNG::Uniform<double>() {
#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
    return std::min<double>(OneMinusEpsilon, (Uniform<uint64_t>() *
                                               5.42101086242752217003726400435e-20));
#else
    return std::min<double>(OneMinusEpsilon, Uniform<uint64_t>() * 0x1p-64);
#endif
}

template <typename T>
inline T RNG::Uniform() { return T::unimplemented; }

inline void RNG::SetSequence(uint64_t initseq) {
    state = 0u;
    inc = (initseq << 1u) | 1u;
    Uniform<uint32_t>();
    state += PCG32_DEFAULT_STATE;
    Uniform<uint32_t>();
}

}  // namespace pbrt

#endif  // PBRT_UTIL_RNG_H
