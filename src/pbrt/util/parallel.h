
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

#ifndef PBRT_UTIL_PARALLEL_H
#define PBRT_UTIL_PARALLEL_H

// core/parallel.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/float.h>
#include <pbrt/util/vecmath.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <string>

namespace pbrt {

// Parallel Declarations
class AtomicFloat {
  public:
    // AtomicFloat Public Methods
    PBRT_HOST_DEVICE_INLINE
    explicit AtomicFloat(float v = 0) {
#ifdef __CUDA_ARCH__
        value = v;
#else
        bits = FloatToBits(v);
#endif
    }
    PBRT_HOST_DEVICE_INLINE
    operator float() const {
#ifdef __CUDA_ARCH__
        return value;
#else
        return BitsToFloat(bits);
#endif
    }
    PBRT_HOST_DEVICE_INLINE
    Float operator=(float v) {
#ifdef __CUDA_ARCH__
        value = v;
        return value;
#else
        bits = FloatToBits(v);
        return v;
#endif
    }
    PBRT_HOST_DEVICE_INLINE
    void Add(float v) {
#ifdef __CUDA_ARCH__
        atomicAdd(&value, v);
#else
        FloatBits oldBits = bits, newBits;
        do {
            newBits = FloatToBits(BitsToFloat(oldBits) + v);
        } while (!bits.compare_exchange_weak(oldBits, newBits));
#endif
    }

    std::string ToString() const;

  private:
    // AtomicFloat Private Data
#ifdef __CUDA_ARCH__
    float value;
#else
    std::atomic<FloatBits> bits;
#endif
};

class AtomicDouble {
  public:
    // AtomicDouble Public Methods
    PBRT_HOST_DEVICE_INLINE
    explicit AtomicDouble(double v = 0) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        value = v;
#else
        bits = FloatToBits(v);
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    operator double() const {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        return value;
#else
        return BitsToFloat(bits);
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    double operator=(double v) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        value = v;
        return value;
#else
        bits = FloatToBits(v);
        return v;
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    void Add(double v) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        atomicAdd(&value, v);
#elif defined(__CUDA_ARCH__)
        uint64_t old = bits, assumed;

        do {
            assumed = old;
            old = atomicCAS((unsigned long long int *)&bits, assumed,
                            __double_as_longlong(v + __longlong_as_double(assumed)));
        } while (assumed != old);
#else
        uint64_t oldBits = bits, newBits;
        do {
            newBits = FloatToBits(BitsToFloat(oldBits) + v);
        } while (!bits.compare_exchange_weak(oldBits, newBits));
#endif
    }

    std::string ToString() const;

  private:
    // AtomicDouble Private Data
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
    double value;
#elif defined(__CUDA_ARCH__)
    uint64_t bits;
#else
    std::atomic<uint64_t> bits;
#endif
};

class Barrier {
public:
    explicit Barrier(int n) : numToBlock(n), numToExit(n) { }

    Barrier(const Barrier&) = delete;
    Barrier& operator=(const Barrier&) = delete;

    // All block. Returns true to only one thread (which should delete the
    // barrier).
    bool Block();

private:
    std::mutex mutex;
    std::condition_variable cv;
    int numToBlock, numToExit;
};

void ParallelFor(int64_t start, int64_t end,
                 std::function<void(int64_t, int64_t)> func,
                 const char *progressName = nullptr);

inline void ParallelFor(int64_t start, int64_t end,
                        std::function<void(int64_t)> func,
                        const char *progressName = nullptr) {
    ParallelFor(start, end, [&func](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i)
            func(i);
    }, progressName);
}

void ParallelFor2D(const Bounds2i &extent,
                   std::function<void(Bounds2i)> func,
                   const char *progressName = nullptr);

inline void ParallelFor2D(const Bounds2i &extent,
                          std::function<void(Point2i)> func,
                          const char *progressName = nullptr) {
    ParallelFor2D(extent, [&func](Bounds2i b) {
        for (Point2i p : b)
            func(p);
    }, progressName);
}

void ForEachThread(std::function<void(void)> func);

extern thread_local int ThreadIndex;

int AvailableCores();
int RunningThreads();
int MaxThreadIndex();

void ParallelInit(int nThreads = -1);
void ParallelCleanup();

}  // namespace pbrt

#endif  // PBRT_UTIL_PARALLEL_H
