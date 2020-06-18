// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

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
    PBRT_CPU_GPU
    explicit AtomicFloat(float v = 0) {
#ifdef PBRT_IS_GPU_CODE
        value = v;
#else
        bits = FloatToBits(v);
#endif
    }
    PBRT_CPU_GPU
    operator float() const {
#ifdef PBRT_IS_GPU_CODE
        return value;
#else
        return BitsToFloat(bits);
#endif
    }
    PBRT_CPU_GPU
    Float operator=(float v) {
#ifdef PBRT_IS_GPU_CODE
        value = v;
        return value;
#else
        bits = FloatToBits(v);
        return v;
#endif
    }
    PBRT_CPU_GPU
    void Add(float v) {
#ifdef PBRT_IS_GPU_CODE
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
#ifdef PBRT_IS_GPU_CODE
    float value;
#else
    std::atomic<FloatBits> bits;
#endif
};

class AtomicDouble {
  public:
    // AtomicDouble Public Methods
    PBRT_CPU_GPU
    explicit AtomicDouble(double v = 0) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        value = v;
#else
        bits = FloatToBits(v);
#endif
    }

    PBRT_CPU_GPU
    operator double() const {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        return value;
#else
        return BitsToFloat(bits);
#endif
    }

    PBRT_CPU_GPU
    double operator=(double v) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
        value = v;
        return value;
#else
        bits = FloatToBits(v);
        return v;
#endif
    }

    PBRT_CPU_GPU
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
    explicit Barrier(int n) : numToBlock(n), numToExit(n) {}

    Barrier(const Barrier &) = delete;
    Barrier &operator=(const Barrier &) = delete;

    // All block. Returns true to only one thread (which should delete the
    // barrier).
    bool Block();

  private:
    std::mutex mutex;
    std::condition_variable cv;
    int numToBlock, numToExit;
};

void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t, int64_t)> func);

inline void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t)> func) {
    ParallelFor(start, end, [&func](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i)
            func(i);
    });
}

void ParallelFor2D(const Bounds2i &extent, std::function<void(Bounds2i)> func);

inline void ParallelFor2D(const Bounds2i &extent, std::function<void(Point2i)> func) {
    ParallelFor2D(extent, [&func](Bounds2i b) {
        for (Point2i p : b)
            func(p);
    });
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
