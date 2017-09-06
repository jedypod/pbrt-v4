
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
#include "pbrt.h"

#include "util/bounds.h"
#include "util/mathutil.h"
#include "util/geometry.h"
#include "util/stats.h"
#include <glog/logging.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>

namespace pbrt {

// Parallel Declarations
class AtomicFloat {
  public:
    // AtomicFloat Public Methods
    explicit AtomicFloat(Float v = 0) { bits = FloatToBits(v); }
    operator Float() const { return BitsToFloat(bits); }
    Float operator=(Float v) {
        bits = FloatToBits(v);
        return v;
    }
    void Add(Float v) {
#ifdef PBRT_FLOAT_AS_DOUBLE
        uint64_t oldBits = bits, newBits;
#else
        uint32_t oldBits = bits, newBits;
#endif
        do {
            newBits = FloatToBits(BitsToFloat(oldBits) + v);
        } while (!bits.compare_exchange_weak(oldBits, newBits));
    }

  private:
// AtomicFloat Private Data
#ifdef PBRT_FLOAT_AS_DOUBLE
    std::atomic<uint64_t> bits;
#else
    std::atomic<uint32_t> bits;
#endif
};

// Simple one-use barrier; ensures that multiple threads all reach a
// particular point of execution before allowing any of them to proceed
// past it.
//
// Note: this should be heap allocated and managed with a shared_ptr, where
// all threads that use it are passed the shared_ptr. This ensures that
// memory for the Barrier won't be freed until all threads have
// successfully cleared it.
class Barrier {
  public:
    static std::shared_ptr<Barrier> Create(int count) {
        // It would be nice to use make_shared, but it's complicated...
        // https://stackoverflow.com/questions/8147027/how-do-i-call-stdmake-shared-on-a-class-with-only-protected-or-private-const
        return std::shared_ptr<Barrier>(new Barrier(count));
    }
    ~Barrier() { CHECK_EQ(count, 0); }
    void Wait();

  private:
    Barrier(int count) : count(count) { CHECK_GT(count, 0); }

    std::mutex mutex;
    std::condition_variable cv;
    int count;
};

void ParallelFor(int64_t start, int64_t end, int chunkSize,
                 std::function<void(int64_t, int64_t)> func);

inline void ParallelFor(int64_t start, int64_t end,
                        std::function<void(int64_t)> func) {
    ParallelFor(start, end, 1, [&func](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i)
                func(i);
        });
}

void ParallelFor2D(const Bounds2i &extent, int chunkSize,
                   std::function<void(Bounds2i)> func);

inline void ParallelFor2D(const Bounds2i &extent,
                          std::function<void(Bounds2i)> func) {
    ParallelFor2D(extent, 1, std::move(func));
}

template <typename T, typename F, typename R> T
ParallelReduce(int64_t start, int64_t end, int chunkSize, F func, R reduce) {
    if (start == end) return T{};
    if (end - start < chunkSize) {
        return func(start, end);
    } else {
        std::mutex mutex;
        T final;
        ParallelFor(start, end, chunkSize,
                    [&](int64_t start, int64_t end) {
                        T result = func(start, end);
                        std::lock_guard<std::mutex> lock(mutex);
                        reduce(final, result);
                    });
        return final;
    }
}

extern PBRT_THREAD_LOCAL int ThreadIndex;

int MaxThreadIndex();

void ParallelInit();
void ParallelCleanup();
void MergeWorkerThreadStats();

}  // namespace pbrt

#endif  // PBRT_UTIL_PARALLEL_H
