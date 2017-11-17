
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
#include <pbrt/core/pbrt.h>

#include <pbrt/util/bounds.h>
#include <pbrt/util/math.h>

#include <atomic>
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
        FloatBits oldBits = bits, newBits;
        do {
            newBits = FloatToBits(BitsToFloat(oldBits) + v);
        } while (!bits.compare_exchange_weak(oldBits, newBits));
    }

  private:
    // AtomicFloat Private Data
    std::atomic<FloatBits> bits;
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

void ForEachWorkerThread(std::function<void(void)> func);

extern thread_local int ThreadIndex;

int AvailableCores();
int MaxThreadIndex();

void ParallelInit(int nThreads = -1);
void ParallelCleanup();

}  // namespace pbrt

#endif  // PBRT_UTIL_PARALLEL_H
