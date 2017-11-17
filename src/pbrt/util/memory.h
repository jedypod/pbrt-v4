
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

#ifndef PBRT_UTIL_MEMORY_H
#define PBRT_UTIL_MEMORY_H

// core/memory.h*
#include <pbrt/core/pbrt.h>

#include <pbrt/util/math.h>

#include <absl/types/span.h>
#include <glog/logging.h>

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>

namespace pbrt {

// Memory Declarations
void *AllocAligned(size_t size);
template <typename T>
T *AllocAligned(size_t count) {
    return (T *)AllocAligned(count * sizeof(T));
}

void FreeAligned(void *);

class alignas(PBRT_L1_CACHE_LINE_SIZE)
MemoryArena {
  public:
    // MemoryArena Public Methods
    MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) {}
    void *Alloc(size_t nRequested, size_t align = 0) {
        if (align == 0) {
#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
        // gcc bug: max_align_t wasn't in std:: until 4.9.0
            align = alignof(::max_align_t);
#else
            align = alignof(std::max_align_t);
#endif
        }
        DCHECK(IsPowerOf2(align));

        size_t nAlloc = nRequested + align - 1;
        if (currentBlockPos + nAlloc > currentBlock.size) {
            // Add current block to _usedBlocks_ list
            if (currentBlock.size) {
                usedBlocks.push_back(std::move(currentBlock));
                currentBlock = {};
            }

            // Get new block of memory for _MemoryArena_

            // Try to get memory block from _availableBlocks_
            for (auto iter = availableBlocks.begin();
                 iter != availableBlocks.end(); ++iter) {
                if (nAlloc <= iter->size) {
                    currentBlock = std::move(*iter);
                    availableBlocks.erase(iter);
                    goto success;
                }
            }
            currentBlock = MemoryBlock(std::max(nAlloc, blockSize));
        success:
            currentBlockPos = 0;
        }

        void *start = currentBlock.ptr.get() + currentBlockPos;
        currentBlockPos += nAlloc;
        void *ptr = std::align(align, nRequested, start, nAlloc);
        CHECK_NOTNULL(ptr);
        return ptr;
    }
    template <typename T>
    T *AllocArray(size_t n = 1, bool runConstructor = true) {
        T *ret = (T *)Alloc(n * sizeof(T), alignof(T));
        if (runConstructor)
            for (size_t i = 0; i < n; ++i) new (&ret[i]) T();
        return ret;
    }
    template<typename T, typename ...Args> T *Alloc(Args&&... args)  {
        T *ptr = (T *)Alloc(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }
    void Reset() {
        currentBlockPos = 0;
        availableBlocks.splice(availableBlocks.begin(), usedBlocks);
    }
    size_t TotalAllocated() const {
        size_t total = currentBlock.size;
        for (const auto &alloc : usedBlocks) total += alloc.size;
        for (const auto &alloc : availableBlocks) total += alloc.size;
        return total;
    }

  private:
    struct MemoryBlock {
        MemoryBlock() = default;
        explicit MemoryBlock(size_t size)
          : ptr(new char[size]), size(size) {}
        std::unique_ptr<char[]> ptr;
        size_t size = 0;
    };

    // MemoryArena Private Data
    const size_t blockSize;
    MemoryBlock currentBlock;
    size_t currentBlockPos = 0;
    std::list<MemoryBlock> usedBlocks, availableBlocks;
};

// Pool of up to a limited number of allocated objects of type T so that they can be
// reused without repeated dynamic memory allocation.
template <typename T> class MemoryPool {
 public:
    MemoryPool(std::function<void(T *)> reset, int maxAlloc = 64)
       : reset(std::move(reset)), maxAlloc(maxAlloc) {}
    ~MemoryPool() {
        CHECK_EQ(nAllocs, pool.size());  // Otherwise they haven't all been returned.
    }

    T *Alloc() {
        std::lock_guard<std::mutex> lock(mutex);
        if (pool.empty()) {
            ++nAllocs;
            // For current usage in the parser, hitting here implies a memory leak.
            // This may not be the case for other use-cases.
            CHECK_LT(nAllocs, maxAlloc);
            return new T;
        }

        T *ptr = pool.back().release();
        pool.pop_back();
        if (reset) reset(ptr);
        return ptr;
    }
    void Release (T *ptr) {
        CHECK_NOTNULL(ptr);
        std::lock_guard<std::mutex> lock(mutex);
#ifndef NDEBUG
        // Check for double-free.
        DCHECK(std::find_if(pool.begin(), pool.end(),
                            [ptr](std::unique_ptr<T> &p) {
                                return ptr == p.get();
                            }) == pool.end());
#endif  // !NDEBUG

        if (pool.size() == maxAlloc)
            delete ptr;
        else
            pool.push_back(std::unique_ptr<T>(ptr));
    }

 private:
    int nAllocs = 0;
    // If provided, this function is applied to reused T *s before they are
    // returned by Alloc() so that they can be restored to a default state.
    std::function<void(T *)> reset;
    const int maxAlloc;
    std::vector<std::unique_ptr<T>> pool;
    std::mutex mutex;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MEMORY_H
