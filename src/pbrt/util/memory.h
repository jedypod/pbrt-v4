
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

// util/memory.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/math.h>
#include <pbrt/util/check.h>
#include <pbrt/util/pstd.h>

#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace pbrt {

// Memory Declarations
void *AllocAligned(size_t size);
template <typename T>
T *AllocAligned(size_t count) {
    return (T *)AllocAligned(count * sizeof(T));
}

void FreeAligned(void *);

size_t GetCurrentRSS();

template <typename T> struct AllocationTraits { using SingleObject = T *; };
template <typename T> struct AllocationTraits<T[]> { using Array = T *; };
template <typename T, size_t n> struct AllocationTraits<T[n]> { struct Invalid { }; };

#ifndef PBRT_L1_CACHE_LINE_SIZE
#define PBRT_L1_CACHE_LINE_SIZE 64
#endif

class MaterialBuffer {
public:
    MaterialBuffer(int size = 4096)
        : size(size) {
        ptr = new uint8_t[size];
    }
    PBRT_HOST_DEVICE
    MaterialBuffer(uint8_t *ptr, int size)
        : ptr(ptr), size(size) { }

    template<typename T, typename ...Args>
    PBRT_HOST_DEVICE
    T *Alloc(Args&&... args)  {
        int align = alignof(T);

        if ((offset % align) != 0)
            offset += align - (offset % align);
        CHECK_LE(offset + sizeof(T), size);

        T *tp = (T *)(ptr + offset);
        offset += sizeof(T);
        return new (tp) T(std::forward<Args>(args)...);
    }

    PBRT_HOST_DEVICE
    void Reset() {
        offset = 0;
    }

private:
    uint8_t *ptr;
    int size, offset = 0;
};

class alignas(PBRT_L1_CACHE_LINE_SIZE)
MemoryArena {
  public:
    // MemoryArena Public Methods
    MemoryArena(size_t blockSize = 262144) : blockSize(blockSize) {}
    MemoryArena(MemoryArena &&a) {
        Swap(a);
    }
    MemoryArena &operator=(MemoryArena &&a) {
        Swap(a);
        return *this;
    }

    void *Alloc(size_t allocSize, size_t align = 0) {
        if (align == 0) {
#if __GNUC__ == 4 && __GNUC_MINOR__ < 9
            // gcc bug: max_align_t wasn't in std:: until 4.9.0
            align = alignof(::max_align_t);
#else
            align = alignof(std::max_align_t);
#endif
        }
        DCHECK(IsPowerOf2(align));

        if (allocSize > blockSize) {
            // We've got a big allocation; let the current block be so that
            // smaller allocations have a chance at using up more of it.
            usedBlocks.push_back(MemoryBlock(allocSize));
            return usedBlocks.back().ptr.get();
        }

        if ((currentBlockPos % align) != 0)
            currentBlockPos += align - (currentBlockPos % align);
        DCHECK_EQ(0, currentBlockPos % align);

        if (currentBlockPos + allocSize > currentBlock.size) {
            // Add current block to _usedBlocks_ list
            if (currentBlock.size) {
                usedBlocks.push_back(std::move(currentBlock));
                currentBlock = {};
            }

            // Get new block of memory for _MemoryArena_

            // Try to get memory block from _availableBlocks_
            for (auto iter = availableBlocks.begin();
                 iter != availableBlocks.end(); ++iter) {
                if (allocSize <= iter->size) {
                    currentBlock = std::move(*iter);
                    availableBlocks.erase(iter);
                    goto success;
                }
            }
            currentBlock = MemoryBlock(blockSize);
        success:
            currentBlockPos = 0;
        }

        void *ptr = currentBlock.ptr.get() + currentBlockPos;
        currentBlockPos += allocSize;
        return ptr;
    }

    template<typename T, typename ...Args>
    typename AllocationTraits<T>::SingleObject Alloc(Args&&... args)  {
        T *ptr = (T *)Alloc(sizeof(T), alignof(T));
        return new (ptr) T(std::forward<Args>(args)...);
    }

    template <typename T>
    typename AllocationTraits<T>::Array Alloc(size_t n = 1, bool runConstructor = true) {
        using ElementType = typename std::remove_extent<T>::type;
        ElementType *ret = (ElementType *)Alloc(n * sizeof(ElementType),
                                                alignof(ElementType));
        if (runConstructor)
            for (size_t i = 0; i < n; ++i) new (&ret[i]) ElementType();
        return ret;
    }

    void Reset() {
        currentBlockPos = 0;
        availableBlocks.splice(availableBlocks.begin(), usedBlocks);
    }
    bool IsReset() const {
        return currentBlockPos == 0 && usedBlocks.size() == 0;
    }
    size_t BytesAllocated() const {
        size_t total = currentBlock.size;
        for (const auto &alloc : usedBlocks) total += alloc.size;
        for (const auto &alloc : availableBlocks) total += alloc.size;
        return total;
    }

    void Swap(MemoryArena &a) {
        std::swap(blockSize, a.blockSize);
        std::swap(currentBlock, a.currentBlock);
        std::swap(currentBlockPos, a.currentBlockPos);
        std::swap(usedBlocks, a.usedBlocks);
        std::swap(availableBlocks, a.availableBlocks);
    }

    std::string ToString() const;

  private:
    struct MemoryBlock {
        MemoryBlock() = default;
        explicit MemoryBlock(size_t size)
          : ptr(std::make_unique<char[]>(size)), size(size) {}

        std::unique_ptr<char[]> ptr;
        size_t size = 0;
    };

    // MemoryArena Private Data
    size_t blockSize = 256 * 1024;
    MemoryBlock currentBlock;
    size_t currentBlockPos = 0;
    std::list<MemoryBlock> usedBlocks, availableBlocks;
};

// MemoryArena repurposed to fulfill the STL allocator interface
template <typename T> class ArenaAllocator {
 public:
    using value_type = T;

    ArenaAllocator(MemoryArena *arena)
        : arena(arena) {}
    template<typename U> ArenaAllocator(const ArenaAllocator<U> &a)
        : arena(a.arena) {}

    T *allocate(size_t n) {
        return (T *)arena->Alloc(n * sizeof(T), alignof(T));
    }
    void deallocate(T *, size_t) { }

    bool operator==(const ArenaAllocator<T> &a) const { return arena == a.arena; }
    bool operator!=(const ArenaAllocator<T> &a) const { return arena != a.arena; }

    std::string ToString() const {
        return std::string("[ ArenaAllocator arena: ") +
            (arena ? arena->ToString() : std::string("(nullptr)")) + " ]";
    }

 private:
    template <typename U> friend class ArenaAllocator;
    MemoryArena *arena = nullptr;
};

class ArenaMemoryResource : public pstd::pmr::memory_resource {
 public:
    ArenaMemoryResource(MemoryArena *arena)
        : arena(arena) {}

    void *do_allocate(size_t bytes, size_t alignment) {
        return arena->Alloc(bytes, alignment);
    }
    void do_deallocate(void *p, size_t bytes, size_t alignment) { }
    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }

 private:
    MemoryArena *arena = nullptr;
};

namespace detail {

std::string MemoryPoolToString(int maxAlloc, size_t poolSize);

}  // namespace detail

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
        CHECK(ptr != nullptr);
        std::lock_guard<std::mutex> lock(mutex);
        // Check for double-free.
        DCHECK(std::find_if(pool.begin(), pool.end(),
                            [ptr](std::unique_ptr<T> &p) {
                                return ptr == p.get();
                            }) == pool.end());

        if (pool.size() == maxAlloc)
            delete ptr;
        else
            pool.push_back(std::unique_ptr<T>(ptr));
    }

    std::string ToString() const {
        return detail::MemoryPoolToString(maxAlloc, pool.size());
    }

 private:
    int nAllocs = 0;
    // If provided, this function is applied to reused T *s before they are
    // returned by Alloc() so that they can be restored to a default state.
    std::function<void(T *)> reset;
    int maxAlloc;
    std::vector<std::unique_ptr<T>> pool;
    std::mutex mutex;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MEMORY_H
