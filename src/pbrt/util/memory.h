// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_MEMORY_H
#define PBRT_UTIL_MEMORY_H

// util/memory.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#ifdef PBRT_BUILD_GPU_RENDERER
#include <unordered_map>
#endif

namespace pbrt {

// Memory Declarations
void *AllocAligned(size_t size, int alignment = PBRT_L1_CACHE_LINE_SIZE);

template <typename T>
T *AllocAligned(size_t count) {
    return (T *)AllocAligned(count * sizeof(T));
}

void FreeAligned(void *);

size_t GetCurrentRSS();

#ifdef PBRT_BUILD_GPU_RENDERER

class CUDAMemoryResource : public pstd::pmr::memory_resource {
    void *do_allocate(size_t size, size_t alignment);
    void do_deallocate(void *p, size_t bytes, size_t alignment);

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }
};

class CUDATrackedMemoryResource : public CUDAMemoryResource {
  public:
    void *do_allocate(size_t size, size_t alignment);
    void do_deallocate(void *p, size_t bytes, size_t alignment);

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }

    void PrefetchToGPU() const;
    size_t BytesAllocated() const { return bytesAllocated; }

  private:
    bool bypassSlab(size_t size) const { return size > slabSize / 4; }

    void *cudaAllocate(size_t size, size_t alignment);

    size_t bytesAllocated = 0;
    uint8_t *currentSlab = nullptr;
    static constexpr int slabSize = 1024 * 1024;
    size_t slabOffset = slabSize;
    std::mutex mutex;
    std::unordered_map<void *, size_t> allocations;
};

extern Allocator gpuMemoryAllocator;

#endif

class TrackedMemoryResource : public pstd::pmr::memory_resource {
  public:
    TrackedMemoryResource(
        pstd::pmr::memory_resource *source = pstd::pmr::get_default_resource())
        : source(source) {}

    void *do_allocate(size_t size, size_t alignment) {
        void *ptr = source->allocate(size, alignment);
        uint64_t currentBytes = allocatedBytes.fetch_add(size) + size;
        uint64_t prevMax = maxAllocatedBytes.load(std::memory_order_relaxed);
        while (prevMax < currentBytes &&
               !maxAllocatedBytes.compare_exchange_weak(prevMax, currentBytes))
            ;
        return ptr;
    }
    void do_deallocate(void *p, size_t bytes, size_t alignment) {
        source->deallocate(p, bytes, alignment);
        allocatedBytes -= bytes;
    }

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }

    size_t CurrentAllocatedBytes() const { return allocatedBytes.load(); }
    size_t MaxAllocatedBytes() const { return maxAllocatedBytes.load(); }

  private:
    pstd::pmr::memory_resource *source;
    std::atomic<uint64_t> allocatedBytes{0}, maxAllocatedBytes{0};
};

template <typename T>
struct AllocationTraits {
    using SingleObject = T *;
};
template <typename T>
struct AllocationTraits<T[]> {
    using Array = T *;
};
template <typename T, size_t n>
struct AllocationTraits<T[n]> {
    struct Invalid {};
};

class alignas(PBRT_L1_CACHE_LINE_SIZE) ScratchBuffer {
  public:
    ScratchBuffer() = default;
    ScratchBuffer(int size) : allocatedBytes(size) {
        ptr = new uint8_t[allocatedBytes];
        freePtr = true;
    }
    PBRT_CPU_GPU
    ScratchBuffer(uint8_t *ptr, int size) : ptr(ptr), allocatedBytes(size) {}

    ScratchBuffer(const ScratchBuffer &) = delete;

    ScratchBuffer(ScratchBuffer &&b) {
        ptr = b.ptr;
        freePtr = b.freePtr;
        allocatedBytes = b.allocatedBytes;
        offset = b.offset;

        b.ptr = nullptr;
        b.freePtr = false;
        b.allocatedBytes = b.offset = 0;
    }

    ~ScratchBuffer() {
        if (freePtr)
            delete[] ptr;
    }

    ScratchBuffer &operator=(const ScratchBuffer &) = delete;

    ScratchBuffer &operator=(ScratchBuffer &&b) {
        std::swap(b.ptr, ptr);
        std::swap(b.freePtr, freePtr);
        std::swap(b.allocatedBytes, allocatedBytes);
        std::swap(b.offset, offset);
        return *this;
    }

    PBRT_CPU_GPU
    void *Alloc(size_t size, size_t align) {
        if ((offset % align) != 0)
            offset += align - (offset % align);
        CHECK_LE(offset + size, allocatedBytes);

        void *p = ptr + offset;
        offset += size;
        return p;
    }

    template <typename T, typename... Args>
    PBRT_CPU_GPU typename AllocationTraits<T>::SingleObject Alloc(Args &&... args) {
        T *p = (T *)Alloc(sizeof(T), alignof(T));
        return new (p) T(std::forward<Args>(args)...);
    }

    template <typename T>
    PBRT_CPU_GPU typename AllocationTraits<T>::Array Alloc(size_t n = 1) {
        using ElementType = typename std::remove_extent<T>::type;
        ElementType *ret =
            (ElementType *)Alloc(n * sizeof(ElementType), alignof(ElementType));
        for (size_t i = 0; i < n; ++i)
            new (&ret[i]) ElementType();
        return ret;
    }

    PBRT_CPU_GPU
    void Reset() { offset = 0; }

  private:
    uint8_t *ptr = nullptr;
    bool freePtr = false;
    int allocatedBytes = 0, offset = 0;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MEMORY_H
