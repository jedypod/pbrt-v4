// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// util/memory.cpp*
#include <pbrt/util/memory.h>

#include <pbrt/util/check.h>
#include <pbrt/util/print.h>

#include <cstdlib>
#ifdef PBRT_HAVE_MALLOC_H
#include <malloc.h>  // for both memalign and _aligned_malloc
#endif
#ifdef PBRT_IS_WINDOWS
#include <psapi.h>
#include <windows.h>
#pragma comment(lib, "psapi.lib")
#endif  // PBRT_IS_WINDOWS
#ifdef PBRT_IS_LINUX
#include <unistd.h>
#include <cstdio>
#endif  // PBRT_IS_LINUX
#ifdef PBRT_IS_OSX
#include <mach/mach.h>
#endif  // PBRT_IS_OSX

#if defined(PBRT_BUILD_GPU_RENDERER)
#include <cuda_runtime.h>
#endif

namespace pbrt {

// Memory Allocation Functions
void *AllocAligned(size_t size, int alignment) {
#if defined(PBRT_HAVE__ALIGNED_MALLOC)
    return _aligned_malloc(size, alignment);
#elif defined(PBRT_HAVE_POSIX_MEMALIGN)
    void *ptr;
    if (alignment < sizeof(void *))
        return malloc(size);
    if (posix_memalign(&ptr, alignment, size) != 0)
        ptr = nullptr;
    return ptr;
#else
    return memalign(alignment, size);
#endif
}

void FreeAligned(void *ptr) {
    if (ptr == nullptr)
        return;
#if defined(PBRT_HAVE__ALIGNED_MALLOC)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

#ifdef PBRT_BUILD_GPU_RENDERER

void *CUDAMemoryResource::do_allocate(size_t size, size_t alignment) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    CHECK_EQ(0, intptr_t(ptr) % alignment);
    return ptr;
}

void CUDAMemoryResource::do_deallocate(void *p, size_t bytes, size_t alignment) {
    CUDA_CHECK(cudaFree(p));
}

void *CUDATrackedMemoryResource::do_allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex);

    // GPU cache line alignment to avoid false sharing...
    alignment = std::max<size_t>(128, alignment);

    if (bypassSlab(size))
        return cudaAllocate(size, alignment);

    if ((slabOffset % alignment) != 0)
        slabOffset += alignment - (slabOffset % alignment);

    if (slabOffset + size > slabSize) {
        currentSlab = (uint8_t *)cudaAllocate(slabSize, alignof(std::max_align_t));
        slabOffset = 0;
    }

    uint8_t *ptr = currentSlab + slabOffset;
    slabOffset += size;
    return ptr;
}

void *CUDATrackedMemoryResource::cudaAllocate(size_t size, size_t alignment) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    DCHECK_EQ(0, intptr_t(ptr) % alignment);

    allocations[ptr] = size;
    bytesAllocated += size;
    return ptr;
}

void CUDATrackedMemoryResource::do_deallocate(void *p, size_t size, size_t alignment) {
    if (!p)
        return;

    if (bypassSlab(size)) {
        CUDA_CHECK(cudaFree(p));

        std::lock_guard<std::mutex> lock(mutex);
        auto iter = allocations.find(p);
        DCHECK(iter != allocations.end());
        allocations.erase(iter);
        bytesAllocated -= size;
    } else
        LOG_VERBOSE("Ignoring dealloc of %d bytes", size);
}

void CUDATrackedMemoryResource::PrefetchToGPU() const {
    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));

    LOG_VERBOSE("Prefetching %d allocations to GPU memory", allocations.size());
    size_t bytes = 0;
    for (auto iter : allocations) {
        CUDA_CHECK(
            cudaMemPrefetchAsync(iter.first, iter.second, deviceIndex, 0 /* stream */));
        bytes += iter.second;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Done prefetching: %d bytes total", bytes);
}

static CUDATrackedMemoryResource cudaTrackedMemoryResource;
Allocator gpuMemoryAllocator(&cudaTrackedMemoryResource);

#endif  // PBRT_BUILD_GPU_RENDERER

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 *
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t GetCurrentRSS() {
#ifdef PBRT_IS_WINDOWS
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
#elif defined(PBRT_IS_OSX)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info,
                  &infoCount) != KERN_SUCCESS)
        return (size_t)0L; /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(PBRT_IS_LINUX)
    FILE *fp;
    if ((fp = fopen("/proc/self/statm", "r")) == nullptr) {
        LOG_ERROR("Unable to open /proc/self/statm");
        return 0;
    }

    long rss = 0L;
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        LOG_ERROR("Unable to read /proc/self/statm");
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
#elif defined(PBRT_IS_GPU_CODE)
    return 0;
#else
#error "TODO: implement GetCurrentRSS() for this target"
    return 0;
    /*    struct rusage rusage;
    CHECK(getrusage(RUSAGE_SELF, &rusage) == 0);
    return rusage.ru_idrss;
    */
#endif
}

}  // namespace pbrt
