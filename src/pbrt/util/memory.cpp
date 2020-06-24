
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


// util/memory.cpp*
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>

#include <cstdlib>
#ifdef PBRT_HAVE_MALLOC_H
  #include <malloc.h>  // for both memalign and _aligned_malloc
#endif
#ifdef PBRT_IS_WINDOWS
  #include <windows.h>
  #include <psapi.h>
  #pragma comment(lib, "psapi.lib")
#endif // PBRT_IS_WINDOWS
#ifdef PBRT_IS_LINUX
  #include <cstdio>
  #include <unistd.h>
#endif // PBRT_IS_LINUX
#ifdef PBRT_IS_OSX
  #include <mach/mach.h>
#endif // PBRT_IS_OSX

namespace pbrt {

// Memory Allocation Functions
void *AllocAligned(size_t size) {
#if defined(PBRT_HAVE__ALIGNED_MALLOC)
    return _aligned_malloc(size, PBRT_L1_CACHE_LINE_SIZE);
#elif defined(PBRT_HAVE_POSIX_MEMALIGN)
    void *ptr;
    if (posix_memalign(&ptr, PBRT_L1_CACHE_LINE_SIZE, size) != 0) ptr = nullptr;
    return ptr;
#else
    return memalign(PBRT_L1_CACHE_LINE_SIZE, size);
#endif
}

void FreeAligned(void *ptr) {
    if (ptr == nullptr) return;
#if defined(PBRT_HAVE__ALIGNED_MALLOC)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

std::string MemoryArena::ToString() const {
    auto str = [](const MemoryBlock &b) {
        return StringPrintf("[ MemoryBlock ptr: %p size: %d ]", b.ptr.get(), b.size);
    };
    std::string s = StringPrintf("[ MemoryArena blockSize: %d currentBlock: %s currentBlockPos: %d ",
                                 blockSize, str(currentBlock).c_str(), currentBlockPos);
    s += "usedBlocks: ";
    for (const MemoryBlock &block : usedBlocks)
        s += str(block);
    s += "availableBlocks: ";
    for (const MemoryBlock &block : availableBlocks)
        s += str(block);
    s += " ]";
    return s;

}

namespace detail {

std::string MemoryPoolToString(int maxAlloc, size_t poolSize) {
    return StringPrintf("[ MemoryPool maxAlloc: %d  pool.size(): %d ]",
                        maxAlloc, poolSize);
}

} // namespace detail

/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 *
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
size_t GetCurrentRSS()
{
#ifdef PBRT_IS_WINDOWS
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;
#elif defined(PBRT_IS_OSX)
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if ( task_info( mach_task_self( ), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount ) != KERN_SUCCESS )
        return (size_t)0L;      /* Can't access? */
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
#elif defined(__CUDA_ARCH__)
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
