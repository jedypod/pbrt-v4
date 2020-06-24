
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

#ifndef PBRT_UTIL_BUFFERCACHE_H
#define PBRT_UTIL_BUFFERCACHE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>

#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace pbrt {

STAT_MEMORY_COUNTER("Memory/Redundant vertex and index buffers", redundantBufferBytes);
STAT_PERCENT("Geometry/Buffer cache hits", nBufferCacheHits, nBufferCacheLookups);

// BufferId stores a hash of the contents of a buffer as well as its size.
// It serves as a key for the BufferCache hash table.
struct BufferId {
    BufferId() = default;
    BufferId(const char *ptr, size_t size)
      : hash(HashBuffer(ptr, size)), size(size) {}

    bool operator==(const BufferId &id) const {
        return hash == id.hash && size == id.size;
    }

    std::string ToString() const {
        return StringPrintf("[ BufferId hash: %d size: %d ]", hash, size);
    }

    uint64_t hash = 0;
    size_t size = 0;
};

// Utility class that computes the hash of a BufferId, using the
// already-computed hash of its buffer.
struct BufferHasher {
    size_t operator()(const BufferId &id) const {
        return id.hash;
    }
};

// The BufferCache class lets us cases such as where a TriangleMesh is
// storing the same vertex indices, positions, uv texture coordinates,
// etc., as another TriangleMesh that has already been created.  In that
// case, the BUfferCache returns a pointer to the pre-existing buffer that
// stores those values, allowing the redundant one to be freed, thus
// reducing memory use. (This case can come up with highly complex scenes,
// especially with large amounts of procedural geometry.)
template <typename T>
class BufferCache {
 public:
    BufferCache(Allocator alloc)
        : alloc(alloc) { }

    const T *LookupOrAdd(std::vector<T> buf) {
        // Hash the provided buffer and see if it's already in the cache.
        // Assumes no padding in T for alignment. (TODO: can we verify this
        // at compile time?)
        BufferId id((const char *)buf.data(), buf.size() * sizeof(T));
        ++nBufferCacheLookups;
        std::lock_guard<std::mutex> lock(mutex);
        auto iter = cache.find(id);
        if (iter != cache.end()) {
            // Success; return the pointer to the start of already-existing
            // one.
            CHECK(std::memcmp(buf.data(), iter->second->data(),
                              buf.size() * sizeof(T)) == 0);
            ++nBufferCacheHits;
            redundantBufferBytes += buf.capacity() * sizeof(T);
            return iter->second->data();
        }
        cache[id] = alloc.new_object<pstd::vector<T>>(buf.begin(), buf.end(), alloc);
        return cache[id]->data();
    }

    size_t BytesUsed() const {
        size_t sum = 0;
        for (const auto &item : cache)
            sum += item.second->capacity() * sizeof(T);
        return sum;
    }

    void Clear() {
        for (const auto &item : cache)
            alloc.delete_object(item.second);
        cache.clear();
    }

    std::string ToString() const {
        return StringPrintf("[ BufferCache cache.size(): %d BytesUsed(): %d ]",
                            cache.size(), BytesUsed());
    }

 private:
    Allocator alloc;
    std::mutex mutex;
    std::unordered_map<BufferId, pstd::vector<T> *, BufferHasher> cache;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_BUFFERCACHE_H
