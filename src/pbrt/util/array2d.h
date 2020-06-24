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

#ifndef PBRT_UTIL_ARRAY2D_H
#define PBRT_UTIL_ARRAY2D_H

// util/array2d.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/print.h>
#include <pbrt/util/check.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>

namespace pbrt {

template <typename T> class Array2D {
public:
    using value_type = T;
    using iterator = value_type *;
    using const_iterator = const value_type *;
    using allocator_type = pstd::pmr::polymorphic_allocator<pstd::byte>;

    Array2D(allocator_type allocator = {}) : Array2D({{0, 0}, {0, 0}}, allocator) { }
    Array2D(const Bounds2i &extent, allocator_type allocator = {})
        : extent(extent), allocator(allocator) {
        int n = extent.Area();
        values = allocator.allocate_object<T>(n);
        for (int i = 0; i < n; ++i)
            allocator.construct(values + i);
    }
    Array2D(const Bounds2i &extent, T def, allocator_type allocator = {})
        : Array2D(extent, allocator) {
        std::fill(begin(), end(), def);
    }

    template <typename InputIt,
              typename = typename std::enable_if_t<
                  !std::is_integral<InputIt>::value &&
                  std::is_base_of<std::input_iterator_tag,
                                  typename std::iterator_traits<InputIt>::iterator_category>::value>>
    Array2D(InputIt first, InputIt last, int nx, int ny, allocator_type allocator = {})
        : Array2D({{0, 0}, {nx, ny}}, allocator) {
        std::copy(first, last, begin());
    }
    Array2D(int nx, int ny, allocator_type allocator = {})
        : Array2D({{0, 0}, {nx, ny}}, allocator) { }
    Array2D(int nx, int ny, T def, allocator_type allocator = {})
        : Array2D({{0, 0}, {nx, ny}}, def, allocator) { }
    Array2D(const Array2D &a, allocator_type allocator = {})
        : Array2D(a.begin(), a.end(), a.xSize(), a.ySize(), allocator) { }

    ~Array2D() {
        int n = extent.Area();
        for (int i = 0; i < n; ++i)
            allocator.destroy(values + i);
        allocator.deallocate_object(values, n);
    }

    Array2D(Array2D &&a, allocator_type allocator = {})
        : extent(a.extent), allocator(allocator) {
        if (allocator == a.allocator) {
            values = a.values;
            a.extent = Bounds2i({0, 0}, {0, 0});
            a.values = nullptr;
        } else {
            values = allocator.allocate_object<T>(extent.Area());
            std::copy(a.begin(), a.end(), begin());
        }
    }
    Array2D &operator=(const Array2D &a) = delete;

    Array2D &operator=(Array2D &&other) {
        if (allocator == other.allocator) {
            pstd::swap(extent, other.extent);
            pstd::swap(values, other.values);
        } else if (extent == other.extent) {
            int n = extent.Area();
            for (int i = 0; i < n; ++i) {
                allocator.destroy(values + i);
                allocator.construct(values + i, other.values[i]);
            }
            extent = other.extent;
        } else {
            int n = extent.Area();
            for (int i = 0; i < n; ++i)
                allocator.destroy(values + i);
            allocator.deallocate_object(values, n);

            int no = other.extent.Area();
            values = allocator.allocate_object<T>(no);
            for (int i = 0; i < no; ++i)
                allocator.construct(values + i, other.values[i]);
        }
        return *this;
    }

    PBRT_HOST_DEVICE_INLINE
    T &operator()(int x, int y) {
        return (*this)[{x, y}];
    }
    PBRT_HOST_DEVICE_INLINE
    T &operator[](Point2i p) {
        DCHECK(InsideExclusive(p, extent));
        p.x -= extent.pMin.x;
        p.y -= extent.pMin.y;
        return values[p.x + (extent.pMax.x - extent.pMin.x) * p.y];
    }
    PBRT_HOST_DEVICE_INLINE
    const T &operator()(int x, int y) const {
        return (*this)[{x, y}];
    }
    PBRT_HOST_DEVICE_INLINE
    const T &operator[](Point2i p) const{
        DCHECK(InsideExclusive(p, extent));
        p.x -= extent.pMin.x;
        p.y -= extent.pMin.y;
        return values[p.x + (extent.pMax.x - extent.pMin.x) * p.y];
    }

    PBRT_HOST_DEVICE_INLINE
    int size() const { return extent.Area(); }
    PBRT_HOST_DEVICE_INLINE
    int xSize() const { return extent.pMax.x - extent.pMin.x; }
    PBRT_HOST_DEVICE_INLINE
    int ySize() const { return extent.pMax.y - extent.pMin.y; }

    PBRT_HOST_DEVICE_INLINE
    iterator begin() { return values; }
    PBRT_HOST_DEVICE_INLINE
    iterator end() { return begin() + size(); }
    PBRT_HOST_DEVICE_INLINE
    const_iterator begin() const { return values; }
    PBRT_HOST_DEVICE_INLINE
    const_iterator end() const { return begin() + size(); }

    PBRT_HOST_DEVICE_INLINE
    operator pstd::span<T>() { return pstd::span<T>(values, size()); }
    PBRT_HOST_DEVICE_INLINE
    operator pstd::span<const T>() const { return pstd::span<const T>(values, size()); }

    std::string ToString() const {
        std::string s = StringPrintf("[ Array2D extent: %s values: [", extent);
        for (int y = extent.pMin.y; y < extent.pMax.y; ++y) {
            s += " [ ";
            for (int x = extent.pMin.x; x < extent.pMax.x; ++x) {
                T value = (*this)(x, y);
                s += StringPrintf("%s, ", value);
            }
            s += "], ";
        }
        s += " ] ]";
        return s;
    }

private:
    Bounds2i extent;
    allocator_type allocator;
    T *values;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_ARRAY2D_H
