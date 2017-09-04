
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

#ifndef PBRT_CORE_TUPLE_H
#define PBRT_CORE_TUPLE_H

// core/tuple.h*
#include "pbrt.h"

#include "stringprint.h"
#include <glog/logging.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace pbrt {

template <typename T>
inline bool isNaN(const T x) {
    return std::isnan(x);
}
template <>
inline bool isNaN(const int x) {
    return false;
}

template <template<typename> class Child, typename T>
class Tuple2 {
  public:
    Tuple2() { x = y = 0; }
    Tuple2(T xx, T yy) : x(xx), y(yy) { DCHECK(!HasNaNs()); }
    bool HasNaNs() const { return isNaN(x) || isNaN(y); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    Tuple2(const Child<T> &v) {
        DCHECK(!v.HasNaNs());
        x = v.x;
        y = v.y;
    }
    Child<T> &operator=(const Child<T> &v) {
        DCHECK(!v.HasNaNs());
        x = v.x;
        y = v.y;
        return *this;
    }
#endif  // !NDEBUG

    template <typename U>
    // WHY doesn't this work?
    // Child<std::common_type<T, U>> operator+(const Child<U> &v) const {
    auto operator+(const Child<U> &v) const -> Child<decltype(T{}+U{})> {
        DCHECK(!v.HasNaNs());
        return { x + v.x, y + v.y };
    }
    template <typename U>
    Child<T> &operator+=(const Child<U> &v) {
        DCHECK(!v.HasNaNs());
        x += v.x;
        y += v.y;
        return *this;
    }

    template <typename U>
    auto operator-(const Child<U> &v) const -> Child<decltype(T{}-U{})> {
        DCHECK(!v.HasNaNs());
        return { x - v.x, y - v.y };
    }
    template <typename U>
    Child<T> &operator-=(const Child<U> &v) {
        DCHECK(!v.HasNaNs());
        x -= v.x;
        y -= v.y;
        return *this;
    }

    bool operator==(const Child<T> &v) const { return x == v.x && y == v.y; }
    bool operator!=(const Child<T> &v) const { return x != v.x || y != v.y; }

    template <typename U>
    auto operator*(U f) const -> Child<decltype(T{}*U{})> {
        return { f * x, f * y };
    }
    template <typename U>
    Child<T> &operator*=(U v) {
        DCHECK(!isNaN(v));
        x *= v;
        y *= v;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    auto operator/(U v) const -> Child<decltype(T{}/U{})> {
        DCHECK(v != 0 && !isNaN(v));
        return { x / v, y / v };
    }
    template <typename U>
    Child<T> &operator/=(U v) {
        CHECK_NE(v, 0);
        DCHECK(isNaN(v));
        x /= v;
        y /= v;
        return static_cast<Child<T> &>(*this);
    }

    Child<T> operator-() const { return { -x, -y }; }

    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 1);
        if (i == 0) return x;
        return y;
    }

    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 1);
        if (i == 0) return x;
        return y;
    }

    std::string ToString() const {
        if (std::is_floating_point<T>::value)
            return StringPrintf("[ %f, %f ]", x, y);
        else
            return StringPrintf("[ %d, %d ]", x, y);
    }

    // Tuple2 Public Data
    T x, y;
};

template <template<class> class C, typename T>
inline std::ostream &operator<<(std::ostream &os, const Tuple2<C, T> &t) {
    return os << t.ToString();
}

template <template<class> class C, typename T>
inline C<T> Abs(const Tuple2<C, T> &v) {
    return { std::abs(v.x), std::abs(v.y) };
}

template <template<class> class C, typename T, typename U>
inline auto operator*(U f, const C<T> &t) -> C<decltype(T{}*U{})> {
    DCHECK(!t.HasNaNs());
    return t * f;
}

template <template<class> class C, typename T>
inline C<T> Floor(const Tuple2<C, T> &t) {
    return { std::floor(t.x), std::floor(t.y) };
}

template <template<class> class C, typename T>
inline C<T> Ceil(const Tuple2<C, T> &t) {
    return { std::ceil(t.x), std::ceil(t.y) };
}

template <template<class> class C, typename T>
inline C<T> Lerp(Float t, const Tuple2<C, T> &t0, const Tuple2<C, T> &t1) {
    return (1 - t) * t0 + t * t1;
}

template <template<class> class C, typename T>
inline C<T> Max(const Tuple2<C, T> &t0, const Tuple2<C, T> &t1) {
    return { std::max(t0.x, t1.x), std::max(t0.y, t1.y) };
}

template <template<class> class C, typename T>
inline C<T> Min(const Tuple2<C, T> &t0, const Tuple2<C, T> &t1) {
    return { std::min(t0.x, t1.x), std::min(t0.y, t1.y) };
}


template <template<typename> class Child, typename T>
class Tuple3 {
  public:
    // Tuple3 Public Methods
    Tuple3() { x = y = z = 0; }
    Tuple3(T x, T y, T z) : x(x), y(y), z(z) { DCHECK(!HasNaNs()); }
    bool HasNaNs() const { return isNaN(x) || isNaN(y) || isNaN(z); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    Tuple3(const Child<T> &v) {
        DCHECK(!v.HasNaNs());
        x = v.x;
        y = v.y;
        z = v.z;
    }

    Tuple3 &operator=(const Child<T> &v) {
        DCHECK(!v.HasNaNs());
        x = v.x;
        y = v.y;
        z = v.z;
        return *this;
    }
#endif  // !NDEBUG

    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    template <typename U>
    auto operator+(const Child<U> &v) const -> Child<decltype(T{}+U{})> {
        DCHECK(!v.HasNaNs());
        return { x + v.x, y + v.y, z + v.z };
    }
    template <typename U>
    Child<T> &operator+=(const Child<U> &v) {
        DCHECK(!v.HasNaNs());
        x += v.x;
        y += v.y;
        z += v.z;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    auto operator-(const Child<U> &v) const -> Child<decltype(T{}-U{})> {
        DCHECK(!v.HasNaNs());
        return { x - v.x, y - v.y, z - v.z };
    }
    template <typename U>
    Child<T> &operator-=(const Child<U> &v) {
        DCHECK(!v.HasNaNs());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return static_cast<Child<T> &>(*this);
    }

    bool operator==(const Child<T> &v) const {
        return x == v.x && y == v.y && z == v.z;
    }
    bool operator!=(const Child<T> &v) const {
        return x != v.x || y != v.y || z != v.z;
    }

    template <typename U>
    auto operator*(U s) const -> Child<decltype(T{} * U{})> {
        return { s * x, s * y, s * z };
    }
    template <typename U>
    Child<T> &operator*=(U s) {
        DCHECK(!isNaN(s));
        x *= s;
        y *= s;
        z *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    auto operator/(U v) const -> Child<decltype(T{} / U{})>{
        CHECK_NE(v, 0);
        return { x / v, y / v, z / v };
    }
    template <typename U>
    Child<T> &operator/=(U v) {
        CHECK_NE(v, 0);
        x /= v;
        y /= v;
        z /= v;
        return static_cast<Child<T> &>(*this);
    }
    Child<T> operator-() const { return { -x, -y, -z }; }

    std::string ToString() const {
        if (std::is_floating_point<T>::value)
            return StringPrintf("[ %f, %f, %f ]", x, y, z);
        else
            return StringPrintf("[ %d, %d, %d ]", x, y, z);
    }

    // Tuple3 Public Data
    T x, y, z;
};

template <template<class> class C, typename T>
inline std::ostream &operator<<(std::ostream &os, const Tuple3<C, T> &t) {
    return os << t.ToString();
}

template <template<class> class C, typename T>
inline C<T> Abs(const Tuple3<C, T> &v) {
    return { std::abs(v.x), std::abs(v.y), std::abs(v.z) };
}

template <template<class> class C, typename T, typename U>
inline auto operator*(U s, const Tuple3<C, T> &v) -> C<decltype(T{} * U{})> {
    return v * s;
}

template <template<class> class C, typename T>
inline T MinComponent(const Tuple3<C, T> &v) {
    return std::min(v.x, std::min(v.y, v.z));
}

template <template<class> class C, typename T>
inline T MaxComponent(const Tuple3<C, T> &v) {
    return std::max(v.x, std::max(v.y, v.z));
}

template <template<class> class C, typename T>
inline int MaxDimension(const Tuple3<C, T> &v) {
    return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
}

template <template<class> class C, typename T>
inline C<T> Permute(const Tuple3<C, T> &v, std::array<int, 3> p) {
    return { v[p[0]], v[p[1]], v[p[2]] };
}

template <template<class> class C, typename T>
inline C<T> Max(const Tuple3<C, T> &p1, const Tuple3<C, T> &p2) {
    return { std::max(p1.x, p2.x), std::max(p1.y, p2.y), std::max(p1.z, p2.z) };
}

template <template<class> class C, typename T>
inline C<T> Min(const Tuple3<C, T> &p1, const Tuple3<C, T> &p2) {
    return { std::min(p1.x, p2.x), std::min(p1.y, p2.y), std::min(p1.z, p2.z) };
}

template <template<class> class C, typename T>
inline C<T> Floor(const Tuple3<C, T> &p) {
    return { std::floor(p.x), std::floor(p.y), std::floor(p.z) };
}

template <template<class> class C, typename T>
inline C<T> Ceil(const Tuple3<C, T> &p) {
    return { std::ceil(p.x), std::ceil(p.y), std::ceil(p.z) };
}

template <template<class> class C, typename T>
inline C<T> Lerp(Float t, const Tuple3<C, T> &v0, const Tuple3<C, T> &v1) {
    return (1 - t) * v0 + t * v1;
}

} // namespace pbrt

#endif // PBRT_CORE_TUPLE_H
