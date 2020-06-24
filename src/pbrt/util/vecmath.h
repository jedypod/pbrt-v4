
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

#ifndef PBRT_GEOMETRY_TUPLE_H
#define PBRT_GEOMETRY_TUPLE_H

// vecmath.h*
#include <pbrt/pbrt.h>
#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <string>

namespace pbrt {

namespace internal {

template <typename T> std::string ToString2(T x, T y);
template <typename T> std::string ToString3(T x, T y, T z);

}

extern template std::string internal::ToString2(float, float);
extern template std::string internal::ToString2(double, double);
extern template std::string internal::ToString2(int, int);
extern template std::string internal::ToString3(float, float, float);
extern template std::string internal::ToString3(double, double, double);
extern template std::string internal::ToString3(int, int, int);

namespace {

template <typename T>
PBRT_HOST_DEVICE_INLINE
typename std::enable_if_t<std::is_floating_point<T>::value, bool>
isNaN(const T x) {
    return std::isnan(x);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
typename std::enable_if_t<std::is_integral<T>::value, bool>
 isNaN(const T x) {
    return false;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool isNaN(Interval<T> fi) {
    return isNaN(T(fi));
}

template <typename T> struct TupleLength {
    using type = Float;
};

template <> struct TupleLength<double> {
    using type = double;
};

template <> struct TupleLength<long double> {
    using type = long double;
};

template <typename T> struct TupleLength<Interval<T>> {
    using type = Interval<typename TupleLength<T>::type>;
};

template <typename T, typename U, typename TT = void> struct CommonType;

// Keep our provided overloads to just scalar types.
template <typename T, typename U>
struct CommonType<T, U, typename std::enable_if_t<std::is_arithmetic<T>::value &&
                                                  std::is_arithmetic<U>::value>> {
    using type = typename std::common_type<T, U>::type;
};

// Lets us sure that given Vector3f v and Float==float, then v * 0.5 is
// still a Vector3f and isn't promoted to a Vector3d.
template <> struct CommonType<float, double> {
    using type = Float;
};

template <> struct CommonType<double, float> {
    using type = Float;
};

template <typename T, typename U>
struct CommonType<Interval<T>, Interval<U>> {
    using type = Interval<typename CommonType<T, U>::type>;
};

template <typename T, typename U>
struct CommonType<T, Interval<U>> {
    using type = Interval<typename CommonType<T, U>::type>;
};

template <typename T, typename U>
struct CommonType<Interval<T>, U> {
    using type = Interval<typename CommonType<T, U>::type>;
};

}  // anonymous namespace

template <template<typename> class Child, typename T>
class Tuple2 {
  public:
    static const int nDimensions = 2;

    PBRT_HOST_DEVICE_INLINE
    Tuple2() { x = y = 0; }
    PBRT_HOST_DEVICE_INLINE
    Tuple2(T x, T y) : x(x), y(y) {
        DCHECK(!HasNaN());
    }
    PBRT_HOST_DEVICE_INLINE
    bool HasNaN() const { return isNaN(x) || isNaN(y); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    PBRT_HOST_DEVICE_INLINE
    Tuple2(const Child<T> &c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
    }
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator=(const Child<T> &c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        return static_cast<Child<T> &>(*this);
    }
#endif  // !NDEBUG

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator+(const Child<U> &c) const -> Child<decltype(T{} + U{})> {
        DCHECK(!c.HasNaN());
        return { x + c.x, y + c.y };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator+=(const Child<U> &c) {
        DCHECK(!c.HasNaN());
        x += c.x;
        y += c.y;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator-(const Child<U> &c) const -> Child<decltype(T{} - U{})> {
        DCHECK(!c.HasNaN());
        return { x - c.x, y - c.y };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator-=(const Child<U> &c) {
        DCHECK(!c.HasNaN());
        x -= c.x;
        y -= c.y;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const Child<T> &c) const { return x == c.x && y == c.y; }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const Child<T> &c) const { return x != c.x || y != c.y; }

    // Hadmard product
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator*(const Child<U> &c) const -> Child<decltype(T{} * U{})> {
        DCHECK(!c.HasNaN());
        return { x * c.x, y * c.y };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator*=(const Child<U> &c) {
        DCHECK(!c.HasNaN());
        x *= c.x;
        y *= c.y;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator*(U s) const -> Child<typename CommonType<T, U>::type> {
        return Child<typename CommonType<T, U>::type>(s * x, s * y);
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator*=(U s) {
        DCHECK(!isNaN(s));
        x *= s;
        y *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator/(U d) const -> Child<typename CommonType<T, U>::type> {
        DCHECK(d != 0 && !isNaN(d));
        return Child<typename CommonType<T, U>::type>(x / d, y / d);
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator/=(U d) {
        DCHECK_NE(d, 0);
        DCHECK(!isNaN(d));
        x /= d;
        y /= d;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_HOST_DEVICE_INLINE
    Child<T> operator-() const { return { -x, -y }; }

    PBRT_HOST_DEVICE_INLINE
    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 1);
        return (i == 0) ? x : y;
    }

    PBRT_HOST_DEVICE_INLINE
    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 1);
        return (i == 0) ? x : y;
    }

    std::string ToString() const {
        return internal::ToString2(x, y);
    }

    // Tuple2 Public Data
    T x, y;
};

template <template<class> class C, typename T, typename U>
PBRT_HOST_DEVICE_INLINE
auto operator*(U s, const Tuple2<C, T> &t) -> C<typename CommonType<T, U>::type> {
    DCHECK(!t.HasNaN());
    return t * s;
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Abs(const Tuple2<C, T> &t) {
    // "argument-dependent lookup..." (here and elsewhere)
    using std::abs;
    return { abs(t.x), abs(t.y) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Ceil(const Tuple2<C, T> &t) {
    using std::ceil;
    return { ceil(t.x), ceil(t.y) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Floor(const Tuple2<C, T> &t) {
    using std::floor;
    return { floor(t.x), floor(t.y) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
auto Lerp(Float t, const Tuple2<C, T> &t0, const Tuple2<C, T> &t1) {
    return (1 - t) * t0 + t * t1;
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> FMA(Float a, const Tuple2<C, T> &b, const Tuple2<C, T> &c) {
    return { FMA(a, b.x, c.x), FMA(a, b.y, c.y) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> FMA(const Tuple2<C, T> &a, Float b, const Tuple2<C, T> &c) {
    return FMA(b, a, c);
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Min(const Tuple2<C, T> &t0, const Tuple2<C, T> &t1) {
    using std::min;
    return { min(t0.x, t1.x), min(t0.y, t1.y) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
T MinComponentValue(const Tuple2<C, T> &t) {
    using std::min;
    return min({t.x, t.y});
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
int MinComponentIndex(const Tuple2<C, T> &t) {
    return (t.x < t.y) ? 0 : 1;
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Max(const Tuple2<C, T> &t0, const Tuple2<C, T> &t1) {
    using std::max;
    return { max(t0.x, t1.x), max(t0.y, t1.y) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
T MaxComponentValue(const Tuple2<C, T> &t) {
    using std::max;
    return max({t.x, t.y});
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
int MaxComponentIndex(const Tuple2<C, T> &t) {
    return (t.x > t.y) ? 0 : 1;
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Permute(const Tuple2<C, T> &t, pstd::array<int, 2> p) {
    return { t[p[0]], t[p[1]] };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
T HProd(const Tuple2<C, T> &t) {
    return t.x * t.y;
}

template <template<typename> class Child, typename T>
class Tuple3 {
  public:
    static const int nDimensions = 3;

    // Tuple3 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Tuple3() { x = y = z = 0; }
    PBRT_HOST_DEVICE_INLINE
    Tuple3(T x, T y, T z) : x(x), y(y), z(z) {
        DCHECK(!HasNaN());
    }
    PBRT_HOST_DEVICE_INLINE
    bool HasNaN() const { return isNaN(x) || isNaN(y) || isNaN(z); }
#ifndef NDEBUG
    // The default versions of these are fine for release builds; for debug
    // we define them so that we can add the Assert checks.
    PBRT_HOST_DEVICE_INLINE
    Tuple3(const Child<T> &c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        z = c.z;
    }

    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator=(const Child<T> &c) {
        DCHECK(!c.HasNaN());
        x = c.x;
        y = c.y;
        z = c.z;
        return static_cast<Child<T> &>(*this);
    }
#endif  // !NDEBUG

    PBRT_HOST_DEVICE_INLINE
    T operator[](int i) const {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }
    PBRT_HOST_DEVICE_INLINE
    T &operator[](int i) {
        DCHECK(i >= 0 && i <= 2);
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator+(const Child<U> &c) const -> Child<decltype(T{} + U{})> {
        DCHECK(!c.HasNaN());
        return { x + c.x, y + c.y, z + c.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator+=(const Child<U> &c) {
        DCHECK(!c.HasNaN());
        x += c.x;
        y += c.y;
        z += c.z;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator-(const Child<U> &c) const -> Child<decltype(T{} - U{})> {
        DCHECK(!c.HasNaN());
        return { x - c.x, y - c.y, z - c.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator-=(const Child<U> &c) {
        DCHECK(!c.HasNaN());
        x -= c.x;
        y -= c.y;
        z -= c.z;
        return static_cast<Child<T> &>(*this);
    }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const Child<T> &c) const {
        return x == c.x && y == c.y && z == c.z;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const Child<T> &c) const {
        return x != c.x || y != c.y || z != c.z;
    }

    // Hadmard product
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator*(const Child<U> &c) const -> Child<decltype(T{} * U{})> {
        DCHECK(!c.HasNaN());
        return { x * c.x, y * c.y, z * c.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator*=(const Child<U> &c) {
        DCHECK(!c.HasNaN());
        x *= c.x;
        y *= c.y;
        z *= c.z;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator*(U s) const -> Child<typename CommonType<T, U>::type> {
        return Child<typename CommonType<T, U>::type>(s * x, s * y, s * z);
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator*=(U s) {
        DCHECK(!isNaN(s));
        x *= s;
        y *= s;
        z *= s;
        return static_cast<Child<T> &>(*this);
    }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator/(U d) const -> Child<typename CommonType<T, U>::type> {
        DCHECK_NE(d, 0);
        return Child<typename CommonType<T, U>::type>(x / d, y / d, z / d);
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Child<T> &operator/=(U d) {
        DCHECK_NE(d, 0);
        x /= d;
        y /= d;
        z /= d;
        return static_cast<Child<T> &>(*this);
    }
    PBRT_HOST_DEVICE_INLINE
    Child<T> operator-() const { return { -x, -y, -z }; }

    std::string ToString() const {
        return internal::ToString3(x, y, z);
    }

    // Tuple3 Public Data
    T x, y, z;
};

template <template<class> class C, typename T, typename U>
PBRT_HOST_DEVICE_INLINE
auto operator*(U s, const Tuple3<C, T> &t) -> C<typename CommonType<T, U>::type> {
    return t * s;
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Abs(const Tuple3<C, T> &t) {
    using std::abs;
    return { abs(t.x), abs(t.y), abs(t.z) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Ceil(const Tuple3<C, T> &t) {
    using std::ceil;
    return { ceil(t.x), ceil(t.y), ceil(t.z) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Floor(const Tuple3<C, T> &t) {
    using std::floor;
    return { floor(t.x), floor(t.y), floor(t.z) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
auto Lerp(Float t, const Tuple3<C, T> &t0, const Tuple3<C, T> &t1) {
    return (1 - t) * t0 + t * t1;
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> FMA(Float a, const Tuple3<C, T> &b, const Tuple3<C, T> &c) {
    return { FMA(a, b.x, c.x), FMA(a, b.y, c.y), FMA(a, b.z, c.z) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> FMA(const Tuple3<C, T> &a, Float b, const Tuple3<C, T> &c) {
    return FMA(b, a, c);
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Min(const Tuple3<C, T> &t1, const Tuple3<C, T> &t2) {
    using std::min;
    return { min(t1.x, t2.x), min(t1.y, t2.y), min(t1.z, t2.z) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
T MinComponentValue(const Tuple3<C, T> &t) {
    using std::min;
    return min({t.x, t.y, t.z});
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
int MinComponentIndex(const Tuple3<C, T> &t) {
    return (t.x < t.y) ? ((t.x < t.z) ? 0 : 2) : ((t.y < t.z) ? 1 : 2);
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Max(const Tuple3<C, T> &t1, const Tuple3<C, T> &t2) {
    using std::max;
    return { max(t1.x, t2.x), max(t1.y, t2.y), max(t1.z, t2.z) };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
T MaxComponentValue(const Tuple3<C, T> &t) {
    using std::max;
    return max({t.x, t.y, t.z});
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
int MaxComponentIndex(const Tuple3<C, T> &t) {
    return (t.x > t.y) ? ((t.x > t.z) ? 0 : 2) : ((t.y > t.z) ? 1 : 2);
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
C<T> Permute(const Tuple3<C, T> &t, pstd::array<int, 3> p) {
    return { t[p[0]], t[p[1]], t[p[2]] };
}

template <template<class> class C, typename T>
PBRT_HOST_DEVICE_INLINE
T HProd(const Tuple3<C, T> &t) {
    return t.x * t.y * t.z;
}

template <typename T>
class Vector2 : public Tuple2<Vector2, T> {
  public:
    // IMHO (but not sure) all of the "::template Vector2" et
    // al. throughout here is unnecessary / a clang bug.
#ifndef PBRT_CRTP_USING_WORKAROUND
    // WAS: using Tuple2<Vector2::template Vector2, T>::x;
    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;
#else
    using Tuple2::x;
    using Tuple2::y;
#endif

    // Vector2 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Vector2() { }
    PBRT_HOST_DEVICE_INLINE
    Vector2(T x, T y) : Tuple2<pbrt::Vector2, T>(x, y) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Vector2(const Point2<U> &p);
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Vector2(const Vector2<U> &v)
        : Tuple2<pbrt::Vector2, T>(T(v.x), T(v.y)) { }
};

template <typename T>
class Vector3 : public Tuple3<Vector3, T> {
  public:
#ifndef PBRT_CRTP_USING_WORKAROUND
    using Tuple3<Vector3, T>::x;
    using Tuple3<Vector3, T>::y;
    using Tuple3<Vector3, T>::z;
#else
    using Tuple3::x;
    using Tuple3::y;
    using Tuple3::z;
#endif

    PBRT_HOST_DEVICE_INLINE
    Vector3() { }
    PBRT_HOST_DEVICE_INLINE
    Vector3(T x, T y, T z) : Tuple3<pbrt::Vector3, T>(x, y, z) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Vector3(const Point3<U> &p);
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Vector3(const Normal3<U> &n);
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Vector3(const Vector3<U> &v)
        : Tuple3<pbrt::Vector3, T>(T(v.x), T(v.y), T(v.z)) { }
};

using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;
using Vector3i = Vector3<int>;

class Vector3fi : public Vector3<Interval<Float>> {
public:
#ifndef PBRT_CRTP_USING_WORKAROUND
    using Tuple3<Vector3, Interval<Float>>::x;
    using Tuple3<Vector3, Interval<Float>>::y;
    using Tuple3<Vector3, Interval<Float>>::z;
    using Tuple3<Vector3, Interval<Float>>::HasNaN;
    using Tuple3<Vector3, Interval<Float>>::operator+;
    using Tuple3<Vector3, Interval<Float>>::operator+=;
    using Tuple3<Vector3, Interval<Float>>::operator*;
    using Tuple3<Vector3, Interval<Float>>::operator*=;
#else
    using Tuple3::x;
    using Tuple3::y;
    using Tuple3::z;
    using Tuple3::HasNaN;
    using Tuple3::operator+;
    using Tuple3::operator+=;
    using Tuple3::operator*;
    using Tuple3::operator*=;
#endif
    Vector3fi() = default;
    PBRT_HOST_DEVICE_INLINE
    Vector3fi(Float x, Float y, Float z)
        : Vector3<Interval<Float>>(Interval<Float>(x), Interval<Float>(y),
                                   Interval<Float>(z)) { }
    PBRT_HOST_DEVICE_INLINE
    Vector3fi(FloatInterval x, FloatInterval y, FloatInterval z)
        : Vector3<Interval<Float>>(x, y, z) { }
    PBRT_HOST_DEVICE_INLINE
    Vector3fi(const Vector3f &p)
        : Vector3<Interval<Float>>(Interval<Float>(p.x), Interval<Float>(p.y),
                                   Interval<Float>(p.z)) { }
    template <typename F>
    PBRT_HOST_DEVICE_INLINE
    Vector3fi(const Vector3<Interval<F>> &pfi)
        : Vector3<Interval<Float>>(pfi) { }
    PBRT_HOST_DEVICE_INLINE
    Vector3fi(const Vector3f &p, const Vector3f &e)
        : Vector3<Interval<Float>>(Interval<Float>::FromValueAndError(p.x, e.x),
                                   Interval<Float>::FromValueAndError(p.y, e.y),
                                   Interval<Float>::FromValueAndError(p.z, e.z)) { }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Error() const { return { x.Width() / 2, y.Width() / 2, z.Width() / 2 }; }
    PBRT_HOST_DEVICE_INLINE
    bool IsExact() const { return x.Width() == 0 && y.Width() == 0 && z.Width() == 0; }
};

// Point Declarations
template <typename T>
class Point2 : public Tuple2<Point2, T> {
  public:
#ifndef PBRT_CRTP_USING_WORKAROUND
    using Tuple2<Point2, T>::x;
    using Tuple2<Point2, T>::y;
    using Tuple2<Point2, T>::HasNaN;
    using Tuple2<Point2, T>::operator+;
    using Tuple2<Point2, T>::operator+=;
    using Tuple2<Point2, T>::operator*;
    using Tuple2<Point2, T>::operator*=;
#else
    using Tuple2::x;
    using Tuple2::y;
    using Tuple2::HasNaN;
    using Tuple2::operator+;
    using Tuple2::operator+=;
    using Tuple2::operator*;
    using Tuple2::operator*=;
#endif

    // Point2 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Point2() { x = y = 0; }
    PBRT_HOST_DEVICE_INLINE
    Point2(T x, T y) : Tuple2<pbrt::Point2, T>(x, y) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Point2(const Point2<U> &v)
        : Tuple2<pbrt::Point2, T>(T(v.x), T(v.y)) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Point2(const Vector2<U> &v)
        : Tuple2<pbrt::Point2, T>(T(v.x), T(v.y)) { }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator+(const Vector2<U> &v) const -> Point2<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return { x + v.x, y + v.y };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point2<T> &operator+=(const Vector2<U> &v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        return *this;
    }

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...
    PBRT_HOST_DEVICE_INLINE
    Point2<T> operator-() const { return { -x, -y }; }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator-(const Point2<U> &p) const -> Vector2<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return { x - p.x, y - p.y };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator-(const Vector2<U> &v) const -> Point2<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return { x - v.x, y - v.y };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point2<T> &operator-=(const Vector2<U> &v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        return *this;
    }
};

template <typename T>
class Point3 : public Tuple3<Point3, T> {
  public:
#ifndef PBRT_CRTP_USING_WORKAROUND
    using Tuple3<Point3, T>::x;
    using Tuple3<Point3, T>::y;
    using Tuple3<Point3, T>::z;
    using Tuple3<Point3, T>::HasNaN;
    using Tuple3<Point3, T>::operator+;
    using Tuple3<Point3, T>::operator+=;
    using Tuple3<Point3, T>::operator*;
    using Tuple3<Point3, T>::operator*=;
#else
    using Tuple3::x;
    using Tuple3::y;
    using Tuple3::z;
    using Tuple3::HasNaN;
    using Tuple3::operator+;
    using Tuple3::operator+=;
    using Tuple3::operator*;
    using Tuple3::operator*=;
#endif

    // Point3 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Point3() { x = y = z = 0; }
    PBRT_HOST_DEVICE_INLINE
    Point3(T x, T y, T z) : Tuple3<pbrt::Point3, T>(x, y, z) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Point3(const Vector3<U> &v)
        : Tuple3<pbrt::Point3, T>(T(v.x), T(v.y), T(v.z)) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Point3(const Point3<U> &p)
        : Tuple3<pbrt::Point3, T>(T(p.x), T(p.y), T(p.z)) { }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator+(const Vector3<U> &v) const -> Point3<decltype(T{} + U{})> {
        DCHECK(!v.HasNaN());
        return { x + v.x, y + v.y, z + v.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3<T> &operator+=(const Vector3<U> &v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    // We can't do using operator- above, since we don't want to pull in
    // the Point-Point -> Point one so that we can return a vector
    // instead...
    PBRT_HOST_DEVICE_INLINE
    Point3<T> operator-() const { return { -x, -y, -z }; }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator-(const Point3<U> &p) const -> Vector3<decltype(T{} - U{})> {
        DCHECK(!p.HasNaN());
        return { x - p.x, y - p.y, z - p.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    auto operator-(const Vector3<U> &v) const -> Point3<decltype(T{} - U{})> {
        DCHECK(!v.HasNaN());
        return { x - v.x, y - v.y, z - v.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3<T> &operator-=(const Vector3<U> &v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
};

using Point2f = Point2<Float>;
using Point2i = Point2<int>;
using Point3f = Point3<Float>;
using Point3i = Point3<int>;

class Point3fi : public Point3<Interval<Float>> {
public:
#ifndef PBRT_CRTP_USING_WORKAROUND
    using Tuple3<Point3, Interval<Float>>::x;
    using Tuple3<Point3, Interval<Float>>::y;
    using Tuple3<Point3, Interval<Float>>::z;
    using Tuple3<Point3, Interval<Float>>::HasNaN;
    using Tuple3<Point3, Interval<Float>>::operator+;
    using Tuple3<Point3, Interval<Float>>::operator+=;
    using Tuple3<Point3, Interval<Float>>::operator*;
    using Tuple3<Point3, Interval<Float>>::operator*=;
#else
    using Tuple3::x;
    using Tuple3::y;
    using Tuple3::z;
    using Tuple3::HasNaN;
    using Tuple3::operator+;
    using Tuple3::operator+=;
    using Tuple3::operator*;
    using Tuple3::operator*=;
#endif
    Point3fi() = default;
    PBRT_HOST_DEVICE_INLINE
    Point3fi(FloatInterval x, FloatInterval y, FloatInterval z)
        : Point3<Interval<Float>>(x, y, z) { }
    PBRT_HOST_DEVICE_INLINE
    Point3fi(Float x, Float y, Float z)
        : Point3<Interval<Float>>(Interval<Float>(x), Interval<Float>(y),
                                  Interval<Float>(z)) { }
    PBRT_HOST_DEVICE_INLINE
    Point3fi(const Point3f &p)
        : Point3<Interval<Float>>(Interval<Float>(p.x), Interval<Float>(p.y),
                                  Interval<Float>(p.z)) { }
    template <typename F>
    PBRT_HOST_DEVICE_INLINE
    Point3fi(const Point3<Interval<F>> &pfi)
        : Point3<Interval<Float>>(pfi) { }
    PBRT_HOST_DEVICE_INLINE
    Point3fi(const Point3f &p, const Vector3f &e)
        : Point3<Interval<Float>>(Interval<Float>::FromValueAndError(p.x, e.x),
                                  Interval<Float>::FromValueAndError(p.y, e.y),
                                  Interval<Float>::FromValueAndError(p.z, e.z)) { }

    PBRT_HOST_DEVICE_INLINE
    Vector3f Error() const { return { x.Width() / 2, y.Width() / 2, z.Width() / 2 }; }
    PBRT_HOST_DEVICE_INLINE
    bool IsExact() const { return x.Width() == 0 && y.Width() == 0 && z.Width() == 0; }

    // Meh--can't seem to get these from Point3 via using declarations...
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3fi operator+(const Vector3<U> &v) const {
        DCHECK(!v.HasNaN());
        return { x + v.x, y + v.y, z + v.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3fi &operator+=(const Vector3<U> &v) {
        DCHECK(!v.HasNaN());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    PBRT_HOST_DEVICE_INLINE
    Point3fi operator-() const { return { -x, -y, -z }; }

    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3fi operator-(const Point3<U> &p) const {
        DCHECK(!p.HasNaN());
        return { x - p.x, y - p.y, z - p.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3fi operator-(const Vector3<U> &v) const {
        DCHECK(!v.HasNaN());
        return { x - v.x, y - v.y, z - v.z };
    }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    Point3fi &operator-=(const Vector3<U> &v) {
        DCHECK(!v.HasNaN());
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
};

// Normal Declarations
template <typename T>
class Normal3 : public Tuple3<Normal3, T> {
  public:
#ifndef PBRT_CRTP_USING_WORKAROUND
    using Tuple3<Normal3, T>::x;
    using Tuple3<Normal3, T>::y;
    using Tuple3<Normal3, T>::z;
    using Tuple3<Normal3, T>::HasNaN;
    using Tuple3<Normal3, T>::operator+;
    using Tuple3<Normal3, T>::operator*;
    using Tuple3<Normal3, T>::operator*=;
#else
    using Tuple3::x;
    using Tuple3::y;
    using Tuple3::z;
    using Tuple3::HasNaN;
    using Tuple3::operator+;
    using Tuple3::operator*;
    using Tuple3::operator*=;
#endif

    // Normal3 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Normal3() { x = y = z = 0; }
    PBRT_HOST_DEVICE_INLINE
    Normal3(T x, T y, T z) : Tuple3<pbrt::Normal3, T>(x, y, z) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Normal3<T>(const Normal3<U> &v)
        : Tuple3<pbrt::Normal3, T>(T(v.x), T(v.y), T(v.z)) { }
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Normal3<T>(const Vector3<U> &v)
        : Tuple3<pbrt::Normal3, T>(T(v.x), T(v.y), T(v.z)) { }
};

using Normal3f = Normal3<Float>;

// Geometry Functions
template <typename T> template <typename U>
Vector2<T>::Vector2(const Point2<U> &p)
    : Tuple2<pbrt::Vector2, T>(T(p.x), T(p.y)) { }

// TODO: book discuss why Dot() and not e.g. a.Dot(b)
template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Dot(const Vector2<T> &v1, const Vector2<T> &v2) -> typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return SumOfProducts(v1.x, v2.x, v1.y, v2.y);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AbsDot(const Vector2<T> &v1, const Vector2<T> &v2) -> typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return std::abs(Dot(v1, v2));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto LengthSquared(const Vector2<T> &v) -> typename TupleLength<T>::type {
    return Sqr(v.x) + Sqr(v.y);
 }

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Length(const Vector2<T> &v) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(v));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Normalize(const Vector2<T> &v) -> Vector2<typename TupleLength<T>::type> {
    return v / Length(v);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Distance(const Point2<T> &p1, const Point2<T> &p2) -> typename TupleLength<T>::type {
    return Length(p1 - p2);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto DistanceSquared(const Point2<T> &p1, const Point2<T> &p2) -> typename TupleLength<T>::type {
    return LengthSquared(p1 - p2);
}

template <typename T> template <typename U>
Vector3<T>::Vector3(const Point3<U> &p)
    : Tuple3<pbrt::Vector3, T>(T(p.x), T(p.y), T(p.z)) { }

template <typename T> template <typename U>
Vector3<T>::Vector3(const Normal3<U> &n)
    : Tuple3<pbrt::Vector3, T>(T(n.x), T(n.y), T(n.z)) { }

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Dot(const Vector3<T> &v1, const Vector3<T> &v2) -> typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return FMA(v1.x, v2.x, SumOfProducts(v1.y, v2.y, v1.z, v2.z));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) -> typename TupleLength<T>::type {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return std::abs(Dot(v1, v2));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Vector3<T> Cross(const Vector3<T> &v1, const Normal3<T> &v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Vector3<T> Cross(const Normal3<T> &v1, const Vector3<T> &v2) {
    DCHECK(!v1.HasNaN() && !v2.HasNaN());
    return {DifferenceOfProducts(v1.y, v2.z, v1.z, v2.y),
            DifferenceOfProducts(v1.z, v2.x, v1.x, v2.z),
            DifferenceOfProducts(v1.x, v2.y, v1.y, v2.x)};
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto LengthSquared(const Vector3<T> &v) -> typename TupleLength<T>::type {
    return Sqr(v.x) + Sqr(v.y) + Sqr(v.z);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Length(const Vector3<T> &v) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(v));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Normalize(const Vector3<T> &v) -> Vector3<typename TupleLength<T>::type> {
    return v / Length(v);
}

// Equivalent to std::acos(Dot(a, b)), but more numerically stable.
// http://www.plunk.org/~hatch/rightway.php
template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AngleBetween(const Vector3<T> &a, const Vector3<T> &b) -> typename TupleLength<T>::type {
#if 1
    if (Dot(a, b) < 0)
        return Pi - 2 * SafeASin(Length(a + b) / 2);
    else
        return 2 * SafeASin(Length(b - a) / 2);
#else
    // Alternative from Kahan, How Futile are Mindless Assessments of Roundoff
    // in Floating-Point Computation ?, p. 46-47.

    // From random testing, it seems to give slightly better average error
    // (1.5x-3x better), but with slightly higher maximum error
    // (1.25-2.5x).  They're both plenty good, but maximum error seems like
    // the more important metric...
    Float la = Length(a), lb = Length(b);
    return 2 * std::atan(Length(a * lb - la * b) / Length(a * lb + la * b));
#endif
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AngleBetween(const Normal3<T> &a, const Normal3<T> &b) -> typename TupleLength<T>::type {
#if 1
    if (Dot(a, b) < 0)
        return Pi - 2 * SafeASin(Length(a + b) / 2);
    else
        return 2 * SafeASin(Length(b - a) / 2);
#else
    Float la = Length(a), lb = Length(b);
    return 2 * std::atan(Length(a * lb - la * b) / Length(a * lb + la * b));
#endif
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2,
                             Vector3<T> *v3) {
    // Was: Hughes-Moller 99
    // Now: Duff et al 2017
    Float sign = std::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + v1.y * v1.y * a, -v1.y);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
void CoordinateSystem(const Normal3<T> &v1, Vector3<T> *v2,
                             Vector3<T> *v3) {
    // Was: Hughes-Moller 99
    // Now: Duff et al 2017
    Float sign = std::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + v1.y * v1.y * a, -v1.y);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Distance(const Point3<T> &p1, const Point3<T> &p2) -> typename TupleLength<T>::type {
    return Length(p1 - p2);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto DistanceSquared(const Point3<T> &p1, const Point3<T> &p2) -> typename TupleLength<T>::type {
    return LengthSquared(p1 - p2);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto LengthSquared(const Normal3<T> &n) -> typename TupleLength<T>::type {
    return Sqr(n.x) + Sqr(n.y) + Sqr(n.z);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Length(const Normal3<T> &n) -> typename TupleLength<T>::type {
    using std::sqrt;
    return sqrt(LengthSquared(n));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Normalize(const Normal3<T> &n) -> Normal3<typename TupleLength<T>::type> {
    return n / Length(n);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Dot(const Normal3<T> &n, const Vector3<T> &v) -> typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Dot(const Vector3<T> &v, const Normal3<T> &n) -> typename TupleLength<T>::type {
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return FMA(n.x, v.x, SumOfProducts(n.y, v.y, n.z, v.z));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto Dot(const Normal3<T> &n1, const Normal3<T> &n2) -> typename TupleLength<T>::type {
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return FMA(n1.x, n2.x, SumOfProducts(n1.y, n2.y, n1.z, n2.z));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AbsDot(const Normal3<T> &n, const Vector3<T> &v) -> typename TupleLength<T>::type {
    DCHECK(!n.HasNaN() && !v.HasNaN());
    return std::abs(Dot(n, v));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AbsDot(const Vector3<T> &v, const Normal3<T> &n) -> typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!v.HasNaN() && !n.HasNaN());
    return abs(Dot(v, n));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
auto AbsDot(const Normal3<T> &n1, const Normal3<T> &n2) -> typename TupleLength<T>::type {
    using std::abs;
    DCHECK(!n1.HasNaN() && !n2.HasNaN());
    return abs(Dot(n1, n2));
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Normal3<T> FaceForward(const Normal3<T> &n, const Vector3<T> &v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Normal3<T> FaceForward(const Normal3<T> &n, const Normal3<T> &n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Vector3<T> FaceForward(const Vector3<T> &v, const Vector3<T> &v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Vector3<T> FaceForward(const Vector3<T> &v, const Normal3<T> &n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

// Quaternion Declarations
struct Quaternion {
    // Quaternion Public Methods
    PBRT_HOST_DEVICE_INLINE
    Quaternion() : v(0, 0, 0), w(1) {}
    PBRT_HOST_DEVICE_INLINE
    Quaternion(const Vector3f &v, Float w) : v(v), w(w) {}

    PBRT_HOST_DEVICE_INLINE
    Quaternion &operator+=(const Quaternion &q) {
        v += q.v;
        w += q.w;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion operator+(const Quaternion &q) const {
        return {v + q.v, w + q.w};
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion &operator-=(const Quaternion &q) {
        v -= q.v;
        w -= q.w;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion operator-() const {
        return {-v, -w};
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion operator-(const Quaternion &q) const {
        return {v - q.v, w - q.w};
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion &operator*=(Float f) {
        v *= f;
        w *= f;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion operator*(Float f) const {
        return {v * f, w * f};
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion &operator/=(Float f) {
        DCHECK_NE(0, f);
        v /= f;
        w /= f;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    Quaternion operator/(Float f) const {
        DCHECK_NE(0, f);
        return {v / f, w / f};
    }

    std::string ToString() const;

    // Quaternion Public Data
    Vector3f v;
    Float w;
};

// Quaternion Functions
PBRT_HOST_DEVICE_INLINE
Quaternion operator*(Float f, const Quaternion &q) { return q * f; }

PBRT_HOST_DEVICE_INLINE
Float Dot(const Quaternion &q1, const Quaternion &q2) {
    return Dot(q1.v, q2.v) + q1.w * q2.w;
}

PBRT_HOST_DEVICE_INLINE
Float Length(const Quaternion &q) {
    return std::sqrt(Dot(q, q));
}

PBRT_HOST_DEVICE_INLINE
Quaternion Normalize(const Quaternion &q) {
    DCHECK_GT(Length(q), 0);
    return q / Length(q);
}

PBRT_HOST_DEVICE_INLINE
Quaternion Slerp(Float t, const Quaternion &q1, const Quaternion &q2) {
    // http://www.plunk.org/~hatch/rightway.php
    // First, robust "angle between" computation.
    Float theta = (Dot(q1, q2) < 0) ? (Pi - 2 * SafeASin(Length(q1 + q2) / 2)) :
        (2 * SafeASin(Length(q2 - q1) / 2));

    // Now, rewrite slerp(t, q1, q2) = sin((1-t)*theta)/sin(theta)*q1 +
    // sin(t*theta)/sin(theta)*q2 by multiplying the first term by
    // (theta*(1-t))/(theta*(1-t)) and the second term by
    // ((t*theta)/(t*theta)).  In turn, we can replace the sin() calls,
    // which head towards 0/0 when theta approaches 0, with robust sin(x)/x
    // calls.
    Float sinThetaOverTheta = SinXOverX(theta);
    return ((SinXOverX((1 - t) * theta) / sinThetaOverTheta) * (1 - t) * q1 +
            (SinXOverX(t * theta) / sinThetaOverTheta) * t * q2);
}

// Bounds Declarations
template <typename T>
class Bounds2 {
  public:
    // Bounds2 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Bounds2() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Point2<T>(maxNum, maxNum);
        pMax = Point2<T>(minNum, minNum);
    }
    PBRT_HOST_DEVICE_INLINE
    explicit Bounds2(const Point2<T> &p) : pMin(p), pMax(p) {}
    PBRT_HOST_DEVICE_INLINE
    Bounds2(const Point2<T> &p1, const Point2<T> &p2)
        : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Bounds2(const Bounds2<U> &b) {
        if (b.IsEmpty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds2<T>();
        else {
            pMin = Point2<T>(b.pMin);
            pMax = Point2<T>(b.pMax);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    Vector2<T> Diagonal() const { return pMax - pMin; }

    PBRT_HOST_DEVICE_INLINE
    T Area() const {
        Vector2<T> d = pMax - pMin;
        return d.x * d.y;
    }

    PBRT_HOST_DEVICE_INLINE
    bool IsEmpty() const {
        return pMin.x >= pMax.x || pMin.y >= pMax.y;
    }

    PBRT_HOST_DEVICE_INLINE
    bool IsDegenerate() const {
        return pMin.x > pMax.x || pMin.y > pMax.y;
    }

    PBRT_HOST_DEVICE_INLINE
    int MaxDimension() const {
        Vector2<T> diag = Diagonal();
        if (diag.x > diag.y)
            return 0;
        else
            return 1;
    }
    PBRT_HOST_DEVICE_INLINE
    const Point2<T> &operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    Point2<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator==(const Bounds2<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const Bounds2<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    Point2<T> Corner(int corner) const {
        DCHECK(corner >= 0 && corner < 4);
        return Point2<T>((*this)[(corner & 1)].x,
                         (*this)[(corner & 2) ? 1 : 0].y);
    }
    PBRT_HOST_DEVICE_INLINE
    Point2<T> Lerp(const Point2f &t) const {
        return Point2<T>(pbrt::Lerp(t.x, pMin.x, pMax.x),
                         pbrt::Lerp(t.y, pMin.y, pMax.y));
    }
    PBRT_HOST_DEVICE_INLINE
    Vector2<T> Offset(const Point2<T> &p) const {
        Vector2<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        return o;
    }
    PBRT_HOST_DEVICE_INLINE
    void BoundingSphere(Point2<T> *c, Float *rad) const {
        *c = (pMin + pMax) / 2;
        *rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
    }

    std::string ToString() const {
        return StringPrintf("[ %s - %s ]", pMin, pMax);
    }

    // Bounds2 Public Data
    Point2<T> pMin, pMax;
};

template <typename T>
class Bounds3 {
  public:
    // Bounds3 Public Methods
    PBRT_HOST_DEVICE_INLINE
    Bounds3() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Point3<T>(maxNum, maxNum, maxNum);
        pMax = Point3<T>(minNum, minNum, minNum);
    }
    PBRT_HOST_DEVICE_INLINE
    explicit Bounds3(const Point3<T> &p) : pMin(p), pMax(p) {}
    PBRT_HOST_DEVICE_INLINE
    Bounds3(const Point3<T> &p1, const Point3<T> &p2)
        : pMin(Min(p1, p2)), pMax(Max(p1, p2)) {}
    template <typename U>
    PBRT_HOST_DEVICE_INLINE
    explicit Bounds3(const Bounds3<U> &b) {
        if (b.IsEmpty())
            // Be careful about overflowing float->int conversions and the
            // like.
            *this = Bounds3<T>();
        else {
            pMin = Point3<T>(b.pMin);
            pMax = Point3<T>(b.pMax);
        }
    }

    PBRT_HOST_DEVICE_INLINE
    const Point3<T> &operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    Point3<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const Bounds3<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const Bounds3<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    PBRT_HOST_DEVICE_INLINE
    Point3<T> Corner(int corner) const {
        DCHECK(corner >= 0 && corner < 8);
        return Point3<T>((*this)[(corner & 1)].x,
                         (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }
    PBRT_HOST_DEVICE_INLINE
    Vector3<T> Diagonal() const { return pMax - pMin; }
    PBRT_HOST_DEVICE_INLINE
    T SurfaceArea() const {
        Vector3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    PBRT_HOST_DEVICE_INLINE
    T Volume() const {
        Vector3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }
    // Tricky: IsEmpty() means zero volume, not necessarily zero surface
    // area.
    PBRT_HOST_DEVICE_INLINE
    bool IsEmpty() const {
        return pMin.x >= pMax.x || pMin.y >= pMax.y || pMin.z >= pMax.z;
    }
    PBRT_HOST_DEVICE_INLINE
    bool IsDegenerate() const {
        return pMin.x > pMax.x || pMin.y > pMax.y || pMin.z > pMax.z;
    }
    PBRT_HOST_DEVICE_INLINE
    int MaxDimension() const {
        Vector3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
    PBRT_HOST_DEVICE_INLINE
    Point3<T> Lerp(const Point3f &t) const {
        return Point3<T>(pbrt::Lerp(t.x, pMin.x, pMax.x),
                         pbrt::Lerp(t.y, pMin.y, pMax.y),
                         pbrt::Lerp(t.z, pMin.z, pMax.z));
    }
    PBRT_HOST_DEVICE_INLINE
    Vector3<T> Offset(const Point3<T> &p) const {
        Vector3<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
    PBRT_HOST_DEVICE_INLINE
    void BoundingSphere(Point3<T> *center, Float *radius) const {
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }
    PBRT_HOST_DEVICE_INLINE
    bool IntersectP(const Point3f &o, const Vector3f &d, Float tMax = Infinity,
                    Float *hitt0 = nullptr, Float *hitt1 = nullptr) const;
    PBRT_HOST_DEVICE_INLINE
    bool IntersectP(const Point3f &o, const Vector3f &d, Float tMax,
                    const Vector3f &invDir, const int dirIsNeg[3]) const;

    std::string ToString() const {
        return StringPrintf("[ %s - %s ]", pMin, pMax);
    }

    // Bounds3 Public Data
    Point3<T> pMin, pMax;
};

using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;

class Bounds2iIterator : public std::forward_iterator_tag {
  public:
    PBRT_HOST_DEVICE_INLINE
    Bounds2iIterator(const Bounds2i &b, const Point2i &pt)
        : p(pt), bounds(&b) {}
    PBRT_HOST_DEVICE_INLINE
    Bounds2iIterator operator++() {
        advance();
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    Bounds2iIterator operator++(int) {
        Bounds2iIterator old = *this;
        advance();
        return old;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator==(const Bounds2iIterator &bi) const {
        return p == bi.p && bounds == bi.bounds;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const Bounds2iIterator &bi) const {
        return p != bi.p || bounds != bi.bounds;
    }

    PBRT_HOST_DEVICE_INLINE
    Point2i operator*() const { return p; }

  private:
    PBRT_HOST_DEVICE_INLINE
    void advance() {
        ++p.x;
        if (p.x == bounds->pMax.x) {
            p.x = bounds->pMin.x;
            ++p.y;
        }
    }
    Point2i p;
    const Bounds2i *bounds;
};

template <typename T>
PBRT_HOST_DEVICE_INLINE
Bounds3<T> Union(const Bounds3<T> &b, const Point3<T> &p) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds3<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds3<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds3<T> b;
    b.pMin = Max(b1.pMin, b2.pMin);
    b.pMax = Min(b1.pMax, b2.pMax);
    return b;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Inside(const Point3<T> &p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x &&
            p.y >= b.pMin.y && p.y <= b.pMax.y &&
            p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Inside(const Bounds3<T> &ba, const Bounds3<T> &bb) {
    return (ba.pMin.x >= bb.pMin.x && ba.pMax.x <= bb.pMax.x &&
            ba.pMin.y >= bb.pMin.y && ba.pMay.y <= bb.pMay.y &&
            ba.pMin.z >= bb.pMin.z && ba.pMay.z <= bb.pMay.z);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool InsideExclusive(const Point3<T> &p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x < b.pMax.x &&
            p.y >= b.pMin.y && p.y < b.pMax.y &&
            p.z >= b.pMin.z && p.z < b.pMax.z);
}

template <typename T, typename U>
PBRT_HOST_DEVICE_INLINE
Bounds3<T> Expand(const Bounds3<T> &b, U delta) {
    Bounds3<T> ret;
    ret.pMin = b.pMin - Vector3<T>(delta, delta, delta);
    ret.pMax = b.pMax + Vector3<T>(delta, delta, delta);
    return ret;
}

// Minimum squared distance from point to box; returns zero if point is
// inside.
template <typename T, typename U>
PBRT_HOST_DEVICE_INLINE
Float DistanceSquared(const Point3<T> &p, const Bounds3<U> &b) {
    Float dx = std::max<Float>({0, b.pMin.x - p.x, p.x - b.pMax.x});
    Float dy = std::max<Float>({0, b.pMin.y - p.y, p.y - b.pMax.y});
    Float dz = std::max<Float>({0, b.pMin.z - p.z, p.z - b.pMax.z});
    return dx * dx + dy * dy + dz * dz;
}

template <typename T, typename U>
PBRT_HOST_DEVICE_INLINE
Float Distance(const Point3<T> &p, const Bounds3<U> &b) {
    return std::sqrt(DistanceSquared(p, b));
}

PBRT_HOST_DEVICE_INLINE
Bounds2iIterator begin(const Bounds2i &b) {
    return Bounds2iIterator(b, b.pMin);
}

PBRT_HOST_DEVICE_INLINE
Bounds2iIterator end(const Bounds2i &b) {
    // Normally, the ending point is at the minimum x value and one past
    // the last valid y value.
    Point2i pEnd(b.pMin.x, b.pMax.y);
    // However, if the bounds are degenerate, override the end point to
    // equal the start point so that any attempt to iterate over the bounds
    // exits out immediately.
    if (b.pMin.x >= b.pMax.x || b.pMin.y >= b.pMax.y)
        pEnd = b.pMin;
    return Bounds2iIterator(b, pEnd);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Bounds2<T> Union(const Bounds2<T> &b, const Point2<T> &p) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pMin = Min(b.pMin, p);
    ret.pMax = Max(b.pMax, p);
    return ret;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Bounds2<T> Union(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> ret;
    ret.pMin = Min(b1.pMin, b2.pMin);
    ret.pMax = Max(b1.pMax, b2.pMax);
    return ret;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
Bounds2<T> Intersect(const Bounds2<T> &b1, const Bounds2<T> &b2) {
    // Be careful to not run the two-point Bounds constructor.
    Bounds2<T> b;
    b.pMin = Max(b1.pMin, b2.pMin);
    b.pMax = Min(b1.pMax, b2.pMax);
    return b;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
    bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
    return (x && y);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Inside(const Point2<T> &pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x <= b.pMax.x &&
            pt.y >= b.pMin.y && pt.y <= b.pMax.y);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Inside(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    return (ba.pMin.x >= bb.pMin.x && ba.pMax.x <= bb.pMax.x &&
            ba.pMin.y >= bb.pMin.y && ba.pMax.y <= bb.pMax.y);
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool InsideExclusive(const Point2<T> &pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x < b.pMax.x &&
            pt.y >= b.pMin.y && pt.y < b.pMax.y);
}

template <typename T, typename U>
PBRT_HOST_DEVICE_INLINE
Bounds2<T> Expand(const Bounds2<T> &b, U delta) {
    Bounds2<T> ret;
    ret.pMin = b.pMin - Vector2<T>(delta, delta);
    ret.pMax = b.pMax + Vector2<T>(delta, delta);
    return ret;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Bounds3<T>::IntersectP(const Point3f &o, const Vector3f &d, Float tMax,
                            Float *hitt0, Float *hitt1) const {
    Float t0 = 0, t1 = tMax;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        Float invRayDir = 1 / d[i];
        Float tNear = (pMin[i] - o[i]) * invRayDir;
        Float tFar = (pMax[i] - o[i]) * invRayDir;

        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar) pstd::swap(tNear, tFar);

        // Update _tFar_ to ensure robust ray--bounds intersection
        tFar *= 1 + 2 * gamma(3);
        t0 = tNear > t0 ? tNear : t0;
        t1 = tFar < t1 ? tFar : t1;
        if (t0 > t1) return false;
    }
    if (hitt0) *hitt0 = t0;
    if (hitt1) *hitt1 = t1;
    return true;
}

template <typename T>
PBRT_HOST_DEVICE_INLINE
bool Bounds3<T>::IntersectP(const Point3f &o, const Vector3f &d, Float raytMax,
                                   const Vector3f &invDir, const int dirIsNeg[3]) const {
    const Bounds3f &bounds = *this;
    // Check for ray intersection against $x$ and $y$ slabs
    Float tMin = (bounds[dirIsNeg[0]].x - o.x) * invDir.x;
    Float tMax = (bounds[1 - dirIsNeg[0]].x - o.x) * invDir.x;
    Float tyMin = (bounds[dirIsNeg[1]].y - o.y) * invDir.y;
    Float tyMax = (bounds[1 - dirIsNeg[1]].y - o.y) * invDir.y;

    // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
    tMax *= 1 + 2 * gamma(3);
    tyMax *= 1 + 2 * gamma(3);
    if (tMin > tyMax || tyMin > tMax) return false;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;

    // Check for ray intersection against $z$ slab
    Float tzMin = (bounds[dirIsNeg[2]].z - o.z) * invDir.z;
    Float tzMax = (bounds[1 - dirIsNeg[2]].z - o.z) * invDir.z;

    // Update _tzMax_ to ensure robust bounds intersection
    tzMax *= 1 + 2 * gamma(3);
    if (tMin > tzMax || tzMin > tMax) return false;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;
    return (tMin < raytMax) && (tMax > 0);
}


PBRT_HOST_DEVICE_INLINE
Float CosTheta(const Vector3f &w) { return w.z; }
PBRT_HOST_DEVICE_INLINE
Float Cos2Theta(const Vector3f &w) { return w.z * w.z; }
PBRT_HOST_DEVICE_INLINE
Float AbsCosTheta(const Vector3f &w) { return std::abs(w.z); }
PBRT_HOST_DEVICE_INLINE
Float Sin2Theta(const Vector3f &w) {
    return std::max<Float>(0, 1 - Cos2Theta(w));
}

PBRT_HOST_DEVICE_INLINE
Float SinTheta(const Vector3f &w) { return std::sqrt(Sin2Theta(w)); }

PBRT_HOST_DEVICE_INLINE
Float TanTheta(const Vector3f &w) { return SinTheta(w) / CosTheta(w); }

PBRT_HOST_DEVICE_INLINE
Float Tan2Theta(const Vector3f &w) {
    return Sin2Theta(w) / Cos2Theta(w);
}

PBRT_HOST_DEVICE_INLINE
Float CosPhi(const Vector3f &w) {
    Float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 1 : Clamp(w.x / sinTheta, -1, 1);
}

PBRT_HOST_DEVICE_INLINE
Float SinPhi(const Vector3f &w) {
    Float sinTheta = SinTheta(w);
    return (sinTheta == 0) ? 0 : Clamp(w.y / sinTheta, -1, 1);
}

PBRT_HOST_DEVICE_INLINE
Float Cos2Phi(const Vector3f &w) { return CosPhi(w) * CosPhi(w); }

PBRT_HOST_DEVICE_INLINE
Float Sin2Phi(const Vector3f &w) { return SinPhi(w) * SinPhi(w); }

PBRT_HOST_DEVICE_INLINE
Float CosDPhi(const Vector3f &wa, const Vector3f &wb) {
    return Clamp(
        (wa.x * wb.x + wa.y * wb.y) / std::sqrt((wa.x * wa.x + wa.y * wa.y) *
                                                (wb.x * wb.x + wb.y * wb.y)),
        -1, 1);
}

PBRT_HOST_DEVICE_INLINE
bool SameHemisphere(const Vector3f &w, const Vector3f &wp) {
    return w.z * wp.z > 0;
}

PBRT_HOST_DEVICE_INLINE
bool SameHemisphere(const Vector3f &w, const Normal3f &wp) {
    return w.z * wp.z > 0;
}

PBRT_HOST_DEVICE_INLINE
Vector3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi) {
    DCHECK(sinTheta >= -1.0001 && sinTheta <= 1.0001);
    sinTheta = Clamp(sinTheta, -1, 1);
    DCHECK(cosTheta >= -1.0001 && cosTheta <= 1.0001);
    cosTheta = Clamp(cosTheta, -1, 1);
    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                    cosTheta);
}

PBRT_HOST_DEVICE_INLINE
Float SphericalTheta(const Vector3f &v) {
    // More robust than std::acos(v.z)
    return 2 * std::asin(.5f * std::sqrt(Sqr(v.x) + Sqr(v.y) + Sqr(v.z - 1)));
}

PBRT_HOST_DEVICE_INLINE
Float SphericalPhi(const Vector3f &v) {
    Float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

PBRT_HOST_DEVICE_INLINE
Float SphericalTriangleArea(const Vector3f &a, const Vector3f &b,
                            const Vector3f &c) {
    // http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
    // Girard's theorem: surface area of a spherical triangle on a unit
    // sphere is the 'excess angle' alpha+beta+gamma-pi, where
    // alpha/beta/gamma are the interior angles at the vertices.
    //
    // Given three vertices on the sphere, a, b, c, then we can compute,
    // for example, the angle c->a->b by
    //
    // cos theta =  Dot(Cross(c, a), Cross(b, a)) /
    //              (Length(Cross(c, a)) * Length(Cross(b, a))).
    //
    // We only need to do three cross products to evaluate the angles at
    // all three vertices, though, since we can take advantage of the fact
    // that Cross(a, b) = -Cross(b, a).
    Vector3f axb = Cross(a, b), bxc = Cross(b, c), cxa = Cross(c, a);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 || LengthSquared(cxa) == 0)
        return 0;
    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxa = Normalize(cxa);

    Float alpha = AngleBetween(cxa, -axb);
    Float beta = AngleBetween(axb, -bxc);
    Float gamma = AngleBetween(bxc, -cxa);

    return std::abs(alpha + beta + gamma - Pi);
}

// Note: if it folds over itself, the total spherical area is returned.
// (i.e. sort of not what we'd want...)
// https://math.stackexchange.com/questions/1228964/area-of-spherical-polygon
PBRT_HOST_DEVICE_INLINE
Float SphericalQuadArea(const Vector3f &a, const Vector3f &b,
                        const Vector3f &c, const Vector3f &d) {
    Vector3f axb = Cross(a, b), bxc = Cross(b, c);
    Vector3f cxd = Cross(c, d), dxa = Cross(d, a);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 ||
        LengthSquared(cxd) == 0 || LengthSquared(dxa) == 0)
        return 0;
    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxd = Normalize(cxd);
    dxa = Normalize(dxa);

    Float alpha = AngleBetween(dxa, -axb);
    Float beta = AngleBetween(axb, -bxc);
    Float gamma = AngleBetween(bxc, -cxd);
    Float delta = AngleBetween(cxd, -dxa);

    return std::abs(alpha + beta + gamma + delta - 2 * Pi);
}

PBRT_HOST_DEVICE
Float SphericalQuadArea(const Vector3f &v0, const Vector3f &v1,
                        const Vector3f &v2, const Vector3f &v3);

PBRT_HOST_DEVICE_INLINE
Point2f ToCylindrical(const Vector3f &v) {
    Float phi = std::atan2(v.y, v.x);
    if (phi < 0) phi += 2 * Pi;
    return {Clamp(v.z, -1, 1), phi};
}

PBRT_HOST_DEVICE_INLINE
Vector3f FromCylindrical(const Point2f &c) {
    Float cosTheta = c[0], sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    return SphericalDirection(sinTheta, cosTheta, c[1]);
}

class DirectionCone {
 public:
    DirectionCone() = default;
    // TODO: Require w to be normalized?
    PBRT_HOST_DEVICE_INLINE
    DirectionCone(const Vector3f &w, Float cosTheta)
        : w(Normalize(w)),
          cosTheta(cosTheta) {
          DCHECK(cosTheta >= -1 && cosTheta <= 1);
    }
    // Single direction
    PBRT_HOST_DEVICE_INLINE
    explicit DirectionCone(const Vector3f &w)
        : DirectionCone(w, 1) { }

    PBRT_HOST_DEVICE_INLINE
    static DirectionCone EntireSphere() {
        return DirectionCone(Vector3f(0,0,1), -1);
    }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    Vector3f ClosestVectorInCone(Vector3f wp) const;

    Vector3f w = Vector3f(0, 0, 1);
    Float cosTheta = -1;
};

// TODO: require normalized w?
PBRT_HOST_DEVICE_INLINE
bool Inside(const DirectionCone &d, const Vector3f &w) {
    return Dot(d.w, Normalize(w)) >= d.cosTheta;
}

// Make this a Bounds3f method?
PBRT_HOST_DEVICE_INLINE
DirectionCone BoundSubtendedDirections(const Bounds3f &b, const Point3f &p) {
    Float radius;
    Point3f pCenter;
    b.BoundingSphere(&pCenter, &radius);
    DirectionCone cSphere;
    if (DistanceSquared(p, pCenter) < radius * radius)
        return DirectionCone::EntireSphere();

    Vector3f w = Normalize(pCenter - p);
    Float sinThetaMax2 = radius * radius / DistanceSquared(pCenter, p);
    Float cosThetaMax = SafeSqrt(1 - sinThetaMax2);
    return DirectionCone(w, cosThetaMax);
}

PBRT_HOST_DEVICE_INLINE
Vector3f DirectionCone::ClosestVectorInCone(Vector3f wp) const {
    wp = Normalize(wp);
    if (Dot(wp, w) > cosTheta)
        // in cone already
        return wp;

    Float sinTheta = -SafeSqrt(1 - cosTheta * cosTheta);
    // closest vector to cone.nb
    // Take the rotation matrix around Normalize(a), then apply it to w,
    // simplify, and we end up with this.
    Vector3f a = Cross(wp, w);
    return cosTheta * w + (sinTheta / Length(a)) *
        Vector3f(w.x * (wp.y * w.y + wp.z * w.z) - wp.x * (Sqr(w.y) + Sqr(w.z)),
                 w.y * (wp.x * w.x + wp.z * w.z) - wp.y * (Sqr(w.x) + Sqr(w.z)),
                 w.z * (wp.x * w.x + wp.y * w.y) - wp.z * (Sqr(w.x) + Sqr(w.y)));
}

PBRT_HOST_DEVICE
DirectionCone Union(const DirectionCone &a, const DirectionCone &b);

class Frame {
 public:
    PBRT_HOST_DEVICE_INLINE
    Frame()
        : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
    PBRT_HOST_DEVICE_INLINE
    Frame(const Vector3f &x, const Vector3f &y, const Vector3f &z);

    PBRT_HOST_DEVICE_INLINE
    static Frame FromXZ(const Vector3f &x, const Vector3f &z);
    PBRT_HOST_DEVICE_INLINE
    static Frame FromXY(const Vector3f &x, const Vector3f &y);
    PBRT_HOST_DEVICE_INLINE
    static Frame FromZ(const Vector3f &n);
    PBRT_HOST_DEVICE_INLINE
    static Frame FromZ(const Normal3f &n);

    PBRT_HOST_DEVICE_INLINE
    Vector3f ToLocal(const Vector3f &v) const;
    PBRT_HOST_DEVICE_INLINE
    Vector3f FromLocal(const Vector3f &v) const {
        return v.x * x + v.y * y + v.z * z;
    }
    PBRT_HOST_DEVICE_INLINE
    Normal3f ToLocal(const Normal3f &n) const;
    PBRT_HOST_DEVICE_INLINE
    Normal3f FromLocal(const Normal3f &n) const {
        return Normal3f(n.x * x + n.y * y + n.z * z);
    }

    std::string ToString() const {
        return StringPrintf("[ Frame x: %s y: %s z: %s ]", x, y, z);
    }

    Vector3f x, y, z;
};

Frame::Frame(const Vector3f &x, const Vector3f &y, const Vector3f &z)
     : x(x), y(y), z(z) {
     DCHECK_LT(std::abs(LengthSquared(x) - 1), 1e-4);
     DCHECK_LT(std::abs(LengthSquared(y) - 1), 1e-4);
     DCHECK_LT(std::abs(LengthSquared(z) - 1), 1e-4);
     DCHECK_LT(std::abs(Dot(x, y)), 1e-4);
     DCHECK_LT(std::abs(Dot(y, z)), 1e-4);
     DCHECK_LT(std::abs(Dot(z, x)), 1e-4);
}

Frame Frame::FromXZ(const Vector3f &x, const Vector3f &z) {
    return Frame(x, Cross(z, x), z);
}

Frame Frame::FromXY(const Vector3f &x, const Vector3f &y) {
    return Frame(x, y, Cross(x, y));
}

Vector3f Frame::ToLocal(const Vector3f &v) const {
    return Vector3f(Dot(v, x), Dot(v, y), Dot(v, z));
}

Normal3f Frame::ToLocal(const Normal3f &n) const {
    return Normal3f(Dot(n, x), Dot(n, y), Dot(n, z));
}

Frame Frame::FromZ(const Vector3f &z) {
    Vector3f x, y;
    CoordinateSystem(z, &x, &y);
    return Frame(x, y, z);
}

Frame Frame::FromZ(const Normal3f &z) {
    return FromZ(Vector3f(z));
}

}  // namespace pbrt

#endif  // PBRT_GEOMETRY_FRAME_H
