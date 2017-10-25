
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

#ifndef PBRT_UTIL_GEOMETRY_H
#define PBRT_UTIL_GEOMETRY_H

// core/geometry.h*
#include <pbrt/core/pbrt.h>

#include <pbrt/util/math.h>
#include <pbrt/util/stringprint.h>
#include <pbrt/util/tuple.h>

namespace pbrt {

template <typename T>
class Vector2 : public Tuple2<Vector2, T> {
  public:
    // IMHO (but not sure) all of the "::template Vector2" et
    // al. throughout here is unnecessary / a clang bug.
    using Tuple2<Vector2::template Vector2, T>::x;
    using Tuple2<Vector2::template Vector2, T>::y;

    // Vector2 Public Methods
    Vector2() { }
    Vector2(T x, T y) : Tuple2<pbrt::Vector2, T>(x, y) { }
    template <typename U> explicit Vector2(const Point2<U> &p);
    template <typename U> explicit Vector2(const Vector2<U> &v)
        : Tuple2<pbrt::Vector2, T>(T(v.x), T(v.y)) { }

    template <typename U> explicit operator Point2<U>() const;
    template <typename U> explicit operator Vector2<U>() const {
        return Vector2<U>(x, y);
    }
};

template <typename T>
class Vector3 : public Tuple3<Vector3, T> {
  public:
    using Tuple3<Vector3::template Vector3, T>::x;
    using Tuple3<Vector3::template Vector3, T>::y;
    using Tuple3<Vector3::template Vector3, T>::z;

    Vector3() { }
    Vector3(T x, T y, T z) : Tuple3<pbrt::Vector3, T>(x, y, z) { }
    explicit Vector3(const Point3<T> &p);
    explicit Vector3(const Normal3<T> &n);
    template <typename U> explicit Vector3(const Vector3<U> &v)
        : Tuple3<pbrt::Vector3, T>(T(v.x), T(v.y), T(v.z)) { }

    template <typename U> explicit operator Point3<U>() const;
    template <typename U> explicit operator Vector3<U>() const {
        return Vector3<U>(x, y, z);
    }
};

using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;
using Vector3i = Vector3<int>;

// Point Declarations
template <typename T>
class Point2 : public Tuple2<Point2, T> {
  public:
    using Tuple2<Point2::template Point2, T>::x;
    using Tuple2<Point2::template Point2, T>::y;
    using Tuple2<Point2::template Point2, T>::HasNaNs;
    using Tuple2<Point2::template Point2, T>::operator+;

    // Point2 Public Methods
    Point2() { x = y = 0; }
    Point2(T xx, T yy) : Tuple2<pbrt::Point2, T>(xx, yy) { }
    template <typename U> explicit Point2(const Point2<U> &v)
        : Tuple2<pbrt::Point2, T>(T(v.x), T(v.y)) { }
    template <typename U> explicit Point2(const Vector2<U> &v)
        : Tuple2<pbrt::Point2, T>(v.x, v.y) { }

    template <typename U>
    explicit operator Point2<U>() const {
        return Point2<U>(x, y);
    }
    template <typename U>
    explicit operator Vector2<U>() const {
        return Vector2<U>(x, y);
    }

    template <typename U>
    auto operator+(const Vector2<U> &v) const -> Point2<decltype(T{}+U{})> {
        DCHECK(!v.HasNaNs());
        return { x + v.x, y + v.y };
    }
    template <typename U>
    Point2<T> &operator+=(const Vector2<U> &v) {
        DCHECK(!v.HasNaNs());
        x += v.x;
        y += v.y;
        return *this;
    }

    template <typename U>
    auto operator-(const Point2<U> &p) const -> Vector2<decltype(T{}-U{})> {
        DCHECK(!p.HasNaNs());
        return { x - p.x, y - p.y };
    }
    template <typename U>
    auto operator-(const Vector2<U> &v) const -> Point2<decltype(T{}-U{})> {
        DCHECK(!v.HasNaNs());
        return { x - v.x, y - v.y };
    }
    template <typename U>
    Point2<T> &operator-=(const Vector2<U> &v) {
        DCHECK(!v.HasNaNs());
        x -= v.x;
        y -= v.y;
        return *this;
    }
};

template <typename T>
class Point3 : public Tuple3<Point3, T> {
  public:
    using Tuple3<Point3::template Point3, T>::x;
    using Tuple3<Point3::template Point3, T>::y;
    using Tuple3<Point3::template Point3, T>::z;
    using Tuple3<Point3::template Point3, T>::HasNaNs;
    using Tuple3<Point3::template Point3, T>::operator+;
    using Tuple3<Point3::template Point3, T>::operator+=;

    // Point3 Public Methods
    Point3() { x = y = z = 0; }
    Point3(T x, T y, T z) : Tuple3<pbrt::Point3, T>(x, y, z) { }
    template <typename U> explicit Point3(const Vector3<U> &v)
        : Tuple3<pbrt::Point3, T>(T(v.x), T(v.y), T(v.z)) { }
    template <typename U> explicit Point3(const Point3<U> &p)
        : Tuple3<pbrt::Point3, T>(T(p.x), T(p.y), T(p.z)) { }

    template <typename U>
    explicit operator Point3<U>() const {
        return Point3<U>(x, y, z);
    }
    template <typename U>
    explicit operator Vector3<U>() const {
        return Vector3<U>(x, y, z);
    }

    Point3<T> operator+(const Vector3<T> &v) const {
        DCHECK(!v.HasNaNs());
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }
    Point3<T> &operator+=(const Vector3<T> &v) {
        DCHECK(!v.HasNaNs());
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    Vector3<T> operator-(const Point3<T> &p) const {
        DCHECK(!p.HasNaNs());
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }
    Point3<T> operator-(const Vector3<T> &v) const {
        DCHECK(!v.HasNaNs());
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }
    Point3<T> &operator-=(const Vector3<T> &v) {
        DCHECK(!v.HasNaNs());
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

// Normal Declarations
template <typename T>
class Normal3 : public Tuple3<Normal3, T> {
  public:
    using Tuple3<Normal3::template Normal3, T>::x;
    using Tuple3<Normal3::template Normal3, T>::y;
    using Tuple3<Normal3::template Normal3, T>::z;
    using Tuple3<Normal3::template Normal3, T>::HasNaNs;
    using Tuple3<Normal3::template Normal3, T>::operator+;

    // Normal3 Public Methods
    Normal3() { x = y = z = 0; }
    Normal3(T x, T y, T z) : Tuple3<pbrt::Normal3, T>(x, y, z) { }
    explicit Normal3<T>(const Vector3<T> &v)
        : Tuple3<pbrt::Normal3, T>(v.x, v.y, v.z) { }
};

using Normal3f = Normal3<Float>;

// Ray Declarations
class Ray {
  public:
    // Ray Public Methods
    Ray() : tMax(Infinity), time(0.f), medium(nullptr) {}
    Ray(const Point3f &o, const Vector3f &d, Float tMax = Infinity,
        Float time = 0.f, const Medium *medium = nullptr)
        : o(o), d(d), tMax(tMax), time(time), medium(medium) {}
    Point3f operator()(Float t) const { return o + d * t; }
    bool HasNaNs() const { return (o.HasNaNs() || d.HasNaNs() || isNaN(tMax)); }

    std::string ToString() const;

    // Ray Public Data
    Point3f o;
    Vector3f d;
    mutable Float tMax;
    Float time;
    const Medium *medium;
};

inline std::ostream &operator<<(std::ostream &os, const Ray &r) {
    return os << r.ToString();
}

class RayDifferential : public Ray {
  public:
    // RayDifferential Public Methods
    RayDifferential() { hasDifferentials = false; }
    RayDifferential(const Point3f &o, const Vector3f &d, Float tMax = Infinity,
                    Float time = 0.f, const Medium *medium = nullptr)
        : Ray(o, d, tMax, time, medium) {
        hasDifferentials = false;
    }
    RayDifferential(const Ray &ray) : Ray(ray) { hasDifferentials = false; }
    bool HasNaNs() const {
        return Ray::HasNaNs() ||
               (hasDifferentials &&
                (rxOrigin.HasNaNs() || ryOrigin.HasNaNs() ||
                 rxDirection.HasNaNs() || ryDirection.HasNaNs()));
    }
    void ScaleDifferentials(Float s) {
        rxOrigin = o + (rxOrigin - o) * s;
        ryOrigin = o + (ryOrigin - o) * s;
        rxDirection = d + (rxDirection - d) * s;
        ryDirection = d + (ryDirection - d) * s;
    }
    std::string ToString() const;

    // RayDifferential Public Data
    bool hasDifferentials;
    Point3f rxOrigin, ryOrigin;
    Vector3f rxDirection, ryDirection;
};

inline std::ostream &operator<<(std::ostream &os, const RayDifferential &r) {
    return os << r.ToString();
}

class Frame {
 public:
    Frame()
        : x(1, 0, 0), y(0, 1, 0), z(0, 0, 1) {}
    Frame(const Vector3f &x, const Vector3f &y, const Vector3f &z);

    static Frame FromXZ(const Vector3f &x, const Vector3f &z);
    Vector3f ToLocal(const Vector3f &v) const;
    Vector3f FromLocal(const Vector3f &v) const {
        return v.x * x + v.y * y + v.z * z;
    }

 private:
    Vector3f x, y, z;
};

// Geometry Inline Functions
template <typename T> template <typename U>
Vector2<T>::operator Point2<U>() const {
    return Point2<U>(x, y);
}

template <typename T> template <typename U>
Vector3<T>::operator Point3<U>() const {
    return Point3<U>(x, y, z);
}

template <typename T>
inline Vector3<T>::Vector3(const Point3<T> &p)
    : Tuple3<pbrt::Vector3, T>(p.x, p.y, p.z) { }

template <typename T>
inline T Dot(const Vector3<T> &v1, const Vector3<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

template <typename T>
inline T AbsDot(const Vector3<T> &v1, const Vector3<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    return std::abs(Dot(v1, v2));
}

template <typename T>
inline Vector3<T> Cross(const Vector3<T> &v1, const Vector3<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    double v1x = v1.x, v1y = v1.y, v1z = v1.z;
    double v2x = v2.x, v2y = v2.y, v2z = v2.z;
    return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
                      (v1x * v2y) - (v1y * v2x));
}

template <typename T>
inline Vector3<T> Cross(const Vector3<T> &v1, const Normal3<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    double v1x = v1.x, v1y = v1.y, v1z = v1.z;
    double v2x = v2.x, v2y = v2.y, v2z = v2.z;
    return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
                      (v1x * v2y) - (v1y * v2x));
}

template <typename T>
inline Vector3<T> Cross(const Normal3<T> &v1, const Vector3<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    double v1x = v1.x, v1y = v1.y, v1z = v1.z;
    double v2x = v2.x, v2y = v2.y, v2z = v2.z;
    return Vector3<T>((v1y * v2z) - (v1z * v2y), (v1z * v2x) - (v1x * v2z),
                      (v1x * v2y) - (v1y * v2x));
}

template <typename T> T LengthSquared(const Vector3<T> &v) {
    return Dot(v, v);
 }

template <typename T> T Length(const Vector3<T> &v) {
    return std::sqrt(LengthSquared(v));
}

template <typename T>
inline Vector3<T> Normalize(const Vector3<T> &v) {
    return v / Length(v);
}

// Equivalent to std::acos(Dot(a, b)), but more numerically stable.
// http://www.plunk.org/~hatch/rightway.php
template <typename T>
inline T AngleBetween(const Vector3<T> &a, const Vector3<T> &b) {
    if (Dot(a, b) < 0)
        return Pi - 2 * SafeASin(Length(a + b) / 2);
    else
        return 2 * SafeASin(Length(b - a) / 2);
}

template <typename T>
inline void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2,
                             Vector3<T> *v3) {
    // Was: Hughes-Moller 99
    // Now: Duff et al 2017
    Float sign = std::copysign(Float(1), v1.z);
    Float a = -1 / (sign + v1.z);
    Float b = v1.x * v1.y * a;
    *v2 = Vector3<T>(1 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);
    *v3 = Vector3<T>(b, sign + v1.y * v1.y * a, -v1.y);
}

template <typename T> template <typename U>
Vector2<T>::Vector2(const Point2<U> &p)
    : Tuple2<pbrt::Vector2, T>(p.x, p.y) { }

template <typename T>
inline Float Dot(const Vector2<T> &v1, const Vector2<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    return v1.x * v2.x + v1.y * v2.y;
}

template <typename T>
inline Float AbsDot(const Vector2<T> &v1, const Vector2<T> &v2) {
    DCHECK(!v1.HasNaNs() && !v2.HasNaNs());
    return std::abs(Dot(v1, v2));
}

template <typename T> T LengthSquared(const Vector2<T> &v) {
    return Dot(v, v);
 }

template <typename T> T Length(const Vector2<T> &v) {
    return std::sqrt(LengthSquared(v));
}

template <typename T>
inline Vector2<T> Normalize(const Vector2<T> &v) {
    return v / Length(v);
}

template <typename T>
inline Float Distance(const Point3<T> &p1, const Point3<T> &p2) {
    return Length(p1 - p2);
}

template <typename T>
inline T DistanceSquared(const Point3<T> &p1, const Point3<T> &p2) {
    return LengthSquared(p1 - p2);
}

template <typename T>
inline Float Distance(const Point2<T> &p1, const Point2<T> &p2) {
    return Length(p1 - p2);
}

template <typename T>
inline T DistanceSquared(const Point2<T> &p1, const Point2<T> &p2) {
    return LengthSquared(p1 - p2);
}

template <typename T> T LengthSquared(const Normal3<T> &n) {
    return Dot(n, n);
 }

template <typename T> T Length(const Normal3<T> &n) {
    return std::sqrt(LengthSquared(n));
}

template <typename T>
inline Normal3<T> Normalize(const Normal3<T> &n) {
    return n / Length(n);
}

template <typename T>
inline Vector3<T>::Vector3(const Normal3<T> &n)
    : Tuple3<pbrt::Vector3, T>(n.x, n.y, n.z) { }

template <typename T>
inline T Dot(const Normal3<T> &n1, const Vector3<T> &v2) {
    DCHECK(!n1.HasNaNs() && !v2.HasNaNs());
    return n1.x * v2.x + n1.y * v2.y + n1.z * v2.z;
}

template <typename T>
inline T Dot(const Vector3<T> &v1, const Normal3<T> &n2) {
    DCHECK(!v1.HasNaNs() && !n2.HasNaNs());
    return v1.x * n2.x + v1.y * n2.y + v1.z * n2.z;
}

template <typename T>
inline T Dot(const Normal3<T> &n1, const Normal3<T> &n2) {
    DCHECK(!n1.HasNaNs() && !n2.HasNaNs());
    return n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
}

template <typename T>
inline T AbsDot(const Normal3<T> &n1, const Vector3<T> &v2) {
    DCHECK(!n1.HasNaNs() && !v2.HasNaNs());
    return std::abs(n1.x * v2.x + n1.y * v2.y + n1.z * v2.z);
}

template <typename T>
inline T AbsDot(const Vector3<T> &v1, const Normal3<T> &n2) {
    DCHECK(!v1.HasNaNs() && !n2.HasNaNs());
    return std::abs(v1.x * n2.x + v1.y * n2.y + v1.z * n2.z);
}

template <typename T>
inline T AbsDot(const Normal3<T> &n1, const Normal3<T> &n2) {
    DCHECK(!n1.HasNaNs() && !n2.HasNaNs());
    return std::abs(n1.x * n2.x + n1.y * n2.y + n1.z * n2.z);
}

template <typename T>
inline Normal3<T> Faceforward(const Normal3<T> &n, const Vector3<T> &v) {
    return (Dot(n, v) < 0.f) ? -n : n;
}

template <typename T>
inline Normal3<T> Faceforward(const Normal3<T> &n, const Normal3<T> &n2) {
    return (Dot(n, n2) < 0.f) ? -n : n;
}

template <typename T>
inline Vector3<T> Faceforward(const Vector3<T> &v, const Vector3<T> &v2) {
    return (Dot(v, v2) < 0.f) ? -v : v;
}

template <typename T>
inline Vector3<T> Faceforward(const Vector3<T> &v, const Normal3<T> &n2) {
    return (Dot(v, n2) < 0.f) ? -v : v;
}

inline Point3f OffsetRayOrigin(const Point3f &p, const Vector3f &pError,
                               const Normal3f &n, const Vector3f &w) {
    Float d = Dot(Abs(n), pError);
#ifdef PBRT_FLOAT_AS_DOUBLE
    // We have tons of precision; for now bump up the offset a bunch just
    // to be extra sure that we start on the right side of the surface
    // (In case of any bugs in the epsilons code...)
    d *= 1024.;
#endif
    Vector3f offset = d * Vector3f(n);
    if (Dot(w, n) < 0) offset = -offset;
    Point3f po = p + offset;
    // Round offset point _po_ away from _p_
    for (int i = 0; i < 3; ++i) {
        if (offset[i] > 0)
            po[i] = NextFloatUp(po[i]);
        else if (offset[i] < 0)
            po[i] = NextFloatDown(po[i]);
    }
    return po;
}

inline Vector3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi) {
    DCHECK(sinTheta >= -1.0001 && sinTheta <= 1.0001);
    sinTheta = Clamp(sinTheta, -1, 1);
    DCHECK(cosTheta >= -1.0001 && cosTheta <= 1.0001);
    cosTheta = Clamp(cosTheta, -1, 1);
    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                    cosTheta);
}

inline Vector3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi,
                                   const Vector3f &x, const Vector3f &y,
                                   const Vector3f &z) {
    DCHECK(sinTheta >= -1.0001 && sinTheta <= 1.0001);
    sinTheta = Clamp(sinTheta, -1, 1);
    DCHECK(cosTheta >= -1.0001 && cosTheta <= 1.0001);
    cosTheta = Clamp(cosTheta, -1, 1);
    return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y +
           cosTheta * z;
}

inline Float SphericalTheta(const Vector3f &v) {
    return SafeACos(v.z);
}

inline Float SphericalPhi(const Vector3f &v) {
    Float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

inline Frame::Frame(const Vector3f &x, const Vector3f &y, const Vector3f &z)
     : x(x), y(y), z(z) {
     DCHECK_LT(std::abs(LengthSquared(x) - 1), 1e-4);
     DCHECK_LT(std::abs(LengthSquared(y) - 1), 1e-4);
     DCHECK_LT(std::abs(LengthSquared(z) - 1), 1e-4);
     DCHECK_LT(std::abs(Dot(x, y)), 1e-4);
     DCHECK_LT(std::abs(Dot(y, z)), 1e-4);
     DCHECK_LT(std::abs(Dot(z, x)), 1e-4);
}

inline Frame Frame::FromXZ(const Vector3f &x, const Vector3f &z) {
    return Frame(x, Cross(z, x), z);
}

inline Vector3f Frame::ToLocal(const Vector3f &v) const {
    return Vector3f(Dot(v, x), Dot(v, y), Dot(v, z));
}

}  // namespace pbrt

#endif  // PBRT_UTIL_GEOMETRY_H
