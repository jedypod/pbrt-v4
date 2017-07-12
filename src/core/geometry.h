
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

#ifndef PBRT_CORE_GEOMETRY_H
#define PBRT_CORE_GEOMETRY_H

// core/geometry.h*
#include "pbrt.h"
#include "stringprint.h"
#include "tuple.h"
#include <iterator>

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
    Vector2(T x, T y) : Tuple2<Vector2::template Vector2, T>(x, y) { }
    template <typename U> explicit Vector2(const Point2<U> &p);
    template <typename U> explicit Vector2(const Vector2<U> &v)
        : Tuple2<Vector2::template Vector2, T>(T(v.x), T(v.y)) { }

    T LengthSquared() const { return x * x + y * y; }
    Float Length() const { return std::sqrt(LengthSquared()); }
};

template <typename T>
class Vector3 : public Tuple3<Vector3, T> {
  public:
    using Tuple3<Vector3::template Vector3, T>::x;
    using Tuple3<Vector3::template Vector3, T>::y;
    using Tuple3<Vector3::template Vector3, T>::z;

    Vector3() { }
    Vector3(T x, T y, T z) : Tuple3<Vector3::template Vector3, T>(x, y, z) { }
    explicit Vector3(const Point3<T> &p);
    explicit Vector3(const Normal3<T> &n);

    T LengthSquared() const { return x * x + y * y + z * z; }
    Float Length() const { return std::sqrt(LengthSquared()); }
};

typedef Vector2<Float> Vector2f;
typedef Vector2<int> Vector2i;
typedef Vector3<Float> Vector3f;
typedef Vector3<int> Vector3i;

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
    Point2(T xx, T yy) : Tuple2<Point2::template Point2, T>(xx, yy) { }
    template <typename U>
    explicit Point2(const Point2<U> &p) {
        x = (T)p.x;
        y = (T)p.y;
        DCHECK(!HasNaNs());
    }

    template <typename U>
    explicit Point2(const Vector2<U> &p) {
        x = (T)p.x;
        y = (T)p.y;
        DCHECK(!HasNaNs());
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
    Point3(T x, T y, T z) : Tuple3<Point3::template Point3, T>(x, y, z) { }
    template <typename U>
    explicit Point3(const Point3<U> &p)
        : Tuple3<Point3::template Point3, T>((T)p.x, (T)p.y, (T)p.z) { }
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

typedef Point2<Float> Point2f;
typedef Point2<int> Point2i;
typedef Point3<Float> Point3f;
typedef Point3<int> Point3i;

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
    Normal3(T x, T y, T z) : Tuple3<Normal3::template Normal3, T>(x, y, z) { }
    explicit Normal3<T>(const Vector3<T> &v)
        : Tuple3<Normal3::template Normal3, T>(v.x, v.y, v.z) { }

    T LengthSquared() const { return x * x + y * y + z * z; }
    Float Length() const { return std::sqrt(LengthSquared()); }
};

typedef Normal3<Float> Normal3f;

// Bounds Declarations
template <typename T>
class Bounds2 {
  public:
    // Bounds2 Public Methods
    Bounds2() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Point2<T>(maxNum, maxNum);
        pMax = Point2<T>(minNum, minNum);
    }
    explicit Bounds2(const Point2<T> &p) : pMin(p), pMax(p) {}
    Bounds2(const Point2<T> &p1, const Point2<T> &p2) {
        pMin = Point2<T>(std::min(p1.x, p2.x), std::min(p1.y, p2.y));
        pMax = Point2<T>(std::max(p1.x, p2.x), std::max(p1.y, p2.y));
    }
    template <typename U>
    explicit operator Bounds2<U>() const {
        return Bounds2<U>((Point2<U>)pMin, (Point2<U>)pMax);
    }

    Vector2<T> Diagonal() const { return pMax - pMin; }
    T Area() const {
        Vector2<T> d = pMax - pMin;
        return (d.x * d.y);
    }
    int MaximumExtent() const {
        Vector2<T> diag = Diagonal();
        if (diag.x > diag.y)
            return 0;
        else
            return 1;
    }
    inline const Point2<T> &operator[](int i) const {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    inline Point2<T> &operator[](int i) {
        DCHECK(i == 0 || i == 1);
        return (i == 0) ? pMin : pMax;
    }
    bool operator==(const Bounds2<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    bool operator!=(const Bounds2<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    Point2<T> Lerp(const Point2f &t) const {
        return Point2<T>(pbrt::Lerp(t.x, pMin.x, pMax.x),
                         pbrt::Lerp(t.y, pMin.y, pMax.y));
    }
    Vector2<T> Offset(const Point2<T> &p) const {
        Vector2<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        return o;
    }
    void BoundingSphere(Point2<T> *c, Float *rad) const {
        *c = (pMin + pMax) / 2;
        *rad = Inside(*c, *this) ? Distance(*c, pMax) : 0;
    }
    friend std::ostream &operator<<(std::ostream &os, const Bounds2<T> &b) {
        os << "[ " << b.pMin << " - " << b.pMax << " ]";
        return os;
    }

    // Bounds2 Public Data
    Point2<T> pMin, pMax;
};

template <typename T>
class Bounds3 {
  public:
    // Bounds3 Public Methods
    Bounds3() {
        T minNum = std::numeric_limits<T>::lowest();
        T maxNum = std::numeric_limits<T>::max();
        pMin = Point3<T>(maxNum, maxNum, maxNum);
        pMax = Point3<T>(minNum, minNum, minNum);
    }
    explicit Bounds3(const Point3<T> &p) : pMin(p), pMax(p) {}
    Bounds3(const Point3<T> &p1, const Point3<T> &p2)
        : pMin(std::min(p1.x, p2.x), std::min(p1.y, p2.y),
               std::min(p1.z, p2.z)),
          pMax(std::max(p1.x, p2.x), std::max(p1.y, p2.y),
               std::max(p1.z, p2.z)) {}
    const Point3<T> &operator[](int i) const;
    Point3<T> &operator[](int i);
    bool operator==(const Bounds3<T> &b) const {
        return b.pMin == pMin && b.pMax == pMax;
    }
    bool operator!=(const Bounds3<T> &b) const {
        return b.pMin != pMin || b.pMax != pMax;
    }
    Point3<T> Corner(int corner) const {
        DCHECK(corner >= 0 && corner < 8);
        return Point3<T>((*this)[(corner & 1)].x,
                         (*this)[(corner & 2) ? 1 : 0].y,
                         (*this)[(corner & 4) ? 1 : 0].z);
    }
    Vector3<T> Diagonal() const { return pMax - pMin; }
    T SurfaceArea() const {
        Vector3<T> d = Diagonal();
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z);
    }
    T Volume() const {
        Vector3<T> d = Diagonal();
        return d.x * d.y * d.z;
    }
    int MaximumExtent() const {
        Vector3<T> d = Diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
    Point3<T> Lerp(const Point3f &t) const {
        return Point3<T>(pbrt::Lerp(t.x, pMin.x, pMax.x),
                         pbrt::Lerp(t.y, pMin.y, pMax.y),
                         pbrt::Lerp(t.z, pMin.z, pMax.z));
    }
    Vector3<T> Offset(const Point3<T> &p) const {
        Vector3<T> o = p - pMin;
        if (pMax.x > pMin.x) o.x /= pMax.x - pMin.x;
        if (pMax.y > pMin.y) o.y /= pMax.y - pMin.y;
        if (pMax.z > pMin.z) o.z /= pMax.z - pMin.z;
        return o;
    }
    void BoundingSphere(Point3<T> *center, Float *radius) const {
        *center = (pMin + pMax) / 2;
        *radius = Inside(*center, *this) ? Distance(*center, pMax) : 0;
    }
    template <typename U>
    explicit operator Bounds3<U>() const {
        return Bounds3<U>((Point3<U>)pMin, (Point3<U>)pMax);
    }
    bool IntersectP(const Ray &ray, Float *hitt0 = nullptr,
                    Float *hitt1 = nullptr) const;
    inline bool IntersectP(const Ray &ray, const Vector3f &invDir,
                           const int dirIsNeg[3]) const;
    friend std::ostream &operator<<(std::ostream &os, const Bounds3<T> &b) {
        os << "[ " << b.pMin << " - " << b.pMax << " ]";
        return os;
    }

    // Bounds3 Public Data
    Point3<T> pMin, pMax;
};

typedef Bounds2<Float> Bounds2f;
typedef Bounds2<int> Bounds2i;
typedef Bounds3<Float> Bounds3f;
typedef Bounds3<int> Bounds3i;

class Bounds2iIterator : public std::forward_iterator_tag {
  public:
    Bounds2iIterator(const Bounds2i &b, const Point2i &pt)
        : p(pt), bounds(&b) {}
    Bounds2iIterator operator++() {
        advance();
        return *this;
    }
    Bounds2iIterator operator++(int) {
        Bounds2iIterator old = *this;
        advance();
        return old;
    }
    bool operator==(const Bounds2iIterator &bi) const {
        return p == bi.p && bounds == bi.bounds;
    }
    bool operator!=(const Bounds2iIterator &bi) const {
        return p != bi.p || bounds != bi.bounds;
    }

    Point2i operator*() const { return p; }

  private:
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
    friend std::ostream &operator<<(std::ostream &os, const Ray &r) {
        os << "[o=" << r.o << ", d=" << r.d << ", tMax=" << r.tMax
           << ", time=" << r.time << "]";
        return os;
    }

    // Ray Public Data
    Point3f o;
    Vector3f d;
    mutable Float tMax;
    Float time;
    const Medium *medium;
};

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
    friend std::ostream &operator<<(std::ostream &os, const RayDifferential &r) {
        os << "[ " << (Ray &)r << " has differentials: " <<
            (r.hasDifferentials ? "true" : "false") << ", xo = " << r.rxOrigin <<
            ", xd = " << r.rxDirection << ", yo = " << r.ryOrigin << ", yd = " <<
            r.ryDirection;
        return os;
    }

    // RayDifferential Public Data
    bool hasDifferentials;
    Point3f rxOrigin, ryOrigin;
    Vector3f rxDirection, ryDirection;
};

// Geometry Inline Functions
template <typename T>
inline Vector3<T>::Vector3(const Point3<T> &p)
    : Tuple3<Vector3::template Vector3, T>(p.x, p.y, p.z) { }

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

template <typename T>
inline Vector3<T> Normalize(const Vector3<T> &v) {
    return v / v.Length();
}

template <typename T>
inline void CoordinateSystem(const Vector3<T> &v1, Vector3<T> *v2,
                             Vector3<T> *v3) {
    if (std::abs(v1.x) > std::abs(v1.y))
        *v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
    else
        *v2 = Vector3<T>(0, v1.z, -v1.y) / std::sqrt(v1.y * v1.y + v1.z * v1.z);
    *v3 = Cross(v1, *v2);
}

template <typename T> template <typename U>
Vector2<T>::Vector2(const Point2<U> &p)
    : Tuple2<Vector2::template Vector2, T>(p.x, p.y) { }

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

template <typename T>
inline Vector2<T> Normalize(const Vector2<T> &v) {
    return v / v.Length();
}

template <typename T>
inline Float Distance(const Point3<T> &p1, const Point3<T> &p2) {
    return (p1 - p2).Length();
}

template <typename T>
inline T DistanceSquared(const Point3<T> &p1, const Point3<T> &p2) {
    return (p1 - p2).LengthSquared();
}

template <typename T>
inline Float Distance(const Point2<T> &p1, const Point2<T> &p2) {
    return (p1 - p2).Length();
}

template <typename T>
inline T DistanceSquared(const Point2<T> &p1, const Point2<T> &p2) {
    return (p1 - p2).LengthSquared();
}

template <typename T>
inline Normal3<T> Normalize(const Normal3<T> &n) {
    return n / n.Length();
}

template <typename T>
inline Vector3<T>::Vector3(const Normal3<T> &n)
  : Tuple3<Vector3::template Vector3, T>(n.x, n.y, n.z) { }

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

template <typename T>
inline const Point3<T> &Bounds3<T>::operator[](int i) const {
    DCHECK(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T>
inline Point3<T> &Bounds3<T>::operator[](int i) {
    DCHECK(i == 0 || i == 1);
    return (i == 0) ? pMin : pMax;
}

template <typename T>
Bounds3<T> Union(const Bounds3<T> &b, const Point3<T> &p) {
    return Bounds3<T>(
        Point3<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y),
                  std::min(b.pMin.z, p.z)),
        Point3<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y),
                  std::max(b.pMax.z, p.z)));
}

template <typename T>
Bounds3<T> Union(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    return Bounds3<T>(Point3<T>(std::min(b1.pMin.x, b2.pMin.x),
                                std::min(b1.pMin.y, b2.pMin.y),
                                std::min(b1.pMin.z, b2.pMin.z)),
                      Point3<T>(std::max(b1.pMax.x, b2.pMax.x),
                                std::max(b1.pMax.y, b2.pMax.y),
                                std::max(b1.pMax.z, b2.pMax.z)));
}

template <typename T>
Bounds3<T> Intersect(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    return Bounds3<T>(Point3<T>(std::max(b1.pMin.x, b2.pMin.x),
                                std::max(b1.pMin.y, b2.pMin.y),
                                std::max(b1.pMin.z, b2.pMin.z)),
                      Point3<T>(std::min(b1.pMax.x, b2.pMax.x),
                                std::min(b1.pMax.y, b2.pMax.y),
                                std::min(b1.pMax.z, b2.pMax.z)));
}

template <typename T>
bool Overlaps(const Bounds3<T> &b1, const Bounds3<T> &b2) {
    bool x = (b1.pMax.x >= b2.pMin.x) && (b1.pMin.x <= b2.pMax.x);
    bool y = (b1.pMax.y >= b2.pMin.y) && (b1.pMin.y <= b2.pMax.y);
    bool z = (b1.pMax.z >= b2.pMin.z) && (b1.pMin.z <= b2.pMax.z);
    return (x && y && z);
}

template <typename T>
bool Inside(const Point3<T> &p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x <= b.pMax.x && p.y >= b.pMin.y &&
            p.y <= b.pMax.y && p.z >= b.pMin.z && p.z <= b.pMax.z);
}

template <typename T>
bool InsideExclusive(const Point3<T> &p, const Bounds3<T> &b) {
    return (p.x >= b.pMin.x && p.x < b.pMax.x && p.y >= b.pMin.y &&
            p.y < b.pMax.y && p.z >= b.pMin.z && p.z < b.pMax.z);
}

template <typename T, typename U>
inline Bounds3<T> Expand(const Bounds3<T> &b, U delta) {
    return Bounds3<T>(b.pMin - Vector3<T>(delta, delta, delta),
                      b.pMax + Vector3<T>(delta, delta, delta));
}

// Minimum squared distance from point to box; returns zero if point is
// inside.
template <typename T, typename U>
inline Float DistanceSquared(const Point3<T> &p, const Bounds3<U> &b) {
    Float dx = std::max({Float(0), b.pMin.x - p.x, p.x - b.pMax.x});
    Float dy = std::max({Float(0), b.pMin.y - p.y, p.y - b.pMax.y});
    Float dz = std::max({Float(0), b.pMin.z - p.z, p.z - b.pMax.z});
    return dx * dx + dy * dy + dz * dz;
}

template <typename T, typename U>
inline Float Distance(const Point3<T> &p, const Bounds3<U> &b) {
    return std::sqrt(DistanceSquared(p, b));
}

inline Bounds2iIterator begin(const Bounds2i &b) {
    return Bounds2iIterator(b, b.pMin);
}

inline Bounds2iIterator end(const Bounds2i &b) {
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
Bounds2<T> Union(const Bounds2<T> &b, const Point2<T> &p) {
    Bounds2<T> ret(Point2<T>(std::min(b.pMin.x, p.x), std::min(b.pMin.y, p.y)),
                   Point2<T>(std::max(b.pMax.x, p.x), std::max(b.pMax.y, p.y)));
    return ret;
}

template <typename T>
Bounds2<T> Union(const Bounds2<T> &b, const Bounds2<T> &b2) {
    Bounds2<T> ret(
        Point2<T>(std::min(b.pMin.x, b2.pMin.x), std::min(b.pMin.y, b2.pMin.y)),
        Point2<T>(std::max(b.pMax.x, b2.pMax.x),
                  std::max(b.pMax.y, b2.pMax.y)));
    return ret;
}

template <typename T>
Bounds2<T> Intersect(const Bounds2<T> &b, const Bounds2<T> &b2) {
    Bounds2<T> ret(
        Point2<T>(std::max(b.pMin.x, b2.pMin.x), std::max(b.pMin.y, b2.pMin.y)),
        Point2<T>(std::min(b.pMax.x, b2.pMax.x),
                  std::min(b.pMax.y, b2.pMax.y)));
    return ret;
}

template <typename T>
bool Overlaps(const Bounds2<T> &ba, const Bounds2<T> &bb) {
    bool x = (ba.pMax.x >= bb.pMin.x) && (ba.pMin.x <= bb.pMax.x);
    bool y = (ba.pMax.y >= bb.pMin.y) && (ba.pMin.y <= bb.pMax.y);
    return (x && y);
}

template <typename T>
bool Inside(const Point2<T> &pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x <= b.pMax.x && pt.y >= b.pMin.y &&
            pt.y <= b.pMax.y);
}

template <typename T>
bool InsideExclusive(const Point2<T> &pt, const Bounds2<T> &b) {
    return (pt.x >= b.pMin.x && pt.x < b.pMax.x && pt.y >= b.pMin.y &&
            pt.y < b.pMax.y);
}

template <typename T, typename U>
Bounds2<T> Expand(const Bounds2<T> &b, U delta) {
    return Bounds2<T>(b.pMin - Vector2<T>(delta, delta),
                      b.pMax + Vector2<T>(delta, delta));
}

template <typename T>
inline bool Bounds3<T>::IntersectP(const Ray &ray, Float *hitt0,
                                   Float *hitt1) const {
    Float t0 = 0, t1 = ray.tMax;
    for (int i = 0; i < 3; ++i) {
        // Update interval for _i_th bounding box slab
        Float invRayDir = 1 / ray.d[i];
        Float tNear = (pMin[i] - ray.o[i]) * invRayDir;
        Float tFar = (pMax[i] - ray.o[i]) * invRayDir;

        // Update parametric interval from slab intersection $t$ values
        if (tNear > tFar) std::swap(tNear, tFar);

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
inline bool Bounds3<T>::IntersectP(const Ray &ray, const Vector3f &invDir,
                                   const int dirIsNeg[3]) const {
    const Bounds3f &bounds = *this;
    // Check for ray intersection against $x$ and $y$ slabs
    Float tMin = (bounds[dirIsNeg[0]].x - ray.o.x) * invDir.x;
    Float tMax = (bounds[1 - dirIsNeg[0]].x - ray.o.x) * invDir.x;
    Float tyMin = (bounds[dirIsNeg[1]].y - ray.o.y) * invDir.y;
    Float tyMax = (bounds[1 - dirIsNeg[1]].y - ray.o.y) * invDir.y;

    // Update _tMax_ and _tyMax_ to ensure robust bounds intersection
    tMax *= 1 + 2 * gamma(3);
    tyMax *= 1 + 2 * gamma(3);
    if (tMin > tyMax || tyMin > tMax) return false;
    if (tyMin > tMin) tMin = tyMin;
    if (tyMax < tMax) tMax = tyMax;

    // Check for ray intersection against $z$ slab
    Float tzMin = (bounds[dirIsNeg[2]].z - ray.o.z) * invDir.z;
    Float tzMax = (bounds[1 - dirIsNeg[2]].z - ray.o.z) * invDir.z;

    // Update _tzMax_ to ensure robust bounds intersection
    tzMax *= 1 + 2 * gamma(3);
    if (tMin > tzMax || tzMin > tMax) return false;
    if (tzMin > tMin) tMin = tzMin;
    if (tzMax < tMax) tMax = tzMax;
    return (tMin < ray.tMax) && (tMax > 0);
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
    return Vector3f(sinTheta * std::cos(phi), sinTheta * std::sin(phi),
                    cosTheta);
}

inline Vector3f SphericalDirection(Float sinTheta, Float cosTheta, Float phi,
                                   const Vector3f &x, const Vector3f &y,
                                   const Vector3f &z) {
    return sinTheta * std::cos(phi) * x + sinTheta * std::sin(phi) * y +
           cosTheta * z;
}

inline Float SphericalTheta(const Vector3f &v) {
    return std::acos(Clamp(v.z, -1, 1));
}

inline Float SphericalPhi(const Vector3f &v) {
    Float p = std::atan2(v.y, v.x);
    return (p < 0) ? (p + 2 * Pi) : p;
}

}  // namespace pbrt

#endif  // PBRT_CORE_GEOMETRY_H
