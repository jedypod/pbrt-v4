
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

#ifndef PBRT_GEOMETRY_TRANSFORM_H
#define PBRT_GEOMETRY_TRANSFORM_H

// geometry/transform.h*
#include <pbrt/pbrt.h>

#include <pbrt/ray.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/float.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdio.h>

namespace pbrt {

// Transform Declarations
class Transform {
  public:
    // Transform Public Methods
    PBRT_HOST_DEVICE
    Transform() {}

    PBRT_HOST_DEVICE
    Transform(const SquareMatrix<4> &m) : m(m) {
        auto inv = Inverse(m);
        if (inv)
            mInv = *inv;
        else
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                    mInv[i][j] = std::numeric_limits<Float>::has_signaling_NaN ?
                        std::numeric_limits<Float>::signaling_NaN() :
                        std::numeric_limits<Float>::quiet_NaN();
    }

    PBRT_HOST_DEVICE
    Transform(const Float mat[4][4])
        : Transform(SquareMatrix<4>(mat)) {}

    PBRT_HOST_DEVICE
    Transform(const SquareMatrix<4> &m, const SquareMatrix<4> &mInv)
        : m(m), mInv(mInv) {}

    PBRT_HOST_DEVICE
    explicit Transform(const Frame &frame)
        : Transform(SquareMatrix<4>(frame.x.x, frame.x.y, frame.x.z, 0.,
                                    frame.y.x, frame.y.y, frame.y.z, 0.,
                                    frame.z.x, frame.z.y, frame.z.z, 0.,
                                    0,   0,   0, 1.)) { }
    PBRT_HOST_DEVICE
    explicit Transform(const Quaternion &q) {
        Float xx = q.v.x * q.v.x, yy = q.v.y * q.v.y, zz = q.v.z * q.v.z;
        Float xy = q.v.x * q.v.y, xz = q.v.x * q.v.z, yz = q.v.y * q.v.z;
        Float wx = q.v.x * q.w, wy = q.v.y * q.w, wz = q.v.z * q.w;

        mInv[0][0] = 1 - 2 * (yy + zz);
        mInv[0][1] = 2 * (xy + wz);
        mInv[0][2] = 2 * (xz - wy);
        mInv[1][0] = 2 * (xy - wz);
        mInv[1][1] = 1 - 2 * (xx + zz);
        mInv[1][2] = 2 * (yz + wx);
        mInv[2][0] = 2 * (xz + wy);
        mInv[2][1] = 2 * (yz - wx);
        mInv[2][2] = 1 - 2 * (xx + yy);

        // Transpose since we are left-handed.  Ugh.
        m = Transpose(mInv);
    }

    PBRT_HOST_DEVICE
    friend Transform Inverse(const Transform &t);

    PBRT_HOST_DEVICE
    friend Transform Transpose(const Transform &t);

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const Transform &t) const {
        return t.m == m;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const Transform &t) const {
        return t.m != m;
    }

    PBRT_HOST_DEVICE_INLINE
    bool IsIdentity() const {
        return m.IsIdentity();
    }

    PBRT_HOST_DEVICE_INLINE
    const SquareMatrix<4> &GetMatrix() const { return m; }
    PBRT_HOST_DEVICE_INLINE
    const SquareMatrix<4> &GetInverseMatrix() const { return mInv; }

    PBRT_HOST_DEVICE
    bool HasScale() const {
        Float la2 = LengthSquared((*this)(Vector3f(1, 0, 0)));
        Float lb2 = LengthSquared((*this)(Vector3f(0, 1, 0)));
        Float lc2 = LengthSquared((*this)(Vector3f(0, 0, 1)));
        return (std::abs(la2 - 1) > 1e-3f || std::abs(lb2 - 1) > 1e-3f ||
                std::abs(lc2 - 1) > 1e-3f);
    }

    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    Point3<T> operator()(const Point3<T> &p) const;
    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    Vector3<T> operator()(const Vector3<T> &v) const;
    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    Normal3<T> operator()(const Normal3<T> &) const;
    PBRT_HOST_DEVICE_INLINE
    Ray operator()(const Ray &r, Float *tMax = nullptr) const;
    PBRT_HOST_DEVICE_INLINE
    RayDifferential operator()(const RayDifferential &r, Float *tMax = nullptr) const;
    PBRT_HOST_DEVICE
    Bounds3f operator()(const Bounds3f &b) const;

    PBRT_HOST_DEVICE
    explicit operator Quaternion() const;

    // These can have 100s of times tighter bounds and (TODO: measure perf)
    // than just using the default operator() implementations that use
    // FloatInterval for everything...
    PBRT_HOST_DEVICE
    Point3fi operator()(const Point3fi &p) const;
    PBRT_HOST_DEVICE
    Vector3fi operator()(const Vector3fi &v) const;
    PBRT_HOST_DEVICE
    Point3fi ApplyInverse(const Point3fi &p) const;

    PBRT_HOST_DEVICE
    Transform operator*(const Transform &t2) const;
    PBRT_HOST_DEVICE
    bool SwapsHandedness() const;

    PBRT_HOST_DEVICE
    Interaction operator()(const Interaction &in) const;
    PBRT_HOST_DEVICE
    Interaction ApplyInverse(const Interaction &in) const;
    PBRT_HOST_DEVICE
    SurfaceInteraction operator()(const SurfaceInteraction &si) const;
    PBRT_HOST_DEVICE
    SurfaceInteraction ApplyInverse(const SurfaceInteraction &in) const;
    PBRT_HOST_DEVICE
    inline Ray ApplyInverse(const Ray &r, Float *tMax = nullptr) const;
    PBRT_HOST_DEVICE
    inline RayDifferential ApplyInverse(const RayDifferential &r, Float *tMax = nullptr) const;

    template <typename T>
    PBRT_HOST_DEVICE
    inline Point3<T> ApplyInverse(const Point3<T> &p) const;
    template <typename T>
    PBRT_HOST_DEVICE
    inline Vector3<T> ApplyInverse(const Vector3<T> &v) const;
    template <typename T>
    PBRT_HOST_DEVICE
    inline Normal3<T> ApplyInverse(const Normal3<T> &) const;

    uint64_t Hash() const { return HashBuffer<sizeof(m)>(&m); }

    void Decompose(Vector3f *T, SquareMatrix<4> *R, SquareMatrix<4> *S) const;

    std::string ToString() const;

  private:
    // Transform Private Data
    SquareMatrix<4> m, mInv;
};

PBRT_HOST_DEVICE
Transform Translate(const Vector3f &delta);
PBRT_HOST_DEVICE
Transform Scale(Float x, Float y, Float z);
PBRT_HOST_DEVICE
Transform RotateX(Float theta);
PBRT_HOST_DEVICE
Transform RotateY(Float theta);
PBRT_HOST_DEVICE
Transform RotateZ(Float theta);
PBRT_HOST_DEVICE
Transform Rotate(Float theta, const Vector3f &axis);
PBRT_HOST_DEVICE
Transform Rotate(Float sinTheta, Float cosTheta, const Vector3f &axis);
PBRT_HOST_DEVICE
Transform LookAt(const Point3f &pos, const Point3f &look, const Vector3f &up);
PBRT_HOST_DEVICE
Transform Orthographic(Float znear, Float zfar);
PBRT_HOST_DEVICE
Transform Perspective(Float fov, Float znear, Float zfar);

// Transform Inline Functions
PBRT_HOST_DEVICE
inline Transform Inverse(const Transform &t) {
    return Transform(t.mInv, t.m);
}

PBRT_HOST_DEVICE
inline Transform Transpose(const Transform &t) {
    return Transform(Transpose(t.m), Transpose(t.mInv));
}

template <typename T>
inline Point3<T> Transform::operator()(const Point3<T> &p) const {
    T x = p.x, y = p.y, z = p.z;
    T xp = m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3];
    T yp = m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3];
    T zp = m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3];
    T wp = m[3][0] * x + m[3][1] * y + m[3][2] * z + m[3][3];
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::operator()(const Vector3<T> &v) const {
    T x = v.x, y = v.y, z = v.z;
    return Vector3<T>(m[0][0] * x + m[0][1] * y + m[0][2] * z,
                      m[1][0] * x + m[1][1] * y + m[1][2] * z,
                      m[2][0] * x + m[2][1] * y + m[2][2] * z);
}

template <typename T>
inline Normal3<T> Transform::operator()(const Normal3<T> &n) const {
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(mInv[0][0] * x + mInv[1][0] * y + mInv[2][0] * z,
                      mInv[0][1] * x + mInv[1][1] * y + mInv[2][1] * z,
                      mInv[0][2] * x + mInv[1][2] * y + mInv[2][2] * z);
}

inline Ray Transform::operator()(const Ray &r, Float *tMax) const {
    Vector3f oError;
    Point3fi o = (*this)(Point3fi(r.o));
    Vector3f d = (*this)(r.d);
    // Offset ray origin to edge of error bounds and compute _tMax_
    Float lengthSquared = LengthSquared(d);
    if (lengthSquared > 0) {
        Float dt = Dot(Abs(d), oError) / lengthSquared;
        o += d * dt;
        if (tMax) *tMax -= dt;
    }
    return Ray(Point3f(o), d, r.time, r.medium);
}

inline RayDifferential Transform::operator()(const RayDifferential &r, Float *tMax) const {
    Ray tr = (*this)(Ray(r), tMax);
    RayDifferential ret(tr.o, tr.d, tr.time, tr.medium);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = (*this)(r.rxOrigin);
    ret.ryOrigin = (*this)(r.ryOrigin);
    ret.rxDirection = (*this)(r.rxDirection);
    ret.ryDirection = (*this)(r.ryDirection);
    return ret;
}

template <typename T>
inline Point3<T> Transform::ApplyInverse(const Point3<T> &p) const {
    T x = p.x, y = p.y, z = p.z;
    T xp = (mInv[0][0] * x + mInv[0][1] * y) + (mInv[0][2] * z + mInv[0][3]);
    T yp = (mInv[1][0] * x + mInv[1][1] * y) + (mInv[1][2] * z + mInv[1][3]);
    T zp = (mInv[2][0] * x + mInv[2][1] * y) + (mInv[2][2] * z + mInv[2][3]);
    T wp = (mInv[3][0] * x + mInv[3][1] * y) + (mInv[3][2] * z + mInv[3][3]);
    CHECK_NE(wp, 0);
    if (wp == 1)
        return Point3<T>(xp, yp, zp);
    else
        return Point3<T>(xp, yp, zp) / wp;
}

template <typename T>
inline Vector3<T> Transform::ApplyInverse(const Vector3<T> &v) const {
    T x = v.x, y = v.y, z = v.z;
    return Vector3<T>(mInv[0][0] * x + mInv[0][1] * y + mInv[0][2] * z,
                      mInv[1][0] * x + mInv[1][1] * y + mInv[1][2] * z,
                      mInv[2][0] * x + mInv[2][1] * y + mInv[2][2] * z);
}

template <typename T>
inline Normal3<T> Transform::ApplyInverse(const Normal3<T> &n) const {
    T x = n.x, y = n.y, z = n.z;
    return Normal3<T>(m[0][0] * x + m[1][0] * y + m[2][0] * z,
                      m[0][1] * x + m[1][1] * y + m[2][1] * z,
                      m[0][2] * x + m[1][2] * y + m[2][2] * z);
}

inline Ray Transform::ApplyInverse(const Ray &r, Float *tMax) const {
    Point3fi o = ApplyInverse(Point3fi(r.o));
    Vector3f d = ApplyInverse(r.d);
    // Offset ray origin to edge of error bounds and compute _tMax_
    Float lengthSquared = LengthSquared(d);
    if (lengthSquared > 0) {
        Vector3f oError(o.x.Width() / 2, o.y.Width() / 2, o.z.Width() / 2);
        Float dt = Dot(Abs(d), oError) / lengthSquared;
        o += d * dt;
        if (tMax) *tMax -= dt;
    }
    return Ray(Point3f(o), d, r.time, r.medium);
}

inline RayDifferential Transform::ApplyInverse(const RayDifferential &r, Float *tMax) const {
    Ray tr = ApplyInverse(Ray(r), tMax);
    RayDifferential ret(tr.o, tr.d, tr.time, tr.medium);
    ret.hasDifferentials = r.hasDifferentials;
    ret.rxOrigin = ApplyInverse(r.rxOrigin);
    ret.ryOrigin = ApplyInverse(r.ryOrigin);
    ret.rxDirection = ApplyInverse(r.rxDirection);
    ret.ryDirection = ApplyInverse(r.ryDirection);
    return ret;
}

// AnimatedTransform Declarations
class AnimatedTransform {
  public:
    AnimatedTransform() = default;
    // AnimatedTransform Public Methods
    explicit AnimatedTransform(const Transform *t)
        : AnimatedTransform(t, 0, t, 1) {}
    AnimatedTransform(const Transform *startTransform,
                      Float startTime,
                      const Transform *endTransform,
                      Float endTime);

    PBRT_HOST_DEVICE
    Transform Interpolate(Float time) const;

    PBRT_HOST_DEVICE
    Ray operator()(const Ray &r, Float *tMax = nullptr) const;
    PBRT_HOST_DEVICE
    Ray ApplyInverse(const Ray &r, Float *tMax = nullptr) const;
    PBRT_HOST_DEVICE
    RayDifferential operator()(const RayDifferential &r, Float *tMax = nullptr) const;

    PBRT_HOST_DEVICE
    Point3f operator()(const Point3f &p, Float time) const;
    PBRT_HOST_DEVICE
    Point3f ApplyInverse(const Point3f &p, Float time) const {
        if (!actuallyAnimated)
            return startTransform->ApplyInverse(p);
        return Interpolate(time).ApplyInverse(p);
    }
    PBRT_HOST_DEVICE
    Vector3f operator()(const Vector3f &v, Float time) const;
    PBRT_HOST_DEVICE
    Vector3f ApplyInverse(const Vector3f &v, Float time) const {
        if (!actuallyAnimated)
            return startTransform->ApplyInverse(v);
        return Interpolate(time).ApplyInverse(v);
    }
    PBRT_HOST_DEVICE
    Normal3f operator()(const Normal3f &n, Float time) const;
    PBRT_HOST_DEVICE
    Normal3f ApplyInverse(const Normal3f &n, Float time) const {
        if (!actuallyAnimated)
            return startTransform->ApplyInverse(n);
        return Interpolate(time).ApplyInverse(n);
    }
    PBRT_HOST_DEVICE
    Interaction operator()(const Interaction &it) const;
    PBRT_HOST_DEVICE
    Interaction ApplyInverse(const Interaction &it) const;
    PBRT_HOST_DEVICE
    SurfaceInteraction operator()(const SurfaceInteraction &it) const;
    PBRT_HOST_DEVICE
    SurfaceInteraction ApplyInverse(const SurfaceInteraction &it) const;
    PBRT_HOST_DEVICE
    bool HasScale() const {
        return startTransform->HasScale() || endTransform->HasScale();
    }
    PBRT_HOST_DEVICE
    Bounds3f MotionBounds(const Bounds3f &b) const;
    PBRT_HOST_DEVICE
    Bounds3f BoundPointMotion(const Point3f &p) const;
    PBRT_HOST_DEVICE
    bool IsAnimated() const { return actuallyAnimated; }

    const Transform *startTransform = nullptr, *endTransform = nullptr;
    Float startTime, endTime;

    std::string ToString() const;

  private:
    // AnimatedTransform Private Data
    bool actuallyAnimated;
    Vector3f T[2];
    Quaternion R[2];
    SquareMatrix<4> S[2];
    bool hasRotation;
    struct DerivativeTerm {
        PBRT_HOST_DEVICE
        DerivativeTerm() {}
        PBRT_HOST_DEVICE
        DerivativeTerm(Float c, Float x, Float y, Float z)
            : kc(c), kx(x), ky(y), kz(z) {}
        Float kc, kx, ky, kz;
        PBRT_HOST_DEVICE
        Float Eval(const Point3f &p) const {
            return kc + kx * p.x + ky * p.y + kz * p.z;
        }
    };
    DerivativeTerm c1[3], c2[3], c3[3], c4[3], c5[3];

    PBRT_HOST_DEVICE
    static void FindZeros(Float c1, Float c2, Float c3, Float c4, Float c5,
                          Float theta, FloatInterval tInterval, pstd::span<Float> zeros,
                          int *zeroCount, int depth = 8);
};

}  // namespace pbrt

#endif  // PBRT_GEOMETRY_TRANSFORM_H
