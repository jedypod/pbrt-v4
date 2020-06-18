// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_RAY_H
#define PBRT_RAY_H

#include <pbrt/pbrt.h>

#include <pbrt/base/medium.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {
// Ray Declarations
class Ray {
  public:
    // Ray Public Methods
    Ray() = default;

    PBRT_CPU_GPU
    Ray(const Point3f &o, const Vector3f &d, Float time = 0.f,
        MediumHandle medium = nullptr)
        : o(o), d(d), time(time), medium(medium) {}

    PBRT_CPU_GPU
    Point3f operator()(Float t) const { return o + d * t; }

    PBRT_CPU_GPU
    bool HasNaN() const { return (o.HasNaN() || d.HasNaN()); }

    std::string ToString() const;

    // Ray Public Data
    Point3f o;
    Vector3f d;
    Float time = 0;
    MediumHandle medium = nullptr;
};

class RayDifferential : public Ray {
  public:
    // RayDifferential Public Methods
    RayDifferential() = default;
    PBRT_CPU_GPU
    RayDifferential(const Point3f &o, const Vector3f &d, Float time = 0.f,
                    MediumHandle medium = nullptr)
        : Ray(o, d, time, medium) {
        hasDifferentials = false;
    }

    PBRT_CPU_GPU
    explicit RayDifferential(const Ray &ray) : Ray(ray) { hasDifferentials = false; }

    PBRT_CPU_GPU
    bool HasNaN() const {
        return Ray::HasNaN() ||
               (hasDifferentials && (rxOrigin.HasNaN() || ryOrigin.HasNaN() ||
                                     rxDirection.HasNaN() || ryDirection.HasNaN()));
    }

    PBRT_CPU_GPU
    void ScaleDifferentials(Float s) {
        rxOrigin = o + (rxOrigin - o) * s;
        ryOrigin = o + (ryOrigin - o) * s;
        rxDirection = d + (rxDirection - d) * s;
        ryDirection = d + (ryDirection - d) * s;
    }

    std::string ToString() const;

    // RayDifferential Public Data
    bool hasDifferentials = false;
    Point3f rxOrigin, ryOrigin;
    Vector3f rxDirection, ryDirection;
};

}  // namespace pbrt

#endif
