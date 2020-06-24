
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

#ifndef PBRT_CORE_INTERACTION_H
#define PBRT_CORE_INTERACTION_H

// core/interaction.h*
#include <pbrt/pbrt.h>

#include <pbrt/ray.h>
#include <pbrt/util/vecmath.h>

#include <limits>

namespace pbrt {

// Interaction Declarations
class Interaction {
 public:
    // Interaction Public Methods
    Interaction() = default;
    // used by surface ctor
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3fi &pi, const Normal3f &n,
                const Point2f &uv, const Vector3f &wo, Float time,
                const MediumInterface *mediumInterface)
        : pi(pi),
          n(n),
          uv(uv),
          wo(Normalize(wo)),
          time(time),
          mediumInterface(mediumInterface) {}
    // used by medium ctor
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3f &p, const Vector3f &wo, Float time,
                const Medium *medium)
        : pi(p), time(time), wo(wo), medium(medium) {}
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3f &p, const Normal3f &n, Float time,
                const Medium *medium)
        : pi(p), n(n), time(time), medium(medium) {}
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3fi &pi, const Normal3f &n, Float time = 0,
                const Point2f &uv = {})
        : pi(pi), n(n), uv(uv), time(time) {}
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3fi &pi, const Normal3f &n, const Point2f &uv)
        : pi(pi), n(n), uv(uv) {}
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3f &p, Float time, const Medium *medium)
        : pi(p), time(time), medium(medium) {}
    PBRT_HOST_DEVICE_INLINE
    Interaction(const Point3f &p, Float time,
                const MediumInterface *mediumInterface)
        : pi(p), time(time), mediumInterface(mediumInterface) {}

    PBRT_HOST_DEVICE_INLINE
    bool IsSurfaceInteraction() const { return n != Normal3f(0, 0, 0); }
    PBRT_HOST_DEVICE_INLINE
    const SurfaceInteraction &AsSurface() const {
        CHECK(IsSurfaceInteraction());
        return (const SurfaceInteraction &)*this;
    }
    PBRT_HOST_DEVICE_INLINE
    SurfaceInteraction &AsSurface() {
        CHECK(IsSurfaceInteraction());
        return (SurfaceInteraction &)*this;
    }

    PBRT_HOST_DEVICE_INLINE
    Point3f OffsetRayOrigin(const Vector3f &w) const {
        Float d = Dot(Abs(n), pi.Error());
        Vector3f offset = d * Vector3f(n);
        if (Dot(w, n) < 0) offset = -offset;
        Point3f po = Point3f(pi) + offset;
        // Round offset point _po_ away from _p_
        for (int i = 0; i < 3; ++i) {
            if (offset[i] > 0)
                po[i] = NextFloatUp(po[i]);
            else if (offset[i] < 0)
                po[i] = NextFloatDown(po[i]);
        }
        return po;
    }
    PBRT_HOST_DEVICE_INLINE
    Point3f OffsetRayOrigin(const Point3f &pt) const {
        return OffsetRayOrigin(pt - p());
    }
    PBRT_HOST_DEVICE_INLINE
    RayDifferential SpawnRay(const Vector3f &d) const {
        return RayDifferential(OffsetRayOrigin(d), d, time, GetMedium(d));
    }
    PBRT_HOST_DEVICE_INLINE
    Ray SpawnRayTo(const Point3f &p2) const {
        Vector3f d = p2 - p();
        return Ray(OffsetRayOrigin(d), d, time, GetMedium(d));
    }
    PBRT_HOST_DEVICE_INLINE
    Ray SpawnRayTo(const Interaction &it) const {
        Point3f po = OffsetRayOrigin(it.p());
        Point3f pt = it.OffsetRayOrigin(po);
        Vector3f d = pt - po;
        return Ray(po, d, time, GetMedium(d));
    }
    PBRT_HOST_DEVICE_INLINE
    bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }
    PBRT_HOST_DEVICE_INLINE
    const MediumInteraction &AsMedium() const {
        CHECK(IsMediumInteraction());
        return (const MediumInteraction &)*this;
    }
    PBRT_HOST_DEVICE_INLINE
    MediumInteraction &AsMedium() {
        CHECK(IsMediumInteraction());
        return (MediumInteraction &)*this;
    }

    PBRT_HOST_DEVICE
    const Medium *GetMedium(const Vector3f &w) const;
    PBRT_HOST_DEVICE
    const Medium *GetMedium() const;
    PBRT_HOST_DEVICE_INLINE
    Point3f p() const { return Point3f(pi); }

    std::string ToString() const;

    // Interaction Public Data
    Point3fi pi;
    Normal3f n;
    Point2f uv;
    Vector3f wo;
    Float time = 0;
    const Medium *medium = nullptr;
    const MediumInterface *mediumInterface = nullptr;
};

class MediumInteraction : public Interaction {
  public:
    // MediumInteraction Public Methods
    PBRT_HOST_DEVICE
    MediumInteraction() : phase(nullptr) {}
    PBRT_HOST_DEVICE
    MediumInteraction(const Point3f &p, const Vector3f &wo, Float time,
                      const Medium *medium, const PhaseFunction *phase)
        : Interaction(p, wo, time, medium), phase(phase) {}

    PBRT_HOST_DEVICE
    bool IsValid() const { return phase != nullptr; }

    std::string ToString() const;

    // MediumInteraction Public Data
    const PhaseFunction *phase;
};

// SurfaceInteraction Declarations
class SurfaceInteraction : public Interaction {
  public:
    // SurfaceInteraction Public Methods
    SurfaceInteraction() = default;
    PBRT_HOST_DEVICE
    SurfaceInteraction(const Point3fi &pi, const Point2f &uv,
                       const Vector3f &wo, const Vector3f &dpdu, const Vector3f &dpdv,
                       const Normal3f &dndu, const Normal3f &dndv, Float time,
                       bool flipNormal, int faceIndex = 0)
        : Interaction(pi, Normal3f(Normalize(Cross(dpdu, dpdv))), uv, wo, time,
                      (MediumInterface *)nullptr),
          dpdu(dpdu),
          dpdv(dpdv),
          dndu(dndu),
          dndv(dndv),
          faceIndex(faceIndex) {
        // Initialize shading geometry from true geometry
        shading.n = n;
        shading.dpdu = dpdu;
        shading.dpdv = dpdv;
        shading.dndu = dndu;
        shading.dndv = dndv;

        // Adjust normal based on orientation and handedness
        if (flipNormal) {
            n *= -1;
            shading.n *= -1;
        }
    }

    PBRT_HOST_DEVICE
    void SetShadingGeometry(const Normal3f &ns, const Vector3f &dpdus,
                            const Vector3f &dpdvs, const Normal3f &dndus,
                            const Normal3f &dndvs, bool orientationIsAuthoritative) {
        // Compute _shading.n_ for _SurfaceInteraction_
        shading.n = ns;
        DCHECK_NE(shading.n, Normal3f(0, 0, 0));
        if (orientationIsAuthoritative)
            n = FaceForward(n, shading.n);
        else
            shading.n = FaceForward(shading.n, n);

        // Initialize _shading_ partial derivative values
        shading.dpdu = dpdus;
        shading.dpdv = dpdvs;
        shading.dndu = dndus;
        shading.dndv = dndvs;

        while (LengthSquared(shading.dpdu) > 1e16 || LengthSquared(shading.dpdv) > 1e16) {
            shading.dpdu /= 1e8;
            shading.dpdv /= 1e8;
        }
    }

    PBRT_HOST_DEVICE
    void ComputeScatteringFunctions(
        const RayDifferential &ray, const SampledWavelengths &lambda,
        const Camera &camera, MaterialBuffer &materialBuffer, Sampler &sampler,
        TransportMode mode = TransportMode::Radiance);
    PBRT_HOST_DEVICE
    void ComputeDifferentials(const RayDifferential &r,
                              const Camera &camera) const;
    PBRT_HOST_DEVICE
    SampledSpectrum Le(const Vector3f &w, const SampledWavelengths &lambda) const;

    using Interaction::SpawnRay;
    PBRT_HOST_DEVICE
    RayDifferential SpawnRay(const RayDifferential &rayi, const Vector3f &wi,
                             BxDFFlags flags) const;
    PBRT_HOST_DEVICE
    void SkipIntersection(RayDifferential *ray, Float t) const;

    std::string ToString() const;

    // SurfaceInteraction Public Data
    Vector3f dpdu, dpdv;
    Normal3f dndu, dndv;
    struct {
        Normal3f n;
        Vector3f dpdu, dpdv;
        Normal3f dndu, dndv;
    } shading;
    BSDF *bsdf = nullptr;
    BSSRDF *bssrdf = nullptr;
    const MaterialHandle *material = nullptr;
    const LightHandle *areaLight = nullptr;

    mutable Vector3f dpdx, dpdy;
    mutable Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;

    int faceIndex = 0;
};

}  // namespace pbrt

#endif  // PBRT_CORE_INTERACTION_H
