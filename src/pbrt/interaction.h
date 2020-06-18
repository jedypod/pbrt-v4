// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_INTERACTION_H
#define PBRT_CORE_INTERACTION_H

// core/interaction.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/bssrdf.h>
#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/medium.h>
#include <pbrt/base/sampler.h>
#include <pbrt/ray.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <limits>

namespace pbrt {

// Interaction Declarations
class Interaction {
  public:
    // Interaction Public Methods
    Interaction() = default;
    // used by surface ctor
    PBRT_CPU_GPU
    Interaction(const Point3fi &pi, const Normal3f &n, const Point2f &uv,
                const Vector3f &wo, Float time, const MediumInterface *mediumInterface)
        : pi(pi),
          n(n),
          uv(uv),
          wo(Normalize(wo)),
          time(time),
          mediumInterface(mediumInterface) {}
    // used by medium ctor
    PBRT_CPU_GPU
    Interaction(const Point3f &p, const Vector3f &wo, Float time, MediumHandle medium)
        : pi(p), time(time), wo(wo), medium(medium) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, const Normal3f &n, Float time, MediumHandle medium)
        : pi(p), n(n), time(time), medium(medium) {}
    PBRT_CPU_GPU
    Interaction(const Point3fi &pi, const Normal3f &n, Float time = 0,
                const Point2f &uv = {})
        : pi(pi), n(n), uv(uv), time(time) {}
    PBRT_CPU_GPU
    Interaction(const Point3fi &pi, const Normal3f &n, const Point2f &uv)
        : pi(pi), n(n), uv(uv) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, Float time, MediumHandle medium)
        : pi(p), time(time), medium(medium) {}
    PBRT_CPU_GPU
    Interaction(const Point3f &p, Float time, const MediumInterface *mediumInterface)
        : pi(p), time(time), mediumInterface(mediumInterface) {}

    PBRT_CPU_GPU
    bool IsSurfaceInteraction() const { return n != Normal3f(0, 0, 0); }
    PBRT_CPU_GPU
    const SurfaceInteraction &AsSurface() const {
        CHECK(IsSurfaceInteraction());
        return (const SurfaceInteraction &)*this;
    }
    PBRT_CPU_GPU
    SurfaceInteraction &AsSurface() {
        CHECK(IsSurfaceInteraction());
        return (SurfaceInteraction &)*this;
    }

    PBRT_CPU_GPU
    Point3f OffsetRayOrigin(const Vector3f &w) const {
        Float d = Dot(Abs(n), pi.Error());
        Vector3f offset = d * Vector3f(n);
        if (Dot(w, n) < 0)
            offset = -offset;
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
    PBRT_CPU_GPU
    Point3f OffsetRayOrigin(const Point3f &pt) const { return OffsetRayOrigin(pt - p()); }
    PBRT_CPU_GPU
    RayDifferential SpawnRay(const Vector3f &d) const {
        return RayDifferential(OffsetRayOrigin(d), d, time, GetMedium(d));
    }
    PBRT_CPU_GPU
    Ray SpawnRayTo(const Point3f &p2) const {
        Vector3f d = p2 - p();
        return Ray(OffsetRayOrigin(d), d, time, GetMedium(d));
    }
    PBRT_CPU_GPU
    Ray SpawnRayTo(const Interaction &it) const {
        Point3f po = OffsetRayOrigin(it.p());
        Point3f pt = it.OffsetRayOrigin(po);
        Vector3f d = pt - po;
        return Ray(po, d, time, GetMedium(d));
    }
    PBRT_CPU_GPU
    bool IsMediumInteraction() const { return !IsSurfaceInteraction(); }
    PBRT_CPU_GPU
    const MediumInteraction &AsMedium() const {
        CHECK(IsMediumInteraction());
        return (const MediumInteraction &)*this;
    }
    PBRT_CPU_GPU
    MediumInteraction &AsMedium() {
        CHECK(IsMediumInteraction());
        return (MediumInteraction &)*this;
    }

    PBRT_CPU_GPU
    MediumHandle GetMedium(const Vector3f &w) const {
        if (mediumInterface != nullptr)
            return Dot(w, n) > 0 ? mediumInterface->outside : mediumInterface->inside;
        return medium;
    }

    PBRT_CPU_GPU
    MediumHandle GetMedium() const {
        if (mediumInterface != nullptr) {
            DCHECK_EQ(mediumInterface->inside, mediumInterface->outside);
            return mediumInterface->inside;
        }
        return medium;
    }

    PBRT_CPU_GPU
    Point3f p() const { return Point3f(pi); }

    std::string ToString() const;

    // Interaction Public Data
    Point3fi pi;
    Normal3f n;
    Point2f uv;
    Vector3f wo;
    Float time = 0;
    MediumHandle medium = nullptr;
    const MediumInterface *mediumInterface = nullptr;
};

class MediumInteraction : public Interaction {
  public:
    // MediumInteraction Public Methods
    PBRT_CPU_GPU
    MediumInteraction() : phase(nullptr) {}
    PBRT_CPU_GPU
    MediumInteraction(const Point3f &p, const Vector3f &wo, Float time,
                      MediumHandle medium, PhaseFunctionHandle phase)
        : Interaction(p, wo, time, medium), phase(phase) {}

    PBRT_CPU_GPU
    bool IsValid() const { return phase != nullptr; }

    std::string ToString() const;

    // MediumInteraction Public Data
    PhaseFunctionHandle phase;
};

// SurfaceInteraction Declarations
class SurfaceInteraction : public Interaction {
  public:
    // SurfaceInteraction Public Methods
    SurfaceInteraction() = default;

    PBRT_CPU_GPU
    SurfaceInteraction(const Point3fi &pi, const Point2f &uv, const Vector3f &wo,
                       const Vector3f &dpdu, const Vector3f &dpdv, const Normal3f &dndu,
                       const Normal3f &dndv, Float time, bool flipNormal,
                       int faceIndex = 0)
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

    PBRT_CPU_GPU
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

    PBRT_CPU_GPU
    void ComputeScatteringFunctions(const RayDifferential &ray,
                                    const SampledWavelengths &lambda, CameraHandle camera,
                                    ScratchBuffer &scratchBuffer, SamplerHandle sampler);
    PBRT_CPU_GPU
    void ComputeDifferentials(const RayDifferential &r, CameraHandle camera) const;
    PBRT_CPU_GPU
    SampledSpectrum Le(const Vector3f &w, const SampledWavelengths &lambda) const;

    using Interaction::SpawnRay;
    PBRT_CPU_GPU
    RayDifferential SpawnRay(const RayDifferential &rayi, const Vector3f &wi,
                             BxDFFlags flags) const;
    PBRT_CPU_GPU
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
    BSSRDFHandle bssrdf;
    MaterialHandle material;
    LightHandle areaLight;

    mutable Vector3f dpdx, dpdy;
    mutable Float dudx = 0, dvdx = 0, dudy = 0, dvdy = 0;

    int faceIndex = 0;
};

}  // namespace pbrt

#endif  // PBRT_CORE_INTERACTION_H
