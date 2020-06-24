
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

#ifndef PBRT_SHAPES_SPHERE_H
#define PBRT_SHAPES_SPHERE_H

// shapes.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/interaction.h>
#include <pbrt/transform.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/vecmath.h>

#include <map>
#include <memory>
#include <vector>

namespace pbrt {

// Sphere Declarations
struct QuadricIntersection {
    Float tHit;
    Point3f pObj;
    Float phi;
};

class Sphere {
  public:
    // Sphere Public Methods
    Sphere(const Transform *worldFromObject, const Transform *objectFromWorld,
           bool reverseOrientation, Float radius, Float zMin, Float zMax,
           Float phiMax)
        : worldFromObject(worldFromObject), objectFromWorld(objectFromWorld),
          reverseOrientation(reverseOrientation),
          transformSwapsHandedness(worldFromObject->SwapsHandedness()),
          radius(radius),
          zMin(Clamp(std::min(zMin, zMax), -radius, radius)),
          zMax(Clamp(std::max(zMin, zMax), -radius, radius)),
          thetaMin(std::acos(Clamp(std::min(zMin, zMax) / radius, -1, 1))),
          thetaMax(std::acos(Clamp(std::max(zMin, zMax) / radius, -1, 1))),
          phiMax(Radians(Clamp(phiMax, 0, 360))) {}

    static Sphere *Create(const Transform *worldFromObject,
                          const Transform *objectFromWorld,
                          bool reverseOrientation,
                          const ParameterDictionary &dict,
                          Allocator alloc);

    PBRT_HOST_DEVICE
    pstd::optional<QuadricIntersection> BasicIntersect(const Ray &r, Float tMax) const {
        ProfilerScope p(ProfilePhase::ShapeIntersect);
        Float phi;
        Point3f pHit;
        // Transform _Ray_ to object space
        Point3fi oi = (*objectFromWorld)(Point3fi(r.o));
        Vector3fi di = (*objectFromWorld)(Vector3fi(r.d));
        Ray ray(Point3f(oi), Vector3f(di), r.time, r.medium);

        // Compute quadratic sphere coefficients
        FloatInterval t0, t1;
        if (!SphereQuadratic(oi, di, &t0, &t1))
            return {};

        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if (t0.UpperBound() > tMax || t1.LowerBound() <= 0) return {};
        FloatInterval tShapeHit = t0;
        if (tShapeHit.LowerBound() <= 0) {
            tShapeHit = t1;
            if (tShapeHit.UpperBound() > tMax) return {};
        }

        // Compute sphere hit position and $\phi$
        pHit = ray((Float)tShapeHit);

        // Refine sphere intersection point
        pHit *= radius / Distance(pHit, Point3f(0, 0, 0));
        if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
        phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) phi += 2 * Pi;

        // Test sphere intersection against clipping parameters
        if ((zMin > -radius && pHit.z < zMin) || (zMax < radius && pHit.z > zMax) ||
            phi > phiMax) {
            if (tShapeHit == t1) return {};
            if (t1.UpperBound() > tMax) return {};
            tShapeHit = t1;
            // Compute sphere hit position and $\phi$
            pHit = ray((Float)tShapeHit);

            // Refine sphere intersection point
            pHit *= radius / Distance(pHit, Point3f(0, 0, 0));
            if (pHit.x == 0 && pHit.y == 0) pHit.x = 1e-5f * radius;
            phi = std::atan2(pHit.y, pHit.x);
            if (phi < 0) phi += 2 * Pi;
            if ((zMin > -radius && pHit.z < zMin) ||
                (zMax < radius && pHit.z > zMax) || phi > phiMax)
                return {};
        }

        return QuadricIntersection{Float(tShapeHit), pHit, phi};
    }

    PBRT_HOST_DEVICE
    SurfaceInteraction InteractionFromIntersection(const QuadricIntersection &isect,
                                                   const Vector3f &wo, Float time) const {
        Point3f pHit = isect.pObj;
        Float phi = isect.phi;

        // Find parametric representation of sphere hit
        Float u = phi / phiMax;
        Float cosTheta = pHit.z / radius;
        Float theta = SafeACos(cosTheta);
        Float v = (theta - thetaMin) / (thetaMax - thetaMin);

        // Compute sphere $\dpdu$ and $\dpdv$
        Float zRadius = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
        Float invZRadius = 1 / zRadius;
        Float cosPhi = pHit.x * invZRadius;
        Float sinPhi = pHit.y * invZRadius;
        Vector3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
        Vector3f dpdv =
            (thetaMax - thetaMin) *
            Vector3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * sinTheta);

        // Compute sphere $\dndu$ and $\dndv$
        Vector3f d2Pduu = -phiMax * phiMax * Vector3f(pHit.x, pHit.y, 0);
        Vector3f d2Pduv =
            (thetaMax - thetaMin) * pHit.z * phiMax * Vector3f(-sinPhi, cosPhi, 0.);
        Vector3f d2Pdvv = -(thetaMax - thetaMin) * (thetaMax - thetaMin) *
            Vector3f(pHit.x, pHit.y, pHit.z);

        // Compute coefficients for fundamental forms
        Float E = Dot(dpdu, dpdu);
        Float F = Dot(dpdu, dpdv);
        Float G = Dot(dpdv, dpdv);
        Vector3f N = Normalize(Cross(dpdu, dpdv));
        Float e = Dot(N, d2Pduu);
        Float f = Dot(N, d2Pduv);
        Float g = Dot(N, d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Float invEGF2 = 1 / (E * G - F * F);
        Normal3f dndu = Normal3f((f * F - e * G) * invEGF2 * dpdu +
                                 (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv = Normal3f((g * F - f * G) * invEGF2 * dpdu +
                                 (f * F - g * E) * invEGF2 * dpdv);

        // Compute error bounds for sphere intersection
        Vector3f pError = gamma(5) * Abs((Vector3f)pHit);

        // Initialize _SurfaceInteraction_ from parametric information
        return (*worldFromObject)(SurfaceInteraction(Point3fi(pHit, pError), Point2f(u, v),
                                                     (*objectFromWorld)(wo), dpdu, dpdv, dndu, dndv,
                                                     time, OrientationIsReversed() ^ TransformSwapsHandedness()));
    }


    PBRT_HOST_DEVICE
    Bounds3f WorldBound() const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const {
        pstd::optional<QuadricIntersection> isect = BasicIntersect(ray, tMax);
        if (!isect)
            return {};

        SurfaceInteraction intr = InteractionFromIntersection(*isect, -ray.d, ray.time);
        return {{intr, isect->tHit}};
    }

    PBRT_HOST_DEVICE
    bool IntersectP(const Ray &r, Float tMax = Infinity) const {
        ProfilerScope p(ProfilePhase::ShapeIntersectP);
        return BasicIntersect(r, tMax).has_value();
    }

    PBRT_HOST_DEVICE
    Float Area() const { return phiMax * radius * (zMax - zMin); }

    PBRT_HOST_DEVICE
    Float SolidAngle(const Point3f &p, int nSamples) const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Interaction &ref, const Point2f &u) const {
        Point3f pCenter = (*worldFromObject)(Point3f(0, 0, 0));

        // Sample uniformly on sphere if $\pt{}$ is inside it
        Point3f pOrigin = ref.OffsetRayOrigin(pCenter);
        if (DistanceSquared(pOrigin, pCenter) <= radius * radius) {
            pstd::optional<ShapeSample> ss = Sample(u);
            if (!ss) return {};
            Vector3f wi = ss->intr.p() - ref.p();
            if (LengthSquared(wi) == 0)
                return {};
            else {
                // Convert from area measure returned by Sample() call above to
                // solid angle measure.
                wi = Normalize(wi);
                ss->pdf *= DistanceSquared(ref.p(), ss->intr.p()) /
                    AbsDot(ss->intr.n, -wi);
            }
            if (std::isinf(ss->pdf)) return {};
            return ss;
        }

        // Compute coordinate system for sphere sampling
        Frame samplingFrame = Frame::FromZ(Normalize(ref.p() - pCenter));

        // Sample sphere uniformly inside subtended cone

        // Compute $\theta$ and $\phi$ values for sample in cone
        Float dc = Distance(ref.p(), pCenter);
        Float invDc = 1 / dc;
        Float sinThetaMax = radius * invDc;
        Float sinThetaMax2 = sinThetaMax * sinThetaMax;
        Float invSinThetaMax = 1 / sinThetaMax;
        Float cosThetaMax = SafeSqrt(1 - sinThetaMax2);
        Float oneMinusCosThetaMax = 1 - cosThetaMax;
        Float cosTheta  = (cosThetaMax - 1) * u[0] + 1;
        Float sinTheta2 = 1 - cosTheta * cosTheta;

        if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */) {
            /* Fall back to a Taylor series expansion for small angles, where
               the standard approach suffers from severe cancellation errors */
            sinTheta2 = sinThetaMax2 * u[0];
            cosTheta = std::sqrt(1 - sinTheta2);

            // Taylor expansion of 1 - sqrt(1 - Sqr(sinThetaMax)) at 0...
            oneMinusCosThetaMax = sinThetaMax2 / 2;
        }

        // Compute angle $\alpha$ from center of sphere to sampled point on surface
        Float cosAlpha = sinTheta2 * invSinThetaMax +
            cosTheta * SafeSqrt(1 - sinTheta2 * invSinThetaMax * invSinThetaMax);
        Float sinAlpha = SafeSqrt(1 - cosAlpha * cosAlpha);
        Float phi = u[1] * 2 * Pi;

        // Compute surface normal and sampled point on sphere
        Vector3f nWorld = samplingFrame.FromLocal(SphericalDirection(sinAlpha, cosAlpha, phi));
        Point3f pWorld = pCenter + radius * Point3f(nWorld.x, nWorld.y, nWorld.z);
        Vector3f pError = gamma(5) * Abs((Vector3f)pWorld);
        Point3fi pi(pWorld, pError);
        Normal3f n(nWorld);
        if (reverseOrientation) n *= -1;

        // Uniform cone PDF.
        DCHECK_NE(oneMinusCosThetaMax, 0); // very small far away sphere
        return ShapeSample{Interaction(pi, n, ref.time), 1 / (2 * Pi * oneMinusCosThetaMax)};
    }

    PBRT_HOST_DEVICE
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_HOST_DEVICE
    Float PDF(const Interaction &ref, const Vector3f &wi) const {
        Point3f pCenter = (*worldFromObject)(Point3f(0, 0, 0));
        // Return uniform PDF if point is inside sphere
        Point3f pOrigin = ref.OffsetRayOrigin(pCenter);
        if (DistanceSquared(pOrigin, pCenter) <= radius * radius)
            return PDF(ref, wi);

        // Compute general sphere PDF
        Float sinThetaMax2 = radius * radius / DistanceSquared(ref.p(), pCenter);
        Float cosThetaMax = SafeSqrt(1 - sinThetaMax2);
        Float oneMinusCosThetaMax = 1 - cosThetaMax;

        if (sinThetaMax2 < 0.00068523f /* sin^2(1.5 deg) */)
            oneMinusCosThetaMax = sinThetaMax2 / 2;

        return 1 / (2 * Pi * oneMinusCosThetaMax);
    }

    PBRT_HOST_DEVICE
    bool OrientationIsReversed() const { return reverseOrientation; }
    PBRT_HOST_DEVICE
    bool TransformSwapsHandedness() const { return transformSwapsHandedness; }

    PBRT_HOST_DEVICE
    DirectionCone NormalBounds() const {
        return DirectionCone::EntireSphere();
    }

    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE
    bool SphereQuadratic(const Point3fi &o, const Vector3fi &d,
                         FloatInterval *t0, FloatInterval *t1) const {
        /* Recap of the approach from Ray Tracing Gems:

           The basic idea is to rewrite b^2 - 4ac to 4a (b^2/4a - c),
           then simplify that to 4a * (r^2 - (Dot(o, o) - Dot(o, d)^2/LengthSquared(d)) =
           4a (r^2 - (Dot(o, o) - Dot(o, d^))) where d^ is Normalize(d).
           Now, consider the decomposition of o into the sum of two vectors,
           d_perp and d_parl, where d_parl is parallel to d^.
           We have d_parl = Dot(o, d^) d^, and d_perp = o - d_parl = o - Dot(o, d^) d^.
           We have a right triangle formed by o, d_perp, and d_parl, and so
           |o|^2 = |d_perp|^2 + |d_parl|^2.
           Note that |d_parl|^2 = Dot(o, d^)^2. Subtrace |d_parl|^2 from both sides and
           we have Dot(o, o) - Dot(o, d^)^2 = |o - Dot(o, d^) d^|^2.

           With the conventional approach, when the ray is long, we end up with
           b^2 \approx 4ac and get hit with catastrophic cancellation.  It's
           extra bad since the magnitudes of the two terms are related to the
           *squared* distance to the ray origin. Even for rays that massively
           miss, we'll end up with a discriminant exactly equal to zero, and
           thence a reported intersection.

           The new approach eliminates c from the computation of the
           discriminant: that's a big win, since it's magnitude is proportional
           to the squared distance to the origin, with accordingly limited
           precision ("accuracy"?)

           Note: the error in the old one is best visualized by going to the
           checkout *before* 1d6e7bd9f6e10991d0c75a2ec74026a2a453522c
           (otherwise everything disappears, since there's too much error in
           the discriminant.)
        */
        // Initialize _FloatInterval_ ray coordinate values
        FloatInterval a = SumSquares(d.x, d.y, d.z);
        FloatInterval b = 2 * (d.x * o.x + d.y * o.y + d.z * o.z);
        FloatInterval c = SumSquares(o.x, o.y, o.z) - Sqr(FloatInterval(radius));

        // Solve quadratic equation for _t_ values
#if 0
        // Original
        FloatInterval b2 = Sqr(b), ac = 4 * a * c;
        FloatInterval odiscrim = b2 - ac; // b * b - FloatInterval(4) * a * c;
#endif
        // RT Gems
        FloatInterval f = b / (2 * a);  // (o . d) / LengthSquared(d)
        Point3fi fp = o - f * d;
        // There's a bit more precision if you compute x^2-y^2 as (x+y)(x-y).
        FloatInterval sqrtf = Sqrt(SumSquares(fp.x, fp.y, fp.z));
        FloatInterval discrim = 4 * a * ((FloatInterval(radius)) - sqrtf) *
            ((FloatInterval(radius)) + sqrtf);

        if (discrim.LowerBound() < 0) return {};
        FloatInterval rootDiscrim = Sqrt(discrim);

        // Compute quadratic _t_ values
        FloatInterval q;
        if ((Float)b < 0)
            q = -.5 * (b - rootDiscrim);
        else
            q = -.5 * (b + rootDiscrim);
        *t0 = q / a;
        *t1 = c / q;
        if (t0->LowerBound() > t1->LowerBound()) pstd::swap(*t0, *t1);
        return true;
    }

    // Sphere Private Data
    Float radius;
    Float zMin, zMax;
    Float thetaMin, thetaMax, phiMax;

    const Transform *worldFromObject, *objectFromWorld;
    bool reverseOrientation;
    bool transformSwapsHandedness;
};


// Disk Declarations
class Disk {
  public:
    // Disk Public Methods
    Disk(const Transform *worldFromObject, const Transform *objectFromWorld,
         bool reverseOrientation, Float height, Float radius, Float innerRadius,
         Float phiMax)
        : worldFromObject(worldFromObject), objectFromWorld(objectFromWorld),
          reverseOrientation(reverseOrientation),
          transformSwapsHandedness(worldFromObject->SwapsHandedness()),
          height(height),
          radius(radius),
          innerRadius(innerRadius),
          phiMax(Radians(Clamp(phiMax, 0, 360))) {}

    static Disk *Create(const Transform *worldFromObject,
                        const Transform *objectFromWorld,
                        bool reverseOrientation,
                        const ParameterDictionary &dict,
                        Allocator alloc);

    PBRT_HOST_DEVICE
    Bounds3f WorldBound() const;

    PBRT_HOST_DEVICE
    pstd::optional<QuadricIntersection> BasicIntersect(const Ray &r, Float tMax) const {
        ProfilerScope p(ProfilePhase::ShapeIntersect);
        // Transform _Ray_ to object space
        Point3fi oi = (*objectFromWorld)(Point3fi(r.o));
        Vector3fi di = (*objectFromWorld)(Vector3fi(r.d));
        Ray ray(Point3f(oi), Vector3f(di), r.time, r.medium);

        // Compute plane intersection for disk

        // Reject disk intersections for rays parallel to the disk's plane
        if (ray.d.z == 0) return {};
        Float tShapeHit = (height - ray.o.z) / ray.d.z;
        if (tShapeHit <= 0 || tShapeHit >= tMax) return {};

        // See if hit point is inside disk radii and $\phimax$
        Point3f pHit = ray(tShapeHit);
        Float dist2 = pHit.x * pHit.x + pHit.y * pHit.y;
        if (dist2 > radius * radius || dist2 < innerRadius * innerRadius)
            return {};

        // Test disk $\phi$ value against $\phimax$
        Float phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) phi += 2 * Pi;
        if (phi > phiMax) return {};

        return QuadricIntersection{tShapeHit, pHit, phi};
    }

    PBRT_HOST_DEVICE
    SurfaceInteraction InteractionFromIntersection(const QuadricIntersection &isect,
                                                   const Vector3f &wo, Float time) const {
        Point3f pHit = isect.pObj;
        Float phi = isect.phi;
        Float dist2 = pHit.x * pHit.x + pHit.y * pHit.y;

        // Find parametric representation of disk hit
        Float u = phi / phiMax;
        Float rHit = std::sqrt(dist2);
        Float v = (radius - rHit) / (radius - innerRadius);
        Vector3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Vector3f dpdv =
            Vector3f(pHit.x, pHit.y, 0.) * (innerRadius - radius) / rHit;
        Normal3f dndu(0, 0, 0), dndv(0, 0, 0);

        // Refine disk intersection point
        pHit.z = height;

        // Compute error bounds for disk intersection
        Vector3f pError(0, 0, 0);

        // Initialize _SurfaceInteraction_ from parametric information
        return (*worldFromObject)(SurfaceInteraction(Point3fi(pHit, pError), Point2f(u, v),
                                                     (*objectFromWorld)(wo), dpdu, dpdv, dndu, dndv,
                                                     time, OrientationIsReversed() ^ TransformSwapsHandedness()));
    }

    PBRT_HOST_DEVICE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const {
        pstd::optional<QuadricIntersection> isect = BasicIntersect(ray, tMax);
        if (!isect)
            return {};

        SurfaceInteraction intr = InteractionFromIntersection(*isect, -ray.d, ray.time);
        return {{intr, isect->tHit}};
    }

    PBRT_HOST_DEVICE
    bool IntersectP(const Ray &r, Float tMax = Infinity) const {
        ProfilerScope p(ProfilePhase::ShapeIntersectP);
        return BasicIntersect(r, tMax).has_value();
    }

    PBRT_HOST_DEVICE
    Float Area() const {
        return phiMax * 0.5f * (radius * radius - innerRadius * innerRadius);
    }

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const {
        Point2f pd = SampleUniformDiskConcentric(u);
        Point3f pObj(pd.x * radius, pd.y * radius, height);
        Point3fi pi = (*worldFromObject)(Point3fi(pObj));
        Normal3f n = Normalize((*worldFromObject)(Normal3f(0, 0, 1)));
        if (reverseOrientation) n *= -1;

        return ShapeSample{Interaction(pi, n), 1 / Area()};
    }

    PBRT_HOST_DEVICE
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Interaction &ref,
                                       const Point2f &u) const {
        pstd::optional<ShapeSample> ss = Sample(u);
        if (!ss) return ss;

        ss->intr.time = ref.time;
        Vector3f wi = ss->intr.p() - ref.p();
        if (LengthSquared(wi) == 0)
            return {};
        else {
            wi = Normalize(wi);
            // Convert from area measure, as returned by the Sample() call
            // above, to solid angle measure.
            ss->pdf *= DistanceSquared(ref.p(), ss->intr.p()) / AbsDot(ss->intr.n, -wi);
            if (std::isinf(ss->pdf)) return {};
        }
        return ss;
    }
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &ref, const Vector3f &wi) const {
        // Intersect sample ray with area light geometry
        Ray ray = ref.SpawnRay(wi);
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si) return 0;

        // Convert light sample weight to solid angle measure
        Float pdf = DistanceSquared(ref.p(), si->intr.p()) /
            (AbsDot(si->intr.n, -wi) * Area());
        if (std::isinf(pdf)) pdf = 0.f;
        return pdf;
    }

    PBRT_HOST_DEVICE
    DirectionCone NormalBounds() const;

    PBRT_HOST_DEVICE
    bool OrientationIsReversed() const { return reverseOrientation; }
    PBRT_HOST_DEVICE
    bool TransformSwapsHandedness() const { return transformSwapsHandedness; }

    PBRT_HOST_DEVICE_INLINE
    Float SolidAngle(const Point3f &p, int nSamples = 512) const {
        return ShapeHandle(this).SampledSolidAngle(p, nSamples);
    }

    std::string ToString() const;

  private:
    // Disk Private Data
    const Transform *worldFromObject, *objectFromWorld;
    bool reverseOrientation;
    bool transformSwapsHandedness;

    Float height, radius, innerRadius, phiMax;
};

// Cylinder Declarations
class Cylinder {
  public:
    // Cylinder Public Methods
    Cylinder(const Transform *worldFromObject, const Transform *objectFromWorld,
             bool reverseOrientation, Float radius, Float zMin, Float zMax,
             Float phiMax)
        : worldFromObject(worldFromObject), objectFromWorld(objectFromWorld),
          reverseOrientation(reverseOrientation),
          transformSwapsHandedness(worldFromObject->SwapsHandedness()),
          radius(radius),
          zMin(std::min(zMin, zMax)),
          zMax(std::max(zMin, zMax)),
          phiMax(Radians(Clamp(phiMax, 0, 360))) {}

    static Cylinder *Create(const Transform *worldFromObject,
                            const Transform *objectFromWorld, bool reverseOrientation,
                            const ParameterDictionary &dict,
                            Allocator alloc);

    PBRT_HOST_DEVICE
    Bounds3f WorldBound() const;

    PBRT_HOST_DEVICE
    pstd::optional<QuadricIntersection> BasicIntersect(const Ray &r, Float tMax) const {
        ProfilerScope p(ProfilePhase::ShapeIntersect);
        Float phi;
        Point3f pHit;
        // Transform _Ray_ to object space
        Point3fi oi = (*objectFromWorld)(Point3fi(r.o));
        Vector3fi di = (*objectFromWorld)(Vector3fi(r.d));
        Ray ray(Point3f(oi), Vector3f(di), r.time, r.medium);

        // Compute quadratic cylinder coefficients

        // Solve quadratic equation for _t_ values
        FloatInterval t0, t1;
        if (!CylinderQuadratic(oi, di, &t0, &t1)) return {};

        // Check quadric shape _t0_ and _t1_ for nearest intersection
        if (t0.UpperBound() > tMax || t1.LowerBound() <= 0) return {};
        FloatInterval tShapeHit = t0;
        if (tShapeHit.LowerBound() <= 0) {
            tShapeHit = t1;
            if (tShapeHit.UpperBound() > tMax) return {};
        }

        // Compute cylinder hit point and $\phi$
        pHit = ray((Float)tShapeHit);

        // Refine cylinder intersection point
        Float hitRad = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
        pHit.x *= radius / hitRad;
        pHit.y *= radius / hitRad;
        phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) phi += 2 * Pi;

        // Test cylinder intersection against clipping parameters
        if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) {
            if (tShapeHit == t1) return {};
            tShapeHit = t1;
            if (t1.UpperBound() > tMax) return {};
            // Compute cylinder hit point and $\phi$
            pHit = ray((Float)tShapeHit);

            // Refine cylinder intersection point
            Float hitRad = std::sqrt(pHit.x * pHit.x + pHit.y * pHit.y);
            pHit.x *= radius / hitRad;
            pHit.y *= radius / hitRad;
            phi = std::atan2(pHit.y, pHit.x);
            if (phi < 0) phi += 2 * Pi;
            if (pHit.z < zMin || pHit.z > zMax || phi > phiMax) return {};
        }

        return QuadricIntersection{(Float)tShapeHit, pHit, phi};
    }

    PBRT_HOST_DEVICE
    SurfaceInteraction InteractionFromIntersection(const QuadricIntersection &isect,
                                                   const Vector3f &wo, Float time) const {
        Point3f pHit = isect.pObj;
        Float phi = isect.phi;

        // Find parametric representation of cylinder hit
        Float u = phi / phiMax;
        Float v = (pHit.z - zMin) / (zMax - zMin);

        // Compute cylinder $\dpdu$ and $\dpdv$
        Vector3f dpdu(-phiMax * pHit.y, phiMax * pHit.x, 0);
        Vector3f dpdv(0, 0, zMax - zMin);

        // Compute cylinder $\dndu$ and $\dndv$
        Vector3f d2Pduu = -phiMax * phiMax * Vector3f(pHit.x, pHit.y, 0);
        Vector3f d2Pduv(0, 0, 0), d2Pdvv(0, 0, 0);

        // Compute coefficients for fundamental forms
        Float E = Dot(dpdu, dpdu);
        Float F = Dot(dpdu, dpdv);
        Float G = Dot(dpdv, dpdv);
        Vector3f N = Normalize(Cross(dpdu, dpdv));
        Float e = Dot(N, d2Pduu);
        Float f = Dot(N, d2Pduv);
        Float g = Dot(N, d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Float invEGF2 = 1 / (E * G - F * F);
        Normal3f dndu = Normal3f((f * F - e * G) * invEGF2 * dpdu +
                                 (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv = Normal3f((g * F - f * G) * invEGF2 * dpdu +
                                 (f * F - g * E) * invEGF2 * dpdv);

        // Compute error bounds for cylinder intersection
        Vector3f pError = gamma(3) * Abs(Vector3f(pHit.x, pHit.y, 0));
        Point3fi pHitError(pHit, pError);

        // Initialize _SurfaceInteraction_ from parametric information
        return (*worldFromObject)(SurfaceInteraction(pHitError, Point2f(u, v),
                                                     (*objectFromWorld)(wo), dpdu, dpdv, dndu, dndv,
                                                     time, OrientationIsReversed() ^ TransformSwapsHandedness()));
    }

    PBRT_HOST_DEVICE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray,
                                                Float tMax = Infinity) const {
        pstd::optional<QuadricIntersection> isect = BasicIntersect(ray, tMax);
        if (!isect)
            return {};

        SurfaceInteraction intr = InteractionFromIntersection(*isect, -ray.d, ray.time);
        return {{intr, isect->tHit}};
    }

    PBRT_HOST_DEVICE
    bool IntersectP(const Ray &r, Float tMax = Infinity) const {
        ProfilerScope p(ProfilePhase::ShapeIntersectP);
        return BasicIntersect(r, tMax).has_value();
    }

    PBRT_HOST_DEVICE
    Float Area() const { return (zMax - zMin) * radius * phiMax; }

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const {
        Float z = Lerp(u[0], zMin, zMax);
        Float phi = u[1] * phiMax;
        Point3f pObj = Point3f(radius * std::cos(phi), radius * std::sin(phi), z);
        // Reproject _pObj_ to cylinder surface and compute _pObjError_
        Float hitRad = std::sqrt(pObj.x * pObj.x + pObj.y * pObj.y);
        pObj.x *= radius / hitRad;
        pObj.y *= radius / hitRad;
        Vector3f pObjError = gamma(3) * Abs(Vector3f(pObj.x, pObj.y, 0));
        Point3fi pi = (*worldFromObject)(Point3fi(pObj, pObjError));

        Normal3f n = Normalize((*worldFromObject)(Normal3f(pObj.x, pObj.y, 0)));
        if (reverseOrientation) n *= -1;

        return ShapeSample{Interaction(pi, n), 1 / Area()};
    }

    PBRT_HOST_DEVICE
    Float PDF(const Interaction &) const { return 1 / Area(); }

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Interaction &ref,
                                       const Point2f &u) const {
        pstd::optional<ShapeSample> ss = Sample(u);
        if (!ss) return ss;

        ss->intr.time = ref.time;
        Vector3f wi = ss->intr.p() - ref.p();
        if (LengthSquared(wi) == 0)
            return {};
        else {
            wi = Normalize(wi);
            // Convert from area measure, as returned by the Sample() call
            // above, to solid angle measure.
            ss->pdf *= DistanceSquared(ref.p(), ss->intr.p()) / AbsDot(ss->intr.n, -wi);
            if (std::isinf(ss->pdf)) return {};
        }
        return ss;
    }
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &ref, const Vector3f &wi) const {
        // Intersect sample ray with area light geometry
        Ray ray = ref.SpawnRay(wi);
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si) return 0;

        // Convert light sample weight to solid angle measure
        Float pdf = DistanceSquared(ref.p(), si->intr.p()) /
            (AbsDot(si->intr.n, -wi) * Area());
        if (std::isinf(pdf)) pdf = 0.f;
        return pdf;
    }


    PBRT_HOST_DEVICE
    bool OrientationIsReversed() const { return reverseOrientation; }
    PBRT_HOST_DEVICE
    bool TransformSwapsHandedness() const { return transformSwapsHandedness; }

    PBRT_HOST_DEVICE_INLINE
    Float SolidAngle(const Point3f &p, int nSamples = 512) const {
        return ShapeHandle(this).SampledSolidAngle(p, nSamples);
    }
    PBRT_HOST_DEVICE
    DirectionCone NormalBounds() const {
        return DirectionCone::EntireSphere();
    }

    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE
    bool CylinderQuadratic(const Point3fi &oi, const Vector3fi &di,
                           FloatInterval *t0, FloatInterval *t1) const {
        FloatInterval a = SumSquares(di.x, di.y);
        FloatInterval b = 2 * (di.x * oi.x + di.y * oi.y);
        FloatInterval c = SumSquares(oi.x, oi.y) - Sqr(FloatInterval(radius));

        // Solve quadratic equation for _t_ values
        //FloatInterval discrim = B * B - FloatInterval(4) * A * C;
        FloatInterval f = b / (2 * a);  // (o . d) / LengthSquared(d)
        FloatInterval fx = oi.x - f * di.x;
        FloatInterval fy = oi.y - f * di.y;
        FloatInterval sqrtf = Sqrt(SumSquares(fx, fy));
        FloatInterval discrim = 4 * a * (FloatInterval(radius) + sqrtf) *
            (FloatInterval(radius) - sqrtf);
        if (discrim.LowerBound() < 0) return false;
        FloatInterval rootDiscrim = Sqrt(discrim);

        // Compute quadratic _t_ values
        FloatInterval q;
        if ((Float)b < 0)
            q = -.5 * (b - rootDiscrim);
        else
            q = -.5 * (b + rootDiscrim);
        *t0 = q / a;
        *t1 = c / q;
        if (t0->LowerBound() > t1->LowerBound()) pstd::swap(*t0, *t1);
        return true;
    }

    // Cylinder Private Data
    const Transform *worldFromObject, *objectFromWorld;
    bool reverseOrientation;
    bool transformSwapsHandedness;

    Float radius, zMin, zMax, phiMax;
};


STAT_MEMORY_COUNTER("Memory/Triangles", triangleBytes);

// Triangle Declarations
class TriangleMesh {
public:
    // TriangleMesh Public Methods
    TriangleMesh(const Transform &worldFromObject, bool reverseOrientation,
                 std::vector<int> vertexIndices, std::vector<Point3f> p,
                 std::vector<Vector3f> S, std::vector<Normal3f> N,
                 std::vector<Point2f> uv, std::vector<int> faceIndices);

    pstd::vector<ShapeHandle> CreateTriangles(Allocator alloc);

    std::string ToString() const;

    static TriangleMesh *Create(const Transform *worldFromObject,
                                bool reverseOrientation,
                                const ParameterDictionary &dict,
                                Allocator alloc);

    PBRT_HOST_DEVICE
    SurfaceInteraction InteractionFromIntersection(int triIndex, pstd::array<Float, 3> b,
                                                   Float time, const Vector3f &wo,
                                                   pstd::optional<Transform> instanceToWorld = {}) const {
        const int *v = &vertexIndices[3 * triIndex];
        Point3f p0 = p[v[0]], p1 = p[v[1]], p2 = p[v[2]];
        if (instanceToWorld) {
            p0 = (*instanceToWorld)(p0);
            p1 = (*instanceToWorld)(p1);
            p2 = (*instanceToWorld)(p2);
        }
        // Compute triangle partial derivatives
        Vector3f dpdu, dpdv;
        pstd::array<Point2f, 3> triuv = uv ? pstd::array<Point2f, 3>({uv[v[0]], uv[v[1]], uv[v[2]]}) :
            pstd::array<Point2f, 3>({Point2f(0, 0), Point2f(1, 0), Point2f(1, 1)});

        // Compute deltas for triangle partial derivatives
        Vector2f duv02 = triuv[0] - triuv[2], duv12 = triuv[1] - triuv[2];
        Vector3f dp02 = p0 - p2, dp12 = p1 - p2;
        Float determinant = DifferenceOfProducts(duv02[0], duv12[1], duv02[1], duv12[0]);
        bool degenerateUV = std::abs(determinant) < 1e-32;
        if (!degenerateUV) {
            Float invdet = 1 / determinant;
            dpdu = DifferenceOfProducts(duv12[1], dp02, duv02[1], dp12) * invdet;
            dpdv = DifferenceOfProducts(duv02[0], dp12, duv12[0], dp02) * invdet;
        }
        if (degenerateUV || LengthSquared(Cross(dpdu, dpdv)) == 0) {
            Vector3f ng = Cross(p2 - p0, p1 - p0);
            if (LengthSquared(ng) == 0) {
                // TODO: should these be eliminated from the start?
                return {};
            }
            // Handle zero determinant for triangle partial derivative matrix
            CoordinateSystem(Normalize(ng), &dpdu, &dpdv);
        }

        // Interpolate $(u,v)$ parametric coordinates and hit point
        Point3f pHit = b[0] * p0 + b[1] * p1 + b[2] * p2;
        Point2f uvHit = b[0] * triuv[0] + b[1] * triuv[1] + b[2] * triuv[2];

        // Compute error bounds for triangle intersection
        Float xAbsSum = (std::abs(b[0] * p0.x) + std::abs(b[1] * p1.x) + std::abs(b[2] * p2.x));
        Float yAbsSum = (std::abs(b[0] * p0.y) + std::abs(b[1] * p1.y) + std::abs(b[2] * p2.y));
        Float zAbsSum = (std::abs(b[0] * p0.z) + std::abs(b[1] * p1.z) + std::abs(b[2] * p2.z));
        Vector3f pError = gamma(7) * Vector3f(xAbsSum, yAbsSum, zAbsSum);
        Point3fi pHitError(pHit, pError);

        // Fill in _SurfaceInteraction_ from triangle hit
        int faceIndex = faceIndices != nullptr ? faceIndices[triIndex] : 0;
        SurfaceInteraction isect(pHitError, uvHit, wo, dpdu, dpdv,
                                 Normal3f(0, 0, 0), Normal3f(0, 0, 0), time,
                                 reverseOrientation ^ transformSwapsHandedness,
                                 faceIndex);

        // Override surface normal in _isect_ for triangle
        // NOTE: this is implicitly assuming a counter-clockwise vertex ordering...
        isect.n = isect.shading.n = Normal3f(Normalize(Cross(dp02, dp12)));
        if (reverseOrientation ^ transformSwapsHandedness)
            isect.n = isect.shading.n = -isect.n;

        if (n != nullptr || s != nullptr) {
            // Initialize _Triangle_ shading geometry

            // Compute shading normal _ns_ for triangle
            Normal3f ns;
            if (n != nullptr) {
                ns = (b[0] * n[v[0]] + b[1] * n[v[1]] + b[2] * n[v[2]]);
                if (instanceToWorld)
                    ns = (*instanceToWorld)(ns);

                if (LengthSquared(ns) > 0)
                    ns = Normalize(ns);
                else
                    ns = isect.n;
            } else
                ns = isect.n;

            // Compute shading tangent _ss_ for triangle
            Vector3f ss;
            if (s != nullptr) {
                ss = (b[0] * s[v[0]] + b[1] * s[v[1]] + b[2] * s[v[2]]);
                if (instanceToWorld)
                    ss = (*instanceToWorld)(ss);

                if (LengthSquared(ss) == 0)
                    ss = isect.dpdu;
            } else
                ss = isect.dpdu;

            // Compute shading bitangent _ts_ for triangle and adjust _ss_
            Vector3f ts = Cross(ns, ss);
            if (LengthSquared(ts) > 0)
                ss = Cross(ts, ns);
            else
                CoordinateSystem(ns, &ss, &ts);

            // Compute $\dndu$ and $\dndv$ for triangle shading geometry
            Normal3f dndu, dndv;
            if (n != nullptr) {
                // Compute deltas for triangle partial derivatives of normal
                Vector2f duv02 = triuv[0] - triuv[2];
                Vector2f duv12 = triuv[1] - triuv[2];
                Normal3f dn1 = n[v[0]] - n[v[2]];
                Normal3f dn2 = n[v[1]] - n[v[2]];
                if (instanceToWorld) {
                    dn1 = (*instanceToWorld)(dn1);
                    dn2 = (*instanceToWorld)(dn2);
                }

                Float determinant = DifferenceOfProducts(duv02[0], duv12[1], duv02[1], duv12[0]);
                bool degenerateUV = std::abs(determinant) < 1e-32;
                if (degenerateUV) {
                    // We can still compute dndu and dndv, with respect to the
                    // same arbitrary coordinate system we use to compute dpdu
                    // and dpdv when this happens. It's important to do this
                    // (rather than giving up) so that ray differentials for
                    // rays reflected from triangles with degenerate
                    // parameterizations are still reasonable.
                    Vector3f dn = Cross(Vector3f(n[v[2]] - n[v[0]]),
                                        Vector3f(n[v[1]] - n[v[0]]));
                    if (instanceToWorld)
                        dn = (*instanceToWorld)(dn);

                    if (LengthSquared(dn) == 0)
                        dndu = dndv = Normal3f(0, 0, 0);
                    else {
                        Vector3f dnu, dnv;
                        CoordinateSystem(dn, &dnu, &dnv);
                        dndu = Normal3f(dnu);
                        dndv = Normal3f(dnv);
                    }
                } else {
                    Float invDet = 1 / determinant;
                    dndu = DifferenceOfProducts(duv12[1], dn1, duv02[1], dn2) * invDet;
                    dndv = DifferenceOfProducts(duv02[0], dn2, duv12[0], dn1) * invDet;
                }
            } else
                dndu = dndv = Normal3f(0, 0, 0);

            if (reverseOrientation) {
                ns = -ns;
                ts = -ts;
            }
            isect.SetShadingGeometry(ns, ss, ts, dndu, dndv, true);
        }
        return isect;
    }

    static void Init(Allocator alloc);

    // TriangleMesh Data
    bool reverseOrientation, transformSwapsHandedness;
    int nTriangles, nVertices;
    const int *vertexIndices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;

    static pstd::vector<const TriangleMesh *> *allMeshes;
};

#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
extern __device__ pstd::vector<const TriangleMesh *> *allTriangleMeshesGPU;
#endif

struct TriangleIntersection {
    Float b0, b1, b2;
    Float t;

    std::string ToString() const;
};

// Note: doesn't inherit from Shape...
class alignas(8) Triangle {
  public:
    // Triangle Public Methods
    Triangle() = default;
    Triangle(int meshIndex, int triIndex)
        : meshIndex(meshIndex), triIndex(triIndex) {
        triangleBytes += sizeof(*this);
    }

    PBRT_HOST_DEVICE
    Bounds3f WorldBound() const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax = Infinity) const;
    PBRT_HOST_DEVICE
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_HOST_DEVICE
    Float Area() const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Interaction &ref, const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &ref, const Vector3f &wi) const;
    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &) const { return 1 / Area(); }

    // Returns the solid angle subtended by the triangle w.r.t. the given
    // reference point p.
    PBRT_HOST_DEVICE_INLINE
    Float SolidAngle(const Point3f &p) const {
        // Project the vertices into the unit sphere around p.
        auto mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];
        Vector3f a = Normalize(mesh->p[v[0]] - p);
        Vector3f b = Normalize(mesh->p[v[1]] - p);
        Vector3f c = Normalize(mesh->p[v[2]] - p);

        return SphericalTriangleArea(a, b, c);
    }

    PBRT_HOST_DEVICE
    DirectionCone NormalBounds() const;

    PBRT_HOST_DEVICE
    bool OrientationIsReversed() const { return GetMesh()->reverseOrientation; }
    PBRT_HOST_DEVICE
    bool TransformSwapsHandedness() const {
        return GetMesh()->transformSwapsHandedness;
    }

    PBRT_HOST_DEVICE
    static pstd::optional<TriangleIntersection> Intersect(const Ray &ray, Float tMax, const Point3f &p0,
                                                          const Point3f &p1, const Point3f &p2);

    std::string ToString() const;

  private:
    PBRT_HOST_DEVICE
    const TriangleMesh *&GetMesh() const {
#ifdef __CUDA_ARCH__
        return (*allTriangleMeshesGPU)[meshIndex];
#else
        return (*TriangleMesh::allMeshes)[meshIndex];
#endif
    }

    // Triangle Private Methods
    PBRT_HOST_DEVICE
    pstd::array<Point2f, 3> GetUVs() const {
        auto mesh = GetMesh();
        if (mesh->uv) {
            const int *v = &mesh->vertexIndices[3 * triIndex];
            return { mesh->uv[v[0]], mesh->uv[v[1]], mesh->uv[v[2]] };
        } else
            return { Point2f(0, 0), Point2f(1, 0), Point2f(1, 1) };
    }

    // Triangle Private Data
    int meshIndex = -1;
    int triIndex = -1;
};

bool WritePlyFile(const std::string &filename,
                  pstd::span<const int> vertexIndices,
                  pstd::span<const Point3f> P, pstd::span<const Vector3f> S,
                  pstd::span<const Normal3f> N, pstd::span<const Point2f> UV,
                  pstd::span<const int> faceIndices);

// CurveType Declarations
enum class CurveType { Flat, Cylinder, Ribbon };

std::string ToString(CurveType type);

// CurveCommon Declarations
struct CurveCommon {
    CurveCommon(pstd::span<const Point3f> c, Float w0, Float w1, CurveType type,
                pstd::span<const Normal3f> norm,
                const Transform *worldFromObject,
                const Transform *objectFromWorld,
                bool reverseOrientation);

    std::string ToString() const;

    CurveType type;
    Point3f cpObj[4];
    Float width[2];
    Normal3f n[2];
    Float normalAngle, invSinNormalAngle;
    const Transform *worldFromObject, *objectFromWorld;
    bool reverseOrientation, transformSwapsHandedness;
};

// Curve Declarations
class Curve {
  public:
    // Curve Public Methods
    Curve(const CurveCommon *common, Float uMin, Float uMax)
        : common(common), uMin(uMin), uMax(uMax) {}

    static pstd::vector<ShapeHandle> Create(const Transform *worldFromObject,
                                            const Transform *objectFromWorld,
                                            bool reverseOrientation,
                                            const ParameterDictionary &dict,
                                            Allocator alloc);

    PBRT_HOST_DEVICE
    Bounds3f WorldBound() const;
    PBRT_HOST_DEVICE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    PBRT_HOST_DEVICE
    bool IntersectP(const Ray &ray, Float tMax) const;
    PBRT_HOST_DEVICE
    Float Area() const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &) const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Interaction &ref, const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &ref, const Vector3f &wi) const;

    PBRT_HOST_DEVICE
    bool OrientationIsReversed() const { return common->reverseOrientation; }
    PBRT_HOST_DEVICE
    bool TransformSwapsHandedness() const {
        return common->transformSwapsHandedness;
    }

    PBRT_HOST_DEVICE_INLINE
    Float SolidAngle(const Point3f &p, int nSamples = 512) const {
        return ShapeHandle(this).SampledSolidAngle(p, nSamples);
    }
    PBRT_HOST_DEVICE
    DirectionCone NormalBounds() const {
        return DirectionCone::EntireSphere();
    }

    std::string ToString() const;

  private:
    // Curve Private Methods
    PBRT_HOST_DEVICE
    bool intersect(const Ray &r, Float tMax, pstd::optional<ShapeIntersection> *si) const;
    PBRT_HOST_DEVICE
    bool recursiveIntersect(const Ray &r, Float tMax, pstd::span<const Point3f> cp,
                            const Transform &ObjectFromRay, Float u0, Float u1,
                            int depth, pstd::optional<ShapeIntersection> *si) const;

    // Curve Private Data
    const CurveCommon *common;
    Float uMin, uMax;
};

// BilinearPatch Declarations
class BilinearPatchMesh {
  public:
    BilinearPatchMesh(const Transform &worldFromObject, bool reverseOrientation,
                      std::vector<int> vertexIndices, std::vector<Point3f> p,
                      std::vector<Normal3f> N, std::vector<Point2f> uv,
                      std::vector<int> faceIndices, Distribution2D *imageDist);

    static pstd::vector<ShapeHandle> Create(const Transform *worldFromObject,
                                            bool reverseOrientation,
                                            const ParameterDictionary &dict,
                                            Allocator alloc);
    static pstd::vector<ShapeHandle> Create(const Transform *worldFromObject,
                                            bool reverseOrientation,
                                            std::vector<int> indices, std::vector<Point3f> p,
                                            std::vector<Normal3f> n, std::vector<Point2f> uv,
                                            std::vector<int> faceIndices,
                                            Distribution2D *imageDist,
                                            Allocator alloc);

    std::string ToString() const;

    static void Init(Allocator alloc);

  private:
    friend class BilinearPatch;

    bool reverseOrientation, transformSwapsHandedness;
    int nPatches, nVertices;
    const int *vertexIndices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;
    Distribution2D *imageDistribution;

    static pstd::vector<const BilinearPatchMesh *> *allMeshes;
};

#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
extern __device__ pstd::vector<const BilinearPatchMesh *> *allBilinearMeshesGPU;
#endif

struct BilinearIntersection {
    Point2f uv;
    Float t;

    std::string ToString() const;
};

class alignas(8) BilinearPatch {
  public:
    // BilinearPatch Public Methods
    BilinearPatch(int meshIndex, int blpIndex);

    PBRT_HOST_DEVICE
    Bounds3f WorldBound() const;
    PBRT_HOST_DEVICE
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax = Infinity) const;
    PBRT_HOST_DEVICE
    bool IntersectP(const Ray &ray, Float tMax = Infinity) const;
    PBRT_HOST_DEVICE
    Float Area() const;

    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Interaction &ref, const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &ref, const Vector3f &wi) const;
    PBRT_HOST_DEVICE
    pstd::optional<ShapeSample> Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &) const;

    PBRT_HOST_DEVICE
    Float SolidAngle(const Point3f &p) const;
    PBRT_HOST_DEVICE
    DirectionCone NormalBounds() const;

    PBRT_HOST_DEVICE
    bool OrientationIsReversed() const { return GetMesh()->reverseOrientation; }
    PBRT_HOST_DEVICE
    bool TransformSwapsHandedness() const { return GetMesh()->transformSwapsHandedness; }

    std::string ToString() const;

    PBRT_HOST_DEVICE
    static pstd::optional<BilinearIntersection> Intersect(const Ray &ray, Float tMax,
                                                          const Point3f &p00, const Point3f &p10,
                                                          const Point3f &p01, const Point3f &p11);

private:
    PBRT_HOST_DEVICE
    bool IsQuad() const;

    PBRT_HOST_DEVICE
    const BilinearPatchMesh *&GetMesh() const {
#ifdef __CUDA_ARCH__
        return (*allBilinearMeshesGPU)[meshIndex];
#else
        return (*BilinearPatchMesh::allMeshes)[meshIndex];
#endif
    }

    // BilinearPatch Private Data
    int meshIndex, blpIndex;
    Float area;
};

inline Bounds3f ShapeHandle::WorldBound() const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->WorldBound();
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->WorldBound();
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->WorldBound();
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->WorldBound();
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->WorldBound();
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->WorldBound();
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline pstd::optional<ShapeIntersection>
ShapeHandle::Intersect(const Ray &ray, Float tMax) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->Intersect(ray, tMax);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->Intersect(ray, tMax);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->Intersect(ray, tMax);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->Intersect(ray, tMax);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->Intersect(ray, tMax);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->Intersect(ray, tMax);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline bool ShapeHandle::IntersectP(const Ray &ray, Float tMax) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->IntersectP(ray, tMax);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->IntersectP(ray, tMax);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->IntersectP(ray, tMax);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->IntersectP(ray, tMax);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->IntersectP(ray, tMax);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->IntersectP(ray, tMax);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline Float ShapeHandle::Area() const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->Area();
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->Area();
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->Area();
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->Area();
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->Area();
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->Area();
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline pstd::optional<ShapeSample> ShapeHandle::Sample(const Point2f &u) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->Sample(u);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->Sample(u);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->Sample(u);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->Sample(u);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->Sample(u);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->Sample(u);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline Float ShapeHandle::PDF(const Interaction &in) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->PDF(in);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->PDF(in);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->PDF(in);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->PDF(in);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->PDF(in);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->PDF(in);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline pstd::optional<ShapeSample> ShapeHandle::Sample(const Interaction &ref,
                                                       const Point2f &u) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->Sample(ref, u);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->Sample(ref, u);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->Sample(ref, u);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->Sample(ref, u);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->Sample(ref, u);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->Sample(ref, u);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline Float ShapeHandle::PDF(const Interaction &ref, const Vector3f &wi) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->PDF(ref, wi);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->PDF(ref, wi);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->PDF(ref, wi);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->PDF(ref, wi);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->PDF(ref, wi);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->PDF(ref, wi);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline Float ShapeHandle::SolidAngle(const Point3f &p, int nSamples) const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->SolidAngle(p);
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->SolidAngle(p);
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->SolidAngle(p);
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->SolidAngle(p, nSamples);
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->SolidAngle(p, nSamples);
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->SolidAngle(p, nSamples);
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline DirectionCone ShapeHandle::NormalBounds() const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->NormalBounds();
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->NormalBounds();
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->NormalBounds();
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->NormalBounds();
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->NormalBounds();
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->NormalBounds();
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline bool ShapeHandle::OrientationIsReversed() const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->OrientationIsReversed();
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->OrientationIsReversed();
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->OrientationIsReversed();
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->OrientationIsReversed();
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->OrientationIsReversed();
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->OrientationIsReversed();
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

inline bool ShapeHandle::TransformSwapsHandedness() const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->TransformSwapsHandedness();
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->TransformSwapsHandedness();
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->TransformSwapsHandedness();
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->TransformSwapsHandedness();
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->TransformSwapsHandedness();
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->TransformSwapsHandedness();
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_SHAPES_H
