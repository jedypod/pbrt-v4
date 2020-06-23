// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// core/interaction.cpp*
#include <pbrt/interaction.h>

#include <pbrt/base/camera.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/samplers.h>
#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>

#include <cmath>

namespace pbrt {

std::string Interaction::ToString() const {
    return StringPrintf(
        "[ Interaction pi: %s n: %s uv: %s wo: %s time: %s "
        "medium: %s mediumInterface: %s ]",
        pi, n, uv, wo, time, medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)");
}

std::string MediumInteraction::ToString() const {
    return StringPrintf(
        "[ MediumInteraction pi: %s n: %s uv: %s wo: %s time: %s "
        "sigma_a: %s sigma_s: %s sigma_maj: %s Le: %s medium: %s mediumInterface: %s "
        "phase: %s ]",
        pi, n, uv, wo, time, sigma_a, sigma_s, sigma_maj, Le,
        medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)",
        phase ? phase.ToString().c_str() : "(nullptr)");
}

// SurfaceInteraction Method Definitions
void SurfaceInteraction::ComputeScatteringFunctions(const RayDifferential &ray,
                                                    const SampledWavelengths &lambda,
                                                    CameraHandle camera,
                                                    ScratchBuffer &scratchBuffer,
                                                    SamplerHandle sampler) {
    ComputeDifferentials(ray, camera);

    if (!material)
        return;

    FloatTextureHandle displacement = material.GetDisplacement();
    if (displacement) {
        Vector3f dpdu, dpdv;
        Bump(UniversalTextureEvaluator(), displacement, *this, &dpdu, &dpdv);
        SetShadingGeometry(Normal3f(Normalize(Cross(dpdu, dpdv))), dpdu, dpdv,
                           shading.dndu, shading.dndv, false);
    }

    {
        bsdf =
            material.GetBSDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);

        if (GetOptions().forceDiffuse) {
            SampledSpectrum r = bsdf->rho(wo, {sampler.Get1D()}, {sampler.Get2D()});
            bsdf = scratchBuffer.Alloc<BSDF>(
                *this, scratchBuffer.Alloc<DiffuseBxDF>(r, SampledSpectrum(0.), 0.),
                bsdf->eta);
        }
    }
    {
        bssrdf =
            material.GetBSSRDF(UniversalTextureEvaluator(), *this, lambda, scratchBuffer);
    }
}

void SurfaceInteraction::ComputeDifferentials(const RayDifferential &ray,
                                              CameraHandle camera) const {
    if (ray.hasDifferentials && Dot(n, ray.rxDirection) != 0 &&
        Dot(n, ray.ryDirection) != 0) {
        // Estimate screen space change in $\pt{}$ and $(u,v)$

        // Compute auxiliary intersection points with plane
        Float d = -Dot(n, Vector3f(p()));
        Float tx = (-Dot(n, Vector3f(ray.rxOrigin)) - d) / Dot(n, ray.rxDirection);
        CHECK(!std::isinf(tx) && !std::isnan(tx));
        Float ty = (-Dot(n, Vector3f(ray.ryOrigin)) - d) / Dot(n, ray.ryDirection);
        CHECK(!std::isinf(ty) && !std::isnan(ty));

        Point3f px = ray.rxOrigin + tx * ray.rxDirection;
        Point3f py = ray.ryOrigin + ty * ray.ryDirection;
        dpdx = px - p();
        dpdy = py - p();
    } else
        camera.ApproximatedPdxy(*this);

    Float a00 = Dot(dpdu, dpdu), a01 = Dot(dpdu, dpdv), a11 = Dot(dpdv, dpdv);
    Float invDet = 1 / (DifferenceOfProducts(a00, a11, a01, a01));

    Float b0x = Dot(dpdu, dpdx), b1x = Dot(dpdv, dpdx);
    Float b0y = Dot(dpdu, dpdy), b1y = Dot(dpdv, dpdy);

    /* Set the UV partials to zero if dpdu and/or dpdv == 0 */
    invDet = std::isfinite(invDet) ? invDet : 0.f;

    dudx = DifferenceOfProducts(a11, b0x, a01, b1x) * invDet;
    dvdx = DifferenceOfProducts(a00, b1x, a01, b0x) * invDet;

    dudy = DifferenceOfProducts(a11, b0y, a01, b1y) * invDet;
    dvdy = DifferenceOfProducts(a00, b1y, a01, b0y) * invDet;
}

SampledSpectrum SurfaceInteraction::Le(const Vector3f &w,
                                       const SampledWavelengths &lambda) const {
    return areaLight ? areaLight.L(*this, w, lambda) : SampledSpectrum(0.f);
}

RayDifferential SurfaceInteraction::SpawnRay(const RayDifferential &rayi,
                                             const Vector3f &wi, BxDFFlags flags) const {
    RayDifferential rd(SpawnRay(wi));
    if (!rayi.hasDifferentials)
        return rd;

    if (flags == (BxDFFlags::Reflection | BxDFFlags::Specular)) {
        rd.hasDifferentials = true;
        rd.rxOrigin = p() + dpdx;
        rd.ryOrigin = p() + dpdy;
        // Compute differential reflected directions
        Normal3f dndx = shading.dndu * dudx + shading.dndv * dvdx;
        Normal3f dndy = shading.dndu * dudy + shading.dndv * dvdy;
        Vector3f dwodx = -rayi.rxDirection - wo, dwody = -rayi.ryDirection - wo;
        Normal3f ns = shading.n;
        Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
        Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);
        rd.rxDirection = wi - dwodx + 2.f * Vector3f(Dot(wo, ns) * dndx + dDNdx * ns);
        rd.ryDirection = wi - dwody + 2.f * Vector3f(Dot(wo, ns) * dndy + dDNdy * ns);
    } else if (flags == (BxDFFlags::Transmission | BxDFFlags::Specular)) {
        rd.hasDifferentials = true;
        rd.rxOrigin = p() + dpdx;
        rd.ryOrigin = p() + dpdy;

        Normal3f dndx = shading.dndu * dudx + shading.dndv * dvdx;
        Normal3f dndy = shading.dndu * dudy + shading.dndv * dvdy;
        Normal3f ns = shading.n;

        // NOTE: eta coming in is now 1/eta from the derivation below, so
        // there's a 1/ here now...
        Float eta = 1 / bsdf->eta;
        if (Dot(wo, ns) < 0) {
            ns = -ns;
            dndx = -dndx;
            dndy = -dndy;
        }

        /*
          Notes on the derivation:
          - pbrt computes the refracted ray as: \wi = -\eta \omega_o + [ \eta
          (\wo \cdot \N) - \cos \theta_t ] \N It flips the normal to lie in the
          same hemisphere as \wo, and then \eta is the relative IOR from \wo's
          medium to \wi's medium.
          - If we denote the term in brackets by \mu, then we have: \wi = -\eta
          \omega_o + \mu \N
          - Now let's take the partial derivative. (We'll use "d" for \partial
          in the following for brevity.) We get: -\eta d\omega_o / dx + \mu
          dN/dx + d\mu/dx N.
          - We have the values of all of these except for d\mu/dx (using bits
          from the derivation of specularly reflected ray deifferentials).
          - The first term of d\mu/dx is easy: \eta d(\wo \cdot N)/dx. We
          already have d(\wo \cdot N)/dx.
          - The second term takes a little more work. We have:
          \cos \theta_i = \sqrt{1 - \eta^2 (1 - (\wo \cdot N)^2)}.
          Starting from (\wo \cdot N)^2 and reading outward, we have \cos^2
          \theta_o, then \sin^2 \theta_o, then \sin^2 \theta_i (via Snell's
          law), then \cos^2 \theta_i and then \cos \theta_i.
          - Let's take the partial derivative of the sqrt expression. We get:
          1 / 2 * 1 / \cos \theta_i * d/dx (1 - \eta^2 (1 - (\wo \cdot N)^2)).
          - That partial derivatve is equal to:
          d/dx \eta^2 (\wo \cdot N)^2 = 2 \eta^2 (\wo \cdot N) d/dx (\wo \cdot
          N).
          - Plugging it in, we have d\mu/dx =
          \eta d(\wo \cdot N)/dx - (\eta^2 (\wo \cdot N) d/dx (\wo \cdot
          N))/(-\wi \cdot N).
        */
        Vector3f dwodx = -rayi.rxDirection - wo, dwody = -rayi.ryDirection - wo;
        Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
        Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);

        Float mu = eta * Dot(wo, ns) - AbsDot(wi, ns);
        Float dmudx = (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdx;
        Float dmudy = (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdy;

        rd.rxDirection = wi - eta * dwodx + Vector3f(mu * dndx + dmudx * ns);
        rd.ryDirection = wi - eta * dwody + Vector3f(mu * dndy + dmudy * ns);
    }

    return rd;
}

void SurfaceInteraction::SkipIntersection(RayDifferential *ray, Float t) const {
    *((Ray *)ray) = SpawnRay(ray->d);
    if (ray->hasDifferentials) {
        ray->rxOrigin = ray->rxOrigin + t * ray->rxDirection;
        ray->ryOrigin = ray->ryOrigin + t * ray->ryDirection;
    }
}

std::string SurfaceInteraction::ToString() const {
    return StringPrintf(
        "[ SurfaceInteraction pi: %s n: %s uv: %s wo: %s time: %s "
        "medium: %s mediumInterface: %s dpdu: %s dpdv: %s dndu: %s dndv: %s "
        "shading.n: %s shading.dpdu: %s shading.dpdv: %s "
        "shading.dndu: %s shading.dndv: %s, bsdf: %s bssrdf: %s material: %s "
        "areaLight: %s dpdx: %s dpdy: %s dudx: %f dvdx: %f "
        "dudy: %f dvdy: %f faceIndex: %d ]",
        pi, n, uv, wo, time, medium ? medium.ToString().c_str() : "(nullptr)",
        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)", dpdu, dpdv,
        dndu, dndv, shading.n, shading.dpdu, shading.dpdv, shading.dndu, shading.dndv,
        bsdf ? bsdf->ToString().c_str() : "(nullptr)",
        bssrdf ? BSSRDFHandle(bssrdf).ToString().c_str() : "(nullptr)",
        material ? material.ToString().c_str() : "(nullptr)",
        areaLight ? areaLight.ToString().c_str() : "(nullptr)", dpdx, dpdy, dudx, dvdx,
        dudy, dvdy, faceIndex);
}

}  // namespace pbrt
