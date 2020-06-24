
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

// core/interaction.cpp*
#include <pbrt/interaction.h>

#include <pbrt/base.h>
#include <pbrt/bsdf.h>
#include <pbrt/bssrdf.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/util/check.h>
#include <pbrt/util/math.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/primitive.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/rng.h>

#include <cmath>

namespace pbrt {

const Medium *Interaction::GetMedium(const Vector3f &w) const {
    if (mediumInterface != nullptr)
        return Dot(w, n) > 0 ? mediumInterface->outside : mediumInterface->inside;
    return medium;
}

const Medium *Interaction::GetMedium() const {
    if (mediumInterface != nullptr) {
        DCHECK_EQ(mediumInterface->inside, mediumInterface->outside);
        return mediumInterface->inside;
    }
    return medium;
}

std::string Interaction::ToString() const {
    return StringPrintf("[ Interaction pi: %s n: %s uv: %s wo: %s time: %s "
                        "medium: %s mediumInterface: %s ]",
                        pi, n, uv, wo, time,
                        medium ? medium->ToString().c_str() : "(nullptr)",
                        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)");
}

std::string MediumInteraction::ToString() const {
    return StringPrintf("[ MediumInteraction pi: %s n: %s uv: %s wo: %s time: %s "
                        "medium: %s mediumInterface: %s phase: %s ]",
                        pi, n, uv, wo, time,
                        medium ? medium->ToString().c_str() : "(nullptr)",
                        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)",
                        phase ? phase->ToString().c_str() : "(nullptr)");
}

// SurfaceInteraction Method Definitions
void SurfaceInteraction::ComputeScatteringFunctions(
    const RayDifferential &ray, const SampledWavelengths &lambda,
    const Camera &camera, MaterialBuffer &materialBuffer, Sampler &sampler,
    TransportMode mode) {
    ComputeDifferentials(ray, camera);

    if (*material == nullptr)
        return;

    FloatTextureHandle displacement = material->GetDisplacement();
    if (displacement) {
        Vector3f dpdu, dpdv;
        CHECK(Bump(UniversalTextureEvaluator(), displacement, *this, &dpdu, &dpdv));
        SetShadingGeometry(Normal3f(Normalize(Cross(dpdu, dpdv))), dpdu, dpdv,
                           shading.dndu, shading.dndv, false);
    }

    {
        ProfilerScope p(ProfilePhase::GetBSDF);
        bsdf = material->GetBSDF(UniversalTextureEvaluator(), *this, lambda,
                                 materialBuffer, mode);

#ifndef __CUDA_ARCH__
        if (PbrtOptions.forceDiffuse) {
            SampledSpectrum r = bsdf->rho(wo, {sampler.Get1D()}, {sampler.Get2D()});
            bsdf = materialBuffer.Alloc<BSDF>(*this,
                                     materialBuffer.Alloc<LambertianBxDF>(r, SampledSpectrum(0.), 0.),
                                     bsdf->eta);
        }
#endif
    }
    {
        ProfilerScope p(ProfilePhase::GetBSSRDF);
        bssrdf = material->GetBSSRDF(*this, lambda, materialBuffer, mode);
    }
}

void SurfaceInteraction::ComputeDifferentials(
        const RayDifferential &ray, const Camera &camera) const {
    if (ray.hasDifferentials) {
        // Estimate screen space change in $\pt{}$ and $(u,v)$

        // Compute auxiliary intersection points with plane
        Float d = -Dot(n, Vector3f(p()));
        Float tx =
            (-Dot(n, Vector3f(ray.rxOrigin)) - d) / Dot(n, ray.rxDirection);
        CHECK_RARE(1e-6, std::isinf(tx) || std::isnan(tx));
        if (std::isinf(tx) || std::isnan(tx)) goto fail;
        Point3f px = ray.rxOrigin + tx * ray.rxDirection;
        Float ty =
            (-Dot(n, Vector3f(ray.ryOrigin)) - d) / Dot(n, ray.ryDirection);

        CHECK_RARE(1e-6, std::isinf(ty) || std::isnan(ty));
        if (std::isinf(ty) || std::isnan(ty)) goto fail;
        Point3f py = ray.ryOrigin + ty * ray.ryDirection;
        dpdx = px - p();
        dpdy = py - p();
    } else
        camera.ApproximatedPdxy(*this);

    // Compute $(u,v)$ offsets at auxiliary points

    // Choose two dimensions to use for ray offset computation
    int dim[2];
    if (std::abs(n.x) > std::abs(n.y) && std::abs(n.x) > std::abs(n.z)) {
        dim[0] = 1;
        dim[1] = 2;
    } else if (std::abs(n.y) > std::abs(n.z)) {
        dim[0] = 0;
        dim[1] = 2;
    } else {
        dim[0] = 0;
        dim[1] = 1;
    }

    // Initialize _A_, _Bx_, and _By_ matrices for offset computation
    {
    Float A[2][2] = {{dpdu[dim[0]], dpdv[dim[0]]},
                     {dpdu[dim[1]], dpdv[dim[1]]}};
    Float Bx[2] = {dpdx[dim[0]], dpdx[dim[1]]};
    Float By[2] = {dpdy[dim[0]], dpdy[dim[1]]};
    if (!SolveLinearSystem2x2(A, Bx, &dudx, &dvdx)) dudx = dvdx = 0;
    if (!SolveLinearSystem2x2(A, By, &dudy, &dvdy)) dudy = dvdy = 0;
    }
    return;

fail:
    dudx = dvdx = 0;
    dudy = dvdy = 0;
    dpdx = dpdy = Vector3f(0, 0, 0);
}

SampledSpectrum SurfaceInteraction::Le(const Vector3f &w,
                                       const SampledWavelengths &lambda) const {
    return (areaLight && *areaLight) ? areaLight->L(*this, w, lambda) : SampledSpectrum(0.f);
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
        rd.rxDirection =
            wi - dwodx + 2.f * Vector3f(Dot(wo, ns) * dndx + dDNdx * ns);
        rd.ryDirection =
            wi - dwody + 2.f * Vector3f(Dot(wo, ns) * dndy + dDNdy * ns);
    } else if (flags == (BxDFFlags::Transmission | BxDFFlags::Specular)) {
        rd.hasDifferentials = true;
        rd.rxOrigin = p() + dpdx;
        rd.ryOrigin = p() + dpdy;

        Normal3f dndx = shading.dndu * dudx + shading.dndv * dvdx;
        Normal3f dndy = shading.dndu * dudy + shading.dndv * dvdy;
        Normal3f ns = shading.n;

        // NOTE: eta coming in is now 1/eta from the derivation below, so there's a 1/ here now...
        Float eta = 1 / bsdf->eta;
        if (Dot(wo, ns) < 0) {
            ns = -ns;
            dndx = -dndx;
            dndy = -dndy;
        }

        /*
          Notes on the derivation:
          - pbrt computes the refracted ray as: \wi = -\eta \omega_o + [ \eta (\wo \cdot \N) - \cos \theta_t ] \N
          It flips the normal to lie in the same hemisphere as \wo, and then \eta is the relative IOR from
          \wo's medium to \wi's medium.
          - If we denote the term in brackets by \mu, then we have: \wi = -\eta \omega_o + \mu \N
          - Now let's take the partial derivative. (We'll use "d" for \partial in the following for brevity.)
          We get: -\eta d\omega_o / dx + \mu dN/dx + d\mu/dx N.
          - We have the values of all of these except for d\mu/dx (using bits from the derivation of specularly
          reflected ray deifferentials).
          - The first term of d\mu/dx is easy: \eta d(\wo \cdot N)/dx. We already have d(\wo \cdot N)/dx.
          - The second term takes a little more work. We have:
          \cos \theta_i = \sqrt{1 - \eta^2 (1 - (\wo \cdot N)^2)}.
          Starting from (\wo \cdot N)^2 and reading outward, we have \cos^2 \theta_o, then \sin^2 \theta_o,
          then \sin^2 \theta_i (via Snell's law), then \cos^2 \theta_i and then \cos \theta_i.
          - Let's take the partial derivative of the sqrt expression. We get:
          1 / 2 * 1 / \cos \theta_i * d/dx (1 - \eta^2 (1 - (\wo \cdot N)^2)).
          - That partial derivatve is equal to:
          d/dx \eta^2 (\wo \cdot N)^2 = 2 \eta^2 (\wo \cdot N) d/dx (\wo \cdot N).
          - Plugging it in, we have d\mu/dx =
          \eta d(\wo \cdot N)/dx - (\eta^2 (\wo \cdot N) d/dx (\wo \cdot N))/(-\wi \cdot N).
        */
        Vector3f dwodx = -rayi.rxDirection - wo,
                 dwody = -rayi.ryDirection - wo;
        Float dDNdx = Dot(dwodx, ns) + Dot(wo, dndx);
        Float dDNdy = Dot(dwody, ns) + Dot(wo, dndy);

        Float mu = eta * Dot(wo, ns) - AbsDot(wi, ns);
        Float dmudx =
            (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdx;
        Float dmudy =
            (eta - (eta * eta * Dot(wo, ns)) / AbsDot(wi, ns)) * dDNdy;

        rd.rxDirection =
            wi - eta * dwodx + Vector3f(mu * dndx + dmudx * ns);
        rd.ryDirection =
            wi - eta * dwody + Vector3f(mu * dndy + dmudy * ns);
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
    return StringPrintf("[ SurfaceInteraction pi: %s n: %s uv: %s wo: %s time: %s "
                        "medium: %s mediumInterface: %s dpdu: %s dpdv: %s dndu: %s dndv: %s "
                        "shading.n: %s shading.dpdu: %s shading.dpdv: %s "
                        "shading.dndu: %s shading.dndv: %s, bsdf: %s bssrdf: %s material: %s "
                        "areaLight: %s dpdx: %s dpdy: %s dudx: %f dvdx: %f "
                        "dudy: %f dvdy: %f faceIndex: %d ]",
                        pi, n, uv, wo, time,
                        medium ? medium->ToString().c_str() : "(nullptr)",
                        mediumInterface ? mediumInterface->ToString().c_str() : "(nullptr)",
                        dpdu, dpdv, dndu, dndv, shading.n, shading.dpdu, shading.dpdv,
                        shading.dndu, shading.dndv,
                        bsdf ? bsdf->ToString().c_str() : "(nullptr)",
                        bssrdf ? bssrdf->ToString().c_str() : "(nullptr)",
                        material ? material->ToString().c_str() : "(nullptr)",
                        areaLight ? areaLight->ToString().c_str() : "(nullptr)",
                        dpdx, dpdy, dudx, dvdx, dudy, dvdy, faceIndex);
}

}  // namespace pbrt
