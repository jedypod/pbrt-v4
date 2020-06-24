
#ifndef PBRT_SCATTERING_H
#define PBRT_SCATTERING_H

#include <pbrt/pbrt.h>

#include <pbrt/util/math.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/spectrum.h>

namespace pbrt {

PBRT_HOST_DEVICE_INLINE
Vector3f Reflect(const Vector3f &wo, const Vector3f &n) {
    return -wo + 2 * Dot(wo, n) * n;
}

PBRT_HOST_DEVICE_INLINE
bool Refract(const Vector3f &wi, const Normal3f &n, Float eta,
             Vector3f *wt) {
    // Compute $\cos \theta_\roman{t}$ using Snell's law
    Float cosTheta_i = Dot(n, wi);
    Float sin2Theta_i = std::max<Float>(0, 1 - cosTheta_i * cosTheta_i);
    Float sin2Theta_t = sin2Theta_i / Sqr(eta);

    // Handle total internal reflection for transmission
    if (sin2Theta_t >= 1) return false;
    Float cosTheta_t = SafeSqrt(1 - sin2Theta_t);
    *wt = -wi / eta + (cosTheta_i / eta - cosTheta_t) * Vector3f(n);
    return true;
}

PBRT_HOST_DEVICE_INLINE
Float FrDielectric(Float cosTheta_i, Float eta) {
    cosTheta_i = Clamp(cosTheta_i, -1, 1);
    // Potentially swap indices of refraction
    bool entering = cosTheta_i > 0.f;
    if (!entering) {
        eta = 1 / eta;
        cosTheta_i = std::abs(cosTheta_i);
    }

    // Compute _cosThetaT_ using Snell's law
    Float sinTheta_i = SafeSqrt(1 - cosTheta_i * cosTheta_i);
    Float sinTheta_t = sinTheta_i / eta;

    // Handle total internal reflection
    if (sinTheta_t >= 1) return 1;
    Float cosTheta_t = SafeSqrt(1 - sinTheta_t * sinTheta_t);
    Float Rparl = (eta * cosTheta_i - cosTheta_t) /
                  (eta * cosTheta_i + cosTheta_t);
    Float Rperp = (cosTheta_i - eta * cosTheta_t) /
                  (cosTheta_i + eta * cosTheta_t);
    return (Rparl * Rparl + Rperp * Rperp) / 2;
}

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
PBRT_HOST_DEVICE_INLINE
SampledSpectrum FrConductor(Float cosTheta_i, const SampledSpectrum &eta,
                            const SampledSpectrum &k) {
    cosTheta_i = Clamp(cosTheta_i, -1, 1);
    SampledSpectrum etak = k;

    Float cos2Theta_i = cosTheta_i * cosTheta_i;
    Float sin2Theta_i = 1 - cos2Theta_i;
    SampledSpectrum eta2 = eta * eta;
    SampledSpectrum etak2 = etak * etak;

    SampledSpectrum t0 = eta2 - etak2 - SampledSpectrum(sin2Theta_i);
    SampledSpectrum a2plusb2 = Sqrt(t0 * t0 + 4 * eta2 * etak2);
    SampledSpectrum t1 = a2plusb2 + SampledSpectrum(cos2Theta_i);
    SampledSpectrum a = Sqrt(0.5f * (a2plusb2 + t0));
    SampledSpectrum t2 = (Float)2 * cosTheta_i * a;
    SampledSpectrum Rs = (t1 - t2) / (t1 + t2);

    SampledSpectrum t3 = cos2Theta_i * a2plusb2 +
        SampledSpectrum(sin2Theta_i * sin2Theta_i);
    SampledSpectrum t4 = t2 * sin2Theta_i;
    SampledSpectrum Rp = Rs * (t3 - t4) / (t3 + t4);

    return 0.5f * (Rp + Rs);
}

PBRT_HOST_DEVICE
Float FresnelMoment1(Float invEta);
PBRT_HOST_DEVICE
Float FresnelMoment2(Float invEta);

// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
//
// The Schlick Fresnel approximation is:
//
// R = R(0) + (1 - R(0)) (1 - cos theta)^5,
//
// where R(0) is the reflectance at normal indicence.
PBRT_HOST_DEVICE_INLINE
Float SchlickWeight(Float cosTheta) {
    Float m = Clamp(1 - cosTheta, 0, 1);
    return Pow<5>(m);
}

PBRT_HOST_DEVICE_INLINE
Float FrSchlick(Float R0, Float cosTheta) {
    return Lerp(SchlickWeight(cosTheta), R0, 1);
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum FrSchlick(const SampledSpectrum &R0, Float cosTheta) {
    return Lerp(SchlickWeight(cosTheta), R0, SampledSpectrum(1.));
}

// For a dielectric, R(0) = (eta - 1)^2 / (eta + 1)^2, assuming we're
// coming from air..
PBRT_HOST_DEVICE_INLINE
Float SchlickR0FromEta(Float eta) { return Sqr(eta - 1) / Sqr(eta + 1); }

PBRT_HOST_DEVICE_INLINE
Float EvaluateHenyeyGreenstein(Float cosTheta, Float g) {
    Float denom = 1 + g * g + 2 * g * cosTheta;
    return Inv4Pi * (1 - g * g) / (denom * SafeSqrt(denom));
}

} // namespace pbrt

#endif //  PBRT_SCATTERING_H
