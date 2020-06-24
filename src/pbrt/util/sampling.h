
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

#ifndef PBRT_SAMPLING_H
#define PBRT_SAMPLING_H

// sampling/sampling.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/array2d.h>
#include <pbrt/util/check.h>
#include <pbrt/util/lowdiscrepancy.h> // yuck: for Hammersley generator...
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <ostream>
#include <string>
#include <vector>

namespace pbrt {

// Sampling Inline Functions
PBRT_HOST_DEVICE_INLINE
Float BalanceHeuristic(int nf, Float fPDF, int ng, Float gPDF) {
    return (nf * fPDF) / (nf * fPDF + ng * gPDF);
}

PBRT_HOST_DEVICE_INLINE
Float PowerHeuristic(int nf, Float fPDF, int ng, Float gPDF) {
    Float f = nf * fPDF, g = ng * gPDF;
    return (f * f) / (f * f + g * g);
}

template <typename AccumType>
class VarianceEstimator {
 public:
    template <typename T>
    PBRT_HOST_DEVICE_INLINE
    void Add(T v) {
        // Welford's algorithm
        ++count;
        AccumType delta = v - mean;
        mean += delta / count;
        AccumType delta2 = v - mean;
        S += delta * delta2;
    }

    PBRT_HOST_DEVICE_INLINE
    void Add(const VarianceEstimator &ve) {
        if (ve.count == 0)
            return;

        // Via Chan et al's parallel algorithm
        mean = (count * mean + ve.count * ve.mean) / (count + ve.count);
        // Eq 2.1b
        S = S + ve.S + AccumType(count) / AccumType(ve.count * (count + ve.count)) *
            Sqr(ve.count * (mean - ve.mean));
        count += ve.count;
    }

    PBRT_HOST_DEVICE_INLINE
    AccumType Mean() const {
        return mean;
    }

    PBRT_HOST_DEVICE_INLINE
    AccumType Variance() const {
        return (count > 1) ? S / (count - 1) : AccumType();
    }

    PBRT_HOST_DEVICE_INLINE
    int64_t Count() const { return count; }

    PBRT_HOST_DEVICE_INLINE
    AccumType RelativeVariance() const {
        if (count < 1 || mean == 0) return {};
        return Variance() / Mean();
    }

  private:
    // S is sum of squares of differences from the current mean:
    // \sum_i^n (x_i - \bar{x}_n)^2
    AccumType mean{}, S{};
    int64_t count = 0;
};

template <typename T, typename Float = double>
class WeightedReservoirSampler {
  public:
    WeightedReservoirSampler() = default;
    PBRT_HOST_DEVICE_INLINE
    WeightedReservoirSampler(uint64_t rngSeed)
        : rng(rngSeed) { }

    PBRT_HOST_DEVICE_INLINE
    void Add(const T &sample, Float weight, int64_t effectiveSamples = 1) {
        weightSum += weight;
        if (nSamplesConsidered == 0)
            reservoir = sample;
        else {
            Float p = weight / weightSum;
            if (rng.Uniform<Float>() < p)
                reservoir = sample;
        }
        nSamplesConsidered += effectiveSamples;
        DCHECK_LT(weightSum, 1e80);
        DCHECK_LT(nSamplesConsidered, ~0ull);
    }

    template <typename F>
    PBRT_HOST_DEVICE_INLINE
    void Add(F func, Float weight, int64_t effectiveSamples = 1) {
        weightSum += weight;
        if (nSamplesConsidered == 0)
            reservoir = func();
        else {
            Float p = weight / weightSum;
            if (rng.Uniform<Float>() < p)
                reservoir = func();
        }
        nSamplesConsidered += effectiveSamples;
        DCHECK_LT(weightSum, 1e80);
        DCHECK_LT(nSamplesConsidered, ~0ull);
    }

    PBRT_HOST_DEVICE_INLINE
    void Copy(const WeightedReservoirSampler &wrs) {
        nSamplesConsidered = wrs.nSamplesConsidered;
        weightSum = wrs.weightSum;
        reservoir = wrs.reservoir;
    }

    PBRT_HOST_DEVICE_INLINE
    void Reset() {
        nSamplesConsidered = 0;
        weightSum = 0;
    }

    PBRT_HOST_DEVICE_INLINE
    void Seed(uint64_t seed) {
        rng.SetSequence(seed);
    }

    PBRT_HOST_DEVICE_INLINE
    void Merge(const WeightedReservoirSampler &wrs) {
        DCHECK_LE(weightSum + wrs.WeightSum(), 1e80);
        DCHECK_GE(nSamplesConsidered + wrs.nSamplesConsidered, nSamplesConsidered);
        if (wrs.HasSample()) {
            Add(wrs.GetSample(), wrs.WeightSum());
            // -1 since Add() added one...
            nSamplesConsidered += wrs.nSamplesConsidered - 1;
        } else
            nSamplesConsidered += wrs.nSamplesConsidered;
    }

    PBRT_HOST_DEVICE_INLINE
    int64_t NSamplesConsidered() const { return nSamplesConsidered; }
    PBRT_HOST_DEVICE_INLINE
    int HasSample() const { return weightSum > 0; }
    PBRT_HOST_DEVICE_INLINE
    Float WeightSum() const { return weightSum; }

    PBRT_HOST_DEVICE_INLINE
    const T &GetSample() const {
        DCHECK(HasSample());
        return reservoir;
    }

    std::string ToString() const {
        return StringPrintf("[ WeightedReservoirSampler rng: %s nSamplesConsidered: %d "
                            "weightSum: %f reservoir: %s ]", rng, nSamplesConsidered,
                            weightSum, reservoir);
   }

  private:
    RNG rng;
    int64_t nSamplesConsidered = 0;
    Float weightSum = 0;
    T reservoir;
};

PBRT_HOST_DEVICE_INLINE
int SampleDiscrete(pstd::span<const Float> weights, Float u, Float *pdf = nullptr,
                   Float *uRemapped = nullptr) {
    if (weights.empty()) {
        if (pdf != nullptr) *pdf = 0;
        return -1;
    }
    Float sum = 0;
    for (Float w : weights) sum += w;
    Float uScaled = u * sum;
    int offset = 0;
    // Need latter condition due to fp roundoff error in the u -= ... term.
    while ((weights[offset] == 0 || uScaled > weights[offset]) &&
           offset < weights.size()) {
        uScaled -= weights[offset];
        ++offset;
    }
    CHECK_RARE(1e-6, offset == weights.size());
    if (offset == weights.size()) offset = weights.size() - 1;

    if (pdf != nullptr) *pdf = weights[offset] / sum;
    if (uRemapped != nullptr)
        *uRemapped = std::min(uScaled / weights[offset], OneMinusEpsilon);
    return offset;
}

PBRT_HOST_DEVICE_INLINE
Float SmoothStepPDF(Float x, Float start, Float end) {
    if (x < start || x > end) return 0;
    DCHECK_LT(start, end);
    return (2 / (end - start)) * SmoothStep(x, start, end);
}

PBRT_HOST_DEVICE_INLINE
Float SampleSmoothStep(Float u, Float start, Float end) {
    DCHECK_LT(start, end);
    auto cdfMinusU = [=](Float x) -> std::pair<Float, Float> {
        Float xp = (x - start) / (end - start);
        return { Pow<3>(xp) * (2 - xp) - u, SmoothStepPDF(x, start, end) };
    };
    return NewtonBisection(start, end, cdfMinusU);
}

PBRT_HOST_DEVICE_INLINE
Float InvertSmoothStepSample(Float x, Float start, Float end) {
    Float xp = (x - start) / (end - start);
    auto CDF = [&](Float x) { return Pow<3>(xp) * (2 - xp); };
    return (CDF(x) - CDF(start)) / (CDF(end) - CDF(start));
}

// Sample ~Lerp(x, a, b). Returned value in [0,1)
PBRT_HOST_DEVICE_INLINE
Float SampleLinear(Float u, Float a, Float b) {
    DCHECK(a >= 0 && b >= 0);
    if (a == b) return u;
    Float x = (a - std::sqrt(Lerp(u, Sqr(a), Sqr(b)))) / (a - b);
    return std::min(x, OneMinusEpsilon);
}

PBRT_HOST_DEVICE_INLINE
Float LinearPDF(Float x, Float a, Float b) {
    DCHECK(a >= 0 && b >= 0);
    if (x < 0 || x > 1) return 0;
    return Lerp(x, a, b) / ((a + b) / 2);
}

PBRT_HOST_DEVICE_INLINE
Float InvertLinearSample(Float x, Float a, Float b) {
    return x * (a * (2 - x) + b * x) / (a + b);
}

// Sample the quadratic function a x^2 + b x + c == 0 over [0,1)
PBRT_HOST_DEVICE
Float SampleQuadratic(Float u, Float a, Float b, Float c, Float *pdf = nullptr);
PBRT_HOST_DEVICE
Float QuadraticPDF(Float x, Float a, Float b, Float c);

PBRT_HOST_DEVICE_INLINE
Float InvertQuadraticSample(Float x, Float a, Float b, Float c) {
    // Just evaluate the CDF...
    Float norm = (a / 3 + b / 2 + c);
    return EvaluatePolynomial(x, 0, c / norm, b / (2 * norm), a / (3 * norm));
}

// Sample the quadratic function going through v0 at x=0, vm at x=0.5, and
// v1 at x=1.
PBRT_HOST_DEVICE_INLINE
Float SampleEquiQuadratic(Float u, Float v0, Float vm, Float v1, Float *pdf = nullptr) {
    // See "fit and sample 1d quadratic.nb"
    // Slightly more math, but cleaner to just compute the coefficients and
    // then reuse SampleQuadratic()?
    pstd::array<Float, 3> c = FitEquiQuadratic(v0, vm, v1);
    return SampleQuadratic(u, c[0], c[1], c[2], pdf);
}

PBRT_HOST_DEVICE_INLINE
Float EquiQuadraticPDF(Float x, Float v0, Float vm, Float v1) {
    // Coefficients of the quadratic going through the three given points
    // at their respective x values.
    pstd::array<Float, 3> c = FitEquiQuadratic(v0, vm, v1);
    return QuadraticPDF(x, c[0], c[1], c[2]);
}

PBRT_HOST_DEVICE_INLINE
Float InvertEquiQuadraticSample(Float x, Float v0, Float vm, Float v1) {
    pstd::array<Float, 3> c = FitEquiQuadratic(v0, vm, v1);
    return InvertQuadraticSample(x, c[0], c[1], c[2]);
}


PBRT_HOST_DEVICE_INLINE
Float SampleBezierCurve(Float u, Float cp0, Float cp1, Float cp2,
                        Float *pdf) {
    // Convert from Bezier to power basis...
    return SampleQuadratic(u, cp0 - 2*cp1 + cp2, -2*cp0 + 2*cp1, cp0, pdf);
}

PBRT_HOST_DEVICE_INLINE
Float BezierCurvePDF(Float x, Float cp0, Float cp1, Float cp2) {
    return QuadraticPDF(x, cp0 - 2*cp1 + cp2, -2*cp0 + 2*cp1, cp0);
}

PBRT_HOST_DEVICE_INLINE
Float InvertBezierCurveSample(Float x, Float cp0, Float cp1, Float cp2) {
    return InvertQuadraticSample(x, cp0 - 2*cp1 + cp2, -2*cp0 + 2*cp1, cp0);
}

PBRT_HOST_DEVICE
Point2f SampleBiquadratic(Point2f u, pstd::array<pstd::array<Float, 3>, 3> w,
                          Float *pdf = nullptr);
PBRT_HOST_DEVICE
Float BiquadraticPDF(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w);
PBRT_HOST_DEVICE
Point2f InvertBiquadraticSample(Point2f p,
                                pstd::array<pstd::array<Float, 3>, 3> w);

PBRT_HOST_DEVICE
Float SampleBezierCurve(Float u, Float cp0, Float cp1, Float cp2, Float *pdf = nullptr);
PBRT_HOST_DEVICE
Float BezierCurvePDF(Float x, Float cp0, Float cp1, Float cp2);
PBRT_HOST_DEVICE
Float InvertBezierCurveSample(Float x);

// w[u][v]
PBRT_HOST_DEVICE
Point2f SampleBezier2D(Point2f u, pstd::array<pstd::array<Float, 3>, 3> w,
                       Float *pdf = nullptr);
PBRT_HOST_DEVICE
Float Bezier2DPDF(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w);
PBRT_HOST_DEVICE
Point2f InvertBezier2DSample(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w);

// v: (0,0), (1,0), (0,1), (1,1)
PBRT_HOST_DEVICE_INLINE
Point2f SampleBilinear(Point2f u, pstd::span<const Float> w) {
    DCHECK_EQ(4, w.size());
    Point2f p;
    // First sample in the v dimension. Compute the endpoints of the line
    // that's the average of the two lines at the edges at u=0 and u=1.
    Float v0 = w[0] + w[1], v1 = w[2] + w[3];
    // Sample along that line.
    p[1] = SampleLinear(u[1], v0, v1);
    // Now in sample in the u direction from the two line end points at the
    // sampled v position.
    p[0] = SampleLinear(u[0], Lerp(p[1], w[0], w[2]), Lerp(p[1], w[1], w[3]));
    return p;
}

// s.t. InvertBilinearSample(SampleBilinear(u, v), v) == u
PBRT_HOST_DEVICE_INLINE
Point2f InvertBilinearSample(Point2f p, pstd::span<const Float> v) {
    // This is just evaluating the CDF at x...
    auto InvertLinear = [](Float x, Float a, Float b) {
        CHECK_RARE(1e-5, !(x >= 0 && x <= 1));
        x = Clamp(x, 0, 1);
        return x * (-a * (x - 2) + b * x) / (a + b);
    };
    return {InvertLinear(p[0], Lerp(p[1], v[0], v[2]), Lerp(p[1], v[1], v[3])),
            InvertLinear(p[1], v[0] + v[1], v[2] + v[3])};
}

PBRT_HOST_DEVICE_INLINE
Float BilinearPDF(Point2f p, pstd::span<const Float> w) {
    DCHECK_EQ(4, w.size());
    if (p.x < 0 || p.x > 1 || p.y < 0 || p.y > 1) return 0;
    if (w[0] + w[1] + w[2] + w[3] == 0) return 1;
    return 4 * Bilerp({p[0], p[1]}, w) / (w[0] + w[1] + w[2] + w[3]);
}

PBRT_HOST_DEVICE
Point2f SampleBilinearGrid(Point2f u, pstd::array<pstd::array<Float, 3>, 3> w,
                           Float *pdf = nullptr);
PBRT_HOST_DEVICE
Point2f InvertBilinearGridSample(Point2f u, pstd::array<pstd::array<Float, 3>, 3> w);
PBRT_HOST_DEVICE
Float BilinearGridPDF(Point2f u, pstd::array<pstd::array<Float, 3>, 3> w);

PBRT_HOST_DEVICE_INLINE
Float SampleTent(Float u, Float radius) {
    if (SampleDiscrete({0.5f, 0.5f}, u, nullptr, &u) == 0)
        return -radius + radius * SampleLinear(u, 0, 1);
    else
        return radius * SampleLinear(u, 1, 0);
}

PBRT_HOST_DEVICE_INLINE
Float TentPDF(Float x, Float radius) {
    if (std::abs(x) >= radius) return 0;
    return 1 / radius - std::abs(x) / Sqr(radius);
}

PBRT_HOST_DEVICE_INLINE
Float InvertTentSample(Float x, Float radius) {
    if (x <= 0)
        return (1 - InvertLinearSample(-x / radius, 1, 0)) / 2;
    else
        return 0.5f + InvertLinearSample(x / radius, 1, 0) / 2;
}

PBRT_HOST_DEVICE_INLINE
Float SampleNormal(Float u, Float mu = 0, Float sigma = 1) {
    // Normal function CDF is 1/2 (1 + erf((x - mu) / (sigma sqrt2))).
    // Set \xi equal to that, solve for x, using erf^-1...
    return mu + Sqrt2 * sigma * ErfInv(2 * u - 1);
}

PBRT_HOST_DEVICE_INLINE
Float NormalPDF(Float x, Float mu = 0, Float sigma = 1) {
    return Gaussian(x, mu, sigma);
}

PBRT_HOST_DEVICE_INLINE
Float InvertNormalSample(Float x, Float mu = 0, Float sigma = 1) {
    return 0.5f * (1 + std::erf((x - mu) / (sigma * std::sqrt(2.f))));
}

PBRT_HOST_DEVICE_INLINE
Point2f SampleTwoNormal(const Point2f &u, Float mu = 0, Float sigma = 1) {
    // Box-Muller transform
    return Point2f(mu + sigma * std::sqrt(-2 * std::log(1 - u[0])) * std::cos(2 * Pi * u[1]),
                   mu + sigma * std::sqrt(-2 * std::log(1 - u[0])) * std::sin(2 * Pi * u[1]));
}

// Sample from e^(-c x), x from 0 to infinity
PBRT_HOST_DEVICE_INLINE
Float SampleExponential(Float u, Float c) {
    return std::log(1 - u) / -c;
}

PBRT_HOST_DEVICE_INLINE
Float ExponentialPDF(Float x, Float c) {
    return c * std::exp(-c * x);
}

PBRT_HOST_DEVICE_INLINE
Float InvertExponentialSample(Float x, Float c) {
    return 1 - std::exp(-c * x);
}

PBRT_HOST_DEVICE_INLINE
Float InvertLogisticSample(Float x, Float s) {
    return 1 / (1 + std::exp(-x / s));
}

PBRT_HOST_DEVICE_INLINE
Float SampleTrimmedLogistic(Float u, Float s, Float a, Float b) {
    DCHECK_LT(a, b);
    u = Lerp(u, InvertLogisticSample(a, s), InvertLogisticSample(b, s));
    Float x = -s * std::log(1 / u - 1);
    DCHECK(!std::isnan(x));
    return Clamp(x, a, b);
}

PBRT_HOST_DEVICE_INLINE
Float TrimmedLogisticPDF(Float x, Float s, Float a, Float b) {
    return Logistic(x, s) / (InvertLogisticSample(b, s) -
                             InvertLogisticSample(a, s));
}

PBRT_HOST_DEVICE_INLINE
Float InvertTrimmedLogisticSample(Float x, Float s, Float a, Float b) {
    DCHECK(a <= x && x <= b);
    return (InvertLogisticSample(x, s) - InvertLogisticSample(a, s)) /
           (InvertLogisticSample(b, s) - InvertLogisticSample(a, s));
}

PBRT_HOST_DEVICE_INLINE
Float SampleCauchy(Float u, Float mu = 0, Float sigma = 1) {
    return mu + sigma * std::tan(Pi * (u - 0.5f));
}

PBRT_HOST_DEVICE_INLINE
Float CauchyPDF(Float x, Float mu = 0, Float sigma = 1) {
    return 1 / (Pi * sigma) * Sqr(sigma) / (Sqr(x - mu) + Sqr(sigma));
}

PBRT_HOST_DEVICE_INLINE
Float InvertCauchySample(Float x, Float mu = 0, Float sigma = 1) {
    return (1 / Pi) * std::atan((x - mu) / sigma) + 0.5f;
}

PBRT_HOST_DEVICE_INLINE
Float SampleTrimmedCauchy(Float u, Float x0, Float x1,
                          Float mu = 0, Float sigma = 1) {
    Float u0 = InvertCauchySample(x0, mu, sigma);
    Float u1 = InvertCauchySample(x1, mu, sigma);
    return SampleCauchy(Lerp(u, u0, u1), mu, sigma);
}

PBRT_HOST_DEVICE_INLINE
Float TrimmedCauchyPDF(Float x, Float x0, Float x1,
                       Float mu = 0, Float sigma = 1) {
    return CauchyPDF(x, mu, sigma) / (InvertCauchySample(x1, mu, sigma) -
                                      InvertCauchySample(x0, mu, sigma));
}

PBRT_HOST_DEVICE_INLINE
Float InvertTrimmedCauchySample(Float x, Float x0, Float x1,
                                Float mu = 0, Float sigma = 1) {
    return (InvertCauchySample(x, mu, sigma) - InvertCauchySample(x0, mu, sigma)) /
           (InvertCauchySample(x1, mu, sigma) - InvertCauchySample(x0, mu, sigma));
}

PBRT_HOST_DEVICE_INLINE
Float SampleXYZMatching(Float u) {
    // "An Improved Technique for Full Spectral Rendering"
    return 538 - std::atanh(Float(0.8569106254698279) -
                            Float(1.8275019724092267) * u) *
        Float(138.88888888888889);
}

PBRT_HOST_DEVICE_INLINE
Float XYZMatchingPDF(Float lambda) {
    if (lambda < 360 || lambda > 830) return 0;

    return Float(0.003939804229326285) /
           Sqr(std::cosh(Float(0.0072) * (lambda - Float(538))));
}

PBRT_HOST_DEVICE_INLINE
Vector3f SampleUniformHemisphere(const Point2f &u) {
    Float z = u[0];
    Float r = SafeSqrt(1 - z * z);
    Float phi = 2 * Pi * u[1];
    return {r * std::cos(phi), r * std::sin(phi), z};
}

PBRT_HOST_DEVICE_INLINE
Float UniformHemispherePDF() { return Inv2Pi; }

PBRT_HOST_DEVICE_INLINE
Point2f InvertUniformHemisphereSample(const Vector3f &v) {
    Float phi = std::atan2(v.y, v.x);
    if (phi < 0) phi += 2 * Pi;
    return Point2f(v.z, phi / (2 * Pi));
}

PBRT_HOST_DEVICE_INLINE
Vector3f SampleUniformSphere(const Point2f &u) {
    Float z = 1 - 2 * u[0];
    Float r = SafeSqrt(1 - z * z);
    Float phi = 2 * Pi * u[1];
    return {r * std::cos(phi), r * std::sin(phi), z};
}

PBRT_HOST_DEVICE_INLINE
Float UniformSpherePDF() { return Inv4Pi; }

PBRT_HOST_DEVICE_INLINE
Point2f InvertUniformSphereSample(const Vector3f &v) {
    Float phi = std::atan2(v.y, v.x);
    if (phi < 0) phi += 2 * Pi;
    return Point2f((1 - v.z) / 2, phi / (2 * Pi));
}

PBRT_HOST_DEVICE_INLINE
Vector3f SampleUniformCone(const Point2f &u, Float cosThetaMax) {
    Float cosTheta = (1 - u[0]) + u[0] * cosThetaMax;
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Float phi = u[1] * 2 * Pi;
    return SphericalDirection(sinTheta, cosTheta, phi);
}

PBRT_HOST_DEVICE_INLINE
Float UniformConePDF(Float cosThetaMax) {
    return 1 / (2 * Pi * (1 - cosThetaMax));
}

PBRT_HOST_DEVICE_INLINE
Point2f InvertUniformConeSample(const Vector3f &v, Float cosThetaMax) {
    Float cosTheta = v.z;
    Float phi = SphericalPhi(v);
    return { (cosTheta - 1) / (cosThetaMax - 1), phi / (2 * Pi) };
}

PBRT_HOST_DEVICE_INLINE
Point2f SampleUniformDiskPolar(const Point2f &u) {
    Float r = std::sqrt(u[0]);
    Float theta = 2 * Pi * u[1];
    return {r * std::cos(theta), r * std::sin(theta)};
}

PBRT_HOST_DEVICE_INLINE
Point2f InvertUniformDiskPolarSample(const Point2f &p) {
    Float phi = std::atan2(p.y, p.x);
    if (phi < 0) phi += 2 * Pi;
    return Point2f(Sqr(p.x) + Sqr(p.y), phi / (2 * Pi));
}

PBRT_HOST_DEVICE_INLINE
Point2f SampleUniformDiskConcentric(const Point2f &u) {
    // Map uniform random numbers to $[-1,1]^2$
    Point2f uOffset = 2.f * u - Vector2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return {0, 0};

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(std::cos(theta), std::sin(theta));
}

PBRT_HOST_DEVICE_INLINE
Point2f InvertUniformDiskConcentricSample(const Point2f &p) {
    Float theta = std::atan2(p.y, p.x); // -pi -> pi
    Float r = std::sqrt(Sqr(p.x) + Sqr(p.y));

    Point2f uo;
    // TODO: can we make this less branchy?
    if (std::abs(theta) < PiOver4 || std::abs(theta) > 3 * PiOver4) {
        uo.x = r = std::copysign(r, p.x);
        if (p.x < 0) {
            if (p.y < 0) {
                uo.y = (Pi + theta) * r / PiOver4;
            } else {
                uo.y = (theta - Pi) * r / PiOver4;
            }
        } else {
            uo.y = (theta * r) / PiOver4;
        }
    } else {
        uo.y = r = std::copysign(r, p.y);
        if (p.y < 0) {
            uo.x = -(PiOver2 + theta) * r / PiOver4;
        } else {
            uo.x = (PiOver2 - theta) * r / PiOver4;
        }
    }

    return { (uo.x + 1) / 2, (uo.y + 1) / 2 };
}

PBRT_HOST_DEVICE_INLINE
Vector3f SampleUniformHemisphereConcentric(const Point2f &u) {
    // Map uniform random numbers to $[-1,1]^2$
    Point2f uOffset = 2.f * u - Vector2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return Vector3f(0, 0, 1);

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }

    return Vector3f(std::cos(theta) * r * std::sqrt(2 - r * r),
                    std::sin(theta) * r * std::sqrt(2 - r * r),
                    1 - r * r);
}

PBRT_HOST_DEVICE_INLINE
pstd::array<Float, 3> SampleUniformTriangle(const Point2f &u) {
    Float b0 = u[0] / 2, b1 = u[1] / 2;
    Float offset = b1 - b0;
    if (offset > 0) b1 += offset;
    else b0 -= offset;
    return { b0, b1, 1 - b0 - b1 };
}

PBRT_HOST_DEVICE_INLINE
Point2f InvertUniformTriangleSample(const pstd::array<Float, 3> &b) {
    if (b[0] > b[1]) {
        // b0 = u[0] - u[1] / 2, b1 = u[1] / 2
        return { b[0] + b[1], 2 * b[1] };
    } else {
        // b1 = u[1] - u[0] / 2, b0 = u[0] / 2
        return { 2 * b[0], b[1] + b[0] };
    }
}


PBRT_HOST_DEVICE
pstd::array<Float, 3> SampleSphericalTriangle(const pstd::array<Point3f, 3> &v,
                                              const Point3f &p, const Point2f &u,
                                              Float *pdf = nullptr);
PBRT_HOST_DEVICE
Point2f InvertSphericalTriangleSample(const pstd::array<Point3f, 3> &v,
                                      const Point3f &p, const Vector3f &w);

PBRT_HOST_DEVICE
Point3f SampleSphericalQuad(const Point3f &p, const Point3f &v00, const Vector3f &ex,
                            const Vector3f &ey, const Point2f &u,
                            Float *pdf = nullptr);
PBRT_HOST_DEVICE
Point2f InvertSphericalQuadSample(const Point3f &pRef, const Point3f &v00, const Vector3f &ex,
                                  const Vector3f &ey, const Point3f &pQuad);

PBRT_HOST_DEVICE
pstd::array<Float, 3> LowDiscrepancySampleTriangle(Float u);

PBRT_HOST_DEVICE_INLINE
Vector3f SampleCosineHemisphere(const Point2f &u) {
    Point2f d = SampleUniformDiskConcentric(u);
    Float z = SafeSqrt(1 - d.x * d.x - d.y * d.y);
    return Vector3f(d.x, d.y, z);
}

PBRT_HOST_DEVICE_INLINE
Float CosineHemispherePDF(Float cosTheta) { return cosTheta * InvPi; }

PBRT_HOST_DEVICE_INLINE
Point2f InvertCosineHemisphereSample(const Vector3f &v) {
    return InvertUniformDiskConcentricSample({v.x, v.y});
}

PBRT_HOST_DEVICE_INLINE
Vector3f SampleTrowbridgeReitz(Float alpha_x, Float alpha_y, const Point2f &u) {
    Float cosTheta = 0, phi = (2 * Pi) * u[1];
    if (alpha_x == alpha_y) {
        Float tanTheta2 = alpha_x * alpha_x * u[0] / (1 - u[0]);
        cosTheta = 1 / std::sqrt(1 + tanTheta2);
    } else {
        phi =
            std::atan(alpha_y / alpha_x * std::tan(2 * Pi * u[1] + .5f * Pi));
        if (u[1] > .5f) phi += Pi;
        Float sinPhi = std::sin(phi), cosPhi = std::cos(phi);
        Float alpha2 = 1 / (Sqr(cosPhi / alpha_x) + Sqr(sinPhi / alpha_y));
        Float tanTheta2 = alpha2 * u[0] / (1 - u[0]);
        cosTheta = 1 / std::sqrt(1 + tanTheta2);
    }
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    return SphericalDirection(sinTheta, cosTheta, phi);
}

PBRT_HOST_DEVICE_INLINE
Vector3f SampleTrowbridgeReitzVisibleArea(const Vector3f &w, Float alpha_x,
                                          Float alpha_y, const Point2f &u) {
    // Section 3.2: transforming the view direction to the hemisphere configuration
    Vector3f wh = Normalize(Vector3f(alpha_x * w.x, alpha_y * w.y, w.z));

    // Section 4.1: orthonormal basis. Can't use CoordinateSystem() since
    // T1 has to be in the tangent plane w.r.t. (0,0,1).
    Vector3f T1 = (wh.z < 0.99999f) ? Normalize(Cross(Vector3f(0, 0, 1), wh)) : Vector3f(1, 0, 0);
    Vector3f T2 = Cross(wh, T1);

    // Section 4.2: parameterization of the projected area
    Float r = std::sqrt(u[0]);
    Float phi = 2 * Pi * u[1];
    Float t1 = r * std::cos(phi), t2 = r * std::sin(phi);
    Float s = 0.5f * (1 + wh.z);
    t2 = (1 - s) * std::sqrt(1 - t1*t1) + s*t2;

    // Section 4.3: reprojection onto hemisphere
    Vector3f nh = t1*T1 + t2*T2 + std::sqrt(std::max<Float>(0, 1 - t1*t1 - t2*t2)) * wh;

    // Section 3.4: transforming the normal back to the ellipsoid configuration
    CHECK_RARE(1e-6, nh.z == 0);
    return Normalize(Vector3f(alpha_x * nh.x, alpha_y * nh.y, std::max<Float>(1e-6f, nh.z)));
}

PBRT_HOST_DEVICE
Vector3f SampleHenyeyGreenstein(const Vector3f &wo, Float g, const Point2f &u,
                                Float *pdf = nullptr);

pstd::vector<Float> Sample1DFunction(std::function<Float(Float)> f, int nSteps,
                                     int nSamples, Float min = 0, Float max = 1,
                                     Norm norm = Norm::LInfinity,
                                     Allocator alloc = {});

Array2D<Float> Sample2DFunction(std::function<Float(Float, Float)> f, int nu,
                                int nv, int nSamples,
                                Bounds2f domain = {Point2f(0, 0),
                                                   Point2f(1, 1)},
                                Norm norm = Norm::LInfinity,
                                Allocator alloc = {});

PBRT_HOST_DEVICE
Float SampleCatmullRom(pstd::span<const Float> nodes, pstd::span<const Float> f,
                       pstd::span<const Float> cdf, Float sample,
                       Float *fval = nullptr, Float *pdf = nullptr);
PBRT_HOST_DEVICE
Float SampleCatmullRom2D(pstd::span<const Float> nodes1, pstd::span<const Float> nodes2,
                         pstd::span<const Float> values, pstd::span<const Float> cdf,
                         Float alpha, Float sample, Float *fval = nullptr,
                         Float *pdf = nullptr);

namespace detail {

template <typename Iterator>
class IndexingIterator {
  public:
    template <typename Generator>
    PBRT_HOST_DEVICE
    IndexingIterator(int i, int n, const Generator *)
        : i(i), n(n) { }

    PBRT_HOST_DEVICE
    bool operator==(const Iterator &it) const { return i == it.i; }
    PBRT_HOST_DEVICE
    bool operator!=(const Iterator &it) const { return !(*this == it); }
    PBRT_HOST_DEVICE
    Iterator &operator++() {
        ++i;
        return (Iterator &)*this;
    }
    PBRT_HOST_DEVICE
    Iterator operator++(int) const {
        Iterator it = *this;
        return ++it;
    }

  protected:
    int i, n;
};

template <typename Generator, typename Iterator>
class IndexingGenerator {
  public:
    PBRT_HOST_DEVICE
    IndexingGenerator(int n) : n(n) {}
    PBRT_HOST_DEVICE
    Iterator begin() const { return Iterator(0, n, (const Generator *)this); }
    PBRT_HOST_DEVICE
    Iterator end() const { return Iterator(n, n, (const Generator *)this); }

  protected:
    int n;
};

class Uniform1DIter;
class Uniform2DIter;
class Uniform3DIter;
class Hammersley2DIter;
class Hammersley3DIter;
class Stratified1DIter;
class Stratified2DIter;
class Stratified3DIter;
template <typename Iterator> class RNGIterator;

template <typename Generator, typename Iterator>
class RNGGenerator : public IndexingGenerator<Generator, Iterator> {
  public:
    PBRT_HOST_DEVICE
    RNGGenerator(int n, uint64_t sequenceIndex = 0)
        : IndexingGenerator<Generator, Iterator>(n),
          sequenceIndex(sequenceIndex) {}

  protected:
    friend RNGIterator<Iterator>;
    uint64_t sequenceIndex;
};

template <typename Iterator>
class RNGIterator : public IndexingIterator<Iterator> {
  public:
    template <typename Generator>
    PBRT_HOST_DEVICE
    RNGIterator(int i, int n, const RNGGenerator<Generator, Iterator> *generator)
        : IndexingIterator<Iterator>(i, n, generator),
          rng(generator->sequenceIndex) {}

  protected:
    RNG rng;
};

}  // namespace detail

class Uniform1D
    : public detail::RNGGenerator<Uniform1D, detail::Uniform1DIter> {
  public:
    using detail::RNGGenerator<Uniform1D, detail::Uniform1DIter>::RNGGenerator;
};

class Uniform2D
    : public detail::RNGGenerator<Uniform2D, detail::Uniform2DIter> {
  public:
    using detail::RNGGenerator<Uniform2D, detail::Uniform2DIter>::RNGGenerator;
};

class Uniform3D
    : public detail::RNGGenerator<Uniform3D, detail::Uniform3DIter> {
  public:
    using detail::RNGGenerator<Uniform3D, detail::Uniform3DIter>::RNGGenerator;
};

class Hammersley2D
    : public detail::IndexingGenerator<Hammersley2D, detail::Hammersley2DIter> {
  public:
    using detail::IndexingGenerator<
        Hammersley2D, detail::Hammersley2DIter>::IndexingGenerator;
};

class Hammersley3D
    : public detail::IndexingGenerator<Hammersley3D, detail::Hammersley3DIter> {
  public:
    using detail::IndexingGenerator<
        Hammersley3D, detail::Hammersley3DIter>::IndexingGenerator;
};

class Stratified1D
    : public detail::RNGGenerator<Stratified1D, detail::Stratified1DIter> {
  public:
    using detail::RNGGenerator<Stratified1D,
                               detail::Stratified1DIter>::RNGGenerator;
};

class Stratified2D
    : public detail::RNGGenerator<Stratified2D, detail::Stratified2DIter> {
  public:
    PBRT_HOST_DEVICE
    Stratified2D(int nx, int ny, uint64_t sequenceIndex = 0)
        : detail::RNGGenerator<Stratified2D, detail::Stratified2DIter>(
              nx * ny, sequenceIndex),
          nx(nx),
          ny(ny) {}

  private:
    friend detail::Stratified2DIter;
    int nx, ny;
};

class Stratified3D
    : public detail::RNGGenerator<Stratified3D, detail::Stratified3DIter> {
  public:
    PBRT_HOST_DEVICE
    Stratified3D(int nx, int ny, int nz, uint64_t sequenceIndex = 0)
        : detail::RNGGenerator<Stratified3D, detail::Stratified3DIter>(
              nx * ny * nz, sequenceIndex),
          nx(nx),
          ny(ny),
          nz(nz) {}

  private:
    friend detail::Stratified3DIter;
    int nx, ny, nz;
};

namespace detail {

class Uniform1DIter : public RNGIterator<Uniform1DIter> {
  public:
    using RNGIterator<Uniform1DIter>::RNGIterator;
    PBRT_HOST_DEVICE
    Float operator*() { return rng.Uniform<Float>(); }
};

class Uniform2DIter : public RNGIterator<Uniform2DIter> {
  public:
    using RNGIterator<Uniform2DIter>::RNGIterator;
    PBRT_HOST_DEVICE
    Point2f operator*() { return {rng.Uniform<Float>(), rng.Uniform<Float>()}; }
};

class Uniform3DIter : public RNGIterator<Uniform3DIter> {
  public:
    using RNGIterator<Uniform3DIter>::RNGIterator;
    PBRT_HOST_DEVICE
    Point3f operator*() {
        return {rng.Uniform<Float>(), rng.Uniform<Float>(),
                rng.Uniform<Float>()};
    }
};

class Stratified1DIter : public RNGIterator<Stratified1DIter> {
  public:
    using RNGIterator<Stratified1DIter>::RNGIterator;
    PBRT_HOST_DEVICE
    Float operator*() { return (i + rng.Uniform<Float>()) / n; }
};

class Stratified2DIter : public RNGIterator<Stratified2DIter> {
  public:
    PBRT_HOST_DEVICE
    Stratified2DIter(int i, int n, const Stratified2D *generator)
        : RNGIterator<Stratified2DIter>(i, n, generator),
          nx(generator->nx),
          ny(generator->ny) {}

    PBRT_HOST_DEVICE
    Point2f operator*() {
        int ix = i % nx, iy = i / nx;
        return {(ix + rng.Uniform<Float>()) / nx,
                (iy + rng.Uniform<Float>()) / ny};
    }

  private:
    int nx, ny;
};

class Stratified3DIter : public RNGIterator<Stratified3DIter> {
  public:
    PBRT_HOST_DEVICE
    Stratified3DIter(int i, int n, const Stratified3D *generator)
        : RNGIterator<Stratified3DIter>(i, n, generator),
          nx(generator->nx), ny(generator->ny), nz(generator->nz) { }

    PBRT_HOST_DEVICE
    Point3f operator*() {
        int ix = i % nx;
        int iy = (i / nx) % ny;
        int iz = i / (nx * ny);
        return {(ix + rng.Uniform<Float>()) / nx,
                (iy + rng.Uniform<Float>()) / ny,
                (iz + rng.Uniform<Float>()) / nz};
    }

  private:
    int nx, ny, nz;
};

class Hammersley2DIter : public IndexingIterator<Hammersley2DIter> {
  public:
    using IndexingIterator<Hammersley2DIter>::IndexingIterator;
    PBRT_HOST_DEVICE
    Point2f operator*() { return {Float(i) / Float(n), RadicalInverse(0, i)}; }
};

class Hammersley3DIter : public IndexingIterator<Hammersley3DIter> {
  public:
    using IndexingIterator<Hammersley3DIter>::IndexingIterator;
    PBRT_HOST_DEVICE
    Point3f operator*() {
        return {Float(i) / Float(n), RadicalInverse(0, i),
                RadicalInverse(1, i)};
    }
};

}  // namespace detail


class Distribution1D {
 public:
    // Distribution1D Public Methods
    Distribution1D() = default;
    Distribution1D(Allocator alloc)
        : func(alloc), cdf(alloc), offsetLUT(alloc) { }
    Distribution1D(pstd::span<const Float> f, Float min, Float max,
                   Allocator alloc = {})
        : func(f.begin(), f.end(), alloc), cdf(f.size() + 1, alloc),
          offsetLUT(alloc), min(min), max(max) {
        CHECK_GT(max, min);
        // Compute integral of step function at $x_i$
        cdf[0] = 0;
        size_t n = f.size();
        for (size_t i = 1; i < n + 1; ++i) {
            CHECK_GE(func[i - 1], 0);
            cdf[i] = cdf[i - 1] + func[i - 1] * (max - min) / n;
        }

        // Transform step function integral into CDF
        funcInt = cdf[n];
        if (funcInt == 0) {
            for (size_t i = 1; i < n + 1; ++i) cdf[i] = Float(i) / Float(n);
        } else {
            for (size_t i = 1; i < n + 1; ++i) cdf[i] /= funcInt;
        }

        // From a not-thorough test, this seems to give about a 20%
        // performance benefit with infinite area light sampling.  OTOH,
        // there's a lot of trig in there, so that may be a bigger speedup
        // for the actual distribution sampling...
//#define U_TO_OFFSET_LUT
#ifdef U_TO_OFFSET_LUT
        offsetLUT = pstd::vector<int>(f.size() / 2, alloc);
        for (size_t i = 0; i < offsetLUT.size(); ++i) {
            Float u = Float(i) / offsetLUT.size();
            int offset = FindInterval((int)cdf.size(),
                                      [&](int index) { return cdf[index] <= u; });
            offsetLUT[i] = offset;
        }
#endif // U_TO_OFFSET_LUT
    }
    Distribution1D(pstd::span<const Float> f, Allocator alloc = {})
        : Distribution1D(f, 0., 1., alloc) { }

    PBRT_HOST_DEVICE_INLINE
    size_t size() const { return func.size(); }

    PBRT_HOST_DEVICE_INLINE
    Float SampleContinuous(Float u, Float *pdf = nullptr, int *off = nullptr) const {
        // Find surrounding CDF segments and _offset_
        int offset = GetOffset(u);
        if (off) *off = offset;
        // Compute offset along CDF segment
        Float du = u - cdf[offset];
        if (cdf[offset + 1] - cdf[offset] > 0)
            du /= cdf[offset + 1] - cdf[offset];
        DCHECK(!std::isnan(du));

        // Compute PDF for sampled offset
        if (pdf != nullptr)
            *pdf = (funcInt > 0) ? func[offset] / funcInt : 0;

        // Return $x$ corresponding to sample
        return Lerp((offset + du) / size(), min, max);
    }

    PBRT_HOST_DEVICE_INLINE
    int SampleDiscrete(Float u, Float *pdf = nullptr,
                       Float *uRemapped = nullptr) const {
        // Find surrounding CDF segments and _offset_
        int offset = GetOffset(u);
        if (pdf != nullptr)
            *pdf = (funcInt > 0) ? func[offset] / (funcInt * size()) : 0;
        if (uRemapped != nullptr)
            *uRemapped = (u - cdf[offset]) / (cdf[offset + 1] - cdf[offset]);
        if (uRemapped) CHECK(*uRemapped >= 0.f && *uRemapped <= 1.f);
        return offset;
    }

    PBRT_HOST_DEVICE_INLINE
    Float DiscretePDF(int index) const {
        CHECK(index >= 0 && index < size());
        return funcInt > 0 ? func[index] / (funcInt * size()) : 0;
    }

    // Given a point in the domain [min,max], return the sample [0,1] that
    // maps to the point.
    PBRT_HOST_DEVICE
    pstd::optional<Float> Inverse(Float v) const {
        if (v < min || v > max) return {};
        Float c = (v - min) / (max - min) * func.size();
        int offset = Clamp(int(c), 0, func.size() - 1);
        DCHECK(offset >= 0 && offset + 1 < cdf.size());
        if (func[offset] == 0) return {};
        Float delta = c - offset;
        return Lerp(delta, cdf[offset], cdf[offset + 1]);
    }

    PBRT_HOST_DEVICE
    size_t BytesUsed() const {
        return (func.capacity() + cdf.capacity()) * sizeof(Float);
    }

    static void TestCompareDistributions(const Distribution1D &da,
                                         const Distribution1D &db,
                                         Float eps = 1e-5);

    PBRT_HOST_DEVICE_INLINE
    int GetOffset(Float u) const {
#ifdef U_TO_OFFSET_LUT
        size_t offsetIndex = size_t(u * offsetLUT.size());
        if (offsetIndex + 1 >= offsetLUT.size())
            return FindInterval((int)cdf.size(),
                                [&](int index) { return cdf[index] <= u; });

        int start = offsetLUT[offsetIndex], end = offsetLUT[offsetIndex + 1] + 1;

        // TODO: why do we need another +1 here given the one added to end?
        int offset = FindInterval(1 + end - start,
                                  [&](int index) {
                                      DCHECK_LT(start + index, cdf.size());
                                      return cdf[start + index] <= u;
                                  }) + start;

        DCHECK_LT(offset + 1, cdf.size());
        DCHECK_GE(u, cdf[offset]);
        DCHECK_LE(u, cdf[offset + 1]);
        return offset;
#else
        return FindInterval((int)cdf.size(),
                            [&](int index) { return cdf[index] <= u; });
#endif
    }

    std::string ToString() const { return "TODO DISTRIB1D TO STRING"; }

    // Distribution1D Public Data
    pstd::vector<Float> func, cdf;
    pstd::vector<int> offsetLUT;
    Float min, max;
    Float funcInt = 0;
};

class Distribution2D {
  public:
    // Distribution2D Public Methods
    Distribution2D() = default;
    Distribution2D(Allocator alloc)
        : pConditionalY(alloc), pMarginal(alloc) { }
    Distribution2D(pstd::span<const Float> data, int nx, int ny,
                   Bounds2f domain, Allocator alloc = {});
    Distribution2D(pstd::span<const Float> data, int nx, int ny,
                   Allocator alloc = {})
        : Distribution2D(data, nx, ny, Bounds2f(Point2f(0, 0), Point2f(1, 1)), alloc) { }
    explicit Distribution2D(const Array2D<Float> &data, Allocator alloc = {})
        : Distribution2D(pstd::span<const Float>(data), data.xSize(), data.ySize(),
                         alloc) { }
    Distribution2D(const Array2D<Float> &data, Bounds2f domain,
                   Allocator alloc = {})
        : Distribution2D(pstd::span<const Float>(data), data.xSize(), data.ySize(),
                         domain, alloc) { }

    PBRT_HOST_DEVICE
    Point2f SampleContinuous(const Point2f &u, Float *pdf = nullptr) const {
        Float pdfs[2];
        int y;
        Float d1 = pMarginal.SampleContinuous(u[1], &pdfs[1], &y);
        Float d0 = pConditionalY[y].SampleContinuous(u[0], &pdfs[0]);
        if (pdf != nullptr) *pdf = pdfs[0] * pdfs[1];
        return Point2f(d0, d1);
    }

    PBRT_HOST_DEVICE
    Point2i SampleDiscrete(const Point2f &u, Float *pdf = nullptr,
                           Point2f *uRemapped = nullptr) const {
        Float pdfs[2];
        int d1 = pMarginal.SampleDiscrete(u[1], &pdfs[1],
                                          uRemapped ? &uRemapped->y : nullptr);
        int d0 = pConditionalY[d1].SampleDiscrete(u[0], &pdfs[0],
                                                  uRemapped ? &uRemapped->x : nullptr);
        if (pdf != nullptr) *pdf = pdfs[0] * pdfs[1];
        return {d0, d1};
    }

    PBRT_HOST_DEVICE
    Float ContinuousPDF(const Point2f &pr) const {
        Point2f p = Point2f(domain.Offset(pr));
        int ix = Clamp(int(p[0] * pConditionalY[0].size()), 0,
                       pConditionalY[0].size() - 1);
        int iy = Clamp(int(p[1] * pMarginal.size()), 0, pMarginal.size() - 1);
        return pConditionalY[iy].func[ix] / pMarginal.funcInt;
    }

    PBRT_HOST_DEVICE
    Float DiscretePDF(const Point2i &p) const {
        return pConditionalY[p[1]].func[p[0]] /
            (pMarginal.funcInt * pMarginal.size() * pConditionalY[p[1]].size());
    }

    PBRT_HOST_DEVICE
    pstd::optional<Point2f> Inverse(const Point2f &p) const {
        pstd::optional<Float> mInv = pMarginal.Inverse(p[1]);
        if (!mInv) return {};
        Float p1o = (p[1] - domain.pMin[1]) / (domain.pMax[1] - domain.pMin[1]);
        if (p1o < 0 || p1o > 1) return {};
        int offset = Clamp(p1o * pConditionalY.size(), 0, pConditionalY.size() - 1);
        pstd::optional<Float> cInv = pConditionalY[offset].Inverse(p[0]);
        if (!cInv) return {};
        return Point2f(*cInv, *mInv);
    }

    PBRT_HOST_DEVICE
    size_t BytesUsed() const {
        return pConditionalY.size() * (pConditionalY[0].BytesUsed() +
                                       sizeof(pConditionalY[0])) +
            pMarginal.BytesUsed();
    }

    PBRT_HOST_DEVICE_INLINE
    Bounds2f Domain() const { return domain; }

    PBRT_HOST_DEVICE_INLINE
    Point2i Resolution() const {
        return {int(pConditionalY[0].size()), int(pMarginal.size())};
    }

    static void TestCompareDistributions(const Distribution2D &da,
                                         const Distribution2D &db,
                                         Float eps = 1e-5);

 private:
    Bounds2f domain;
    pstd::vector<Distribution1D> pConditionalY;
    Distribution1D pMarginal;
};

class LinearDistribution1D {
  public:
    LinearDistribution1D(pstd::span<const Float> v, Allocator alloc = {})
        : values(v.begin(), v.end(), alloc) {
        // Note takes at least two values to specify f(0) and f(1)
        CHECK_GE(values.size(), 2);

        std::vector<Float> p(values.size() - 1);
        for (int i = 0; i < values.size() - 1; ++i)
            p[i] = values[i] + values[i + 1];
        distrib = Distribution1D(p, alloc);
    }

    PBRT_HOST_DEVICE
    Float Sample(Float u, Float *pdf = nullptr) {
        Float uRemapped;
        int index = distrib.SampleDiscrete(u, pdf, &uRemapped);
        CHECK_LT(index + 1, values.size());
        Float x = SampleLinear(uRemapped, values[index], values[index + 1]);
        if (pdf != nullptr)
            *pdf *= (values.size() - 1) *
                LinearPDF(x, values[index], values[index + 1]);
        return Float(index + x) / (values.size() - 1);
    }

    PBRT_HOST_DEVICE
    Float PDF(Float x) {
        int index = std::min<int>(x * (values.size() - 1), values.size() - 1);
        Float pdf = distrib.DiscretePDF(index);
        // Fractional part
        Float fx = x * (values.size() - 1) - index;
        return pdf * (values.size() - 1) *
               LinearPDF(fx, values[index], values[index + 1]);
    }

  private:
    Distribution1D distrib;
    pstd::vector<Float> values;
};

class LinearDistribution2D {
  public:
    LinearDistribution2D(pstd::span<const Float> v, int nx, int ny,
                         Allocator alloc = {})
        : nx(nx), ny(ny), values(v.begin(), v.end(), alloc) {
        CHECK_GE(nx, 2);
        CHECK_GE(ny, 2);
        CHECK_EQ(values.size(), nx * ny);

        std::vector<Float> p((nx - 1) * (ny - 1));
        for (int y = 0; y < ny - 1; ++y)
            for (int x = 0; x < nx - 1; ++x)
                p[y * (nx - 1) + x] = (values[y * nx + x] +
                                       values[y * nx + x + 1] +
                                       values[(y + 1) * nx + x] +
                                       values[(y + 1) * nx + x + 1]);
        distrib = Distribution2D(p, nx - 1, ny - 1, alloc);
    }

    PBRT_HOST_DEVICE
    Point2f Sample(const Point2f &u, Float *pdf = nullptr) {
        Point2f uRemapped;
        Point2i p = distrib.SampleDiscrete(u, pdf, &uRemapped);
        pstd::array<Float, 4> v = { values[p[1] * nx + p[0]],
                                    values[p[1] * nx + p[0] + 1],
                                    values[(p[1] + 1) * nx + p[0]],
                                    values[(p[1] + 1) * nx + p[0] + 1] };
        Point2f x = SampleBilinear(uRemapped, v);
        if (pdf != nullptr)
            *pdf *= (nx - 1) * (ny - 1) * BilinearPDF(x, v);
        return Point2f((p[0] + x[0]) / (nx - 1), (p[1] + x[1]) / (ny - 1));
    }

    PBRT_HOST_DEVICE
    Float PDF(const Point2f &x) {
        Point2i p(std::min<int>(x[0] * (nx - 1), nx - 1),
                  std::min<int>(x[1] * (ny - 1), ny - 1));
        Float pdf = distrib.DiscretePDF(p);
        // Fractional part
        Point2f fp(x[0] * (nx - 1) - p[0], x[1] * (ny - 1) - p[1]);
        pstd::array<Float, 4> v = { values[p[1] * nx + p[0]],
                                   values[p[1] * nx + p[0] + 1],
                                   values[(p[1] + 1) * nx + p[0]],
                                   values[(p[1] + 1) * nx + p[0] + 1] };
        return pdf * (nx - 1) * (ny - 1) * BilinearPDF(fp, v);
    }

  private:
    Distribution2D distrib;
    int nx, ny;
    pstd::vector<Float> values;
};

class DynamicDistribution1D {
  public:
    DynamicDistribution1D(int count)
        // Just allocate the extra nodes and let them hold zero forever.
        // it's fine.
        : nValidLeaves(count), nodes(2 * RoundUpPow2(count) - 1, Float(0)) {
        // The tree is laid out breadth-first in |nodes|.
        firstLeafOffset = nodes.size() - RoundUpPow2(count);
    }
    DynamicDistribution1D(pstd::span<const Float> v)
        : DynamicDistribution1D(v.size()) {
        for (size_t i = 0; i < v.size(); ++i)
            (*this)[i] = v[i];
        UpdateAll();
    }

    Float &operator[](int index) {
        DCHECK(index >= 0 && index + firstLeafOffset < nodes.size());
        return nodes[index + firstLeafOffset];
    }
    Float operator[](int index) const {
        DCHECK(index >= 0 && index + firstLeafOffset < nodes.size());
        return nodes[index + firstLeafOffset];
    }
    size_t size() const { return nValidLeaves; }

    // Update probabilities after a single value has been modified. O(log n).
    void Update(int index) {
        DCHECK(index >= 0 && index + firstLeafOffset < nodes.size());
        index += firstLeafOffset;
        while (index != 0) {
            int parentIndex = (index - 1) / 2;
            nodes[parentIndex] =
                (nodes[2 * parentIndex + 1] + nodes[2 * parentIndex + 2]);
            index = parentIndex;
        }
    }

    // Update all probabilities. O(n).
    void UpdateAll();

    std::string ToString() const {
        std::string ret;
        int newline = 0, nextCount = 2;
        for (size_t i = 0; i < nodes.size(); ++i) {
            ret += std::to_string(nodes[i]) + ' ';
            if (i == newline) {
                ret += '\n';
                newline = i + nextCount;
                nextCount *= 2;
            }
        }
        return ret;
    }

    int SampleDiscrete(Float u, Float *pdf = nullptr) const {
        int index = 0;
        if (pdf != nullptr)
            *pdf = 1;
        while (index < firstLeafOffset) {
            Float p[2] = { nodes[2 * index + 1], nodes[2 * index + 2] };
            Float q = p[0] / (p[0] + p[1]);
            if (u < q) {
                if (pdf != nullptr)
                    *pdf *= q;
                u = std::min(u / q, OneMinusEpsilon);
                index = 2 * index + 1;
            } else {
                if (pdf != nullptr)
                    *pdf *= 1 - q;
                u = std::min((u - q) / (1 - q), OneMinusEpsilon);
                index = 2 * index + 2;
            }
        }
        return index - firstLeafOffset;
    }

    Float PDF(int index) const {
        Float pdf = 1;
        index += firstLeafOffset;
        while (index != 0) {
            int parentIndex = (index - 1) / 2;
            Float psum = (nodes[2 * parentIndex + 1] +
                          nodes[2 * parentIndex + 2]);
            pdf *= nodes[index]  / psum;
            index = parentIndex;
        }
        return pdf;
    }

  private:
    int nValidLeaves;
    std::vector<Float> nodes;
    size_t firstLeafOffset;
};

// Both Distribution2D and Hierarchical2DWarp work for the warp here
#if 0
template <typename W>
Image WarpedStrataVisualization(const W &warp, int xs = 16, int ys = 16) {
    Image im(PixelFormat::Half, {warp.Resolution().x / 2, warp.Resolution().y / 2}, { "R", "G", "B" });
    for (int y = 0; y < im.Resolution().y; ++y) {
        for (int x = 0; x < im.Resolution().x; ++x) {
            Point2f target = warp.Domain().Lerp({(x + .5f) / im.Resolution().x,
                                                 (y + .5f) / im.Resolution().y});
            if (warp.ContinuousPDF(target) == 0) continue;

            pstd::optional<Point2f> u = warp.Inverse(target);
            if (!u.has_value()) {
#if 0
                LOG(WARNING) << "No value at target " << target << ", though cont pdf = " <<
                    tabdist.ContinuousPDF(target);
#endif
                continue;
            }

#if 1
            int tile = int(u->x * xs) + xs * int(u->y * ys);
            Float rgb[3] = { RadicalInverse(0, tile), RadicalInverse(1, tile),
                             RadicalInverse(2, tile) };
            im.SetChannels({x, int(y)}, {rgb[0], rgb[1], rgb[2]});
#else
            Float gray = ((int(u->x * xs) + int(u->y * ys)) & 1) ? 0.8 : 0.2;
            im.SetChannel({x, int(y)}, 0, gray);
#endif
        }
    }
    return im;
}
#endif


class Hierarchical2DWarp {
  public:
    // Take optional Bounds2f domain, use it.
    Hierarchical2DWarp() = default;
    Hierarchical2DWarp(pstd::span<const Float> values, int nx, int ny,
                       const Bounds2f &domain, Allocator alloc = {});
    explicit Hierarchical2DWarp(const Array2D<Float> &values, Allocator alloc = {})
        : Hierarchical2DWarp(values, values.xSize(), values.ySize(), alloc) { }
    Hierarchical2DWarp(const Array2D<Float> &values, const Bounds2f &domain,
                       Allocator alloc = {})
        : Hierarchical2DWarp(values, values.xSize(), values.ySize(), domain, alloc) { }
    Hierarchical2DWarp(pstd::span<const Float> values, int nx, int ny,
                       Allocator alloc = {})
        : Hierarchical2DWarp(values, nx, ny, Bounds2f(Point2f(0,0), Point2f(1,1)),
                             alloc) { }

    PBRT_HOST_DEVICE
    Point2i SampleDiscrete(Point2f u, Float *pdf = nullptr) const;
    PBRT_HOST_DEVICE
    Float DiscretePDF(Point2i p) const;

    PBRT_HOST_DEVICE
    Point2f SampleContinuous(Point2f u, Float *pdf = nullptr) const;
    PBRT_HOST_DEVICE
    Float ContinuousPDF(const Point2f &p) const;

    PBRT_HOST_DEVICE
    pstd::optional<Point2f> Inverse(const Point2f &p) const;

    std::string ToString() const;

    PBRT_HOST_DEVICE
    Bounds2f Domain() const { return domain; }
    PBRT_HOST_DEVICE
    Point2i Resolution() const { return Resolution(levels.size() - 1); }

  private:
    PBRT_HOST_DEVICE
    Point2i Resolution(int level) const {
        DCHECK(level >= 0 && level < levels.size());
        return {levels[level].xSize(), levels[level].ySize()};
    }

    PBRT_HOST_DEVICE
    Float Lookup(int level, int x, int y) const {
        DCHECK(level >= 0 && level < levels.size());
        DCHECK(x >= 0 && y >= 0);

        if (x >= levels[level].xSize() || y >= levels[level].ySize())
            return 0;
        return levels[level](x, y);
    }

    Bounds2f domain;
    pstd::vector<Array2D<Float>> levels;
};

// *****************************************************************************
// Marginal-conditional warp
// *****************************************************************************

/**
 * \brief Implements a marginal sample warping scheme for 2D distributions
 * with linear interpolation and an optional dependence on additional parameters
 *
 * This class takes a rectangular floating point array as input and constructs
 * internal data structures to efficiently map uniform variates from the unit
 * square <tt>[0, 1]^2</tt> to a function on <tt>[0, 1]^2</tt> that linearly
 * interpolates the input array.
 *
 * The mapping is constructed via the inversion method, which is applied to
 * a marginal distribution over rows, followed by a conditional distribution
 * over columns.
 *
 * The implementation also supports <em>conditional distributions</em>, i.e. 2D
 * distributions that depend on an arbitrary number of parameters (indicated
 * via the \c Dimension template parameter).
 *
 * In this case, the input array should have dimensions <tt>N0 x N1 x ... x Nn
 * x res[1] x res[0]</tt> (where the last dimension is contiguous in memory),
 * and the <tt>param_res</tt> should be set to <tt>{ N0, N1, ..., Nn }</tt>,
 * and <tt>param_values</tt> should contain the parameter values where the
 * distribution is discretized. Linear interpolation is used when sampling or
 * evaluating the distribution for in-between parameter values.
 */
template <size_t Dimension = 0>
class Marginal2D {
  private:
    using FloatStorage = pstd::vector<float>;

#if !defined(_MSC_VER) && !defined(__CUDACC__)
    static constexpr size_t ArraySize = Dimension;
#else
    static constexpr size_t ArraySize = (Dimension != 0) ? Dimension : 1;
#endif

  public:
    Marginal2D(Allocator alloc)
        : m_param_values(alloc), m_data(alloc), m_marginal_cdf(alloc),
          m_conditional_cdf(alloc) {
        for (int i = 0; i < ArraySize; ++i)
            m_param_values.emplace_back(alloc);
    }

    /**
     * Construct a marginal sample warping scheme for floating point
     * data of resolution \c size.
     *
     * \c param_res and \c param_values are only needed for conditional
     * distributions (see the text describing the Marginal2D class).
     *
     * If \c normalize is set to \c false, the implementation will not
     * re-scale the distribution so that it integrates to \c 1. It can
     * still be sampled (proportionally), but returned density values
     * will reflect the unnormalized values.
     *
     * If \c build_cdf is set to \c false, the implementation will not
     * construct the cdf needed for sample warping, which saves memory in case
     * this functionality is not needed (e.g. if only the interpolation in \c
     * eval() is used).
     */
    Marginal2D(Allocator alloc, const float *data, int xSize, int ySize,
               pstd::array<int, Dimension> param_res = {},
               pstd::array<const float *, Dimension> param_values = {},
               bool normalize = true, bool build_cdf = true)
        : m_size(xSize, ySize),
          m_patch_size(1.f / (xSize - 1), 1.f / (ySize - 1)),
          m_inv_patch_size(m_size - Vector2i(1, 1)),
          m_param_values(alloc), m_data(alloc), m_marginal_cdf(alloc),
          m_conditional_cdf(alloc) {
        if (build_cdf && !normalize)
            LOG_FATAL("Marginal2D: build_cdf implies normalize=true");

        /* Keep track of the dependence on additional parameters (optional) */
        uint32_t slices = 1;
        for (int i = 0; i < ArraySize; ++i)
            m_param_values.emplace_back(alloc);
        for (int i = (int)Dimension - 1; i >= 0; --i) {
            if (param_res[i] < 1)
                LOG_FATAL("Marginal2D(): parameter resolution must be >= 1!");

            m_param_size[i] = param_res[i];
            m_param_values[i] = FloatStorage(param_res[i]);
            memcpy(m_param_values[i].data(), param_values[i],
                   sizeof(float) * param_res[i]);
            m_param_strides[i] = param_res[i] > 1 ? slices : 0;
            slices *= m_param_size[i];
        }

        uint32_t n_values = xSize * ySize;

        m_data = FloatStorage(slices * n_values);

        if (build_cdf) {
            m_marginal_cdf = FloatStorage(slices * m_size.y);
            m_conditional_cdf = FloatStorage(slices * n_values);

            float *marginal_cdf = m_marginal_cdf.data(),
                  *conditional_cdf = m_conditional_cdf.data(),
                  *data_out = m_data.data();

            for (uint32_t slice = 0; slice < slices; ++slice) {
                /* Construct conditional CDF */
                for (uint32_t y = 0; y < m_size.y; ++y) {
                    double sum = 0.0;
                    size_t i = y * xSize;
                    conditional_cdf[i] = 0.f;
                    for (uint32_t x = 0; x < m_size.x - 1; ++x, ++i) {
                        sum += .5 * ((double)data[i] + (double)data[i + 1]);
                        conditional_cdf[i + 1] = (float)sum;
                    }
                }

                /* Construct marginal CDF */
                marginal_cdf[0] = 0.f;
                double sum = 0.0;
                for (uint32_t y = 0; y < m_size.y - 1; ++y) {
                    sum += .5 * ((double)conditional_cdf[(y + 1) * xSize - 1] +
                                 (double)conditional_cdf[(y + 2) * xSize - 1]);
                    marginal_cdf[y + 1] = (float)sum;
                }

                /* Normalize CDFs and PDF (if requested) */
                float normalization = 1.f / marginal_cdf[m_size.y - 1];
                for (size_t i = 0; i < n_values; ++i)
                    conditional_cdf[i] *= normalization;
                for (size_t i = 0; i < m_size.y; ++i)
                    marginal_cdf[i] *= normalization;
                for (size_t i = 0; i < n_values; ++i)
                    data_out[i] = data[i] * normalization;

                marginal_cdf += m_size.y;
                conditional_cdf += n_values;
                data_out += n_values;
                data += n_values;
            }
        } else {
            float *data_out = m_data.data();

            for (uint32_t slice = 0; slice < slices; ++slice) {
                float normalization = 1.f / HProd(m_inv_patch_size);
                if (normalize) {
                    double sum = 0.0;
                    for (uint32_t y = 0; y < m_size.y - 1; ++y) {
                        size_t i = y * xSize;
                        for (uint32_t x = 0; x < m_size.x - 1; ++x, ++i) {
                            float v00 = data[i], v10 = data[i + 1],
                                  v01 = data[i + xSize],
                                  v11 = data[i + 1 + xSize],
                                  avg = .25f * (v00 + v10 + v01 + v11);
                            sum += (double)avg;
                        }
                    }
                    normalization = float(1.0 / sum);
                }

                for (uint32_t k = 0; k < n_values; ++k)
                    data_out[k] = data[k] * normalization;

                data += n_values;
                data_out += n_values;
            }
        }
    }

    struct Sample {
        Vector2f p;
        float pdf;
    };

    /**
     * \brief Given a uniformly distributed 2D sample, draw a sample from the
     * distribution (parameterized by \c param if applicable)
     *
     * Returns the warped sample and associated probability density.
     */
    PBRT_HOST_DEVICE
    Sample sample(Vector2f sample, const Float *param = nullptr) const {
        /* Avoid degeneracies at the extrema */
        sample[0] = Clamp(sample[0], 1 - OneMinusEpsilon, OneMinusEpsilon);
        sample[1] = Clamp(sample[1], 1 - OneMinusEpsilon, OneMinusEpsilon);

        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize];
        uint32_t slice_offset = 0u;
        for (size_t dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index =
                FindInterval(m_param_size[dim], [&](uint32_t idx) {
                    return m_param_values[dim].data()[idx] <= param[dim];
                });

            float p0 = m_param_values[dim][param_index],
                  p1 = m_param_values[dim][param_index + 1];

            param_weight[2 * dim + 1] =
                Clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Sample the row first */
        uint32_t offset = 0;
        if (Dimension != 0) offset = slice_offset * m_size.y;

        auto fetch_marginal = [&](uint32_t idx) -> float {
            return lookup<Dimension>(m_marginal_cdf.data(), offset + idx,
                                     m_size.y, param_weight);
        };

        uint32_t row = FindInterval(m_size.y, [&](uint32_t idx) {
            return fetch_marginal(idx) < sample.y;
        });

        sample.y -= fetch_marginal(row);

        uint32_t slice_size = HProd(m_size);
        offset = row * m_size.x;
        if (Dimension != 0) offset += slice_offset * slice_size;

        float r0 = lookup<Dimension>(m_conditional_cdf.data(),
                                     offset + m_size.x - 1, slice_size,
                                     param_weight),
              r1 = lookup<Dimension>(m_conditional_cdf.data(),
                                     offset + (m_size.x * 2 - 1), slice_size,
                                     param_weight);

        bool is_const = std::abs(r0 - r1) < 1e-4f * (r0 + r1);
        sample.y = is_const
                       ? (2.f * sample.y)
                       : (r0 - SafeSqrt(r0 * r0 - 2.f * sample.y * (r0 - r1)));
        sample.y /= is_const ? (r0 + r1) : (r0 - r1);

        /* Sample the column next */
        sample.x *= (1.f - sample.y) * r0 + sample.y * r1;

        auto fetch_conditional = [&](uint32_t idx) -> float {
            float v0 = lookup<Dimension>(m_conditional_cdf.data(), offset + idx,
                                         slice_size, param_weight),
                  v1 =
                      lookup<Dimension>(m_conditional_cdf.data() + m_size.x,
                                        offset + idx, slice_size, param_weight);

            return (1.f - sample.y) * v0 + sample.y * v1;
        };

        uint32_t col = FindInterval(m_size.x, [&](uint32_t idx) {
            return fetch_conditional(idx) < sample.x;
        });

        sample.x -= fetch_conditional(col);

        offset += col;

        float v00 = lookup<Dimension>(m_data.data(), offset, slice_size,
                                      param_weight),
              v10 = lookup<Dimension>(m_data.data() + 1, offset, slice_size,
                                      param_weight),
              v01 = lookup<Dimension>(m_data.data() + m_size.x, offset,
                                      slice_size, param_weight),
              v11 = lookup<Dimension>(m_data.data() + m_size.x + 1, offset,
                                      slice_size, param_weight),
              c0 = std::fma((1.f - sample.y), v00, sample.y * v01),
              c1 = std::fma((1.f - sample.y), v10, sample.y * v11);

        is_const = std::abs(c0 - c1) < 1e-4f * (c0 + c1);
        sample.x = is_const
                       ? (2.f * sample.x)
                       : (c0 - SafeSqrt(c0 * c0 - 2.f * sample.x * (c0 - c1)));
        sample.x /= is_const ? (c0 + c1) : (c0 - c1);

        return {
            (Vector2f(col, row) + sample) * m_patch_size,
            ((1.f - sample.x) * c0 + sample.x * c1) * HProd(m_inv_patch_size)};
    }

    /// Inverse of the mapping implemented in \c sample()
    PBRT_HOST_DEVICE
    Sample invert(Vector2f sample, const Float *param = nullptr) const {
        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize];
        uint32_t slice_offset = 0u;
        for (size_t dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index =
                FindInterval(m_param_size[dim], [&](uint32_t idx) {
                    return m_param_values[dim][idx] <= param[dim];
                });

            float p0 = m_param_values[dim][param_index],
                  p1 = m_param_values[dim][param_index + 1];

            param_weight[2 * dim + 1] =
                Clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Fetch values at corners of bilinear patch */
        sample *= m_inv_patch_size;
        Vector2i pos = Min(Vector2i(sample), m_size - Vector2i(2, 2));
        sample -= Vector2f(pos);

        uint32_t offset = pos.x + pos.y * m_size.x;
        uint32_t slice_size = HProd(m_size);
        if (Dimension != 0) offset += slice_offset * slice_size;

        /* Invert the X component */
        float v00 = lookup<Dimension>(m_data.data(), offset, slice_size,
                                      param_weight),
              v10 = lookup<Dimension>(m_data.data() + 1, offset, slice_size,
                                      param_weight),
              v01 = lookup<Dimension>(m_data.data() + m_size.x, offset,
                                      slice_size, param_weight),
              v11 = lookup<Dimension>(m_data.data() + m_size.x + 1, offset,
                                      slice_size, param_weight);

        Vector2f w1 = sample, w0 = Vector2f(1, 1) - w1;

        float c0 = std::fma(w0.y, v00, w1.y * v01),
              c1 = std::fma(w0.y, v10, w1.y * v11),
              pdf = std::fma(w0.x, c0, w1.x * c1);

        sample.x *= c0 + .5f * sample.x * (c1 - c0);

        float v0 = lookup<Dimension>(m_conditional_cdf.data(), offset,
                                     slice_size, param_weight),
              v1 = lookup<Dimension>(m_conditional_cdf.data() + m_size.x,
                                     offset, slice_size, param_weight);

        sample.x += (1.f - sample.y) * v0 + sample.y * v1;

        offset = pos.y * m_size.x;
        if (Dimension != 0) offset += slice_offset * slice_size;

        float r0 = lookup<Dimension>(m_conditional_cdf.data(),
                                     offset + m_size.x - 1, slice_size,
                                     param_weight),
              r1 = lookup<Dimension>(m_conditional_cdf.data(),
                                     offset + (m_size.x * 2 - 1), slice_size,
                                     param_weight);

        sample.x /= (1.f - sample.y) * r0 + sample.y * r1;

        /* Invert the Y component */
        sample.y *= r0 + .5f * sample.y * (r1 - r0);

        offset = pos.y;
        if (Dimension != 0) offset += slice_offset * m_size.y;

        sample.y += lookup<Dimension>(m_marginal_cdf.data(), offset, m_size.y,
                                      param_weight);

        return {sample, pdf * HProd(m_inv_patch_size)};
    }

    /**
     * \brief Evaluate the density at position \c pos. The distribution is
     * parameterized by \c param if applicable.
     */
    PBRT_HOST_DEVICE
    float eval(Vector2f pos, const Float *param = nullptr) const {
        /* Look up parameter-related indices and weights (if Dimension != 0) */
        float param_weight[2 * ArraySize];
        uint32_t slice_offset = 0u;

        for (size_t dim = 0; dim < Dimension; ++dim) {
            if (m_param_size[dim] == 1) {
                param_weight[2 * dim] = 1.f;
                param_weight[2 * dim + 1] = 0.f;
                continue;
            }

            uint32_t param_index =
                FindInterval(m_param_size[dim], [&](uint32_t idx) {
                    return m_param_values[dim][idx] <= param[dim];
                });

            float p0 = m_param_values[dim][param_index],
                  p1 = m_param_values[dim][param_index + 1];

            param_weight[2 * dim + 1] =
                Clamp((param[dim] - p0) / (p1 - p0), 0.f, 1.f);
            param_weight[2 * dim] = 1.f - param_weight[2 * dim + 1];
            slice_offset += m_param_strides[dim] * param_index;
        }

        /* Compute linear interpolation weights */
        pos *= m_inv_patch_size;
        Vector2i offset = Min(Vector2i(pos), m_size - Vector2i(2, 2));

        Vector2f w1 = pos - Vector2f(Vector2i(offset)),
                 w0 = Vector2f(1, 1) - w1;

        uint32_t index = offset.x + offset.y * m_size.x;

        uint32_t size = HProd(m_size);
        if (Dimension != 0) index += slice_offset * size;

        float v00 = lookup<Dimension>(m_data.data(), index, size, param_weight),
              v10 = lookup<Dimension>(m_data.data() + 1, index, size,
                                      param_weight),
              v01 = lookup<Dimension>(m_data.data() + m_size.x, index, size,
                                      param_weight),
              v11 = lookup<Dimension>(m_data.data() + m_size.x + 1, index, size,
                                      param_weight);

        return std::fma(w0.y, std::fma(w0.x, v00, w1.x * v10),
                        w1.y * std::fma(w0.x, v01, w1.x * v11)) *
               HProd(m_inv_patch_size);
    }

    PBRT_HOST_DEVICE
    size_t BytesUsed() const {
        size_t sum = 4 * (m_data.capacity() + m_marginal_cdf.capacity() +
                          m_conditional_cdf.capacity());
        for (int i = 0; i < ArraySize; ++i)
            sum += m_param_values[i].capacity();
        return sum;
    }

  private:
    template <size_t Dim, std::enable_if_t<Dim != 0, int> = 0>
    PBRT_HOST_DEVICE
    float lookup(const float *data, uint32_t i0, uint32_t size,
                 const float *param_weight) const {
        uint32_t i1 = i0 + m_param_strides[Dim - 1] * size;

        float w0 = param_weight[2 * Dim - 2], w1 = param_weight[2 * Dim - 1],
              v0 = lookup<Dim - 1>(data, i0, size, param_weight),
              v1 = lookup<Dim - 1>(data, i1, size, param_weight);

        return std::fma(v0, w0, v1 * w1);
    }

    template <size_t Dim, std::enable_if_t<Dim == 0, int> = 0>
    PBRT_HOST_DEVICE
    float lookup(const float *data, uint32_t index, uint32_t,
                 const float *) const {
        return data[index];
    }

    /// Resolution of the discretized density function
    Vector2i m_size;

    /// Size of a bilinear patch in the unit square
    Vector2f m_patch_size, m_inv_patch_size;

    /// Resolution of each parameter (optional)
    uint32_t m_param_size[ArraySize];

    /// Stride per parameter in units of sizeof(float)
    uint32_t m_param_strides[ArraySize];

    /// Discretization of each parameter domain
    pstd::vector<FloatStorage> m_param_values;

    /// Density values
    FloatStorage m_data;

    /// Marginal and conditional PDFs
    FloatStorage m_marginal_cdf;
    FloatStorage m_conditional_cdf;
};

}  // namespace pbrt

#endif  // PBRT_SAMPLING_H
