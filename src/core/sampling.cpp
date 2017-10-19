
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


// core/sampling.cpp*
#include "sampling.h"

#include "lowdiscrepancy.h"
#include "util/mathutil.h"
#include "util/stats.h"
#include "util/transform.h"
#include <tests/gtest/gtest.h>

#include <numeric>


namespace pbrt {

// Sampling Function Definitions
void StratifiedSample1D(absl::Span<Float> samples, RNG &rng,
                        bool jitter) {
    Float invNSamples = (Float)1 / samples.size();
    for (size_t i = 0; i < samples.size(); ++i) {
        Float delta = jitter ? rng.UniformFloat() : 0.5f;
        samples[i] = std::min((i + delta) * invNSamples, OneMinusEpsilon);
    }
}

void StratifiedSample2D(absl::Span<Point2f> samp, int nx, int ny,
                        RNG &rng, bool jitter) {
    CHECK_EQ(samp.size(), (size_t)nx * (size_t)ny);
    Float dx = (Float)1 / nx, dy = (Float)1 / ny;
    int offset = 0;
    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x, ++offset) {
            Float jx = jitter ? rng.UniformFloat() : 0.5f;
            Float jy = jitter ? rng.UniformFloat() : 0.5f;
            samp[offset].x = std::min((x + jx) * dx, OneMinusEpsilon);
            samp[offset].y = std::min((y + jy) * dy, OneMinusEpsilon);
        }
}

void LatinHypercube(absl::Span<Float> samples, int nDim, RNG &rng) {
    // Generate LHS samples along diagonal
    DCHECK_EQ(0, samples.size() % nDim);
    int nSamples = samples.size() / nDim;
    Float invNSamples = (Float)1 / nSamples;
    for (size_t i = 0; i < nSamples; ++i)
        for (int j = 0; j < nDim; ++j) {
            Float sj = (i + (rng.UniformFloat())) * invNSamples;
            samples[nDim * i + j] = std::min(sj, OneMinusEpsilon);
        }

    // Permute LHS samples in each dimension
    for (int i = 0; i < nDim; ++i) {
        for (size_t j = 0; j < nSamples; ++j) {
            size_t other = j + rng.UniformUInt32(nSamples - j);
            std::swap(samples[nDim * j + i], samples[nDim * other + i]);
        }
    }
}

Point2f RejectionSampleDisk(RNG &rng) {
    Point2f p;
    do {
        p.x = 1 - 2 * rng.UniformFloat();
        p.y = 1 - 2 * rng.UniformFloat();
    } while (p.x * p.x + p.y * p.y > 1);
    return p;
}

Vector3f UniformSampleHemisphere(const Point2f &u) {
    Float z = u[0];
    Float r = SafeSqrt(1 - z * z);
    Float phi = 2 * Pi * u[1];
    return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
}

Float UniformHemispherePdf() { return Inv2Pi; }

Vector3f UniformSampleSphere(const Point2f &u) {
    Float z = 1 - 2 * u[0];
    Float r = SafeSqrt(1 - z * z);
    Float phi = 2 * Pi * u[1];
    return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
}

Float UniformSpherePdf() { return Inv4Pi; }

Point2f UniformSampleDisk(const Point2f &u) {
    Float r = std::sqrt(u[0]);
    Float theta = 2 * Pi * u[1];
    return Point2f(r * std::cos(theta), r * std::sin(theta));
}

Point2f ConcentricSampleDisk(const Point2f &u) {
    // Map uniform random numbers to $[-1,1]^2$
    Point2f uOffset = 2.f * u - Vector2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return Point2f(0, 0);

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

Float UniformConePdf(Float cosThetaMax) {
    return 1 / (2 * Pi * (1 - cosThetaMax));
}

Vector3f UniformSampleCone(const Point2f &u, Float cosThetaMax) {
    Float cosTheta = (1 - u[0]) + u[0] * cosThetaMax;
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Float phi = u[1] * 2 * Pi;
    return Vector3f(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta,
                    cosTheta);
}

Vector3f UniformSampleCone(const Point2f &u, Float cosThetaMax,
                           const Vector3f &x, const Vector3f &y,
                           const Vector3f &z) {
    Float cosTheta = Lerp(u[0], cosThetaMax, 1.f);
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Float phi = u[1] * 2 * Pi;
    return std::cos(phi) * sinTheta * x + std::sin(phi) * sinTheta * y +
           cosTheta * z;
}

std::array<Float, 3> UniformSampleTriangle(const Point2f &u) {
    Float su0 = std::sqrt(u[0]);
    std::array<Float, 3> b;
    b[0] = 1 - su0;
    b[1] = u[1] * su0;
    b[2] = 1 - b[0] - b[1];
    return b;
}

void Distribution1D::TestCompareDistributions(const Distribution1D &da,
                                              const Distribution1D &db,
                                              Float eps) {
    ASSERT_EQ(da.func.size(), db.func.size());
    ASSERT_EQ(da.cdf.size(), db.cdf.size());
    for (size_t i = 0; i < da.func.size(); ++i) {
        Float pdfa = da.func[i] / da.funcInt, pdfb = db.func[i] / db.funcInt;
        Float err = std::abs(pdfa - pdfb) / ((pdfa + pdfb) / 2);
        EXPECT_LT(err, eps) << pdfa << " - " << pdfb;
    }
}

Distribution2D::Distribution2D(absl::Span<const Float> func, int nu, int nv) {
    CHECK_EQ(func.size(), (size_t)nu * (size_t)nv);
    pConditionalV.reserve(nv);
    for (int v = 0; v < nv; ++v)
        // Compute conditional sampling distribution for $\tilde{v}$
        pConditionalV.push_back(Distribution1D(func.subspan(v * nu, nu)));
    // Compute marginal sampling distribution $p[\tilde{v}]$
    std::vector<Float> marginalFunc;
    marginalFunc.reserve(nv);
    for (int v = 0; v < nv; ++v)
        marginalFunc.push_back(pConditionalV[v].funcInt);
    pMarginal = Distribution1D(marginalFunc);
}


Distribution2D Distribution2D::SampleFunction(std::function<Float(Point2f)> f,
                                              int nu, int nv,
                                              int nSamples, Norm norm) {
    std::vector<Point2f> samples(nSamples);
    for (int i = 0; i < nSamples; ++i)
        samples[i] = Point2f(RadicalInverse(0, i), RadicalInverse(1, i));
    if (norm == Norm::LInfinity) {
        // Check the corners, too.
        samples.push_back(Point2f(0, 1));
        samples.push_back(Point2f(1, 0));
        samples.push_back(Point2f(1, 1));
    }

    std::vector<Float> values(nu * nv, Float(0));
    for (int v = 0; v < nv; ++v) {
        for (int u = 0; u < nu; ++u) {
            double accum = 0;
            for (size_t i = 0; i < samples.size(); ++i) {
                Point2f p((u + samples[i][0]) / nu, (v + samples[i][1]) / nv);
                Float fuv = std::abs(f(p));
                switch (norm) {
                case Norm::L1:
                    accum += fuv;
                    break;
                case Norm::L2:
                    accum += fuv * fuv;
                    break;
                case Norm::LInfinity:
                    accum = std::max<double>(accum, fuv);
                    break;
                default:
                    LOG(FATAL) << "Unhandled norm";
                }
            }
            // There's actually no need for the divide by nSamples, since
            // these are normalzed into a PDF anyway.
            if (norm == Norm::L2) accum = std::sqrt(accum);
            values[u + v * nu] = accum;
        }
    }

    return Distribution2D(values, nu, nv);
}

void Distribution2D::TestCompareDistributions(const Distribution2D &da,
                                              const Distribution2D &db,
                                              Float eps) {
    Distribution1D::TestCompareDistributions(da.pMarginal, db.pMarginal, eps);

    ASSERT_EQ(da.pConditionalV.size(), db.pConditionalV.size());
    for (size_t i = 0; i < da.pConditionalV.size(); ++i)
        Distribution1D::TestCompareDistributions(da.pConditionalV[i],
                                                 db.pConditionalV[i], eps);
}

int SampleDiscrete(absl::Span<const Float> weights, Float u, Float *pdf,
                   Float *uRemapped) {
    if (weights.empty()) {
        *pdf = 0;
        return -1;
    }
    Float sum = std::accumulate(weights.begin(), weights.end(), Float(0));
    Float uScaled = u * sum;
    int offset = 0;
    // Need latter condition due to fp roundoff error in the u -= ... term.
    while (uScaled > weights[offset] && offset < weights.size()) {
        uScaled -= weights[offset];
        ++offset;
    }
    *pdf = weights[offset] / sum;
    if (uRemapped) *uRemapped = std::min(uScaled / weights[offset],
                                         OneMinusEpsilon);
    return offset;
}

// TODO: work on fp robustness.
//
// https://www.solidangle.com/research/egsr2013_spherical_rectangle.pdf
// discusses the issue, but seems to just do a bunch of clamping.
//
// See also http://graphics.pixar.com/library/StatFrameworkForImportance/paper.pdf
// for some discussion of this.
std::array<Float, 3> SphericalSampleTriangle(const std::array<Point3f, 3> &v,
                                             const Point3f &p, const Point2f &u,
                                             Float *pdf) {
    using Vector3d = Vector3<double>;
    Vector3d a(v[0] - p), b(v[1] - p), c(v[2] - p);
    CHECK_GT(LengthSquared(a), 0);
    CHECK_GT(LengthSquared(b), 0);
    CHECK_GT(LengthSquared(c), 0);
    a = Normalize(a);
    b = Normalize(b);
    c = Normalize(c);

    // TODO: have a shared snippet that goes from here to computing
    // alpha/beta/gamma, use it also in Triangle::SolidAngle().
    Vector3d axb = Cross(a, b), bxc = Cross(b, c), cxa = Cross(c, a);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 || LengthSquared(cxa) == 0) {
        *pdf = 0;
        return {};
    }
    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxa = Normalize(cxa);

    // See comment in Triangle::SolidAngle() for ordering...
    double alpha = AngleBetween(cxa, -axb);
    double beta = AngleBetween(axb, -bxc);
    double gamma = AngleBetween(bxc, -cxa);

    // Spherical area of the triangle.
    double A = alpha + beta + gamma - Pi;
    if (A <= 0) {
        *pdf = 0;
        return {};
    }
    *pdf = 1 / A;

    // Uniformly sample triangle area
    double Ap = u[0] * A;

    // Compute sin beta' and cos beta' for the point along the edge b
    // corresponding to the area sampled, A'.

    // TODO? Permute vertices so we always sample along the longest edge?
    double sinPhi = std::sin(Ap - alpha);
    // This doesn't always work... Can we compute cos and then do
    // sine this way?
//CO    double cosPhi = std::sqrt(std::max(double(0), 1 - sinPhi * sinPhi));
    double cosPhi = std::cos(Ap - alpha);

    double cosAlpha = std::cos(alpha);
    double uu = cosPhi - cosAlpha;
//CO    double sinAlpha = SafeSqrt(1 - cosAlpha * cosAlpha);
    double sinAlpha = std::sin(alpha);

    double vv = sinPhi + sinAlpha * Dot(a, b) /* cos c */;
    double cosBetap = (((vv * cosPhi - uu * sinPhi) * cosAlpha - vv) /
                      ((vv * sinPhi + uu * cosPhi) * sinAlpha));
    CHECK_RARE(1e-6, cosBetap < -1.00001 || cosBetap > 1.00001);
    // Happens if the triangle basically covers the entire hemisphere.
    // We currently depend on calling code to detect this case, which
    // is sort of ugly/unfortunate.
    CHECK(!isNaN(cosBetap));
    cosBetap = Clamp(cosBetap, -1, 1);
    double sinBetap = SafeSqrt(1 - cosBetap * cosBetap);

    // Gram-Schmidt
    auto GS = [](const Vector3d &a, const Vector3d &b) {
        return Normalize(a - Dot(a, b) * b);
    };

    // Compute c', the point along the arc between b' and a.
    Vector3d cp = cosBetap * a + sinBetap * GS(c, a);

    double cosTheta = 1 - u[1] * (1 - Dot(cp, b));
    double sinTheta = SafeSqrt(1 - cosTheta * cosTheta);

    // Compute direction on the sphere.
    Vector3d w = cosTheta * b + sinTheta * GS(cp, b);

    // Compute barycentrics. Subset of Moller-Trumbore intersection test.
    Vector3d e1(v[1] - v[0]), e2(v[2] - v[0]);
    Vector3d s1 = Cross(w, e2);
    double divisor = Dot(s1, e1);

    CHECK_RARE(1e-6, divisor == 0);
    if (divisor == 0) {
        // This happens with triangles that cover (nearly) the whole
        // hemisphere.
        LOG(ERROR) << "Divisor 0. A = " << A;
        return {1.f/3.f, 1.f/3.f, 1.f/3.f};
    }
    double invDivisor = 1 / divisor;

    // Compute first barycentric coordinate
    Vector3d s(p - v[0]);
    double b1 = Dot(s, s1) * invDivisor;

    // Compute second barycentric coordinate
    Vector3d s2 = Cross(s, e1);
    double b2 = Dot(w, s2) * invDivisor;

    // We get goofy barycentrics for very small and very large (w.r.t. the sphere) triangles. Again,
    // we expect the caller to not use this in that case.
    CHECK_RARE(1e-6, b1 < -1e-4 || b1 > 1.0001 || b2 < -1e-4 || b2 > 1.0001);
    b1 = Clamp(b1, 0, 1);
    b2 = Clamp(b2, 0, 1);
    if (b1 + b2 > 1) {
        b1 /= b1 + b2;
        b2 /= b1 + b2;
    }

    return {Float(1 - b1 - b2), Float(b1), Float(b2)};
}

Float SampleSmoothstep(Float u, Float start, Float end) {
    CHECK_LT(start, end);
    auto cdfMinusU = [=](Float x) -> std::pair<Float, Float> {
        Float xp = (x - start) / (end - start);
        return { Pow<3>(xp) * (2 - xp) - u, SmoothstepPdf(x, start, end) };
    };
    // Bump the endpoint down so that the CDF function is negative there
    // and so that we bracket the root going into NewtonBisection().
    return NewtonBisection(start - Float(.0001) * (end - start), end, cdfMinusU);
}

Float SmoothstepPdf(Float x, Float start, Float end) {
    CHECK_LT(start, end);
    return (2 / (end - start)) * Smoothstep(x, start, end);
}

}  // namespace pbrt
