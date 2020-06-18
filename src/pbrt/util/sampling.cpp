// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// Include this first, since it has a method named Infinity(), and we
// #define that for __CUDA_ARCH__ builds.
#include <gtest/gtest.h>

#include <pbrt/util/sampling.h>

#include <pbrt/util/check.h>
#include <pbrt/util/float.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/scattering.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <set>

namespace pbrt {

// Sampling Function Definitions

Float SampleQuadratic(Float u, Float a, Float b, Float c, Float *pdf) {
    // Make sure it doesn't go negative over [0,1]...
    if (c < 0) {  // x = 0
        DCHECK_RARE(1e-5, c < -1e-5);
        c = -c;
    }
    if (a + b + c < 0) {  // x == 1
        DCHECK_RARE(1e-5, a + b + c < -1e-5);
        c += -2 * (a + b + c);
    }
    // inflection point
    if (a != 0) {
        Float t = -b / (2 * a);
        if (t >= 0 && t <= 1) {
            Float v = EvaluatePolynomial(t, c, b, a);
            DCHECK_RARE(1e-5, v < -1e-5);
            if (v < 0)
                // TODO: do what here?
                c += -2 * v;
        }
    }

    DCHECK(u >= 0 && u <= 1);
    Float factor = 6 / (2 * a + 3 * b + 6 * c);

    // CDF(x) = factor * (a x^3 / 3 + b x^2 / 2 + c x)
    Float x = NewtonBisection(
        0., 1.,
        [&](Float x) -> std::pair<Float, Float> {
            return std::make_pair(
                EvaluatePolynomial(x, -u, factor * c, factor * b / 2, factor * a / 3),
                factor * EvaluatePolynomial(x, c, b, a));
        },
        1e-4f, 1e-4f);
    x = std::min(x, OneMinusEpsilon);

    if (pdf != nullptr)
        // Float integ = a / 3 + b / 2 + c;  // integral over [0,1] Factor is
        // 1/that
        *pdf = EvaluatePolynomial(x, c, b, a) * factor;

    return x;
}

Float QuadraticPDF(Float x, Float a, Float b, Float c) {
    // Make sure it doesn't go negative over [0,1]...
    if (c < 0) {  // x = 0
        CHECK_RARE(1e-5, c < -1e-5);
        c = -c;
    }
    if (a + b + c < 0) {  // x == 1
        CHECK_RARE(1e-5, a + b + c < -1e-5);
        c += -2 * (a + b + c);
    }
    // inflection point
    if (a != 0) {
        Float t = -b / (2 * a);
        if (t >= 0 && t <= 1) {
            Float v = EvaluatePolynomial(t, c, b, a);
            CHECK_RARE(1e-5, v < -1e-5);
            if (v < 0)
                // TODO: do what here?
                c += -2 * v;
        }
    }

    Float integ = a / 3 + b / 2 + c;  // integral over [0,1]
    return EvaluatePolynomial(x, c, b, a) / integ;
}

Point2f SampleBiquadratic(Point2f su, pstd::array<pstd::array<Float, 3>, 3> w,
                          Float *pdf) {
    // "sample biquadratic.nb"

    // Sample marginal polynomial in v
    Float vc = w[0][0] + 4 * w[1][0] + w[2][0];
    Float vb = -3 * w[0][0] + 4 * w[0][1] - w[0][2] - 12 * w[1][0] + 16 * w[1][1] -
               4 * w[1][2] - 3 * w[2][0] + 4 * w[2][1] - w[2][2];
    Float va = 2 * w[0][0] + 2 * (-2 * w[0][1] + w[0][2] + 4 * w[1][0] - 8 * w[1][1] +
                                  4 * w[1][2] + w[2][0] - 2 * w[2][1] + w[2][2]);
    Float vPDF;
    Float v = SampleQuadratic(su[1], va, vb, vc, pdf != nullptr ? &vPDF : nullptr);

    // Conditional polynomial in u, given v
    Float uc = (1 - 3 * v + 2 * v * v) * w[0][0] + 4 * v * w[0][1] - 4 * v * v * w[0][1] -
               v * w[0][2] + 2 * v * v * w[0][2];
    Float ub = (-3 * (1 - 3 * v + 2 * v * v) * w[0][0] - 12 * v * w[0][1] +
                12 * v * v * w[0][1] + 3 * v * w[0][2] - 6 * v * v * w[0][2] +
                4 * w[1][0] - 12 * v * w[1][0] + 8 * v * v * w[1][0] + 16 * v * w[1][1] -
                16 * v * v * w[1][1] - 4 * v * w[1][2] + 8 * v * v * w[1][2] - w[2][0] +
                3 * v * w[2][0] - 2 * v * v * w[2][0] - 4 * v * w[2][1] +
                4 * v * v * w[2][1] + v * w[2][2] - 2 * v * v * w[2][2]);
    Float ua =
        (2 * (1 - 3 * v + 2 * v * v) * w[0][0] + 8 * v * w[0][1] - 8 * v * v * w[0][1] -
         2 * v * w[0][2] + 4 * v * v * w[0][2] - 4 * w[1][0] + 12 * v * w[1][0] -
         8 * v * v * w[1][0] - 16 * v * w[1][1] + 16 * v * v * w[1][1] + 4 * v * w[1][2] -
         8 * v * v * w[1][2] + 2 * w[2][0] - 6 * v * w[2][0] + 4 * v * v * w[2][0] +
         8 * v * w[2][1] - 8 * v * v * w[2][1] - 2 * v * w[2][2] + 4 * v * v * w[2][2]);
    Float uPDF;
    Float u = SampleQuadratic(su[0], ua, ub, uc, pdf != nullptr ? &uPDF : nullptr);

    if (pdf != nullptr)
        *pdf = uPDF * vPDF;

    return {u, v};
}

Float BiquadraticPDF(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w) {
    // Marginal quadratic PDF coefficients
    Float vc = w[0][0] + 4 * w[1][0] + w[2][0];
    Float vb = -3 * w[0][0] + 4 * w[0][1] - w[0][2] - 12 * w[1][0] + 16 * w[1][1] -
               4 * w[1][2] - 3 * w[2][0] + 4 * w[2][1] - w[2][2];
    Float va = 2 * w[0][0] + 2 * (-2 * w[0][1] + w[0][2] + 4 * w[1][0] - 8 * w[1][1] +
                                  4 * w[1][2] + w[2][0] - 2 * w[2][1] + w[2][2]);

    // Conditional quadratic PDF in u, given v
    Float v = p[1];
    Float uc = (1 - 3 * v + 2 * v * v) * w[0][0] + 4 * v * w[0][1] - 4 * v * v * w[0][1] -
               v * w[0][2] + 2 * v * v * w[0][2];
    Float ub = (-3 * (1 - 3 * v + 2 * v * v) * w[0][0] - 12 * v * w[0][1] +
                12 * v * v * w[0][1] + 3 * v * w[0][2] - 6 * v * v * w[0][2] +
                4 * w[1][0] - 12 * v * w[1][0] + 8 * v * v * w[1][0] + 16 * v * w[1][1] -
                16 * v * v * w[1][1] - 4 * v * w[1][2] + 8 * v * v * w[1][2] - w[2][0] +
                3 * v * w[2][0] - 2 * v * v * w[2][0] - 4 * v * w[2][1] +
                4 * v * v * w[2][1] + v * w[2][2] - 2 * v * v * w[2][2]);
    Float ua =
        (2 * (1 - 3 * v + 2 * v * v) * w[0][0] + 8 * v * w[0][1] - 8 * v * v * w[0][1] -
         2 * v * w[0][2] + 4 * v * v * w[0][2] - 4 * w[1][0] + 12 * v * w[1][0] -
         8 * v * v * w[1][0] - 16 * v * w[1][1] + 16 * v * v * w[1][1] + 4 * v * w[1][2] -
         8 * v * v * w[1][2] + 2 * w[2][0] - 6 * v * w[2][0] + 4 * v * v * w[2][0] +
         8 * v * w[2][1] - 8 * v * v * w[2][1] - 2 * v * w[2][2] + 4 * v * v * w[2][2]);

    return QuadraticPDF(p[0], ua, ub, uc) * QuadraticPDF(p[1], va, vb, vc);
}

Point2f InvertBiquadraticSample(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w) {
    Point2f u;

    // Evaluate the v-marginal CDF at p[1] to compute u[1].
    u[1] = (p[1] *
            ((6 - 9 * p[1] + 4 * p[1] * p[1]) * w[0][0] + 6 * (4 * w[1][0] + w[2][0]) +
             3 * p[1] *
                 (4 * w[0][1] - w[0][2] - 12 * w[1][0] + 16 * w[1][1] - 4 * w[1][2] -
                  3 * w[2][0] + 4 * w[2][1] - w[2][2]) +
             4 * p[1] * p[1] *
                 (-2 * w[0][1] + w[0][2] + 4 * w[1][0] - 8 * w[1][1] + 4 * w[1][2] +
                  w[2][0] - 2 * w[2][1] + w[2][2]))) /
           (w[0][0] + 4 * w[0][1] + w[0][2] + 4 * w[1][0] + 16 * w[1][1] + 4 * w[1][2] +
            w[2][0] + 4 * w[2][1] + w[2][2]);

    u[0] =
        (p[0] *
         ((6 - 9 * p[0] + 4 * p[0] * p[0]) * (1 - 3 * p[1] + 2 * p[1] * p[1]) * w[0][0] +
          p[0] * ((12 - 8 * p[0]) * w[1][0] + (-3 + 4 * p[0]) * w[2][0]) -
          2 * p[1] * p[1] *
              (2 * (6 - 9 * p[0] + 4 * p[0] * p[0]) * w[0][1] +
               (-6 + 9 * p[0] - 4 * p[0] * p[0]) * w[0][2] +
               p[0] * (-12 * w[1][0] + 8 * p[0] * w[1][0] + 24 * w[1][1] -
                       16 * p[0] * w[1][1] - 12 * w[1][2] + 8 * p[0] * w[1][2] +
                       3 * w[2][0] - 4 * p[0] * w[2][0] - 6 * w[2][1] +
                       8 * p[0] * w[2][1] + 3 * w[2][2] - 4 * p[0] * w[2][2])) +
          p[1] * (4 * (6 - 9 * p[0] + 4 * p[0] * p[0]) * w[0][1] +
                  (-6 + 9 * p[0] - 4 * p[0] * p[0]) * w[0][2] +
                  p[0] * (-36 * w[1][0] + 24 * p[0] * w[1][0] + 48 * w[1][1] -
                          32 * p[0] * w[1][1] - 12 * w[1][2] + 8 * p[0] * w[1][2] +
                          9 * w[2][0] - 12 * p[0] * w[2][0] - 12 * w[2][1] +
                          16 * p[0] * w[2][1] + 3 * w[2][2] - 4 * p[0] * w[2][2])))) /
        ((1 - 3 * p[1] + 2 * p[1] * p[1]) * w[0][0] + 4 * w[1][0] + w[2][0] +
         p[1] * (4 * w[0][1] - w[0][2] - 12 * w[1][0] + 16 * w[1][1] - 4 * w[1][2] -
                 3 * w[2][0] + 4 * w[2][1] - w[2][2]) +
         2 * p[1] * p[1] *
             (-2 * w[0][1] + w[0][2] + 4 * w[1][0] - 8 * w[1][1] + 4 * w[1][2] + w[2][0] -
              2 * w[2][1] + w[2][2]));

    return u;
}

Point2f SampleBezier2D(Point2f su, pstd::array<pstd::array<Float, 3>, 3> w, Float *pdf) {
    // Sample the marginal quadratic in v
    Float vp[3] = {w[0][0] + w[1][0] + w[2][0], w[0][1] + w[1][1] + w[2][1],
                   w[0][2] + w[1][2] + w[2][2]};
    Float vPDF;
    Float v =
        SampleBezierCurve(su[1], vp[0], vp[1], vp[2], pdf != nullptr ? &vPDF : nullptr);

    Float up[3] = {Lerp(v, Lerp(v, w[0][0], w[0][1]), Lerp(v, w[0][1], w[0][2])),
                   Lerp(v, Lerp(v, w[1][0], w[1][1]), Lerp(v, w[1][1], w[1][2])),
                   Lerp(v, Lerp(v, w[2][0], w[2][1]), Lerp(v, w[2][1], w[2][2]))};
    Float uPDF;
    Float u =
        SampleBezierCurve(su[0], up[0], up[1], up[2], pdf != nullptr ? &uPDF : nullptr);

    if (pdf != nullptr)
        *pdf = uPDF * vPDF;

    return {u, v};
}

Float Bezier2DPDF(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w) {
    Float vp[3] = {w[0][0] + w[1][0] + w[2][0], w[0][1] + w[1][1] + w[2][1],
                   w[0][2] + w[1][2] + w[2][2]};

    Float up[3] = {
        Lerp(p[1], Lerp(p[1], w[0][0], w[0][1]), Lerp(p[1], w[0][1], w[0][2])),
        Lerp(p[1], Lerp(p[1], w[1][0], w[1][1]), Lerp(p[1], w[1][1], w[1][2])),
        Lerp(p[1], Lerp(p[1], w[2][0], w[2][1]), Lerp(p[1], w[2][1], w[2][2]))};

    return (BezierCurvePDF(p[0], up[0], up[1], up[2]) *
            BezierCurvePDF(p[1], vp[0], vp[1], vp[2]));
}

Point2f InvertBezier2DSample(Point2f p, pstd::array<pstd::array<Float, 3>, 3> w) {
    Point2f u;
    // Marginal quadratic in v
    Float vp[3] = {w[0][0] + w[1][0] + w[2][0], w[0][1] + w[1][1] + w[2][1],
                   w[0][2] + w[1][2] + w[2][2]};
    u[1] = InvertBezierCurveSample(p[1], vp[0], vp[1], vp[2]);

    Float up[3] = {
        Lerp(p[1], Lerp(p[1], w[0][0], w[0][1]), Lerp(p[1], w[0][1], w[0][2])),
        Lerp(p[1], Lerp(p[1], w[1][0], w[1][1]), Lerp(p[1], w[1][1], w[1][2])),
        Lerp(p[1], Lerp(p[1], w[2][0], w[2][1]), Lerp(p[1], w[2][1], w[2][2]))};
    u[0] = InvertBezierCurveSample(p[0], up[0], up[1], up[2]);

    return u;
}

pstd::vector<Float> Sample1DFunction(std::function<Float(Float)> f, int nSteps,
                                     int nSamples, Float min, Float max,
                                     Allocator alloc) {
    pstd::vector<Float> values(nSteps, Float(0), alloc);
    for (int i = 0; i < nSteps; ++i) {
        double accum = 0;
        // One extra so that we sample at the very start and the very end.
        for (int j = 0; j < nSamples + 1; ++j) {
            Float delta = Float(j) / nSamples;
            Float v = Lerp((i + delta) / Float(nSteps), min, max);
            Float fv = std::abs(f(v));
            accum = std::max<double>(accum, fv);
        }
        // There's actually no need for the divide by nSamples, since
        // these are normalzed into a PDF anyway.
        values[i] = accum;
    }
    return values;
}

Array2D<Float> Sample2DFunction(std::function<Float(Float, Float)> f, int nu, int nv,
                                int nSamples, Bounds2f domain, Allocator alloc) {
    std::vector<Point2f> samples(nSamples);
    for (int i = 0; i < nSamples; ++i)
        samples[i] = Point2f(RadicalInverse(0, i), RadicalInverse(1, i));
    // Check the corners, too.
    samples.push_back(Point2f(0, 1));
    samples.push_back(Point2f(1, 0));
    samples.push_back(Point2f(1, 1));

    Array2D<Float> values(nu, nv, alloc);
    for (int v = 0; v < nv; ++v) {
        for (int u = 0; u < nu; ++u) {
            double accum = 0;
            for (size_t i = 0; i < samples.size(); ++i) {
                Point2f p = domain.Lerp(
                    Point2f((u + samples[i][0]) / nu, (v + samples[i][1]) / nv));
                Float fuv = std::abs(f(p.x, p.y));
                accum = std::max<double>(accum, fuv);
            }
            // There's actually no need for the divide by nSamples, since
            // these are normalzed into a PDF anyway.
            values(u, v) = accum;
        }
    }

    return values;
}

// TODO: work on fp robustness.
//
// https://www.solidangle.com/research/egsr2013_spherical_rectangle.pdf
// discusses the issue, but seems to just do a bunch of clamping.
//
// See also
// http://graphics.pixar.com/library/StatFrameworkForImportance/paper.pdf for
// some discussion of this.
pstd::array<Float, 3> SampleSphericalTriangle(const pstd::array<Point3f, 3> &v,
                                              const Point3f &p, const Point2f &u,
                                              Float *pdf) {
    using Vector3d = Vector3<Float>;
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
        if (pdf != nullptr)
            *pdf = 0;
        return {};
    }
    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxa = Normalize(cxa);

    // See comment in Triangle::SolidAngle() for ordering...
    Float alpha = AngleBetween(cxa, -axb);
    Float beta = AngleBetween(axb, -bxc);
    Float gamma = AngleBetween(bxc, -cxa);

    // Spherical area of the triangle.
    Float A = alpha + beta + gamma - Pi;
    if (A <= 0) {
        if (pdf != nullptr)
            *pdf = 0;
        return {};
    }
    if (pdf != nullptr)
        *pdf = 1 / A;

    // Uniformly sample triangle area
    Float Ap = u[0] * A;

    // Compute sin beta' and cos beta' for the point along the edge b
    // corresponding to the area sampled, A'.

    Float cosAlpha = std::cos(alpha), sinAlpha = std::sin(alpha);

    // TODO? Permute vertices so we always sample along the longest edge?
    // via Max:
    // s = sin(\hat A)cos(alpha) - cos(\hat A)sin(alpha) = sin(\hat A)cos(alpha)
    // - cos(\hat A) sqrt(1 - cos(alpha)^2); t = cos(\hat A)cos(alpha) +
    // sin(\hat A)sin(alpha) = cos(\hat A)cos(alpha) + sin(\hat A) sqrt(1 -
    // cos(alpha)^2);
    Float sinPhi = std::sin(Ap) * cosAlpha - std::cos(Ap) * SafeSqrt(1 - Sqr(cosAlpha));
    Float cosPhi = std::cos(Ap) * cosAlpha + std::sin(Ap) * SafeSqrt(1 - Sqr(cosAlpha));

    Float uu = cosPhi - cosAlpha;
    Float vv = sinPhi + sinAlpha * Dot(a, b) /* cos c */;
    Float cosBetap = (((vv * cosPhi - uu * sinPhi) * cosAlpha - vv) /
                      ((vv * sinPhi + uu * cosPhi) * sinAlpha));
#if 0
    CHECK_RARE(1e-6, cosBetap < -1.001 || cosBetap > 1.001);
    if (cosBetap < -1.001 || cosBetap > 1.001)
        LOG_ERROR("cbp %f", cosBetap);
#endif

    // Happens if the triangle basically covers the entire hemisphere.
    // We currently depend on calling code to detect this case, which
    // is sort of ugly/unfortunate.
    CHECK(!std::isnan(cosBetap));
    cosBetap = Clamp(cosBetap, -1, 1);
    Float sinBetap = SafeSqrt(1 - cosBetap * cosBetap);

    // Gram-Schmidt
    auto GS = [](const Vector3d &a, const Vector3d &b) {
        return Normalize(a - Dot(a, b) * b);
    };

    // Compute c', the point along the arc between b' and a.
    Vector3d cp = cosBetap * a + sinBetap * GS(c, a);

    Float cosTheta = 1 - u[1] * (1 - Dot(cp, b));
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);

    // Compute direction on the sphere.
    Vector3d w = cosTheta * b + sinTheta * GS(cp, b);

    // Compute barycentrics. Subset of Moller-Trumbore intersection test.
    Vector3d e1(v[1] - v[0]), e2(v[2] - v[0]);
    Vector3d s1 = Cross(w, e2);
    Float divisor = Dot(s1, e1);

    CHECK_RARE(1e-6, divisor == 0);
    if (divisor == 0) {
        // This happens with triangles that cover (nearly) the whole
        // hemisphere.
        // LOG_ERROR("Divisor 0. A = %f", A);
        return {1.f / 3.f, 1.f / 3.f, 1.f / 3.f};
    }
    Float invDivisor = 1 / divisor;

    // Compute first barycentric coordinate
    Vector3d s(p - v[0]);
    Float b1 = Dot(s, s1) * invDivisor;

    // Compute second barycentric coordinate
    Vector3d s2 = Cross(s, e1);
    Float b2 = Dot(w, s2) * invDivisor;

    // We get goofy barycentrics for very small and very large (w.r.t. the
    // sphere) triangles.
    b1 = Clamp(b1, 0, 1);
    b2 = Clamp(b2, 0, 1);
    if (b1 + b2 > 1) {
        b1 /= b1 + b2;
        b2 /= b1 + b2;
    }

    return {Float(1 - b1 - b2), Float(b1), Float(b2)};
}

// Via Jim Arvo's SphTri.C
Point2f InvertSphericalTriangleSample(const pstd::array<Point3f, 3> &v, const Point3f &p,
                                      const Vector3f &w) {
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
    CHECK_RARE(1e-5, LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 ||
                         LengthSquared(cxa) == 0);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 || LengthSquared(cxa) == 0)
        return Point2f(0.5, 0.5);

    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxa = Normalize(cxa);

    // See comment in Triangle::SolidAngle() for ordering...
    double alpha = AngleBetween(cxa, -axb);
    double beta = AngleBetween(axb, -bxc);
    double gamma = AngleBetween(bxc, -cxa);

    // Spherical area of the triangle.
    double A = alpha + beta + gamma - Pi;

    // Assume that w is normalized...

    // Compute the new C vertex, which lies on the arc defined by b-w
    // and the arc defined by a-c.
    Vector3d cp = Normalize(Cross(Cross(b, Vector3d(w)), Cross(c, a)));

    // Adjust the sign of cp.  Make sure it's on the arc between A and C.
    if (Dot(cp, a + c) < 0)
        cp = -cp;

    // Compute x1, the area of the sub-triangle over the original area.
    // The AngleBetween() calls are computing the dihedral angles (a, b, cp)
    // and (a, cp, b) respectively, FWIW...
    Vector3d cnxb = Cross(cp, b), axcn = Cross(a, cp);
    CHECK_RARE(1e-5, LengthSquared(cnxb) == 0 || LengthSquared(axcn) == 0);
    if (LengthSquared(cnxb) == 0 || LengthSquared(axcn) == 0)
        return Point2f(0.5, 0.5);
    cnxb = Normalize(cnxb);
    axcn = Normalize(axcn);

    Float sub_area = alpha + AngleBetween(axb, cnxb) + AngleBetween(axcn, -cnxb) - Pi;
    Float u0 = sub_area / A;

    // Now compute the second coordinate using the new C vertex.
    Float z = Dot(Vector3d(w), b);
    Float u1 = (1 - z) / (1 - Dot(cp, b));

    return Point2f(Clamp(u0, 0, 1), Clamp(u1, 0, 1));
}

Point3f SampleSphericalQuad(const Point3f &pRef, const Point3f &s, const Vector3f &ex,
                            const Vector3f &ey, const Point2f &u, Float *pdf) {
    // SphQuadInit()
    // local reference system ’R’
    Float exl = Length(ex), eyl = Length(ey);
    Frame R = Frame::FromXY(ex / exl, ey / eyl);

    // compute rectangle coords in local reference system
    Vector3f d = s - pRef;
    Vector3f dLocal = R.ToLocal(d);
    Float z0 = dLocal.z;

    // flip ’z’ to make it point against ’Q’
    if (z0 > 0) {
        R.z = -R.z;
        z0 *= -1;
    }
    Float z0sq = Sqr(z0);
    Float x0 = dLocal.x;
    Float y0 = dLocal.y;
    Float x1 = x0 + exl;
    Float y1 = y0 + eyl;
    Float y0sq = Sqr(y0), y1sq = Sqr(y1);

    // create vectors to four vertices
    Vector3f v00(x0, y0, z0), v01(x0, y1, z0);
    Vector3f v10(x1, y0, z0), v11(x1, y1, z0);

    // compute normals to edges
    Vector3f n0 = Normalize(Cross(v00, v10));
    Vector3f n1 = Normalize(Cross(v10, v11));
    Vector3f n2 = Normalize(Cross(v11, v01));
    Vector3f n3 = Normalize(Cross(v01, v00));

    // compute internal angles (gamma_i)
    Float g0 = AngleBetween(-n0, n1);
    Float g1 = AngleBetween(-n1, n2);
    Float g2 = AngleBetween(-n2, n3);
    Float g3 = AngleBetween(-n3, n0);

    // compute predefined constants
    Float b0 = n0.z, b1 = n2.z, b0sq = Sqr(b0), b1sq = Sqr(b1);

    // compute solid angle from internal angles
    Float solidAngle = double(g0) + double(g1) + double(g2) + double(g3) - 2. * Pi;
    CHECK_RARE(1e-5, solidAngle <= 0);
    if (solidAngle <= 0) {
        if (pdf != nullptr)
            *pdf = 0;
        return Point3f(s + u[0] * ex + u[1] * ey);
    }
    if (pdf != nullptr)
        *pdf = std::max<Float>(0, 1 / solidAngle);

    if (solidAngle < 1e-3)
        return Point3f(s + u[0] * ex + u[1] * ey);

    // SphQuadSample
    // 1. compute ’cu’
    // Float au = u[0] * solidAngle + k;   // original
    Float au = u[0] * solidAngle - g2 - g3;
    Float fu = (std::cos(au) * b0 - b1) / std::sin(au);
    Float fusq = Sqr(fu);
    Float cu = std::copysign(1 / std::sqrt(Sqr(fu) + b0sq), fu);
    cu = Clamp(cu, -OneMinusEpsilon, OneMinusEpsilon);  // avoid NaNs

    // 2. compute ’xu’
    Float xu = -(cu * z0) / SafeSqrt(1 - Sqr(cu));
    xu = Clamp(xu, x0, x1);  // avoid Infs

    // 3. compute ’yv’
    Float dd = std::sqrt(Sqr(xu) + z0sq);
    Float h0 = y0 / std::sqrt(Sqr(dd) + y0sq);
    Float h1 = y1 / std::sqrt(Sqr(dd) + y1sq);
    Float hv = h0 + u[1] * (h1 - h0), hvsq = Sqr(hv);
    const Float eps = 1e-6;
    Float yv = (hvsq < 1 - eps) ? (hv * dd) / std::sqrt(1 - hvsq) : y1;

    // 4. transform (xu,yv,z0) to world coords
    return pRef + R.FromLocal(Vector3f(xu, yv, z0));
}

Point2f InvertSphericalQuadSample(const Point3f &pRef, const Point3f &s,
                                  const Vector3f &ex, const Vector3f &ey,
                                  const Point3f &pQuad) {
    // TODO: Delete anything unused in the below...

    // SphQuadInit()
    // local reference system ’R’
    Float exl = Length(ex), eyl = Length(ey);
    Frame R = Frame::FromXY(ex / exl, ey / eyl);

    // compute rectangle coords in local reference system
    Vector3f d = s - pRef;
    Vector3f dLocal = R.ToLocal(d);
    Float z0 = dLocal.z;

    // flip ’z’ to make it point against ’Q’
    if (z0 > 0) {
        R.z = -R.z;
        z0 *= -1;
    }
    Float z0sq = Sqr(z0);
    Float x0 = dLocal.x;
    Float y0 = dLocal.y;
    Float x1 = x0 + exl;
    Float y1 = y0 + eyl;
    Float y0sq = Sqr(y0), y1sq = Sqr(y1);

    // create vectors to four vertices
    Vector3f v00(x0, y0, z0), v01(x0, y1, z0);
    Vector3f v10(x1, y0, z0), v11(x1, y1, z0);

    // compute normals to edges
    Vector3f n0 = Normalize(Cross(v00, v10));
    Vector3f n1 = Normalize(Cross(v10, v11));
    Vector3f n2 = Normalize(Cross(v11, v01));
    Vector3f n3 = Normalize(Cross(v01, v00));

    // compute internal angles (gamma_i)
    Float g0 = AngleBetween(-n0, n1);
    Float g1 = AngleBetween(-n1, n2);
    Float g2 = AngleBetween(-n2, n3);
    Float g3 = AngleBetween(-n3, n0);

    // compute predefined constants
    Float b0 = n0.z, b1 = n2.z, b0sq = Sqr(b0), b1sq = Sqr(b1);

    // compute solid angle from internal angles
    Float solidAngle = double(g0) + double(g1) + double(g2) + double(g3) - 2. * Pi;

    // TODO: this (rarely) goes differently than sample. figure out why...
    if (solidAngle < 1e-3) {
        Vector3f pq = pQuad - s;
        return Point2f(Dot(pq, ex) / LengthSquared(ex), Dot(pq, ey) / LengthSquared(ey));
    }

    Vector3f v = R.ToLocal(pQuad - pRef);
    Float xu = v.x, yv = v.y;

    xu = Clamp(xu, x0, x1);  // avoid Infs
    if (xu == 0)
        xu = 1e-10;

    // DOing all this in double actually makes things slightly worse???!?
    // Float fusq = (1 - b0sq * Sqr(cu)) / Sqr(cu);
    // Float fusq = 1 / Sqr(cu) - b0sq;  // more stable
    Float invcusq = 1 + z0sq / Sqr(xu);
    Float fusq = invcusq - b0sq;  // the winner so far
    Float fu = std::copysign(std::sqrt(fusq), xu);
    // Note, though have 1 + z^2/x^2 - b0^2, which isn't great if b0 \approx 1
    // double fusq = 1. - Sqr(double(b0)) + Sqr(double(z0) / double(xu));  //
    // this is worse?? double fu = std::copysign(std::sqrt(fusq), cu);
    CHECK_RARE(1e-6, fu == 0);

    // State of the floating point world: in the bad cases, about half the
    // error seems to come from inaccuracy in fu and half comes from
    // inaccuracy in sqrt/au.
    //
    // For fu, the main issue comes adding a small value to 1+ in invcusq
    // and then having b0sq be close to one, so having catistrophic
    // cancellation affect fusq. Approximating it as z0sq / Sqr(xu) when
    // b0sq is close to one doesn't help, however..
    //
    // For au, DifferenceOfProducts doesn't seem to help with the two
    // factors. Furthermore, while it would be nice to think about this
    // like atan(y/x) and then rewrite/simplify y/x, we need to do so in a
    // way that doesn't flip the sign of x and y, which would be fine if we
    // were computing y/x, but messes up atan2's quadrant-determinations...

    Float sqrt = SafeSqrt(DifferenceOfProducts(b0, b0, b1, b1) + fusq);
    // No benefit to difference of products here...
    Float au = std::atan2(-(b1 * fu) - std::copysign(b0 * sqrt, fu * b0),
                          b0 * b1 - sqrt * std::abs(fu));
    if (au > 0)
        au -= 2 * Pi;

    if (fu == 0)
        au = Pi;
    Float u0 = (au + g2 + g3) / solidAngle;

    Float ddsq = Sqr(xu) + z0sq;
    Float dd = std::sqrt(ddsq);
    Float h0 = y0 / std::sqrt(ddsq + y0sq);
    Float h1 = y1 / std::sqrt(ddsq + y1sq);
    Float yvsq = Sqr(yv);

    Float u1[2] = {(DifferenceOfProducts(h0, h0, h0, h1) -
                    std::abs(h0 - h1) * std::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq)) /
                       Sqr(h0 - h1),
                   (DifferenceOfProducts(h0, h0, h0, h1) +
                    std::abs(h0 - h1) * std::sqrt(yvsq * (ddsq + yvsq)) / (ddsq + yvsq)) /
                       Sqr(h0 - h1)};

    // TODO: yuck is there a better way to figure out which is the right
    // solution?
    Float hv[2] = {Lerp(u1[0], h0, h1), Lerp(u1[1], h0, h1)};
    Float hvsq[2] = {Sqr(hv[0]), Sqr(hv[1])};
    Float yz[2] = {(hv[0] * dd) / std::sqrt(1 - hvsq[0]),
                   (hv[1] * dd) / std::sqrt(1 - hvsq[1])};

    Point2f u = (std::abs(yz[0] - yv) < std::abs(yz[1] - yv))
                    ? Point2f(Clamp(u0, 0, 1), u1[0])
                    : Point2f(Clamp(u0, 0, 1), u1[1]);

    return u;
}

pstd::array<Float, 3> LowDiscrepancySampleTriangleReference(Float u) {
    uint32_t uf = u * (1ull << 32);  // Fixed point
    Point3f A(1, 0, 0), B(0, 1, 0), C(0, 0, 1);
    for (int i = 0; i < 16; ++i) {
        int d = (uf >> (2 * (15 - i))) & 0x3;
        Point3f An, Bn, Cn;
        switch (d) {
        case 0:
            An = (B + C) / 2;
            Bn = (A + C) / 2;
            Cn = (A + B) / 2;
            break;
        case 1:
            An = A;
            Bn = (A + B) / 2;
            Cn = (A + C) / 2;
            break;
        case 2:
            An = (B + A) / 2;
            Bn = B;
            Cn = (B + C) / 2;
            break;
        case 3:
            An = (C + A) / 2;
            Bn = (C + B) / 2;
            Cn = C;
            break;
        }
        A = An;
        B = Bn;
        C = Cn;
    }
    Point3f mid = (A + B + C) / 3;
    return {mid.x, mid.y, mid.z};
}

/*
Benchmarks:
reference implementation: 708ms
  not that this is 1M points in 71ms -> 71ns / point
drop 3rd barycentric coord: 705(???)
  note that clang seems to actually use SIMD for this--woo!
ping-pong Point2f ABC[3][2] -> 813ms, inhibits putting in registers
avx2 (not working but looks like right instr mix): 327ms. feh.
*/
pstd::array<Float, 3> LowDiscrepancySampleTriangle(Float u) {
    uint32_t uf = u * (1ull << 32);  // Fixed point

    Float cx = 0.0f, cy = 0.0f;
    Float w = 0.5f;

    for (int i = 0; i < 16; i++) {
        uint32_t uu = uf >> 30;
        bool flip = (uu & 3) == 0;

        cy += ((uu & 1) == 0) * w;
        cx += ((uu & 2) == 0) * w;

        w *= flip ? -0.5f : 0.5f;

        uf <<= 2;
    }

    Float b0 = cx + w / 3.0f, b1 = cy + w / 3.0f;
    return {b0, b1, 1 - b0 - b1};
}

Vector3f SampleHenyeyGreenstein(const Vector3f &wo, Float g, const Point2f &u,
                                Float *pdf) {
    // Compute $\cos \theta$ for Henyey--Greenstein sample
    Float cosTheta;
    if (std::abs(g) < 1e-3)
        cosTheta = 1 - 2 * u[0];
    else {
        Float sqrTerm = (1 - g * g) / (1 - g + 2 * g * u[0]);
        cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
    }

    // Compute direction _wi_ for Henyey--Greenstein sample
    Float sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
    Float phi = 2 * Pi * u[1];

    Frame wFrame = Frame::FromZ(-wo);
    Vector3f wi = wFrame.FromLocal(SphericalDirection(sinTheta, cosTheta, phi));
    if (pdf)
        *pdf = EvaluateHenyeyGreenstein(-cosTheta, g);
    return wi;
}

void PiecewiseConstant1D::TestCompareDistributions(const PiecewiseConstant1D &da,
                                                   const PiecewiseConstant1D &db,
                                                   Float eps) {
    ASSERT_EQ(da.func.size(), db.func.size());
    ASSERT_EQ(da.cdf.size(), db.cdf.size());
    ASSERT_EQ(da.min, db.min);
    ASSERT_EQ(da.max, db.max);
    for (size_t i = 0; i < da.func.size(); ++i) {
        Float pdfa = da.func[i] / da.funcInt, pdfb = db.func[i] / db.funcInt;
        Float err = std::abs(pdfa - pdfb) / ((pdfa + pdfb) / 2);
        EXPECT_LT(err, eps) << pdfa << " - " << pdfb;
    }
}

PiecewiseConstant2D::PiecewiseConstant2D(pstd::span<const Float> func, int nx, int ny,
                                         Bounds2f domain, Allocator alloc)
    : domain(domain), pConditionalY(alloc), pMarginal(alloc) {
    CHECK_EQ(func.size(), (size_t)nx * (size_t)ny);
    pConditionalY.reserve(ny);
    for (int y = 0; y < ny; ++y)
        // Compute conditional sampling distribution for $\tilde{y}$
        // TODO: emplace_back is key so the alloc sticks. WHY?
        pConditionalY.emplace_back(func.subspan(y * nx, nx), domain.pMin[0],
                                   domain.pMax[0], alloc);
    // Compute marginal sampling distribution $p[\tilde{v}]$
    std::vector<Float> marginalFunc;
    marginalFunc.reserve(ny);
    for (int y = 0; y < ny; ++y)
        marginalFunc.push_back(pConditionalY[y].funcInt);
    pMarginal = PiecewiseConstant1D(marginalFunc, domain.pMin[1], domain.pMax[1], alloc);
}

void PiecewiseConstant2D::TestCompareDistributions(const PiecewiseConstant2D &da,
                                                   const PiecewiseConstant2D &db,
                                                   Float eps) {
    PiecewiseConstant1D::TestCompareDistributions(da.pMarginal, db.pMarginal, eps);

    ASSERT_EQ(da.pConditionalY.size(), db.pConditionalY.size());
    ASSERT_EQ(da.domain, db.domain);
    for (size_t i = 0; i < da.pConditionalY.size(); ++i)
        PiecewiseConstant1D::TestCompareDistributions(da.pConditionalY[i],
                                                      db.pConditionalY[i], eps);
}

Float SampleCatmullRom(pstd::span<const Float> x, pstd::span<const Float> f,
                       pstd::span<const Float> F, Float u, Float *fval, Float *pdf) {
    CHECK_EQ(x.size(), f.size());
    CHECK_EQ(f.size(), F.size());

    // Map _u_ to a spline interval by inverting _F_
    u *= F.back();
    int i = FindInterval(F.size(), [&](int i) { return F[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = x[i], x1 = x[i + 1];
    Float f0 = f[i], f1 = f[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0, d1;
    if (i > 0)
        d0 = width * (f1 - f[i - 1]) / (x1 - x[i - 1]);
    else
        d0 = f1 - f0;
    if (i + 2 < x.size())
        d1 = width * (f[i + 2] - f0) / (x[i + 2] - x0);
    else
        d1 = f1 - f0;

    // Re-scale _u_ for continous spline sampling step
    u = (u - F[i]) / width;

    // Invert definite integral over spline segment and return solution
    Float Fhat, fhat;
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        Fhat =
            EvaluatePolynomial(t, 0, f0, .5f * d0, (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                               .25f * (d0 + d1) + .5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    // Return the sample position and function value
    if (fval != nullptr)
        *fval = fhat;
    if (pdf != nullptr)
        *pdf = fhat / F.back();
    return x0 + width * t;
}

Float SampleCatmullRom2D(pstd::span<const Float> nodes1, pstd::span<const Float> nodes2,
                         pstd::span<const Float> values, pstd::span<const Float> cdf,
                         Float alpha, Float u, Float *fval, Float *pdf) {
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    Float weights[4];
    if (!CatmullRomWeights(nodes1, alpha, &offset, weights))
        return 0;

    // Define a lambda function to interpolate table entries
    auto interpolate = [&](pstd::span<const Float> array, int idx) {
        Float value = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                value += array[(offset + i) * nodes2.size() + idx] * weights[i];
        return value;
    };

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    Float maximum = interpolate(cdf, nodes2.size() - 1);
    u *= maximum;
    int idx =
        FindInterval(nodes2.size(), [&](int i) { return interpolate(cdf, i) <= u; });

    // Look up node positions and interpolated function values
    Float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    Float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;

    // Re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0)
        d0 = width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1]);
    else
        d0 = f1 - f0;
    if (idx + 2 < nodes2.size())
        d1 = width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert definite integral over spline segment and return solution

    // Set initial guess for $t$ by importance sampling a linear interpolant
    Float Fhat, fhat;
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        Fhat =
            EvaluatePolynomial(t, 0, f0, .5f * d0, (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                               .25f * (d0 + d1) + .5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    // Return the sample position and function value
    if (fval != nullptr)
        *fval = fhat;
    if (pdf != nullptr)
        *pdf = fhat / maximum;
    return x0 + width * t;
}

AliasTable::AliasTable(pstd::span<const Float> values, Allocator alloc)
    : p(values.size(), alloc), pdf(values.size(), alloc), alias(values.size(), alloc) {
    // Compute PDF
    // Double precision here seems important; otherwise at
    // a96543654534c274e, area-blp-tri-mlt.pbrt fails--the issue is that
    // the first few buckets aren't initialized, due to round-off error
    // causing us not have as much PDF mass as expected to fill everything
    // up.
    //
    // TODO: it may be worth using doubles for Item::p and the computation
    // of |pg| below, just to be safe.
    Float sum = std::accumulate(values.begin(), values.end(), 0.);
    for (size_t i = 0; i < values.size(); ++i)
        pdf[i] = values[i] / sum;

    // Create worklists
    struct Item {
        Float p;
        size_t index;
    };
    std::vector<Item> small, large;
    for (size_t i = 0; i < pdf.size(); ++i) {
        Float p = pdf[i] * pdf.size();
        if (p < 1)
            small.push_back(Item{p, i});
        else
            large.push_back(Item{p, i});
    }

    // Build alias table
    // Vose's method, via https://www.keithschwarz.com/darts-dice-coins/
    while (!small.empty() && !large.empty()) {
        Item l = small.back();
        small.pop_back();
        Item g = large.back();
        large.pop_back();

        p[l.index] = l.p;
        alias[l.index] = g.index;

        Float pg = (l.p + g.p) - 1;
        if (pg < 1)
            small.push_back(Item{pg, g.index});
        else
            large.push_back(Item{pg, g.index});
    }

    while (!large.empty()) {
        Item g = large.back();
        large.pop_back();

        p[g.index] = 1;
        alias[g.index] = -1;
    }

    while (!small.empty()) {
        Item l = small.back();
        small.pop_back();

        p[l.index] = 1;
        alias[l.index] = -1;
    }
}

int AliasTable::Sample(Float u, Float *pdfOut, Float *uRemapped) const {
    int offset = std::min<int>(u * p.size(), p.size() - 1);
    Float up = std::min<Float>(u * p.size() - offset, OneMinusEpsilon);
    if (up < p[offset]) {
        DCHECK_GT(pdf[offset], 0);
        if (pdfOut)
            *pdfOut = pdf[offset];
        if (uRemapped)
            *uRemapped = std::min<Float>(up / p[offset], OneMinusEpsilon);
        return offset;
    } else {
        DCHECK_GE(alias[offset], 0);
        DCHECK_GT(pdf[alias[offset]], 0);
        if (pdfOut)
            *pdfOut = pdf[alias[offset]];
        if (uRemapped)
            *uRemapped =
                std::min<Float>((up - p[offset]) / (1 - p[offset]), OneMinusEpsilon);
        return alias[offset];
    }
}

std::string AliasTable::ToString() const {
    return StringPrintf("[ AliasTable p: %s pdf: %s alias: %s ]", p, pdf, alias);
}

std::string SummedAreaTable::ToString() const {
    return StringPrintf("[ SummedAreaTable sum: %s ]", sum);
}

}  // namespace pbrt
