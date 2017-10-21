
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

// core/interpolation.cpp*
#include "util/interpolation.h"

#include "util/mathutil.h"
#include <glog/logging.h>


namespace pbrt {

// Spline Interpolation Definitions
Float CatmullRom(absl::Span<const Float> nodes, absl::Span<const Float> values, Float x) {
    CHECK_EQ(nodes.size(), values.size());
    if (!(x >= nodes.front() && x <= nodes.back())) return 0;
    int idx = FindInterval(nodes.size(), [&](int i) { return nodes[i] <= x; });
    Float x0 = nodes[idx], x1 = nodes[idx + 1];
    Float f0 = values[idx], f1 = values[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;
    if (idx > 0)
        d0 = width * (f1 - values[idx - 1]) / (x1 - nodes[idx - 1]);
    else
        d0 = f1 - f0;

    if (idx + 2 < nodes.size())
        d1 = width * (values[idx + 2] - f0) / (nodes[idx + 2] - x0);
    else
        d1 = f1 - f0;

    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 +
           (t3 - 2 * t2 + t) * d0 + (t3 - t2) * d1;
}

bool CatmullRomWeights(absl::Span<const Float> nodes, Float x, int *offset,
                       absl::Span<Float> weights) {
    CHECK_GE(weights.size(), 4);
    // Return _false_ if _x_ is out of bounds
    if (!(x >= nodes.front() && x <= nodes.back())) return false;

    // Search for the interval _idx_ containing _x_
    int idx = FindInterval(nodes.size(), [&](int i) { return nodes[i] <= x; });
    *offset = idx - 1;
    Float x0 = nodes[idx], x1 = nodes[idx + 1];

    // Compute the $t$ parameter and powers
    Float t = (x - x0) / (x1 - x0), t2 = t * t, t3 = t2 * t;

    // Compute initial node weights $w_1$ and $w_2$
    weights[1] = 2 * t3 - 3 * t2 + 1;
    weights[2] = -2 * t3 + 3 * t2;

    // Compute first node weight $w_0$
    if (idx > 0) {
        Float w0 = (t3 - 2 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        Float w0 = t3 - 2 * t2 + t;
        weights[0] = 0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    // Compute last node weight $w_3$
    if (idx + 2 < nodes.size()) {
        Float w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        Float w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0;
    }
    return true;
}

Float SampleCatmullRom(absl::Span<const Float> x, absl::Span<const Float> f,
                       absl::Span<const Float> F, Float u, Float *fval, Float *pdf) {
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
        Fhat = EvaluatePolynomial(t, 0, f0, .5f * d0,
                                  (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                                  .25f * (d0 + d1) + .5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    // Return the sample position and function value
    if (fval) *fval = fhat;
    if (pdf) *pdf = fhat / F.back();
    return x0 + width * t;
}

Float SampleCatmullRom2D(absl::Span<const Float> nodes1, absl::Span<const Float> nodes2,
                         absl::Span<const Float> values, absl::Span<const Float> cdf,
                         Float alpha, Float u, Float *fval, Float *pdf) {
    // Determine offset and coefficients for the _alpha_ parameter
    int offset;
    Float weights[4];
    if (!CatmullRomWeights(nodes1, alpha, &offset, weights)) return 0;

    // Define a lambda function to interpolate table entries
    auto interpolate = [&](absl::Span<const Float> array, int idx) {
        Float value = 0;
        for (int i = 0; i < 4; ++i)
            if (weights[i] != 0)
                value += array[(offset + i) * nodes2.size() + idx] * weights[i];
        return value;
    };

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    Float maximum = interpolate(cdf, nodes2.size() - 1);
    u *= maximum;
    int idx = FindInterval(nodes2.size(),
                           [&](int i) { return interpolate(cdf, i) <= u; });

    // Look up node positions and interpolated function values
    Float f0 = interpolate(values, idx), f1 = interpolate(values, idx + 1);
    Float x0 = nodes2[idx], x1 = nodes2[idx + 1];
    Float width = x1 - x0;
    Float d0, d1;

    // Re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    if (idx > 0)
        d0 = width * (f1 - interpolate(values, idx - 1)) /
             (x1 - nodes2[idx - 1]);
    else
        d0 = f1 - f0;
    if (idx + 2 < nodes2.size())
        d1 = width * (interpolate(values, idx + 2) - f0) /
             (nodes2[idx + 2] - x0);
    else
        d1 = f1 - f0;

    // Invert definite integral over spline segment and return solution

    // Set initial guess for $t$ by importance sampling a linear interpolant
    Float Fhat, fhat;
    auto eval = [&](Float t) -> std::pair<Float, Float> {
        Fhat = EvaluatePolynomial(t, 0, f0, .5f * d0,
                                  (1.f / 3.f) * (-2 * d0 - d1) + f1 - f0,
                                  .25f * (d0 + d1) + .5f * (f0 - f1));
        fhat = EvaluatePolynomial(t, f0, d0, -2 * d0 - d1 + 3 * (f1 - f0),
                                  d0 + d1 + 2 * (f0 - f1));
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    // Return the sample position and function value
    if (fval) *fval = fhat;
    if (pdf) *pdf = fhat / maximum;
    return x0 + width * t;
}

Float IntegrateCatmullRom(absl::Span<const Float> x, absl::Span<const Float> values,
                          absl::Span<Float> cdf) {
    CHECK_EQ(x.size(), values.size());
    Float sum = 0;
    cdf[0] = 0;
    for (int i = 0; i < x.size() - 1; ++i) {
        // Look up $x_i$ and function values of spline segment _i_
        Float x0 = x[i], x1 = x[i + 1];
        Float f0 = values[i], f1 = values[i + 1];
        Float width = x1 - x0;

        // Approximate derivatives using finite differences
        Float d0, d1;
        if (i > 0)
            d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
        else
            d0 = f1 - f0;
        if (i + 2 < x.size())
            d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
        else
            d1 = f1 - f0;

        // Keep a running sum and build a cumulative distribution function
        sum += ((d0 - d1) * (1.f / 12.f) + (f0 + f1) * .5f) * width;
        cdf[i + 1] = sum;
    }
    return sum;
}

Float InvertCatmullRom(absl::Span<const Float> x, absl::Span<const Float> values, Float u) {
    // Stop when _u_ is out of bounds
    if (!(u > values.front()))
        return x.front();
    else if (!(u < values.back()))
        return x.back();

    // Map _u_ to a spline interval by inverting _values_
    int i = FindInterval(values.size(), [&](int i) { return values[i] <= u; });

    // Look up $x_i$ and function values of spline segment _i_
    Float x0 = x[i], x1 = x[i + 1];
    Float f0 = values[i], f1 = values[i + 1];
    Float width = x1 - x0;

    // Approximate derivatives using finite differences
    Float d0, d1;
    if (i > 0)
        d0 = width * (f1 - values[i - 1]) / (x1 - x[i - 1]);
    else
        d0 = f1 - f0;
    if (i + 2 < x.size())
        d1 = width * (values[i + 2] - f0) / (x[i + 2] - x0);
    else
        d1 = f1 - f0;

    auto eval = [&](Float t) -> std::pair<Float, Float> {
        // Compute powers of _t_
        Float t2 = t * t, t3 = t2 * t;

        // Set _Fhat_ using Equation (8.27)
        Float Fhat = (2 * t3 - 3 * t2 + 1) * f0 + (-2 * t3 + 3 * t2) * f1 +
               (t3 - 2 * t2 + t) * d0 + (t3 - t2) * d1;
        // Set _fhat_ using Equation (not present)
        Float fhat = (6 * t2 - 6 * t) * f0 + (-6 * t2 + 6 * t) * f1 +
               (3 * t2 - 4 * t + 1) * d0 + (3 * t2 - 2 * t) * d1;
        return {Fhat - u, fhat};
    };
    Float t = NewtonBisection(0, 1, eval);

    return x0 + t * width;
}

// Fourier Interpolation Definitions
Float Fourier(absl::Span<const Float> a, double cosPhi) {
    double value = 0.0;
    // Initialize cosine iterates
    double cosKMinusOnePhi = cosPhi;
    double cosKPhi = 1;
    for (int k = 0; k < a.size(); ++k) {
        // Add the current summand and update the cosine iterates
        value += a[k] * cosKPhi;
        double cosKPlusOnePhi = 2 * cosPhi * cosKPhi - cosKMinusOnePhi;
        cosKMinusOnePhi = cosKPhi;
        cosKPhi = cosKPlusOnePhi;
    }
    return value;
}

Float SampleFourier(absl::Span<const Float> ak, absl::Span<const Float> recip, Float u,
                    Float *pdf, Float *phiPtr) {
    bool flip = (u >= 0.5);
    if (flip)
        u = 1 - 2 * (u - .5f);
    else
        u *= 2;

    // Hacky: trust that the last time NewtonBisection evaluates the lambda
    // will be at the phi value that's returned. In turn, by declaring 'f'
    // outside of the lambda (but capturing it by reference), we can grab
    // the value of the derivative at phi below to compute the PDF.
    double F, f;
    auto eval = [&](double phi) -> std::pair<double, double> {
        // Evaluate $F(\phi)$ and its derivative $f(\phi)$

        // Initialize sine and cosine iterates
        double cosPhi = std::cos(phi);
        double sinPhi = SafeSqrt(1 - cosPhi * cosPhi);
        double cosPhiPrev = cosPhi, cosPhiCur = 1;
        double sinPhiPrev = -sinPhi, sinPhiCur = 0;

        // Initialize _F_ and _f_ with the first series term
        F = ak[0] * phi;
        f = ak[0];
        for (int k = 1; k < ak.size(); ++k) {
            // Compute next sine and cosine iterates
            double sinPhiNext = 2 * cosPhi * sinPhiCur - sinPhiPrev;
            double cosPhiNext = 2 * cosPhi * cosPhiCur - cosPhiPrev;
            sinPhiPrev = sinPhiCur;
            sinPhiCur = sinPhiNext;
            cosPhiPrev = cosPhiCur;
            cosPhiCur = cosPhiNext;

            // Add the next series term to _F_ and _f_
            F += ak[k] * recip[k] * sinPhiNext;
            f += ak[k] * cosPhiNext;
        }
        F -= u * ak[0] * Pi;
        return { F, f };
    };
    double phi = NewtonBisection(0., double(Pi), eval);

    // Potentially flip $\phi$ and return the result
    if (flip) phi = 2 * Pi - phi;
    *pdf = (Float)(Inv2Pi * f / ak[0]);
    *phiPtr = (Float)phi;
    return f;
}

}  // namespace pbrt
