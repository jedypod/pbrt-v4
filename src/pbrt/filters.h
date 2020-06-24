
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

#ifndef PBRT_FILTERS_BOX_H
#define PBRT_FILTERS_BOX_H

// filters.h*
#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/util/math.h>

#include <memory>
#include <string>

namespace pbrt {

// Box Filter Declarations
class BoxFilter final : public Filter {
  public:
    BoxFilter(const Vector2f &radius = Vector2f(0.5, 0.5)) : Filter(radius) {}

    static BoxFilter *Create(const ParameterDictionary &dict, Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const;
    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float Integral() const final { return 2 * radius.x * 2 * radius.y; }

    std::string ToString() const;
};

// Gaussian Filter Declarations
class GaussianFilter final : public Filter {
  public:
    // GaussianFilter Public Methods
    GaussianFilter(const Vector2f &radius, Float sigma = 0.5f)
        : Filter(radius),
          sigma(sigma),
          expX(Gaussian(radius.x, 0, sigma)),
          expY(Gaussian(radius.y, 0, sigma)),
          sampler(this) { }
    static GaussianFilter *Create(const ParameterDictionary &dict, Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const;
    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const final { return sampler.Sample(u); }
    PBRT_HOST_DEVICE
    Float Integral() const;

    std::string ToString() const;

  private:
    // GaussianFilter Private Data
    Float sigma;
    Float expX, expY;
    FilterSampler sampler;
};


// Mitchell Filter Declarations
class MitchellFilter final : public Filter {
  public:
    // MitchellFilter Public Methods
    MitchellFilter(const Vector2f &radius, Float B = 1.f/3.f, Float C = 1.f/3.f)
        : Filter(radius), B(B), C(C), sampler(this) { }
    static MitchellFilter *Create(const ParameterDictionary &dict, Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const;
    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const final { return sampler.Sample(u); }
    PBRT_HOST_DEVICE
    Float Integral() const {
        // integrate filters.nb
        return radius.x * radius.y / 4;
    }

    std::string ToString() const;

  private:
    Float B, C;
    FilterSampler sampler;

    PBRT_HOST_DEVICE
    Float Mitchell1D(Float x) const {
        x = std::abs(2 * x);
        if (x <= 1)
            return ((12 - 9 * B - 6 * C) * x * x * x +
                    (-18 + 12 * B + 6 * C) * x * x + (6 - 2 * B)) *
                   (1.f / 6.f);
        else if (x <= 2)
            return ((-B - 6 * C) * x * x * x + (6 * B + 30 * C) * x * x +
                    (-12 * B - 48 * C) * x + (8 * B + 24 * C)) *
                   (1.f / 6.f);
        else
            return 0;
    }
};


// Sinc Filter Declarations
class LanczosSincFilter final : public Filter {
  public:
    // LanczosSincFilter Public Methods
    LanczosSincFilter(const Vector2f &radius, Float tau = 3.f)
        : Filter(radius), tau(tau), sampler(this) { }
    static LanczosSincFilter *Create(const ParameterDictionary &dict, Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const;
    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const final { return sampler.Sample(u); }

    std::string ToString() const;

  private:
    Float tau;
    FilterSampler sampler;
};

// Triangle Filter Declarations
class TriangleFilter final : public Filter {
  public:
    TriangleFilter(const Vector2f &radius) : Filter(radius) {}
    static TriangleFilter *Create(const ParameterDictionary &dict, Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const;
    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const;
    PBRT_HOST_DEVICE
    Float Integral() const final { return radius.x * radius.x * radius.y * radius.y; }

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_FILTERS_H
