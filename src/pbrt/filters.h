
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
#include <pbrt/util/sampling.h>

#include <cmath>
#include <memory>
#include <string>

namespace pbrt {

class FilterSampler {
  public:
    FilterSampler(FilterHandle filter, int freq = 64, Allocator alloc = {});

    PBRT_HOST_DEVICE_INLINE
    FilterSample Sample(const Point2f &u) const {
        Point2f p = distrib.SampleContinuous(u);
        Point2f p01 = Point2f(domain.Offset(p));
        Point2i pi(Clamp(p01.x * values.xSize() + 0.5f, 0, values.xSize() - 1),
                   Clamp(p01.y * values.ySize() + 0.5f, 0, values.ySize() - 1));
        return { p, values[pi] < 0 ? -1.f : 1.f };
    }

    std::string ToString() const;

 private:
    Bounds2f domain;
    Array2D<Float> values;
    Distribution2D distrib;
};

// Box Filter Declarations
class FilterBase {
public:
    PBRT_HOST_DEVICE
    Vector2f Radius() const { return radius; }

protected:
    FilterBase(Vector2f radius) : radius(radius) { }
    Vector2f radius;
};

class BoxFilter : public FilterBase {
  public:
    BoxFilter(const Vector2f &radius = Vector2f(0.5, 0.5))
        : FilterBase(radius) {}

    static BoxFilter *Create(const ParameterDictionary &dict, const FileLoc *loc,
                             Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const {
        return (std::abs(p.x) <= radius.x && std::abs(p.y) <= radius.y) ? 1 : 0;
    }

    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const {
        return { Point2f(Lerp(u[0], -radius.x, radius.x),
                         Lerp(u[1], -radius.y, radius.y)), 1.f };
    }

    PBRT_HOST_DEVICE
    Float Integral() const {
        return 2 * radius.x * 2 * radius.y;
    }

    std::string ToString() const;
};

// Gaussian Filter Declarations
class GaussianFilter : public FilterBase {
  public:
    // GaussianFilter Public Methods
    GaussianFilter(const Vector2f &radius, Float sigma = 0.5f, Allocator alloc = {})
        : FilterBase(radius),
          sigma(sigma),
          expX(Gaussian(radius.x, 0, sigma)),
          expY(Gaussian(radius.y, 0, sigma)),
          sampler(this, 64, alloc) { }

    static GaussianFilter *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                  Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const {
        return (std::max<Float>(0, Gaussian(p.x, 0, sigma) - expX) *
                std::max<Float>(0, Gaussian(p.y, 0, sigma) - expY));
    }

    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const {
        return sampler.Sample(u);
    }

    PBRT_HOST_DEVICE
    Float Integral() const {
        return ((GaussianIntegral(-radius.x, radius.x, 0, sigma) - 2 * radius.x * expX) *
                (GaussianIntegral(-radius.y, radius.y, 0, sigma) - 2 * radius.y * expY));
    }

    std::string ToString() const;

  private:
    // GaussianFilter Private Data
    Float sigma;
    Float expX, expY;
    FilterSampler sampler;
};


// Mitchell Filter Declarations
class MitchellFilter : public FilterBase {
  public:
    // MitchellFilter Public Methods
    MitchellFilter(const Vector2f &radius, Float B = 1.f/3.f, Float C = 1.f/3.f)
        : FilterBase(radius), B(B), C(C), sampler(this) { }
    static MitchellFilter *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                  Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const {
        return Mitchell1D(p.x / radius.x) * Mitchell1D(p.y / radius.y);
    }

    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const {
        return sampler.Sample(u);
    }

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
class LanczosSincFilter : public FilterBase {
  public:
    // LanczosSincFilter Public Methods
    LanczosSincFilter(const Vector2f &radius, Float tau = 3.f)
        : FilterBase(radius), tau(tau), sampler(this) { }

    static LanczosSincFilter *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                     Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const {
        return WindowedSinc(p.x, radius.x, tau) * WindowedSinc(p.y, radius.y, tau);
    }

    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const {
        return sampler.Sample(u);
    }

    PBRT_HOST_DEVICE
    Float Integral() const;

    std::string ToString() const;

  private:
    Float tau;
    FilterSampler sampler;
};

// Triangle Filter Declarations
class TriangleFilter : public FilterBase {
  public:
    TriangleFilter(const Vector2f &radius) : FilterBase(radius) {}

    static TriangleFilter *Create(const ParameterDictionary &dict, const FileLoc *loc,
                                  Allocator alloc);

    PBRT_HOST_DEVICE
    Float Evaluate(const Point2f &p) const {
        return std::max<Float>(0, radius.x - std::abs(p.x)) *
            std::max<Float>(0, radius.y - std::abs(p.y));
    }

    PBRT_HOST_DEVICE
    FilterSample Sample(const Point2f &u) const {
        return { Point2f(SampleTent(u[0], radius.x),
                         SampleTent(u[1], radius.y)), 1.f };
    }

    PBRT_HOST_DEVICE
    Float Integral() const {
        return radius.x * radius.x * radius.y * radius.y;
    }

    std::string ToString() const;
};

inline Float FilterHandle::Evaluate(const Point2f &p) const {
    switch (Tag()) {
    case TypeIndex<BoxFilter>():
        return Cast<BoxFilter>()->Evaluate(p);
    case TypeIndex<GaussianFilter>():
        return Cast<GaussianFilter>()->Evaluate(p);
    case TypeIndex<MitchellFilter>():
        return Cast<MitchellFilter>()->Evaluate(p);
    case TypeIndex<LanczosSincFilter>():
        return Cast<LanczosSincFilter>()->Evaluate(p);
    case TypeIndex<TriangleFilter>():
        return Cast<TriangleFilter>()->Evaluate(p);
    default:
        LOG_FATAL("Unhandled Filter type");
        return {};
    }
}

inline FilterSample FilterHandle::Sample(const Point2f &u) const {
    switch (Tag()) {
    case TypeIndex<BoxFilter>():
        return Cast<BoxFilter>()->Sample(u);
    case TypeIndex<GaussianFilter>():
        return Cast<GaussianFilter>()->Sample(u);
    case TypeIndex<MitchellFilter>():
        return Cast<MitchellFilter>()->Sample(u);
    case TypeIndex<LanczosSincFilter>():
        return Cast<LanczosSincFilter>()->Sample(u);
    case TypeIndex<TriangleFilter>():
        return Cast<TriangleFilter>()->Sample(u);
    default:
        LOG_FATAL("Unhandled Filter type");
        return {};
    }
}

inline Vector2f FilterHandle::Radius() const {
    switch (Tag()) {
    case TypeIndex<BoxFilter>():
        return Cast<BoxFilter>()->Radius();
    case TypeIndex<GaussianFilter>():
        return Cast<GaussianFilter>()->Radius();
    case TypeIndex<MitchellFilter>():
        return Cast<MitchellFilter>()->Radius();
    case TypeIndex<LanczosSincFilter>():
        return Cast<LanczosSincFilter>()->Radius();
    case TypeIndex<TriangleFilter>():
        return Cast<TriangleFilter>()->Radius();
    default:
        LOG_FATAL("Unhandled Filter type");
        return {};
    }
}

inline Float FilterHandle::Integral() const {
    switch (Tag()) {
    case TypeIndex<BoxFilter>():
        return Cast<BoxFilter>()->Integral();
    case TypeIndex<GaussianFilter>():
        return Cast<GaussianFilter>()->Integral();
    case TypeIndex<MitchellFilter>():
        return Cast<MitchellFilter>()->Integral();
    case TypeIndex<LanczosSincFilter>():
        return Cast<LanczosSincFilter>()->Integral();
    case TypeIndex<TriangleFilter>():
        return Cast<TriangleFilter>()->Integral();
    default:
        LOG_FATAL("Unhandled Filter type");
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_FILTERS_H
