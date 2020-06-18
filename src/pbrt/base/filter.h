// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_FILTER_H
#define PBRT_BASE_FILTER_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

// Filter Declarations
struct FilterSample;
class BoxFilter;
class GaussianFilter;
class MitchellFilter;
class LanczosSincFilter;
class TriangleFilter;

class FilterHandle : public TaggedPointer<BoxFilter, GaussianFilter, MitchellFilter,
                                          LanczosSincFilter, TriangleFilter> {
  public:
    using TaggedPointer::TaggedPointer;

    static FilterHandle Create(const std::string &name,
                               const ParameterDictionary &parameters, const FileLoc *loc,
                               Allocator alloc);

    PBRT_CPU_GPU inline Float Evaluate(const Point2f &p) const;

    PBRT_CPU_GPU inline FilterSample Sample(const Point2f &u) const;

    PBRT_CPU_GPU inline Vector2f Radius() const;

    PBRT_CPU_GPU inline Float Integral() const;

    std::string ToString() const;
};

}  // namespace pbrt

#endif  // PBRT_BASE_FILTER_H
