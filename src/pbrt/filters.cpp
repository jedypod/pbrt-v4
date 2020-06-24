
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


// filters.cpp*
#include <pbrt/filters.h>

#include <pbrt/util/math.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>

#include <cmath>

namespace pbrt {

// Box Filter Method Definitions
Float BoxFilter::Evaluate(const Point2f &p) const {
    return (std::abs(p.x) <= radius.x && std::abs(p.y) <= radius.y) ? 1 : 0;
}

FilterSample BoxFilter::Sample(const Point2f &u) const {
    return { Point2f(Lerp(u[0], -radius.x, radius.x),
                     Lerp(u[1], -radius.y, radius.y)), 1.f };
}

std::string BoxFilter::ToString() const {
    return StringPrintf("[ BoxFilter radius: %s ]", radius);
}

BoxFilter *BoxFilter::Create(const ParameterDictionary &dict, Allocator alloc) {
    Float xw = dict.GetOneFloat("xradius", 0.5f);
    Float yw = dict.GetOneFloat("yradius", 0.5f);
    return alloc.new_object<BoxFilter>(Vector2f(xw, yw));
}

// Gaussian Filter Method Definitions
Float GaussianFilter::Evaluate(const Point2f &p) const {
    return (std::max<Float>(0, Gaussian(p.x, 0, sigma) - expX) *
            std::max<Float>(0, Gaussian(p.y, 0, sigma) - expY));
}

Float GaussianFilter::Integral() const {
    return ((GaussianIntegral(-radius.x, radius.x, 0, sigma) - 2 * radius.x * expX) *
            (GaussianIntegral(-radius.y, radius.y, 0, sigma) - 2 * radius.y * expY));
}

std::string GaussianFilter::ToString() const {
    return StringPrintf("[ GaussianFilter radius: %s sigma: %f expX: %f expY: %f sampler: %s ]",
                        radius, sigma, expX, expY, sampler);
}

GaussianFilter *GaussianFilter::Create(const ParameterDictionary &dict, Allocator alloc) {
    // Find common filter parameters
    Float xw = dict.GetOneFloat("xradius", 1.5f);
    Float yw = dict.GetOneFloat("yradius", 1.5f);
    Float sigma = dict.GetOneFloat("sigma", 0.5f);  // equivalent to old alpha = 2
    return alloc.new_object<GaussianFilter>(Vector2f(xw, yw), sigma);
}

// Mitchell Filter Method Definitions
Float MitchellFilter::Evaluate(const Point2f &p) const {
    return Mitchell1D(p.x / radius.x) * Mitchell1D(p.y / radius.y);
}

std::string MitchellFilter::ToString() const {
    return StringPrintf("[ MitchellFilter radius: %s B: %f C: %f sampler: %s ]",
                        radius, B, C, sampler);
}

MitchellFilter *MitchellFilter::Create(const ParameterDictionary &dict, Allocator alloc) {
    // Find common filter parameters
    Float xw = dict.GetOneFloat("xradius", 2.f);
    Float yw = dict.GetOneFloat("yradius", 2.f);
    Float B = dict.GetOneFloat("B", 1.f / 3.f);
    Float C = dict.GetOneFloat("C", 1.f / 3.f);
    return alloc.new_object<MitchellFilter>(Vector2f(xw, yw), B, C);
}

// Sinc Filter Method Definitions
Float LanczosSincFilter::Evaluate(const Point2f &p) const {
    return WindowedSinc(p.x, radius.x, tau) * WindowedSinc(p.y, radius.y, tau);
}

std::string LanczosSincFilter::ToString() const {
    return StringPrintf("[ LanczosSincFilter radius: %s tau: %f sampler: %s ]",
                        radius, tau, sampler);
}

LanczosSincFilter *LanczosSincFilter::Create(const ParameterDictionary &dict, Allocator alloc) {
    Float xw = dict.GetOneFloat("xradius", 4.);
    Float yw = dict.GetOneFloat("yradius", 4.);
    Float tau = dict.GetOneFloat("tau", 3.f);
    return alloc.new_object<LanczosSincFilter>(Vector2f(xw, yw), tau);
}

// Triangle Filter Method Definitions
Float TriangleFilter::Evaluate(const Point2f &p) const {
    return std::max<Float>(0, radius.x - std::abs(p.x)) *
           std::max<Float>(0, radius.y - std::abs(p.y));
}

FilterSample TriangleFilter::Sample(const Point2f &u) const {
    return { Point2f(SampleTent(u[0], radius.x),
                     SampleTent(u[1], radius.y)), 1.f };
}

std::string TriangleFilter::ToString() const {
    return StringPrintf("[ TriangleFilter radius: %s ]", radius);
}

TriangleFilter *TriangleFilter::Create(const ParameterDictionary &dict, Allocator alloc) {
    // Find common filter parameters
    Float xw = dict.GetOneFloat("xradius", 2.f);
    Float yw = dict.GetOneFloat("yradius", 2.f);
    return alloc.new_object<TriangleFilter>(Vector2f(xw, yw));
}

Filter *Filter::Create(const std::string &name,
                       const ParameterDictionary &dict,
                       const FileLoc *loc, Allocator alloc) {
    Filter *filter = nullptr;
    if (name == "box")
        filter = BoxFilter::Create(dict, alloc);
    else if (name == "gaussian")
        filter = GaussianFilter::Create(dict, alloc);
    else if (name == "mitchell")
        filter = MitchellFilter::Create(dict, alloc);
    else if (name == "sinc")
        filter = LanczosSincFilter::Create(dict, alloc);
    else if (name == "triangle")
        filter = TriangleFilter::Create(dict, alloc);
    else
        ErrorExit(loc, "%s: filter type unknown.", name);

    if (!filter)
        ErrorExit(loc, "%s: unable to create filter.", name);

    dict.ReportUnused();
    return filter;
}

}  // namespace pbrt
