
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

#include <pbrt/paramdict.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>

namespace pbrt {

std::string FilterHandle::ToString() const {
    switch (Tag()) {
    case TypeIndex<BoxFilter>():
        return Cast<BoxFilter>()->ToString();
    case TypeIndex<GaussianFilter>():
        return Cast<GaussianFilter>()->ToString();
    case TypeIndex<MitchellFilter>():
        return Cast<MitchellFilter>()->ToString();
    case TypeIndex<LanczosSincFilter>():
        return Cast<LanczosSincFilter>()->ToString();
    case TypeIndex<TriangleFilter>():
        return Cast<TriangleFilter>()->ToString();
    default:
        LOG_FATAL("Unhandled Filter type");
        return {};
    }
}

// Box Filter Method Definitions
std::string BoxFilter::ToString() const {
    return StringPrintf("[ BoxFilter radius: %s ]", radius);
}

BoxFilter *BoxFilter::Create(const ParameterDictionary &dict, const FileLoc *loc,
                             Allocator alloc) {
    Float xw = dict.GetOneFloat("xradius", 0.5f);
    Float yw = dict.GetOneFloat("yradius", 0.5f);
    return alloc.new_object<BoxFilter>(Vector2f(xw, yw));
}

// Gaussian Filter Method Definitions
std::string GaussianFilter::ToString() const {
    return StringPrintf("[ GaussianFilter radius: %s sigma: %f expX: %f expY: %f sampler: %s ]",
                        radius, sigma, expX, expY, sampler);
}

GaussianFilter *GaussianFilter::Create(const ParameterDictionary &dict, const FileLoc *loc,
                                       Allocator alloc) {
    // Find common filter parameters
    Float xw = dict.GetOneFloat("xradius", 1.5f);
    Float yw = dict.GetOneFloat("yradius", 1.5f);
    Float sigma = dict.GetOneFloat("sigma", 0.5f);  // equivalent to old alpha = 2
    return alloc.new_object<GaussianFilter>(Vector2f(xw, yw), sigma, alloc);
}

// Mitchell Filter Method Definitions
std::string MitchellFilter::ToString() const {
    return StringPrintf("[ MitchellFilter radius: %s B: %f C: %f sampler: %s ]",
                        radius, B, C, sampler);
}

MitchellFilter *MitchellFilter::Create(const ParameterDictionary &dict, const FileLoc *loc,
                                       Allocator alloc) {
    // Find common filter parameters
    Float xw = dict.GetOneFloat("xradius", 2.f);
    Float yw = dict.GetOneFloat("yradius", 2.f);
    Float B = dict.GetOneFloat("B", 1.f / 3.f);
    Float C = dict.GetOneFloat("C", 1.f / 3.f);
    return alloc.new_object<MitchellFilter>(Vector2f(xw, yw), B, C);
}

// Sinc Filter Method Definitions
Float LanczosSincFilter::Integral() const {
    Float sum = 0;
    int sqrtSamples = 64;
    int nSamples = sqrtSamples * sqrtSamples;
    Float area = 2 * radius.x * 2 * radius.y;
    RNG rng;
    for (int y = 0; y < sqrtSamples; ++y) {
        for (int x = 0; x < sqrtSamples; ++x) {
            Point2f u((x + rng.Uniform<Float>()) / sqrtSamples,
                      (y + rng.Uniform<Float>()) / sqrtSamples);
            Point2f p(Lerp(u.x, -radius.x, radius.x), Lerp(u.y, -radius.y, radius.y));
            sum += Evaluate(p);
        }
    }
    return sum / nSamples * area;
}

std::string LanczosSincFilter::ToString() const {
    return StringPrintf("[ LanczosSincFilter radius: %s tau: %f sampler: %s ]",
                        radius, tau, sampler);
}

LanczosSincFilter *LanczosSincFilter::Create(const ParameterDictionary &dict, const FileLoc *loc,
                                             Allocator alloc) {
    Float xw = dict.GetOneFloat("xradius", 4.);
    Float yw = dict.GetOneFloat("yradius", 4.);
    Float tau = dict.GetOneFloat("tau", 3.f);
    return alloc.new_object<LanczosSincFilter>(Vector2f(xw, yw), tau);
}

// Triangle Filter Method Definitions
std::string TriangleFilter::ToString() const {
    return StringPrintf("[ TriangleFilter radius: %s ]", radius);
}

TriangleFilter *TriangleFilter::Create(const ParameterDictionary &dict, const FileLoc *loc,
                                       Allocator alloc) {
    // Find common filter parameters
    Float xw = dict.GetOneFloat("xradius", 2.f);
    Float yw = dict.GetOneFloat("yradius", 2.f);
    return alloc.new_object<TriangleFilter>(Vector2f(xw, yw));
}

FilterHandle FilterHandle::Create(const std::string &name,
                                  const ParameterDictionary &dict,
                                  const FileLoc *loc, Allocator alloc) {
    FilterHandle filter = nullptr;
    if (name == "box")
        filter = BoxFilter::Create(dict, loc, alloc);
    else if (name == "gaussian")
        filter = GaussianFilter::Create(dict, loc, alloc);
    else if (name == "mitchell")
        filter = MitchellFilter::Create(dict, loc, alloc);
    else if (name == "sinc")
        filter = LanczosSincFilter::Create(dict, loc, alloc);
    else if (name == "triangle")
        filter = TriangleFilter::Create(dict, loc, alloc);
    else
        ErrorExit(loc, "%s: filter type unknown.", name);

    if (!filter)
        ErrorExit(loc, "%s: unable to create filter.", name);

    dict.ReportUnused();
    return filter;
}

FilterSampler::FilterSampler(FilterHandle filter, int freq, Allocator alloc)
    :  domain(Point2f(-filter.Radius()), Point2f(filter.Radius())),
       values(int(16 * 2 * filter.Radius().x),
              int(16 * 2 * filter.Radius().y), alloc),
       distrib(alloc) {
    for (int y = 0; y < values.ySize(); ++y) {
        for (int x = 0; x < values.xSize(); ++x) {
            Point2f p = domain.Lerp(Point2f((x + 0.5f) / values.xSize(),
                                            (y + 0.5f) / values.ySize()));
            values(x, y) = std::abs(filter.Evaluate(p));
        }
    }

    distrib = std::move(Distribution2D(values, domain, alloc));

    // And again without the abs() for use in Sample...
    for (int y = 0; y < values.ySize(); ++y) {
        for (int x = 0; x < values.xSize(); ++x) {
            Point2f p = domain.Lerp(Point2f((x + 0.5f) / values.xSize(),
                                            (y + 0.5f) / values.ySize()));
            values(x, y) = filter.Evaluate(p);
        }
    }
}

std::string FilterSampler::ToString() const {
#if 0
    return StringPrintf("[ FilterSampler domain: %s values: %s distrib: %s ]",
                        domian, values, distrib);
#endif
    return "[ FilterSampler TODO ]";
}

}  // namespace pbrt
