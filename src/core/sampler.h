
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

#ifndef PBRT_CORE_SAMPLER_H
#define PBRT_CORE_SAMPLER_H

// core/sampler.h*
#include "pbrt.h"

#include "camera.h"
#include "geometry.h"
#include "rng.h"
#include "stringprint.h"
#include "ext/google/array_slice.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// Sampler Declarations
class Sampler {
  public:
    // Sampler Interface
    virtual ~Sampler();
    Sampler(int samplesPerPixel);
    virtual void StartSequence(const Point2i &p, int sampleIndex);
    virtual Float Get1D() = 0;
    virtual Point2f Get2D() = 0;
    CameraSample GetCameraSample(const Point2i &pRaster);
    virtual int RoundCount(int n) const { return n; }
    virtual void Request1DArray(int n) = 0;
    virtual void Request2DArray(int n) = 0;
    virtual gtl::ArraySlice<Float> Get1DArray(int n) = 0;
    virtual gtl::ArraySlice<Point2f> Get2DArray(int n) = 0;
    virtual std::unique_ptr<Sampler> Clone() = 0;

    int GetDiscrete1D(int n) {
        return std::min(int(Get1D() * n), n - 1);
    }
    // Sampler Public Data
    const int samplesPerPixel;

  protected:
    // Sampler Protected Data
    Point2i currentPixel;
    int64_t currentPixelSampleIndex;
};

class PixelSampler : public Sampler {
  public:
    // PixelSampler Public Methods
    PixelSampler(int samplesPerPixel, int nSampledDimensions);
    void StartSequence(const Point2i &p, int sampleIndex) final;
    Float Get1D() final;
    Point2f Get2D() final;
    void Request1DArray(int n) final;
    void Request2DArray(int n) final;
    gtl::ArraySlice<Float> Get1DArray(int n) final;
    gtl::ArraySlice<Point2f> Get2DArray(int n) final;

    virtual void GeneratePixelSamples(RNG &rng) = 0;

  protected:
    // PixelSampler Protected Data
    std::vector<std::vector<Float>> samples1D;
    std::vector<std::vector<Point2f>> samples2D;

    std::vector<int> samples1DArraySizes, samples2DArraySizes;
    std::vector<std::vector<Float>> sampleArray1D;
    std::vector<std::vector<Point2f>> sampleArray2D;

  private:
    int current1DDimension = 0, current2DDimension = 0;
    RNG rng;
    size_t array1DOffset, array2DOffset;
};

class GlobalSampler : public Sampler {
  public:
    // GlobalSampler Public Methods
    void StartSequence(const Point2i &p, int sampleIndex) final;
    Float Get1D() final;
    Point2f Get2D() final;
    GlobalSampler(int samplesPerPixel) : Sampler(samplesPerPixel) {}
    void Request1DArray(int n) final;
    void Request2DArray(int n) final;
    gtl::ArraySlice<Float> Get1DArray(int n) final;
    gtl::ArraySlice<Point2f> Get2DArray(int n) final;

    virtual int64_t GetIndexForSample(int64_t sampleNum) const = 0;
    virtual Float SampleDimension(int64_t index, int dimension) const = 0;

  private:
    // GlobalSampler Private Data
    int dimension;
    int64_t intervalSampleIndex;

    static const int arrayStartDim = 5;
    int arrayEndDim = arrayStartDim;
    // Offsets into sampleArray1D and sampleArray2D, respectively, as
    // we consume sample arrays for the current pixel sample.
    size_t array1DOffset, array2DOffset;
    // Note: we need separate vectors for each requested array since
    // callers may want to have multiple of them outstanding at the same
    // time.
    std::vector<std::vector<Float>> sampleArray1D;
    std::vector<std::vector<Point2f>> sampleArray2D;
};

}  // namespace pbrt

#endif  // PBRT_CORE_SAMPLER_H
