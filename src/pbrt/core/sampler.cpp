
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


// core/sampler.cpp*
#include <pbrt/core/sampler.h>

#include <pbrt/core/sampling.h>
#include <pbrt/util/stats.h>

#include <limits>

namespace pbrt {

// Sampler Method Definitions
Sampler::~Sampler() {}

Sampler::Sampler(int samplesPerPixel) : samplesPerPixel(samplesPerPixel) {
    currentPixel.x = currentPixel.y = std::numeric_limits<int>::lowest();
}

CameraSample Sampler::GetCameraSample(const Point2i &pRaster) {
    CameraSample cs;
    cs.pFilm = (Point2f)pRaster + Get2D();
    cs.time = Get1D();
    cs.pLens = Get2D();
    return cs;
}

void Sampler::StartSequence(const Point2i &p, int index) {
    CHECK_LT(index, samplesPerPixel);
    currentPixel = p;
    currentPixelSampleIndex = index;
}

PixelSampler::PixelSampler(int samplesPerPixel, int nSampledDimensions)
    : Sampler(samplesPerPixel) {
    for (int i = 0; i < nSampledDimensions; ++i) {
        samples1D.push_back(std::vector<Float>(samplesPerPixel));
        samples2D.push_back(std::vector<Point2f>(samplesPerPixel));
    }
}

void PixelSampler::StartSequence(const Point2i &p, int sampleIndex) {
    ProfilePhase _(Prof::StartSequence);

    current1DDimension = current2DDimension = 0;
    array1DOffset = array2DOffset = 0;

    if (p != currentPixel) {
        rng.SetSequence(p.x + 65536 * p.y);
        GeneratePixelSamples(rng);
    }

    // Always start at the begining of the sequence.
    rng.SetSequence(p.x + 65536 * p.y);
    // And now advance past the values used in the implementation of
    // GeneratePixelSamples().
    rng.Advance((sampleIndex + 1) * 16384);

    Sampler::StartSequence(p, sampleIndex);
}

Float PixelSampler::Get1D() {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    if (current1DDimension < samples1D.size())
        return samples1D[current1DDimension++][currentPixelSampleIndex];
    else
        return rng.Uniform<Float>();
}

Point2f PixelSampler::Get2D() {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    if (current2DDimension < samples2D.size())
        return samples2D[current2DDimension++][currentPixelSampleIndex];
    else
        return Point2f(rng.Uniform<Float>(), rng.Uniform<Float>());
}

void PixelSampler::Request1DArray(int n) {
    CHECK_EQ(RoundCount(n), n);
    samples1DArraySizes.push_back(n);
    sampleArray1D.push_back(std::vector<Float>(n * samplesPerPixel));
}

void PixelSampler::Request2DArray(int n) {
    CHECK_EQ(RoundCount(n), n);
    samples2DArraySizes.push_back(n);
    sampleArray2D.push_back(std::vector<Point2f>(n * samplesPerPixel));
}

absl::Span<const Float> PixelSampler::Get1DArray(int n) {
    ProfilePhase _(Prof::GetSample);
    if (array1DOffset == sampleArray1D.size()) return {};
    CHECK_EQ(samples1DArraySizes[array1DOffset], n);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    int offset = array1DOffset++;
    return {&sampleArray1D[offset][currentPixelSampleIndex * n],
            (size_t)samples1DArraySizes[offset]};
}

absl::Span<const Point2f> PixelSampler::Get2DArray(int n) {
    ProfilePhase _(Prof::GetSample);
    if (array2DOffset == sampleArray2D.size()) return {};
    CHECK_EQ(samples2DArraySizes[array2DOffset], n);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    int offset = array2DOffset++;
    return {&sampleArray2D[offset][currentPixelSampleIndex * n],
            (size_t)samples2DArraySizes[offset]};
}

void GlobalSampler::StartSequence(const Point2i &p, int sampleIndex) {
    ProfilePhase _(Prof::StartSequence);

    array1DOffset = array2DOffset = 0;

    Sampler::StartSequence(p, sampleIndex);

    dimension = 0;
    intervalSampleIndex = GetIndexForSample(sampleIndex);
}

void GlobalSampler::Request1DArray(int n) {
    CHECK_EQ(RoundCount(n), n);
    sampleArray1D.push_back(std::vector<Float>(n));
    ++arrayEndDim;
}

void GlobalSampler::Request2DArray(int n) {
    CHECK_EQ(RoundCount(n), n);
    sampleArray2D.push_back(std::vector<Point2f>(n));
    arrayEndDim += 2;
}

absl::Span<const Float> GlobalSampler::Get1DArray(int n) {
    ProfilePhase _(Prof::GetSample);
    if (array1DOffset == sampleArray1D.size()) return {};

    CHECK_EQ(n, sampleArray1D[array1DOffset].size());
    int firstSample = currentPixelSampleIndex * n;
    // Use the dimensions from arrayStartDim to arrayStartDim +
    // sampleArray1D.size() for the 1D arrays.
    int dim = arrayStartDim + array1DOffset;
    for (int i = 0; i < n; ++i) {
        int intervalSampleIndex = GetIndexForSample(firstSample + i);
        sampleArray1D[array1DOffset][i] = SampleDimension(intervalSampleIndex, dim);
    }
    return sampleArray1D[array1DOffset++];
}

absl::Span<const Point2f> GlobalSampler::Get2DArray(int n) {
    ProfilePhase _(Prof::GetSample);
    if (array2DOffset == sampleArray2D.size()) return {};

    CHECK_EQ(n, sampleArray2D[array2DOffset].size());

    int firstSample = currentPixelSampleIndex * n;
    // The dimensions for 2D samples are all taken after the dimensions
    // used for 1D array samples.
    int dim = arrayStartDim + sampleArray1D.size() + 2 * array2DOffset;
    for (int i = 0; i < n; ++i) {
        int invervalSampleIndex = GetIndexForSample(firstSample + i);
        sampleArray2D[array2DOffset][i].x = SampleDimension(invervalSampleIndex, dim);
        sampleArray2D[array2DOffset][i].y = SampleDimension(invervalSampleIndex, dim + 1);
    }
    return sampleArray2D[array2DOffset++];
}

Float GlobalSampler::Get1D() {
    ProfilePhase _(Prof::GetSample);
    if (dimension >= arrayStartDim && dimension < arrayEndDim)
        dimension = arrayEndDim;
    return SampleDimension(intervalSampleIndex, dimension++);
}

Point2f GlobalSampler::Get2D() {
    ProfilePhase _(Prof::GetSample);
    if (dimension + 1 >= arrayStartDim && dimension < arrayEndDim)
        dimension = arrayEndDim;
    Point2f p(SampleDimension(intervalSampleIndex, dimension),
              SampleDimension(intervalSampleIndex, dimension + 1));
    dimension += 2;
    return p;
}

}  // namespace pbrt
