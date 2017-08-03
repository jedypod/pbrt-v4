
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

// samplers/random.cpp*
#include "samplers/random.h"
#include "paramset.h"
#include "sampling.h"
#include "stats.h"

namespace pbrt {

RandomSampler::RandomSampler(int ns) : Sampler(ns) {}

Float RandomSampler::Get1D() {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    return rng.UniformFloat();
}

Point2f RandomSampler::Get2D() {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(currentPixelSampleIndex, samplesPerPixel);
    return {rng.UniformFloat(), rng.UniformFloat()};
}

std::unique_ptr<Sampler> RandomSampler::Clone() {
    return std::make_unique<RandomSampler>(*this);
}

void RandomSampler::StartSequence(const Point2i &p, int sampleIndex) {
    ProfilePhase _(Prof::StartSequence);

    rng.SetSequence(p.x + p.y * 65536);
    // Assume we won't use more than 64k sample dimensions in a pixel...
    rng.Advance(sampleIndex * 65536);

    array1DOffset = array2DOffset = 0;

    Sampler::StartSequence(p, sampleIndex);
}

void RandomSampler::Request1DArray(int n) {
    sampleArray1D.push_back(std::vector<Float>(n));
}

void RandomSampler::Request2DArray(int n) {
    sampleArray2D.push_back(std::vector<Point2f>(n));
}

gtl::ArraySlice<Float> RandomSampler::Get1DArray(int n) {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(array1DOffset, sampleArray1D.size());
    for (int i = 0; i < n; ++i)
        sampleArray1D[array1DOffset][i] = rng.UniformFloat();
    return sampleArray1D[array1DOffset++];
}

gtl::ArraySlice<Point2f> RandomSampler::Get2DArray(int n) {
    ProfilePhase _(Prof::GetSample);
    CHECK_LT(array2DOffset, sampleArray2D.size());
    for (int i = 0; i < n; ++i) {
        sampleArray2D[array2DOffset][i].x = rng.UniformFloat();
        sampleArray2D[array2DOffset][i].y = rng.UniformFloat();
    }
    return sampleArray2D[array2DOffset++];
}

std::unique_ptr<RandomSampler> CreateRandomSampler(const ParamSet &params) {
    int ns = params.GetOneInt("pixelsamples", 4);
    return std::make_unique<RandomSampler>(ns);
}

}  // namespace pbrt
