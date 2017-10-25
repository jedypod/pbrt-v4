
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

// samplers/stratified.cpp*
#include <pbrt/samplers/stratified.h>
#include <pbrt/core/paramset.h>
#include <pbrt/core/sampling.h>

namespace pbrt {

// StratifiedSampler Method Definitions
void StratifiedSampler::GeneratePixelSamples(RNG &rng) {
    // Generate single stratified samples for the pixel
    for (size_t i = 0; i < samples1D.size(); ++i) {
        StratifiedSample1D(absl::Span<Float>(samples1D[i]), rng, jitterSamples);
        Shuffle<Float>(absl::Span<Float>(samples1D[i]), 1, rng);
    }
    for (size_t i = 0; i < samples2D.size(); ++i) {
        StratifiedSample2D(absl::Span<Point2f>(samples2D[i]), xPixelSamples,
                           yPixelSamples, rng, jitterSamples);
        Shuffle<Point2f>(absl::Span<Point2f>(samples2D[i]), 1, rng);
    }

    // Generate arrays of stratified samples for the pixel
    for (size_t i = 0; i < samples1DArraySizes.size(); ++i)
        for (int64_t j = 0; j < samplesPerPixel; ++j) {
            int count = samples1DArraySizes[i];
            absl::Span<Float> samples(&sampleArray1D[i][j * count], count);
            StratifiedSample1D(samples, rng, jitterSamples);
            Shuffle(samples, 1, rng);
        }
    for (size_t i = 0; i < samples2DArraySizes.size(); ++i)
        for (int64_t j = 0; j < samplesPerPixel; ++j) {
            int count = samples2DArraySizes[i];
            absl::Span<Float> samples(&sampleArray2D[i][j * count].x, 2 * count);
            LatinHypercube(samples, 2, rng);
        }
}

std::unique_ptr<Sampler> StratifiedSampler::Clone() {
    return std::make_unique<StratifiedSampler>(*this);
}

std::unique_ptr<StratifiedSampler> CreateStratifiedSampler(
    const ParamSet &params) {
    bool jitter = params.GetOneBool("jitter", true);
    int xsamp = params.GetOneInt("xsamples", 4);
    int ysamp = params.GetOneInt("ysamples", 4);
    int sd = params.GetOneInt("dimensions", 4);
    if (PbrtOptions.quickRender) xsamp = ysamp = 1;
    return std::make_unique<StratifiedSampler>(xsamp, ysamp, jitter, sd);
}

}  // namespace pbrt
