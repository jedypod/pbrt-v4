
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

// samplers.cpp*
#include <pbrt/samplers.h>

#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/options.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/error.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pmj02tables.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/shuffle.h>


namespace pbrt {

// HaltonSampler Method Definitions
HaltonSampler::HaltonSampler(int samplesPerPixel, const Point2i &fullResolution, int seed)
    : Sampler(samplesPerPixel), haltonPixelIndexer(fullResolution) {
    // Generate random digit permutations for Halton sampler
    digitPermutations = ComputeRadicalInversePermutations(seed);
    ownDigitPermutations = true;
    digitPermutationsSeed = seed;
}

HaltonSampler::~HaltonSampler() {
    if (ownDigitPermutations)
        digitPermutations->get_allocator().delete_object(digitPermutations);
}

void HaltonSampler::ImplStartPixelSample(const Point2i &p, int pixelSample) {
    if (p != pixelForIndex) {
        // Compute Halton sample offset for _currentPixel_
        haltonPixelIndexer.SetPixel(p);
        pixelForIndex = p;
    }
    haltonPixelIndexer.SetPixelSample(pixelSample);

    rng.SetSequence(p.x + p.y * 65536);
    rng.Advance(pixelSample * 65536);
}

Float HaltonSampler::ImplGet1D(const SamplerState &s) {
    if (s.dimension >= PrimeTableSize)
        return rng.Uniform<Float>();

    // Note (for book): it's about 5x faster to use precomputed tables than
    // to generate the permutations on the fly...
    // TODO: check on GPU though...
    return ScrambledRadicalInverse(s.dimension, haltonPixelIndexer.SampleIndex(),
                                   (*digitPermutations)[s.dimension]);
}

Point2f HaltonSampler::ImplGet2D(const SamplerState &s) {
    if (s.dimension == 0)
        return haltonPixelIndexer.SampleFirst2D();
    else {
        if (s.dimension + 1 >= PrimeTableSize)
            return {rng.Uniform<Float>(), rng.Uniform<Float>()};

        return {ScrambledRadicalInverse(s.dimension, haltonPixelIndexer.SampleIndex(),
                                        (*digitPermutations)[s.dimension]),
                ScrambledRadicalInverse(s.dimension + 1, haltonPixelIndexer.SampleIndex(),
                                        (*digitPermutations)[s.dimension + 1])};
    }
}

std::unique_ptr<Sampler> HaltonSampler::Clone() {
    std::unique_ptr<HaltonSampler> s = std::make_unique<HaltonSampler>(*this);
    s->ownDigitPermutations = false;
    return s;
}

std::string HaltonSampler::ToString() const {
    return StringPrintf("[ HaltonSampler %s digitPermutations: (elided) "
                        "digitPermutationsSeed: %d ownDigitPermutations: %s "
                        "haltonPixelIndexer: %s pixelForIndex: %s rng: %s ]",
                        BaseToString(), digitPermutationsSeed, ownDigitPermutations,
                        haltonPixelIndexer, pixelForIndex, rng);
}

std::unique_ptr<HaltonSampler> HaltonSampler::Create(
    const ParameterDictionary &dict, const Point2i &fullResolution) {
    int nsamp = dict.GetOneInt("pixelsamples", 16);
    if (PbrtOptions.pixelSamples)
        nsamp = *PbrtOptions.pixelSamples;
    int seed = dict.GetOneInt("seed", PbrtOptions.seed);
    if (PbrtOptions.quickRender) nsamp = 1;
    return std::make_unique<HaltonSampler>(nsamp, fullResolution, seed);
}


// PaddedSobolSampler Method Definitions
PaddedSobolSampler::PaddedSobolSampler(int samplesPerPixel, RandomizeStrategy randomizer)
    : Sampler(RoundUpPow2(samplesPerPixel)), randomizeStrategy(randomizer) {
    if (!IsPowerOf2(samplesPerPixel))
        Warning(
            "Pixel samples being rounded up to power of 2 (from %d to %d).",
            samplesPerPixel, RoundUpPow2(samplesPerPixel));
}

Float PaddedSobolSampler::ImplGet1D(const SamplerState &s) {
    uint64_t hash = s.Hash();
    int index = PermutationElement(s.sampleIndex, samplesPerPixel, hash);

    if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
        return SampleGeneratorMatrix(CVanDerCorput, index,
                                     CranleyPattersonRotator(BlueNoise(s.dimension, s.p.x, s.p.y)));
    else
        return generateSample(CVanDerCorput, index, hash >> 32);
}

Point2f PaddedSobolSampler::ImplGet2D(const SamplerState &s) {
    uint64_t hash = s.Hash();
    int index = PermutationElement(s.sampleIndex, samplesPerPixel, hash);

    if (randomizeStrategy == RandomizeStrategy::CranleyPatterson)
        return {SampleGeneratorMatrix(CSobol[0], index,
                                      CranleyPattersonRotator(BlueNoise(s.dimension, s.p.x, s.p.y))),
                SampleGeneratorMatrix(CSobol[1], index,
                                      CranleyPattersonRotator(BlueNoise(s.dimension + 1, s.p.x, s.p.y)))};
    else
        // Note: we're reusing the low 32 bits of the hash both for the permutation and
        // for the random scrambling in the first dimension. This should(?) be fine.
        return {generateSample(CSobol[0], index, hash >> 8), generateSample(CSobol[1], index, hash >> 32)};
}

std::string PaddedSobolSampler::ToString() const {
    return StringPrintf("[ PaddedSobolSampler %s randomizeStrategy: %s ]",
                        BaseToString(), randomizeStrategy);
}

std::unique_ptr<Sampler> PaddedSobolSampler::Clone() {
    return std::make_unique<PaddedSobolSampler>(*this);
}

std::unique_ptr<PaddedSobolSampler> PaddedSobolSampler::Create(
    const ParameterDictionary &dict) {
    int nsamp = dict.GetOneInt("pixelsamples", 16);
    if (PbrtOptions.pixelSamples)
        nsamp = *PbrtOptions.pixelSamples;
    if (PbrtOptions.quickRender) nsamp = 1;

    RandomizeStrategy randomizer;
    std::string s = dict.GetOneString("randomization",
                                      nsamp <= 2 ? "cranleypatterson" : "owen");
    if (s == "none")
        randomizer = RandomizeStrategy::None;
    else if (s == "cranleypatterson")
        randomizer = RandomizeStrategy::CranleyPatterson;
    else if (s == "xor")
        randomizer = RandomizeStrategy::Xor;
    else if (s == "owen")
        randomizer = RandomizeStrategy::Owen;
    else
        ErrorExit("%s: unknown randomization strategy given to PaddedSobolSampler", s);

    return std::make_unique<PaddedSobolSampler>(nsamp, randomizer);
}


/* notes:
- generally wins (in part due to blue noise at low sampling densities)
  - less clear with volumetrics
- big win on bdpt contemporary bathroom?
  - versus halton, totally. why is halton so bad here?
    - is it that the camera path gets all the good early dimensions??
    -> halton is probably a definitely wrong default. leaning for pmj02bn...
  - sobol is close (do a rigorous check with a reference image)
- worse on sibenik-gonio-bdpt
  - why? would be great to explain and discuss
*/

PMJ02BNSampler::PMJ02BNSampler(int samplesPerPixel, Allocator alloc)
    : Sampler(samplesPerPixel) {
    if (!IsPowerOf4(samplesPerPixel))
        Warning("PMJ02BNSampler results are best with power-of-4 samples per pixel (1, 4, 16, 64, ...)");

    pixelSamples = GetSortedPMJ02BNPixelSamples(samplesPerPixel, alloc, &pixelTileSize);
}

void PMJ02BNSampler::ImplStartPixelSample(const Point2i &p, int pixelSample) {
    pmjInstance = 0;
}

Float PMJ02BNSampler::ImplGet1D(const SamplerState &s) {
    uint64_t hash = s.Hash();
    int index = PermutationElement(s.sampleIndex, samplesPerPixel, hash);
    Float cpOffset = BlueNoise(s.dimension, s.p.x, s.p.y);
    Float u = (index + cpOffset) / samplesPerPixel;
    if (u >= 1) u -= 1;
    return std::min(u, OneMinusEpsilon);
}

Point2f PMJ02BNSampler::ImplGet2D(const SamplerState &s) {
    // Don't start permuting until the second time through: when we
    // permute, that breaks the progressive part of the pattern and in
    // turn, convergence is similar to random until the very end. This way,
    // we generally do well for intermediate images as well.
    int sampleIndex = (pmjInstance >= nPMJ02bnSets) ?
        PermutationElement(s.sampleIndex, samplesPerPixel, s.Hash()) :
        s.sampleIndex;
    if (s.dimension == 0) {
        // special case the pixel sample
        int offset = pixelSampleOffset(Point2i(s.p));
        return (*pixelSamples)[offset + sampleIndex];
    } else {
        Vector2f cpOffset(BlueNoise(s.dimension, s.p.x, s.p.y),
                          BlueNoise(s.dimension + 1, s.p.x, s.p.y));
        Point2f u = GetPMJ02BNSample(pmjInstance++, sampleIndex) + cpOffset;
        if (u.x >= 1) u.x -= 1;
        if (u.y >= 1) u.y -= 1;
        return {std::min(u.x, OneMinusEpsilon), std::min(u.y, OneMinusEpsilon)};
    }
}

std::unique_ptr<Sampler> PMJ02BNSampler::Clone() {
    return std::make_unique<PMJ02BNSampler>(*this);
}

std::string PMJ02BNSampler::ToString() const {
    return StringPrintf("[ PMJ02BNSampler %s pixelTileSize: %d pmjInstance: %d "
                        " pixelSamples: (elided)]",
                        BaseToString(), pixelTileSize, pmjInstance);
}

std::unique_ptr<PMJ02BNSampler> PMJ02BNSampler::Create(const ParameterDictionary &dict) {
    int nsamp = dict.GetOneInt("pixelsamples", 16);
    if (PbrtOptions.pixelSamples)
        nsamp = *PbrtOptions.pixelSamples;
    if (PbrtOptions.quickRender) nsamp = 1;
    return std::make_unique<PMJ02BNSampler>(nsamp);
}


RandomSampler::RandomSampler(int ns, int seed) : Sampler(ns), seed(seed) {}

Float RandomSampler::ImplGet1D(const SamplerState &s) {
    return rng.Uniform<Float>();
}

Point2f RandomSampler::ImplGet2D(const SamplerState &s) {
    return {rng.Uniform<Float>(), rng.Uniform<Float>()};
}

void RandomSampler::ImplStartPixelSample(const Point2i &p, int pixelSample) {
    rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
    // Assume we won't use more than 64k sample dimensions in a pixel...
    rng.Advance(pixelSample * 65536);
}

std::string RandomSampler::ToString() const {
    return StringPrintf("[ RandomSampler %s seed: %d rng: %s ]",
                        BaseToString(), seed, rng);
}

std::unique_ptr<Sampler> RandomSampler::Clone() {
    return std::make_unique<RandomSampler>(*this);
}

std::unique_ptr<RandomSampler> RandomSampler::Create(const ParameterDictionary &dict) {
    int ns = dict.GetOneInt("pixelsamples", 4);
    if (PbrtOptions.pixelSamples)
        ns = *PbrtOptions.pixelSamples;
    int seed = dict.GetOneInt("seed", PbrtOptions.seed);
    return std::make_unique<RandomSampler>(ns, seed);
}


// SobolSampler Method Definitions
void SobolSampler::ImplStartPixelSample(const Point2i &p, int pixelSample) {
    if (p != pixelForIndex || pixelSample != pixelSampleForIndex) {
        sequenceIndex = SobolIntervalToIndex(log2Resolution, pixelSample, p);
        pixelForIndex = p;
        pixelSampleForIndex = pixelSample;
    }
    rng.SetSequence(p.x + p.y * 65536);
    rng.Advance(pixelSample * 65536);
}

Float SobolSampler::ImplGet1D(const SamplerState &s) {
    if (s.dimension >= NSobolDimensions)
        return rng.Uniform<Float>();

    return sampleDimension(s.dimension);
}

Point2f SobolSampler::ImplGet2D(const SamplerState &s) {
    if (s.dimension + 1 >= NSobolDimensions)
        return {rng.Uniform<Float>(), rng.Uniform<Float>()};

    Point2f u = {sampleDimension(s.dimension), sampleDimension(s.dimension + 1)};

    if (s.dimension == 0) {
        // Remap Sobol$'$ dimensions used for pixel samples
        for (int dim = 0; dim < 2; ++dim) {
            u[dim] = u[dim] * resolution;
            CHECK_RARE(1e-7, u[dim] - s.p[dim] < 0);
            CHECK_RARE(1e-7, u[dim] - s.p[dim] > 1);
            u[dim] = Clamp(u[dim] - s.p[dim], (Float)0, OneMinusEpsilon);
        }
    }
    return u;
}

std::unique_ptr<Sampler> SobolSampler::Clone() {
    return std::make_unique<SobolSampler>(*this);
}

std::string SobolSampler::ToString() const {
    return StringPrintf("[ SobolSampler %s fullResolution: %s resolution: %d "
                        "log2Resolution: %d pixelForIndex: %s pixelSampleForIndex: %d "
                        "sequenceIndex: %d rng: %s randomizeStrategy: %s]",
                        BaseToString(), fullResolution, resolution, log2Resolution,
                        pixelForIndex, pixelSampleForIndex, sequenceIndex, rng,
                        randomizeStrategy);
}

std::unique_ptr<SobolSampler> SobolSampler::Create(const ParameterDictionary &dict,
                                                   const Point2i &fullResolution) {
    int nsamp = dict.GetOneInt("pixelsamples", 16);
    if (PbrtOptions.pixelSamples)
        nsamp = *PbrtOptions.pixelSamples;
    if (PbrtOptions.quickRender) nsamp = 1;

    RandomizeStrategy randomizer;
    std::string s = dict.GetOneString("randomization", "owen");
    if (s == "none")
        randomizer = RandomizeStrategy::None;
    else if (s == "cranleypatterson")
        randomizer = RandomizeStrategy::CranleyPatterson;
    else if (s == "xor")
        randomizer = RandomizeStrategy::Xor;
    else if (s == "owen")
        randomizer = RandomizeStrategy::Owen;
    else
        ErrorExit("%s: unknown randomization strategy given to SobolSampler", s);

    return std::make_unique<SobolSampler>(nsamp, fullResolution, randomizer);
}


// StratifiedSampler Method Definitions
void StratifiedSampler::ImplStartPixelSample(const Point2i &p, int pixelSample) {
    rng.SetSequence((p.x + p.y * 65536) | (uint64_t(seed) << 32));
    // Assume we won't use more than 64k sample dimensions in a pixel...
    rng.Advance(pixelSample * 65536);
}

Float StratifiedSampler::ImplGet1D(const SamplerState &s) {
    int stratum = PermutationElement(s.sampleIndex, samplesPerPixel, s.Hash());
    Float delta = jitter ? rng.Uniform<Float>() : 0.5f;
    return (stratum + delta) / samplesPerPixel;
}

Point2f StratifiedSampler::ImplGet2D(const SamplerState &s) {
    int stratum = PermutationElement(s.sampleIndex, samplesPerPixel, s.Hash());
    int x = stratum % xPixelSamples;
    int y = stratum / xPixelSamples;
    Float dx = jitter ? rng.Uniform<Float>() : 0.5f;
    Float dy = jitter ? rng.Uniform<Float>() : 0.5f;
    return {(x + dx) / xPixelSamples, (y + dy) / yPixelSamples};
}

std::string StratifiedSampler::ToString() const {
    return StringPrintf("[ StratifiedSampler %s xPixelSamples: %d "
                        "yPixelSamples: %d jitter: %s rng: %s ]",
                        BaseToString(), xPixelSamples, yPixelSamples, jitter,
                        rng);
}

std::unique_ptr<Sampler> StratifiedSampler::Clone() {
    return std::make_unique<StratifiedSampler>(*this);
}

std::unique_ptr<StratifiedSampler> StratifiedSampler::Create(
    const ParameterDictionary &dict) {
    bool jitter = dict.GetOneBool("jitter", true);
    int xsamp = dict.GetOneInt("xsamples", 4);
    int ysamp = dict.GetOneInt("ysamples", 4);
    if (PbrtOptions.quickRender) xsamp = ysamp = 1;
    int seed = dict.GetOneInt("seed", PbrtOptions.seed);
    return std::make_unique<StratifiedSampler>(xsamp, ysamp, jitter, seed);
}

}  // namespace pbrt
