
#ifndef PBRT_SAMPLING_PMJ02TABLES_H
#define PBRT_SAMPLING_PMJ02TABLES_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>

namespace pbrt {

constexpr int nPMJ02bnSets = 5;
constexpr int nPMJ02bnSamples = 65536;

extern PBRT_CONST uint32_t pmj02bnSamples[nPMJ02bnSets][nPMJ02bnSamples][2];

PBRT_HOST_DEVICE_INLINE
Point2f GetPMJ02BNSample(int setIndex, int sampleIndex) {
    setIndex %= nPMJ02bnSets;
    DCHECK_LT(sampleIndex, nPMJ02bnSamples);
    sampleIndex %= nPMJ02bnSamples;

    // Convert from fixed-point.
#ifdef __CUDA_ARCH__
    return Point2f(pmj02bnSamples[setIndex][sampleIndex][0] * 0x1p-32f,
                   pmj02bnSamples[setIndex][sampleIndex][1] * 0x1p-32f);
#else
    // Double precision is key here for the pixel sample sorting, but not
    // necessary otherwise.
    return Point2f(pmj02bnSamples[setIndex][sampleIndex][0] * 0x1p-32,
                   pmj02bnSamples[setIndex][sampleIndex][1] * 0x1p-32);
#endif
}

pstd::vector<Point2f> *GetSortedPMJ02BNPixelSamples(int samplesPerPixel,
                                                    Allocator alloc,
                                                    int *pixelTileSize);

} // namespace pbrt

#endif // PBRT_SAMPLING_PMJ02TABLES_H
