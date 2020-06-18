// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// core/options.cpp*
#include <pbrt/options.h>

#include <pbrt/util/print.h>

namespace pbrt {

ExtendedOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER)
__constant__ BasicOptions OptionsGPU;
#endif

std::string ExtendedOptions::ToString() const {
    return StringPrintf(
        "[ ExtendedOptions nThreads: %d seed: %d quickRender: %s quiet: %s "
        "recordPixelStatistics: %s "
        "upgrade: %s disablePixelJitter: %s "
        "disableWavelengthJitter: %s forceDiffuse: %s "
        "imageFile: %s mseReferenceImage: %s mseReferenceOutput: %s "
        "debugStart: %s displayServer: %s cropWindow: %s pixelBounds: %s ]",
        nThreads, seed, quickRender, quiet, recordPixelStatistics, upgrade,
        disablePixelJitter, disableWavelengthJitter, forceDiffuse, imageFile,
        mseReferenceImage, mseReferenceOutput, debugStart, displayServer, cropWindow,
        pixelBounds);
}

}  // namespace pbrt
