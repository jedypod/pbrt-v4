// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_OPTIONS_H
#define PBRT_CORE_OPTIONS_H

// core/options.h*
#include <pbrt/pbrt.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

struct BasicOptions {
    int nThreads = 0;
    int seed = 0;
    bool quickRender = false;
    bool quiet = false;
    bool recordPixelStatistics = false;
    bool upgrade = false;
    bool disablePixelJitter = false, disableWavelengthJitter = false;
    bool forceDiffuse = false;
};

struct ExtendedOptions : BasicOptions {
    pstd::optional<int> pixelSamples;
    pstd::optional<int> gpuDevice;
    pstd::optional<std::string> imageFile;
    pstd::optional<std::string> mseReferenceImage, mseReferenceOutput;
    pstd::optional<std::string> debugStart;
    pstd::optional<std::string> displayServer;
    pstd::optional<Bounds2f> cropWindow;
    pstd::optional<Bounds2i> pixelBounds;

    std::string ToString() const;
};

extern ExtendedOptions *Options;

#if defined(PBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
extern __constant__ BasicOptions OptionsGPU;
#endif

PBRT_CPU_GPU inline const BasicOptions &GetOptions() {
#if defined(PBRT_IS_GPU_CODE)
    return OptionsGPU;
#else
    return *Options;
#endif
}

}  // namespace pbrt

#endif  // PBRT_CORE_OPTIONS_H
