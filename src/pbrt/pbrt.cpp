// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

// core/api.cpp*
#include <pbrt/pbrt.h>

#include <pbrt/gpu/init.h>
#include <pbrt/options.h>
#include <pbrt/shapes.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/display.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/stats.h>

namespace pbrt {

// API Function Definitions
void InitPBRT(const ExtendedOptions &opt, const LogConfig &logConfig) {
    Options = new ExtendedOptions(opt);
    // API Initialization

    if (Options->quiet)
        SuppressErrorMessages();

    InitLogging(logConfig);

    // General \pbrt Initialization
    int nThreads = Options->nThreads != 0 ? Options->nThreads : AvailableCores();
    ParallelInit(nThreads);  // Threads must be launched before the
                             // profiler is initialized.

#ifdef PBRT_BUILD_GPU_RENDERER
    GPUInit();

    CUDA_CHECK(cudaMemcpyToSymbol(OptionsGPU, &Options, sizeof(OptionsGPU)));

    SPDs::Init(gpuMemoryAllocator);
    RGBToSpectrumTable::Init(gpuMemoryAllocator);

    RGBColorSpace::Init(gpuMemoryAllocator);
    InitBufferCaches(gpuMemoryAllocator);
    Triangle::Init(gpuMemoryAllocator);
    BilinearPatch::Init(gpuMemoryAllocator);
#else
    // Before RGBColorSpace::Init!
    SPDs::Init(Allocator{});
    RGBToSpectrumTable::Init(Allocator{});

    RGBColorSpace::Init(Allocator{});
    InitBufferCaches({});
    Triangle::Init({});
    BilinearPatch::Init({});
#endif

    if (Options->displayServer)
        ConnectToDisplayServer(*Options->displayServer);
}

void CleanupPBRT() {
    ForEachThread(ReportThreadStats);

    if (Options->recordPixelStatistics)
        StatsWritePixelImages();

    if (!Options->quiet) {
        PrintStats(stdout);
        ClearStats();
    }
    if (PrintCheckRare(stdout))
        ErrorExit("CHECK_RARE failures");

    if (Options->displayServer)
        DisconnectFromDisplayServer();

    // API Cleanup
    ParallelCleanup();

    // CO    delete Options;
    Options = nullptr;
}

}  // namespace pbrt
