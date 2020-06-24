
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

// core/api.cpp*
#include <pbrt/pbrt.h>

#include <pbrt/gpu.h>
#include <pbrt/options.h>
#include <pbrt/shapes.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/spectrum.h>

namespace pbrt {

// API Function Definitions
void InitPBRT(const Options &opt, const LogConfig &logConfig) {
    PbrtOptions = opt;
    // API Initialization

    InitLogging(logConfig);

    // General \pbrt Initialization
    int nThreads = PbrtOptions.nThreads != 0 ? PbrtOptions.nThreads : AvailableCores();
    ParallelInit(nThreads);  // Threads must be launched before the
                             // profiler is initialized.
    if (PbrtOptions.profile)
        InitProfiler();

#ifdef PBRT_HAVE_OPTIX
    GPUInit();

    CUDA_CHECK(cudaMemcpyToSymbol(PbrtOptionsGPU, &PbrtOptions, sizeof(PbrtOptionsGPU)));

    // Yuck: this has to stick around since ShapeHandle::InitBufferCaches() ends
    // up making copies of the Allocator its been given...
    // TODO: have a static Allocator gpuAllocator ?
    static CUDAMemoryResource mr;
    // Before RGBColorSpace::Init!
    SPDs::Init(Allocator(&mr));
    RGBToSpectrumTable::Init(Allocator(&mr));

    RGBColorSpace::Init(Allocator(&mr));
    ShapeHandle::InitBufferCaches(Allocator(&mr));
    TriangleMesh::Init(Allocator(&mr));
    BilinearPatchMesh::Init(Allocator(&mr));
#else
    // Before RGBColorSpace::Init!
    SPDs::Init(Allocator{});
    RGBToSpectrumTable::Init(Allocator{});

    RGBColorSpace::Init(Allocator{});
    ShapeHandle::InitBufferCaches({});
    TriangleMesh::Init({});
    BilinearPatchMesh::Init({});
#endif
}

void CleanupPBRT() {
    // API Cleanup
    ParallelCleanup();

    if (PbrtOptions.profile)
        CleanupProfiler();
}

}  // namespace pbrt
