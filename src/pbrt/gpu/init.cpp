// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/gpu/init.h>

#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>

#include <cuda.h>

namespace pbrt {

void GPUInit() {
    cudaFree(nullptr);

    int driverVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    int runtimeVersion;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    auto versionToString = [](int version) {
        int major = version / 1000;
        int minor = (version - major * 1000) / 10;
        return StringPrintf("%d.%d", major, minor);
    };
    LOG_VERBOSE("GPU CUDA driver %s, CUDA runtime %s", versionToString(driverVersion),
                versionToString(runtimeVersion));

    int nDevices;
    CUDA_CHECK(cudaGetDeviceCount(&nDevices));
    for (int i = 0; i < nDevices; ++i) {
        cudaDeviceProp deviceProperties;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, i));
        CHECK(deviceProperties.canMapHostMemory);

        size_t stackSize;
        CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
        size_t printfFIFOSize;
        CUDA_CHECK(cudaDeviceGetLimit(&printfFIFOSize, cudaLimitPrintfFifoSize));

        LOG_VERBOSE(
            "CUDA device %d (%s) with %f MiB, %d SMs running at %f MHz "
            "with shader model  %d.%d, max stack %d printf FIFO %d",
            i, deviceProperties.name, deviceProperties.totalGlobalMem / (1024. * 1024.),
            deviceProperties.multiProcessorCount, deviceProperties.clockRate / 1000.,
            deviceProperties.major, deviceProperties.minor, stackSize, printfFIFOSize);
    }

    int device = Options->gpuDevice ? *Options->gpuDevice : 0;
    LOG_VERBOSE("Selecting GPU device %d", device);
    CUDA_CHECK(cudaSetDevice(device));

    // CO#ifndef NDEBUG
    // This causes the stack smasher check to fail on Linux with CUDA 11...
//CO    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 2 * 8192));
    // CO#else
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
    // CO#endif
    size_t stackSize;
    CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
    LOG_VERBOSE("Reset stack size to %d", stackSize);

    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32 * 1024 * 1024));

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}

}  // namespace pbrt
