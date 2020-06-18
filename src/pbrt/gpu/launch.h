// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_GPU_LAUNCH_H
#define PBRT_GPU_LAUNCH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/log.h>

#include <tuple>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace pbrt {

struct GPUKernelStats {
    GPUKernelStats() = default;
    GPUKernelStats(const char *description) : description(description) {
        launchEvents.reserve(256);
    }

    std::string description;
    int blockSize = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> launchEvents;
};

GPUKernelStats &GetGPUKernelStats(std::type_index typeIndex, const char *description);

template <typename T>
inline GPUKernelStats &GetGPUKernelStats(const char *description) {
    return GetGPUKernelStats(std::type_index(typeid(T)), description);
}

// https://stackoverflow.com/a/27885283
// primary template.
template <class T>
struct function_traits : function_traits<decltype(&T::operator())> {};

// partial specialization for function type
template <class R, class... Args>
struct function_traits<R(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<Args...>;
};

// partial specialization for function pointer
template <class R, class... Args>
struct function_traits<R (*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<Args...>;
};

// partial specialization for std::function
template <class R, class... Args>
struct function_traits<std::function<R(Args...)>> {
    using result_type = R;
    using argument_types = std::tuple<Args...>;
};

// partial specialization for pointer-to-member-function (i.e., operator()'s)
template <class T, class R, class... Args>
struct function_traits<R (T::*)(Args...)> {
    using result_type = R;
    using argument_types = std::tuple<Args...>;
};

template <class T, class R, class... Args>
struct function_traits<R (T::*)(Args...) const> {
    using result_type = R;
    using argument_types = std::tuple<Args...>;
};

// additional cv-qualifier and ref-qualifier combinations omitted
// sprinkle with C-style variadics if desired

template <class T>
using first_argument_type =
    typename std::tuple_element<0, typename function_traits<T>::argument_types>::type;

template <typename F, typename... Args>
__global__ void Kernel(F func, int nItems, Args... args) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems)
        return;

    func(first_argument_type<F>(tid), args...);
}

template <typename F, typename... Args>
void GPUParallelFor(const char *description, int nItems, F func, Args... args) {
    auto kernel = &Kernel<F, Args...>;

    GPUKernelStats &kernelStats = GetGPUKernelStats<F>(description);
    if (kernelStats.blockSize == 0) {
        int minGridSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
            &minGridSize, &kernelStats.blockSize, kernel, 0, 0));

        LOG_VERBOSE("[%s]: block size %d", description, kernelStats.blockSize);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifndef NDEBUG
    LOG_VERBOSE("Launching %s", description);
#endif
    cudaEventRecord(start);
    int gridSize = (nItems + kernelStats.blockSize - 1) / kernelStats.blockSize;
    kernel<<<gridSize, kernelStats.blockSize>>>(func, nItems, args...);
    cudaEventRecord(stop);

    kernelStats.launchEvents.push_back(std::make_pair(start, stop));

#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSynchronize());
    LOG_VERBOSE("Post-sync %s", description);
#endif
}

#define GPUDo(description, ...) \
    GPUParallelFor(description, 1, [=] __device__(int) { __VA_ARGS__ })

void ReportKernelStats();

}  // namespace pbrt

#endif  // PBRT_GPU_LAUNCH_H
