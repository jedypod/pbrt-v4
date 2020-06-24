
#ifdef PBRT_HAVE_OPTIX

#include <pbrt/gpu.h>

#include <pbrt/base.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/lights.h>
#include <pbrt/lightsampling.h>
#include <pbrt/materials.h>
#include <pbrt/samplers.h>
#include <pbrt/textures.h>
#include <pbrt/util/bluenoise.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/file.h>
#include <pbrt/util/image.h>
#include <pbrt/util/log.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/stats.h>
#include <pbrt/util/taggedptr.h>

#include <cstring>
#include <iostream>
#include <map>
#include <typeinfo>
#include <typeindex>

#include <cuda/std/atomic>

#include <cuda.h>
#include <cuda_runtime.h>

namespace pbrt {

class SampledSpectrumSOA {
public:
    SampledSpectrumSOA(Allocator alloc, size_t n)
        : alloc(alloc), n(n) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] = alloc.allocate_object<Float>(n);
    }
    ~SampledSpectrumSOA() {
        for (int i = 0; i < NSpectrumSamples; ++i)
            alloc.deallocate_object(v[i], n);
    }

    SampledSpectrumSOA() = delete;
    SampledSpectrumSOA(const SampledSpectrumSOA &) = delete;
    SampledSpectrumSOA &operator=(const SampledSpectrumSOA &) = delete;

    PBRT_HOST_DEVICE
    SampledSpectrum at(size_t offset) const {
        DCHECK_LT(offset, n);
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = v[i][offset];
        return s;
    }

    class SampledSpectrumRef {
    public:
        PBRT_HOST_DEVICE_INLINE
        SampledSpectrumRef(pstd::array<Float *, NSpectrumSamples> ptrs)
            : ptrs(ptrs) { }

        PBRT_HOST_DEVICE_INLINE
        operator SampledSpectrum() const {
            SampledSpectrum s;
            for (int i = 0; i < NSpectrumSamples; ++i)
                s[i] = *(ptrs[i]);
            return s;
        }
        PBRT_HOST_DEVICE_INLINE
        void operator=(const SampledSpectrum &s) {
            for (int i = 0; i < NSpectrumSamples; ++i)
                *(ptrs[i]) = s[i];
        }
    private:
        pstd::array<Float *, NSpectrumSamples> ptrs;
    };

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrumRef at(size_t offset) {
        pstd::array<Float *, NSpectrumSamples> ptrs;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ptrs[i] = &v[i][offset];
        return SampledSpectrumRef(ptrs);
    }

private:
    Allocator alloc;
    pstd::array<Float *, NSpectrumSamples> v;
    size_t n;
};

void *CUDAMemoryResource::do_allocate(size_t size, size_t alignment) {
    void *ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    CHECK_EQ(0, intptr_t(ptr) % alignment);
    return ptr;
}

void CUDAMemoryResource::do_deallocate(void *p, size_t bytes, size_t alignment) {
    CUDA_CHECK(cudaFree(p));
}

static CUDAMemoryResource cudaMemoryResource;

struct GPUKernelStats {
    GPUKernelStats() = default;
    GPUKernelStats(const char *description)
        : description(description) {
        launchEvents.reserve(256);
    }

    std::string description;
    int blockSize = 0;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> launchEvents;
};

static std::vector<std::type_index> gpuKernelLaunchOrder;
static std::map<std::type_index, GPUKernelStats> gpuKernels;

template <typename T>
GPUKernelStats &GetGPUKernelStats(const char *description) {
    auto iter = gpuKernels.find(std::type_index(typeid(T)));
    if (iter != gpuKernels.end()) {
        // This will probably hit if IntersectClosest/Shadow are called
        // multiple times with different descriptions...
        CHECK_EQ(iter->second.description, std::string(description));
        return iter->second;
    }

    std::type_index typeIndex = typeid(T);
    gpuKernelLaunchOrder.push_back(typeIndex);
    gpuKernels[typeIndex] = GPUKernelStats(description);
    return gpuKernels.find(typeIndex)->second;
}

template <typename F, typename... Args>
__global__ void Kernel(F func, int nItems, Args... args) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nItems) return;

    func(tid, args...);
}

template <typename F, typename... Args>
void GPUParallelFor(int nItems, const char *description, F func, Args... args) {
    auto kernel = &Kernel<F, Args...>;

    GPUKernelStats &kernelStats = GetGPUKernelStats<F>(description);
    if (kernelStats.blockSize == 0) {
        int minGridSize;
        CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &kernelStats.blockSize,
                                                      kernel, 0, 0));

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

void ReportKernelStats() {
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute total milliseconds over all kernels and launches
    float totalms = 0.f;
    for (const auto kernelTypeId : gpuKernelLaunchOrder) {
        const GPUKernelStats &stats = gpuKernels[kernelTypeId];
        for (const auto &launch : stats.launchEvents) {
            cudaEventSynchronize(launch.second);
            float ms = 0;
            cudaEventElapsedTime(&ms, launch.first, launch.second);
            totalms += ms;
        }
    }

    printf("GPU Kernel Profile:\n");
    int otherLaunches = 0;
    float otherms = 0;
    const float otherCutoff = 0.0025f * totalms;
    for (const auto kernelTypeId : gpuKernelLaunchOrder) {
        float summs = 0.f, minms = 1e30, maxms = 0;
        const GPUKernelStats &stats = gpuKernels[kernelTypeId];
        for (const auto &launch : stats.launchEvents) {
            float ms = 0;
            cudaEventElapsedTime(&ms, launch.first, launch.second);
            summs += ms;
            minms = std::min(minms, ms);
            maxms = std::max(maxms, ms);
        }

        if (summs > otherCutoff)
            Printf("  %-45s %5d launches %9.2f ms / %5.1f%% (avg %6.3f, min %6.3f, max %7.3f)\n",
                   stats.description, stats.launchEvents.size(), summs,
                   100.f * summs / totalms, summs / stats.launchEvents.size(),
                   minms, maxms);
        else {
            otherms += summs;
            otherLaunches += stats.launchEvents.size();
        }
    }
    Printf("  %-45s %5d launches %9.2f ms / %5.1f%% (avg %6.3f)\n",
           "Other", otherLaunches, otherms, 100.f * otherms / totalms,
           otherms / otherLaunches);

    Printf("\n");
}

void GPUInit() {
    cudaFree(nullptr);

    InitGPULogging();

    int driverVersion;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    int runtimeVersion;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    auto versionToString = [](int version) {
        int major = version / 1000;
        int minor = (version - major * 1000) / 10;
        return StringPrintf("%d.%d", major, minor);
    };
    LOG_VERBOSE("GPU CUDA driver %s, CUDA runtime %s",
                versionToString(driverVersion),
                versionToString(runtimeVersion));

    int deviceIndex;
    CUDA_CHECK(cudaGetDevice(&deviceIndex));

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

        LOG_VERBOSE("CUDA device %d (%s) with %f MiB, %d SMs running at %f MHz "
                    "with shader model  %d.%d, max stack %d printf FIFO %d", i,
                    deviceProperties.name, deviceProperties.totalGlobalMem / (1024.*1024.),
                    deviceProperties.multiProcessorCount,
                    deviceProperties.clockRate / 1000., deviceProperties.major,
                    deviceProperties.minor, stackSize, printfFIFOSize);
    }

    CUDA_CHECK(cudaSetDevice(0));
    LOG_VERBOSE("Selected device %d", 0);

#ifndef NDEBUG
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
    size_t stackSize;
    CUDA_CHECK(cudaDeviceGetLimit(&stackSize, cudaLimitStackSize));
    LOG_VERBOSE("Reset stack size to %d", stackSize);
#endif
    CUDA_CHECK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 32*1024*1024));

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
}

template <int base>
__device__
inline Float RadicalInverseSpecialized(uint64_t a) {
    const Float invBase = (Float)1 / (Float)base;
    uint64_t reversedDigits = 0;
    Float invBaseN = 1;
    while (a) {
        uint64_t next = a / base;
        uint64_t digit = a - next * base;
        reversedDigits = reversedDigits * base + digit;
        invBaseN *= invBase;
        a = next;
    }
    DCHECK_LT(reversedDigits * invBaseN, 1.00001);
    return std::min(reversedDigits * invBaseN, OneMinusEpsilon);
}

template <typename Camera>
__device__ void generateCameraRays(const Camera *camera,
                                   int pixelIndex, int sampleIndex,
                                   Point2i *pPixelArray,
                                   SamplerHandle *samplerArray,
                                   FilterHandle filter,
                                   SampledWavelengths *lambdaArray,
                                   cuda::std::atomic<int> *numActiveRays,
                                   int *rayIndexToPixelIndex,
                                   Point3fSOA *rayo,
                                   Vector3fSOA *rayd,
                                   Float *tMaxArray,
                                   SampledSpectrumSOA *cameraRayWeight,
                                   Float *filterSampleWeight) {
    Point2i pPixel = pPixelArray[pixelIndex];
    if (!InsideExclusive(pPixel, camera->film->pixelBounds))
        return;

    SamplerHandle sampler = samplerArray[pixelIndex];

    CameraSample cameraSample = sampler.GetCameraSample(pPixel, filter);

    Float lu = RadicalInverseSpecialized<3>(sampleIndex) + BlueNoise(47, pPixel.x, pPixel.y);
    if (lu >= 1) lu -= 1;
    if (PbrtOptionsGPU.disableWavelengthJitter)
        lu = 0.5f;
    SampledWavelengths lambda = SampledWavelengths::SampleImportance(lu);

    pstd::optional<CameraRay> cr = camera->GenerateRay(cameraSample, lambda);
    if (!cr) {
        cameraRayWeight->at(pixelIndex) = SampledSpectrum(0);
        return;
    }

    // EnqueueRay...
    int rayIndex = numActiveRays->fetch_add(1, cuda::std::memory_order_relaxed);
    rayIndexToPixelIndex[rayIndex] = pixelIndex;
    rayo->at(rayIndex) = cr->ray.o;
    rayd->at(rayIndex) = cr->ray.d;
    tMaxArray[rayIndex] = 1e20f;

    // It's sort of a hack to do this here...
    lambdaArray[pixelIndex] = lambda;
    filterSampleWeight[pixelIndex] = cameraSample.weight;
    cameraRayWeight->at(pixelIndex) = cr->weight;
}

struct PathState {
    SampledSpectrum L;
    SampledSpectrum beta;
    Float etaScale;
};

struct BSDFSlice {
    int count;
    int *bsdfIndexToRayIndex;
};

struct KernelParameters {
    KernelParameters(int nPixels, Point2i fullResolution, int spp, Allocator alloc) {
        pPixelArray = alloc.allocate_object<Point2i>(nPixels);
        for (int i = 0; i < nPixels; ++i) alloc.construct(&pPixelArray[i]);

        lambdaArray = alloc.allocate_object<SampledWavelengths>(nPixels);
        for (int i = 0; i < nPixels; ++i) alloc.construct(&lambdaArray[i]);

        numActiveRays[0] = alloc.new_object<cuda::std::atomic<int>>(0);
        numActiveRays[1] = alloc.new_object<cuda::std::atomic<int>>(0);
        rayIndexToPixelIndex[0] = alloc.allocate_object<int>(nPixels);
        rayIndexToPixelIndex[1] = alloc.allocate_object<int>(nPixels);

        rayoSOA = alloc.new_object<Point3fSOA>(alloc, nPixels);
        raydSOA = alloc.new_object<Vector3fSOA>(alloc, nPixels);

        pixelIndexToIndirectRayIndex = alloc.allocate_object<int>(nPixels);  // just for subsurf...
        for (int i = 0; i < nPixels; ++i)
            pixelIndexToIndirectRayIndex[i] = -1000000000;

        numShadowRays = alloc.new_object<cuda::std::atomic<int>>(0);
        shadowRayIndexToPixelIndex = alloc.allocate_object<int>(nPixels);
        shadowRayoSOA = alloc.new_object<Point3fSOA>(alloc, nPixels);
        shadowRaydSOA = alloc.new_object<Vector3fSOA>(alloc, nPixels);

        numSubsurfaceRays = alloc.new_object<cuda::std::atomic<int>>(0);
        subsurfaceMaterialArray = alloc.allocate_object<MaterialHandle>(nPixels);
        subsurfaceRayIndexToPathRayIndex = alloc.allocate_object<int>(nPixels);
        for (int i = 0; i < nPixels; ++i)
            subsurfaceRayIndexToPathRayIndex[i] = -1000000000;
        subsurfaceRayoSOA = alloc.new_object<Point3fSOA>(alloc, nPixels);
        subsurfaceRaydSOA = alloc.new_object<Vector3fSOA>(alloc, nPixels);
        subsurfaceReservoirSamplerArray =
            alloc.allocate_object<WeightedReservoirSampler<SurfaceInteraction, Float>>(nPixels);
        for (int i = 0; i < nPixels; ++i)
            alloc.construct(&subsurfaceReservoirSamplerArray[i]);

        cameraRayWeightsSOA = alloc.new_object<SampledSpectrumSOA>(alloc, nPixels);

        intersectionsArray[0] = alloc.allocate_object<SurfaceInteraction>(nPixels);
        intersectionsArray[1] = alloc.allocate_object<SurfaceInteraction>(nPixels);
        for (int i = 0; i < nPixels; ++i) {
            alloc.construct(&intersectionsArray[0][i]);
            alloc.construct(&intersectionsArray[1][i]);
        }

        filterSampleWeightsArray = alloc.allocate_object<Float>(nPixels);
        bsdfPDFArray = alloc.allocate_object<Float>(nPixels);
        sampledTransmissionArray = alloc.allocate_object<int>(nPixels);
        tMaxArray = alloc.allocate_object<Float>(nPixels);
        occludedArray = alloc.allocate_object<int>(nPixels);

        LiSOAArray = alloc.new_object<SampledSpectrumSOA>(alloc, nPixels);

        pathStateArray = alloc.allocate_object<PathState>(nPixels);
        for (int i = 0; i < nPixels; ++i) alloc.construct(&pathStateArray[i]);

        sampledLightArray = alloc.allocate_object<LightHandle>(nPixels);
        for (int i = 0; i < nPixels; ++i) sampledLightArray[i] = nullptr;
        sampledLightPDFArray = alloc.allocate_object<Float>(nPixels);

        constexpr int materialBufferSize = 512;
        uint8_t *materialBufferMemory =
            alloc.allocate_object<uint8_t>(nPixels * materialBufferSize);
        materialBufferArray = alloc.allocate_object<MaterialBuffer>(nPixels);
        for (int i = 0; i < nPixels; ++i)
            materialBufferArray[i] =
                MaterialBuffer(materialBufferMemory + i * materialBufferSize,
                               materialBufferSize);

        bsdfSlices = alloc.allocate_object<BSDFSlice>(BxDFHandle::MaxTag() + 1);
        for (int i = 0; i < BxDFHandle::MaxTag() + 1; ++i)
            bsdfSlices[i].bsdfIndexToRayIndex = alloc.allocate_object<int>(nPixels);

        visibleSurfaceArray = alloc.allocate_object<pstd::optional<VisibleSurface>>(nPixels);
        for (int i = 0; i < nPixels; ++i)
            alloc.construct(&visibleSurfaceArray[i]);

        samplerArray = alloc.allocate_object<SamplerHandle>(nPixels);
    }

    Point2i *pPixelArray;
    SampledWavelengths *lambdaArray;
    cuda::std::atomic<int> *numActiveRays[2];
    int *rayIndexToPixelIndex[2];
    Point3fSOA *rayoSOA;
    Vector3fSOA *raydSOA;
    int *pixelIndexToIndirectRayIndex;

    cuda::std::atomic<int> *numShadowRays;
    int *shadowRayIndexToPixelIndex;
    Point3fSOA *shadowRayoSOA;
    Vector3fSOA *shadowRaydSOA;

    cuda::std::atomic<int> *numSubsurfaceRays;
    MaterialHandle *subsurfaceMaterialArray;
    int *subsurfaceRayIndexToPathRayIndex;
    Point3fSOA *subsurfaceRayoSOA;
    Vector3fSOA *subsurfaceRaydSOA;
    WeightedReservoirSampler<SurfaceInteraction, Float> *subsurfaceReservoirSamplerArray;

    SampledSpectrumSOA *cameraRayWeightsSOA;
    SurfaceInteraction *intersectionsArray[2];
    Float *filterSampleWeightsArray;
    Float *bsdfPDFArray;
    int *sampledTransmissionArray;
    Float *tMaxArray;
    int *occludedArray;
    SampledSpectrumSOA *LiSOAArray;
    PathState *pathStateArray;
    LightHandle *sampledLightArray;
    Float *sampledLightPDFArray;
    MaterialBuffer *materialBufferArray;
    BSDFSlice *bsdfSlices;
    pstd::optional<VisibleSurface> *visibleSurfaceArray;
    SamplerHandle *samplerArray;
};

template <typename BxDF>
void SampleDirect(int nPixels, const char *name, KernelParameters *kp, int depth) {
    auto EnqueueShadowRay = [=] __device__ (Point3f p, Vector3f w, Float t,
                                            SampledSpectrum L, int pixelIndex) {
        int shadowRayIndex = kp->numShadowRays->fetch_add(1, cuda::std::memory_order_relaxed);
        DCHECK(shadowRayIndex >= 0 && shadowRayIndex < nPixels);
        kp->shadowRayIndexToPixelIndex[shadowRayIndex] = pixelIndex;
        kp->shadowRayoSOA->at(shadowRayIndex) = p;
        kp->shadowRaydSOA->at(shadowRayIndex) = w;
        kp->LiSOAArray->at(shadowRayIndex) = L;
        kp->tMaxArray[shadowRayIndex] = t;
    };

    int bxdfTag = BxDFHandle::TypeIndex<BxDF>();
    GPUParallelFor(nPixels, StringPrintf("Sample direct - %s", name).c_str(),
    [=] __device__ (int bsdfEvalIndex) {
        // IMPORTANT: must not read rayo/rayd in this kernel...
        if (bsdfEvalIndex >= kp->bsdfSlices[bxdfTag].count)
            return;

        int rayIndex = kp->bsdfSlices[bxdfTag].bsdfIndexToRayIndex[bsdfEvalIndex];
        int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
        SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
        const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
        PathState *pathState = &kp->pathStateArray[pixelIndex];

        SampledSpectrum beta = pathState->beta;
        if (!beta || !intersection.material)
            return;

        LightHandle light = kp->sampledLightArray[pixelIndex];
        if (!light)
            return;

        SamplerHandle sampler = kp->samplerArray[pixelIndex];
        Point2f u = sampler.Get2D();
        Float lightChoicePDF = kp->sampledLightPDFArray[pixelIndex];

        pstd::optional<LightLiSample> ls = light.Sample_Li(intersection, u, lambda);
        if (!ls || ls->pdf == 0 || !ls->L)
            return;

        BSDF *bsdf = intersection.bsdf;
        Vector3f wo = intersection.wo;
        SampledSpectrum f = bsdf->f<BxDF>(wo, ls->wi);
        if (!f)
            return;

        Float cosTheta = AbsDot(ls->wi, intersection.shading.n);
        Float lightPDF = ls->pdf * lightChoicePDF;
        Float weight = 1;
        if (!IsDeltaLight(light->type)) {
            Float bsdfPDF = bsdf->PDF<BxDF>(wo, ls->wi);
            weight = PowerHeuristic(1, lightPDF, 1, bsdfPDF);
        }

        SampledSpectrum Li = beta * ls->L * f * (weight * cosTheta / lightPDF);
        DCHECK(!Li.HasNaNs());
        Ray ray = intersection.SpawnRayTo(ls->pLight);
        EnqueueShadowRay(ray.o, ray.d, 1 - ShadowEpsilon, Li, pixelIndex);
    });
}

template <typename BxDF>
void SampleIndirect(int nPixels, const char *name, KernelParameters *kp, int depth, bool overrideRay = false) {
    auto EnqueueRay = [=] __device__ (Point3f p, Vector3f w, int pixelIndex, int rayIndex, int bsdfEvalIndex) {
        int newRayIndex;
        if (overrideRay)
            newRayIndex = kp->pixelIndexToIndirectRayIndex[pixelIndex];
        else {
            newRayIndex = kp->numActiveRays[(depth & 1) ^ 1]->fetch_add(1, cuda::std::memory_order_relaxed);
            kp->pixelIndexToIndirectRayIndex[pixelIndex] = newRayIndex;
        }
        DCHECK(newRayIndex >= 0 && newRayIndex < nPixels);

        kp->rayoSOA->at(newRayIndex) = p;
        kp->raydSOA->at(newRayIndex) = w;
        kp->rayIndexToPixelIndex[(depth & 1) ^ 1][newRayIndex] = pixelIndex;
        kp->tMaxArray[newRayIndex] = 1e20f;
    };

    int bxdfTag = BxDFHandle::TypeIndex<BxDF>();
    GPUParallelFor(nPixels, StringPrintf("Sample indirect - %s", name).c_str(),
    [=] __device__ (int bsdfEvalIndex) {
        // IMPORTANT: must not read rayo/rayd in this kernel...
        if (bsdfEvalIndex >= kp->bsdfSlices[bxdfTag].count)
            return;

        int rayIndex = kp->bsdfSlices[bxdfTag].bsdfIndexToRayIndex[bsdfEvalIndex];
        int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
        SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
        const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
        PathState *pathState = &kp->pathStateArray[pixelIndex];

        SampledSpectrum beta = pathState->beta;
        if (!beta || !intersection.material)
            return;

        BSDF *bsdf = intersection.bsdf;
        Vector3f wo = intersection.wo;

        SamplerHandle sampler = kp->samplerArray[pixelIndex];
        Point2f u = sampler.Get2D();
        Float uc = sampler.Get1D();

        pstd::optional<BSDFSample> bs = bsdf->Sample_f<BxDF>(wo, uc, u);
        if (!bs || !bs->f || bs->pdf == 0) {
            pathState->beta = SampledSpectrum(0);
            return;
        }
        beta *= bs->f * (AbsDot(intersection.shading.n, bs->wi) / bs->pdf);
        DCHECK(!beta.HasNaNs());

        kp->bsdfPDFArray[pixelIndex] = bs->IsSpecular() ? -1 :
            (bsdf->PDFIsApproximate() ? bsdf->PDF(wo, bs->wi) : bs->pdf);

        kp->sampledTransmissionArray[pixelIndex] = bs->IsTransmission();

        if (bs->IsTransmission())
            // Update the term that tracks radiance scaling for refraction.
            pathState->etaScale *= Sqr(bsdf->eta);

        // russian roulette
        SampledSpectrum rrBeta = beta * pathState->etaScale;
        if (rrBeta.MaxComponentValue() < 1 && depth > 3) {
            Float q = std::max<Float>(0, 1 - rrBeta.MaxComponentValue());
            if (sampler.Get1D() < q) {
                pathState->beta = SampledSpectrum(0);
                return;
            }
            beta /= 1 - q;
            DCHECK(!beta.HasNaNs());
        }

        EnqueueRay(intersection.OffsetRayOrigin(bs->wi), bs->wi, pixelIndex, rayIndex, bsdfEvalIndex);

        pathState->beta = beta;
    });
}

static inline __device__ RhoHemiDirSample rhoSample(int i) {
    RhoHemiDirSample sample;
    sample.u = RadicalInverse(0, i + 1);
    sample.u2 = Point2f(RadicalInverse(1, i + 1),
                        RadicalInverse(2, i + 1));
    return sample;
}

void GPURender(GeneralScene &scene) {
    ProfilerScope _(ProfilePhase::SceneConstruction);

    // Get GPU stuff ready to go...
    StatRegisterer::CallInitializationCallbacks();

    pstd::pmr::polymorphic_allocator<pstd::byte> alloc(&cudaMemoryResource);

    FilterHandle filter = FilterHandle::Create(scene.filter.name, scene.filter.parameters,
                                               &scene.filter.loc, alloc);

    const RGBColorSpace *colorSpace = RGBColorSpace::sRGB;
    Film *film = Film::Create(scene.film.name, scene.film.parameters, &scene.film.loc,
                              filter, alloc);
    RGBFilm *rgbFilm = dynamic_cast<RGBFilm *>(film);
    AOVFilm *aovFilm = dynamic_cast<AOVFilm *>(film);
    CHECK(rgbFilm || aovFilm);

    SamplerHandle sampler = SamplerHandle::Create(scene.sampler.name, scene.sampler.parameters,
                                                  film->fullResolution, &scene.sampler.loc,
                                                  alloc);
    int spp = sampler.SamplesPerPixel();
    int maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    Camera *camera = Camera::Create(scene.camera.name, scene.camera.parameters,
                                    nullptr /* medium */, scene.camera.worldFromCamera,
                                    film, &scene.camera.loc, alloc);
    PerspectiveCamera *perspectiveCamera = dynamic_cast<PerspectiveCamera *>(camera);
    OrthographicCamera *orthographicCamera = dynamic_cast<OrthographicCamera *>(camera);
    RealisticCamera *realisticCamera = dynamic_cast<RealisticCamera *>(camera);
    SphericalCamera *sphericalCamera = dynamic_cast<SphericalCamera *>(camera);

    UniformInfiniteLight *uniformEnvLight = nullptr;
    ImageInfiniteLight *imageEnvLight = nullptr;
    pstd::vector<LightHandle> allLights(alloc);
    {
    ProfilerScope _(ProfilePhase::LightConstruction);

    for (const auto &light : scene.lights) {
        LightHandle l = LightHandle::Create(light.name, light.parameters, light.worldFromObject,
                                            nullptr /* Medium * */, &light.loc, alloc);
        if (l.Is<UniformInfiniteLight>()) {
            if (uniformEnvLight)
                Warning(&light.loc, "Multiple uniform infinite lights provided. Using this one.");
            uniformEnvLight = l.Cast<UniformInfiniteLight>();
        }
        if (l.Is<ImageInfiniteLight>()) {
            if (imageEnvLight)
                Warning(&light.loc, "Multiple image infinite lights provided. Using this one.");
            imageEnvLight = l.Cast<ImageInfiniteLight>();
        }

        allLights.push_back(l);
    }
    }

    // Area lights...
    std::map<int, pstd::vector<LightHandle> *> shapeIndexToAreaLights;
    {
    ProfilerScope _(ProfilePhase::LightConstruction);
    for (size_t i = 0; i < scene.shapes.size(); ++i) {
        const auto &shape = scene.shapes[i];
        if (shape.lightIndex == -1)
            continue;
        CHECK_LT(shape.lightIndex, scene.areaLights.size());
        const auto &areaLightEntity = scene.areaLights[shape.lightIndex];
        AnimatedTransform worldFromLight(shape.worldFromObject);

        pstd::vector<ShapeHandle> shapeHandles =
            ShapeHandle::Create(shape.name, shape.worldFromObject, shape.objectFromWorld,
                                shape.reverseOrientation, shape.parameters,
                                &shape.loc, alloc);

        if (shapeHandles.empty())
            continue;

        pstd::vector<LightHandle> *lightsForShape =
            alloc.new_object<pstd::vector<LightHandle>>(alloc);
        for (ShapeHandle sh : shapeHandles) {
            DiffuseAreaLight *area =
                DiffuseAreaLight::Create(worldFromLight, nullptr /*mediumInterface.outside*/,
                                         areaLightEntity.parameters,
                                         areaLightEntity.parameters.ColorSpace(),
                                         &areaLightEntity.loc, alloc, sh);
            allLights.push_back(area);
            lightsForShape->push_back(area);
        }
        shapeIndexToAreaLights[i] = lightsForShape;
    }
    }

    LOG_VERBOSE("%d lights", allLights.size());
    GPUAccel accel(scene, nullptr /* cuda stream */, shapeIndexToAreaLights);

    // preprocess...
    for (LightHandle light : allLights)
        light.Preprocess(accel.Bounds());

    if (allLights.size() == 0)
        ErrorExit("No light sources specified");

    BVHLightSampler *lightSampler = alloc.new_object<BVHLightSampler>(allLights, alloc);

    ///////////////////////// Render!

    // Same hack as in cpurender.cpp...
    if (PbrtOptions.profile) {
        CHECK_EQ(CurrentProfilerState(), ProfilePhaseToBits(ProfilePhase::SceneConstruction));
        ProfilerState = ProfilePhaseToBits(ProfilePhase::IntegratorRender);
    }

    Vector2i resolution = film->pixelBounds.Diagonal();
    int nPixels = resolution.x * resolution.y;

    KernelParameters *kp = alloc.new_object<KernelParameters>(nPixels, film->fullResolution,
                                                              spp, alloc);
    KernelParameters *kpHost = new KernelParameters(*kp);

    std::vector<SamplerHandle> clonedSamplers = sampler.Clone(nPixels, alloc);
    for (int i = 0; i < nPixels; ++i)
        kp->samplerArray[i] = clonedSamplers[i];

    Timer timer;

    // Initialize state for the first sample
    GPUParallelFor(nPixels, "Initialize pPixelArray",
    [=] __device__ (int pixelIndex, Point2i pMin, Point2i resolution) {
        kp->pPixelArray[pixelIndex] = Point2i(pMin.x + pixelIndex % resolution.x,
                                              pMin.y + pixelIndex / resolution.x);
    }, film->pixelBounds.pMin, Point2i(resolution));

    ProgressReporter progress(spp, "Rendering", true /* GPU */);

    for (int pixelSample = 0; pixelSample < spp; ++pixelSample) {
        // Generate camera rays
        GPUParallelFor(1, "Reset Num Camera Rays",
        [=] __device__ (int) {
            *kp->numActiveRays[0] = 0;
        });

        GPUParallelFor(nPixels, "Reset sampler dimension",
        [=] __device__ (int pixelIndex) {
            SamplerHandle sampler = kp->samplerArray[pixelIndex];
            sampler.StartPixelSample(kp->pPixelArray[pixelIndex], pixelSample);
        });

        if (perspectiveCamera)
            GPUParallelFor(nPixels, "Generate PerspectiveCamera rays",
                [=] __device__ (int pixelIndex) {
                    generateCameraRays(perspectiveCamera, pixelIndex, pixelSample, kp->pPixelArray,
                                       kp->samplerArray, filter,
                                       kp->lambdaArray, kp->numActiveRays[0], kp->rayIndexToPixelIndex[0],
                                       kp->rayoSOA, kp->raydSOA, kp->tMaxArray, kp->cameraRayWeightsSOA,
                                       kp->filterSampleWeightsArray);
                });
        else if (orthographicCamera)
            GPUParallelFor(nPixels, "Generate OrthographicCamera rays",
                [=] __device__ (int pixelIndex) {
                    generateCameraRays(orthographicCamera, pixelIndex, pixelSample, kp->pPixelArray,
                                       kp->samplerArray, filter,
                                       kp->lambdaArray, kp->numActiveRays[0], kp->rayIndexToPixelIndex[0],
                                       kp->rayoSOA, kp->raydSOA, kp->tMaxArray, kp->cameraRayWeightsSOA,
                                       kp->filterSampleWeightsArray);
                });
        else if (sphericalCamera)
            GPUParallelFor(nPixels, "Generate SphericalCamera rays",
                [=] __device__ (int pixelIndex) {
                    generateCameraRays(sphericalCamera, pixelIndex, pixelSample, kp->pPixelArray,
                                       kp->samplerArray, filter,
                                       kp->lambdaArray, kp->numActiveRays[0], kp->rayIndexToPixelIndex[0],
                                       kp->rayoSOA, kp->raydSOA, kp->tMaxArray, kp->cameraRayWeightsSOA,
                                       kp->filterSampleWeightsArray);
                });
        else {
            CHECK(realisticCamera != nullptr);
            GPUParallelFor(nPixels, "Generate RealisticCamera rays",
                [=] __device__ (int pixelIndex) {
                    generateCameraRays(realisticCamera, pixelIndex, pixelSample, kp->pPixelArray,
                                       kp->samplerArray, filter,
                                       kp->lambdaArray, kp->numActiveRays[0], kp->rayIndexToPixelIndex[0],
                                       kp->rayoSOA, kp->raydSOA, kp->tMaxArray, kp->cameraRayWeightsSOA,
                                       kp->filterSampleWeightsArray);
                });
        }

        // path tracing
        GPUParallelFor(nPixels, "Initialize PathState",
        [=] __device__ (int pixelIndex) {
            // Initialize all of them, to be sure we have zero L
            // for rays that didn't spawn.
            PathState &pathState = kp->pathStateArray[pixelIndex];
            pathState.L = SampledSpectrum(0.f);
            pathState.beta = SampledSpectrum(1.f);
            pathState.etaScale = 1.f;
        });

        for (int depth = 0; true; ++depth) {
            GPUParallelFor(nPixels, "Clear intersections",
            [=] __device__ (int pixelIndex) {
                kp->intersectionsArray[depth & 1][pixelIndex].bsdf = nullptr;
                kp->intersectionsArray[depth & 1][pixelIndex].bssrdf = nullptr;
                kp->intersectionsArray[depth & 1][pixelIndex].material = nullptr;
                kp->intersectionsArray[depth & 1][pixelIndex].areaLight = nullptr;
            });

            auto events = accel.IntersectClosest(nPixels, kpHost->numActiveRays[depth & 1],
                             kpHost->rayIndexToPixelIndex[depth & 1], kpHost->rayoSOA,
                             kpHost->raydSOA, kpHost->tMaxArray,
                             kpHost->intersectionsArray[depth & 1]);
            struct IsectHack { };
            GetGPUKernelStats<IsectHack>("Path tracing closest hit rays").launchEvents.push_back(events);

            GPUParallelFor(nPixels, "Handle ray-found emission",
            [=] __device__ (int rayIndex) {
                if (rayIndex >= kp->numActiveRays[depth & 1]->load(cuda::std::memory_order_relaxed))
                    return;

                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];

                Point3f o = kp->rayoSOA->at(rayIndex);

                SampledSpectrum beta = pathState->beta;
                if (!beta)
                    return;

                auto rayd = kp->raydSOA->at(rayIndex);

                // Hit something; add surface emission if there is any
                if (intersection.areaLight) {
                    Vector3f wo = -Vector3f(rayd);
                    const DiffuseAreaLight *light = intersection.areaLight.Cast<const DiffuseAreaLight>();
                    SampledSpectrum Le = light->L(intersection, wo, lambda);
                    if (Le) {
                        Float weight;
                        if (depth == 0 || kp->bsdfPDFArray[pixelIndex] < 0 /* specular */)
                            weight = 1;
                        else {
                            const SurfaceInteraction &prevIntr =
                                kp->intersectionsArray[(depth & 1) ^ 1][pixelIndex];
                            // Compute MIS pdf...
                            Float lightChoicePDF = lightSampler->PDFSingle(prevIntr, intersection.areaLight);
                            Float lightPDF = lightChoicePDF * light->Pdf_Li(prevIntr, rayd);
                            Float bsdfPDF = kp->bsdfPDFArray[pixelIndex];
                            weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                        }
                        pathState->L += beta * weight * Le;
                    }
                }

                if (intersection.material)
                    return;

                if (uniformEnvLight) {
                    auto rayo = kp->rayoSOA->at(rayIndex);
                    SampledSpectrum Le = uniformEnvLight->Le(Ray(rayo, rayd), lambda);
                    if (Le) {
                        if (depth == 0 || kp->bsdfPDFArray[pixelIndex] < 0 /* aka specular */)
                            pathState->L += beta * Le;
                        else {
                            const SurfaceInteraction &prevIntersection =
                                kp->intersectionsArray[(depth & 1) ^ 1][pixelIndex];
                            // Compute MIS pdf...
                            Float lightPDF = /*lightSampler->PDF(prevIntr, *light) * */
                                uniformEnvLight->Pdf_Li(prevIntersection, rayd);
                            Float bsdfPDF = kp->bsdfPDFArray[pixelIndex];
                            Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                            // beta already includes 1 / bsdf pdf.
                            pathState->L += beta * weight * Le;
                        }
                    }
                }
                if (imageEnvLight) {
                    auto rayo = kp->rayoSOA->at(rayIndex);
                    SampledSpectrum Le = imageEnvLight->Le(Ray(rayo, rayd), lambda);
                    if (Le) {
                        if (depth == 0 || kp->bsdfPDFArray[pixelIndex] < 0 /* aka specular */)
                            pathState->L += beta * Le;
                        else {
                            const SurfaceInteraction &prevIntersection =
                                kp->intersectionsArray[(depth & 1) ^ 1][pixelIndex];
                            // Compute MIS pdf...
                            Float lightPDF = /*lightSampler->PDF(prevIntr, *light) * */
                                imageEnvLight->Pdf_Li(prevIntersection, rayd);
                            Float bsdfPDF = kp->bsdfPDFArray[pixelIndex];
                            Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                            // beta already includes 1 / bsdf pdf.
                            pathState->L += beta * weight * Le;
                        }
                    }
                }
            });

            if (depth == 1 && aovFilm) {
                // Set Ld in the visible surface
                GPUParallelFor(nPixels, "Initialize VisibleSurface.Ld",
                [=] __device__ (int pixelIndex) {
                    SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];

                    if (intersection.material) {
                        PathState &pathState = kp->pathStateArray[pixelIndex];
                        kp->visibleSurfaceArray[pixelIndex]->Ld =
                            pathState.L - kp->visibleSurfaceArray[pixelIndex]->Le;
                    }
                });
            }

            if (depth == maxDepth)
                break;

            GPUParallelFor(1, "Reset Num Rays and BSDFSlices",
            [=] __device__ (int) {
                *kp->numShadowRays = 0;
                *kp->numActiveRays[(depth & 1) ^ 1] = 0;
                for (int i = 0; i < BxDFHandle::MaxTag() + 1; ++i)
                    kp->bsdfSlices[i].count = 0;
            });

            GPUParallelFor(nPixels, "Bump and Material::GetBSDF/GetBSSRDF",
            [=] __device__ (int rayIndex) {
                if (rayIndex >= kp->numActiveRays[depth & 1]->load(cuda::std::memory_order_relaxed))
                    return;

                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];

                if (!pathState->beta || !intersection.material)
                    return;

                FloatTextureHandle displacement = intersection.material.GetDisplacement();
                if (displacement) {
                    Vector3f dpdu, dpdv;
                    Bump(BasicTextureEvaluator(), //UniversalTextureEvaluator(),
                         displacement, intersection, &dpdu, &dpdv);
                    intersection.SetShadingGeometry(Normal3f(Normalize(Cross(dpdu, dpdv))), dpdu, dpdv,
                                                    Normal3f(0,0,0), Normal3f(0,0,0), false);
                }

                // rayIndex?
                MaterialBuffer &materialBuffer = kp->materialBufferArray[pixelIndex];
                materialBuffer.Reset();

                intersection.bsdf = intersection.material.GetBSDF(BasicTextureEvaluator(),
                                                                  //UniversalTextureEvaluator(),
                                                                  intersection, lambda,
                                                                  materialBuffer, TransportMode::Radiance);

                if (intersection.bsdf) {
                    if (PbrtOptionsGPU.forceDiffuse) {
                        int nRhoSamples = 16;
                        SampledSpectrum r = intersection.bsdf->rho(intersection.wo, rhoSample, nRhoSamples);

                        intersection.bsdf = materialBuffer.Alloc<BSDF>(intersection,
                                     materialBuffer.Alloc<LambertianBxDF>(r, SampledSpectrum(0.), 0.),
                                     intersection.bsdf->eta);
                    }

                    int tag = intersection.bsdf->GetBxDF().Tag();
                    int bsdfIndex = atomicAdd(&kp->bsdfSlices[tag].count, 1);
                    DCHECK(bsdfIndex >= 0 && bsdfIndex < nPixels);
                    kp->bsdfSlices[tag].bsdfIndexToRayIndex[bsdfIndex] = rayIndex;
                }

                intersection.bssrdf = intersection.material.GetBSSRDF(BasicTextureEvaluator(), intersection,
                                                                      lambda, materialBuffer,
                                                                      TransportMode::Radiance);
            });

            if (depth == 0 && aovFilm) {
                // Set the pixel's VisibleSurface for the first visible point.
                GPUParallelFor(nPixels, "Initialize VisibleSurface",
                [=] __device__ (int pixelIndex) {
                    SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                    const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
                    PathState *pathState = &kp->pathStateArray[pixelIndex];

                    if (intersection.material)
                        kp->visibleSurfaceArray[pixelIndex] = VisibleSurface(intersection, *camera, lambda);
                });
            }

            auto TraceShadowRays = [=]() {
                GPUParallelFor(nPixels, "Clear occludedArray",
                [=] __device__ (int shadowRayIndex) {
                    kp->occludedArray[shadowRayIndex] = 0;
                });

                auto events = accel.IntersectShadow(nPixels, kpHost->numShadowRays,
                                                    kpHost->shadowRayoSOA, kpHost->shadowRaydSOA, kpHost->tMaxArray,
                                                    kpHost->occludedArray);
                struct IsectShadowHack { };
                GetGPUKernelStats<IsectShadowHack>("Path tracing shadow rays").launchEvents.push_back(events);

                // Add contribution if light was visible
                // TODO: fold this into the optix code?
                GPUParallelFor(nPixels, "Incorporate shadow ray contribution",
                [=] __device__ (int shadowRayIndex) {
                    if (shadowRayIndex > kp->numShadowRays->load(cuda::std::memory_order_relaxed) ||
                        kp->occludedArray[shadowRayIndex])
                        return;

                    SampledSpectrum Li = kp->LiSOAArray->at(shadowRayIndex);
                    int pixelIndex = kp->shadowRayIndexToPixelIndex[shadowRayIndex];
                    DCHECK(pixelIndex >= 0 && pixelIndex < nPixels);
                    PathState &pathState = kp->pathStateArray[pixelIndex];
                    pathState.L += Li;
                });

                GPUParallelFor(1, "Reset Num Shadow Rays",
                [=] __device__ (int) {
                    *kp->numShadowRays = 0;
                });
            };

            GPUParallelFor(nPixels, "Sample Light BVH",
            [=] __device__ (int rayIndex) {
                if (rayIndex >= kp->numActiveRays[depth & 1]->load(cuda::std::memory_order_relaxed))
                    return;

                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];

                if (!pathState->beta || !intersection.material || !intersection.bsdf)
                    return;

                if (intersection.bsdf->IsSpecular() &&
                    !intersection.bsdf->IsNonSpecular()) { // TODO: is the second check needed?
                    kp->sampledLightArray[pixelIndex] = nullptr;
                    return;
                }

                SamplerHandle sampler = kp->samplerArray[pixelIndex];

                struct CandidateLight {
                    LightHandle light;
                    Float pdf;
                };

                uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
                WeightedReservoirSampler<CandidateLight, float> wrs(seed);
                lightSampler->SampleSingle(intersection, sampler.Get1D(),
                    [&](LightHandle light, Float pdf) {
                        wrs.Add(CandidateLight{light, pdf}, 1.f);
                    });

                if (wrs.HasSample()) {
                    kp->sampledLightArray[pixelIndex] = wrs.GetSample().light;
                    kp->sampledLightPDFArray[pixelIndex] = wrs.GetSample().pdf / wrs.WeightSum();
                } else
                    kp->sampledLightArray[pixelIndex] = nullptr;
           });

            // TODO: can we synthesize these calls automatically?
            SampleDirect<LambertianBxDF>(nPixels, "Lambertian", kp, depth);
            SampleDirect<CoatedDiffuseBxDF>(nPixels, "CoatedDiffuse", kp, depth);
            SampleDirect<GeneralLayeredBxDF>(nPixels, "GeneralLayered", kp, depth);
            SampleDirect<DielectricInterfaceBxDF>(nPixels, "DielectricInterface", kp, depth);
            SampleDirect<ThinDielectricBxDF>(nPixels, "ThinDielectric", kp, depth);
            SampleDirect<SpecularReflectionBxDF>(nPixels, "SpecularReflection", kp, depth);
            SampleDirect<HairBxDF>(nPixels, "Hair", kp, depth);
            SampleDirect<MeasuredBxDF>(nPixels, "Measured", kp, depth);
            SampleDirect<MixBxDF>(nPixels, "Mix", kp, depth);
            SampleDirect<MicrofacetReflectionBxDF>(nPixels, "MicrofacetReflection", kp, depth);
            SampleDirect<MicrofacetTransmissionBxDF>(nPixels, "MicrofacetTransmission", kp, depth);

            TraceShadowRays();

            SampleIndirect<LambertianBxDF>(nPixels, "Lambertian", kp, depth);
            SampleIndirect<CoatedDiffuseBxDF>(nPixels, "CoatedDiffuse", kp, depth);
            SampleIndirect<GeneralLayeredBxDF>(nPixels, "GeneralLayered", kp, depth);
            SampleIndirect<DielectricInterfaceBxDF>(nPixels, "DielectricInterface", kp, depth);
            SampleIndirect<ThinDielectricBxDF>(nPixels, "ThinDielectric", kp, depth);
            SampleIndirect<SpecularReflectionBxDF>(nPixels, "SpecularReflection", kp, depth);
            SampleIndirect<HairBxDF>(nPixels, "Hair", kp, depth);
            SampleIndirect<MeasuredBxDF>(nPixels, "Measured", kp, depth);
            SampleIndirect<MixBxDF>(nPixels, "Mix", kp, depth);
            SampleIndirect<MicrofacetReflectionBxDF>(nPixels, "MicrofacetReflection", kp, depth);
            SampleIndirect<MicrofacetTransmissionBxDF>(nPixels, "MicrofacetTransmission", kp, depth);

            // Subsurface scattering
            // TODO: skip this entirely if there are no subsurface materials in the scene...
            GPUParallelFor(1, "Reset num subsurface rays",
            [=] __device__ (int) {
                *kp->numSubsurfaceRays = 0;
            });

            GPUParallelFor(nPixels, "Sample subsurface scattering",
            [=] __device__ (int rayIndex) {
                if (rayIndex >= kp->numActiveRays[depth & 1]->load(cuda::std::memory_order_relaxed))
                    return;

                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];

                if (!pathState->beta || !intersection.material || !intersection.bsdf ||
                    !intersection.bssrdf || !kp->sampledTransmissionArray[pixelIndex])
                    return;

                SamplerHandle sampler = kp->samplerArray[pixelIndex];
                pstd::optional<BSSRDFProbeSegment> probeSeg =
                    intersection.bssrdf.Sample(sampler.Get1D(), sampler.Get2D());
                if (!probeSeg)
                    return;

                // Enqueue ray
                int ssRayIndex = kp->numSubsurfaceRays->fetch_add(1, cuda::std::memory_order_relaxed);
                DCHECK(ssRayIndex >= 0 && ssRayIndex < nPixels);
                kp->subsurfaceMaterialArray[ssRayIndex] = intersection.material;
                kp->subsurfaceRayIndexToPathRayIndex[ssRayIndex] = rayIndex;
                kp->subsurfaceRayoSOA->at(ssRayIndex) = probeSeg->p0;
                kp->subsurfaceRaydSOA->at(ssRayIndex) = probeSeg->p1 - probeSeg->p0;
                kp->tMaxArray[ssRayIndex] = 1;

                kp->subsurfaceReservoirSamplerArray[ssRayIndex].Reset();
                uint64_t seed = MixBits(FloatToBits(sampler.Get1D()));
                kp->subsurfaceReservoirSamplerArray[ssRayIndex].Seed(seed);
            });

            events = accel.IntersectOneRandom(nPixels, kpHost->numSubsurfaceRays,
                                              kpHost->subsurfaceMaterialArray,
                                              kpHost->subsurfaceRayoSOA, kpHost->subsurfaceRaydSOA,
                                              kpHost->tMaxArray, kpHost->subsurfaceReservoirSamplerArray);
            struct IsectRandomHack { };
            GetGPUKernelStats<IsectRandomHack>(
                "Tracing subsurface scattering probe rays").launchEvents.push_back(events);

            GPUParallelFor(nPixels, "Incorporate subsurface S factor",
            [=] __device__ (int ssRayIndex) {
                if (ssRayIndex >= kp->numSubsurfaceRays->load(cuda::std::memory_order_relaxed))
                    return;

                int rayIndex = kp->subsurfaceRayIndexToPathRayIndex[ssRayIndex]; // incident ray
                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];
                if (!pathState->beta)
                    return;

                WeightedReservoirSampler<SurfaceInteraction, Float> &interactionSampler =
                    kp->subsurfaceReservoirSamplerArray[ssRayIndex];

                if (!interactionSampler.HasSample())
                    return;

                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                MaterialBuffer &materialBuffer = kp->materialBufferArray[pixelIndex];
                BSSRDFSample bssrdfSample =
                    intersection.bssrdf.ProbeIntersectionToSample(interactionSampler.GetSample(),
                                                                  materialBuffer);

                if (!bssrdfSample.S || bssrdfSample.pdf == 0) {
                    pathState->beta = SampledSpectrum(0.);
                    return;
                }
                pathState->beta *= bssrdfSample.S * interactionSampler.WeightSum() /
                    bssrdfSample.pdf;
                DCHECK(!pathState->beta.HasNaNs());

                // This copy is annoying, but it us reuse SampleDirect and
                // SampleIndirect...
                intersection.pi = bssrdfSample.si.pi;
                intersection.dpdu = bssrdfSample.si.dpdu;
                intersection.dpdv = bssrdfSample.si.dpdv;
                intersection.dndu = bssrdfSample.si.dndu;
                intersection.dndv = bssrdfSample.si.dndv;
                intersection.n = bssrdfSample.si.n;
                intersection.uv = bssrdfSample.si.uv;
                intersection.wo = bssrdfSample.si.wo;
                intersection.shading = bssrdfSample.si.shading;
                intersection.bsdf = bssrdfSample.si.bsdf; // important!

                int tag = intersection.bsdf->GetBxDF().Tag();
                DCHECK(tag == BxDFHandle::TypeIndex<BSSRDFAdapter>());
                int bsdfIndex = atomicAdd(&kp->bsdfSlices[tag].count, 1);
                DCHECK(bsdfIndex < nPixels);
                kp->bsdfSlices[tag].bsdfIndexToRayIndex[bsdfIndex] = rayIndex;
            });

            SampleDirect<BSSRDFAdapter>(nPixels, "BSSRDFAdapter", kp, depth);

            TraceShadowRays();

            SampleIndirect<BSSRDFAdapter>(nPixels, "BSSRDFAdapter", kp, depth, true);
        }

        // Update film
        if (rgbFilm) {
            GPUParallelFor(nPixels, "Update RGBFilm",
            [=] __device__ (int pixelIndex) {
                const PathState &pathState = kp->pathStateArray[pixelIndex];
                const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];

                SampledSpectrum Lw = pathState.L * kp->cameraRayWeightsSOA->at(pixelIndex);
                // NOTE: assumes that no more than one thread is
                // working on each pixel.
                rgbFilm->AddSample(kp->pPixelArray[pixelIndex], Lw, lambda, {},
                                   kp->filterSampleWeightsArray[pixelIndex]);
            });
        } else {
            CHECK(aovFilm != nullptr);

            GPUParallelFor(nPixels, "Update AOVFilm",
            [=] __device__ (int pixelIndex) {
                const PathState &pathState = kp->pathStateArray[pixelIndex];
                const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];

                SampledSpectrum Lw = pathState.L * kp->cameraRayWeightsSOA->at(pixelIndex);
                // NOTE: assumes that no more than one thread is
                // working on each pixel.
                aovFilm->AddSample(kp->pPixelArray[pixelIndex], Lw, lambda,
                                   kp->visibleSurfaceArray[pixelIndex],
                                   kp->filterSampleWeightsArray[pixelIndex]);
            });
        }

        progress.Update();
    }
    progress.Done();
    CUDA_CHECK(cudaDeviceSynchronize());

    LOG_VERBOSE("Total rendering time: %.3f s", timer.ElapsedSeconds());

    if (PbrtOptions.profile) {
        CHECK_EQ(CurrentProfilerState(), ProfilePhaseToBits(ProfilePhase::IntegratorRender));
        ProfilerState = ProfilePhaseToBits(ProfilePhase::SceneConstruction);
    }

    ReportKernelStats();

    ImageMetadata metadata;
    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    metadata.samplesPerPixel = spp;
    metadata.colorSpace = colorSpace;
    camera->InitMetadata(&metadata);

#if 0
    std::vector<GPULogItem> logs = ReadGPULogs();
    for (const auto &item : logs) {
        if (item.level < LOGGING_logConfig.level)
            continue;
        if (item.level < LogLevel::Error)
            LOG_VERBOSE("GPU %s(%d): %s", item.file, item.line, item.message);
        else
            LOG_ERROR("GPU %s(%d): %s", item.file, item.line, item.message);
    }
#endif

    film->WriteImage(metadata);

    // Cleanup
    ShapeHandle::FreeBufferCaches();
}

}  // namespace pbrt

#endif // PBRT_HAVE_OPTIX
