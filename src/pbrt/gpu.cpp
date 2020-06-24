
#ifdef PBRT_HAVE_OPTIX

#include <pbrt/gpu.h>

#include <pbrt/base.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/genscene.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/gpu/accel.h>
#include <pbrt/lights.h>
#include <pbrt/lightsampling.h>
#include <pbrt/materials.h>
#include <pbrt/plymesh.h>
#include <pbrt/shapes.h>
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

#include <array>
#include <cstring>
#include <iostream>
#include <map>
#include <typeinfo>
#include <typeindex>

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

        LOG_VERBOSE("CUDA device %d (%s) with %f MiB, %d SMs running at %f MHz "
                    "with shader model  %d.%d, max stack %d", i,
                    deviceProperties.name, deviceProperties.totalGlobalMem / (1024.*1024.),
                    deviceProperties.multiProcessorCount,
                    deviceProperties.clockRate / 1000., deviceProperties.major,
                    deviceProperties.minor, stackSize);
    }

    CUDA_CHECK(cudaSetDevice(0));
    LOG_VERBOSE("Selected device %d", 0);

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

struct SampleState {
    Point2i pPixel;
    int sampleIndex;  // which pixel sample
    int dimension;
};

class GPUHaltonSampler {
public:
    PBRT_HOST_DEVICE_INLINE
    GPUHaltonSampler(SampleState *sampleState,
                     const pstd::vector<DigitPermutation> &permutations)
        : permutations(permutations) {
        indexer.SetPixel(sampleState->pPixel);
        indexer.SetPixelSample(sampleState->sampleIndex);
        dimensionPointer = &sampleState->dimension;
        dimension = *dimensionPointer;
    }

    PBRT_HOST_DEVICE_INLINE
    ~GPUHaltonSampler() {
        *dimensionPointer = dimension;
    }

    PBRT_HOST_DEVICE_INLINE
    Point2f Get2D() {
        Point2f u;
        if (dimension == 0)
            u = indexer.SampleFirst2D();
        else
            u = Point2f(ScrambledRadicalInverse(dimension, indexer.SampleIndex(),
                                                permutations[dimension]),
                        ScrambledRadicalInverse(dimension + 1, indexer.SampleIndex(),
                                                permutations[dimension + 1]));
        dimension += 2;
        return u;
    }

    PBRT_HOST_DEVICE_INLINE
    Float Get1D() {
        Float u = ScrambledRadicalInverse(dimension, indexer.SampleIndex(),
                                          permutations[dimension]);
        ++dimension;
        return u;
    }

private:
    Halton128PixelIndexer indexer;
    int dimension;
    int *dimensionPointer;
    const pstd::vector<DigitPermutation> &permutations;
};

template <typename Camera>
__device__ void generateCameraRays(const Camera *camera,
                                   int pixelIndex,
                                   SampleState *sampleState,
                                   const FilterSampler *filterSampler,
                                   SampledWavelengths *lambdaArray,
                                   int *numActiveRays,
                                   int *rayIndexToPixelIndex,
                                   Point3fSOA *rayo,
                                   Vector3fSOA *rayd,
                                   SampledSpectrumSOA *cameraRayWeight,
                                   Float *filterSampleWeight,
                                   const pstd::vector<DigitPermutation> &permutations) {
    if (!InsideExclusive(sampleState[pixelIndex].pPixel, camera->film->pixelBounds))
        return;

    GPUHaltonSampler sampler(&sampleState[pixelIndex], permutations);

    FilterSample fs = filterSampler->Sample(sampler.Get2D());

    CameraSample sample;
    sample.pFilm = Point2f(sampleState[pixelIndex].pPixel) + Vector2f(fs.p) + Vector2f(0.5f, 0.5f);
    sample.pLens = sampler.Get2D();

    Float lu = RadicalInverseSpecialized<3>(sampleState[pixelIndex].sampleIndex) +
        BlueNoise(47, sampleState[pixelIndex].pPixel.x, sampleState[pixelIndex].pPixel.y);
    if (lu >= 1) lu -= 1;
    SampledWavelengths lambda = SampledWavelengths::SampleImportance(lu);

    pstd::optional<CameraRay> cr = camera->GenerateRay(sample, lambda);
    if (!cr) {
        cameraRayWeight->at(pixelIndex) = SampledSpectrum(0);
        return;
    }

    // EnqueueRay...
    int rayIndex = atomicAdd(numActiveRays, 1);
    rayIndexToPixelIndex[rayIndex] = pixelIndex;
    rayo->at(rayIndex) = cr->ray.o;
    rayd->at(rayIndex) = cr->ray.d;

    // It's sort of a hack to do this here...
    lambdaArray[pixelIndex] = lambda;
    filterSampleWeight[pixelIndex] = fs.weight;
    cameraRayWeight->at(pixelIndex) = cr->weight;
}

struct PathState {
    SampledSpectrum L;
    SampledSpectrum beta;
    Float etaScale;
    RNG rng;
};

struct BSDFSlice {
    int count;
    int *bsdfIndexToRayIndex;
};

struct KernelParameters {
    KernelParameters(int nPixels, Allocator alloc) {
        sampleStateArray = alloc.allocate_object<SampleState>(nPixels);
        for (int i = 0; i < nPixels; ++i) alloc.construct(&sampleStateArray[i]);

        lambdaArray = alloc.allocate_object<SampledWavelengths>(nPixels);
        for (int i = 0; i < nPixels; ++i) alloc.construct(&lambdaArray[i]);

        numActiveRays[0] = alloc.new_object<int>(0);
        numActiveRays[1] = alloc.new_object<int>(0);
        rayIndexToPixelIndex[0] = alloc.allocate_object<int>(nPixels);
        rayIndexToPixelIndex[1] = alloc.allocate_object<int>(nPixels);

        rayoSOA = alloc.new_object<Point3fSOA>(alloc, nPixels);
        raydSOA = alloc.new_object<Vector3fSOA>(alloc, nPixels);

        numShadowRays = alloc.new_object<int>(0);
        shadowRayIndexToPixelIndex = alloc.allocate_object<int>(nPixels);
        shadowRayoSOA = alloc.new_object<Point3fSOA>(alloc, nPixels);
        shadowRaydSOA = alloc.new_object<Vector3fSOA>(alloc, nPixels);

        cameraRayWeightsSOA = alloc.new_object<SampledSpectrumSOA>(alloc, nPixels);

        intersectionsArray[0] = alloc.allocate_object<SurfaceInteraction>(nPixels);
        intersectionsArray[1] = alloc.allocate_object<SurfaceInteraction>(nPixels);
        for (int i = 0; i < nPixels; ++i) {
            alloc.construct(&intersectionsArray[0][i]);
            alloc.construct(&intersectionsArray[1][i]);
        }

        filterSampleWeightsArray = alloc.allocate_object<Float>(nPixels);
        bsdfPDFArray = alloc.allocate_object<Float>(nPixels);
        tMaxArray = alloc.allocate_object<Float>(nPixels);
        occludedArray = alloc.allocate_object<uint8_t>(nPixels);

        LiArray = alloc.allocate_object<SampledSpectrum>(nPixels);

        pathStateArray = alloc.allocate_object<PathState>(nPixels);
        for (int i = 0; i < nPixels; ++i) alloc.construct(&pathStateArray[i]);

        sampledLightArray = alloc.allocate_object<LightHandle>(nPixels);
        for (int i = 0; i < nPixels; ++i) sampledLightArray[i] = nullptr;
        sampledLightPDFArray = alloc.allocate_object<Float>(nPixels);

        materialBuffers =
            alloc.allocate_object<uint8_t>(nPixels * materialBufferSize);

        permutations = ComputeRadicalInversePermutations(6502 /* seed */, alloc);

        bsdfSlices = alloc.allocate_object<BSDFSlice>(BxDFHandle::MaxTag());
        for (int i = 0; i < BxDFHandle::MaxTag(); ++i)
            bsdfSlices[i].bsdfIndexToRayIndex = alloc.allocate_object<int>(nPixels);
    }

    static constexpr int materialBufferSize = 256;

    SampleState *sampleStateArray;
    SampledWavelengths *lambdaArray;
    int *numActiveRays[2];
    int *rayIndexToPixelIndex[2];
    Point3fSOA *rayoSOA;
    Vector3fSOA *raydSOA;
    int *numShadowRays;
    int *shadowRayIndexToPixelIndex;
    Point3fSOA *shadowRayoSOA;
    Vector3fSOA *shadowRaydSOA;
    SampledSpectrumSOA *cameraRayWeightsSOA;
    SurfaceInteraction *intersectionsArray[2];
    Float *filterSampleWeightsArray;
    Float *bsdfPDFArray;
    Float *tMaxArray;
    uint8_t *occludedArray;
    SampledSpectrum *LiArray;
    PathState *pathStateArray;
    LightHandle *sampledLightArray;
    Float *sampledLightPDFArray;
    uint8_t *materialBuffers;
    pstd::vector<DigitPermutation> *permutations;
    BSDFSlice *bsdfSlices;
};

template <typename BxDF>
void SampleDirect(int nPixels, const char *name, LightHandle light, Float lightChoicePDF,
                  KernelParameters *kp, int depth) {
    auto EnqueueShadowRay = [=] __device__ (Point3f p, Vector3f w, Float t,
                                            SampledSpectrum L, int pixelIndex) {
        int shadowRayIndex = atomicAdd(kp->numShadowRays, 1);
        SampledSpectrum *shadowRayLi = &kp->LiArray[shadowRayIndex];
        kp->shadowRayIndexToPixelIndex[shadowRayIndex] = pixelIndex;
        kp->shadowRayoSOA->at(shadowRayIndex) = p;
        kp->shadowRaydSOA->at(shadowRayIndex) = w;
        *shadowRayLi = L;
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
        SampleState *sampleState = &kp->sampleStateArray[pixelIndex];
        PathState *pathState = &kp->pathStateArray[pixelIndex];

        SampledSpectrum beta = pathState->beta;
        if (!beta || !intersection.material)
            return;

        GPUHaltonSampler sampler(sampleState, *kp->permutations);
        Point2f u = sampler.Get2D();

        LightHandle localLight = light;
        Float localLightChoicePDF = lightChoicePDF;
        if (!light) {
            localLight = kp->sampledLightArray[pixelIndex];
            localLightChoicePDF = kp->sampledLightPDFArray[pixelIndex];

            if (!localLight)
                return;
        }

        pstd::optional<LightLiSample> ls = localLight.Sample_Li(intersection, u, lambda);
        if (ls && ls->pdf > 0 && ls->L) {
            BSDF *bsdf = intersection.bsdf;
            Vector3f wo = intersection.wo;
            SampledSpectrum f = bsdf->f<BxDF>(wo, ls->wi);
            if (f) {
                Float cosTheta = AbsDot(ls->wi, intersection.shading.n);
                Float lightPDF = ls->pdf * localLightChoicePDF;
                Float weight = 1;
                if (!IsDeltaLight(localLight->type)) {
                    Float bsdfPDF = bsdf->PDF<BxDF>(wo, ls->wi);
                    weight = PowerHeuristic(1, lightPDF, 1, bsdfPDF);
                }

                SampledSpectrum Li = beta * ls->L * f * (weight * cosTheta / lightPDF);
                Ray ray = intersection.SpawnRayTo(ls->pLight);
                EnqueueShadowRay(ray.o, ray.d, 1 - ShadowEpsilon, Li, pixelIndex);
            }
        }
    });
}

template <typename BxDF>
void SampleIndirect(int nPixels, const char *name, KernelParameters *kp, int depth) {
    auto EnqueueRay = [=] __device__ (Point3f p, Vector3f w, int pixelIndex) {
        int newRayIndex = atomicAdd(kp->numActiveRays[(depth & 1) ^ 1], 1);
        kp->rayoSOA->at(newRayIndex) = p;
        kp->raydSOA->at(newRayIndex) = w;
        kp->rayIndexToPixelIndex[(depth & 1) ^ 1][newRayIndex] = pixelIndex;
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
        SampleState *sampleState = &kp->sampleStateArray[pixelIndex];
        PathState *pathState = &kp->pathStateArray[pixelIndex];

        SampledSpectrum beta = pathState->beta;
        if (!beta || !intersection.material)
            return;

        BSDF *bsdf = intersection.bsdf;
        Vector3f wo = intersection.wo;

        GPUHaltonSampler sampler(sampleState, *kp->permutations);
        Point2f u = sampler.Get2D();
        Float uc = sampler.Get1D();

        pstd::optional<BSDFSample> bs = bsdf->Sample_f<BxDF>(wo, uc, u);
        if (!bs || !bs->f || bs->pdf == 0) {
            pathState->beta = SampledSpectrum(0);
            return;
        }
        beta *= bs->f * (AbsDot(intersection.shading.n, bs->wi) / bs->pdf);

        kp->bsdfPDFArray[pixelIndex] = bs->IsSpecular() ? -1 :
            (bsdf->PDFIsApproximate() ? bsdf->PDF(wo, bs->wi) : bs->pdf);

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
        }

        EnqueueRay(intersection.OffsetRayOrigin(bs->wi), bs->wi, pixelIndex);

        pathState->beta = beta;
    });
}

void GPURender(GeneralScene &scene) {
    // Get GPU stuff ready to go...
    StatRegisterer::CallInitializationCallbacks();

    pstd::pmr::polymorphic_allocator<pstd::byte> alloc(&cudaMemoryResource);

    pstd::unique_ptr<Filter> filter(Filter::Create(scene.filter.name, scene.filter.parameters,
                                                   &scene.filter.loc, alloc));
    FilterSampler *filterSampler = alloc.new_object<FilterSampler>(filter.get(), 64, alloc);
    CHECK_EQ(scene.film.name, "rgb");

    const RGBColorSpace *colorSpace = RGBColorSpace::sRGB;
    RGBFilm *film = RGBFilm::Create(scene.film.parameters, std::move(filter),
                                    colorSpace, alloc);

    auto gpuize = [&alloc](const AnimatedTransform &at) {
        const Transform *t[2] = {
                                 alloc.new_object<Transform>(*at.startTransform),
                                 alloc.new_object<Transform>(*at.endTransform)
        };
        return AnimatedTransform(t[0], at.startTime,
                                 t[1], at.endTime);
    };

    int spp = scene.sampler.parameters.GetOneInt("pixelsamples", 16);
    if (PbrtOptions.pixelSamples)
        spp = *PbrtOptions.pixelSamples;
    int maxDepth = scene.integrator.parameters.GetOneInt("maxdepth", 5);

    AnimatedTransform wfcGPU = gpuize(scene.camera.worldFromCamera);
    PerspectiveCamera *pc = nullptr;
    RealisticCamera *rc = nullptr;
    if (scene.camera.name == "perspective")
        pc = PerspectiveCamera::Create(scene.camera.parameters,
                                       wfcGPU, std::unique_ptr<Film>(film),
                                       nullptr /* medium */, alloc);
    else if (scene.camera.name == "realistic")
        rc = RealisticCamera::Create(scene.camera.parameters,
                                     wfcGPU, std::unique_ptr<Film>(film),
                                     nullptr /* medium */, alloc);
    else
        LOG_FATAL("\"%s\" camera not yet supported", scene.camera.name);

    // Env light, if specified.
    ImageInfiniteLight *envLight = nullptr;
    pstd::vector<LightHandle> allLights(alloc);  // except for env... (and should skip distant...)
    for (const auto &light : scene.lights) {
        LightHandle l = LightHandle::Create(light.name, light.parameters,
                                            gpuize(light.worldFromObject),
                                            nullptr /* Medium * */, light.loc,
                                            alloc);
        if (light.name == "infinite")
            // TODO: warn if already have one...
            envLight = l.CastOrNullptr<ImageInfiniteLight>();
        else
            allLights.push_back(l);
    }


    // Area lights...
    std::map<int, pstd::vector<LightHandle *> *> shapeIndexToAreaLights;
    for (size_t i = 0; i < scene.shapes.size(); ++i) {
        const auto &shape = scene.shapes[i];
        if (shape.lightIndex == -1)
            continue;
        CHECK_LT(shape.lightIndex, scene.areaLights.size());
        const auto &areaLightEntity = scene.areaLights[shape.lightIndex];
        AnimatedTransform worldFromLight(shape.worldFromObject);

        pstd::vector<ShapeHandle> shapeHandles =
            ShapeHandle::Create(shape.name,
                                alloc.new_object<Transform>(*shape.worldFromObject),
                                alloc.new_object<Transform>(*shape.objectFromWorld),
                                shape.reverseOrientation, shape.parameters, alloc,
                                shape.loc);

        if (shapeHandles.empty())
            continue;

        pstd::vector<LightHandle *> *lightsForShape =
            alloc.new_object<pstd::vector<LightHandle *>>(alloc);
        for (ShapeHandle sh : shapeHandles) {
            DiffuseAreaLight *area =
                DiffuseAreaLight::Create(gpuize(worldFromLight), nullptr /*mediumInterface.outside*/,
                                         areaLightEntity.parameters,
                                         areaLightEntity.parameters.ColorSpace(),
                                         alloc, sh);
            allLights.push_back(area);
            lightsForShape->push_back(alloc.new_object<LightHandle>(area));
        }
        shapeIndexToAreaLights[i] = lightsForShape;
    }
    LOG_VERBOSE("%d lights", allLights.size());

    GPUAccel accel(scene, nullptr /* cuda stream */, shapeIndexToAreaLights);

    // preprocess...
    if (envLight)
        envLight->Preprocess(accel.Bounds());
    for (LightHandle light : allLights)
        light.Preprocess(accel.Bounds());

    BVHLightSampler *lightSampler = alloc.new_object<BVHLightSampler>(allLights, alloc);

    ///////////////////////// Render!

    Vector2i resolution = film->pixelBounds.Diagonal();
    int nPixels = resolution.x * resolution.y;

    KernelParameters *kp = alloc.new_object<KernelParameters>(nPixels, alloc);
    KernelParameters *kpHost = new KernelParameters(*kp);

    Timer timer;

    // Initialize state for the first sample
    GPUParallelFor(nPixels, "Initialize SampleState",
    [=] __device__ (int pixelIndex, Point2i pMin, Point2i resolution) {
            SampleState &sampleState = kp->sampleStateArray[pixelIndex];

            sampleState.pPixel = Point2i(pMin.x + pixelIndex % resolution.x,
                                         pMin.y + pixelIndex / resolution.x);
            sampleState.sampleIndex = 0;
            sampleState.dimension = 0;
    }, film->pixelBounds.pMin, Point2i(resolution));

    for (int pixelSample = 0; pixelSample < spp; ++pixelSample) {
        // Generate camera rays
        GPUParallelFor(1, "Reset Num Camera Rays",
        [=] __device__ (int) {
            *kp->numActiveRays[0] = 0;
        });

        GPUParallelFor(nPixels, "Reset sampler dimension",
        [=] __device__ (int pixelIndex) {
            SampleState &sampleState = kp->sampleStateArray[pixelIndex];
            sampleState.dimension = 0;
        });

        if (pc)
            GPUParallelFor(nPixels, "Generate PerspectiveCamera rays",
                [=] __device__ (int pixelIndex) {
                    generateCameraRays(pc, pixelIndex, kp->sampleStateArray, filterSampler, kp->lambdaArray,
                                       kp->numActiveRays[0], kp->rayIndexToPixelIndex[0],
                                       kp->rayoSOA, kp->raydSOA, kp->cameraRayWeightsSOA,
                                       kp->filterSampleWeightsArray, *kp->permutations);
                });
        else
            GPUParallelFor(nPixels, "Generate RealisticCamera rays",
                [=] __device__ (int pixelIndex) {
                    generateCameraRays(rc, pixelIndex, kp->sampleStateArray, filterSampler, kp->lambdaArray,
                                       kp->numActiveRays[0], kp->rayIndexToPixelIndex[0],
                                       kp->rayoSOA, kp->raydSOA, kp->cameraRayWeightsSOA,
                                       kp->filterSampleWeightsArray, *kp->permutations);
                });

        // path tracing
        GPUParallelFor(nPixels, "Initialize PathState",
        [=] __device__ (int pixelIndex) {
            // Initialize all of them, to be sure we have zero L
            // for rays that didn't spawn.
            PathState &pathState = kp->pathStateArray[pixelIndex];
            pathState.L = SampledSpectrum(0.f);
            pathState.beta = SampledSpectrum(1.f);
            pathState.etaScale = 1.f;

            const SampleState &sampleState = kp->sampleStateArray[pixelIndex];
            pathState.rng.SetSequence(13 * pixelIndex ^ (sampleState.sampleIndex << 8));
        });

        for (int depth = 0; true; ++depth) {
            GPUParallelFor(nPixels, "Clear intersections",
            [=] __device__ (int pixelIndex) {
                kp->intersectionsArray[depth & 1][pixelIndex].material = nullptr;
                kp->intersectionsArray[depth & 1][pixelIndex].areaLight = nullptr;
            });

            auto events = accel.IntersectClosest(nPixels, kpHost->numActiveRays[depth & 1],
                             kpHost->rayIndexToPixelIndex[depth & 1], kpHost->rayoSOA,
                             kpHost->raydSOA, kpHost->intersectionsArray[depth & 1]);
            struct IsectHack { };
            GetGPUKernelStats<IsectHack>("Path tracing closest hit rays").launchEvents.push_back(events);

            GPUParallelFor(nPixels, "Handle ray-found emission",
            [=] __device__ (int rayIndex) {
                if (rayIndex >= *kp->numActiveRays[depth & 1])
                    return;

                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];

                SampledSpectrum beta = pathState->beta;
                if (!beta)
                    return;

                auto rayd = kp->raydSOA->at(rayIndex);

                if (intersection.material) {
                    // Hit something; add surface emission if there is any
                    if (intersection.areaLight && *intersection.areaLight) {
                        printf("wot hai area light\n");
                        Vector3f wo = -Vector3f(rayd);
                        const DiffuseAreaLight *light = intersection.areaLight->Cast<const DiffuseAreaLight>();
                        SampledSpectrum Le = light->L(intersection, wo, lambda);
                        if (Le) {
                            Float weight;
                            if (depth == 0 || kp->bsdfPDFArray[pixelIndex] < 0 /* specular */)
                                weight = 1;
                            else {
                                const SurfaceInteraction &prevIntr =
                                    kp->intersectionsArray[(depth & 1) ^ 1][pixelIndex];
                                // Compute MIS pdf...
                                Float lightChoicePDF = lightSampler->PDF(prevIntr, *intersection.areaLight);
                                Float lightPDF = lightChoicePDF * light->Pdf_Li(prevIntr, rayd);
                                Float bsdfPDF = kp->bsdfPDFArray[pixelIndex];
                                weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                            }
                            pathState->L += beta * weight * Le;
                        }
                    }
                    return;
                }

                auto rayo = kp->rayoSOA->at(rayIndex);

                if (envLight) {
                    SampledSpectrum Le = envLight->Le(Ray(rayo, rayd), lambda);
                    if (Le) {
                        if (depth == 0 || kp->bsdfPDFArray[pixelIndex] < 0 /* aka specular */)
                            pathState->L += beta * Le;
                        else {
                            const SurfaceInteraction &prevIntersection =
                                kp->intersectionsArray[(depth & 1) ^ 1][pixelIndex];
                            // Compute MIS pdf...
                            Float lightPDF = /*lightSampler->PDF(prevIntr, *light) * */
                                envLight->Pdf_Li(prevIntersection, rayd);
                            Float bsdfPDF = kp->bsdfPDFArray[pixelIndex];
                            Float weight = PowerHeuristic(1, bsdfPDF, 1, lightPDF);
                            // beta already includes 1 / bsdf pdf.
                            pathState->L += beta * weight * Le;
                        }
                    }
                }
            });

            if (depth == maxDepth)
                break;

            GPUParallelFor(1, "Reset Num Rays and BSDFSlices",
            [=] __device__ (int) {
                *kp->numShadowRays = 0;
                *kp->numActiveRays[(depth & 1) ^ 1] = 0;
                for (int i = 0; i < BxDFHandle::MaxTag(); ++i)
                    kp->bsdfSlices[i].count = 0;
            });

            GPUParallelFor(nPixels, "Bump and Material::GetBSDF",
            [=] __device__ (int rayIndex) {
                if (rayIndex >= *kp->numActiveRays[depth & 1])
                    return;

                int pixelIndex = kp->rayIndexToPixelIndex[depth & 1][rayIndex];
                SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];
                PathState *pathState = &kp->pathStateArray[pixelIndex];

                if (!pathState->beta || !intersection.material)
                    return;

                FloatTextureHandle displacement = intersection.material->GetDisplacement();
                if (displacement) {
                    Vector3f dpdu, dpdv;
                    Bump(BasicTextureEvaluator(), //UniversalTextureEvaluator(),
                         displacement, intersection, &dpdu, &dpdv);
                    intersection.SetShadingGeometry(Normal3f(Normalize(Cross(dpdu, dpdv))), dpdu, dpdv,
                                                    Normal3f(0,0,0), Normal3f(0,0,0), false);
                }

                // rayIndex?
                MaterialBuffer materialBuffer(kp->materialBuffers + pixelIndex * kp->materialBufferSize,
                                              kp->materialBufferSize);

                intersection.bsdf = intersection.material->GetBSDF(BasicTextureEvaluator(),
                                                                   //UniversalTextureEvaluator(),
                                                                   intersection, lambda,
                                                                   materialBuffer, TransportMode::Radiance);

                int tag = intersection.bsdf->GetBxDF().Tag();
                int bsdfIndex = atomicAdd(&kp->bsdfSlices[tag].count, 1);
                kp->bsdfSlices[tag].bsdfIndexToRayIndex[bsdfIndex] = rayIndex;
            });

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
                    if (shadowRayIndex > *kp->numShadowRays || kp->occludedArray[shadowRayIndex])
                        return;

                    const SampledSpectrum &Li = kp->LiArray[shadowRayIndex];
                    int pixelIndex = kp->shadowRayIndexToPixelIndex[shadowRayIndex];
                    PathState &pathState = kp->pathStateArray[pixelIndex];
                    pathState.L += Li;
                });

                GPUParallelFor(1, "Reset Num Shadow Rays",
                [=] __device__ (int) {
                    *kp->numShadowRays = 0;
                });
            };

            // TODO: can we synthesize these calls automatically?
            // TODO: SeparableBSSRDFAdapter

            if (envLight) {
                SampleDirect<LambertianBxDF>(nPixels, "Lambertian", envLight, 1.f, kp, depth);
                SampleDirect<CoatedDiffuseBxDF>(nPixels, "CoatedDiffuse", envLight, 1.f, kp, depth);
                SampleDirect<GeneralLayeredBxDF>(nPixels, "GeneralLayered", envLight, 1.f, kp, depth);
                SampleDirect<DielectricInterfaceBxDF>(nPixels, "DielectricInterface", envLight, 1.f, kp, depth);
                SampleDirect<ThinDielectricBxDF>(nPixels, "ThinDielectric", envLight, 1.f, kp, depth);
                SampleDirect<SpecularReflectionBxDF>(nPixels, "SpecularReflection", envLight, 1.f, kp, depth);
                SampleDirect<SpecularTransmissionBxDF>(nPixels, "SpecularTransmission", envLight, 1.f, kp, depth);
                SampleDirect<HairBxDF>(nPixels, "Hair", envLight, 1.f, kp, depth);
                SampleDirect<MeasuredBxDF>(nPixels, "Measured", envLight, 1.f, kp, depth);
                SampleDirect<MixBxDF>(nPixels, "Mix", envLight, 1.f, kp, depth);
                SampleDirect<MicrofacetReflectionBxDF>(nPixels, "MicrofacetReflection", envLight, 1.f, kp, depth);
                SampleDirect<MicrofacetTransmissionBxDF>(nPixels, "MicrofacetTransmission", envLight, 1.f, kp, depth);
                SampleDirect<DisneyBxDF>(nPixels, "Disney", envLight, 1.f, kp, depth);

                TraceShadowRays();
            }

            if (!allLights.empty()) {
                GPUParallelFor(nPixels, "Sample Light BVH",
                [=] __device__ (int pixelIndex) {
                    SurfaceInteraction &intersection = kp->intersectionsArray[depth & 1][pixelIndex];
                    PathState *pathState = &kp->pathStateArray[pixelIndex];

                    if (!pathState->beta || !intersection.material)
                        return;

                    GPUHaltonSampler sampler(&kp->sampleStateArray[pixelIndex], *kp->permutations);

                    struct CandidateLight {
                        LightHandle light;
                        Float pdf;
                    };
                    WeightedReservoirSampler<CandidateLight, float> wrs(pathState->rng.Uniform<uint32_t>());
                    // TODO: better sampling pattern
                    lightSampler->Sample(intersection, sampler.Get1D(),
                        [&](LightHandle light, Float pdf) {
                            wrs.Add(CandidateLight{light, pdf}, 1.f);
                        });

                    if (wrs.HasSample()) {
                        kp->sampledLightArray[pixelIndex] = wrs.GetSample().light;
                        kp->sampledLightPDFArray[pixelIndex] = wrs.GetSample().pdf / wrs.WeightSum();
                    } else
                        kp->sampledLightArray[pixelIndex] = nullptr;
               });


                LightHandle light = nullptr;
                SampleDirect<LambertianBxDF>(nPixels, "Lambertian", light, 0.f, kp, depth);
                SampleDirect<CoatedDiffuseBxDF>(nPixels, "CoatedDiffuse", light, 0.f, kp, depth);
                SampleDirect<GeneralLayeredBxDF>(nPixels, "GeneralLayered", light, 0.f, kp, depth);
                SampleDirect<DielectricInterfaceBxDF>(nPixels, "DielectricInterface", light, 0.f, kp, depth);
                SampleDirect<ThinDielectricBxDF>(nPixels, "ThinDielectric", light, 0.f, kp, depth);
                SampleDirect<SpecularReflectionBxDF>(nPixels, "SpecularReflection", light, 0.f, kp, depth);
                SampleDirect<SpecularTransmissionBxDF>(nPixels, "SpecularTransmission", light, 0.f, kp, depth);
                SampleDirect<HairBxDF>(nPixels, "Hair", light, 0.f, kp, depth);
                SampleDirect<MeasuredBxDF>(nPixels, "Measured", light, 0.f, kp, depth);
                SampleDirect<MixBxDF>(nPixels, "Mix", light, 0.f, kp, depth);
                SampleDirect<MicrofacetReflectionBxDF>(nPixels, "MicrofacetReflection", light, 0.f, kp, depth);
                SampleDirect<MicrofacetTransmissionBxDF>(nPixels, "MicrofacetTransmission", light, 0.f, kp, depth);
                SampleDirect<DisneyBxDF>(nPixels, "Disney", light, 0.f, kp, depth);

                TraceShadowRays();
            }

            SampleIndirect<LambertianBxDF>(nPixels, "Lambertian", kp, depth);
            SampleIndirect<CoatedDiffuseBxDF>(nPixels, "CoatedDiffuse", kp, depth);
            SampleIndirect<GeneralLayeredBxDF>(nPixels, "GeneralLayered", kp, depth);
            SampleIndirect<DielectricInterfaceBxDF>(nPixels, "DielectricInterface", kp, depth);
            SampleIndirect<ThinDielectricBxDF>(nPixels, "ThinDielectric", kp, depth);
            SampleIndirect<SpecularReflectionBxDF>(nPixels, "SpecularReflection", kp, depth);
            SampleIndirect<SpecularTransmissionBxDF>(nPixels, "SpecularTransmission", kp, depth);
            SampleIndirect<HairBxDF>(nPixels, "Hair", kp, depth);
            SampleIndirect<MeasuredBxDF>(nPixels, "Measured", kp, depth);
            SampleIndirect<MixBxDF>(nPixels, "Mix", kp, depth);
            SampleIndirect<MicrofacetReflectionBxDF>(nPixels, "MicrofacetReflection", kp, depth);
            SampleIndirect<MicrofacetTransmissionBxDF>(nPixels, "MicrofacetTransmission", kp, depth);
            SampleIndirect<DisneyBxDF>(nPixels, "Disney", kp, depth);
        }

        // Update film
        GPUParallelFor(nPixels, "Update film",
        [=] __device__ (int pixelIndex) {
            const SampleState &sampleState = kp->sampleStateArray[pixelIndex];
            const PathState &pathState = kp->pathStateArray[pixelIndex];
            const SampledWavelengths &lambda = kp->lambdaArray[pixelIndex];

            SampledSpectrum Lw = pathState.L * kp->cameraRayWeightsSOA->at(pixelIndex);
            // NOTE: assumes that no more than one thread is
            // working on each pixel.
            film->AddSample(sampleState.pPixel, Lw, lambda, {},
                            kp->filterSampleWeightsArray[pixelIndex]);
        });

        // Get ready for the next round
        if (pixelSample < spp - 1) {
            GPUParallelFor(nPixels, "Increment sampleIndex",
            [=] __device__ (int tid) {
                SampleState &sampleState = kp->sampleStateArray[tid];
                ++sampleState.sampleIndex;
            });
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    LOG_VERBOSE("Total rendering time: %.3f s", timer.ElapsedSeconds());

    ReportKernelStats();

    ImageMetadata metadata;
    metadata.renderTimeSeconds = timer.ElapsedSeconds();
    // metadata.samplesPerPixel =
    // pixelBounds / fullResolution
    // cameraFromWorld/ ndcFromWorld...
    metadata.colorSpace = colorSpace;

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
}

}  // namespace pbrt

#endif // PBRT_HAVE_OPTIX
