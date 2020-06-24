
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

#include <pbrt/lightsampling.h>

#include <pbrt/lights.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/interaction.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <atomic>
#include <numeric>
#include <cstdint>
#include <vector>

namespace pbrt {

std::string SampledLight::ToString() const {
    return StringPrintf("[ SampledLight light: %s pdf: %f ]",
                        light ? light.ToString().c_str() : "(nullptr)", pdf);
}

LightSampler::~LightSampler() {}

LightSampler *LightSampler::Create(const std::string &name, pstd::span<const LightHandle> lights,
                                   Allocator alloc) {
    if (name == "uniform" || lights.size() == 1)
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    else if (name == "all")
        return alloc.new_object<AllLightsSampler>(lights, alloc);
    else if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    else if (name == "exhaustive")
        return alloc.new_object<ExhaustiveLightSampler>(lights, alloc);
    else if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    else {
        Error(
            R"(Light sample distribution type "%s" unknown. Using "bvh".)",
            name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }
}

///////////////////////////////////////////////////////////////////////////
// AllLightsSampler

SampledLightVector AllLightsSampler::Sample(const Interaction &intr,
                                            Float u, MemoryArena &arena) const {
    ArenaAllocator<SampledLight> alloc(&arena);
    SampledLightVector vec(alloc);
    vec.reserve(lights.size());
    for (const auto &light : lights)
        vec.push_back({light, (Float)1});
    return vec;
}

///////////////////////////////////////////////////////////////////////////
// FixedLightSampler

FixedLightSampler::FixedLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
    : LightSampler(lights, alloc), lightToIndex(alloc) {
    // Compute a reverse mapping from light pointers to offsets into the
    // lights vector (and, equivalently, offsets into lightSampler).
    for (size_t i = 0; i < lights.size(); ++i)
        lightToIndex.Insert(lights[i], i);
}

FixedLightSampler::~FixedLightSampler() {}

SampledLightVector FixedLightSampler::Sample(const Interaction &intr, Float u,
                                             MemoryArena &arena) const {
    ArenaAllocator<SampledLight> alloc(&arena);
    SampledLightVector vec(alloc);
    Float pdf;
    LightHandle light = Sample(u, &pdf);
    if (light && pdf > 0)
        vec.push_back({light, pdf});
    return vec;
}

///////////////////////////////////////////////////////////////////////////
// UniformLightSampler

UniformLightSampler::UniformLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
    : FixedLightSampler(lights, alloc) {
    if (lights.empty()) return;
    std::vector<Float> prob(lights.size(), Float(1));
    distrib = std::make_unique<Distribution1D>(prob, alloc);
}

///////////////////////////////////////////////////////////////////////////
// PowerLightSampler

PowerLightSampler::PowerLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
    : FixedLightSampler(lights, alloc) {
    if (lights.empty()) return;

    std::vector<Float> lightPower;
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    for (const auto &light : lights)
        lightPower.push_back(light.Phi(lambda).Average());
    distrib = std::make_unique<Distribution1D>(lightPower, alloc);
}

std::string PowerLightSampler::ToString() const {
    return StringPrintf("[ PowerLightSampler distrib: %s ]", *distrib);
}

///////////////////////////////////////////////////////////////////////////
// BVHLightSampler

// TODO: make animated light sources an exercise

// For bvh build
STAT_MEMORY_COUNTER("Memory/Light BVH", lightBVHBytes);
STAT_INT_DISTRIBUTION("Integrator/Lights sampled per lookup", nLightsSampled);
STAT_INT_DISTRIBUTION("Integrator/Light BVH depth", bvhDepth);

BVHLightSampler::BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
    : LightSampler(lights, alloc), lightToNode(alloc), infiniteLights(alloc) {
    std::vector<LightInfo> lightInfo;
    for (const auto &light : lights) {
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds)
            infiniteLights.push_back(light);
        else {
            LightInfo li(light, *lightBounds);
            if (li.lightBounds.phi > 0)
                lightInfo.push_back(li);
        }
    }

    if (lightInfo.empty()) {
        root = nullptr;
        return;
    }

    ProfilerScope _(ProfilePhase::LightDistribCreation);
    int nNodes = 0;
    root = buildBVH(lightInfo, 0, lightInfo.size(), alloc, &nNodes);
    lightBVHBytes += nNodes * sizeof(LightBVHNode);

#if 0
    // FIXME
    int minDepth = 1000000, maxDepth = 0;
    for (const auto &l : lightToNode) {
        LightBVHNode *n = l.second;
        int depth = 0;
        while (n != nullptr) {
            ++depth;
            n = n->parent;
        }
        ReportValue(bvhDepth, depth);
    }
#endif
}

std::string BVHLightSampler::ToString() const {
    return StringPrintf("[ BVHLightSampler root: %s ]", *root);
}

LightBVHNode *BVHLightSampler::buildBVH(std::vector<LightInfo> &lightInfo,
                                        int start, int end, Allocator alloc,
                                        int *nNodes) {
//CO    LOG(WARNING) << "Build range " << start << " - " << end;
    CHECK_LT(start, end);
    (*nNodes)++;
    int nLights = end - start;
    if (nLights == 1) {
        LightBVHNode *node = alloc.new_object<LightBVHNode>(lightInfo[start]);
        lightToNode.Insert(lightInfo[start].light, node);
        return node;
    }

    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        bounds = Union(bounds, lightInfo[i].lightBounds.b);
        centroidBounds = Union(centroidBounds, lightInfo[i].centroid());
    }
//CO    LOG(WARNING) << "bounds: " << bounds << ", centroid bounds: " <<
//CO        centroidBounds;

    // Modified SAH
    // Replace # of primitives with emitter power
    // TODO: use the more efficient bounds/cost sweep calculation from v4

    Float minCost = Infinity;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;

    for (int dim = 0; dim < 3; ++dim) {
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
//CO            LOG(WARNING) << "Degenerate bounds. dim = " << dim <<
//CO                " centroidBounds = " << centroidBounds <<
//CO                " bounds = " << bounds << " start = " << start <<
//CO                " end = " << end;
            continue;
        }

        LightBounds bucketLightBounds[nBuckets];

        for (int i = start; i < end; ++i) {
            Point3f pc = lightInfo[i].centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
//CO            LOG(WARNING) << "Light " << i << ", bounds " << lightInfo[i].lightBounds << " -> bucket " << b;
            if (b == nBuckets) b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b],
                                         lightInfo[i].lightBounds);
        }

//CO        for (int i = 0; i < nBuckets; ++i)
//CO            LOG(WARNING) << "Bucket light bounds " << i << " -> " << bucketLightBounds[i].b;

        // Compute costs for splitting after each bucket
        Float cost[nBuckets - 1];
        for (int i = 0; i < nBuckets - 1; ++i) {
            LightBounds b0, b1;

            for (int j = 0; j <= i; ++j)
                b0 = Union(b0, bucketLightBounds[j]);
            for (int j = i + 1; j < nBuckets; ++j)
                b1 = Union(b1, bucketLightBounds[j]);

            auto Momega = [](const LightBounds &b) {
                Float theta_w = std::min(b.theta_o + b.theta_e, Pi);
                return 2 * Pi * (1 - std::cos(b.theta_o)) +
                Pi / 2 * (2 * theta_w * std::sin(b.theta_o) -
                          std::cos(b.theta_o - 2 * theta_w) -
                          2 * b.theta_o * std::sin(b.theta_o) +
                          std::cos(b.theta_o));
            };

            // Can simplify since we always split
            Float Kr = MaxComponentValue(bounds.Diagonal()) / bounds.Diagonal()[dim];
            cost[i] = Kr * (b0.phi * Momega(b0) * b0.b.SurfaceArea() +
                            b1.phi * Momega(b1) * b1.b.SurfaceArea());
//CO            if (b0.b.SurfaceArea() == 0 || b1.b.SurfaceArea() == 0)
//CO                cost[i] = Kr *
//CO                    (b0.phi * Momega(b0) * MaxComponent(b0.b.Diagonal()) +
//CO                     b1.phi * Momega(b1) * MaxComponent(b1.b.Diagonal()));
//CO            LOG(WARNING) << "dim " << dim << ", bucket " << i << " Kr " << Kr <<
//CO                " cost " << cost[i] << " b0 " << b0 << ", b1 " << b1 <<
//CO                " momega 0 " << Momega(b0) << ", momega b1 " << Momega(b1) <<
//CO                " b0 sa " << b0.b.SurfaceArea() <<
//CO                " b1 sa " << b1.b.SurfaceArea();
//CO            LOG(WARNING) << "dim " << dim << ", bucket " << i << " Kr " << Kr <<
//CO                " cost " << cost[i];
        }

        // Find bucket to split at that minimizes SAH metric
        for (int i = 1; i < nBuckets - 1; ++i) {
            if (cost[i] > 0 && cost[i] < minCost) {
                minCost = cost[i];
                minCostSplitBucket = i;
                minCostSplitDim = dim;
            }
        }
    }

    int mid;
    if (minCostSplitDim == -1) {
        // TODO: stat for this instead?
//CO        LOG(WARNING) << "Found no good split...";
        // just split arbitrarily
        mid = (start + end) / 2;
    } else {
//CO        LOG(WARNING) << "Splitting dimension " << minCostSplitDim <<
//CO            " @ bucket " << minCostSplitBucket << " centroid bounds " <<
//CO            centroidBounds;
//CO        for (int i = start; i < end; ++i) {
//CO            LOG(WARNING) << "light " << i << ": info " << lightInfo[i] <<
//CO                ", centroid " << lightInfo[i].centroid() << " -> bucket " <<
//CO                nBuckets *
//CO                centroidBounds.Offset(lightInfo[i].centroid())[minCostSplitDim];
//CO        }

        LightInfo *pmid = std::partition(
            &lightInfo[start], &lightInfo[end - 1] + 1,
            [=](const LightInfo &li) {
                int b = nBuckets *
                    centroidBounds.Offset(li.centroid())[minCostSplitDim];
                if (b == nBuckets) b = nBuckets - 1;
                CHECK_GE(b, 0);
                CHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &lightInfo[0];
//CO        LOG(WARNING) << "mid @ " << mid;

        if (mid == start || mid == end) {
//CO            LOG(WARNING) << "feh: min dim " << minCostSplitDim <<
//CO                " @ bucket " << minCostSplitBucket;
//CO            for (int i = start; i < end; ++i) {
//CO                LOG(WARNING) << "light " << i << ": " << lightInfo[i];
//CO            }
            mid = (start + end) / 2;
        }
        CHECK(mid > start && mid < end);
    }

    LightBVHNode *node = alloc.new_object<LightBVHNode>(
        buildBVH(lightInfo, start, mid, alloc, nNodes),
        buildBVH(lightInfo, mid, end, alloc, nNodes));
    node->children[0]->parent = node->children[1]->parent = node;
    return node;
}

SampledLightVector BVHLightSampler::Sample(const Interaction &intr,
                                           Float u, MemoryArena &arena) const {
    ProfilerScope _(ProfilePhase::LightDistribLookup);

#ifdef __CUDA_ARCH__
    assert(!"unsupported on GPU");
#else
    ArenaAllocator<SampledLight> alloc(&arena);
    SampledLightVector vec(alloc);
    Sample(intr, u, [&](LightHandle light, Float pdf) {
                        vec.push_back({light, pdf});
                    });
    ReportValue(nLightsSampled, vec.size());
    return vec;
#endif
}

///////////////////////////////////////////////////////////////////////////
// ExhaustiveLightSampler

ExhaustiveLightSampler::ExhaustiveLightSampler(pstd::span<const LightHandle> lights,
                                               Allocator alloc)
    : LightSampler(lights, alloc), infiniteLights(alloc), lightBounds(alloc),
      lightToIndex(alloc) {
    std::vector<Float> prob(lights.size(), 0.f);
    for (const auto &light : lights) {
        pstd::optional<LightBounds> lb = light.Bounds();
        if (!lb) {
            infiniteLights.push_back(light);
        } else {
            lightToIndex.Insert(light, lightBounds.size());
            lightBounds.push_back(*lb);
        }
    }
}

SampledLightVector ExhaustiveLightSampler::Sample(
        const Interaction &intr, Float u, MemoryArena &arena) const {
#ifdef __CUDA_ARCH__
    assert(!"unsupported on GPU");
    ArenaAllocator<SampledLight> alloc(&arena);
    SampledLightVector vec(alloc);
    return vec;
#else
    std::vector<Float> prob(lights.size());
    for (size_t i = 0; i < lights.size(); ++i)
        prob[i] = lightBounds[i].Importance(intr);
    Distribution1D distrib(prob);

    Float pdf;
    int index = distrib.SampleDiscrete(u, &pdf);

    ArenaAllocator<SampledLight> alloc(&arena);
    SampledLightVector vec(alloc);
    vec.push_back({lights[index], pdf});
    for (LightHandle light : infiniteLights)
        vec.push_back({light, 1.f});
    return vec;
#endif
}

Float ExhaustiveLightSampler::PDF(const Interaction &intr,
                                  LightHandle light) const {
#ifdef __CUDA_ARCH__
    assert(!"unsupported on GPU");
    return {};
#else
    if (std::find(infiniteLights.begin(), infiniteLights.end(), light) !=
        infiniteLights.end())
        return 1;

    size_t lightIndex = lightToIndex[light];
    std::vector<Float> prob(lights.size());
    for (size_t i = 0; i < lights.size(); ++i)
        prob[i] = lightBounds[i].Importance(intr);
    Distribution1D distrib(prob);
    return distrib.DiscretePDF(lightIndex);
#endif
}

std::string ExhaustiveLightSampler::ToString() const {
    return StringPrintf("[ ExhaustiveLightSampler lightBounds: %s]", lightBounds);
}

}  // namespace pbrt
