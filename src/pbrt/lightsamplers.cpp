// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/lightsamplers.h>

#include <pbrt/interaction.h>
#include <pbrt/lights.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/spectrum.h>

#include <atomic>
#include <cstdint>
#include <numeric>
#include <vector>

namespace pbrt {

std::string SampledLight::ToString() const {
    return StringPrintf("[ SampledLight light: %s pdf: %f ]",
                        light ? light.ToString().c_str() : "(nullptr)", pdf);
}

LightSamplerHandle LightSamplerHandle::Create(const std::string &name,
                                              pstd::span<const LightHandle> lights,
                                              Allocator alloc) {
    if (name == "uniform")
        return alloc.new_object<UniformLightSampler>(lights, alloc);
    else if (name == "power")
        return alloc.new_object<PowerLightSampler>(lights, alloc);
    else if (name == "bvh")
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    else if (name == "exhaustive")
        return alloc.new_object<ExhaustiveLightSampler>(lights, alloc);
    else {
        Error(R"(Light sample distribution type "%s" unknown. Using "bvh".)",
              name.c_str());
        return alloc.new_object<BVHLightSampler>(lights, alloc);
    }
}

std::string LightSamplerHandle::ToString() const {
    if (!ptr())
        return "(nullptr)";

    auto ts = [&](auto ptr) { return ptr->ToString(); };
    return ApplyCPU<std::string>(ts);
}

///////////////////////////////////////////////////////////////////////////
// PowerLightSampler

PowerLightSampler::PowerLightSampler(pstd::span<const LightHandle> lights,
                                     Allocator alloc)
    : aliasTable(alloc),
      lights(lights.begin(), lights.end(), alloc),
      lightToIndex(alloc) {
    if (lights.empty())
        return;

    for (size_t i = 0; i < lights.size(); ++i)
        lightToIndex.Insert(lights[i], i);

    std::vector<Float> lightPower;
    SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
    for (const auto &light : lights)
        lightPower.push_back(light.Phi(lambda).Average());

#if 0
    // Power can be a bit tricky... e.g. one could create a triangle light
    // source with arbitrarily large area, but at the end of the day, it
    // can't subtend more than a hemisphere at a point. So we'll try to be
    // a little defensive about that and not let the probabilities get too
    // far skewed.  (See e.g. volpath-direct-geom-only-two-lights.pbrt
    // with PowerLightSampler...)
    //
    // This helps with that case, but seems to hurt in general...
    Float avgPower = std::accumulate(lightPower.begin(), lightPower.end(), 0.) /
        lights.size();
    for (Float &power : lightPower)
        power = std::max(power, .1f * avgPower);
#endif

    aliasTable = AliasTable(lightPower, alloc);
}

std::string PowerLightSampler::ToString() const {
    return StringPrintf("[ PowerLightSampler aliasTable: %s ]", aliasTable);
}

///////////////////////////////////////////////////////////////////////////
// BVHLightSampler

// TODO: make animated light sources an exercise

// For bvh build
STAT_MEMORY_COUNTER("Memory/Light BVH", lightBVHBytes);
STAT_INT_DISTRIBUTION("Integrator/Lights sampled per lookup", nLightsSampled);
STAT_INT_DISTRIBUTION("Integrator/Light BVH depth", bvhDepth);

BVHLightSampler::BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      infiniteLights(alloc),
      lightToNode(alloc) {
    std::vector<std::pair<LightHandle, LightBounds>> bvhLights;
    for (const auto &light : lights) {
        pstd::optional<LightBounds> lightBounds = light.Bounds();
        if (!lightBounds)
            infiniteLights.push_back(light);
        else if (lightBounds->phi > 0)
            bvhLights.push_back(std::make_pair(light, *lightBounds));
    }

    if (bvhLights.empty())
        return;

#if 0
    // Helps a bit with kitchen, but hurts some with measure-one, and tons
    // with contemporary-bathroom (tons of small emissive triangles in
    // light sources?)

    // Harmonize phi values
    double sumPhi = 0;
    for (const auto &bl : bvhLights)
        sumPhi += bl.second.phi;
    Float avgPhi = sumPhi / bvhLights.size();
    for (auto &bl : bvhLights)
        bl.second.phi = std::max(bl.second.phi, .1f * avgPhi);
#endif

    int nNodes = 0;
    root = buildBVH(bvhLights, 0, bvhLights.size(), alloc, &nNodes);
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
    return StringPrintf("[ BVHLightSampler root: %s ]",
                        root ? root->ToString() : std::string("(nullptr)"));
}

std::string LightBVHNode::ToString() const {
    std::string s =
        StringPrintf("[ LightBVHNode lightBounds: %s isLeaf: %s ", lightBounds, isLeaf);
    if (isLeaf)
        s += StringPrintf("light: %s ", light);
    else
        s += StringPrintf("children[0]: %s children[1]: %s", *children[0], *children[1]);
    return s + "]";
}

LightBVHNode *BVHLightSampler::buildBVH(
    std::vector<std::pair<LightHandle, LightBounds>> &lights, int start, int end,
    Allocator alloc, int *nNodes) {
    // CO    LOG(WARNING) << "Build range " << start << " - " << end;
    CHECK_LT(start, end);
    (*nNodes)++;
    int nLights = end - start;
    if (nLights == 1) {
        LightBVHNode *node =
            alloc.new_object<LightBVHNode>(lights[start].first, lights[start].second);
        lightToNode.Insert(lights[start].first, node);
        return node;
    }

    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        const LightBounds &lb = lights[i].second;
        bounds = Union(bounds, lb.b);
        centroidBounds = Union(centroidBounds, lb.Centroid());
    }
    // CO    LOG(WARNING) << "bounds: " << bounds << ", centroid bounds: " <<
    // CO        centroidBounds;

    // Modified SAH
    // Replace # of primitives with emitter power
    // TODO: use the more efficient bounds/cost sweep calculation from v4

    Float minCost = Infinity;
    int minCostSplitBucket = -1, minCostSplitDim = -1;
    constexpr int nBuckets = 12;

    for (int dim = 0; dim < 3; ++dim) {
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            // CO            LOG(WARNING) << "Degenerate bounds. dim = " << dim
            // << CO                " centroidBounds = " << centroidBounds << CO
            // " bounds = " << bounds << " start = " << start << CO " end = " <<
            // end;
            continue;
        }

        LightBounds bucketLightBounds[nBuckets];

        for (int i = start; i < end; ++i) {
            Point3f pc = lights[i].second.Centroid();
            int b = nBuckets * centroidBounds.Offset(pc)[dim];
            // CO            LOG(WARNING) << "Light " << i << ", bounds " <<
            // lightInfo[i].lightBounds << " -> bucket " << b;
            if (b == nBuckets)
                b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            bucketLightBounds[b] = Union(bucketLightBounds[b], lights[i].second);
        }

        // CO        for (int i = 0; i < nBuckets; ++i)
        // CO            LOG(WARNING) << "Bucket light bounds " << i << " -> "
        // << bucketLightBounds[i].b;

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
                       Pi / 2 *
                           (2 * theta_w * std::sin(b.theta_o) -
                            std::cos(b.theta_o - 2 * theta_w) -
                            2 * b.theta_o * std::sin(b.theta_o) + std::cos(b.theta_o));
            };

            // Can simplify since we always split
            Float Kr = MaxComponentValue(bounds.Diagonal()) / bounds.Diagonal()[dim];
            cost[i] = Kr * (b0.phi * Momega(b0) * b0.b.SurfaceArea() +
                            b1.phi * Momega(b1) * b1.b.SurfaceArea());
            // CO            if (b0.b.SurfaceArea() == 0 || b1.b.SurfaceArea()
            // == 0) CO                cost[i] = Kr * CO (b0.phi * Momega(b0) *
            // MaxComponent(b0.b.Diagonal()) + CO                     b1.phi *
            // Momega(b1) * MaxComponent(b1.b.Diagonal())); CO LOG(WARNING) <<
            // "dim " << dim << ", bucket " << i << " Kr " << Kr << CO " cost "
            // << cost[i] << " b0 " << b0 << ", b1 " << b1 << CO " momega 0 " <<
            // Momega(b0) << ", momega b1 " << Momega(b1) << CO " b0 sa " <<
            // b0.b.SurfaceArea() << CO                " b1 sa " <<
            // b1.b.SurfaceArea(); CO            LOG(WARNING) << "dim " << dim
            // <<
            // ", bucket " << i << " Kr " << Kr << CO                " cost " <<
            // cost[i];
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
        // CO        LOG(WARNING) << "Found no good split...";
        // just split arbitrarily
        mid = (start + end) / 2;
    } else {
        // CO        LOG(WARNING) << "Splitting dimension " << minCostSplitDim
        // << CO            " @ bucket " << minCostSplitBucket << " centroid
        // bounds " << CO            centroidBounds; CO        for (int i =
        // start; i < end; ++i) { CO            LOG(WARNING) << "light " << i <<
        // ": info " << lightInfo[i] << CO                ", centroid " <<
        // lightInfo[i].centroid() << " -> bucket " << CO nBuckets
        // * CO centroidBounds.Offset(lightInfo[i].centroid())[minCostSplitDim];
        // CO        }

        const auto *pmid = std::partition(
            &lights[start], &lights[end - 1] + 1,
            [=](const std::pair<LightHandle, LightBounds> &l) {
                int b = nBuckets *
                        centroidBounds.Offset(l.second.Centroid())[minCostSplitDim];
                if (b == nBuckets)
                    b = nBuckets - 1;
                CHECK_GE(b, 0);
                CHECK_LT(b, nBuckets);
                return b <= minCostSplitBucket;
            });
        mid = pmid - &lights[0];
        // CO        LOG(WARNING) << "mid @ " << mid;

        if (mid == start || mid == end) {
            // CO            LOG(WARNING) << "feh: min dim " << minCostSplitDim
            // << CO                " @ bucket " << minCostSplitBucket; CO for
            // (int i = start; i < end; ++i) { CO                LOG(WARNING) <<
            // "light " << i << ": " << lightInfo[i]; CO            }
            mid = (start + end) / 2;
        }
        CHECK(mid > start && mid < end);
    }

    LightBVHNode *node =
        alloc.new_object<LightBVHNode>(buildBVH(lights, start, mid, alloc, nNodes),
                                       buildBVH(lights, mid, end, alloc, nNodes));
    node->children[0]->parent = node->children[1]->parent = node;

    return node;
}

///////////////////////////////////////////////////////////////////////////
// ExhaustiveLightSampler

ExhaustiveLightSampler::ExhaustiveLightSampler(pstd::span<const LightHandle> lights,
                                               Allocator alloc)
    : lights(lights.begin(), lights.end(), alloc),
      boundedLights(alloc),
      infiniteLights(alloc),
      lightBounds(alloc),
      lightToBoundedIndex(alloc) {
    for (const auto &light : lights) {
        pstd::optional<LightBounds> lb = light.Bounds();
        if (lb) {
            lightToBoundedIndex.Insert(light, boundedLights.size());
            lightBounds.push_back(*lb);
            boundedLights.push_back(light);
        } else
            infiniteLights.push_back(light);
    }
}

pstd::optional<SampledLight> ExhaustiveLightSampler::Sample(const Interaction &intr,
                                                            Float u) const {
    Float sampleInfiniteProbability =
        Float(infiniteLights.size()) /
        Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    // Note: shared with BVH light sampler...
    if (u < sampleInfiniteProbability) {
        u = std::min<Float>(u * sampleInfiniteProbability, OneMinusEpsilon);
        int index = std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
        Float pdf = sampleInfiniteProbability * 1.f / infiniteLights.size();
        return SampledLight{infiniteLights[index], pdf};
    } else {
        u = std::min<Float>(
            (u - sampleInfiniteProbability) / (1 - sampleInfiniteProbability),
            OneMinusEpsilon);

        uint64_t seed = MixBits(FloatToBits(u));
        WeightedReservoirSampler<LightHandle> wrs(seed);

        for (size_t i = 0; i < boundedLights.size(); ++i)
            wrs.Add(boundedLights[i], lightBounds[i].Importance(intr));

        if (!wrs.HasSample())
            return {};

        Float pdf = (1.f - sampleInfiniteProbability) * wrs.Weight() / wrs.WeightSum();
        return SampledLight{wrs.GetSample(), pdf};
    }
}

Float ExhaustiveLightSampler::PDF(const Interaction &intr, LightHandle light) const {
    if (!lightToBoundedIndex.HasKey(light))
        return 1.f / (infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));

    Float importanceSum = 0;
    Float lightImportance = 0;
    for (size_t i = 0; i < boundedLights.size(); ++i) {
        Float importance = lightBounds[i].Importance(intr);
        importanceSum += importance;
        if (light == boundedLights[i])
            lightImportance = importance;
    }

    Float sampleInfiniteProbability =
        Float(infiniteLights.size()) /
        Float(infiniteLights.size() + (!lightBounds.empty() ? 1 : 0));
    Float pdf = lightImportance / importanceSum * (1. - sampleInfiniteProbability);
    return pdf;
}

std::string ExhaustiveLightSampler::ToString() const {
    return StringPrintf("[ ExhaustiveLightSampler lightBounds: %s]", lightBounds);
}

}  // namespace pbrt
