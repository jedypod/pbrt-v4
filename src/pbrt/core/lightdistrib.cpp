
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

// TODO (maybe): have integrators pre-prime the cache by rendering a very
// low res image first?

#include <pbrt/core/lightdistrib.h>

#include <pbrt/core/error.h>
#include <pbrt/core/interaction.h>
#include <pbrt/core/integrator.h> // FIXME only for ComputeLightPowerDistribution()
#include <pbrt/core/light.h>
#include <pbrt/core/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/core/sampling.h>
#include <pbrt/core/scene.h>
#include <pbrt/core/spectrum.h>
#include <pbrt/util/stats.h>
#include <glog/logging.h>

#include <atomic>
#include <numeric>
#include <cstdint>
#include <vector>

namespace pbrt {

LightDistribution::~LightDistribution() {}

std::unique_ptr<LightDistribution> CreateLightSampleDistribution(
    const std::string &name, const Scene &scene) {
    if (name == "uniform" || scene.lights.size() == 1)
        return std::make_unique<UniformLightDistribution>(scene);
    else if (name == "bvh")
        return std::make_unique<BVHLightDistribution>(scene);
    else if (name == "exhaustive")
        return std::make_unique<ExhaustiveLightDistribution>(scene);
    else if (name == "power")
        return std::make_unique<PowerLightDistribution>(scene);
    else if (name == "spatial")
        return std::make_unique<SpatialLightDistribution>(scene);
    else {
        Error(
            "Light sample distribution type \"%s\" unknown. Using \"spatial\".",
            name.c_str());
        return std::make_unique<SpatialLightDistribution>(scene);
    }
}

///////////////////////////////////////////////////////////////////////////
// FixedLightDistribution

FixedLightDistribution::FixedLightDistribution(const Scene &scene)
    : LightDistribution(scene) {
    // Compute a reverse mapping from light pointers to offsets into the
    // scene lights vector (and, equivalently, offsets into lightDistr).
    for (size_t i = 0; i < scene.lights.size(); ++i)
        lightToIndex[scene.lights[i].get()] = i;
}

FixedLightDistribution::~FixedLightDistribution() {}

const Light *FixedLightDistribution::Sample(Float u, Float *pdf) const {
    if (!distrib) {
        *pdf = 0;
        return nullptr;
    }

    int lightIndex = distrib->SampleDiscrete(u, pdf);
    return scene.lights[lightIndex].get();
}

Float FixedLightDistribution::Pdf(const Light *light) const {
    if (!distrib)
        return 0;

    auto iter = lightToIndex.find(light);
    CHECK(iter != lightToIndex.end());
    size_t index = iter->second;
    return distrib->DiscretePDF(index);
}

///////////////////////////////////////////////////////////////////////////
// UniformLightDistribution

UniformLightDistribution::UniformLightDistribution(const Scene &scene)
    : FixedLightDistribution(scene) {
    if (scene.lights.size() == 0) return;
    std::vector<Float> prob(scene.lights.size(), Float(1));
    distrib = std::make_unique<Distribution1D>(prob);
}

///////////////////////////////////////////////////////////////////////////
// PowerLightDistribution

PowerLightDistribution::PowerLightDistribution(const Scene &scene)
    : FixedLightDistribution(scene) {
    if (scene.lights.size() == 0) return;

    std::vector<Float> lightPower;
    for (const auto &light : scene.lights)
        lightPower.push_back(light->Phi().y());
    distrib = std::make_unique<Distribution1D>(lightPower);
}

///////////////////////////////////////////////////////////////////////////
// SpatialLightDistribution

STAT_COUNTER("SpatialLightDistribution/Distributions created", nCreated);
STAT_RATIO("SpatialLightDistribution/Lookups per distribution", nLookups, nDistributions);
STAT_INT_DISTRIBUTION("SpatialLightDistribution/Hash probes per lookup", nProbesPerLookup);

// Voxel coordinates are packed into a uint64_t for hash table lookups;
// 10 bits are allocated to each coordinate.  invalidPackedPos is an impossible
// packed coordinate value, which we use to represent
static const uint64_t invalidPackedPos = 0xffffffffffffffff;

SpatialLightDistribution::SpatialLightDistribution(const Scene &scene,
                                                   int maxVoxels)
    : LightDistribution(scene) {
    // Compute the number of voxels so that the widest scene bounding box
    // dimension has maxVoxels voxels and the other dimensions have a number
    // of voxels so that voxels are roughly cube shaped.
    Bounds3f b = scene.WorldBound();
    Vector3f diag = b.Diagonal();
    Float bmax = diag[b.MaximumExtent()];
    for (int i = 0; i < 3; ++i) {
        nVoxels[i] = std::max(1, int(std::round(diag[i] / bmax * maxVoxels)));
        // In the Lookup() method, we require that 20 or fewer bits be
        // sufficient to represent each coordinate value. It's fairly hard
        // to imagine that this would ever be a problem.
        CHECK_LT(nVoxels[i], 1 << 20);
    }

    hashTableSize = 4 * nVoxels[0] * nVoxels[1] * nVoxels[2];
    hashTable = std::make_unique<HashEntry[]>(hashTableSize);
    for (int i = 0; i < hashTableSize; ++i) {
        hashTable[i].packedPos.store(invalidPackedPos);
        hashTable[i].distribution.store(nullptr);
    }

    for (size_t i = 0; i < scene.lights.size(); ++i)
        lightToIndex[scene.lights[i].get()] = i;

    LOG(INFO) << "SpatialLightDistribution: scene bounds " << b <<
        ", voxel res (" << nVoxels[0] << ", " << nVoxels[1] << ", " <<
        nVoxels[2] << ")";
}

SpatialLightDistribution::~SpatialLightDistribution() {
    // Gather statistics about how well the computed distributions are across
    // the buckets.
    for (size_t i = 0; i < hashTableSize; ++i) {
        HashEntry &entry = hashTable[i];
        if (entry.distribution.load())
            delete entry.distribution.load();
    }
}

const Light *SpatialLightDistribution::Sample(const Point3f &p, Float u, Float *pdf) const {
    if (scene.lights.size() == 0) {
        *pdf = 0;
        return nullptr;
    }
    const Distribution1D *distrib = GetDistribution(p);
    int lightIndex = distrib->SampleDiscrete(u, pdf);
    return scene.lights[lightIndex].get();
}

Float SpatialLightDistribution::Pdf(const Point3f &p, const Light *light) const {
    if (scene.lights.size() == 0)
        return 0;

    const Distribution1D *distrib = GetDistribution(p);

    auto iter = lightToIndex.find(light);
    CHECK(iter != lightToIndex.end());
    size_t index = iter->second;
    return distrib->DiscretePDF(index);
}

const Distribution1D *SpatialLightDistribution::GetDistribution(const Point3f &p) const {
    ProfilePhase _(Prof::LightDistribLookup);
    ++nLookups;

    // First, compute integer voxel coordinates for the given point |p|
    // with respect to the overall voxel grid.
    Vector3f offset = scene.WorldBound().Offset(p);  // offset in [0,1].
    Point3i pi;
    for (int i = 0; i < 3; ++i)
        // The clamp should almost never be necessary, but is there to be
        // robust to computed intersection points being slightly outside
        // the scene bounds due to floating-point roundoff error.
        pi[i] = Clamp(int(offset[i] * nVoxels[i]), 0, nVoxels[i] - 1);

    // Pack the 3D integer voxel coordinates into a single 64-bit value.
    uint64_t packedPos = (uint64_t(pi[0]) << 40) | (uint64_t(pi[1]) << 20) | pi[2];
    CHECK_NE(packedPos, invalidPackedPos);

    // Compute a hash value from the packed voxel coordinates.  We could
    // just take packedPos mod the hash table size, but since packedPos
    // isn't necessarily well distributed on its own, it's worthwhile to do
    // a little work to make sure that its bits values are individually
    // fairly random. For details of and motivation for the following, see:
    // http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
    uint64_t hash = packedPos;
    hash ^= (hash >> 31);
    hash *= 0x7fb5d329728ea185;
    hash ^= (hash >> 27);
    hash *= 0x81dadef4bc2dd44d;
    hash ^= (hash >> 33);
    hash %= hashTableSize;
    CHECK_GE(hash, 0);

    // Now, see if the hash table already has an entry for the voxel. We'll
    // use quadratic probing when the hash table entry is already used for
    // another value; step stores the square root of the probe step.
    int step = 1;
    int nProbes = 0;
    while (true) {
        ++nProbes;
        HashEntry &entry = hashTable[hash];
        // Does the hash table entry at offset |hash| match the current point?
        uint64_t entryPackedPos = entry.packedPos.load(std::memory_order_acquire);
        if (entryPackedPos == packedPos) {
            // Yes! Most of the time, there should already by a light
            // sampling distribution available.
            Distribution1D *dist = entry.distribution.load(std::memory_order_acquire);
            if (dist == nullptr) {
                // Rarely, another thread will have already done a lookup
                // at this point, found that there isn't a sampling
                // distribution, and will already be computing the
                // distribution for the point.  In this case, we spin until
                // the sampling distribution is ready.  We assume that this
                // is a rare case, so don't do anything more sophisticated
                // than spinning.
                ProfilePhase _(Prof::LightDistribSpinWait);
                while ((dist = entry.distribution.load(std::memory_order_acquire)) ==
                       nullptr)
                    // spin :-(. If we were fancy, we'd have any threads
                    // that hit this instead help out with computing the
                    // distribution for the voxel...
                    ;
            }
            // We have a valid sampling distribution.
            ReportValue(nProbesPerLookup, nProbes);
            return dist;
        } else if (entryPackedPos != invalidPackedPos) {
            // The hash table entry we're checking has already been
            // allocated for another voxel. Advance to the next entry with
            // quadratic probing.
            hash += step * step;
            if (hash >= hashTableSize)
                hash %= hashTableSize;
            ++step;
        } else {
            // We have found an invalid entry. (Though this may have
            // changed since the load into entryPackedPos above.)  Use an
            // atomic compare/exchange to try to claim this entry for the
            // current position.
            uint64_t invalid = invalidPackedPos;
            if (entry.packedPos.compare_exchange_weak(invalid, packedPos)) {
                // Success; we've claimed this position for this voxel's
                // distribution. Now compute the sampling distribution and
                // add it to the hash table. As long as packedPos has been
                // set but the entry's distribution pointer is nullptr, any
                // other threads looking up the distribution for this voxel
                // will spin wait until the distribution pointer is
                // written.
                Distribution1D *dist = ComputeDistribution(pi);
                entry.distribution.store(dist, std::memory_order_release);
                ReportValue(nProbesPerLookup, nProbes);
                return dist;
            }
        }
    }
}

Distribution1D *
SpatialLightDistribution::ComputeDistribution(Point3i pi) const {
    ProfilePhase _(Prof::LightDistribCreation);
    ++nCreated;
    ++nDistributions;

    // Compute the world-space bounding box of the voxel corresponding to
    // |pi|.
    Point3f p0(Float(pi[0]) / Float(nVoxels[0]),
               Float(pi[1]) / Float(nVoxels[1]),
               Float(pi[2]) / Float(nVoxels[2]));
    Point3f p1(Float(pi[0] + 1) / Float(nVoxels[0]),
               Float(pi[1] + 1) / Float(nVoxels[1]),
               Float(pi[2] + 1) / Float(nVoxels[2]));
    Bounds3f voxelBounds(scene.WorldBound().Lerp(p0),
                         scene.WorldBound().Lerp(p1));

    // Compute the sampling distribution. Sample a number of points inside
    // voxelBounds using a 3D Halton sequence; at each one, sample each
    // light source and compute a weight based on Li/pdf for the light's
    // sample (ignoring visibility between the point in the voxel and the
    // point on the light source) as an approximation to how much the light
    // is likely to contribute to illumination in the voxel.
    int nSamples = 128;
    std::vector<Float> lightContrib(scene.lights.size(), Float(0));
    for (int i = 0; i < nSamples; ++i) {
        Point3f po = voxelBounds.Lerp(Point3f(
            RadicalInverse(0, i), RadicalInverse(1, i), RadicalInverse(2, i)));
        Interaction intr(po, Normal3f(), Vector3f(), Vector3f(1, 0, 0),
                         0 /* time */, MediumInterface());

        // Use the next two Halton dimensions to sample a point on the
        // light source.
        Point2f u(RadicalInverse(3, i), RadicalInverse(4, i));
        for (size_t j = 0; j < scene.lights.size(); ++j) {
            Float pdf;
            Vector3f wi;
            VisibilityTester vis;
            Spectrum Li = scene.lights[j]->Sample_Li(intr, u, &wi, &pdf, &vis);
            if (pdf > 0) {
                // TODO: look at tracing shadow rays / computing beam
                // transmittance.  Probably shouldn't give those full weight
                // but instead e.g. have an occluded shadow ray scale down
                // the contribution by 10 or something.
                lightContrib[j] += Li.y() / pdf;
            }
        }
    }

    // We don't want to leave any lights with a zero probability; it's
    // possible that a light contributes to points in the voxel even though
    // we didn't find such a point when sampling above.  Therefore, compute
    // a minimum (small) weight and ensure that all lights are given at
    // least the corresponding probability.
    Float sumContrib =
        std::accumulate(lightContrib.begin(), lightContrib.end(), Float(0));
    Float avgContrib = sumContrib / (nSamples * lightContrib.size());
    Float minContrib = (avgContrib > 0) ? .001 * avgContrib : 1;
    for (size_t i = 0; i < lightContrib.size(); ++i) {
        VLOG(2) << "Voxel pi = " << pi << ", light " << i << " contrib = "
                << lightContrib[i];
        lightContrib[i] = std::max(lightContrib[i], minContrib);
    }
    LOG(INFO) << "Initialized light distribution in voxel pi= " <<  pi <<
        ", avgContrib = " << avgContrib;

    // Compute a sampling distribution from the accumulated contributions.
    return new Distribution1D(lightContrib);
}

///////////////////////////////////////////////////////////////////////////
// BVHLightDistribution

/*
// Shirley TOG direct lighting paper: discuses light source sampling well
// including tentative contribution of each light to form PDF, spatial
// suvdiv, ...

He also cites related work before.

Spatial subdivision idea.  Computing isosurfaces of contribution up to some distance.

There's also a recent jcgt paper on stochastic light culling.

Keller learning stuff

*/

// TODO: add Bounds3f Light::WorldBound(/* TODO axis angles stuff */
// Vector3f *axis, Float *thetaX, Float *thetaY) /// ???????

// TODO: infinite lights? distant lights? (lights w/o bounds...)
// can these be handled as special ones at the top?

class LightInfo {
public:
    LightInfo(const Light *light)
        : light(light) {
        bounds = light->Bounds();
        centroid = .5f * bounds.worldBound.pMin + .5f * bounds.worldBound.pMax;
    }

    const Light *light;
    LightBounds bounds;
    Point3f centroid;
    // todo: axis and theta angles
};

class LightBVHNode {
public:
    LightBVHNode(LightInfo *lightInfo) {
        isLeaf = true;
        light = lightInfo->light;
        bounds = lightInfo->bounds.worldBound;
        maxLiContrib = lightInfo->bounds.maxLiContrib.y();
    }
    LightBVHNode(LightBVHNode *c0, LightBVHNode *c1) {
        isLeaf = false;
        children[0] = c0;
        children[1] = c1;
        bounds = Union(c0->bounds, c1->bounds);
        maxLiContrib = c0->maxLiContrib + c1->maxLiContrib;
    }

    const Light *Sample(const Interaction &ref, Float u,
                        Float *pdf) const;

    Bounds3f bounds;
    Float maxLiContrib;
    // todo: axis and theta angles

    bool isLeaf;
    union {
        const Light *light;
        LightBVHNode *children[2];
    };
    LightBVHNode *parent = nullptr;
};

BVHLightDistribution::BVHLightDistribution(const Scene &scene)
    : LightDistribution(scene) {
    std::vector<LightInfo> lightInfo;
    for (const auto &light : scene.lights)
        lightInfo.push_back(LightInfo(light.get()));

    static MemoryArena arena;  // YOLO FIXME
    int nNodes = 0;
    root = buildBVH(lightInfo, 0, lightInfo.size(), arena, &nNodes);

    // TODO: flatten (or keep the arena around!)
    // Note: have to do lightToNode for flattened nodes, if that's what we do...
}

LightBVHNode *BVHLightDistribution::buildBVH(std::vector<LightInfo> &lightInfo,
                                             int start, int end, MemoryArena &arena,
                                             int *nNodes) {
    CHECK_LT(start, end);
    (*nNodes)++;
    int nLights = end - start;
    if (nLights == 1) {
        LightBVHNode *node = arena.Alloc<LightBVHNode>(&lightInfo[start]);
        lightToNode[lightInfo[start].light] = node;
        return node;
    }

    Bounds3f bounds, centroidBounds;
    for (int i = start; i < end; ++i) {
        bounds = Union(bounds, lightInfo[i].bounds.worldBound);
        centroidBounds = Union(centroidBounds, lightInfo[i].centroid);
    }
    int mid;
    int dim = centroidBounds.MaximumExtent();
    if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
        LOG(WARNING) << "Degenerate bounds. dim = " << dim <<
            " bounds = " << centroidBounds << " start = " << start <<
            " end = " << end;
        mid = (start + end) / 2;
//CO        LOG(FATAL) << "TODO degenerate bounds in dim";
    } else {
    // Modified SAH
    // Replace # of primitives with emitter power
    // TODO: more efficient bounds/cost sweep calculation from v4

    struct BucketInfo {
        Float maxLiContrib = 0;
        Bounds3f bounds;
        // TODO: axis, angle bounds
    };

    constexpr int nBuckets = 12;
    BucketInfo buckets[nBuckets];

    for (int i = start; i < end; ++i) {
        int b = nBuckets *
            centroidBounds.Offset(lightInfo[i].centroid)[dim];
        if (b == nBuckets) b = nBuckets - 1;
        CHECK_GE(b, 0);
        CHECK_LT(b, nBuckets);
        buckets[b].maxLiContrib += lightInfo[i].bounds.maxLiContrib.y();
        buckets[b].bounds =
            Union(buckets[b].bounds, lightInfo[i].bounds.worldBound);
    }

    // Compute costs for splitting after each bucket
    Float cost[nBuckets - 1];
    for (int i = 0; i < nBuckets - 1; ++i) {
        Bounds3f b0, b1;
        int maxLi0 = 0, maxLi1 = 0;
        for (int j = 0; j <= i; ++j) {
            b0 = Union(b0, buckets[j].bounds);
            maxLi0 += buckets[j].maxLiContrib;
        }
        for (int j = i + 1; j < nBuckets; ++j) {
            b1 = Union(b1, buckets[j].bounds);
            maxLi1 += buckets[j].maxLiContrib;
        }
        // Can simplify since we always split
        cost[i] = maxLi0 * b0.SurfaceArea() + maxLi1 * b1.SurfaceArea();
    }

    // Find bucket to split at that minimizes SAH metric
    Float minCost = cost[0];
    int minCostSplitBucket = 0;
    for (int i = 1; i < nBuckets - 1; ++i) {
        if (cost[i] < minCost) {
            minCost = cost[i];
            minCostSplitBucket = i;
        }
    }

    LightInfo *pmid = std::partition(
        &lightInfo[start], &lightInfo[end - 1] + 1,
        [=](const LightInfo &li) {
            int b = nBuckets * centroidBounds.Offset(li.centroid)[dim];
            if (b == nBuckets) b = nBuckets - 1;
            CHECK_GE(b, 0);
            CHECK_LT(b, nBuckets);
            return b <= minCostSplitBucket;
        });
    mid = pmid - &lightInfo[0];

    if (mid == start || mid == end)
        mid = (start + end) / 2;
    }

    LightBVHNode *node = arena.Alloc<LightBVHNode>(
        buildBVH(lightInfo, start, mid, arena, nNodes),
        buildBVH(lightInfo, mid, end, arena, nNodes));
    node->children[0]->parent = node->children[1]->parent = node;
    return node;
}

static std::array<Float, 2> computeImportance(const Point3f &p, const LightBVHNode * const children[2]) {
    Float d2[2] = { DistanceSquared(p, children[0]->bounds),
                    DistanceSquared(p, children[1]->bounds) };
    // TODO: think through what's right to do here
    //
    // Sorta want to keep going until we get to solid ground, peel off the
    // subtree that is valid, then choose from a (small?) array of better
    // grounded options...
    if (d2[0] == 0) d2[0] = .1 * d2[1];
    if (d2[1] == 0) d2[1] = .1 * d2[0];
    if (d2[0] == 0)
        // both zero. Could be overlapping bounds? In any case, the 1/r^2
        // term doesn't matter...
        d2[0] = d2[1] = 1;

    // TODO: once the importance function starts getting more
    // sophisticated, I think it will be possible to have cases where we
    // recurse into a node with > 0 importance but then both children have
    // zero importance.  Need to think about / handle this case...
    std::array<Float, 2> importance;
    for (int c = 0; c < 2; ++c) {
        if (/**************/ false && children[c]->isLeaf)
            importance[c] = children[c]->light->MaxLiContribution(p).y();
        else {
            importance[c] = children[c]->maxLiContrib / d2[c];
            CHECK_GT(importance[c], 0);
        }
    }
    return importance;
}

const Light *sampleRecursive(const LightBVHNode *node, const Point3f &p,
                             Float u, Float *pdf) {
    if (node->isLeaf)
        return node->light;

    std::array<Float, 2> importance = computeImportance(p, node->children);
    if (importance[0] == 0 && importance[1] == 0) {
        LOG(WARNING) << "Hit 0/0 importances...";
        *pdf = 0;
        return nullptr;
    }
    Float p0 = importance[0] / (importance[0] + importance[1]);

    if (u < p0) {
        *pdf *= p0;
        u /= p0; // remap (TODO helper func?)
        return sampleRecursive(node->children[0], p, u, pdf);
    } else {
        *pdf *= 1 - p0;
        u = Clamp((u - p0) / (1 - p0), 0, 1);
        return sampleRecursive(node->children[1], p, u, pdf);
    }
}

const Light *BVHLightDistribution::Sample(const Point3f &p, Float u, Float *pdf) const {
    ProfilePhase _(Prof::LightDistribLookup);
    *pdf = 1;
    return sampleRecursive(root, p, u, pdf);
}

Float BVHLightDistribution::Pdf(const Point3f &p, const Light *light) const {
    ProfilePhase _(Prof::LightDistribLookup);

    const auto iter = lightToNode.find(light);
    CHECK(iter != lightToNode.end());
    LightBVHNode *node = iter->second;
    Float pdf = 1;

    while (node->parent) {
        std::array<Float, 2> importance = computeImportance(p, node->parent->children);
        int childIndex = (node == node->parent->children[1]);
        CHECK_NE(importance[childIndex], 0);
        pdf *= importance[childIndex] / (importance[0] + importance[1]);
        node = node->parent;
    }

    return pdf;
}

///////////////////////////////////////////////////////////////////////////
// ExhaustiveLightDistribution

ExhaustiveLightDistribution::ExhaustiveLightDistribution(const Scene &scene)
    : LightDistribution(scene) {}

const Light *ExhaustiveLightDistribution::Sample(const Point3f &p, Float u,
                                                 Float *pdf) const {
    std::vector<Float> lp(scene.lights.size(), Float(0));
    for (int i = 0; i < scene.lights.size(); ++i)
        lp[i] = scene.lights[i]->MaxLiContribution(p).y();

    Distribution1D dist(lp);
    int index = dist.SampleDiscrete(u, pdf);
    return scene.lights[index].get();
}

Float ExhaustiveLightDistribution::Pdf(const Point3f &p, const Light *light) const {
    LOG(FATAL) << "TODO";
    return 0;
}

}  // namespace pbrt
