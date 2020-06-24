
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

#ifndef PBRT_SAMPLING_LIGHTS_H
#define PBRT_SAMPLING_LIGHTS_H

#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/lights.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace pbrt {

struct SampledLight {
    LightHandle light;
    Float pdf;

    std::string ToString() const;
};

using SampledLightVector = std::vector<SampledLight, ArenaAllocator<SampledLight>>;

// LightSampler defines a general interface for classes that provide
// probability distributions for sampling light sources at a given point in
// space.
class LightSampler {
  public:
    static LightSampler *Create(const std::string &name, pstd::span<const LightHandle> lights,
                                Allocator alloc);

    virtual ~LightSampler();

    PBRT_HOST_DEVICE
    virtual SampledLightVector Sample(const Interaction &intr,
                                      Float u, MemoryArena &arena) const = 0;

    // Returns the PDF for sampling the light |light| at the point |p|.
    PBRT_HOST_DEVICE
    virtual Float PDF(const Interaction &intr, LightHandle light) const = 0;

    virtual std::string ToString() const = 0;

 protected:
    LightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
        : lights(lights.begin(), lights.end(), alloc) {}

    pstd::vector<LightHandle> lights;
};

class AllLightsSampler final : public LightSampler {
 public:
    AllLightsSampler(pstd::span<const LightHandle> lights, Allocator alloc)
        : LightSampler(lights, alloc) {}

    PBRT_HOST_DEVICE
    SampledLightVector Sample(const Interaction &intr,
                              Float u, MemoryArena &arena) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &intr, LightHandle light) const {
        return 1;
    }

    std::string ToString() const { return "[ AllLightsSampler ]"; }
};

// FixedLightSampler is a LightSampler specialization for
// implementations that use the sampling probabilities at all points in the
// scene. Those just need to initialize a corresponding Distribution1D in
// their constructor and the implementation here takes care of the rest.
class FixedLightSampler : public LightSampler {
 public:
    virtual ~FixedLightSampler();

    // If the caller knows they have a FixedLightSampler, they can
    // sample and compute the PDF without providing a point.
    PBRT_HOST_DEVICE
    LightHandle Sample(Float u, Float *pdf) const {
        if (!distrib) {
            *pdf = 0;
            return nullptr;
        }

        int lightIndex = distrib->SampleDiscrete(u, pdf);
        return lights[lightIndex];
    }

    PBRT_HOST_DEVICE
    Float PDF(LightHandle light) const {
        if (!distrib)
            return 0;

        size_t index = lightToIndex[light];
        return distrib->DiscretePDF(index);
    }

    PBRT_HOST_DEVICE
    SampledLightVector Sample(const Interaction &intr, Float u,
                              MemoryArena &arena) const final;

    PBRT_HOST_DEVICE
    Float PDF(const Interaction &intr, LightHandle light) const final {
        return PDF(light);
    }

 protected:
    FixedLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    std::unique_ptr<Distribution1D> distrib;

 private:
    // Inverse mapping from elements in scene.lights to entries
    // in the |distrib| array.
    HashMap<LightHandle, size_t, LightHandleHash> lightToIndex;
};

// The simplest possible implementation of LightSampler: this uses a
// uniform distribution over all light sources. This approach works well
// for very simple scenes, but is quite ineffective for scenes with more
// than a handful of light sources. (This was the sampling method
// originally used for the PathIntegrator and the VolPathIntegrator in the
// printed book, though without the UniformLightSampler class.)
class UniformLightSampler final : public FixedLightSampler {
  public:
    UniformLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    std::string ToString() const { return "UniformLightSampler"; }
};

// PowerLightSampler returns a distribution with sampling probability
// proportional to the total emitted power for each light. (It also ignores
// the provided point |p|.)  This approach works well for scenes where
// there the most powerful lights are also the most important contributors
// to lighting in the scene, but doesn't do well if there are many lights
// and if different lights are relatively important in some areas of the
// scene and unimportant in others. (This was the default sampling method
// used for the BDPT integrator and MLT integrator in the printed book,
// though also without the PowerLightSampler class.)
class PowerLightSampler final : public FixedLightSampler {
  public:
    PowerLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    std::string ToString() const;
};

class LightInfo {
public:
    LightInfo(LightHandle light, LightBounds lightBounds)
        : light(light), lightBounds(lightBounds) { }

    std::string ToString() const {
        return StringPrintf("[ LightInfo: [ light: %s lightBounds: %s ] ]",
                            light, lightBounds);
    }

    LightHandle light;
    LightBounds lightBounds;
    Point3f centroid() const {
        return (lightBounds.b.pMin + lightBounds.b.pMax) / 2;
    }
};

class LightBVHNode {
public:
    LightBVHNode(const LightInfo &lightInfo)
        : lightBounds(lightInfo.lightBounds) {
        isLeaf = true;
        light = lightInfo.light;
    }
    LightBVHNode(LightBVHNode *c0, LightBVHNode *c1)
        : lightBounds(Union(c0->lightBounds, c1->lightBounds)) {
        isLeaf = false;
        children[0] = c0;
        children[1] = c1;
    }

    PBRT_HOST_DEVICE
    LightHandle Sample(const Interaction &ref, Float u, Float *pdf) const;

    std::string ToString() const {
        std::string s = StringPrintf("[ LightBVHNode lightBounds: %s isLeaf: %s ",
                                     lightBounds, isLeaf);
        if (isLeaf)
            s += StringPrintf("light: %s ", light);
        else
            s += StringPrintf("children[0]: %s children[1]: %s",
                              *children[0], *children[1]);
        return s + "]";
    }

    LightBounds lightBounds;
    bool isLeaf;
    union {
        LightHandle light;
        LightBVHNode *children[2];
    };
    LightBVHNode *parent = nullptr;
};

class BVHLightSampler final : public LightSampler {
 public:
    BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_HOST_DEVICE
    SampledLightVector Sample(const Interaction &intr, Float u,
                              MemoryArena &arena) const;

    template <typename F>
    PBRT_HOST_DEVICE
    void Sample(const Interaction &intr, Float u, F func) const {
        // First to improve execution convergence
        for (LightHandle light : infiniteLights)
            func(light, 1.f);
        if (root)
            SampleRecursive(root, intr, u, 1.f, func);
    }

    PBRT_HOST_DEVICE
    Float PDF(const Interaction &intr, LightHandle light) const {
        ProfilerScope _(ProfilePhase::LightDistribLookup);

#ifndef __CUDA_ARCH__
        // Why the hell is this being flagged as calling a host function?
        if (light->type == LightType::DeltaDirection ||
            light->type == LightType::Infinite)
            return 1;
#endif

        LightBVHNode *node = lightToNode[light];
        Float pdf = 1;

        if (node->lightBounds.Importance(intr) == 0)
            return 0;

        for (; node->parent != nullptr; node = node->parent) {
#ifndef __CUDA_ARCH__
            if (Inside(intr.p(), node->parent->children[0]->lightBounds.b) ||
                Inside(intr.p(), node->parent->children[1]->lightBounds.b))
                // PDF isn't affected since we always follow both children from
                // the parent in this case.
                continue;
#endif

            pstd::array<Float, 2> ci = {
                node->parent->children[0]->lightBounds.Importance(intr),
                node->parent->children[1]->lightBounds.Importance(intr)
            };
            int childIndex = static_cast<int>(node == node->parent->children[1]);
            DCHECK_GT(ci[childIndex], 0);
            pdf *= ci[childIndex] / (ci[0] + ci[1]);
        }

        return pdf;
    }

    template <typename F>
    PBRT_HOST_DEVICE
    void SampleSingle(const Interaction &intr, Float u, F func) const {
        // First to improve execution convergence
        for (LightHandle light : infiniteLights)
            func(light, 1.f);
        if (root)
            SampleSingleRecursive(root, intr, u, 1.f, func);
    }

    PBRT_HOST_DEVICE
    Float PDFSingle(const Interaction &intr, LightHandle light) const {
        ProfilerScope _(ProfilePhase::LightDistribLookup);

#ifndef __CUDA_ARCH__
        // Why the hell is this being flagged as calling a host function?
        if (light->type == LightType::DeltaDirection ||
            light->type == LightType::Infinite)
            return 1;
#endif

        LightBVHNode *node = lightToNode[light];
        Float pdf = 1;

        if (node->lightBounds.Importance(intr) == 0)
            return 0;

        for (; node->parent != nullptr; node = node->parent) {
            pstd::array<Float, 2> ci = {
                node->parent->children[0]->lightBounds.Importance(intr),
                node->parent->children[1]->lightBounds.Importance(intr)
            };
            int childIndex = static_cast<int>(node == node->parent->children[1]);
            DCHECK_GT(ci[childIndex], 0);
            pdf *= ci[childIndex] / (ci[0] + ci[1]);
        }

        return pdf;
    }

    std::string ToString() const;

 private:
    template <typename F>
    PBRT_HOST_DEVICE
    static void SampleRecursive(const LightBVHNode *node, const Interaction &intr,
                                Float u, Float pdf, F func) {
        if (node->isLeaf) {
            if (node->lightBounds.Importance(intr) > 0)
                func(node->light, pdf);
#ifndef __CUDA_ARCH__
        } else if (Inside(intr.p(), node->children[0]->lightBounds.b) ||
                   Inside(intr.p(), node->children[1]->lightBounds.b)) {
            // TODO?: use the variance-based test
            SampleRecursive(node->children[0], intr, u, pdf, func);
            SampleRecursive(node->children[1], intr, u, pdf, func);
#endif
        } else {
            pstd::array<Float, 2> ci = { node->children[0]->lightBounds.Importance(intr),
                                         node->children[1]->lightBounds.Importance(intr) };
            if (ci[0] == 0 && ci[1] == 0)
                // It may happen that we follow a path down the tree and later find that there
                // aren't any lights that illuminate our point; a natural consequence of the
                // bounds tightening up on the way down.
                return;

            Float nodePDF;
            int child = SampleDiscrete(ci, u, &nodePDF, &u);
            SampleRecursive(node->children[child], intr, u, pdf * nodePDF, func);
        }
    }

    template <typename F>
    PBRT_HOST_DEVICE
    static void SampleSingleRecursive(const LightBVHNode *node, const Interaction &intr,
                                      Float u, Float pdf, F func) {
        if (node->isLeaf) {
            if (node->lightBounds.Importance(intr) > 0)
                func(node->light, pdf);
        } else {
            pstd::array<Float, 2> ci = { node->children[0]->lightBounds.Importance(intr),
                                         node->children[1]->lightBounds.Importance(intr) };
            if (ci[0] == 0 && ci[1] == 0)
                // It may happen that we follow a path down the tree and later find that there
                // aren't any lights that illuminate our point; a natural consequence of the
                // bounds tightening up on the way down.
                return;

            Float nodePDF;
            int child = SampleDiscrete(ci, u, &nodePDF, &u);
            SampleSingleRecursive(node->children[child], intr, u, pdf * nodePDF, func);
        }
    }

    LightBVHNode *buildBVH(std::vector<LightInfo> &lightInfo,
                           int start, int end, Allocator alloc,
                           int *nNodes);

    LightBVHNode *root;
    HashMap<LightHandle, LightBVHNode *, LightHandleHash> lightToNode;
    pstd::vector<LightHandle> infiniteLights;
};

class ExhaustiveLightSampler final : public LightSampler {
 public:
    ExhaustiveLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_HOST_DEVICE
    SampledLightVector Sample(const Interaction &intr, Float u,
                              MemoryArena &arena) const;
    PBRT_HOST_DEVICE
    Float PDF(const Interaction &intr, LightHandle light) const;

    std::string ToString() const;

 private:
    pstd::vector<LightHandle> infiniteLights;
    pstd::vector<LightBounds> lightBounds;
    HashMap<LightHandle, size_t, LightHandleHash> lightToIndex; // in lightBounds
};

}  // namespace pbrt

#endif  // PBRT_SAMPLING_LIGHTS_H
