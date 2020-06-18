// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_SAMPLING_LIGHTS_H
#define PBRT_SAMPLING_LIGHTS_H

#include <pbrt/pbrt.h>

#include <pbrt/base/light.h>
#include <pbrt/base/lightsampler.h>
#include <pbrt/lights.h>  // LightBounds. Should that live elsewhere?
#include <pbrt/util/containers.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>
#include <string>

namespace pbrt {

struct LightHandleHash {
    PBRT_CPU_GPU
    size_t operator()(LightHandle lightHandle) const { return Hash(lightHandle.ptr()); }
};

class UniformLightSampler {
  public:
    UniformLightSampler(pstd::span<const LightHandle> lights, Allocator alloc)
        : lights(lights.begin(), lights.end(), alloc) {}

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};

        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const Interaction &intr, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PDF(const Interaction &intr, LightHandle light) const { return PDF(light); }

    std::string ToString() const { return "UniformLightSampler"; }

  private:
    pstd::vector<LightHandle> lights;
};

class PowerLightSampler {
  public:
    PowerLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (!aliasTable.size())
            return {};

        Float pdf;
        int lightIndex = aliasTable.Sample(u, &pdf);
        return SampledLight{lights[lightIndex], pdf};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (!aliasTable.size())
            return 0;

        size_t index = lightToIndex[light];
        return aliasTable.PDF(index);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const Interaction &intr, Float u) const {
        return Sample(u);
    }

    PBRT_CPU_GPU
    Float PDF(const Interaction &intr, LightHandle light) const { return PDF(light); }

    std::string ToString() const;

  private:
    AliasTable aliasTable;
    pstd::vector<LightHandle> lights;
    HashMap<LightHandle, size_t, LightHandleHash> lightToIndex;
};

class LightBVHNode {
  public:
    LightBVHNode(LightHandle light, const LightBounds &lightBounds)
        : light(light), lightBounds(lightBounds) {
        isLeaf = true;
    }
    LightBVHNode(LightBVHNode *c0, LightBVHNode *c1)
        : lightBounds(Union(c0->lightBounds, c1->lightBounds)) {
        isLeaf = false;
        children[0] = c0;
        children[1] = c1;
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const Interaction &ref, Float u) const;

    std::string ToString() const;

    LightBounds lightBounds;
    bool isLeaf;
    union {
        LightHandle light;
        LightBVHNode *children[2];
    };
    LightBVHNode *parent = nullptr;
};

class BVHLightSampler {
  public:
    BVHLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const Interaction &intr, Float u) const {
        // FIXME: handle no lights at all w/o a NaN...
        Float sampleInfiniteProbability =
            Float(infiniteLights.size()) /
            Float(infiniteLights.size() + (root != nullptr ? 1 : 0));

        if (u < sampleInfiniteProbability) {
            u = std::min<Float>(u * sampleInfiniteProbability, OneMinusEpsilon);
            int index =
                std::min<int>(u * infiniteLights.size(), infiniteLights.size() - 1);
            Float pdf = sampleInfiniteProbability * 1.f / infiniteLights.size();
            return SampledLight{infiniteLights[index], pdf};
        } else {
            if (root == nullptr)
                return {};

            u = std::min<Float>(
                (u - sampleInfiniteProbability) / (1 - sampleInfiniteProbability),
                OneMinusEpsilon);
            pstd::optional<SampledLight> sampledLight =
                SampleRecursive(root, intr, u, 1.f);
            if (sampledLight)
                sampledLight->pdf *= (1 - sampleInfiniteProbability);
            return sampledLight;
        }
    }

    PBRT_CPU_GPU
    Float PDF(const Interaction &intr, LightHandle light) const {
        if (!lightToNode.HasKey(light))
            return 1.f / (infiniteLights.size() + (root != nullptr ? 1 : 0));

        LightBVHNode *node = lightToNode[light];
        Float pdf = 1;

        if (node->lightBounds.Importance(intr) == 0)
            return 0;

        for (; node->parent != nullptr; node = node->parent) {
            pstd::array<Float, 2> ci = {
                node->parent->children[0]->lightBounds.Importance(intr),
                node->parent->children[1]->lightBounds.Importance(intr)};
            int childIndex = static_cast<int>(node == node->parent->children[1]);
            DCHECK_GT(ci[childIndex], 0);
            pdf *= ci[childIndex] / (ci[0] + ci[1]);
        }

        Float sampleInfiniteProbability =
            Float(infiniteLights.size()) / Float(infiniteLights.size() + 1);
        return pdf * (1.f - sampleInfiniteProbability);
    }

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};

        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    std::string ToString() const;

  private:
    PBRT_CPU_GPU
    static pstd::optional<SampledLight> SampleRecursive(const LightBVHNode *node,
                                                        const Interaction &intr, Float u,
                                                        Float pdf) {
        if (node->isLeaf) {
            if (node->lightBounds.Importance(intr) == 0)
                return {};
            return SampledLight{node->light, pdf};
        } else {
            pstd::array<Float, 2> ci = {node->children[0]->lightBounds.Importance(intr),
                                        node->children[1]->lightBounds.Importance(intr)};
            if (ci[0] == 0 && ci[1] == 0)
                // It may happen that we follow a path down the tree and later
                // find that there aren't any lights that illuminate our point;
                // a natural consequence of the bounds tightening up on the way
                // down.
                return {};

            Float nodePDF;
            int child = SampleDiscrete(ci, u, &nodePDF, &u);
            return SampleRecursive(node->children[child], intr, u, pdf * nodePDF);
        }
    }

    LightBVHNode *buildBVH(std::vector<std::pair<LightHandle, LightBounds>> &lights,
                           int start, int end, Allocator alloc, int *nNodes);

    LightBVHNode *root = nullptr;
    pstd::vector<LightHandle> lights, infiniteLights;
    HashMap<LightHandle, LightBVHNode *, LightHandleHash> lightToNode;
};

class ExhaustiveLightSampler {
  public:
    ExhaustiveLightSampler(pstd::span<const LightHandle> lights, Allocator alloc);

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(const Interaction &intr, Float u) const;

    PBRT_CPU_GPU
    Float PDF(const Interaction &intr, LightHandle light) const;

    PBRT_CPU_GPU
    pstd::optional<SampledLight> Sample(Float u) const {
        if (lights.empty())
            return {};

        int lightIndex = std::min<int>(u * lights.size(), lights.size() - 1);
        return SampledLight{lights[lightIndex], 1.f / lights.size()};
    }

    PBRT_CPU_GPU
    Float PDF(LightHandle light) const {
        if (lights.empty())
            return 0;
        return 1.f / lights.size();
    }

    std::string ToString() const;

  private:
    pstd::vector<LightHandle> lights, boundedLights, infiniteLights;
    pstd::vector<LightBounds> lightBounds;
    HashMap<LightHandle, size_t, LightHandleHash> lightToBoundedIndex;
};

inline pstd::optional<SampledLight> LightSamplerHandle::Sample(const Interaction &intr,
                                                               Float u) const {
    auto s = [&](auto ptr) { return ptr->Sample(intr, u); };
    return Apply<pstd::optional<SampledLight>>(s);
}

inline Float LightSamplerHandle::PDF(const Interaction &intr, LightHandle light) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(intr, light); };
    return Apply<Float>(pdf);
}

inline pstd::optional<SampledLight> LightSamplerHandle::Sample(Float u) const {
    auto sample = [&](auto ptr) { return ptr->Sample(u); };
    return Apply<pstd::optional<SampledLight>>(sample);
}

inline Float LightSamplerHandle::PDF(LightHandle light) const {
    auto pdf = [&](auto ptr) { return ptr->PDF(light); };
    return Apply<Float>(pdf);
}

}  // namespace pbrt

#endif  // PBRT_SAMPLING_LIGHTS_H
