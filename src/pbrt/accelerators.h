
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

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ACCELERATORS_BVH_H
#define PBRT_ACCELERATORS_BVH_H

// accelerators/bvh.h*
#include <pbrt/pbrt.h>
#include <pbrt/primitive.h>

#include <atomic>
#include <memory>
#include <vector>

namespace pbrt {

PrimitiveHandle CreateAccelerator(const std::string &name, std::vector<PrimitiveHandle> prims,
                                  const ParameterDictionary &dict);

struct BVHBuildNode;

// BVHAccel Forward Declarations
struct BVHPrimitiveInfo;
struct MortonPrimitive;
struct LinearBVHNode;

// BVHAccel Declarations
class BVHAccel : public Primitive {
  public:
    // BVHAccel Public Types
    enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

    // BVHAccel Public Methods
    BVHAccel(std::vector<PrimitiveHandle> p,
             int maxPrimsInNode = 1,
             SplitMethod splitMethod = SplitMethod::SAH);

    static BVHAccel *Create(std::vector<PrimitiveHandle> prims,
                            const ParameterDictionary &dict);

    Bounds3f WorldBound() const;
    ~BVHAccel();
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // BVHAccel Private Methods
    BVHBuildNode *recursiveBuild(
        std::vector<MemoryArena> &threadArenas,
        std::vector<BVHPrimitiveInfo> &primitiveInfo,
        int start, int end, std::atomic<int> *totalNodes,
        std::vector<PrimitiveHandle> &orderedPrims,
        std::atomic<int> *orderedPrimsOffset);
    BVHBuildNode *HLBVHBuild(
        MemoryArena &arena, const std::vector<BVHPrimitiveInfo> &primitiveInfo,
        std::atomic<int> *totalNodes,
        std::vector<PrimitiveHandle> &orderedPrims);
    BVHBuildNode *emitLBVH(
        BVHBuildNode *&buildNodes,
        const std::vector<BVHPrimitiveInfo> &primitiveInfo,
        MortonPrimitive *mortonPrims, int nPrimitives, int *totalNodes,
        std::vector<PrimitiveHandle> &orderedPrims,
        std::atomic<int> *orderedPrimsOffset, int bitIndex);
    BVHBuildNode *buildUpperSAH(MemoryArena &arena,
                                std::vector<BVHBuildNode *> &treeletRoots,
                                int start, int end, std::atomic<int> *totalNodes) const;
    int flattenBVHTree(BVHBuildNode *node, int *offset);

    // BVHAccel Private Data
    int maxPrimsInNode;
    SplitMethod splitMethod;
    std::vector<PrimitiveHandle> primitives;
    LinearBVHNode *nodes = nullptr;
};

// KdTreeAccel Declarations
struct KdAccelNode;
struct BoundEdge;
class KdTreeAccel : public Primitive {
  public:
    // KdTreeAccel Public Methods
    KdTreeAccel(std::vector<PrimitiveHandle> p,
                int isectCost = 80, int traversalCost = 1,
                Float emptyBonus = 0.5, int maxPrims = 1, int maxDepth = -1);

    static KdTreeAccel *Create(std::vector<PrimitiveHandle> prims,
                               const ParameterDictionary &dict);

    Bounds3f WorldBound() const { return bounds; }
    ~KdTreeAccel();
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // KdTreeAccel Private Methods
    void buildTree(int nodeNum, const Bounds3f &bounds,
                   const std::vector<Bounds3f> &primBounds, int *primNums,
                   int nprims, int depth,
                   const std::unique_ptr<BoundEdge[]> edges[3], int *prims0,
                   int *prims1, int badRefines = 0);

    // KdTreeAccel Private Data
    int isectCost, traversalCost, maxPrims;
    Float emptyBonus;
    std::vector<PrimitiveHandle> primitives;
    std::vector<int> primitiveIndices;
    KdAccelNode *nodes;
    int nAllocedNodes, nextFreeNode;
    Bounds3f bounds;
};

struct KdToDo {
    const KdAccelNode *node;
    Float tMin, tMax;
};

}  // namespace pbrt

#endif  // PBRT_ACCELERATORS_KDTREE_H
