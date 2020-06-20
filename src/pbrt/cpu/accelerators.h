// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_ACCELERATORS_BVH_H
#define PBRT_ACCELERATORS_BVH_H

// accelerators/bvh.h*
#include <pbrt/pbrt.h>

#include <pbrt/cpu/primitive.h>

#include <atomic>
#include <memory>
#include <vector>

namespace pbrt {

PrimitiveHandle CreateAccelerator(const std::string &name,
                                  std::vector<PrimitiveHandle> prims,
                                  const ParameterDictionary &parameters);

struct BVHBuildNode;

// BVHAccel Forward Declarations
struct BVHPrimitiveInfo;
struct MortonPrimitive;
struct LinearBVHNode;

// BVHAccel Declarations
class BVHAccel {
  public:
    // BVHAccel Public Types
    enum class SplitMethod { SAH, HLBVH, Middle, EqualCounts };

    // BVHAccel Public Methods
    BVHAccel(std::vector<PrimitiveHandle> p, int maxPrimsInNode = 1,
             SplitMethod splitMethod = SplitMethod::SAH);

    static BVHAccel *Create(std::vector<PrimitiveHandle> prims,
                            const ParameterDictionary &parameters);

    Bounds3f Bounds() const;
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // BVHAccel Private Methods
    BVHBuildNode *recursiveBuild(std::vector<Allocator> &threadAllocators,
                                 std::vector<BVHPrimitiveInfo> &primitiveInfo, int start,
                                 int end, std::atomic<int> *totalNodes,
                                 std::vector<PrimitiveHandle> &orderedPrims,
                                 std::atomic<int> *orderedPrimsOffset);
    BVHBuildNode *HLBVHBuild(Allocator alloc,
                             const std::vector<BVHPrimitiveInfo> &primitiveInfo,
                             std::atomic<int> *totalNodes,
                             std::vector<PrimitiveHandle> &orderedPrims);
    BVHBuildNode *emitLBVH(BVHBuildNode *&buildNodes,
                           const std::vector<BVHPrimitiveInfo> &primitiveInfo,
                           MortonPrimitive *mortonPrims, int nPrimitives, int *totalNodes,
                           std::vector<PrimitiveHandle> &orderedPrims,
                           std::atomic<int> *orderedPrimsOffset, int bitIndex);
    BVHBuildNode *buildUpperSAH(Allocator alloc,
                                std::vector<BVHBuildNode *> &treeletRoots, int start,
                                int end, std::atomic<int> *totalNodes) const;
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
class KdTreeAccel {
  public:
    // KdTreeAccel Public Methods
    KdTreeAccel(std::vector<PrimitiveHandle> p, int isectCost = 80, int traversalCost = 1,
                Float emptyBonus = 0.5, int maxPrims = 1, int maxDepth = -1);

    static KdTreeAccel *Create(std::vector<PrimitiveHandle> prims,
                               const ParameterDictionary &parameters);

    Bounds3f Bounds() const { return bounds; }
    pstd::optional<ShapeIntersection> Intersect(const Ray &ray, Float tMax) const;
    bool IntersectP(const Ray &ray, Float tMax) const;

  private:
    // KdTreeAccel Private Methods
    void buildTree(int nodeNum, const Bounds3f &bounds,
                   const std::vector<Bounds3f> &primBounds, int *primNums, int nprims,
                   int depth, const std::unique_ptr<BoundEdge[]> edges[3], int *prims0,
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
