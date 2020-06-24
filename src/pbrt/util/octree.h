
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

#ifndef PBRT_GEOMETRY_OCTREE_H
#define PBRT_GEOMETRY_OCTREE_H

// geometry/octree.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <algorithm>
#include <cmath>

namespace pbrt {

PBRT_HOST_DEVICE_INLINE
int firstNode(Float tx0, Float ty0, Float tz0, Float txm, Float tym,
              Float tzm) {
    Float tmax = std::max({tx0, ty0, tz0});
    CHECK(!std::isnan(tmax));

    if (tx0 == tmax)
        // yz plane entry
        return ((tym < tx0) ? 2 : 0) + ((tzm < tx0) ? 4 : 0);
    else if (ty0 == tmax)
        // xz plane entry
        return ((txm < ty0) ? 1 : 0) + ((tzm < ty0) ? 4 : 0);
    else
        // xy plane entry
        return ((txm < tz0) ? 1 : 0) + ((tym < tz0) ? 2 : 0);
}

PBRT_HOST_DEVICE
static int nodeStep(Float tx, int nx, Float ty, int ny, Float tz, int nz) {
    Float tmin = std::min({tx, ty, tz});
    CHECK(!std::isnan(tmin));

    if (tx == tmin) return nx;
    return (ty == tmin) ? ny : nz;
}

PBRT_HOST_DEVICE
inline Bounds3f OctreeChildBounds(const Bounds3f &parentBounds, int child) {
    Point3f pMid = (parentBounds.pMin + parentBounds.pMax) * 0.5f;
    Bounds3f childBounds = parentBounds;
    for (int c = 0; c < 3; ++c)
        childBounds[(child & (1 << c)) ? 0 : 1][c] = pMid[c];
    return childBounds;
}

enum class OctreeTraversal {
    Continue,
    Abort,
};

template <typename N, typename F>
PBRT_HOST_DEVICE
static OctreeTraversal traverseNode(const N *node, const Bounds3f &bounds, const Point3f &o,
                                    const Vector3f &d, Float raytMax, int mirrorMask, Float tx0,
                                    Float ty0, Float tz0, Float tx1, Float ty1, Float tz1,
                                    F nodeCallback) {
    if (tx1 < 0 || ty1 < 0 || tz1 < 0)
        return OctreeTraversal::Continue;
    if (std::max({tx0, ty0, tz0}) >= raytMax)
        return OctreeTraversal::Abort;

    // FIXME: should we have a hasChildren() method?
    if (node->child(0) == nullptr)
        // Leaf
        return nodeCallback(*node,
                            std::max<Float>({0, tx0, ty0, tz0}),
                            std::min({tx1, ty1, tz1, raytMax}));

    Float t[3][3] = {{tx0, 0, tx1},
                     {ty0, 0, ty1},
                     {tz0, 0, tz1}};

    if (d.x != 0)
        t[0][1] = (tx0 + tx1) * 0.5f;
    else
        t[0][1] = o.x < (bounds.pMin.x + bounds.pMax.x) / 2 ? Infinity : -Infinity;
    if (d.y != 0)
        t[1][1] = (ty0 + ty1) * 0.5f;
    else
        t[1][1] = o.y < (bounds.pMin.y + bounds.pMax.y) / 2 ? Infinity : -Infinity;
    if (d.z != 0)
        t[2][1] = (tz0 + tz1) * 0.5f;
    else
        t[2][1] = o.z < (bounds.pMin.z + bounds.pMax.z) / 2 ? Infinity : -Infinity;

    int nodeIndex = firstNode(tx0, ty0, tz0, t[0][1], t[1][1], t[2][1]);
    while (nodeIndex < 8) {
        auto bitSet = [nodeIndex](int b) {
            return nodeIndex & (1 << b) ? 1 : 0;
        };
        int child = nodeIndex ^ mirrorMask;
        if (traverseNode(node->child(child), OctreeChildBounds(bounds, child),
                         o, d, raytMax, mirrorMask,
                         t[0][bitSet(0)], t[1][bitSet(1)], t[2][bitSet(2)],
                         t[0][1 + bitSet(0)], t[1][1 + bitSet(1)],
                         t[2][1 + bitSet(2)], nodeCallback) == OctreeTraversal::Abort)
            return OctreeTraversal::Abort;

        nodeIndex += nodeStep(t[0][1 + bitSet(0)], bitSet(0) ? 8 : 1,
                              t[1][1 + bitSet(1)], bitSet(1) ? 8 : 2,
                              t[2][1 + bitSet(2)], bitSet(2) ? 8 : 4);
    }
    return OctreeTraversal::Continue;
}

// Over [0,1]^3
// TODO: document requirements on N
// child() method
// convention that child(0) is nullptr for a leaf node
// ... ?
template <typename N, typename F>
PBRT_HOST_DEVICE_INLINE
void TraverseOctree(const N *root, Point3f o, Vector3f d,
                    Float raytMax, F nodeCallback) {
    int mirrorMask = 0;
    for (int c = 0; c < 3; ++c) {
        if (d[c] < 0) {
            o[c] = 1 - o[c];
            d[c] = -d[c];
            mirrorMask |= 1 << c;
        }
    }

    Float tx0 = -o.x / d.x;
    Float tx1 = (1 - o.x) / d.x;
    Float ty0 = -o.y / d.y;
    Float ty1 = (1 - o.y) / d.y;
    Float tz0 = -o.z / d.z;
    Float tz1 = (1 - o.z) / d.z;

    Bounds3f rootBounds(Point3f(0, 0, 0), Point3f(1, 1, 1));

    if (std::max({tx0, ty0, tz0}) < std::min({tx1, ty1, tz1}))
        traverseNode(root, rootBounds, o, d, raytMax, mirrorMask, tx0, ty0, tz0, tx1, ty1, tz1,
                     nodeCallback);
}

}  // namespace pbrt

#endif  // PBRT_GEOMETRY_OCTREE_H
