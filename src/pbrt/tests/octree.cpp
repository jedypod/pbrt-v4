
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/ray.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/octree.h>

#include <algorithm>
using namespace pbrt;

struct OctreeNode {
    OctreeNode() {
        for (int i = 0; i < 8; ++i)
            children[i] = nullptr;
    }

    int data = 0;
    OctreeNode *children[8];

    OctreeNode *child(int n) { return children[n]; }
    const OctreeNode *child(int n) const { return children[n]; }
};

TEST(Octree, VisitRoot) {
    OctreeNode root;
    root.data = 1;

    bool visited = false;
    Ray ray(Point3f(-4, -2, -1), Vector3f(4.2, 2.2, 1.2));
    TraverseOctree(&root, ray.o, ray.d, Infinity, [&](const OctreeNode &node, Float, Float) {
            visited = true;
            EXPECT_EQ(node.data, root.data);
            return OctreeTraversal::Continue;
        });
    EXPECT_TRUE(visited);

    // Ray starting inside the octree
    visited = false;
    ray = Ray(Point3f(.1, .35, .55), Vector3f(.9, .3, -.2));
    TraverseOctree(&root, ray.o, ray.d, Infinity, [&](const OctreeNode &node, Float, Float) {
            visited = true;
            EXPECT_EQ(root.data, node.data);
            return OctreeTraversal::Continue;
        });
    EXPECT_TRUE(visited);
}

TEST(Octree, VisitRootRandomRays) {
    OctreeNode root;
    root.data = 1;

    RNG rng;
    // Octree's bounds
    Bounds3f bounds(Point3f(0,0,0), Point3f(1,1,1));
    int count = 0;
    int nSamples = 10000;
    for (int i = 0; i < nSamples; ++i) {
        // Choose two random points on a sphere of random radius centered
        // around (0.5, 0.5, 0.5) (the center of the octree). Small radii
        // ensure that some rays start inside the octree.
        Float radius = .1 + 2 * rng.Uniform<Float>();
        Point2f u0{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point3f p0 = Point3f(radius * SampleUniformSphere(u0) + Vector3f(0.5, 0.5, 0.5));
        Point2f u1{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point3f p1 = Point3f(radius * SampleUniformSphere(u1) + Vector3f(0.5, 0.5, 0.5));
        Ray ray(p0, p1 - p0);
        Float bt0, bt1;
        if (bounds.IntersectP(ray.o, ray.d, Infinity, &bt0, &bt1)) {
            ++count;
            bool visited = false;
            TraverseOctree(&root, ray.o, ray.d, Infinity, [&](const OctreeNode &node,
                                                              Float t0, Float t1) {
                    visited = true;
                    EXPECT_EQ(root.data, node.data);
                    EXPECT_LT(std::abs(bt0 - t0), 1e-4);
                    EXPECT_LT(std::abs(bt1 - t1), 1e-4);
                    return OctreeTraversal::Continue;
                });
            EXPECT_TRUE(visited);
        }
    }
    // Make sure we in fact tested some rays above
    EXPECT_GE(count, int(.4 * nSamples));
}


TEST(Octree, OneLevel) {
    OctreeNode root, children[8];
    Bounds3f childBounds[8];
    root.data = -1;
    Bounds3f bounds(Point3f(0,0,0), Point3f(1,1,1));

    for (int i = 0; i < 8; ++i) {
        root.children[i] = &children[i];
        children[i].data = i;

        childBounds[i] = OctreeChildBounds(bounds, i);
    }

    RNG rng;
    int count = 0;
    int nSamples = 10000;
    for (int i = 0; i < nSamples; ++i) {
        // Choose two random points on a sphere of random radius centered
        // around (0.5, 0.5, 0.5) (the center of the octree). Small radii
        // ensure that some rays start inside the octree.
        Float radius = .1 + 2 * rng.Uniform<Float>();
        Point2f u0{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point3f p0 = Point3f(radius * SampleUniformSphere(u0) + Vector3f(0.5, 0.5, 0.5));
        Point2f u1{rng.Uniform<Float>(), rng.Uniform<Float>()};
        Point3f p1 = Point3f(radius * SampleUniformSphere(u1) + Vector3f(0.5, 0.5, 0.5));
        Ray ray(p0, p1 - p0);

        // Zero out one or two direction components for some of the rays
        // to make sure we don't get into NaN trouble with thise.
        if ((i % 5) == 0)
            ray.d[rng.Uniform<int>(3)] = 0;
        if ((i % 11) == 0)
            ray.d[rng.Uniform<int>(3)] = 0;

        Float t0, t1;
        if (!bounds.IntersectP(ray.o, ray.d, 1, &t0, &t1))
            continue;

        struct ChildHit {
            Float t, t1;
            int child;
            bool operator<(const ChildHit &h) const {
                return t < h.t;
            }
        };
        std::vector<ChildHit> childHits;
        for (int j = 0; j < 8; ++j) {
            if (childBounds[j].IntersectP(ray.o, ray.d, 1, &t0, &t1)) {
                ASSERT_GT(t1, 0);
                childHits.push_back({t0, t1, j});
            }
        }
        std::sort(childHits.begin(), childHits.end());
        int hitIndex = 0;
        ++count;

        TraverseOctree(&root, ray.o, ray.d, 1, [&](const OctreeNode &node,
                                                   Float t0, Float t1) {
                // Want ASSERT_LT, but it returns out of the function
                // without returning a value, which is incompatible with
                // the bool return...
                EXPECT_LT(hitIndex, childHits.size());
                if (hitIndex == childHits.size())
                    return OctreeTraversal::Abort;

                EXPECT_EQ(node.data, childHits[hitIndex].child) << i;
                EXPECT_LT(std::abs(childHits[hitIndex].t - t0), 1e-4);
                EXPECT_LT(std::abs(childHits[hitIndex].t1 - t1), 1e-4);

                ++hitIndex;
                return OctreeTraversal::Continue;
            });
        EXPECT_EQ(hitIndex, childHits.size());
    }
    // Make sure we in fact tested some rays above
    EXPECT_GE(count, int(.4 * nSamples));
}
