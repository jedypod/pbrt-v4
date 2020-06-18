// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <gtest/gtest.h>

#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>

#include <set>
#include <string>

using namespace pbrt;

TEST(Array2D, Basics) {
    const int nx = 5, ny = 9;
    Array2D<Float> a(nx, ny);

    EXPECT_EQ(nx, a.xSize());
    EXPECT_EQ(ny, a.ySize());
    EXPECT_EQ(nx * ny, a.size());

    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
            a(x, y) = 1000 * x + y;

    for (int y = 0; y < ny; ++y)
        for (int x = 0; x < nx; ++x)
            EXPECT_EQ(1000 * x + y, a(x, y));
}

TEST(Array2D, Bounds) {
    Bounds2i b(Point2i(-4, 3), Point2i(10, 7));
    Array2D<Point2f> a(b);

    EXPECT_EQ(b.pMax.x - b.pMin.x, a.xSize());
    EXPECT_EQ(b.pMax.y - b.pMin.y, a.ySize());

    for (Point2i p : b)
        a[p] = Point2f(p.y, p.x);

    for (Point2i p : b)
        EXPECT_EQ(Point2f(p.y, p.x), a[p]);
}

TEST(AoSoA, Basic) {
    int n = 8192;
    Allocator alloc;

    AoSoA<float> singleFloat(alloc, n);

    EXPECT_EQ(singleFloat.size(), n);

    for (int i = 0; i < n; ++i)
        singleFloat.at<0>(i) = 2 * i;
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(2 * i, singleFloat.at<0>(i));
}

TEST(AoSoA, Pair) {
    int n = 8192;
    Allocator alloc;

    AoSoA<float, float> floatPair(alloc, n);

    EXPECT_EQ(floatPair.size(), n);

    for (int i = 0; i < n; ++i) {
        floatPair.at<0>(i) = 2 * i;
        floatPair.at<1>(i) = -i;
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(2 * i, floatPair.at<0>(i));
        EXPECT_EQ(-i, floatPair.at<1>(i));
    }
}

TEST(AoSoA, Mixed) {
    int n = 8192;
    Allocator alloc;

    AoSoA<float, short, int> mixed(alloc, n);

    EXPECT_EQ(mixed.size(), n);

    for (int i = 0; i < n; ++i) {
        mixed.at<0>(i) = 2 * i;
        mixed.at<1>(i) = -i;
        mixed.at<2>(i) = 3 * i;
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(2 * i, mixed.at<0>(i));
        EXPECT_EQ(-i, mixed.at<1>(i));
        EXPECT_EQ(3 * i, mixed.at<2>(i));
    }
}

TEST(AoSoA, MixedVariousSizes) {
    RNG rng;
    Allocator alloc;

    for (int iter = 0; iter < 1000; ++iter) {
        int n = 1 + rng.Uniform<int>(512);

        AoSoA<float, short, int> mixed(alloc, n);

        EXPECT_EQ(mixed.size(), n);

        for (int i = 0; i < n; ++i) {
            mixed.at<0>(i) = 2 * i;
            mixed.at<1>(i) = -i;
            mixed.at<2>(i) = 3 * i;
        }
        for (int i = 0; i < n; ++i) {
            EXPECT_EQ(2 * i, mixed.at<0>(i));
            EXPECT_EQ(-i, mixed.at<1>(i));
            EXPECT_EQ(3 * i, mixed.at<2>(i));
        }
    }
}

TEST(HashMap, Basics) {
    Allocator alloc;
    HashMap<int, std::string, std::hash<int>> map(alloc);

    map.Insert(1, std::string("yolo"));
    map.Insert(10, std::string("hello"));
    map.Insert(42, std::string("test"));

    EXPECT_EQ(3, map.size());
    EXPECT_GE(map.capacity(), 3);
    EXPECT_TRUE(map.HasKey(1));
    EXPECT_TRUE(map.HasKey(10));
    EXPECT_TRUE(map.HasKey(42));
    EXPECT_FALSE(map.HasKey(0));
    EXPECT_FALSE(map.HasKey(1240));
    EXPECT_EQ("yolo", map[1]);
    EXPECT_EQ("hello", map[10]);
    EXPECT_EQ("test", map[42]);

    map.Insert(10, std::string("hai"));
    EXPECT_EQ(3, map.size());
    EXPECT_GE(map.capacity(), 3);
    EXPECT_EQ("hai", map[10]);
}

TEST(HashMap, Randoms) {
    Allocator alloc;
    HashMap<int, int, std::hash<int>> map(alloc);
    std::set<int> values;
    RNG rng(1234);

    for (int i = 0; i < 10000; ++i) {
        int v = rng.Uniform<int>();
        values.insert(v);
        map.Insert(v, -v);
    }

    // Could have a collision so thus less...
    EXPECT_EQ(map.size(), values.size());

    // Reset
    rng.SetSequence(1234);
    for (int i = 0; i < 10000; ++i) {
        int v = rng.Uniform<int>();
        ASSERT_TRUE(map.HasKey(v));
        EXPECT_EQ(-v, map[v]);
    }

    int nVisited = 0;
    for (auto iter = map.begin(); iter != map.end(); ++iter) {
        ++nVisited;

        EXPECT_EQ(iter->first, -iter->second);

        int v = iter->first;
        auto siter = values.find(v);
        ASSERT_NE(siter, values.end());
        values.erase(siter);
    }

    EXPECT_EQ(nVisited, 10000);
    EXPECT_EQ(0, values.size());
}
