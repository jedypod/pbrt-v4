
#include <gtest/gtest.h>

#include <pbrt/util/containers.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/rng.h>

#include <set>
#include <string>

using namespace pbrt;

TEST(AoSoA, Basic) {
    int n = 8192;
    Allocator alloc;

    AoSoA<float> singleFloat(n, alloc);

    EXPECT_EQ(singleFloat.size(), n);

    for (int i = 0; i < n; ++i)
        singleFloat.at<0>(i) = 2*i;
    for (int i = 0; i < n; ++i)
        EXPECT_EQ(2*i, singleFloat.at<0>(i));
}

TEST(AoSoA, Pair) {
    int n = 8192;
    Allocator alloc;

    AoSoA<float, float> floatPair(n, alloc);

    EXPECT_EQ(floatPair.size(), n);

    for (int i = 0; i < n; ++i) {
        floatPair.at<0>(i) = 2*i;
        floatPair.at<1>(i) = -i;
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(2*i, floatPair.at<0>(i));
        EXPECT_EQ(-i, floatPair.at<1>(i));
    }
}

TEST(AoSoA, Mixed) {
    int n = 8192;
    Allocator alloc;

    AoSoA<float, short, int> mixed(n, alloc);

    EXPECT_EQ(mixed.size(), n);

    for (int i = 0; i < n; ++i) {
        mixed.at<0>(i) = 2*i;
        mixed.at<1>(i) = -i;
        mixed.at<2>(i) = 3*i;
    }
    for (int i = 0; i < n; ++i) {
        EXPECT_EQ(2*i, mixed.at<0>(i));
        EXPECT_EQ(-i, mixed.at<1>(i));
        EXPECT_EQ(3*i, mixed.at<2>(i));
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

