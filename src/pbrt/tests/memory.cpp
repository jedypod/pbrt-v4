
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>
#include <pbrt/util/bits.h>
#include <pbrt/util/memory.h>

#include <cstdint>
#include <set>

using namespace pbrt;

TEST(MemoryArena, Alignment) {
    MemoryArena arena(4096);

    for (int i = 0; i < 1000; ++i) {
        (void)arena.Alloc<char>(1);

        int16_t *i16 = arena.Alloc<int16_t>(1);
        EXPECT_EQ(0, (uintptr_t)i16 & 1);

        int32_t *i32 = arena.Alloc<int32_t>(1);
        EXPECT_EQ(0, (uintptr_t)i32 & 3);

        int64_t *i64 = arena.Alloc<int64_t>(1);
        EXPECT_EQ(0, (uintptr_t)i64 & 7);
    }
}

TEST(MemoryArena, NoOverlaps) {
    MemoryArena arena(4096);

    for (int pass = 0; pass < 10; ++pass) {
        std::set<void *> seen;
        std::vector<char *> cp;
        std::vector<int16_t *> i16p;
        std::vector<int32_t *> i32p;
        std::vector<int64_t *> i64p;

        for (int i = 0; i < 100; ++i) {
            char *c = arena.Alloc<char>(1);
            EXPECT_TRUE(seen.find(c) == seen.end());
            seen.insert(c);
            cp.push_back(c);
            *c = MixBits((uintptr_t)c);

            // Switch order so that the i16 can be immediately after i32
            // allocs.
            int32_t *i32 = arena.Alloc<int32_t>(1);
            EXPECT_TRUE(seen.find(i32) == seen.end());
            seen.insert(i32);
            i32p.push_back(i32);
            *i32 = MixBits((uintptr_t)i32);

            int16_t *i16 = arena.Alloc<int16_t>(1);
            EXPECT_TRUE(seen.find(i16) == seen.end());
            seen.insert(i16);
            i16p.push_back(i16);
            *i16 = MixBits((uintptr_t)i16);

            int64_t *i64 = arena.Alloc<int64_t>(1);
            EXPECT_TRUE(seen.find(i64) == seen.end());
            seen.insert(i64);
            i64p.push_back(i64);
            *i64 = MixBits((uintptr_t)i64);
        }

        // Make sure all of the expected values are in memory
        for (char *p : cp)
            EXPECT_EQ(*p, (char)MixBits((uintptr_t)p));
        for (int16_t *p : i16p)
            EXPECT_EQ(*p, (int16_t)MixBits((uintptr_t)p));
        for (int32_t *p : i32p)
            EXPECT_EQ(*p, (int32_t)MixBits((uintptr_t)p));
        for (int64_t *p : i64p)
            EXPECT_EQ(*p, (int64_t)MixBits((uintptr_t)p));

        arena.Reset();
    }
}

TEST(ArenaAllocator, Basics) {
    MemoryArena arena;
    ArenaAllocator<int> alloc(&arena);
    std::vector<int, ArenaAllocator<int>> vec(alloc);

    const int n = 1000000;
    for (int i = 0; i < n; ++i)
        vec.push_back(i);

    for (int i = 0; i < n; ++i)
        EXPECT_EQ(i, vec[i]);

    vec.clear();
    vec.shrink_to_fit();
    ASSERT_EQ(0, vec.capacity());
}
