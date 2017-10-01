
#include "tests/gtest/gtest.h"
#include <cmath>

#include "pbrt.h"
#include "util/mathutil.h"

using namespace pbrt;

TEST(Math, Pow) {
    EXPECT_EQ(Pow<0>(2.f), 1 << 0);
    EXPECT_EQ(Pow<1>(2.f), 1 << 1);
    EXPECT_EQ(Pow<2>(2.f), 1 << 2);
    // Test remainder of pow template powers to 29
    EXPECT_EQ(Pow<3>(2.f), 1 << 3);
    EXPECT_EQ(Pow<4>(2.f), 1 << 4);
    EXPECT_EQ(Pow<5>(2.f), 1 << 5);
    EXPECT_EQ(Pow<6>(2.f), 1 << 6);
    EXPECT_EQ(Pow<7>(2.f), 1 << 7);
    EXPECT_EQ(Pow<8>(2.f), 1 << 8);
    EXPECT_EQ(Pow<9>(2.f), 1 << 9);
    EXPECT_EQ(Pow<10>(2.f), 1 << 10);
    EXPECT_EQ(Pow<11>(2.f), 1 << 11);
    EXPECT_EQ(Pow<12>(2.f), 1 << 12);
    EXPECT_EQ(Pow<13>(2.f), 1 << 13);
    EXPECT_EQ(Pow<14>(2.f), 1 << 14);
    EXPECT_EQ(Pow<15>(2.f), 1 << 15);
    EXPECT_EQ(Pow<16>(2.f), 1 << 16);
    EXPECT_EQ(Pow<17>(2.f), 1 << 17);
    EXPECT_EQ(Pow<18>(2.f), 1 << 18);
    EXPECT_EQ(Pow<19>(2.f), 1 << 19);
    EXPECT_EQ(Pow<20>(2.f), 1 << 20);
    EXPECT_EQ(Pow<21>(2.f), 1 << 21);
    EXPECT_EQ(Pow<22>(2.f), 1 << 22);
    EXPECT_EQ(Pow<23>(2.f), 1 << 23);
    EXPECT_EQ(Pow<24>(2.f), 1 << 24);
    EXPECT_EQ(Pow<25>(2.f), 1 << 25);
    EXPECT_EQ(Pow<26>(2.f), 1 << 26);
    EXPECT_EQ(Pow<27>(2.f), 1 << 27);
    EXPECT_EQ(Pow<28>(2.f), 1 << 28);
    EXPECT_EQ(Pow<29>(2.f), 1 << 29);
}
