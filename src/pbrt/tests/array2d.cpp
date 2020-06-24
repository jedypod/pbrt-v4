
#include <gtest/gtest.h>

#include <pbrt/util/array2d.h>

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
