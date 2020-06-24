
#include <gtest/gtest.h>

#include <pbrt/pbrt.h>

#include <pbrt/interaction.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/sampling.h>
#include <pbrt/shapes.h>
#include <pbrt/util/parallel.h>

#include <cmath>
#include <functional>

using namespace pbrt;

static Float pExp(RNG &rng, Float exp = 8.) {
    Float logu = Lerp(rng.Uniform<Float>(), -exp, exp);
    return std::pow(10, logu);
}

static Float pUnif(RNG &rng, Float range = 10.) {
    return Lerp(rng.Uniform<Float>(), -range, range);
}

TEST(Triangle, Watertight) {
    RNG rng(12111);
    int nTheta = 16, nPhi = 16;
    ASSERT_GE(nTheta, 3);
    ASSERT_GE(nPhi, 4);

    // Make a triangle mesh representing a triangulated sphere (with
    // vertices randomly offset along their normal), centered at the
    // origin.
    int nVertices = nTheta * nPhi;
    std::vector<Point3f> vertices;
    for (int t = 0; t < nTheta; ++t) {
        Float theta = Pi * (Float)t / (Float)(nTheta - 1);
        Float cosTheta = std::cos(theta);
        Float sinTheta = std::sin(theta);
        for (int p = 0; p < nPhi; ++p) {
            Float phi = 2 * Pi * (Float)p / (Float)(nPhi - 1);
            Float radius = 1;
            // Make sure all of the top and bottom vertices are coincident.
            if (t == 0)
                vertices.push_back(Point3f(0, 0, radius));
            else if (t == nTheta - 1)
                vertices.push_back(Point3f(0, 0, -radius));
            else if (p == nPhi - 1)
                // Close it up exactly at the end
                vertices.push_back(vertices[vertices.size() - (nPhi - 1)]);
            else {
                radius += 5 * rng.Uniform<Float>();
                vertices.push_back(
                    Point3f(0, 0, 0) +
                    radius * SphericalDirection(sinTheta, cosTheta, phi));
            }
        }
    }
    EXPECT_EQ(nVertices, vertices.size());

    std::vector<int> indices;
    // fan at top
    auto offset = [nPhi](int t, int p) { return t * nPhi + p; };
    for (int p = 0; p < nPhi - 1; ++p) {
        indices.push_back(offset(0, 0));
        indices.push_back(offset(1, p));
        indices.push_back(offset(1, p + 1));
    }

    // quads in the middle rows
    for (int t = 1; t < nTheta - 2; ++t) {
        for (int p = 0; p < nPhi - 1; ++p) {
            indices.push_back(offset(t, p));
            indices.push_back(offset(t + 1, p));
            indices.push_back(offset(t + 1, p + 1));

            indices.push_back(offset(t, p));
            indices.push_back(offset(t + 1, p + 1));
            indices.push_back(offset(t, p + 1));
        }
    }

    // fan at bottom
    for (int p = 0; p < nPhi - 1; ++p) {
        indices.push_back(offset(nTheta - 1, 0));
        indices.push_back(offset(nTheta - 2, p));
        indices.push_back(offset(nTheta - 2, p + 1));
    }

    Transform identity;
    TriangleMesh mesh(identity, false, indices, vertices, {}, {}, {}, {});
    pstd::vector<ShapeHandle> tris = mesh.CreateTriangles(Allocator());

    for (int i = 0; i < 100000; ++i) {
        RNG rng(i);
        // Choose a random point in sphere of radius 0.5 around the origin.
        Point2f u;
        u[0] = rng.Uniform<Float>();
        u[1] = rng.Uniform<Float>();
        Point3f p = Point3f(0, 0, 0) + Float(0.5) * SampleUniformSphere(u);

        // Choose a random direction.
        u[0] = rng.Uniform<Float>();
        u[1] = rng.Uniform<Float>();
        Ray r(p, SampleUniformSphere(u));
        int nHits = 0;
        for (const auto &tri : tris) {
            if (tri.Intersect(r)) ++nHits;
        }
        EXPECT_GE(nHits, 1);

        // Now tougher: shoot directly at a vertex.
        Point3f pVertex = vertices[rng.Uniform<uint32_t>(vertices.size())];
        r.d = pVertex - r.o;
        nHits = 0;
        for (const auto &tri : tris) {
            if (tri.Intersect(r).has_value()) ++nHits;
        }
        EXPECT_GE(nHits, 1) << pVertex;
    }
}

ShapeHandle GetRandomTriangle(std::function<Float()> value) {
    // Triangle vertices
    Point3f v[3];
    for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) v[j][k] = value();

    if (LengthSquared(Cross(v[1] - v[0], v[2] - v[0])) < 1e-20)
        // Don't into trouble with ~degenerate triangles.
        return nullptr;

    // Create the corresponding Triangle.
    static Transform identity;
    int indices[3] = {0, 1, 2};
    TriangleMesh mesh(identity, false, {indices, indices + 3}, {v, v + 3},
                      {}, {}, {}, {});
    pstd::vector<ShapeHandle> triVec = mesh.CreateTriangles(Allocator());
    EXPECT_EQ(1, triVec.size());
    return triVec[0];
}

TEST(Triangle, Reintersect) {
    for (int i = 0; i < 1000; ++i) {
        RNG rng(i);
        ShapeHandle tri = GetRandomTriangle([&]() { return pExp(rng); });
        if (!tri) continue;

        // Sample a point on the triangle surface to shoot the ray toward.
        Point2f u;
        u[0] = rng.Uniform<Float>();
        u[1] = rng.Uniform<Float>();
        auto ts = tri.Sample(u);
        ASSERT_TRUE(ts.has_value());

        // Choose a ray origin.
        Point3f o;
        for (int j = 0; j < 3; ++j) o[j] = pExp(rng);

        // Intersect the ray with the triangle.
        Ray r(o, ts->intr.p() - o);
        auto si = tri.Intersect(r);
        if (!si)
            // We should almost always find an intersection, but rarely
            // miss, due to round-off error. Just do another go-around in
            // this case.
            continue;
        SurfaceInteraction &isect = si->intr;

        // Now trace a bunch of rays leaving the intersection point.
        for (int j = 0; j < 10000; ++j) {
            // Random direction leaving the intersection point.
            Point2f u;
            u[0] = rng.Uniform<Float>();
            u[1] = rng.Uniform<Float>();
            Vector3f w = SampleUniformSphere(u);
            Ray rOut = isect.SpawnRay(w);
            EXPECT_FALSE(tri.IntersectP(rOut));
            EXPECT_FALSE(tri.Intersect(rOut).has_value());

            // Choose a random point to trace rays to.
            Point3f p2;
            for (int k = 0; k < 3; ++k) p2[k] = pExp(rng);
            rOut = isect.SpawnRayTo(p2);

            EXPECT_FALSE(tri.IntersectP(rOut, 1));
            EXPECT_FALSE(tri.Intersect(rOut, 1).has_value());
        }
    }
}

// Computes the projected solid angle subtended by a series of random
// triangles both using uniform spherical sampling as well as
// Triangle::Sample(), in order to verify Triangle::Sample().
TEST(Triangle, Sampling) {
    for (int i = 0; i < 30; ++i) {
        const Float range = 10;
        RNG rng(i);
        ShapeHandle tri =
            GetRandomTriangle([&]() { return pUnif(rng, range); });
        if (!tri) continue;

        // Ensure that the reference point isn't too close to the
        // triangle's surface (which makes the Monte Carlo stuff have more
        // variance, thus requiring more samples).
        Point3f pc{pUnif(rng, range), pUnif(rng, range), pUnif(rng, range)};
        pc[rng.Uniform<uint32_t>() % 3] =
            rng.Uniform<Float>() > .5 ? (-range - 3) : (range + 3);

        // Compute reference value using Monte Carlo with uniform spherical
        // sampling.
        const int count = 512 * 1024;
        int hits = 0;
        for (int j = 0; j < count; ++j) {
            Point2f u{RadicalInverse(0, j), RadicalInverse(1, j)};
            Vector3f w = SampleUniformSphere(u);
            if (tri.IntersectP(Ray(pc, w))) ++hits;
        }
        double unifEstimate = hits / double(count * UniformSpherePDF());

        // Now use Triangle::Sample()...
        Interaction ref(pc, Normal3f(), 0, (const Medium *)nullptr);
        double triSampleEstimate = 0;
        for (int j = 0; j < count; ++j) {
            Point2f u{RadicalInverse(0, j), RadicalInverse(1, j)};
            auto ss = tri.Sample(ref, u);
            ASSERT_TRUE(ss.has_value());
            EXPECT_GT(ss->pdf, 0);
            triSampleEstimate += 1. / (count * ss->pdf);
        }

        // Now make sure that the two computed solid angle values are
        // fairly close.
        // Absolute error for small solid angles, relative for large.
        auto error = [](Float a, Float b) {
            if (std::abs(a) < 1e-4 || std::abs(b) < 1e-4)
                return std::abs(a - b);
            return std::abs((a - b) / b);
        };

        // Don't compare really small triangles, since uniform sampling
        // doesn't get a good estimate for them.
        if (triSampleEstimate > 1e-3)
            // The error tolerance is fairly large so that we can use a
            // reasonable number of samples.  It has been verified that for
            // larger numbers of Monte Carlo samples, the error continues to
            // tighten.
            EXPECT_LT(error(triSampleEstimate, unifEstimate), .1)
                << "Unif sampling: " << unifEstimate
                << ", triangle sampling: " << triSampleEstimate
                << ", tri index " << i;
    }
}

// Checks the closed-form solid angle computation for triangles against a
// Monte Carlo estimate of it.
TEST(Triangle, SolidAngle) {
    for (int i = 0; i < 50; ++i) {
        const Float range = 10;
        RNG rng(100 +
                i);  // Use different triangles than the Triangle/Sample test.
        ShapeHandle tri = GetRandomTriangle([&]() { return pUnif(rng, range); });
        if (!tri) continue;

        // Ensure that the reference point isn't too close to the
        // triangle's surface (which makes the Monte Carlo stuff have more
        // variance, thus requiring more samples).
        Point3f pc{pUnif(rng, range), pUnif(rng, range), pUnif(rng, range)};
        pc[rng.Uniform<uint32_t>() % 3] =
            rng.Uniform<Float>() > .5 ? (-range - 3) : (range + 3);

        // Compute a reference value using Triangle::Sample()
        const int count = 64 * 1024;
        Interaction ref(pc, Normal3f(), 0, (const Medium *)nullptr);
        double triSampleEstimate = 0;
        for (int j = 0; j < count; ++j) {
            Point2f u{RadicalInverse(0, j), RadicalInverse(1, j)};
            auto ss = tri.Sample(ref, u);
            ASSERT_TRUE(ss.has_value());
            EXPECT_GT(ss->pdf, 0);
            triSampleEstimate += 1. / (count * ss->pdf);
        }

        auto error = [](Float a, Float b) {
            if (std::abs(a) < 1e-4 || std::abs(b) < 1e-4)
                return std::abs(a - b);
            return std::abs((a - b) / b);
        };

        // Now compute the subtended solid angle of the triangle in closed
        // form.
        Float sphericalArea = tri.SolidAngle(pc);

        EXPECT_LT(error(sphericalArea, triSampleEstimate), .015)
            << "spherical area: " << sphericalArea
            << ", tri sampling: " << triSampleEstimate << ", pc = " << pc
            << ", tri index " << i;
    }
}

// Use Quasi Monte Carlo with uniform sphere sampling to esimate the solid
// angle subtended by the given shape from the given point.
static Float mcSolidAngle(const Point3f &p, const ShapeHandle shape, int nSamples) {
    int nHits = 0;
    for (int i = 0; i < nSamples; ++i) {
        Point2f u{RadicalInverse(0, i), RadicalInverse(1, i)};
        Vector3f w = SampleUniformSphere(u);
        if (shape.IntersectP(Ray(p, w))) ++nHits;
    }
    return nHits / (UniformSpherePDF() * nSamples);
}

TEST(Sphere, SolidAngle) {
    Transform tr = Translate(Vector3f(1, .5, -.8)) * RotateX(30);
    Transform trInv = Inverse(tr);
    Sphere sphere(&tr, &trInv, false, 1, -1, 1, 360);

    // Make sure we get a subtended solid angle of 4pi for a point
    // inside the sphere.
    Point3f pInside(1, .9, -.8);
    const int nSamples = 128 * 1024;
    EXPECT_LT(std::abs(mcSolidAngle(pInside, &sphere, nSamples) - 4 * Pi), .01);
    EXPECT_LT(std::abs(sphere.SolidAngle(pInside, nSamples) - 4 * Pi), .01);

    // Now try a point outside the sphere
    Point3f p(-.25, -1, .8);
    Float mcSA = mcSolidAngle(p, &sphere, nSamples);
    Float sphereSA = sphere.SolidAngle(p, nSamples);
    EXPECT_LT(std::abs(mcSA - sphereSA), .001);
}

TEST(Cylinder, SolidAngle) {
    Transform tr = Translate(Vector3f(1, .5, -.8)) * RotateX(30);
    Transform trInv = Inverse(tr);
    Cylinder cyl(&tr, &trInv, false, .25, -1, 1, 360.);

    Point3f p(.5, .25, .5);
    const int nSamples = 128 * 1024;
    Float solidAngle = mcSolidAngle(p, &cyl, nSamples);
    EXPECT_LT(std::abs(solidAngle - cyl.SolidAngle(p, nSamples)), .001);
}

TEST(Disk, SolidAngle) {
    Transform tr = Translate(Vector3f(1, .5, -.8)) * RotateX(30);
    Transform trInv = Inverse(tr);
    Disk disk(&tr, &trInv, false, 0, 1.25, 0, 360);

    Point3f p(.5, -.8, .5);
    const int nSamples = 128 * 1024;
    Float solidAngle = mcSolidAngle(p, &disk, nSamples);
    EXPECT_LT(std::abs(solidAngle - disk.SolidAngle(p, nSamples)), .001);
}

// Check for incorrect self-intersection: assumes that the shape is convex,
// such that if the dot product of an outgoing ray and the surface normal
// at a point is positive, then a ray leaving that point in that direction
// should never intersect the shape.
static int TestReintersectConvex(ShapeHandle shape, RNG &rng) {
    // Ray origin
    Point3f o;
    for (int c = 0; c < 3; ++c) o[c] = pExp(rng);

    // Destination: a random point in the shape's bounding box.
    Bounds3f bbox = shape.WorldBound();
    Point3f t;
    for (int c = 0; c < 3; ++c) t[c] = rng.Uniform<Float>();
    Point3f p2 = bbox.Lerp(t);

    // Ray to intersect with the shape.
    Ray r(o, p2 - o);
    if (rng.Uniform<Float>() < .5) r.d = Normalize(r.d);

    // We should usually (but not always) find an intersection.
    auto si = shape.Intersect(r);
    if (!si) return 0;
    SurfaceInteraction &isect = si->intr;

    // Now trace a bunch of rays leaving the intersection point.
    int n = 0;
    for (int j = 0; j < 10000; ++j) {
        // Random direction leaving the intersection point.
        Point2f u;
        u[0] = rng.Uniform<Float>();
        u[1] = rng.Uniform<Float>();
        Vector3f w = SampleUniformSphere(u);
        // Make sure it's in the same hemisphere as the surface normal.
        w = FaceForward(w, isect.n);
        Ray rOut = isect.SpawnRay(w);

        if (shape.IntersectP(rOut)) ++n;
        if (shape.Intersect(rOut).has_value()) ++n;

        // Choose a random point to trace rays to.
        Point3f p2;
        for (int c = 0; c < 3; ++c) p2[c] = pExp(rng);
        // Make sure that the point we're tracing rays toward is in the
        // hemisphere about the intersection point's surface normal.
        w = p2 - isect.p();
        w = FaceForward(w, isect.n);
        p2 = isect.p() + w;
        rOut = isect.SpawnRayTo(p2);

        if (shape.IntersectP(rOut, 1)) ++n;
        if (shape.Intersect(rOut, 1).has_value()) ++n;
    }

    return n;
}

static Transform randomTransform(RNG &rng) {
    Transform t;
    auto rr = [&rng]() { return -10. + 20. * rng.Uniform<Float>(); };
    auto rt = [&rng]() {
       Float f = pExp(rng);
       return rng.Uniform<Float>() > .5 ? f : -f;
    };

    for (int i = 0; i < 1; ++i) {
        t = t * Scale(pExp(rng, 4), pExp(rng, 4), pExp(rng, 4));
        t = t * Translate(Vector3f(rt(), rt(), rt()));
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        t = t * Rotate(rr() * 20., SampleUniformSphere(u));
    }
    return t;
}

const int nReintersect = 1000;

TEST(FullSphere, Reintersect) {
    ParallelFor(0, nReintersect, [](int64_t i) {
        RNG rng(i);
        Transform identity;
        Float radius = pExp(rng, 4);
        Float zMin = -radius;
        Float zMax = radius;
        Float phiMax = 360;
        Sphere sphere(&identity, &identity, false, radius, zMin, zMax, phiMax);
        EXPECT_EQ(0, TestReintersectConvex(&sphere, rng)) << i;

        Transform tr = randomTransform(rng);
        Transform ti = Inverse(tr);
        Sphere tsphere(&tr, &ti, false, radius, zMin, zMax, phiMax);
        EXPECT_EQ(0, TestReintersectConvex(&tsphere, rng)) << i << ", tr " << tr;
    }
        );
}

TEST(ParialSphere, Normal) {
    for (int i = 0; i < 100; ++i) {
        RNG rng(i);
        Transform identity;
        Float radius = pExp(rng, 4);
        Float zMin = rng.Uniform<Float>() < 0.5
                         ? -radius
                         : Lerp(rng.Uniform<Float>(), -radius, radius);
        Float zMax = rng.Uniform<Float>() < 0.5
                         ? radius
                         : Lerp(rng.Uniform<Float>(), -radius, radius);
        Float phiMax =
            rng.Uniform<Float>() < 0.5 ? 360. : rng.Uniform<Float>() * 360.;
        Sphere sphere(&identity, &identity, false, radius, zMin, zMax, phiMax);

        // Ray origin
        Point3f o;
        for (int c = 0; c < 3; ++c) o[c] = pExp(rng);

        // Destination: a random point in the shape's bounding box.
        Bounds3f bbox = sphere.WorldBound();
        Point3f t;
        for (int c = 0; c < 3; ++c) t[c] = rng.Uniform<Float>();
        Point3f p2 = bbox.Lerp(t);

        // Ray to intersect with the shape.
        Ray r(o, p2 - o);
        if (rng.Uniform<Float>() < .5) r.d = Normalize(r.d);

        // We should usually (but not always) find an intersection.
        auto si = sphere.Intersect(r, Infinity);
        if (!si) continue;

        Float dot = Dot(Normalize(si->intr.n),
                        Normalize(Vector3f(si->intr.p())));
        EXPECT_FLOAT_EQ(1., dot);
    }
}

TEST(PartialSphere, Reintersect) {
    ParallelFor(0, nReintersect, [](int64_t i) {
        RNG rng(i);
        Transform identity;
        Float radius = pExp(rng, 4);
        Float zMin = rng.Uniform<Float>() < 0.5
                         ? -radius
                         : Lerp(rng.Uniform<Float>(), -radius, radius);
        Float zMax = rng.Uniform<Float>() < 0.5
                         ? radius
                         : Lerp(rng.Uniform<Float>(), -radius, radius);
        Float phiMax =
            rng.Uniform<Float>() < 0.5 ? 360. : rng.Uniform<Float>() * 360.;
        Sphere sphere(&identity, &identity, false, radius, zMin, zMax, phiMax);

        EXPECT_EQ(0, TestReintersectConvex(&sphere, rng)) << i;
    }
        );
}

TEST(Cylinder, Reintersect) {
    ParallelFor(0, nReintersect, [](int64_t i) {
        RNG rng(i);
        Transform identity;
        Float radius = pExp(rng, 4);
        Float zMin = pExp(rng, 4) * (rng.Uniform<Float>() < 0.5 ? -1 : 1);
        Float zMax = pExp(rng, 4) * (rng.Uniform<Float>() < 0.5 ? -1 : 1);
        Float phiMax =
            rng.Uniform<Float>() < 0.5 ? 360. : rng.Uniform<Float>() * 360.;
        Cylinder cyl(&identity, &identity, false, radius, zMin, zMax, phiMax);

        EXPECT_EQ(0, TestReintersectConvex(&cyl, rng)) << i;

        Transform tr = randomTransform(rng);
        Transform ti = Inverse(tr);
        Cylinder tcyl(&tr, &ti, false, radius, zMin, zMax, phiMax);
        EXPECT_EQ(0, TestReintersectConvex(&tcyl, rng)) << i;
    });
}

TEST(Triangle, BadCases) {
    Transform identity;
    std::vector<int> indices{ 0, 1, 2 };
    std::vector<Point3f> p { Point3f( -1113.45459, -79.049614, -56.2431908),
                             Point3f(-1113.45459, -87.0922699, -56.2431908),
                             Point3f(-1113.45459, -79.2090149, -56.2431908) };
    TriangleMesh mesh(identity, false, indices, p, {}, {}, {}, {});
    auto tris = mesh.CreateTriangles(Allocator());
    ASSERT_EQ(1, tris.size());

    Ray ray(Point3f( -1081.47925, 99.9999542, 87.7701111),
            Vector3f(-32.1072998, -183.355865, -144.607635), 0.9999);

    EXPECT_FALSE(tris[0].Intersect(ray).has_value());
}
