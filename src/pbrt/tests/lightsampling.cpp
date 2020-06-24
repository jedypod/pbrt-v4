
#include <gtest/gtest.h>

#include <pbrt/interaction.h>
#include <pbrt/lightsampling.h>
#include <pbrt/scene.h>
#include <pbrt/transform.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/lights.h>
#include <pbrt/shapes.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/math.h>
#include <pbrt/util/rng.h>

#include <memory>
#include <tuple>
#include <vector>

using namespace pbrt;

TEST(BVHLightSampling, OneSpot) {
    Transform id;
    std::vector<LightHandle> lights;
    lights.push_back(new SpotLight(AnimatedTransform(&id),
                                   MediumInterface(),
                                   SPDs::One(),
                                   45.f /* total width */,
                                   44.f /* falloff start */,
                                   Allocator()));
    Scene scene(nullptr, std::move(lights));
    BVHLightSampler distrib(scene.lights, Allocator());

    RNG rng;
    MemoryArena arena;
    for (int i = 0; i < 100; ++i) {
        // Random point in [-5, 5]
        Point3f p(Lerp(rng.Uniform<Float>(), -5, 5),
                  Lerp(rng.Uniform<Float>(), -5, 5),
                  Lerp(rng.Uniform<Float>(), -5, 5));

        Interaction in(p, 0., (const Medium *)nullptr);
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
        pstd::optional<LightLiSample> ls = scene.lights[0].Sample_Li(in, u, lambda);

        SampledLightVector vec = distrib.Sample(in, rng.Uniform<Float>(), arena);

        if (vec.empty()) {
            EXPECT_FALSE(ls.has_value());
            continue;
        } else
            EXPECT_TRUE(ls.has_value());

        ASSERT_EQ(1, vec.size());
        LightHandle light = vec[0].light;
        Float distribPdf = vec[0].pdf;

        EXPECT_EQ(1, distribPdf);
        EXPECT_TRUE(light == scene.lights[0]);
        EXPECT_TRUE((bool)ls->L) << ls->L << " @ " << p;
    }
}

// For a random collection of point lights, make sure that they're all sampled
// with an appropriate ratio of frequency to pdf.
TEST(BVHLightSampling, Point) {
    RNG rng;
    std::vector<LightHandle> lights;
    std::unordered_map<LightHandle, int, LightHandleHash> lightToIndex;
    for (int i = 0; i < 33; ++i) {
        // Random point in [-5, 5]
        Vector3f p(Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5));
        lights.push_back(new PointLight(
            // FIXME: leak
            AnimatedTransform(new Transform(Translate(p))),
            MediumInterface(),
            SPDs::One(), Allocator()));
        lightToIndex[lights.back()] = i;
    }
    Scene scene(nullptr, std::move(lights));
    BVHLightSampler distrib(scene.lights, Allocator());
    MemoryArena arena;

    for (int i = 0; i < 10; ++i) {
        // Don't get too close to the light bbox
        auto r = [&rng]() {
            return rng.Uniform<Float>() < .5 ?
            Lerp(rng.Uniform<Float>(), -15, -7) :
            Lerp(rng.Uniform<Float>(), 7, 16);
        };
        Point3f p(r(), r(), r());

        std::vector<Float> sumWt(scene.lights.size(), 0.f);
        const int nSamples = 100000;
        for (int j = 0; j < nSamples; ++j) {
            Float u = (j + rng.Uniform<Float>()) / nSamples;
            Interaction intr(p, 0, (const Medium *)nullptr);
            SampledLightVector vec = distrib.Sample(intr, u, arena);
            // Can assume this because it's all point lights
            ASSERT_EQ(1, vec.size());

            LightHandle light = vec[0].light;
            Float pdf = vec[0].pdf;
            EXPECT_GT(pdf, 0);
            sumWt[lightToIndex[light]] += 1 / (pdf * nSamples);

            EXPECT_FLOAT_EQ(pdf, distrib.PDF(intr, light));
        }

        for (int i = 0; i < scene.lights.size(); ++i) {
            EXPECT_GE(sumWt[i], .98);
            EXPECT_LT(sumWt[i], 1.02);
        }
    }
}

// Similar to BVHLightSampling.Point, but vary light power
TEST(BVHLightSampling, PointVaryPower) {
    RNG rng(53251);
    std::vector<LightHandle> lights;
    std::vector<Float> lightPower;
    std::vector<std::unique_ptr<ConstantSpectrum>> lightSpectra;
    Float sumPower = 0;
    std::unordered_map<LightHandle, int, LightHandleHash> lightToIndex;
    for (int i = 0; i < 82; ++i) {
        // Random point in [-5, 5]
        Vector3f p(Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5));
        lightPower.push_back(rng.Uniform<Float>());
        lightSpectra.push_back(std::make_unique<ConstantSpectrum>(lightPower.back()));
        sumPower += lightPower.back();
        lights.push_back(new PointLight(
            // FIXME: Leak
            AnimatedTransform(new Transform(Translate(p))),
            MediumInterface(),
            lightSpectra.back().get(),
            Allocator()));
        lightToIndex[lights.back()] = i;
    }
    Scene scene(nullptr, std::move(lights));
    BVHLightSampler distrib(scene.lights, Allocator());
    MemoryArena arena;

    for (int i = 0; i < 10; ++i) {
        // Don't get too close to the light bbox
        auto r = [&rng]() {
            return rng.Uniform<Float>() < .5 ?
            Lerp(rng.Uniform<Float>(), -15, -7) :
            Lerp(rng.Uniform<Float>(), 7, 16);
        };
        Point3f p(r(), r(), r());

        std::vector<Float> sumWt(scene.lights.size(), 0.f);
        const int nSamples = 200000;
        for (int j = 0; j < nSamples; ++j) {
            Float u = (j + rng.Uniform<Float>()) / nSamples;
            Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
            SampledLightVector vec = distrib.Sample(intr, u, arena);
            // Again because it's all point lights...
            ASSERT_EQ(1, vec.size());

            LightHandle light = vec[0].light;
            Float pdf = vec[0].pdf;
            EXPECT_GT(pdf, 0);
            sumWt[lightToIndex[light]] += 1 / (pdf * nSamples);

            EXPECT_LT(std::abs(distrib.PDF(intr, light) - pdf) / pdf, 1e-4);
        }

        for (int i = 0; i < scene.lights.size(); ++i) {
            EXPECT_GE(sumWt[i], .98);
            EXPECT_LT(sumWt[i], 1.02);
        }
    }

    // Now, for very far away points (so d^2 is about the same for all
    // lights), make sure that sampling frequencies for each light are
    // basically proportional to their power
    for (int i = 0; i < 10; ++i) {
        // Don't get too close to the light bbox
        auto r = [&rng]() {
            return rng.Uniform<Float>() < .5 ?
            Lerp(rng.Uniform<Float>(), -15, -7) :
            Lerp(rng.Uniform<Float>(), 7, 16);
        };
        Point3f p(10000 * r(), 10000 * r(), 10000 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));

        std::vector<int> counts(scene.lights.size(), 0);
        const int nSamples = 100000;
        for (int j = 0; j < nSamples; ++j) {
            Float u = (j + rng.Uniform<Float>()) / nSamples;
            SampledLightVector vec = distrib.Sample(intr, u, arena);
            ASSERT_EQ(1, vec.size());
            LightHandle light = vec[0].light;
            Float pdf = vec[0].pdf;
            EXPECT_GT(pdf, 0);
            ++counts[lightToIndex[light]];

            EXPECT_FLOAT_EQ(pdf, distrib.PDF(intr, light));
        }

        for (int i = 0; i < scene.lights.size(); ++i) {
            Float expected = nSamples * lightPower[i] / sumPower;
            EXPECT_GE(counts[i], .97 * expected);
            EXPECT_LT(counts[i], 1.03 * expected);
        }
    }
}

TEST(BVHLightSampling, OneTri) {
    RNG rng(5251);
    Transform id;
    std::vector<int> indices { 0, 1, 2 };
    // Light is illuminating points with z > 0
    std::vector<Point3f> p { Point3f(-1, -1, 0), Point3f(1, -1, 0), Point3f(0, 1, 0) };
    MemoryArena arena;
    TriangleMesh mesh(id, false /* rev orientation */, indices, p, {}, {}, {}, {});
    auto tris = mesh.CreateTriangles(Allocator());

    ASSERT_EQ(1, tris.size());
    std::vector<LightHandle> lights;
    lights.push_back(new DiffuseAreaLight(AnimatedTransform(&id),
                                          MediumInterface(),
                                          SPDs::One(),
                                          1.f,
                                          tris[0],
                                          pstd::optional<Image>{}, nullptr,
                                          false /* two sided */,
                                          Allocator()));

    Scene scene(nullptr, std::move(lights));
    BVHLightSampler distrib(scene.lights, Allocator());
    for (int i = 0; i < 10; ++i) {
        // Random point in [-5, 5]
        Point3f p(Lerp(rng.Uniform<Float>(), -5, 5),
                  Lerp(rng.Uniform<Float>(), -5, 5),
                  Lerp(rng.Uniform<Float>(), -5, 5));

        Interaction in(p, 0., (const Medium *)nullptr);
        Point2f u(rng.Uniform<Float>(), rng.Uniform<Float>());
        SampledWavelengths lambda = SampledWavelengths::SampleEqui(0.5);
        pstd::optional<LightLiSample> ls = scene.lights[0].Sample_Li(in, u, lambda);

        SampledLightVector vec = distrib.Sample(in, rng.Uniform<Float>(), arena);
        if (!vec.empty()) {
            EXPECT_TRUE(ls.has_value());
            ASSERT_EQ(1, vec.size());
            EXPECT_EQ(1, vec[0].pdf);
            EXPECT_TRUE(vec[0].light == scene.lights[0]);
            EXPECT_FLOAT_EQ(vec[0].pdf, distrib.PDF(in, vec[0].light));
            // Li may be 0 due to approximations in the importance metric.
        } else {
            EXPECT_FALSE(ls.has_value());
        }
    }
}

static std::tuple<std::vector<LightHandle>,
                  std::vector<ShapeHandle>> randomLights(int n, MemoryArena *arena) {
    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> allTris;
    RNG rng(6502);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    Transform id;
    for (int i = 0; i < n; ++i) {
        // Triangle
        {
        std::vector<int> indices { 0, 1, 2 };
        std::vector<Point3f> p { Point3f(r(), r(), r()), Point3f(r(), r(), r()),
                                 Point3f(r(), r(), r()) };
        TriangleMesh mesh(id, false /* rev orientation */,
                          indices, p, {}, {}, {}, {});
        auto tris = mesh.CreateTriangles(Allocator());
        CHECK_EQ(1, tris.size());  // EXPECT doesn't work since this is in a function :-p
        static Transform id;
        lights.push_back(new DiffuseAreaLight(
                 AnimatedTransform(&id),
                 MediumInterface(),
                 arena->Alloc<ConstantSpectrum>(r()),
                 1.f,
                 tris[0],
                 pstd::optional<Image>{}, nullptr,
                 false /* two sided */,
                 Allocator()));
        allTris.push_back(tris[0]);
        }

        // Random point light
        {
        Vector3f p(Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5),
                   Lerp(rng.Uniform<Float>(), -5, 5));
        // FIXME: leaks
        lights.push_back(new PointLight(
            AnimatedTransform(new Transform(Translate(p))),
            MediumInterface(),
            arena->Alloc<ConstantSpectrum>(r()),
            Allocator()));
        }
    }

    return {std::move(lights), std::move(allTris)};
}

TEST(BVHLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    MemoryArena arena;
    std::tie(lights, tris) = randomLights(20, &arena);
    Scene scene(nullptr, std::move(lights));

    BVHLightSampler distrib(scene.lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Float u = rng.Uniform<Float>();
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        SampledLightVector vec = distrib.Sample(intr, u, arena);
        // It's actually legit to sometimes get no lights; as the bounds
        // tighten up as we get deeper in the tree, it may turn out that
        // the path we followed didn't have any lights after all.
        for (const auto &light : vec)
            EXPECT_FLOAT_EQ(light.pdf, distrib.PDF(intr, light.light));
    }
}

TEST(ExhaustiveLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    MemoryArena arena;
    std::tie(lights, tris) = randomLights(20, &arena);

    Scene scene(nullptr, std::move(lights));

    ExhaustiveLightSampler distrib(scene.lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        SampledLightVector vec = distrib.Sample(intr, rng.Uniform<Float>(), arena);
        ASSERT_GT(vec.size(), 0) << i << " - " << p;
        for (const auto &light : vec)
            EXPECT_FLOAT_EQ(light.pdf, distrib.PDF(intr, light.light));
    }
}

TEST(UniformLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    MemoryArena arena;
    std::tie(lights, tris) = randomLights(20, &arena);

    Scene scene(nullptr, std::move(lights));

    UniformLightSampler distrib(scene.lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        SampledLightVector vec = distrib.Sample(intr, rng.Uniform<Float>(), arena);
        ASSERT_GT(vec.size(), 0) << i << " - " << p;
        for (const auto &light : vec)
            EXPECT_FLOAT_EQ(light.pdf, distrib.PDF(intr, light.light));
    }
}

TEST(PowerLightSampling, PdfMethod) {
    RNG rng(5251);
    auto r = [&rng]() { return rng.Uniform<Float>(); };

    std::vector<LightHandle> lights;
    std::vector<ShapeHandle> tris;
    MemoryArena arena;
    std::tie(lights, tris) = randomLights(20, &arena);

    Scene scene(nullptr, std::move(lights));

    PowerLightSampler distrib(scene.lights, Allocator());
    for (int i = 0; i < 100; ++i) {
        Point3f p(-1 + 3 * r(), -1 + 3 * r(), -1 + 3 * r());
        Interaction intr(Point3fi(p), Normal3f(0, 0, 0), Point2f(0, 0));
        SampledLightVector vec =
            distrib.Sample(intr, rng.Uniform<Float>(), arena);
        ASSERT_GT(vec.size(), 0) << i << " - " << p;
        for (const auto &light : vec)
            EXPECT_FLOAT_EQ(light.pdf, distrib.PDF(intr, light.light));
    }
}
