
#include <gtest/gtest.h>
#include <pbrt/core/pbrt.h>
#include <pbrt/core/image.h>
#include <pbrt/accelerators/bvh.h>
#include <pbrt/core/api.h>
#include <pbrt/cameras/orthographic.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/core/film.h>
#include <pbrt/filters/box.h>
#include <pbrt/util/geometry.h>
#include <pbrt/integrators/bdpt.h>
#include <pbrt/integrators/mlt.h>
#include <pbrt/integrators/path.h>
#include <pbrt/integrators/volpath.h>
#include <pbrt/lights/diffuse.h>
#include <pbrt/lights/point.h>
#include <pbrt/materials/matte.h>
#include <pbrt/materials/mirror.h>
#include <pbrt/materials/uber.h>
#include <pbrt/samplers/halton.h>
#include <pbrt/samplers/random.h>
#include <pbrt/samplers/stratified.h>
#include <pbrt/samplers/sobol.h>
#include <pbrt/samplers/zerotwosequence.h>
#include <pbrt/core/scene.h>
#include <pbrt/shapes/sphere.h>
#include <pbrt/core/spectrum.h>
#include <pbrt/textures/constant.h>

using namespace pbrt;

static std::string inTestDir(const std::string &path) { return path; }

struct TestScene {
    std::shared_ptr<Scene> scene;
    std::string description;
    float expected;
};

struct TestIntegrator {
    Integrator *integrator;
    Film *film;
    std::string description;
    TestScene scene;
};

void PrintTo(const TestIntegrator &tr, ::std::ostream *os) {
    *os << tr.description;
}

void CheckSceneAverage(const std::string &filename, float expected) {
    absl::optional<Image> image = Image::Read(filename);
    ASSERT_TRUE((bool)image);
    ASSERT_EQ(image->nChannels(), 3);

    float delta = .02;
    float sum = 0;

    for (int t = 0; t < image->resolution[1]; ++t)
      for (int s = 0; s < image->resolution[0]; ++s)
        for (int c = 0; c < 3; ++c) sum += image->GetChannel(Point2i(s, t), c);
    int nPixels = image->resolution.x * image->resolution.y * 3;
    EXPECT_NEAR(expected, sum / nPixels, delta);
}

std::vector<TestScene> GetScenes() {
    std::vector<TestScene> scenes;

    auto id = std::make_shared<const Transform>();
    {
        // Unit sphere, Kd = 0.5, point light I = 3.1415 at center
        // -> With GI, should have radiance of 1.
        std::shared_ptr<Shape> sphere = std::make_shared<Sphere>(
            id, id, true /* reverse orientation */, 1, -1, 1, 360, nullptr);

        std::shared_ptr<Texture<Spectrum>> Kd =
            std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.5));
        std::shared_ptr<Texture<Float>> sigma =
            std::make_shared<ConstantTexture<Float>>(0.);
        std::shared_ptr<Material> material =
            std::make_shared<MatteMaterial>(Kd, sigma, nullptr, nullptr);

        MediumInterface mediumInterface;
        std::vector<std::shared_ptr<Primitive>> prims;
        prims.push_back(std::make_shared<GeometricPrimitive>(
            sphere, material, nullptr, mediumInterface));
        std::shared_ptr<BVHAccel> bvh = std::make_shared<BVHAccel>(prims);

        std::vector<std::shared_ptr<Light>> lights;
        lights.push_back(
            std::make_shared<PointLight>(Transform(), nullptr, Spectrum(Pi),
                                         std::make_shared<ParamSet>()));

        std::unique_ptr<Scene> scene = std::make_unique<Scene>(bvh, lights);

        scenes.push_back({std::move(scene), "Sphere, 1 light, Kd = 0.5", 1.0});
    }

    {
        // Unit sphere, Kd = 0.5, 4 point lights I = 3.1415/4 at center
        // -> With GI, should have radiance of 1.
        std::shared_ptr<Shape> sphere = std::make_shared<Sphere>(
            id, id, true /* reverse orientation */, 1, -1, 1, 360, nullptr);

        std::shared_ptr<Texture<Spectrum>> Kd =
            std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.5));
        std::shared_ptr<Texture<Float>> sigma =
            std::make_shared<ConstantTexture<Float>>(0.);
        std::shared_ptr<Material> material =
            std::make_shared<MatteMaterial>(Kd, sigma, nullptr, nullptr);

        MediumInterface mediumInterface;
        std::vector<std::shared_ptr<Primitive>> prims;
        prims.push_back(std::make_shared<GeometricPrimitive>(
            sphere, material, nullptr, mediumInterface));
        std::shared_ptr<BVHAccel> bvh = std::make_shared<BVHAccel>(prims);

        std::vector<std::shared_ptr<Light>> lights;
        lights.push_back(std::make_shared<PointLight>(Transform(), nullptr,
                                                      Spectrum(Pi / 4),
                                                      std::make_shared<ParamSet>()));
        lights.push_back(std::make_shared<PointLight>(Transform(), nullptr,
                                                      Spectrum(Pi / 4),
                                                      std::make_shared<ParamSet>()));
        lights.push_back(std::make_shared<PointLight>(Transform(), nullptr,
                                                      Spectrum(Pi / 4),
                                                      std::make_shared<ParamSet>()));
        lights.push_back(std::make_shared<PointLight>(Transform(), nullptr,
                                                      Spectrum(Pi / 4),
                                                      std::make_shared<ParamSet>()));

        std::unique_ptr<Scene> scene = std::make_unique<Scene>(bvh, lights);

        scenes.push_back({std::move(scene), "Sphere, 1 light, Kd = 0.5", 1.0});
    }

    {
        // Unit sphere, Kd = 0.5, Le = 0.5
        // -> With GI, should have radiance of 1.
        std::shared_ptr<Shape> sphere = std::make_shared<Sphere>(
            id, id, true /* reverse orientation */, 1, -1, 1, 360, nullptr);

        std::shared_ptr<Texture<Spectrum>> Kd =
            std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.5));
        std::shared_ptr<Texture<Float>> sigma =
            std::make_shared<ConstantTexture<Float>>(0.);
        std::shared_ptr<Material> material =
            std::make_shared<MatteMaterial>(Kd, sigma, nullptr, nullptr);

        std::shared_ptr<AreaLight> areaLight =
            std::make_shared<DiffuseAreaLight>(Transform(), nullptr,
                                               Spectrum(0.5), sphere, false,
                                               std::make_shared<ParamSet>());

        std::vector<std::shared_ptr<Light>> lights;
        lights.push_back(areaLight);

        MediumInterface mediumInterface;
        std::vector<std::shared_ptr<Primitive>> prims;
        prims.push_back(std::make_shared<GeometricPrimitive>(
            sphere, material, areaLight, mediumInterface));
        std::shared_ptr<BVHAccel> bvh = std::make_shared<BVHAccel>(prims);

        std::unique_ptr<Scene> scene = std::make_unique<Scene>(bvh, lights);

        scenes.push_back({std::move(scene), "Sphere, Kd = 0.5, Le = 0.5", 1.0});
    }

    {
        // Unit sphere, Kd = 0.25, Kr = .5, point light I = 7.4 at center
        // -> With GI, should have radiance of ~1.
        std::shared_ptr<Shape> sphere = std::make_shared<Sphere>(
            id, id, true /* reverse orientation */, 1, -1, 1, 360, nullptr);

        std::shared_ptr<Texture<Spectrum>> Kd =
            std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.25));
        std::shared_ptr<Texture<Spectrum>> Kr =
            std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.5));
        std::shared_ptr<Texture<Spectrum>> black =
            std::make_shared<ConstantTexture<Spectrum>>(0.);
        std::shared_ptr<Texture<Spectrum>> white =
            std::make_shared<ConstantTexture<Spectrum>>(1.);
        std::shared_ptr<Texture<Float>> zero =
            std::make_shared<ConstantTexture<Float>>(0.);
        std::shared_ptr<Texture<Float>> one =
            std::make_shared<ConstantTexture<Float>>(1.);
        std::shared_ptr<Material> material = std::make_shared<UberMaterial>(
            Kd, black, Kr, black, zero, zero, zero, white, one, nullptr, false, nullptr);

        MediumInterface mediumInterface;
        std::vector<std::shared_ptr<Primitive>> prims;
        prims.push_back(std::make_shared<GeometricPrimitive>(
            sphere, material, nullptr, mediumInterface));
        std::shared_ptr<BVHAccel> bvh = std::make_shared<BVHAccel>(prims);

        std::vector<std::shared_ptr<Light>> lights;
        lights.push_back(std::make_shared<PointLight>(Transform(), nullptr,
                                                      Spectrum(3. * Pi),
                                                      std::make_shared<ParamSet>()));

        std::unique_ptr<Scene> scene = std::make_unique<Scene>(bvh, lights);

        scenes.push_back(
            {std::move(scene), "Sphere, 1 light, Kd = 0.25 Kr = 0.5", 1.0});
    }

#if 0
  {
    // Unit sphere, Kd = 0.25, Kr = .5, Le .587
    // -> With GI, should have radiance of ~1.
    std::shared_ptr<Shape> sphere = std::make_shared<Sphere>(
        id, id, true /* reverse orientation */, 1, -1, 1, 360, nullptr);

    std::shared_ptr<Texture<Spectrum>> Kd =
        std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.25));
    std::shared_ptr<Texture<Spectrum>> Kr =
        std::make_shared<ConstantTexture<Spectrum>>(Spectrum(0.5));
    std::shared_ptr<Texture<Spectrum>> black =
        std::make_shared<ConstantTexture<Spectrum>>(0.);
    std::shared_ptr<Texture<Spectrum>> white =
        std::make_shared<ConstantTexture<Spectrum>>(1.);
    std::shared_ptr<Texture<Float>> zero =
        std::make_shared<ConstantTexture<Float>>(0.);
    std::shared_ptr<Texture<Float>> one =
        std::make_shared<ConstantTexture<Float>>(1.);
    std::shared_ptr<Material> material = std::make_shared<UberMaterial>(
        Kd, black, Kr, black, zero, zero, zero, white, one, nullptr, false, nullptr);

    std::shared_ptr<AreaLight> areaLight = std::make_shared<DiffuseAreaLight>(
        Transform(), nullptr, Spectrum(0.587), 8, sphere, true, false,
        std::make_shared<ParamSet>());

    MediumInterface mediumInterface;
    std::vector<std::shared_ptr<Primitive>> prims;
    prims.push_back(std::make_shared<GeometricPrimitive>(
        sphere, material, areaLight, mediumInterface));
    std::shared_ptr<BVHAccel> bvh = std::make_shared<BVHAccel>(prims);

    std::vector<std::shared_ptr<Light>> lights;
    lights.push_back(areaLight);

    std::unique_ptr<Scene> scene = std::make_unique<Scene>(bvh, lights);

    scenes.push_back({std::move(scene),
            "Sphere, Kd = 0.25 Kr = 0.5, Le = 0.587", 1.0});
  }
#endif

    return scenes;
}

std::vector<std::pair<std::unique_ptr<Sampler>, std::string>> GetSamplers(
    const Bounds2i &sampleBounds) {
    std::vector<std::pair<std::unique_ptr<Sampler>, std::string>> samplers;

    samplers.push_back(std::make_pair(
        std::make_unique<HaltonSampler>(256, sampleBounds), "Halton 256"));
    samplers.push_back(std::make_pair(
        std::make_unique<ZeroTwoSequenceSampler>(256), "(0,2)-seq 256"));
    samplers.push_back(std::make_pair(
        std::make_unique<SobolSampler>(256, sampleBounds), "Sobol 256"));
    samplers.push_back(
        std::make_pair(std::make_unique<RandomSampler>(256), "Random 256"));
    samplers.push_back(
        std::make_pair(std::make_unique<StratifiedSampler>(16, 16, true, 8),
                       "Stratified 16x16"));

    return samplers;
}

std::vector<TestIntegrator> GetIntegrators() {
    std::vector<TestIntegrator> integrators;

    Point2i resolution(10, 10);
    auto id = std::make_shared<const Transform>();
    AnimatedTransform identity(id, 0, id, 1);

    for (auto scene : GetScenes()) {
        // Path tracing integrators
        for (auto &sampler : GetSamplers(Bounds2i(Point2i(0, 0), resolution))) {
            std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
            std::unique_ptr<Film> film =
                std::make_unique<Film>(resolution, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                         std::move(filter), 1., inTestDir("test.exr"), 1.);
            std::shared_ptr<Camera> camera =
                std::make_shared<PerspectiveCamera>(
                    identity, Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 1.,
                    0., 10., 45, std::move(film), nullptr);

            Integrator *integrator =
                new PathIntegrator(8, camera, std::move(sampler.first),
                                   camera->film->croppedPixelBounds);
            integrators.push_back({integrator, camera->film.get(),
                                   "Path, depth 8, Perspective, " +
                                       sampler.second + ", " +
                                       scene.description,
                                   scene});
        }

        for (auto &sampler : GetSamplers(Bounds2i(Point2i(0, 0), resolution))) {
            std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
            std::unique_ptr<Film> film =
                std::make_unique<Film>(resolution, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                         std::move(filter), 1., inTestDir("test.exr"), 1.);
            std::shared_ptr<Camera> camera =
                std::make_shared<OrthographicCamera>(
                    identity, Bounds2f(Point2f(-.1, -.1), Point2f(.1, .1)), 0.,
                    1., 0., 10., std::move(film), nullptr);

            Integrator *integrator =
                new PathIntegrator(8, camera, std::move(sampler.first),
                                   camera->film->croppedPixelBounds);
            integrators.push_back({integrator, camera->film.get(),
                                   "Path, depth 8, Ortho, " + sampler.second +
                                       ", " + scene.description,
                                   scene});
        }

        // Volume path tracing integrators
        for (auto &sampler : GetSamplers(Bounds2i(Point2i(0, 0), resolution))) {
            std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
            std::unique_ptr<Film> film =
                std::make_unique<Film>(resolution, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                         std::move(filter), 1., inTestDir("test.exr"), 1.);
            std::shared_ptr<Camera> camera =
                std::make_shared<PerspectiveCamera>(
                    identity, Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 1.,
                    0., 10., 45, std::move(film), nullptr);

            Integrator *integrator =
                new VolPathIntegrator(8, camera, std::move(sampler.first),
                                      camera->film->croppedPixelBounds);
            integrators.push_back({integrator, camera->film.get(),
                                   "VolPath, depth 8, Perspective, " +
                                       sampler.second + ", " +
                                       scene.description,
                                   scene});
        }
        for (auto &sampler : GetSamplers(Bounds2i(Point2i(0, 0), resolution))) {
            std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
            std::unique_ptr<Film> film =
                std::make_unique<Film>(resolution, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                         std::move(filter), 1., inTestDir("test.exr"), 1.);
            std::shared_ptr<Camera> camera =
                std::make_shared<OrthographicCamera>(
                    identity, Bounds2f(Point2f(-.1, -.1), Point2f(.1, .1)), 0.,
                    1., 0., 10., std::move(film), nullptr);

            Integrator *integrator =
                new VolPathIntegrator(8, camera, std::move(sampler.first),
                                      camera->film->croppedPixelBounds);
            integrators.push_back({integrator, camera->film.get(),
                                   "VolPath, depth 8, Ortho, " +
                                       sampler.second + ", " +
                                       scene.description,
                                   scene});
        }

        // BDPT
        for (auto &sampler : GetSamplers(Bounds2i(Point2i(0, 0), resolution))) {
            std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
            std::unique_ptr<Film> film =
                std::make_unique<Film>(resolution, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                         std::move(filter), 1., inTestDir("test.exr"), 1.);
            std::shared_ptr<Camera> camera =
                std::make_shared<PerspectiveCamera>(
                    identity, Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 1.,
                    0., 10., 45, std::move(film), nullptr);

            Integrator *integrator =
                new BDPTIntegrator(std::move(sampler.first), camera, 6, false, false,
                                   camera->film->croppedPixelBounds);
            integrators.push_back({integrator, camera->film.get(),
                                   "BDPT, depth 8, Perspective, " +
                                       sampler.second + ", " +
                                       scene.description,
                                   scene});
        }
#if 0
    // Ortho camera not currently supported with BDPT.
    for (auto &sampler : GetSamplers(Bounds2i(Point2i(0,0), resolution))) {
      std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
      std::unique_ptr<Film> film = std::make_unique<Film>(resolution, Bounds2f(Point2f(0,0), Point2f(1,1)),
                            std::move(filter), 1., inTestDir("test.exr"), 1.);
      std::shared_ptr<Camera> camera = std::make_shared<OrthographicCamera>(
          identity, Bounds2f(Point2f(-.1,-.1), Point2f(.1,.1)), 0., 1.,
          0., 10., std::move(film), nullptr);

      Integrator *integrator = new BDPTIntegrator(std::move(sampler.first), camera, 8,
                                            false, false);
      integrators.push_back({integrator, camera->film.get(),
              "BDPT, depth 8, Ortho, " + sampler.second + ", " +
              scene.description, scene});
    }
#endif

        // MLT
        {
            std::unique_ptr<Filter> filter = std::make_unique<BoxFilter>(Vector2f(0.5, 0.5));
            std::unique_ptr<Film> film =
                std::make_unique<Film>(resolution, Bounds2f(Point2f(0, 0), Point2f(1, 1)),
                         std::move(filter), 1., inTestDir("test.exr"), 1.);
            std::shared_ptr<Camera> camera =
                std::make_shared<PerspectiveCamera>(
                    identity, Bounds2f(Point2f(-1, -1), Point2f(1, 1)), 0., 1.,
                    0., 10., 45, std::move(film), nullptr);

            Integrator *integrator = new MLTIntegrator(
                camera, 8 /* depth */, 100000 /* n bootstrap */,
                1000 /* nchains */, 1024 /* mutations per pixel */,
                0.01 /* sigma */, 0.3 /* large step prob */);
            integrators.push_back({integrator, camera->film.get(),
                 "MLT, depth 8, Perspective, " + scene.description, scene});
        }
    }

    return integrators;
}

struct RenderTest : public testing::TestWithParam<TestIntegrator> {};

TEST_P(RenderTest, RadianceMatches) {
    Options options;
    options.quiet = true;
    pbrtInit(options);

    const TestIntegrator &tr = GetParam();
    tr.integrator->Render(*tr.scene.scene);
    CheckSceneAverage(inTestDir("test.exr"), tr.scene.expected);
    // The SpatialLightDistribution class keeps a per-thread cache that
    // must be cleared out between test runs. In turn, this means that we
    // must delete the Integrator here in order to make sure that its
    // destructor runs. (This is ugly and should be fixed in a better way.)
    delete tr.integrator;

    pbrtCleanup();

    EXPECT_EQ(0, remove(inTestDir("test.exr").c_str()));
}

INSTANTIATE_TEST_CASE_P(AnalyticTestScenes, RenderTest,
                        testing::ValuesIn(GetIntegrators()));
