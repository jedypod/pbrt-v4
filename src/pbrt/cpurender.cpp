
#include <pbrt/cpurender.h>

#include <pbrt/genscene.h>
#include <pbrt/accelerators.h>
#include <pbrt/cameras.h>
#include <pbrt/film.h>
#include <pbrt/filters.h>
#include <pbrt/integrators.h>
#include <pbrt/lights.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/plymesh.h>
#include <pbrt/samplers.h>
#include <pbrt/scene.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/colorspace.h>

namespace pbrt {

#if 0
    std::vector<PrimitiveHandle> &in =
        renderOptions->instances[name];
    if (in.empty()) return;

    if (in.size() > 1) {
        // Create aggregate for instance _Primitive_s
        PrimitiveHandle accel =
            CreateAccelerator(renderOptions->AcceleratorName, std::move(in),
                              renderOptions->AcceleratorParams);
        in.clear();
        in.push_back(std::move(accel));
    }
    static_assert(MaxTransforms == 2,
                  "TransformCache assumes only two transforms");

    TransformSet worldFromCameraT = Inverse(renderOptions->cameraFromWorldT);

    // Create _animatedWorldFromInstance_ transform for instance
    if (CTMIsAnimated()) {
        const Transform *WorldFromInstance[2] = {
            transformCache.Lookup(GetCTM(0) * worldFromCameraT[0]),
            transformCache.Lookup(GetCTM(1) * worldFromCameraT[1])
        };
        AnimatedTransform animatedWorldFromInstance(
            WorldFromInstance[0], renderOptions->transformStartTime,
            WorldFromInstance[1], renderOptions->transformEndTime);
        PrimitiveHandle prim(
            new AnimatedPrimitive(std::move(in[0]), animatedWorldFromInstance));
        renderOptions->primitives.push_back(std::move(prim));
    } else {
        const Transform *WorldFromInstance =
            transformCache.Lookup(GetCTM(0) * worldFromCameraT[0]);
        PrimitiveHandle prim(new TransformedPrimitive(in[0], WorldFromInstance));
        renderOptions->primitives.push_back(std::move(prim));
    }
#endif

static std::unique_ptr<Medium> CreateMedium(const std::string &name,
                                            const ParameterDictionary &dict,
                                            const Transform &worldFromMedium, FileLoc loc,
                                            Allocator alloc) {
    std::unique_ptr<Medium> m;
    if (name == "homogeneous")
        m = HomogeneousMedium::Create(dict, alloc);
    else if (name == "heterogeneous")
        m = GridDensityMedium::Create(dict, worldFromMedium, alloc);
    else
        ErrorExit(&loc, "%s: medium unknown.", name);

    if (!m)
        ErrorExit(&loc, "%s: unable to create medium.", name);

    dict.ReportUnused();
    return m;
}

static PrimitiveHandle CreateAccelerator(
    const std::string &name, std::vector<PrimitiveHandle> prims,
    const ParameterDictionary &dict) {
    PrimitiveHandle accel;
    if (name == "bvh")
        accel = BVHAccel::Create(std::move(prims), dict);
    else if (name == "kdtree")
        accel = KdTreeAccel::Create(std::move(prims), dict);
    else
        ErrorExit("%s: accelerator type unknown.", name);

    if (!accel)
        ErrorExit("%s: unable to create accelerator.", name);

    dict.ReportUnused();
    return accel;
}

#if 0
    // Check if volume scattering integrator is expected but not specified.
    if (haveScatteringMedia &&
        IntegratorName != "volpath" &&
        IntegratorName != "bdpt" &&
        IntegratorName != "mlt") {
        Warning("Scene has scattering media but \"%s\" integrator doesn't support "
                "volume scattering. Consider using \"volpath\", \"bdpt\", or "
                "\"mlt\".", IntegratorName);
    }

    // Warn if no light sources are defined
    if ((*scene)->lights.empty() && IntegratorName != "ambientocclusion" &&
        IntegratorName != "aov")
        Warning("No light sources defined in scene; rendering a black image.");
}
#endif

static std::unique_ptr<Camera> CreateCamera(
    const std::string &name, const ParameterDictionary &dict,
    const Medium *medium, const AnimatedTransform &worldFromCamera,
    std::unique_ptr<Film> film, const FileLoc *loc) {
    std::unique_ptr<Camera> camera;

    if (name == "perspective")
        camera.reset(PerspectiveCamera::Create(dict, worldFromCamera, std::move(film),
                                               medium));
    else if (name == "orthographic")
        camera.reset(OrthographicCamera::Create(dict, worldFromCamera, std::move(film),
                                                medium));
    else if (name == "realistic")
        camera.reset(RealisticCamera::Create(dict, worldFromCamera, std::move(film),
                                             medium));
    else if (name == "spherical")
        camera.reset(SphericalCamera::Create(dict, worldFromCamera, std::move(film),
                                             medium));
    else
        ErrorExit(loc, "%s: camera type unknown.", name);

    if (!camera)
        ErrorExit(loc, "%s: unable to create camera.", name);

    dict.ReportUnused();
    return camera;
}

static std::unique_ptr<Sampler> CreateSampler(const std::string &name,
                                              const ParameterDictionary &dict,
                                              const Point2i &fullResolution,
                                              const FileLoc *loc) {
    std::unique_ptr<Sampler> sampler;
    if (name == "paddedsobol")
        sampler = PaddedSobolSampler::Create(dict);
    else if (name == "halton")
        sampler = HaltonSampler::Create(dict, fullResolution);
    else if (name == "sobol")
        sampler = SobolSampler::Create(dict, fullResolution);
    else if (name == "random")
        sampler = RandomSampler::Create(dict);
    else if (name == "pmj02bn")
        sampler = PMJ02BNSampler::Create(dict);
    else if (name == "stratified")
        sampler = StratifiedSampler::Create(dict);
    else
        ErrorExit(loc, "%s: sampler type unknown.", name);

    if (!sampler)
        ErrorExit(loc, "%s: unable to create sampler.", name);

    dict.ReportUnused();
    return sampler;
}

static std::unique_ptr<Film> CreateFilm(
    const std::string &name, const ParameterDictionary &dict, const FileLoc *loc,
    pstd::unique_ptr<Filter> filter) {
    std::unique_ptr<Film> film;
    if (name == "rgb")
        film.reset(RGBFilm::Create(dict, std::move(filter), dict.ColorSpace()));
    else if (name == "aov")
        film = AOVFilm::Create(dict, std::move(filter), dict.ColorSpace());
    else
        ErrorExit(loc, "%s: film type unknown.", name);

    if (!film)
        ErrorExit(loc, "%s: unable to create film.", name);

    dict.ReportUnused();
    return film;
}

static std::unique_ptr<Integrator> CreateIntegrator(const std::string &name,
                                                    const ParameterDictionary &dict,
                                                    const Scene &scene,
                                                    std::unique_ptr<Camera> camera,
                                                    std::unique_ptr<Sampler> sampler,
                                                    const RGBColorSpace *colorSpace, FileLoc loc) {
    std::unique_ptr<Integrator> integrator;
    if (name == "whitted")
        integrator = WhittedIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
    else if (name == "path")
        integrator = PathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
    else if (name == "simplepath")
        integrator = SimplePathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
    else if (name == "lightpath")
        integrator = LightPathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
    else if (name == "volpath")
        integrator = VolPathIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
#ifdef PBRT_DISABLE_BDPT_MLT
    else if (name == "bdpt")
        integrator = BDPTIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
    else if (name == "mlt")
        integrator = MLTIntegrator::Create(dict, scene, std::move(camera));
#endif
    else if (name == "ambientocclusion")
        integrator = AOIntegrator::Create(dict, scene, &colorSpace->illuminant,
                                          std::move(camera), std::move(sampler));
    else if (name == "ris")
        integrator = RISIntegrator::Create(dict, scene, std::move(camera), std::move(sampler));
    else if (name == "sppm")
        integrator = SPPMIntegrator::Create(dict, scene, colorSpace, std::move(camera));
    else
        ErrorExit(&loc, "%s: integrator type unknown.", name);

    if (!integrator)
        ErrorExit(&loc, "%s: unable to create integrator.", name);

    dict.ReportUnused();
    return integrator;
}

void CPURender(GeneralScene &genScene) {
    ProfilerScope _(ProfilePhase::CPUSceneConstruction);
    MemoryArena arena;  // TODO: merge with scene arena from api....
    Allocator alloc;

    LOG_VERBOSE("Scene: %s", genScene);

    // Create media first (so have them for the camera...)
    std::map<std::string, std::unique_ptr<Medium>> media;
    for (const auto &m : genScene.media) {
        std::string type = m.second.parameters.GetOneString("type", "");
        if (type.empty())
            ErrorExit(&m.second.loc, "No parameter string \"type\" found for medium.");

        if (m.second.worldFromObject.IsAnimated())
            Warning(&m.second.loc, "Animated transformation provided for medium. Only the "
                    "start transform will be used.");
        std::unique_ptr<Medium> medium = CreateMedium(type, m.second.parameters,
                                                      *m.second.worldFromObject.startTransform,
                                                      m.second.loc, alloc);
        media[m.first] = std::move(medium);
    }
    auto findMedium = [&media](const std::string &s, const FileLoc *loc) -> Medium * {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        return iter->second.get();
    };

    // Filter
    pstd::unique_ptr<Filter> filter(genScene.filter.name.empty() ?
                                    Filter::Create("gaussian", {}, nullptr, alloc) :
                                    Filter::Create(genScene.filter.name,
                                                   genScene.filter.parameters,
                                                   &genScene.filter.loc, alloc));

    // Film
    ParameterDictionary defaultFilmDict({}, RGBColorSpace::sRGB);
    std::unique_ptr<Film> film = genScene.film.name.empty() ?
        CreateFilm("rgb", defaultFilmDict, nullptr, std::move(filter)) :
        CreateFilm(genScene.film.name, genScene.film.parameters, &genScene.film.loc,
                   std::move(filter));

    // Camera
    const Medium *cameraMedium = findMedium(genScene.camera.medium, &genScene.camera.loc);
    std::unique_ptr<Camera> camera =
        genScene.camera.name.empty() ?
        CreateCamera("perspective", {}, nullptr, genScene.camera.worldFromCamera,
                     std::move(film), nullptr) :
        CreateCamera(genScene.camera.name, genScene.camera.parameters, cameraMedium,
                     genScene.camera.worldFromCamera, std::move(film), &genScene.camera.loc);

    // Sampler
    std::unique_ptr<Sampler> sampler = genScene.sampler.name.empty() ?
        CreateSampler("pmj02bn", {}, camera->film->fullResolution, nullptr) :
        CreateSampler(genScene.sampler.name, genScene.sampler.parameters,
                      camera->film->fullResolution, &genScene.sampler.loc);

    // Textures
    std::map<std::string, FloatTextureHandle> floatTextures;
    std::map<std::string, SpectrumTextureHandle> spectrumTextures;
    for (const auto &tex : genScene.floatTextures) {
        if (tex.second.worldFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                    "Using start transform.");
        // TODO: Texture could hold a texture pointer...
        Transform worldFromTexture = *tex.second.worldFromObject.startTransform;

        TextureParameterDictionary texDict(&tex.second.parameters, &floatTextures,
                                           &spectrumTextures);
        FloatTextureHandle t =
            FloatTextureHandle::Create(tex.second.texName, worldFromTexture,
                                       texDict, alloc, tex.second.loc, false);
        floatTextures[tex.first] = std::move(t);
    }
    for (const auto &tex : genScene.spectrumTextures) {
        if (tex.second.worldFromObject.IsAnimated())
            Warning(&tex.second.loc, "Animated world to texture transform not supported. "
                    "Using start transform.");

        Transform worldFromTexture = *tex.second.worldFromObject.startTransform;

        TextureParameterDictionary texDict(&tex.second.parameters, &floatTextures,
                                           &spectrumTextures);
        SpectrumTextureHandle t =
            SpectrumTextureHandle::Create(tex.second.texName, worldFromTexture,
                                          texDict, alloc, tex.second.loc, false);
        spectrumTextures[tex.first] = std::move(t);
    }

    // Materials
    std::map<std::string, MaterialHandle> namedMaterials;
    std::vector<MaterialHandle> materials;
    genScene.CreateMaterials(floatTextures, spectrumTextures, alloc, &namedMaterials,
                             &materials);

    // Lights (area lights with shapes...)
    std::vector<LightHandle> lights;
    lights.reserve(genScene.lights.size() + genScene.areaLights.size());
    for (const auto &light : genScene.lights) {
        ProfilerScope _(ProfilePhase::LightConstruction);
        const Medium *outsideMedium = findMedium(light.medium, &light.loc);
        LightHandle l = LightHandle::Create(light.name, light.parameters, light.worldFromObject,
                                            outsideMedium, light.loc, Allocator{} /* FIXME: leaks*/);
        lights.push_back(l);
    }

    // Primitives
    auto getAlphaTexture = [&](const ParameterDictionary &parameters,
                               const FileLoc *loc) -> FloatTextureHandle {
        std::string alphaTexName = parameters.GetTexture("alpha");
        if (!alphaTexName.empty()) {
            if (floatTextures.find(alphaTexName) != floatTextures.end())
                return floatTextures[alphaTexName];
            else
                ErrorExit(loc, "%s: couldn't find float texture for \"alpha\" parameter.",
                          alphaTexName);
        } else if (parameters.GetOneFloat("alpha", 1.f) == 0.f)
            return alloc.new_object<FloatConstantTexture>(0.f);
        else
            return nullptr;
    };

    // Non-animated shapes
    auto CreatePrimitivesForShapes = [&](const std::vector<ShapeSceneEntity> &shapes)
        -> std::vector<PrimitiveHandle> {
        ProfilerScope _(ProfilePhase::ShapeConstruction);
        std::vector<PrimitiveHandle> primitives;
        for (const auto &sh : shapes) {
            pstd::vector<ShapeHandle> shapes =
                ShapeHandle::Create(sh.name, sh.worldFromObject, sh.objectFromWorld,
                                    sh.reverseOrientation, sh.parameters,
                                    alloc, sh.loc);
            if (shapes.empty())
                continue;

            FloatTextureHandle alphaTex = getAlphaTexture(sh.parameters, &sh.loc);
            sh.parameters.ReportUnused(); // do now so can grab alpha...

            MaterialHandle mtl = nullptr;
            if (!sh.materialName.empty()) {
                auto iter = namedMaterials.find(sh.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
                mtl = iter->second;
            } else {
                CHECK_LT(sh.materialIndex, materials.size());
                mtl = materials[sh.materialIndex];
            }

            MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                               findMedium(sh.outsideMedium, &sh.loc));

            for (auto &s : shapes) {
                // Possibly create area light for shape
                LightHandle areaHandle = nullptr;
                if (sh.lightIndex != -1) {
                    ProfilerScope _(ProfilePhase::LightConstruction);
                    CHECK_LT(sh.lightIndex, genScene.areaLights.size());
                    const auto &areaLightEntity = genScene.areaLights[sh.lightIndex];

                    // Unique ptr ok since we're using default allocator.
                    LightHandle area =
                        LightHandle::CreateArea(areaLightEntity.name, areaLightEntity.parameters,
                                                AnimatedTransform(sh.worldFromObject),
                                                mi, s, areaLightEntity.loc, Allocator{} /* FIXME LEAK */);
                    areaHandle = area;
                    if (area) lights.push_back(area);
                }
                if (areaHandle == nullptr && !mi.IsMediumTransition() && !alphaTex)
                    primitives.push_back(new SimplePrimitive(s, mtl));
                else
                    primitives.push_back(new GeometricPrimitive(s, mtl, areaHandle, mi, alphaTex));
            }
        }
        return primitives;
    };

    std::vector<PrimitiveHandle> primitives = CreatePrimitivesForShapes(genScene.shapes);

    // Animated shapes
    auto CreatePrimitivesForAnimatedShapes = [&](const std::vector<AnimatedShapeSceneEntity> &shapes)
        -> std::vector<PrimitiveHandle> {
        ProfilerScope _(ProfilePhase::ShapeConstruction);
        std::vector<PrimitiveHandle> primitives;
        primitives.reserve(shapes.size());

        for (const auto &sh : shapes) {
            pstd::vector<ShapeHandle> shapes =
                ShapeHandle::Create(sh.name, sh.identity, sh.identity,
                                    sh.reverseOrientation, sh.parameters,
                                    alloc, sh.loc);
            if (shapes.empty())
                continue;

            FloatTextureHandle alphaTex = getAlphaTexture(sh.parameters, &sh.loc);
            sh.parameters.ReportUnused(); // do now so can grab alpha...

            // Create initial shape or shapes for animated shape

            MaterialHandle mtl = nullptr;
            if (!sh.materialName.empty()) {
                auto iter = namedMaterials.find(sh.materialName);
                if (iter == namedMaterials.end())
                    ErrorExit(&sh.loc, "%s: no named material defined.", sh.materialName);
                mtl = iter->second;
            } else {
                CHECK_LT(sh.materialIndex, materials.size());
                mtl = materials[sh.materialIndex];
            }

            MediumInterface mi(findMedium(sh.insideMedium, &sh.loc),
                               findMedium(sh.outsideMedium, &sh.loc));

            std::vector<PrimitiveHandle> prims;
            for (auto &s : shapes) {
                // Possibly create area light for shape
                LightHandle areaHandle = nullptr;
                if (sh.lightIndex != -1) {
                    ProfilerScope _(ProfilePhase::LightConstruction);
                    CHECK_LT(sh.lightIndex, genScene.areaLights.size());
                    const auto &areaLightEntity = genScene.areaLights[sh.lightIndex];

                    LightHandle area =
                        LightHandle::CreateArea(areaLightEntity.name, areaLightEntity.parameters,
                                                sh.worldFromObject, mi, s, sh.loc, Allocator{});
                    areaHandle = area;
                    if (area) lights.push_back(area);
                }
                if (areaHandle == nullptr && !mi.IsMediumTransition() && !alphaTex)
                    prims.push_back(new SimplePrimitive(s, mtl));
                else
                    prims.push_back(new GeometricPrimitive(s, mtl, areaHandle, mi, alphaTex));
            }

            // TODO: could try to be greedy or even segment them according
            // to same sh.worldFromObject...

            // Create single _Primitive_ for _prims_
            if (prims.size() > 1) {
                PrimitiveHandle bvh = new BVHAccel(std::move(prims));
                prims.clear();
                prims.push_back(bvh);
            }
            primitives.push_back(new AnimatedPrimitive(prims[0], sh.worldFromObject));
        }
        return primitives;
    };
    std::vector<PrimitiveHandle> animatedPrimitives =
        CreatePrimitivesForAnimatedShapes(genScene.animatedShapes);
    primitives.insert(primitives.end(), animatedPrimitives.begin(),
                      animatedPrimitives.end());

    // Instance definitions
    std::map<std::string, PrimitiveHandle> instanceDefinitions;
    for (const auto &inst : genScene.instanceDefinitions) {
        if (instanceDefinitions.find(inst.first) != instanceDefinitions.end())
            ErrorExit("%s: object instance redefined", inst.first);

        std::vector<PrimitiveHandle> instancePrimitives =
            CreatePrimitivesForShapes(inst.second.first);
        std::vector<PrimitiveHandle> movingInstancePrimitives =
            CreatePrimitivesForAnimatedShapes(inst.second.second);
        instancePrimitives.insert(instancePrimitives.end(),
                                  movingInstancePrimitives.begin(),
                                  movingInstancePrimitives.end());
        if (instancePrimitives.size() != 1) {
            PrimitiveHandle bvh = new BVHAccel(std::move(instancePrimitives));
            instancePrimitives.clear();
            instancePrimitives.push_back(bvh);
        }
        instanceDefinitions[inst.first] = instancePrimitives[0];
    }

    // Instances
    for (const auto &inst : genScene.instances) {
        auto iter = instanceDefinitions.find(inst.name);
        if (iter == instanceDefinitions.end())
            ErrorExit(&inst.loc, "%s: object instance not defined", inst.name);

        if (inst.worldFromInstance)
            primitives.push_back(new TransformedPrimitive(iter->second,
                                                          inst.worldFromInstance));
        else
            primitives.push_back(new AnimatedPrimitive(iter->second,
                                                       inst.worldFromInstanceAnim));
    }

    // Accelerator
    PrimitiveHandle accel = genScene.accelerator.name.empty() ?
        new BVHAccel(std::move(primitives)) :
        CreateAccelerator(genScene.accelerator.name, std::move(primitives),
                          genScene.accelerator.parameters);

    // Scene
    Scene scene(accel, std::move(lights));

    // Integrator
    const RGBColorSpace *integratorColorSpace = genScene.film.parameters.ColorSpace();
    std::unique_ptr<Integrator> integrator = genScene.integrator.name.empty() ?
        CreateIntegrator("path", {}, scene, std::move(camera), std::move(sampler),
                         integratorColorSpace, {}) :
        CreateIntegrator(genScene.integrator.name, genScene.integrator.parameters,
                         scene, std::move(camera), std::move(sampler),
                         integratorColorSpace, genScene.integrator.loc);

    LOG_VERBOSE("Memory used after scene creation: %d", GetCurrentRSS());

    // This is kind of ugly; we directly override the current profiler
    // state to switch from parsing/scene construction related stuff to
    // rendering stuff and then switch it back below. The underlying
    // issue is that all the rest of the profiling system assumes
    // hierarchical inheritance of profiling state; this is the only
    // place where that isn't the case.
    if (PbrtOptions.profile) {
        CHECK_EQ(CurrentProfilerState(), (ProfilePhaseToBits(ProfilePhase::SceneConstruction) |
                                          ProfilePhaseToBits(ProfilePhase::ParsingAndGenScene) |
                                          ProfilePhaseToBits(ProfilePhase::CPUSceneConstruction)));
        ProfilerState = ProfilePhaseToBits(ProfilePhase::IntegratorRender);
    }

    // Render!
    integrator->Render();

    LOG_VERBOSE("Memory used after rendering: %s", GetCurrentRSS());

    if (PbrtOptions.profile) {
        CHECK_EQ(CurrentProfilerState(), ProfilePhaseToBits(ProfilePhase::IntegratorRender));
        ProfilerState = (ProfilePhaseToBits(ProfilePhase::SceneConstruction) |
                         ProfilePhaseToBits(ProfilePhase::ParsingAndGenScene) |
                         ProfilePhaseToBits(ProfilePhase::CPUSceneConstruction));
    }

    ImageTextureBase::ClearCache();
    ShapeHandle::FreeBufferCaches();
}

} // namespace pbrt
