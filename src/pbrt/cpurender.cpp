
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

void CPURender(GeneralScene &genScene) {
    ProfilerScope _(ProfilePhase::SceneConstruction);
    Allocator alloc;

    // Create media first (so have them for the camera...)
    std::map<std::string, std::unique_ptr<Medium>> media;
    for (const auto &m : genScene.media) {
        std::string type = m.second.parameters.GetOneString("type", "");
        if (type.empty())
            ErrorExit(&m.second.loc, "No parameter string \"type\" found for medium.");

        if (m.second.worldFromObject.IsAnimated())
            Warning(&m.second.loc, "Animated transformation provided for medium. Only the "
                    "start transform will be used.");
        std::unique_ptr<Medium> medium(Medium::Create(type, m.second.parameters,
                                                      *m.second.worldFromObject.startTransform,
                                                      &m.second.loc, alloc));
        media[m.first] = std::move(medium);
    }

    bool haveScatteringMedia = false;
    auto findMedium = [&media,&haveScatteringMedia](const std::string &s,
                                                    const FileLoc *loc) -> Medium * {
        if (s.empty())
            return nullptr;

        auto iter = media.find(s);
        if (iter == media.end())
            ErrorExit(loc, "%s: medium not defined", s);
        haveScatteringMedia = true;
        return iter->second.get();
    };

    // Filter
    FilterHandle filter =genScene.filter.name.empty() ?
        FilterHandle::Create("gaussian", {}, nullptr, alloc) :
        FilterHandle::Create(genScene.filter.name, genScene.filter.parameters,
                             &genScene.filter.loc, alloc);

    // Film
    ParameterDictionary defaultFilmDict({}, RGBColorSpace::sRGB);
    Film *film(genScene.film.name.empty() ?
        Film::Create("rgb", defaultFilmDict, nullptr, filter, alloc) :
        Film::Create(genScene.film.name, genScene.film.parameters, &genScene.film.loc,
                     filter, alloc));

    // Camera
    const Medium *cameraMedium = findMedium(genScene.camera.medium, &genScene.camera.loc);
    Camera *camera =
        genScene.camera.name.empty() ?
        Camera::Create("perspective", {}, nullptr, genScene.camera.worldFromCamera,
                       film, nullptr, alloc) :
        Camera::Create(genScene.camera.name, genScene.camera.parameters, cameraMedium,
                       genScene.camera.worldFromCamera, film,
                       &genScene.camera.loc, alloc);

    // Sampler
    SamplerHandle sampler(genScene.sampler.name.empty() ?
        SamplerHandle::Create("pmj02bn", {}, camera->film->fullResolution, nullptr, alloc) :
        SamplerHandle::Create(genScene.sampler.name, genScene.sampler.parameters,
                              camera->film->fullResolution, &genScene.sampler.loc, alloc));

    // Textures
    std::map<std::string, FloatTextureHandle> floatTextures;
    std::map<std::string, SpectrumTextureHandle> spectrumTextures;
    genScene.CreateTextures(&floatTextures, &spectrumTextures, alloc, false);

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
                                            outsideMedium, &light.loc, alloc);
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
                                    &sh.loc, alloc);
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
                                                mi, s, &areaLightEntity.loc, Allocator{} /* FIXME LEAK */);
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
                                    &sh.loc, alloc);
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
                                                sh.worldFromObject, mi, s, &sh.loc, Allocator{});
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
            CreatePrimitivesForShapes(inst.second.shapes);
        std::vector<PrimitiveHandle> movingInstancePrimitives =
            CreatePrimitivesForAnimatedShapes(inst.second.animatedShapes);
        instancePrimitives.insert(instancePrimitives.end(),
                                  movingInstancePrimitives.begin(),
                                  movingInstancePrimitives.end());
        if (instancePrimitives.empty()) {
            Warning(&inst.second.loc, "Empty object instance");
            instanceDefinitions[inst.first] = nullptr;
        } else {
            if (instancePrimitives.size() > 1) {
                PrimitiveHandle bvh = new BVHAccel(std::move(instancePrimitives));
                instancePrimitives.clear();
                instancePrimitives.push_back(bvh);
            }
            instanceDefinitions[inst.first] = instancePrimitives[0];
        }
    }

    // Instances
    for (const auto &inst : genScene.instances) {
        auto iter = instanceDefinitions.find(inst.name);
        if (iter == instanceDefinitions.end())
            ErrorExit(&inst.loc, "%s: object instance not defined", inst.name);

        if (iter->second == nullptr)
            // empty instance
            continue;

        if (inst.worldFromInstance)
            primitives.push_back(new TransformedPrimitive(iter->second,
                                                          inst.worldFromInstance));
        else
            primitives.push_back(new AnimatedPrimitive(iter->second,
                                                       inst.worldFromInstanceAnim));
    }

    // Accelerator
    PrimitiveHandle accel = nullptr;
    if (!primitives.empty())
        accel = genScene.accelerator.name.empty() ?
            new BVHAccel(std::move(primitives)) :
            CreateAccelerator(genScene.accelerator.name, std::move(primitives),
                              genScene.accelerator.parameters);

    // Scene
    Scene scene(accel, std::move(lights));

    // Integrator
    const RGBColorSpace *integratorColorSpace = genScene.film.parameters.ColorSpace();
    std::unique_ptr<Integrator> integrator(genScene.integrator.name.empty() ?
        Integrator::Create("path", {}, scene, std::unique_ptr<Camera>(camera), std::move(sampler),
                           integratorColorSpace, {}) :
        Integrator::Create(genScene.integrator.name, genScene.integrator.parameters,
                           scene, std::unique_ptr<Camera>(camera), std::move(sampler),
                           integratorColorSpace, &genScene.integrator.loc));

    // Helpful warnings
    if (haveScatteringMedia &&
        genScene.integrator.name != "volpath" &&
        genScene.integrator.name != "bdpt" &&
        genScene.integrator.name != "mlt")
        Warning("Scene has scattering media but \"%s\" integrator doesn't support "
                "volume scattering. Consider using \"volpath\", \"bdpt\", or "
                "\"mlt\".", genScene.integrator.name);

    if (scene.lights.empty() && genScene.integrator.name != "ambientocclusion" &&
        genScene.integrator.name != "aov")
        Warning("No light sources defined in scene; rendering a black image.");

    LOG_VERBOSE("Memory used after scene creation: %d", GetCurrentRSS());

    // This is kind of ugly; we directly override the current profiler
    // state to switch from parsing/scene construction related stuff to
    // rendering stuff and then switch it back below. The underlying
    // issue is that all the rest of the profiling system assumes
    // hierarchical inheritance of profiling state; this is the only
    // place where that isn't the case.
    if (PbrtOptions.profile) {
        CHECK_EQ(CurrentProfilerState(), ProfilePhaseToBits(ProfilePhase::SceneConstruction));
        ProfilerState = ProfilePhaseToBits(ProfilePhase::IntegratorRender);
    }

    // Render!
    integrator->Render();

    LOG_VERBOSE("Memory used after rendering: %s", GetCurrentRSS());

    if (PbrtOptions.profile) {
        CHECK_EQ(CurrentProfilerState(), ProfilePhaseToBits(ProfilePhase::IntegratorRender));
        ProfilerState = ProfilePhaseToBits(ProfilePhase::SceneConstruction);
    }

    ImageTextureBase::ClearCache();
    ShapeHandle::FreeBufferCaches();
}

} // namespace pbrt
