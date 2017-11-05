
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

// core/api.cpp*
#include <pbrt/core/api.h>

#include <pbrt/core/error.h>
#include <pbrt/util/parallel.h>
#include <pbrt/core/paramset.h>
#include <pbrt/core/spectrum.h>
#include <pbrt/core/scene.h>
#include <pbrt/core/film.h>
#include <pbrt/core/medium.h>
#include <pbrt/util/stats.h>
#include <pbrt/core/texcache.h>

// API Additional Headers
#include <pbrt/accelerators/bvh.h>
#include <pbrt/accelerators/kdtreeaccel.h>
#include <pbrt/cameras/environment.h>
#include <pbrt/cameras/orthographic.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/cameras/realistic.h>
#include <pbrt/filters/box.h>
#include <pbrt/filters/gaussian.h>
#include <pbrt/filters/mitchell.h>
#include <pbrt/filters/sinc.h>
#include <pbrt/filters/triangle.h>
#include <pbrt/integrators/aov.h>
#include <pbrt/integrators/bdpt.h>
#include <pbrt/integrators/mlt.h>
#include <pbrt/integrators/ao.h>
#include <pbrt/integrators/path.h>
#include <pbrt/integrators/sppm.h>
#include <pbrt/integrators/volpath.h>
#include <pbrt/integrators/whitted.h>
#include <pbrt/lights/diffuse.h>
#include <pbrt/lights/distant.h>
#include <pbrt/lights/goniometric.h>
#include <pbrt/lights/infinite.h>
#include <pbrt/lights/point.h>
#include <pbrt/lights/projection.h>
#include <pbrt/lights/spot.h>
#include <pbrt/materials/disney.h>
#include <pbrt/materials/fourier.h>
#include <pbrt/materials/glass.h>
#include <pbrt/materials/hair.h>
#include <pbrt/materials/kdsubsurface.h>
#include <pbrt/materials/matte.h>
#include <pbrt/materials/metal.h>
#include <pbrt/materials/mirror.h>
#include <pbrt/materials/mixmat.h>
#include <pbrt/materials/plastic.h>
#include <pbrt/materials/substrate.h>
#include <pbrt/materials/subsurface.h>
#include <pbrt/materials/translucent.h>
#include <pbrt/materials/uber.h>
#include <pbrt/samplers/halton.h>
#include <pbrt/samplers/maxmin.h>
#include <pbrt/samplers/random.h>
#include <pbrt/samplers/sobol.h>
#include <pbrt/samplers/stratified.h>
#include <pbrt/samplers/zerotwosequence.h>
#include <pbrt/shapes/cone.h>
#include <pbrt/shapes/curve.h>
#include <pbrt/shapes/cylinder.h>
#include <pbrt/shapes/disk.h>
#include <pbrt/shapes/hyperboloid.h>
#include <pbrt/shapes/loopsubdiv.h>
#include <pbrt/shapes/nurbs.h>
#include <pbrt/shapes/paraboloid.h>
#include <pbrt/shapes/sphere.h>
#include <pbrt/shapes/triangle.h>
#include <pbrt/shapes/plymesh.h>
#include <pbrt/textures/bilerp.h>
#include <pbrt/textures/checkerboard.h>
#include <pbrt/textures/constant.h>
#include <pbrt/textures/dots.h>
#include <pbrt/textures/fbm.h>
#include <pbrt/textures/imagemap.h>
#include <pbrt/textures/marble.h>
#include <pbrt/textures/mix.h>
#include <pbrt/textures/ptex.h>
#include <pbrt/textures/scale.h>
#include <pbrt/textures/uv.h>
#include <pbrt/textures/windy.h>
#include <pbrt/textures/wrinkled.h>
#include <pbrt/media/grid.h>
#include <pbrt/media/homogeneous.h>
#include <map>
#include <stdio.h>


namespace pbrt {

// API Global Variables
Options PbrtOptions;

// API Local Classes
constexpr int MaxTransforms = 2;
constexpr int StartTransformBits = 1 << 0;
constexpr int EndTransformBits = 1 << 1;
constexpr int AllTransformsBits = (1 << MaxTransforms) - 1;
struct TransformSet {
    // TransformSet Public Methods
    Transform &operator[](int i) {
        CHECK_GE(i, 0);
        CHECK_LT(i, MaxTransforms);
        return t[i];
    }
    const Transform &operator[](int i) const {
        CHECK_GE(i, 0);
        CHECK_LT(i, MaxTransforms);
        return t[i];
    }
    friend TransformSet Inverse(const TransformSet &ts) {
        TransformSet tInv;
        for (int i = 0; i < MaxTransforms; ++i) tInv.t[i] = Inverse(ts.t[i]);
        return tInv;
    }
    bool IsAnimated() const {
        for (int i = 0; i < MaxTransforms - 1; ++i)
            if (t[i] != t[i + 1]) return true;
        return false;
    }

  private:
    Transform t[MaxTransforms];
};

struct RenderOptions {
    void CreateSceneAndIntegrator(std::unique_ptr<Scene> *scene,
                                  std::unique_ptr<Integrator> *integrator);

    // RenderOptions Public Data
    Float transformStartTime = 0, transformEndTime = 1;
    std::string FilterName = "box";
    ParamSet FilterParams;
    std::string FilmName = "image";
    ParamSet FilmParams;
    std::string SamplerName = "halton";
    ParamSet SamplerParams;
    std::string AcceleratorName = "bvh";
    ParamSet AcceleratorParams;
    std::string IntegratorName = "path";
    ParamSet IntegratorParams;
    std::string CameraName = "perspective";
    ParamSet CameraParams;
    TransformSet CameraToWorld;
    std::map<std::string, std::shared_ptr<Medium>> namedMedia;
    std::vector<std::shared_ptr<Light>> lights;
    std::vector<std::shared_ptr<Primitive>> primitives;
    std::map<std::string, std::vector<std::shared_ptr<Primitive>>> instances;
    std::vector<std::shared_ptr<Primitive>> *currentInstance = nullptr;
    bool haveScatteringMedia = false;
};

// MaterialInstance represents both an instance of a material as well as
// the information required to create another instance of it (possibly with
// different parameters from the shape).
struct MaterialInstance {
    MaterialInstance() = default;
    MaterialInstance(const std::string &name, const std::shared_ptr<Material> &mtl,
                     ParamSet params)
        : name(name), material(mtl), params(std::move(params)) {}

    std::string name;
    std::shared_ptr<Material> material;
    ParamSet params;
};

struct GraphicsState {
    // Graphics State Methods
    GraphicsState() {
        ParamSet params;
        TextureParams tp({&params}, floatTextures, spectrumTextures);
        params.EnableLookupTracking();
        std::shared_ptr<Material> mtl(CreateMatteMaterial(tp, nullptr));
        currentMaterial = std::make_shared<MaterialInstance>("matte", mtl, params);

        shapeAttributes = std::make_shared<ParamSet>();
        lightAttributes = std::make_shared<ParamSet>();
        materialAttributes = std::make_shared<ParamSet>();
        mediumAttributes = std::make_shared<ParamSet>();
    }
    std::shared_ptr<Material> GetMaterialForShape(const ParamSet &geomParams);
    MediumInterface CreateMediumInterface();

    // Graphics State
    std::string currentInsideMedium, currentOutsideMedium;
    std::map<std::string, std::shared_ptr<Texture<Float>>> floatTextures;
    std::map<std::string, std::shared_ptr<Texture<Spectrum>>> spectrumTextures;
    std::map<std::string, std::shared_ptr<MaterialInstance>> namedMaterials;
    std::shared_ptr<MaterialInstance> currentMaterial;
    std::shared_ptr<ParamSet> areaLightParams;
    std::string areaLightName;
    bool reverseOrientation = false;
    std::shared_ptr<ParamSet> shapeAttributes, lightAttributes;
    std::shared_ptr<ParamSet> materialAttributes, mediumAttributes;
};

class TransformCache {
  public:
    // TransformCache Public Methods
    void Lookup(const Transform &t, std::shared_ptr<const Transform> *tCached,
                std::shared_ptr<const Transform> *tCachedInverse) {
        auto iter = cache.find(t);
        if (iter == cache.end()) {
            std::shared_ptr<const Transform> tr =
                std::make_shared<const Transform>(t);
            std::shared_ptr<const Transform> tinv =
                std::make_shared<const Transform>(Inverse(t));
            cache[t] = CacheItem{tr, tinv};
            iter = cache.find(t);
        }
        if (tCached) *tCached = iter->second.first;
        if (tCachedInverse) *tCachedInverse = iter->second.second;
    }
    void Clear() {
        cache.erase(cache.begin(), cache.end());
    }

  private:
    // TransformCache Private Data
    using CacheItem = std::pair<std::shared_ptr<const Transform>,
                                std::shared_ptr<const Transform>>;
    std::map<Transform, CacheItem> cache;
};

// API Static Data
enum class APIState { Uninitialized, OptionsBlock, WorldBlock };
static APIState currentApiState = APIState::Uninitialized;
static TransformSet curTransform;
static uint32_t activeTransformBits = AllTransformsBits;
static std::map<std::string, TransformSet> namedCoordinateSystems;
static std::unique_ptr<RenderOptions> renderOptions;
static GraphicsState graphicsState;
static std::vector<GraphicsState> pushedGraphicsStates;
static std::vector<TransformSet> pushedTransforms;
static std::vector<uint32_t> pushedActiveTransformBits;
static TransformCache transformCache;
int catIndentCount = 0;

// API Macros
#define VERIFY_INITIALIZED(func)                           \
    if (!(PbrtOptions.cat || PbrtOptions.toPly) &&           \
        currentApiState == APIState::Uninitialized) {        \
        Error(                                             \
            "pbrtInit() must be before calling \"%s()\". " \
            "Ignoring.",                                   \
            func);                                         \
        return;                                            \
    } else /* swallow trailing semicolon */
#define VERIFY_OPTIONS(func)                             \
    VERIFY_INITIALIZED(func);                            \
    if (!(PbrtOptions.cat || PbrtOptions.toPly) &&       \
        currentApiState == APIState::WorldBlock) {       \
        Error(                                           \
            "Options cannot be set inside world block; " \
            "\"%s\" not allowed.  Ignoring.",            \
            func);                                       \
        return;                                          \
    } else /* swallow trailing semicolon */
#define VERIFY_WORLD(func)                                   \
    VERIFY_INITIALIZED(func);                                \
    if (!(PbrtOptions.cat || PbrtOptions.toPly) &&           \
        currentApiState == APIState::OptionsBlock) {         \
        Error(                                               \
            "Scene description must be inside world block; " \
            "\"%s\" not allowed. Ignoring.",                 \
            func);                                           \
        return;                                              \
    } else /* swallow trailing semicolon */
#define FOR_ACTIVE_TRANSFORMS(expr)           \
    for (int i = 0; i < MaxTransforms; ++i)   \
        if (activeTransformBits & (1 << i)) { \
            expr                              \
        }
#define WARN_IF_ANIMATED_TRANSFORM(func)                             \
    do {                                                             \
        if (curTransform.IsAnimated())                               \
            Warning(                                                 \
                "Animated transformations set; ignoring for \"%s\" " \
                "and using the start transform only",                \
                func);                                               \
    } while (false) /* swallow trailing semicolon */

// Object Creation Function Definitions
static std::vector<std::shared_ptr<Shape>> MakeShapes(
        const std::string &name, std::shared_ptr<const Transform> object2world,
        std::shared_ptr<const Transform> world2object, bool reverseOrientation,
        const ParamSet &paramSet) {
    std::vector<std::shared_ptr<Shape>> shapes;
    std::shared_ptr<Shape> s;
    if (name == "sphere")
        s = CreateSphereShape(object2world, world2object, reverseOrientation,
                              paramSet, graphicsState.shapeAttributes);
    // Create remaining single _Shape_ types
    else if (name == "cylinder")
        s = CreateCylinderShape(object2world, world2object, reverseOrientation,
                                paramSet, graphicsState.shapeAttributes);
    else if (name == "disk")
        s = CreateDiskShape(object2world, world2object, reverseOrientation,
                            paramSet, graphicsState.shapeAttributes);
    else if (name == "cone")
        s = CreateConeShape(object2world, world2object, reverseOrientation,
                            paramSet, graphicsState.shapeAttributes);
    else if (name == "paraboloid")
        s = CreateParaboloidShape(object2world, world2object,
                                  reverseOrientation, paramSet,
                                  graphicsState.shapeAttributes);
    else if (name == "hyperboloid")
        s = CreateHyperboloidShape(object2world, world2object,
                                   reverseOrientation, paramSet,
                                   graphicsState.shapeAttributes);
    if (s != nullptr) shapes.push_back(s);

    // Create multiple-_Shape_ types
    else if (name == "curve")
        shapes = CreateCurveShape(object2world, world2object, reverseOrientation,
                                  paramSet, graphicsState.shapeAttributes);
    else if (name == "trianglemesh") {
        if (PbrtOptions.toPly) {
            static int count = 1;
            const char *plyPrefix =
                getenv("PLY_PREFIX") ? getenv("PLY_PREFIX") : "mesh";
            std::string fn = StringPrintf("%s_%05d.ply", plyPrefix, count++);

            absl::Span<const int> vi = paramSet.GetIntArray("indices");
            absl::Span<const Point3f> P = paramSet.GetPoint3fArray("P");
            absl::Span<const Point2f> uvs = paramSet.GetPoint2fArray("uv");
            if (uvs.empty()) uvs = paramSet.GetPoint2fArray("st");
            std::vector<Point2f> tempUVs;
            if (uvs.empty()) {
                absl::Span<const Float> fuv = paramSet.GetFloatArray("uv");
                if (fuv.empty()) fuv = paramSet.GetFloatArray("st");
                if (!fuv.empty()) {
                    tempUVs.reserve(fuv.size() / 2);
                    for (size_t i = 0; i < fuv.size() / 2; ++i)
                        tempUVs.push_back(Point2f(fuv[2 * i], fuv[2 * i + 1]));
                    uvs = tempUVs;
                }
            }
            absl::Span<const Normal3f> N = paramSet.GetNormal3fArray("N");
            absl::Span<const Vector3f> S = paramSet.GetVector3fArray("S");
            // TODO: check that if non-empty, N and S are at least as big
            // as P.

            if (!WritePlyFile(fn, vi, P, S, N, uvs))
                Error("Unable to write PLY file \"%s\"", fn.c_str());

            printf("%*sShape \"plymesh\" \"string filename\" \"%s\" ",
                   catIndentCount, "", fn.c_str());

            std::string alphaTex = paramSet.FindTexture("alpha");
            if (alphaTex != "")
                printf("\n%*s\"texture alpha\" \"%s\" ", catIndentCount + 8, "",
                       alphaTex.c_str());
            else {
                absl::Span<const Float> alpha = paramSet.GetFloatArray("alpha");
                if (!alpha.empty())
                    printf("\n%*s\"float alpha\" %f ", catIndentCount + 8, "",
                           alpha[0]);
            }

            std::string shadowAlphaTex = paramSet.FindTexture("shadowalpha");
            if (shadowAlphaTex != "")
                printf("\n%*s\"texture shadowalpha\" \"%s\" ",
                       catIndentCount + 8, "", shadowAlphaTex.c_str());
            else {
                absl::Span<const Float> alpha = paramSet.GetFloatArray("shadowalpha");
                if (!alpha.empty())
                    printf("\n%*s\"float shadowalpha\" %f ", catIndentCount + 8,
                           "", alpha[0]);
            }
            printf("\n");
        } else
            shapes = CreateTriangleMeshShape(object2world, world2object,
                                             reverseOrientation, paramSet,
                                             graphicsState.shapeAttributes);
    } else if (name == "plymesh")
        shapes = CreatePLYMesh(object2world, world2object, reverseOrientation,
                               paramSet, graphicsState.shapeAttributes);
    else if (name == "loopsubdiv")
        shapes = CreateLoopSubdiv(object2world, world2object,
                                  reverseOrientation, paramSet, graphicsState.shapeAttributes);
    else if (name == "nurbs")
        shapes = CreateNURBS(object2world, world2object, reverseOrientation,
                             paramSet, graphicsState.shapeAttributes);
    else
        ErrorExit("Shape \"%s\" unknown.", name.c_str());

    return shapes;
}

STAT_COUNTER("Scene/Materials created", nMaterialsCreated);

static std::shared_ptr<Material> MakeMaterial(const std::string &name,
                                              const TextureParams &mp) {
    std::shared_ptr<Material> material;
    if (name == "" || name == "none")
        return nullptr;
    else if (name == "matte")
        material = CreateMatteMaterial(mp, graphicsState.materialAttributes);
    else if (name == "plastic")
        material = CreatePlasticMaterial(mp, graphicsState.materialAttributes);
    else if (name == "translucent")
        material = CreateTranslucentMaterial(mp, graphicsState.materialAttributes);
    else if (name == "glass")
        material = CreateGlassMaterial(mp, graphicsState.materialAttributes);
    else if (name == "mirror")
        material = CreateMirrorMaterial(mp, graphicsState.materialAttributes);
    else if (name == "hair")
        material = CreateHairMaterial(mp, graphicsState.materialAttributes);
    else if (name == "disney")
        material = CreateDisneyMaterial(mp, graphicsState.materialAttributes);
    else if (name == "mix") {
        std::string m1 = mp.GetOneString("namedmaterial1", "");
        std::string m2 = mp.GetOneString("namedmaterial2", "");
        std::shared_ptr<Material> mat1, mat2;
        if (graphicsState.namedMaterials.find(m1) ==
            graphicsState.namedMaterials.end()) {
            Error("Named material \"%s\" undefined.  Using \"matte\"",
                  m1.c_str());
            mat1 = MakeMaterial("matte", mp);
        } else
            mat1 = graphicsState.namedMaterials[m1]->material;

        if (graphicsState.namedMaterials.find(m2) ==
            graphicsState.namedMaterials.end()) {
            Error("Named material \"%s\" undefined.  Using \"matte\"",
                  m2.c_str());
            mat2 = MakeMaterial("matte", mp);
        } else
            mat2 = graphicsState.namedMaterials[m2]->material;
        material = CreateMixMaterial(mp, mat1, mat2, graphicsState.materialAttributes);
    } else if (name == "metal")
        material = CreateMetalMaterial(mp, graphicsState.materialAttributes);
    else if (name == "substrate")
        material = CreateSubstrateMaterial(mp, graphicsState.materialAttributes);
    else if (name == "uber")
        material = CreateUberMaterial(mp, graphicsState.materialAttributes);
    else if (name == "subsurface")
        material = CreateSubsurfaceMaterial(mp, graphicsState.materialAttributes);
    else if (name == "kdsubsurface")
        material = CreateKdSubsurfaceMaterial(mp, graphicsState.materialAttributes);
    else if (name == "fourier")
        material = CreateFourierMaterial(mp, graphicsState.materialAttributes);
    else
        ErrorExit("Material \"%s\" unknown.", name.c_str());

    if ((name == "subsurface" || name == "kdsubsurface") &&
        (renderOptions->IntegratorName != "path" &&
         (renderOptions->IntegratorName != "volpath")))
        Warning(
            "Subsurface scattering material \"%s\" used, but \"%s\" "
            "integrator doesn't support subsurface scattering. "
            "Use \"path\" or \"volpath\".",
            name.c_str(), renderOptions->IntegratorName.c_str());

    mp.ReportUnused();
    CHECK(material);
    ++nMaterialsCreated;
    return material;
}

static std::shared_ptr<Texture<Float>> MakeFloatTexture(
        const std::string &name, const Transform &tex2world,
        const TextureParams &tp) {
    std::shared_ptr<Texture<Float>> tex;
    if (name == "constant")
        tex = CreateConstantFloatTexture(tex2world, tp);
    else if (name == "scale")
        tex = CreateScaleFloatTexture(tex2world, tp);
    else if (name == "mix")
        tex = CreateMixFloatTexture(tex2world, tp);
    else if (name == "bilerp")
        tex = CreateBilerpFloatTexture(tex2world, tp);
    else if (name == "imagemap")
        tex = CreateImageFloatTexture(tex2world, tp);
    else if (name == "uv")
        tex = CreateUVFloatTexture(tex2world, tp);
    else if (name == "checkerboard")
        tex = CreateCheckerboardFloatTexture(tex2world, tp);
    else if (name == "dots")
        tex = CreateDotsFloatTexture(tex2world, tp);
    else if (name == "fbm")
        tex = CreateFBmFloatTexture(tex2world, tp);
    else if (name == "wrinkled")
        tex = CreateWrinkledFloatTexture(tex2world, tp);
    else if (name == "marble")
        tex = CreateMarbleFloatTexture(tex2world, tp);
    else if (name == "windy")
        tex = CreateWindyFloatTexture(tex2world, tp);
    else if (name == "ptex")
        tex = CreatePtexFloatTexture(tex2world, tp);
    else
        Warning("Float texture \"%s\" unknown.", name.c_str());

    tp.ReportUnused();
    return tex;
}

static std::shared_ptr<Texture<Spectrum>> MakeSpectrumTexture(
        const std::string &name, const Transform &tex2world,
        const TextureParams &tp) {
    std::shared_ptr<Texture<Spectrum>> tex;
    if (name == "constant")
        tex = CreateConstantSpectrumTexture(tex2world, tp);
    else if (name == "scale")
        tex = CreateScaleSpectrumTexture(tex2world, tp);
    else if (name == "mix")
        tex = CreateMixSpectrumTexture(tex2world, tp);
    else if (name == "bilerp")
        tex = CreateBilerpSpectrumTexture(tex2world, tp);
    else if (name == "imagemap")
        tex = CreateImageSpectrumTexture(tex2world, tp);
    else if (name == "uv")
        tex = CreateUVSpectrumTexture(tex2world, tp);
    else if (name == "checkerboard")
        tex = CreateCheckerboardSpectrumTexture(tex2world, tp);
    else if (name == "dots")
        tex = CreateDotsSpectrumTexture(tex2world, tp);
    else if (name == "fbm")
        tex = CreateFBmSpectrumTexture(tex2world, tp);
    else if (name == "wrinkled")
        tex = CreateWrinkledSpectrumTexture(tex2world, tp);
    else if (name == "marble")
        tex = CreateMarbleSpectrumTexture(tex2world, tp);
    else if (name == "windy")
        tex = CreateWindySpectrumTexture(tex2world, tp);
    else if (name == "ptex")
        tex = CreatePtexSpectrumTexture(tex2world, tp);
    else
        Warning("Spectrum texture \"%s\" unknown.", name.c_str());

    tp.ReportUnused();
    return tex;
}

static std::shared_ptr<Medium> MakeMedium(const std::string &name,
                                          const ParamSet &paramSet,
                                          const Transform &medium2world) {
    std::shared_ptr<Medium> m;
    if (name == "homogeneous")
        m = HomogeneousMedium::Create(paramSet, graphicsState.mediumAttributes);
    else if (name == "heterogeneous")
        m = GridDensityMedium::Create(paramSet, medium2world,
                                      graphicsState.mediumAttributes);
    else
        ErrorExit("Medium \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return m;
}

static std::shared_ptr<Light> MakeLight(const std::string &name,
                                        const ParamSet &paramSet,
                                        const Transform &light2world,
                                        const MediumInterface &mediumInterface) {
    std::shared_ptr<Light> light;
    if (name == "point")
        light =
            CreatePointLight(light2world, mediumInterface.outside, paramSet,
                             graphicsState.lightAttributes);
    else if (name == "spot")
        light = CreateSpotLight(light2world, mediumInterface.outside, paramSet,
                                graphicsState.lightAttributes);
    else if (name == "goniometric")
        light = CreateGoniometricLight(light2world, mediumInterface.outside,
                                       paramSet, graphicsState.lightAttributes);
    else if (name == "projection")
        light = CreateProjectionLight(light2world, mediumInterface.outside,
                                      paramSet, graphicsState.lightAttributes);
    else if (name == "distant")
        light = CreateDistantLight(light2world, paramSet,
                                   graphicsState.lightAttributes);
    else if (name == "infinite")
        light = CreateInfiniteLight(light2world, paramSet,
                                    graphicsState.lightAttributes);
    else
        ErrorExit("Light \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return light;
}

static std::shared_ptr<AreaLight> MakeAreaLight(
    const std::string &name, const Transform &light2world,
    const MediumInterface &mediumInterface, const ParamSet &paramSet,
    const std::shared_ptr<Shape> &shape) {
    std::shared_ptr<AreaLight> area;
    if (name == "diffuse")
        area = CreateDiffuseAreaLight(light2world, mediumInterface.outside,
                                      paramSet, shape, graphicsState.lightAttributes);
    else
        ErrorExit("Area light \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return area;
}

static std::shared_ptr<Primitive> MakeAccelerator(
    const std::string &name, const std::vector<std::shared_ptr<Primitive>> &prims,
    const ParamSet &paramSet) {
    std::shared_ptr<Primitive> accel;
    if (name == "bvh")
        accel = CreateBVHAccelerator(prims, paramSet);
    else if (name == "kdtree")
        accel = CreateKdTreeAccelerator(prims, paramSet);
    else
        ErrorExit("Accelerator \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return accel;
}

static std::shared_ptr<Camera> MakeCamera(
    const std::string &name, const ParamSet &paramSet,
    const TransformSet &cam2worldSet, Float transformStart, Float transformEnd,
    const Scene &scene, std::unique_ptr<Film> film) {
    std::shared_ptr<Camera> camera;
    MediumInterface mediumInterface = graphicsState.CreateMediumInterface();
    static_assert(MaxTransforms == 2,
                  "TransformCache assumes only two transforms");
    std::shared_ptr<const Transform> cam2world[2];
    transformCache.Lookup(cam2worldSet[0], &cam2world[0], nullptr);
    transformCache.Lookup(cam2worldSet[1], &cam2world[1], nullptr);
    AnimatedTransform animatedCam2World(cam2world[0], transformStart,
                                        cam2world[1], transformEnd);
    if (name == "perspective")
        camera = CreatePerspectiveCamera(paramSet, animatedCam2World, std::move(film),
                                         mediumInterface.outside, scene);
    else if (name == "orthographic")
        camera = CreateOrthographicCamera(paramSet, animatedCam2World, std::move(film),
                                          mediumInterface.outside, scene);
    else if (name == "realistic")
        camera = CreateRealisticCamera(paramSet, animatedCam2World, std::move(film),
                                       mediumInterface.outside, scene);
    else if (name == "environment")
        camera = CreateEnvironmentCamera(paramSet, animatedCam2World, std::move(film),
                                         mediumInterface.outside, scene);
    else
        ErrorExit("Camera \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return camera;
}

static std::unique_ptr<Sampler> MakeSampler(const std::string &name,
                                            const ParamSet &paramSet,
                                            const Bounds2i &sampleBounds) {
    std::unique_ptr<Sampler> sampler;
    if (name == "02sequence")
        sampler = CreateZeroTwoSequenceSampler(paramSet);
    else if (name == "maxmindist")
        sampler = CreateMaxMinDistSampler(paramSet);
    else if (name == "halton")
        sampler = CreateHaltonSampler(paramSet, sampleBounds);
    else if (name == "sobol")
        sampler = CreateSobolSampler(paramSet, sampleBounds);
    else if (name == "random")
        sampler = CreateRandomSampler(paramSet);
    else if (name == "stratified")
        sampler = CreateStratifiedSampler(paramSet);
    else
        ErrorExit("Sampler \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return sampler;
}

static std::unique_ptr<Filter> MakeFilter(const std::string &name,
                                          const ParamSet &paramSet) {
    std::unique_ptr<Filter> filter;
    if (name == "box")
        filter = CreateBoxFilter(paramSet);
    else if (name == "gaussian")
        filter = CreateGaussianFilter(paramSet);
    else if (name == "mitchell")
        filter = CreateMitchellFilter(paramSet);
    else if (name == "sinc")
        filter = CreateSincFilter(paramSet);
    else if (name == "triangle")
        filter = CreateTriangleFilter(paramSet);
    else
        ErrorExit("Filter \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return filter;
}

static std::unique_ptr<Film> MakeFilm(const std::string &name, const ParamSet &paramSet,
                                      std::unique_ptr<Filter> filter) {
    std::unique_ptr<Film> film;
    if (name == "image")
        film = CreateFilm(paramSet, std::move(filter));
    else
        ErrorExit("Film \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return film;
}

static std::unique_ptr<Integrator> MakeIntegrator(const std::string &name,
                                                  const ParamSet &paramSet,
                                                  const Scene &scene,
                                                  std::shared_ptr<Camera> camera,
                                                  std::unique_ptr<Sampler> sampler) {
    std::unique_ptr<Integrator> integrator;
    if (name == "whitted")
        integrator = CreateWhittedIntegrator(paramSet, scene, camera, std::move(sampler));
    else if (name == "path")
        integrator = CreatePathIntegrator(paramSet, scene, camera, std::move(sampler));
    else if (name == "volpath")
        integrator = CreateVolPathIntegrator(paramSet, scene, camera, std::move(sampler));
    else if (name == "bdpt")
        integrator = CreateBDPTIntegrator(paramSet, scene, camera, std::move(sampler));
    else if (name == "aov")
        integrator = CreateAOVIntegrator(paramSet, scene, camera);
    else if (name == "mlt")
        integrator = CreateMLTIntegrator(paramSet, scene, camera);
    else if (name == "ambientocclusion")
        integrator = CreateAOIntegrator(paramSet, scene, camera, std::move(sampler));
    else if (name == "sppm")
        integrator = CreateSPPMIntegrator(paramSet, scene, camera);
    else
        ErrorExit("Integrator \"%s\" unknown.", name.c_str());

    paramSet.ReportUnused();
    return integrator;
}

// API Function Definitions
void pbrtInit(const Options &opt) {
    PbrtOptions = opt;
    // API Initialization
    if (currentApiState != APIState::Uninitialized)
        Error("pbrtInit() has already been called.");
    currentApiState = APIState::OptionsBlock;
    renderOptions = std::make_unique<RenderOptions>();
    graphicsState = GraphicsState();
    catIndentCount = 0;

    // General \pbrt Initialization
    SampledSpectrum::Init();
    ParallelInit();  // Threads must be launched before the profiler is
                     // initialized.
    InitProfiler();
}

void pbrtCleanup() {
    // API Cleanup
    if (currentApiState == APIState::Uninitialized)
        Error("pbrtCleanup() called without pbrtInit().");
    else if (currentApiState == APIState::WorldBlock)
        Error("pbrtCleanup() called while inside world block.");
    currentApiState = APIState::Uninitialized;
    ParallelCleanup();
    renderOptions = nullptr;
    CleanupProfiler();
}

void pbrtIdentity() {
    VERIFY_INITIALIZED("Identity");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = Transform();)
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sIdentity\n", catIndentCount, "");
}

void pbrtTranslate(Float dx, Float dy, Float dz) {
    VERIFY_INITIALIZED("Translate");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = curTransform[i] *
                                            Translate(Vector3f(dx, dy, dz));)
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sTranslate %.9g %.9g %.9g\n", catIndentCount, "", dx, dy,
               dz);
}

void pbrtTransform(Float tr[16]) {
    VERIFY_INITIALIZED("Transform");
    FOR_ACTIVE_TRANSFORMS(
        curTransform[i] = Transform(Matrix4x4(
            tr[0], tr[4], tr[8], tr[12], tr[1], tr[5], tr[9], tr[13], tr[2],
            tr[6], tr[10], tr[14], tr[3], tr[7], tr[11], tr[15]));)
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sTransform [ ", catIndentCount, "");
        for (int i = 0; i < 16; ++i) printf("%.9g ", tr[i]);
        printf(" ]\n");
    }
}

void pbrtConcatTransform(Float tr[16]) {
    VERIFY_INITIALIZED("ConcatTransform");
    FOR_ACTIVE_TRANSFORMS(
        curTransform[i] =
            curTransform[i] *
            Transform(Matrix4x4(tr[0], tr[4], tr[8], tr[12], tr[1], tr[5],
                                tr[9], tr[13], tr[2], tr[6], tr[10], tr[14],
                                tr[3], tr[7], tr[11], tr[15]));)
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sConcatTransform [ ", catIndentCount, "");
        for (int i = 0; i < 16; ++i) printf("%.9g ", tr[i]);
        printf(" ]\n");
    }
}

void pbrtRotate(Float angle, Float dx, Float dy, Float dz) {
    VERIFY_INITIALIZED("Rotate");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] =
                              curTransform[i] *
                              Rotate(angle, Vector3f(dx, dy, dz));)
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sRotate %.9g %.9g %.9g %.9g\n", catIndentCount, "", angle,
               dx, dy, dz);
}

void pbrtScale(Float sx, Float sy, Float sz) {
    VERIFY_INITIALIZED("Scale");
    FOR_ACTIVE_TRANSFORMS(curTransform[i] =
                              curTransform[i] * Scale(sx, sy, sz);)
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sScale %.9g %.9g %.9g\n", catIndentCount, "", sx, sy, sz);
}

void pbrtLookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                Float ux, Float uy, Float uz) {
    VERIFY_INITIALIZED("LookAt");
    Transform lookAt =
        LookAt(Point3f(ex, ey, ez), Point3f(lx, ly, lz), Vector3f(ux, uy, uz));
    FOR_ACTIVE_TRANSFORMS(curTransform[i] = curTransform[i] * lookAt;);
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf(
            "%*sLookAt %.9g %.9g %.9g\n%*s%.9g %.9g %.9g\n"
            "%*s%.9g %.9g %.9g\n",
            catIndentCount, "", ex, ey, ez, catIndentCount + 8, "", lx, ly, lz,
            catIndentCount + 8, "", ux, uy, uz);
}

void pbrtCoordinateSystem(const std::string &name) {
    VERIFY_INITIALIZED("CoordinateSystem");
    namedCoordinateSystems[name] = curTransform;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sCoordinateSystem \"%s\"\n", catIndentCount, "",
               name.c_str());
}

void pbrtCoordSysTransform(const std::string &name) {
    VERIFY_INITIALIZED("CoordSysTransform");
    if (namedCoordinateSystems.find(name) != namedCoordinateSystems.end())
        curTransform = namedCoordinateSystems[name];
    else
        Warning("Couldn't find named coordinate system \"%s\"", name.c_str());
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sCoordSysTransform \"%s\"\n", catIndentCount, "",
               name.c_str());
}

void pbrtActiveTransformAll() {
    activeTransformBits = AllTransformsBits;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sActiveTransform All\n", catIndentCount, "");
}

void pbrtActiveTransformEndTime() {
    activeTransformBits = EndTransformBits;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sActiveTransform EndTime\n", catIndentCount, "");
}

void pbrtActiveTransformStartTime() {
    activeTransformBits = StartTransformBits;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sActiveTransform StartTime\n", catIndentCount, "");
}

void pbrtTransformTimes(Float start, Float end) {
    VERIFY_OPTIONS("TransformTimes");
    renderOptions->transformStartTime = start;
    renderOptions->transformEndTime = end;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sTransformTimes %.9g %.9g\n", catIndentCount, "", start,
               end);
}

void pbrtPixelFilter(const std::string &name, ParamSet params) {
    VERIFY_OPTIONS("PixelFilter");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sPixelFilter \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    renderOptions->FilterName = name;
    renderOptions->FilterParams = std::move(params);
}

void pbrtFilm(const std::string &type, ParamSet params) {
    VERIFY_OPTIONS("Film");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sFilm \"%s\" ", catIndentCount, "", type.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    renderOptions->FilmName = type;
    renderOptions->FilmParams = std::move(params);
}

void pbrtSampler(const std::string &name, ParamSet params) {
    VERIFY_OPTIONS("Sampler");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sSampler \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    renderOptions->SamplerName = name;
    renderOptions->SamplerParams = std::move(params);
}

void pbrtAccelerator(const std::string &name, ParamSet params) {
    VERIFY_OPTIONS("Accelerator");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sAccelerator \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    renderOptions->AcceleratorName = name;
    renderOptions->AcceleratorParams = std::move(params);
}

void pbrtIntegrator(const std::string &name, ParamSet params) {
    VERIFY_OPTIONS("Integrator");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sIntegrator \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    renderOptions->IntegratorName = name;
    renderOptions->IntegratorParams = std::move(params);
}

void pbrtCamera(const std::string &name, ParamSet params) {
    VERIFY_OPTIONS("Camera");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sCamera \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    renderOptions->CameraName = name;
    renderOptions->CameraParams = std::move(params);
    renderOptions->CameraToWorld = Inverse(curTransform);
    namedCoordinateSystems["camera"] = renderOptions->CameraToWorld;
}

void pbrtMakeNamedMedium(const std::string &name, ParamSet params) {
    VERIFY_INITIALIZED("MakeNamedMedium");
    WARN_IF_ANIMATED_TRANSFORM("MakeNamedMedium");
    std::string type = params.GetOneString("type", "");
    if (type == "")
        Error("No parameter string \"type\" found in MakeNamedMedium");
    else {
        std::shared_ptr<Medium> medium =
            MakeMedium(type, params, curTransform[0]);
        if (medium) renderOptions->namedMedia[name] = medium;
    }
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sMakeNamedMedium \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
}

void pbrtMediumInterface(const std::string &insideName,
                         const std::string &outsideName) {
    VERIFY_INITIALIZED("MediumInterface");
    graphicsState.currentInsideMedium = insideName;
    graphicsState.currentOutsideMedium = outsideName;
    renderOptions->haveScatteringMedia = true;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sMediumInterface \"%s\" \"%s\"\n", catIndentCount, "",
               insideName.c_str(), outsideName.c_str());
}

void pbrtWorldBegin() {
    VERIFY_OPTIONS("WorldBegin");
    currentApiState = APIState::WorldBlock;
    for (int i = 0; i < MaxTransforms; ++i) curTransform[i] = Transform();
    activeTransformBits = AllTransformsBits;
    namedCoordinateSystems["world"] = curTransform;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("\n\nWorldBegin\n\n");
}

void pbrtAttributeBegin() {
    VERIFY_WORLD("AttributeBegin");
    pushedGraphicsStates.push_back(graphicsState);
    pushedTransforms.push_back(curTransform);
    pushedActiveTransformBits.push_back(activeTransformBits);
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("\n%*sAttributeBegin\n", catIndentCount, "");
        catIndentCount += 4;
    }
}

void pbrtAttributeEnd() {
    VERIFY_WORLD("AttributeEnd");
    if (!pushedGraphicsStates.size()) {
        Error("Unmatched AttributeEnd encountered. Ignoring it.");
        return;
    }

    graphicsState = std::move(pushedGraphicsStates.back());
    pushedGraphicsStates.pop_back();
    curTransform = pushedTransforms.back();
    pushedTransforms.pop_back();
    activeTransformBits = pushedActiveTransformBits.back();
    pushedActiveTransformBits.pop_back();
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        catIndentCount -= 4;
        printf("%*sAttributeEnd\n", catIndentCount, "");
    }
}

void pbrtAttribute(const std::string &target, const NamedValues &attrib) {
    VERIFY_INITIALIZED("Attribute");
    CHECK(attrib.next == nullptr);

    std::shared_ptr<ParamSet> *attributes = nullptr;
    if (target == "shape")
        attributes = &graphicsState.shapeAttributes;
    else if (target == "light")
        attributes = &graphicsState.lightAttributes;
    else if (target == "material")
        attributes = &graphicsState.materialAttributes;
    else if (target == "medium")
        attributes = &graphicsState.mediumAttributes;
    else {
        Error("Unknown attribute target \"%s\". Must be \"shape\", \"light\", "
              "\"material\", or \"medium\".", target.c_str());
        return;
    }
    if (attributes->use_count() > 1)
        *attributes = std::make_shared<ParamSet>(**attributes);
    (*attributes)->Parse(&attrib, SpectrumType::Reflectance);

    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sAttribute \"%s\" ", catIndentCount, "", target.c_str());
        printf("%*s%s\n", catIndentCount, "", attrib.ToString().c_str());
    }
}

void pbrtTransformBegin() {
    VERIFY_WORLD("TransformBegin");
    pushedTransforms.push_back(curTransform);
    pushedActiveTransformBits.push_back(activeTransformBits);
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sTransformBegin\n", catIndentCount, "");
        catIndentCount += 4;
    }
}

void pbrtTransformEnd() {
    VERIFY_WORLD("TransformEnd");
    if (!pushedTransforms.size()) {
        Error("Unmatched TransformEnd encountered. Ignoring it.");
        return;
    }
    curTransform = pushedTransforms.back();
    pushedTransforms.pop_back();
    activeTransformBits = pushedActiveTransformBits.back();
    pushedActiveTransformBits.pop_back();
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        catIndentCount -= 4;
        printf("%*sTransformEnd\n", catIndentCount, "");
    }
}

void pbrtTexture(const std::string &name, const std::string &type,
                 const std::string &texname, ParamSet params) {
    VERIFY_WORLD("Texture");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sTexture \"%s\" \"%s\" \"%s\" ", catIndentCount, "",
               name.c_str(), type.c_str(), texname.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }

    TextureParams tp({&params}, graphicsState.floatTextures,
                     graphicsState.spectrumTextures);
    if (type == "float") {
        // Create _Float_ texture and store in _floatTextures_
        if (graphicsState.floatTextures.find(name) !=
            graphicsState.floatTextures.end())
            Warning("Texture \"%s\" being redefined", name.c_str());
        WARN_IF_ANIMATED_TRANSFORM("Texture");
        std::shared_ptr<Texture<Float>> ft =
            MakeFloatTexture(texname, curTransform[0], tp);
        if (ft) graphicsState.floatTextures[name] = ft;
    } else if (type == "color" || type == "spectrum") {
        // Create _color_ texture and store in _spectrumTextures_
        if (graphicsState.spectrumTextures.find(name) !=
            graphicsState.spectrumTextures.end())
            Warning("Texture \"%s\" being redefined", name.c_str());
        WARN_IF_ANIMATED_TRANSFORM("Texture");
        std::shared_ptr<Texture<Spectrum>> st =
            MakeSpectrumTexture(texname, curTransform[0], tp);
        if (st) graphicsState.spectrumTextures[name] = st;
    } else
        ErrorExit("Texture type \"%s\" unknown.", type.c_str());
}

void pbrtMaterial(const std::string &name, ParamSet params) {
    VERIFY_WORLD("Material");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sMaterial \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    TextureParams tp({&params}, graphicsState.floatTextures,
                     graphicsState.spectrumTextures);
    params.EnableLookupTracking();
    std::shared_ptr<Material> mtl = MakeMaterial(name, tp);
    graphicsState.currentMaterial =
        std::make_shared<MaterialInstance>(name, mtl, std::move(params));
}

void pbrtMakeNamedMaterial(const std::string &name, ParamSet params) {
    VERIFY_WORLD("MakeNamedMaterial");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sMakeNamedMaterial \"%s\" ", catIndentCount, "",
               name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    } else {
        // error checking, warning if replace, what to use for transform?
        std::string matName = params.GetOneString("type", "");
        WARN_IF_ANIMATED_TRANSFORM("MakeNamedMaterial");
        if (matName == "")
            Error("No parameter string \"type\" found in MakeNamedMaterial");

        bool setNameAttribute = false;
        if (graphicsState.materialAttributes->GetOneString("name", "") == "") {
            // The user hasn't explicitly specified a name via
            // 'Attribute "material" "string name" "..."', so set it based
            // on the given mmaterial name;
            setNameAttribute = true;
            pbrtAttributeBegin();

            std::string decl = "string name";
            NamedValues nv;
            nv.name = &decl;
            nv.strings.push_back(&name);
            pbrtAttribute("material", nv);
        }

        TextureParams mp({&params}, graphicsState.floatTextures,
                         graphicsState.spectrumTextures);
        params.EnableLookupTracking();
        std::shared_ptr<Material> mtl = MakeMaterial(matName, mp);

        if (setNameAttribute)
            pbrtAttributeEnd();

        if (graphicsState.namedMaterials.find(name) !=
            graphicsState.namedMaterials.end())
            Warning("Named material \"%s\" redefined.", name.c_str());

        graphicsState.namedMaterials[name] =
            std::make_shared<MaterialInstance>(matName, mtl, std::move(params));
    }
}

void pbrtNamedMaterial(const std::string &name) {
    VERIFY_WORLD("NamedMaterial");
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sNamedMaterial \"%s\"\n", catIndentCount, "", name.c_str());
    else {
        auto iter = graphicsState.namedMaterials.find(name);
        if (iter == graphicsState.namedMaterials.end()) {
            Error("NamedMaterial \"%s\" unknown.", name.c_str());
            return;
        }
        graphicsState.currentMaterial = iter->second;
    }
}

void pbrtLightSource(const std::string &name, ParamSet params) {
    VERIFY_WORLD("LightSource");
    WARN_IF_ANIMATED_TRANSFORM("LightSource");
    MediumInterface mi = graphicsState.CreateMediumInterface();
    std::shared_ptr<Light> lt = MakeLight(name, params, curTransform[0], mi);
    renderOptions->lights.push_back(lt);

    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sLightSource \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
}

void pbrtAreaLightSource(const std::string &name, ParamSet params) {
    VERIFY_WORLD("AreaLightSource");
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sAreaLightSource \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }
    graphicsState.areaLightName = name;
    graphicsState.areaLightParams = std::make_shared<ParamSet>(std::move(params));
}

void pbrtShape(const std::string &name, ParamSet params) {
    VERIFY_WORLD("Shape");
    std::vector<std::shared_ptr<Primitive>> prims;
    std::vector<std::shared_ptr<AreaLight>> areaLights;
    if (PbrtOptions.cat || (PbrtOptions.toPly && name != "trianglemesh")) {
        printf("%*sShape \"%s\" ", catIndentCount, "", name.c_str());
        printf("%s\n", params.ToString(catIndentCount).c_str());
    }

    std::shared_ptr<Texture<Float>> alphaTex;
    std::string alphaTexName = params.FindTexture("alpha");
    if (alphaTexName != "") {
        if (graphicsState.floatTextures.find(alphaTexName) != graphicsState.floatTextures.end())
            alphaTex = graphicsState.floatTextures[alphaTexName];
        else
            Error("Couldn't find float texture \"%s\" for \"alpha\" parameter",
                  alphaTexName.c_str());
    } else if (params.GetOneFloat("alpha", 1.f) == 0.f)
        alphaTex = std::make_shared<ConstantTexture<Float>>(0.f);

    std::shared_ptr<Texture<Float>> shadowAlphaTex;
    std::string shadowAlphaTexName = params.FindTexture("shadowalpha");
    if (shadowAlphaTexName != "") {
        if (graphicsState.floatTextures.find(shadowAlphaTexName) != graphicsState.floatTextures.end())
            shadowAlphaTex = graphicsState.floatTextures[shadowAlphaTexName];
        else
            Error(
                "Couldn't find float texture \"%s\" for \"shadowalpha\" "
                "parameter",
                shadowAlphaTexName.c_str());
    } else if (params.GetOneFloat("shadowalpha", 1.f) == 0.f)
        shadowAlphaTex = std::make_shared<ConstantTexture<Float>>(0.f);

    if (!curTransform.IsAnimated()) {
        // Initialize _prims_ and _areaLights_ for static shape

        // Create shapes for shape _name_
        std::shared_ptr<const Transform> ObjToWorld, WorldToObj;
        transformCache.Lookup(curTransform[0], &ObjToWorld, &WorldToObj);
        std::vector<std::shared_ptr<Shape>> shapes =
            MakeShapes(name, ObjToWorld, WorldToObj,
                       graphicsState.reverseOrientation, params);
        if (shapes.empty()) return;
        std::shared_ptr<Material> mtl = graphicsState.GetMaterialForShape(params);
        params.ReportUnused();
        MediumInterface mi = graphicsState.CreateMediumInterface();
        for (auto &s : shapes) {
            // Possibly create area light for shape
            std::shared_ptr<AreaLight> area;
            if (!graphicsState.areaLightName.empty()) {
                area = MakeAreaLight(graphicsState.areaLightName, curTransform[0],
                                     mi, *graphicsState.areaLightParams, s);
                if (area) areaLights.push_back(area);
            }
            if (!area && !mi.IsMediumTransition() && !alphaTex &&
                !shadowAlphaTex)
                prims.push_back(std::make_shared<SimplePrimitive>(s, mtl));
            else
                prims.push_back(
                    std::make_shared<GeometricPrimitive>(s, mtl, area, mi, alphaTex,
                                                         shadowAlphaTex));
        }
        params.ReportUnused();
    } else {
        // Initialize _prims_ and _areaLights_ for animated shape

        // Create initial shape or shapes for animated shape
        if (!graphicsState.areaLightName.empty())
            Warning(
                "Ignoring currently set area light when creating "
                "animated shape");
        std::shared_ptr<const Transform> identity;
        transformCache.Lookup(Transform(), &identity, nullptr);
        std::vector<std::shared_ptr<Shape>> shapes = MakeShapes(
            name, identity, identity, graphicsState.reverseOrientation, params);
        if (shapes.empty()) return;

        // Create _GeometricPrimitive_(s) for animated shape
        std::shared_ptr<Material> mtl = graphicsState.GetMaterialForShape(params);
        params.ReportUnused();
        MediumInterface mi = graphicsState.CreateMediumInterface();

        for (auto &s : shapes) {
            if (!mi.IsMediumTransition() && !alphaTex && !shadowAlphaTex)
                prims.push_back(std::make_shared<SimplePrimitive>(s, mtl));
            else
                prims.push_back(
                    std::make_shared<GeometricPrimitive>(s, mtl, nullptr, mi, alphaTex,
                                                         shadowAlphaTex));
        }
        params.ReportUnused();

        // Create single _TransformedPrimitive_ for _prims_

        // Get _animatedObjectToWorld_ transform for shape
        static_assert(MaxTransforms == 2,
                      "TransformCache assumes only two transforms");
        std::shared_ptr<const Transform> ObjToWorld[2];
        transformCache.Lookup(curTransform[0], &ObjToWorld[0], nullptr);
        transformCache.Lookup(curTransform[1], &ObjToWorld[1], nullptr);
        AnimatedTransform animatedObjectToWorld(
            ObjToWorld[0], renderOptions->transformStartTime, ObjToWorld[1],
            renderOptions->transformEndTime);
        if (prims.size() > 1) {
            std::shared_ptr<Primitive> bvh = std::make_shared<BVHAccel>(prims);
            prims.clear();
            prims.push_back(bvh);
        }
        prims[0] = std::make_shared<TransformedPrimitive>(
            prims[0], animatedObjectToWorld);
    }
    // Add _prims_ and _areaLights_ to scene or current instance
    if (renderOptions->currentInstance) {
        if (areaLights.size())
            Warning("Area lights not supported with object instancing");
        renderOptions->currentInstance->insert(
            renderOptions->currentInstance->end(), prims.begin(), prims.end());
    } else {
        renderOptions->primitives.insert(renderOptions->primitives.end(),
                                         prims.begin(), prims.end());
        if (areaLights.size())
            renderOptions->lights.insert(renderOptions->lights.end(),
                                         areaLights.begin(), areaLights.end());
    }
}

void pbrtReverseOrientation() {
    VERIFY_WORLD("ReverseOrientation");
    graphicsState.reverseOrientation = !graphicsState.reverseOrientation;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sReverseOrientation\n", catIndentCount, "");
}

void pbrtObjectBegin(const std::string &name) {
    VERIFY_WORLD("ObjectBegin");
    pbrtAttributeBegin();

    // Set the shape name attribute using the instance name.
    std::string decl = "string name";
    NamedValues nv;
    nv.name = &decl;
    nv.strings.push_back(&name);
    pbrtAttribute("shape", nv);

    if (renderOptions->currentInstance)
        Error("ObjectBegin called inside of instance definition");
    renderOptions->instances[name] = std::vector<std::shared_ptr<Primitive>>();
    renderOptions->currentInstance = &renderOptions->instances[name];
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sObjectBegin \"%s\"\n", catIndentCount, "", name.c_str());
}

STAT_COUNTER("Scene/Object instances created", nObjectInstancesCreated);

void pbrtObjectEnd() {
    VERIFY_WORLD("ObjectEnd");
    if (!renderOptions->currentInstance)
        Error("ObjectEnd called outside of instance definition");
    renderOptions->currentInstance = nullptr;
    pbrtAttributeEnd();
    ++nObjectInstancesCreated;
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sObjectEnd\n", catIndentCount, "");
}

STAT_COUNTER("Scene/Object instances used", nObjectInstancesUsed);

void pbrtObjectInstance(const std::string &name) {
    VERIFY_WORLD("ObjectInstance");
    // Perform object instance error checking
    if (PbrtOptions.cat || PbrtOptions.toPly)
        printf("%*sObjectInstance \"%s\"\n", catIndentCount, "", name.c_str());
    if (renderOptions->currentInstance) {
        Error("ObjectInstance can't be called inside instance definition");
        return;
    }
    if (renderOptions->instances.find(name) == renderOptions->instances.end()) {
        Error("Unable to find instance named \"%s\"", name.c_str());
        return;
    }
    std::vector<std::shared_ptr<Primitive>> &in =
        renderOptions->instances[name];
    if (in.empty()) return;
    ++nObjectInstancesUsed;
    if (in.size() > 1) {
        // Create aggregate for instance _Primitive_s
        std::shared_ptr<Primitive> accel(
            MakeAccelerator(renderOptions->AcceleratorName, in,
                            renderOptions->AcceleratorParams));
        if (!accel) accel = std::make_shared<BVHAccel>(in);
        in.erase(in.begin(), in.end());
        in.push_back(accel);
    }
    static_assert(MaxTransforms == 2,
                  "TransformCache assumes only two transforms");
    // Create _animatedInstanceToWorld_ transform for instance
    std::shared_ptr<const Transform> InstanceToWorld[2];
    transformCache.Lookup(curTransform[0], &InstanceToWorld[0], nullptr);
    transformCache.Lookup(curTransform[1], &InstanceToWorld[1], nullptr);
    AnimatedTransform animatedInstanceToWorld(
        InstanceToWorld[0], renderOptions->transformStartTime,
        InstanceToWorld[1], renderOptions->transformEndTime);
    std::shared_ptr<Primitive> prim(
        std::make_shared<TransformedPrimitive>(in[0], animatedInstanceToWorld));
    renderOptions->primitives.push_back(prim);
}

void pbrtWorldEnd() {
    VERIFY_WORLD("WorldEnd");
    // Ensure there are no pushed graphics states
    while (pushedGraphicsStates.size()) {
        Warning("Missing end to AttributeBegin");
        pushedGraphicsStates.pop_back();
        pushedTransforms.pop_back();
    }
    while (pushedTransforms.size()) {
        Warning("Missing end to TransformBegin");
        pushedTransforms.pop_back();
    }

    // Create scene and render
    if (PbrtOptions.cat || PbrtOptions.toPly) {
        printf("%*sWorldEnd\n", catIndentCount, "");
    } else {
        std::unique_ptr<Scene> scene;
        std::unique_ptr<Integrator> integrator;
        renderOptions->CreateSceneAndIntegrator(&scene, &integrator);

        // This is kind of ugly; we directly override the current profiler
        // state to switch from parsing/scene construction related stuff to
        // rendering stuff and then switch it back below. The underlying
        // issue is that all the rest of the profiling system assumes
        // hierarchical inheritance of profiling state; this is the only
        // place where that isn't the case.
        CHECK_EQ(CurrentProfilerState(), ProfToBits(Prof::SceneConstruction));
        ProfilerState = ProfToBits(Prof::IntegratorRender);

        // Render!
        integrator->Render();

        CHECK_EQ(CurrentProfilerState(), ProfToBits(Prof::IntegratorRender));
        ProfilerState = ProfToBits(Prof::SceneConstruction);
    }

    // Clean up after rendering. Do this before reporting stats so that
    // destructors can run and update stats as needed.
    if (CachedTexelProvider::textureCache) {
        delete CachedTexelProvider::textureCache;
        CachedTexelProvider::textureCache = nullptr;
    }
    graphicsState = GraphicsState();
    transformCache.Clear();
    currentApiState = APIState::OptionsBlock;
    ImageTexture<Float>::ClearCache();
    ImageTexture<Spectrum>::ClearCache();

    if (!PbrtOptions.cat && !PbrtOptions.toPly) {
        MergeWorkerThreadStats();
        ReportThreadStats();
        if (!PbrtOptions.quiet) {
            PrintStats(stdout);
            ReportProfilerResults(stdout);
            ClearStats();
            ClearProfiler();
        }
        if (PrintCheckRare(stdout))
            ErrorExit("CHECK_RARE failures");
    }

    for (int i = 0; i < MaxTransforms; ++i) curTransform[i] = Transform();
    activeTransformBits = AllTransformsBits;
    namedCoordinateSystems.erase(namedCoordinateSystems.begin(),
                                 namedCoordinateSystems.end());
}

std::shared_ptr<Material> GraphicsState::GetMaterialForShape(
    const ParamSet &shapeParams) {
    CHECK(currentMaterial);
    // Only create a unique material for the shape if the shape's
    // parameters are going to provide values for some of the material
    // parameters.
    if (shapeParams.ShadowsAny(currentMaterial->params)) {
        TextureParams mp({&shapeParams, &currentMaterial->params}, floatTextures,
                         spectrumTextures);
        return MakeMaterial(currentMaterial->name, mp);
    } else
        return currentMaterial->material;
}

MediumInterface GraphicsState::CreateMediumInterface() {
    MediumInterface m;
    if (currentInsideMedium != "") {
        if (renderOptions->namedMedia.find(currentInsideMedium) !=
            renderOptions->namedMedia.end())
            m.inside = renderOptions->namedMedia[currentInsideMedium].get();
        else
            Error("Named medium \"%s\" undefined.",
                  currentInsideMedium.c_str());
    }
    if (currentOutsideMedium != "") {
        if (renderOptions->namedMedia.find(currentOutsideMedium) !=
            renderOptions->namedMedia.end())
            m.outside = renderOptions->namedMedia[currentOutsideMedium].get();
        else
            Error("Named medium \"%s\" undefined.",
                  currentOutsideMedium.c_str());
    }
    return m;
}

void RenderOptions::CreateSceneAndIntegrator(std::unique_ptr<Scene> *scene,
                                             std::unique_ptr<Integrator> *integrator) {
    // Create Scene
    std::shared_ptr<Primitive> accelerator =
        MakeAccelerator(AcceleratorName, primitives, AcceleratorParams);
    *scene = std::make_unique<Scene>(accelerator, lights);

    // Erase primitives and lights from _RenderOptions_
    primitives.erase(primitives.begin(), primitives.end());
    lights.erase(lights.begin(), lights.end());

    // Create Filter, Film, and Camera
    std::unique_ptr<Filter> filter = MakeFilter(FilterName, FilterParams);
    std::unique_ptr<Film> film = MakeFilm(FilmName, FilmParams, std::move(filter));
    std::shared_ptr<Camera> camera =
        MakeCamera(CameraName, CameraParams, CameraToWorld,
                   transformStartTime, transformEndTime, **scene,
                   std::move(film));

    // Create Sampler and Integrator
    std::unique_ptr<Sampler> sampler =
        MakeSampler(SamplerName, SamplerParams, camera->film->GetSampleBounds());
    *integrator = MakeIntegrator(IntegratorName, IntegratorParams, **scene,
                                 camera, std::move(sampler));

    // Check if volume scattering integrator is expected but not specified.
    if (haveScatteringMedia &&
        IntegratorName != "volpath" &&
        IntegratorName != "bdpt" &&
        IntegratorName != "mlt") {
        Warning("Scene has scattering media but \"%s\" integrator doesn't support "
                "volume scattering. Consider using \"volpath\", \"bdpt\", or "
                "\"mlt\".", IntegratorName.c_str());
    }

    // Warn if no light sources are defined
    if ((*scene)->lights.empty() && IntegratorName != "ambientocclusion" &&
        IntegratorName != "aov")
        Warning("No light sources defined in scene; rendering a black image.");
}

}  // namespace pbrt
