// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_GENSCENE_H
#define PBRT_GENSCENE_H

#include <pbrt/pbrt.h>

#include <pbrt/cameras.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/print.h>
#include <pbrt/util/transform.h>

#include <map>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace pbrt {

struct SceneEntity {
    SceneEntity() = default;
    SceneEntity(const std::string &name, ParameterDictionary parameters, FileLoc loc)
        : name(name), parameters(parameters), loc(loc) {}

    std::string ToString() const {
        return StringPrintf("[ SceneEntity name: %s parameters: %s loc: %s ]", name,
                            parameters, loc);
    }

    std::string name;
    ParameterDictionary parameters;
    FileLoc loc;
};

struct TransformedSceneEntity : public SceneEntity {
    TransformedSceneEntity() = default;
    TransformedSceneEntity(const std::string &name, ParameterDictionary parameters,
                           FileLoc loc, const AnimatedTransform &worldFromObject)
        : SceneEntity(name, parameters, loc), worldFromObject(worldFromObject) {}

    std::string ToString() const {
        return StringPrintf("[ TransformedSeneEntity name: %s parameters: %s loc: %s "
                            "worldFromObject: %s ]",
                            name, parameters, loc, worldFromObject);
    }

    AnimatedTransform worldFromObject;
};

struct CameraSceneEntity : public SceneEntity {
    CameraSceneEntity() = default;
    CameraSceneEntity(const std::string &name, ParameterDictionary parameters,
                      FileLoc loc, const CameraTransform &cameraTransform,
                      const std::string &medium)
        : SceneEntity(name, parameters, loc),
          cameraTransform(cameraTransform),
          medium(medium) {}

    std::string ToString() const {
        return StringPrintf("[ CameraSeneEntity name: %s parameters: %s loc: %s "
                            "cameraTransform: %s medium: %s ]",
                            name, parameters, loc, cameraTransform, medium);
    }

    CameraTransform cameraTransform;
    std::string medium;
};

struct ShapeSceneEntity : public SceneEntity {
    ShapeSceneEntity() = default;
    ShapeSceneEntity(const std::string &name, ParameterDictionary parameters, FileLoc loc,
                     const Transform *worldFromObject, const Transform *objectFromWorld,
                     bool reverseOrientation, int materialIndex,
                     const std::string &materialName, int lightIndex,
                     const std::string &insideMedium, const std::string &outsideMedium)
        : SceneEntity(name, parameters, loc),
          worldFromObject(worldFromObject),
          objectFromWorld(objectFromWorld),
          reverseOrientation(reverseOrientation),
          materialIndex(materialIndex),
          materialName(materialName),
          lightIndex(lightIndex),
          insideMedium(insideMedium),
          outsideMedium(outsideMedium) {}

    std::string ToString() const {
        return StringPrintf(
            "[ ShapeSeneEntity name: %s parameters: %s loc: %s "
            "worldFromObject: %s objectFromWorld: %s reverseOrientation: %s "
            "materialIndex: %d materialName: %s lightIndex: %d "
            "insideMedium: %s outsideMedium: %s]",
            name, parameters, loc, *worldFromObject, *objectFromWorld, reverseOrientation,
            materialIndex, materialName, lightIndex, insideMedium, outsideMedium);
    }

    const Transform *worldFromObject = nullptr, *objectFromWorld = nullptr;
    bool reverseOrientation = false;
    int materialIndex;  // one of these two...  std::variant?
    std::string materialName;
    int lightIndex = -1;
    std::string insideMedium, outsideMedium;
};

struct AnimatedShapeSceneEntity : public TransformedSceneEntity {
    AnimatedShapeSceneEntity() = default;
    AnimatedShapeSceneEntity(const std::string &name, ParameterDictionary parameters,
                             FileLoc loc, const AnimatedTransform &worldFromObject,
                             const Transform *identity, bool reverseOrientation,
                             int materialIndex, const std::string &materialName,
                             int lightIndex, const std::string &insideMedium,
                             const std::string &outsideMedium)
        : TransformedSceneEntity(name, parameters, loc, worldFromObject),
          identity(identity),
          reverseOrientation(reverseOrientation),
          materialIndex(materialIndex),
          materialName(materialName),
          lightIndex(lightIndex),
          insideMedium(insideMedium),
          outsideMedium(outsideMedium) {}

    std::string ToString() const {
        return StringPrintf(
            "[ ShapeSeneEntity name: %s parameters: %s loc: %s "
            "worldFromObject: %s reverseOrientation: %s materialIndex: %d "
            "materialName: %s insideMedium: %s outsideMedium: %s]",
            name, parameters, loc, worldFromObject, reverseOrientation, materialIndex,
            materialName, insideMedium, outsideMedium);
    }

    const Transform *identity = nullptr;
    bool reverseOrientation = false;
    int materialIndex;  // one of these two...  std::variant?
    std::string materialName;
    int lightIndex = -1;
    std::string insideMedium, outsideMedium;
};

struct InstanceDefinitionSceneEntity {
    InstanceDefinitionSceneEntity() = default;
    InstanceDefinitionSceneEntity(const std::string &name, FileLoc loc)
        : name(name), loc(loc) {}

    std::string ToString() const {
        return StringPrintf("[ InstanceDefinitionSceneEntity name: %s loc: %s "
                            " shapes: %s animatedShapes: %s ]",
                            name, loc, shapes, animatedShapes);
    }

    std::string name;
    FileLoc loc;
    std::vector<ShapeSceneEntity> shapes;
    std::vector<AnimatedShapeSceneEntity> animatedShapes;
};

struct TextureSceneEntity : public TransformedSceneEntity {
    TextureSceneEntity() = default;
    TextureSceneEntity(const std::string &texName, ParameterDictionary parameters,
                       FileLoc loc, const AnimatedTransform &worldFromObject)
        : TransformedSceneEntity("", std::move(parameters), loc, worldFromObject),
          texName(texName) {}

    std::string ToString() const {
        return StringPrintf("[ TextureSeneEntity name: %s parameters: %s loc: %s "
                            "worldFromObject: %s texName: %s ]",
                            name, parameters, loc, worldFromObject, texName);
    }

    std::string texName;
};

struct LightSceneEntity : public TransformedSceneEntity {
    LightSceneEntity() = default;
    LightSceneEntity(const std::string &name, ParameterDictionary parameters, FileLoc loc,
                     const AnimatedTransform &worldFromLight, const std::string &medium)
        : TransformedSceneEntity(name, parameters, loc, worldFromLight), medium(medium) {}

    std::string ToString() const {
        return StringPrintf("[ LightSeneEntity name: %s parameters: %s loc: %s "
                            "worldFromObject: %s medium: %s ]",
                            name, parameters, loc, worldFromObject, medium);
    }

    std::string medium;
};

struct InstanceSceneEntity : public SceneEntity {
    InstanceSceneEntity() = default;
    InstanceSceneEntity(const std::string &name, FileLoc loc,
                        const AnimatedTransform &worldFromInstanceAnim,
                        const Transform *worldFromInstance)
        : SceneEntity(name, {}, loc),
          worldFromInstanceAnim(worldFromInstanceAnim),
          worldFromInstance(worldFromInstance) {}

    std::string ToString() const {
        return StringPrintf(
            "[ InstanceSeneEntity name: %s loc: %s "
            "worldFromInstanceAnim: %s worldFromInstance: %s ]",
            name, loc, worldFromInstanceAnim,
            worldFromInstance ? worldFromInstance->ToString() : std::string("nullptr"));
    }

    AnimatedTransform worldFromInstanceAnim;
    const Transform *worldFromInstance;
};

struct TransformHash {
    size_t operator()(const Transform *t) const { return t->Hash(); }
};

class TransformCache {
  public:
    TransformCache(Allocator alloc)
        : bufferResource(alloc.resource()),
          alloc(&bufferResource) {}
    ~TransformCache();

    // TransformCache Public Methods
    const Transform *Lookup(const Transform &t);

  private:
    // TransformCache Private Data
    pstd::pmr::monotonic_buffer_resource bufferResource;
    Allocator alloc;
    std::unordered_set<Transform *, TransformHash> hashTable;
};

constexpr int MaxTransforms = 2;

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
        for (int i = 0; i < MaxTransforms; ++i)
            tInv.t[i] = Inverse(ts.t[i]);
        return tInv;
    }
    bool IsAnimated() const {
        for (int i = 0; i < MaxTransforms - 1; ++i)
            if (t[i] != t[i + 1])
                return true;
        return false;
    }

  private:
    Transform t[MaxTransforms];
};

class ParsedScene : public SceneRepresentation {
  public:
    ParsedScene(Allocator alloc);
    ~ParsedScene();

    void Option(const std::string &name, const std::string &value, FileLoc loc);
    void Identity(FileLoc loc);
    void Translate(Float dx, Float dy, Float dz, FileLoc loc);
    void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc);
    void Scale(Float sx, Float sy, Float sz, FileLoc loc);
    void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux,
                Float uy, Float uz, FileLoc loc);
    void ConcatTransform(Float transform[16], FileLoc loc);
    void Transform(Float transform[16], FileLoc loc);
    void CoordinateSystem(const std::string &, FileLoc loc);
    void CoordSysTransform(const std::string &, FileLoc loc);
    void ActiveTransformAll(FileLoc loc);
    void ActiveTransformEndTime(FileLoc loc);
    void ActiveTransformStartTime(FileLoc loc);
    void TransformTimes(Float start, Float end, FileLoc loc);
    void ColorSpace(const std::string &n, FileLoc loc);
    void PixelFilter(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Film(const std::string &type, ParsedParameterVector params, FileLoc loc);
    void Sampler(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Accelerator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Integrator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Camera(const std::string &, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void MediumInterface(const std::string &insideName, const std::string &outsideName,
                         FileLoc loc);
    void WorldBegin(FileLoc loc);
    void AttributeBegin(FileLoc loc);
    void AttributeEnd(FileLoc loc);
    void Attribute(const std::string &target, ParsedParameterVector params, FileLoc loc);
    void TransformBegin(FileLoc loc);
    void TransformEnd(FileLoc loc);
    void Texture(const std::string &name, const std::string &type,
                 const std::string &texname, ParsedParameterVector params, FileLoc loc);
    void Material(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                           FileLoc loc);
    void NamedMaterial(const std::string &name, FileLoc loc);
    void LightSource(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void AreaLightSource(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void Shape(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void ReverseOrientation(FileLoc loc);
    void ObjectBegin(const std::string &name, FileLoc loc);
    void ObjectEnd(FileLoc loc);
    void ObjectInstance(const std::string &name, FileLoc loc);

    void EndOfFiles();

    CameraSceneEntity camera;
    SceneEntity film;
    SceneEntity sampler;
    SceneEntity integrator;
    SceneEntity filter;
    SceneEntity accelerator;

    std::vector<std::pair<std::string, SceneEntity>> namedMaterials;
    std::vector<SceneEntity> materials;
    std::map<std::string, TransformedSceneEntity> media;
    std::vector<std::pair<std::string, TextureSceneEntity>> floatTextures;
    std::vector<std::pair<std::string, TextureSceneEntity>> spectrumTextures;
    std::map<std::string, InstanceDefinitionSceneEntity> instanceDefinitions;

    std::vector<LightSceneEntity> lights;
    std::vector<SceneEntity> areaLights;
    std::vector<ShapeSceneEntity> shapes;
    std::vector<AnimatedShapeSceneEntity> animatedShapes;
    std::vector<InstanceSceneEntity> instances;

    std::string ToString() const;

    void CreateTextures(std::map<std::string, FloatTextureHandle> *floatTextureMap,
                        std::map<std::string, SpectrumTextureHandle> *spectrumTextureMap,
                        Allocator alloc, bool gpu) const;

    void CreateMaterials(
        /*const*/ std::map<std::string, FloatTextureHandle> &floatTextures,
        /*const*/
        std::map<std::string, SpectrumTextureHandle> &spectrumTextures, Allocator alloc,
        std::map<std::string, MaterialHandle> *namedMaterials,
        std::vector<MaterialHandle> *materials) const;

    std::map<std::string, MediumHandle> CreateMedia(Allocator alloc) const;

  private:
    class Transform GetCTM(int index) const {
        // I believe that this GetMatrix() is related to precision: we get
        // a more accurate inverse if we re-invert from scratch than to
        // take the accumulated inverse...
        return pbrt::Transform((renderFromWorld * curTransform[index]).GetMatrix());
    }

    bool CTMIsAnimated() const { return curTransform.IsAnimated(); }

    Float transformStartTime = 0, transformEndTime = 1;
    class Transform renderFromWorld;
    InstanceDefinitionSceneEntity *currentInstance = nullptr;

    class GraphicsState;

    static constexpr int StartTransformBits = 1 << 0;
    static constexpr int EndTransformBits = 1 << 1;
    static constexpr int AllTransformsBits = (1 << MaxTransforms) - 1;

    enum class APIState { Uninitialized, OptionsBlock, WorldBlock };
    APIState currentApiState = APIState::Uninitialized;
    TransformSet curTransform;
    uint32_t activeTransformBits = AllTransformsBits;
    std::map<std::string, TransformSet> namedCoordinateSystems;
    GraphicsState *graphicsState;
    std::vector<GraphicsState> pushedGraphicsStates;
    std::vector<TransformSet> pushedTransforms;
    std::vector<uint32_t> pushedActiveTransformBits;
    std::vector<std::pair<char, FileLoc>>
        pushStack;  // 'a': attribute, 't': transform, 'o': object

    TransformCache transformCache;
};

class FormattingScene : public SceneRepresentation {
  public:
    FormattingScene(bool toPly, bool upgrade) : toPly(toPly), upgrade(upgrade) {}
    ~FormattingScene();

    void Option(const std::string &name, const std::string &value, FileLoc loc);
    void Identity(FileLoc loc);
    void Translate(Float dx, Float dy, Float dz, FileLoc loc);
    void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc);
    void Scale(Float sx, Float sy, Float sz, FileLoc loc);
    void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux,
                Float uy, Float uz, FileLoc loc);
    void ConcatTransform(Float transform[16], FileLoc loc);
    void Transform(Float transform[16], FileLoc loc);
    void CoordinateSystem(const std::string &, FileLoc loc);
    void CoordSysTransform(const std::string &, FileLoc loc);
    void ActiveTransformAll(FileLoc loc);
    void ActiveTransformEndTime(FileLoc loc);
    void ActiveTransformStartTime(FileLoc loc);
    void TransformTimes(Float start, Float end, FileLoc loc);
    void ColorSpace(const std::string &n, FileLoc loc);
    void PixelFilter(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Film(const std::string &type, ParsedParameterVector params, FileLoc loc);
    void Sampler(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Accelerator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Integrator(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void Camera(const std::string &, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void MediumInterface(const std::string &insideName, const std::string &outsideName,
                         FileLoc loc);
    void WorldBegin(FileLoc loc);
    void AttributeBegin(FileLoc loc);
    void AttributeEnd(FileLoc loc);
    void Attribute(const std::string &target, ParsedParameterVector params, FileLoc loc);
    void TransformBegin(FileLoc loc);
    void TransformEnd(FileLoc loc);
    void Texture(const std::string &name, const std::string &type,
                 const std::string &texname, ParsedParameterVector params, FileLoc loc);
    void Material(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                           FileLoc loc);
    void NamedMaterial(const std::string &name, FileLoc loc);
    void LightSource(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void AreaLightSource(const std::string &name, ParsedParameterVector params,
                         FileLoc loc);
    void Shape(const std::string &name, ParsedParameterVector params, FileLoc loc);
    void ReverseOrientation(FileLoc loc);
    void ObjectBegin(const std::string &name, FileLoc loc);
    void ObjectEnd(FileLoc loc);
    void ObjectInstance(const std::string &name, FileLoc loc);

    void EndOfFiles();

    std::string indent(int extra = 0) const {
        return std::string(catIndentCount + 4 * extra, ' ');
    }

  private:
    std::string upgradeMaterialIndex(const std::string &name, ParameterDictionary *dict,
                                     FileLoc loc) const;
    std::string upgradeMaterial(std::string *name, ParameterDictionary *dict,
                                FileLoc loc) const;

    int catIndentCount = 0;
    bool toPly, upgrade;
    std::map<std::string, std::string> definedTextures;
    std::map<std::string, std::string> definedNamedMaterials;
    std::map<std::string, ParameterDictionary> namedMaterialDictionaries;
    std::map<std::string, std::string> definedObjectInstances;
};

}  // namespace pbrt

#endif  // PBRT_GENSCENE_H
