
// this stuff gets compiled to PTX and embedded in the binary...

#define PBRT_HAVE_OPTIX 1

#include <pbrt/pbrt.h>

#include <pbrt/gpu/optix.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/ray.h>
#include <pbrt/shapes.h>
#include <pbrt/transform.h>
#include <pbrt/textures.h>
#include <pbrt/util/float.h>
#include <pbrt/util/vecmath.h>

#include <pbrt/transform.cpp> // :-(

using namespace pbrt;

#include <optix_device.h>

extern "C" {
    extern __constant__ pbrt::RayIntersectParameters params;
}

static __forceinline__ __device__ void *
unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    return reinterpret_cast<void *>(uptr);
}

static __forceinline__ __device__ uint32_t
packPointer0(void *ptr) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return uptr >> 32;
}

static __forceinline__ __device__ uint32_t
packPointer1(void *ptr) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T *getPayload() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

static __forceinline__ __device__
SurfaceInteraction getTriangleIntersection() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    const TriangleMesh *mesh = rec.mesh;

    float b1 = optixGetTriangleBarycentrics().x;
    float b2 = optixGetTriangleBarycentrics().y;
    float b0 = 1 - b1 - b2;

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    assert(optixGetTransformListSize() == 1);
    float worldFromObj[12], objFromWorld[12];
    optixGetObjectToWorldTransformMatrix(worldFromObj);
    optixGetWorldToObjectTransformMatrix(objFromWorld);
    SquareMatrix<4> worldFromObjM(worldFromObj[0], worldFromObj[1], worldFromObj[2],  worldFromObj[3],
                                  worldFromObj[4], worldFromObj[5], worldFromObj[6],  worldFromObj[7],
                                  worldFromObj[8], worldFromObj[9], worldFromObj[10], worldFromObj[11],
                                  0.f, 0.f, 0.f, 1.f);
    SquareMatrix<4> objFromWorldM(objFromWorld[0], objFromWorld[1], objFromWorld[2],  objFromWorld[3],
                                  objFromWorld[4], objFromWorld[5], objFromWorld[6],  objFromWorld[7],
                                  objFromWorld[8], objFromWorld[9], objFromWorld[10], objFromWorld[11],
                                  0.f, 0.f, 0.f, 1.f);

    Transform worldFromInstance(worldFromObjM, objFromWorldM);
    return mesh->InteractionFromIntersection(optixGetPrimitiveIndex(),
                                             {b0, b1, b2}, 0. /* time */, wo,
                                             worldFromInstance);
}

static __forceinline__ __device__ bool alphaKilled(const TriangleMeshRecord &rec) {
    if (!rec.alphaTexture)
        return false;

    SurfaceInteraction intr = getTriangleIntersection();

    BasicTextureEvaluator eval;
    TextureEvalContext ctx(intr);
    Float alpha = eval(rec.alphaTexture, ctx);
    return alpha == 0;
}

struct Payload {
    SurfaceInteraction *intr;
    FloatTextureHandle alphaTexture;
};

extern "C" __global__ void __raygen__findClosest() {
    int rayIndex = optixGetLaunchIndex().x;

    if (rayIndex >= params.numActiveRays->load(cuda::std::memory_order_relaxed))
        return;

    int pixelIndex = params.rayIndexToPixelIndex[rayIndex];
    uint32_t p0 = packPointer0(&params.intersections[pixelIndex]);
    uint32_t p1 = packPointer1(&params.intersections[pixelIndex]);

    Point3f rayo = params.rayo->at(rayIndex);
    Vector3f rayd = params.rayd->at(rayIndex);

    optixTrace(params.traversable,
               make_float3(rayo.x, rayo.y, rayo.z),
               make_float3(rayd.x, rayd.y, rayd.z),
               0.f,    // tmin
               params.tMax[rayIndex],
               0.0f,   // rayTime
               OptixVisibilityMask(255),
               // OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               OPTIX_RAY_FLAG_NONE,
               0, /* ray type */
               1, /* total number of ray types */
               0 /* missSBTIndex */,
               p0, p1);
}

extern "C" __global__ void __miss__noop() {
}

///////////////////////////////////////////////////////////////////////////
// Triangles

extern "C" __global__ void __closesthit__triangle() {
    const TriangleMeshRecord &mesh = *(const TriangleMeshRecord *)optixGetSbtDataPointer();
    SurfaceInteraction intr = getTriangleIntersection();

    getPayload<SurfaceInteraction>()->pi = intr.pi;
    getPayload<SurfaceInteraction>()->dpdu = intr.dpdu;
    getPayload<SurfaceInteraction>()->dpdv = intr.dpdv;
    getPayload<SurfaceInteraction>()->dndu = intr.dndu;
    getPayload<SurfaceInteraction>()->dndv = intr.dndv;
    getPayload<SurfaceInteraction>()->n = intr.n;
    getPayload<SurfaceInteraction>()->uv = intr.uv;
    getPayload<SurfaceInteraction>()->wo = intr.wo;
    getPayload<SurfaceInteraction>()->shading = intr.shading;
    getPayload<SurfaceInteraction>()->material = mesh.material;
    getPayload<SurfaceInteraction>()->areaLight = mesh.areaLights ?
        mesh.areaLights[optixGetPrimitiveIndex()] : nullptr;

    int rayIndex = optixGetLaunchIndex().x;
    params.tMax[rayIndex] = optixGetRayTmax();
}

extern "C" __global__ void __anyhit__triangle() {
    const TriangleMeshRecord &mesh = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (alphaKilled(mesh))
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__shadowTriangle() {
    const TriangleMeshRecord &mesh = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (mesh.material && mesh.material.IsTransparent())
        optixIgnoreIntersection();

    if (alphaKilled(mesh))
        optixIgnoreIntersection();
}

///////////////////////////////////////////////////////////////////////////
// Shadows

extern "C" __global__ void __raygen__shadow() {
    int rayIndex = optixGetLaunchIndex().x;

    if (rayIndex >= params.numActiveRays->load(cuda::std::memory_order_relaxed))
        return;

    if (params.occluded[rayIndex] == 1)
        return;

    int missed = 0;

    uint32_t p0 = packPointer0(&missed);
    uint32_t p1 = packPointer1(&missed);

    Point3f rayo = params.rayo->at(rayIndex);
    Vector3f rayd = params.rayd->at(rayIndex);

    optixTrace(params.traversable,
               make_float3(rayo.x, rayo.y, rayo.z),
               make_float3(rayd.x, rayd.y, rayd.z),
               1e-5f,    // tmin
               params.tMax[rayIndex],
               0.0f,   // rayTime
               OptixVisibilityMask(255),
               /* TODO: if no alpha mapped stuff, pass:
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, */
               OPTIX_RAY_FLAG_NONE,
               0, /* ray type */
               1, /* total number of ray types */
               0, /* missSBTIndex */
               p0, p1);

    if (!missed)
        params.occluded[rayIndex] = 1;
}

extern "C" __global__ void __miss__shadow() {
    *getPayload<int>() = 1;
}

/////////////////////////////////////////////////////////////////////////////////////
// Quadric

static __device__ inline SurfaceInteraction getQuadricIntersection() {
    QuadricRecord &shapeRecord = *((QuadricRecord *)optixGetSbtDataPointer());

    QuadricIntersection si;
    si.pObj = Point3f(BitsToFloat(optixGetAttribute_0()),
                      BitsToFloat(optixGetAttribute_1()),
                      BitsToFloat(optixGetAttribute_2()));
    si.phi = BitsToFloat(optixGetAttribute_3());
    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    SurfaceInteraction intr;

    if (const Sphere *sphere = shapeRecord.shape.CastOrNullptr<Sphere>())
        intr = sphere->InteractionFromIntersection(si, wo, 0. /* time */);
    else if (const Cylinder *cylinder = shapeRecord.shape.CastOrNullptr<Cylinder>())
        intr = cylinder->InteractionFromIntersection(si, wo, 0. /* time */);
    else if (const Disk *disk = shapeRecord.shape.CastOrNullptr<Disk>())
        intr = disk->InteractionFromIntersection(si, wo, 0. /* time */);
    else
        assert(!"unexpected quadric");

    return intr;
}

extern "C" __global__ void __closesthit__quadric() {
    QuadricRecord &quadricRecord = *((QuadricRecord *)optixGetSbtDataPointer());
    SurfaceInteraction intr = getQuadricIntersection();

    getPayload<SurfaceInteraction>()->pi = intr.pi;
    getPayload<SurfaceInteraction>()->dpdu = intr.dpdu;
    getPayload<SurfaceInteraction>()->dpdv = intr.dpdv;
    getPayload<SurfaceInteraction>()->dndu = intr.dndu;
    getPayload<SurfaceInteraction>()->dndv = intr.dndv;
    getPayload<SurfaceInteraction>()->n = intr.n;
    getPayload<SurfaceInteraction>()->uv = intr.uv;
    getPayload<SurfaceInteraction>()->wo = intr.wo;
    getPayload<SurfaceInteraction>()->shading = intr.shading;
    getPayload<SurfaceInteraction>()->material = quadricRecord.material;
    getPayload<SurfaceInteraction>()->areaLight = quadricRecord.areaLight;

    int rayIndex = optixGetLaunchIndex().x;
    params.tMax[rayIndex] = optixGetRayTmax();
}

static __forceinline__ __device__
bool alphaKilled(const QuadricRecord &quadricRecord) {
    if (!quadricRecord.alphaTexture)
        return false;

    SurfaceInteraction intr = getQuadricIntersection();

    BasicTextureEvaluator eval;
    TextureEvalContext ctx(intr);
    Float alpha = eval(quadricRecord.alphaTexture, ctx);
    return alpha == 0;
}

extern "C" __global__ void __anyhit__quadric() {
    QuadricRecord &quadricRecord = *((QuadricRecord *)optixGetSbtDataPointer());

    if (alphaKilled(quadricRecord))
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__shadowQuadric() {
    QuadricRecord &quadricRecord = *((QuadricRecord *)optixGetSbtDataPointer());

    if (quadricRecord.material && quadricRecord.material.IsTransparent())
        optixIgnoreIntersection();

    if (alphaKilled(quadricRecord))
        optixIgnoreIntersection();
}

extern "C" __global__ void __intersection__quadric() {
    QuadricRecord &quadricRecord = *((QuadricRecord *)optixGetSbtDataPointer());

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));
    pstd::optional<QuadricIntersection> isect;

    if (const Sphere *sphere = quadricRecord.shape.CastOrNullptr<Sphere>())
        isect = sphere->BasicIntersect(ray, tMax);
    else if (const Cylinder *cylinder = quadricRecord.shape.CastOrNullptr<Cylinder>())
        isect = cylinder->BasicIntersect(ray, tMax);
    else if (const Disk *disk = quadricRecord.shape.CastOrNullptr<Disk>())
        isect = disk->BasicIntersect(ray, tMax);

    if (isect)
        optixReportIntersection(isect->tHit,
                                0 /* hit kind */,
                                FloatToBits(isect->pObj.x),
                                FloatToBits(isect->pObj.y),
                                FloatToBits(isect->pObj.z),
                                FloatToBits(isect->phi));
}

///////////////////////////////////////////////////////////////////////////
// Bilinear patches

static __forceinline__ __device__
SurfaceInteraction getBilinearPatchIntersection() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    Point2f uv(BitsToFloat(optixGetAttribute_0()),
               BitsToFloat(optixGetAttribute_1()));
    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    return rec.mesh->InteractionFromIntersection(optixGetPrimitiveIndex(), uv,
                                                 0.f /* time */, wo);
}

extern "C" __global__ void __closesthit__bilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());
    SurfaceInteraction intr = getBilinearPatchIntersection();

    getPayload<SurfaceInteraction>()->pi = intr.pi;
    getPayload<SurfaceInteraction>()->dpdu = intr.dpdu;
    getPayload<SurfaceInteraction>()->dpdv = intr.dpdv;
    getPayload<SurfaceInteraction>()->dndu = intr.dndu;
    getPayload<SurfaceInteraction>()->dndv = intr.dndv;
    getPayload<SurfaceInteraction>()->n = intr.n;
    getPayload<SurfaceInteraction>()->uv = intr.uv;
    getPayload<SurfaceInteraction>()->wo = intr.wo;
    getPayload<SurfaceInteraction>()->shading = intr.shading;
    getPayload<SurfaceInteraction>()->material = rec.material;
    getPayload<SurfaceInteraction>()->areaLight = rec.areaLights ? rec.areaLights[optixGetPrimitiveIndex()] :
        nullptr;

    int rayIndex = optixGetLaunchIndex().x;
    params.tMax[rayIndex] = optixGetRayTmax();
}

static __forceinline__ __device__
bool alphaKilled(const BilinearMeshRecord &rec) {
    if (!rec.alphaTexture)
        return false;

    SurfaceInteraction intr = getBilinearPatchIntersection();
    BasicTextureEvaluator eval;
    TextureEvalContext ctx(intr);
    Float alpha = eval(rec.alphaTexture, ctx);
    return alpha == 0;

}

extern "C" __global__ void __anyhit__bilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    if (alphaKilled(rec))
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__shadowBilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    if (rec.material && rec.material.IsTransparent())
        optixIgnoreIntersection();

    if (alphaKilled(rec))
        optixIgnoreIntersection();
}

extern "C" __global__ void __intersection__bilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));

    int vertexIndex = 4 * optixGetPrimitiveIndex();
    Point3f p00 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex]];
    Point3f p10 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 1]];
    Point3f p01 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 2]];
    Point3f p11 = rec.mesh->p[rec.mesh->vertexIndices[vertexIndex + 3]];
    pstd::optional<BilinearIntersection> isect =
        BilinearPatch::Intersect(ray, tMax, p00, p10, p01, p11);

    if (isect)
        optixReportIntersection(isect->t,
                                0 /* hit kind */,
                                FloatToBits(isect->uv[0]),
                                FloatToBits(isect->uv[1]));
}

///////////////////////////////////////////////////////////////////////////
// Random hit

struct RandomHitPayload {
    MaterialHandle material;
    WeightedReservoirSampler<SurfaceInteraction, Float> *reservoirSampler;
};

extern "C" __global__ void __raygen__randomHit() {
    int rayIndex = optixGetLaunchIndex().x;

    if (rayIndex >= params.numActiveRays->load(cuda::std::memory_order_relaxed))
        return;

    RandomHitPayload payload;
    payload.material = params.materialArray[rayIndex];
    payload.reservoirSampler = &params.reservoirSamplerArray[rayIndex];
    uint32_t p0 = packPointer0(&payload);
    uint32_t p1 = packPointer1(&payload);

    Point3f rayo = params.rayo->at(rayIndex);
    Vector3f rayd = params.rayd->at(rayIndex);

    optixTrace(params.traversable,
               make_float3(rayo.x, rayo.y, rayo.z),
               make_float3(rayd.x, rayd.y, rayd.z),
               0.f,    // tmin
               params.tMax[rayIndex],
               0.0f,   // rayTime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               0, /* ray type */
               1, /* total number of ray types */
               0 /* missSBTIndex */,
               p0, p1);
}

extern "C" __global__ void __anyhit__randomHitTriangle() {
    const TriangleMeshRecord &mesh = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (mesh.material == getPayload<RandomHitPayload>()->material) {
        // It's a candidate...
        getPayload<RandomHitPayload>()->reservoirSampler->Add(
            [&] __device__ () { return getTriangleIntersection(); }, 1.f);
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__randomHitBilinearPatch() {
    BilinearMeshRecord &rec = *(BilinearMeshRecord *)optixGetSbtDataPointer();

    if (rec.material == getPayload<RandomHitPayload>()->material) {
        getPayload<RandomHitPayload>()->reservoirSampler->Add(
            [&] __device__ () { return getBilinearPatchIntersection(); }, 1.f);
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__randomHitQuadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    if (rec.material == getPayload<RandomHitPayload>()->material) {
        getPayload<RandomHitPayload>()->reservoirSampler->Add(
            [&] __device__ () { return getQuadricIntersection(); }, 1.f);
    }

    optixIgnoreIntersection();
}
