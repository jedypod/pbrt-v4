
// this stuff gets compiled to PTX and embedded in the binary...

#define PBRT_HAVE_OPTIX 1

#include <pbrt/pbrt.h>

#include <pbrt/gpu/optix.h>
#include <pbrt/interaction.h>
#include <pbrt/ray.h>
#include <pbrt/transform.h>
#include <pbrt/textures.h>
#include <pbrt/util/float.h>
#include <pbrt/util/vecmath.h>

#include <pbrt/shapes.cpp> // :-(
#include <pbrt/transform.cpp> // :-(

using namespace pbrt;

#include <optix_device.h>

extern "C" {
    extern __constant__ pbrt::LaunchParams optixLaunchParams;
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

static __forceinline__ __device__ bool alphaKilled(const MeshRecord &rec) {
    assert(rec.alphaTextureHandle != nullptr);

    const TriangleMesh *mesh = rec.mesh;
    const int *idx = mesh->vertexIndices + 3 * optixGetPrimitiveIndex();

    float b1 = optixGetTriangleBarycentrics().x;
    float b2 = optixGetTriangleBarycentrics().y;
    float b0 = 1 - b1 - b2;

    const Point3f *vertices = mesh->p;

    Point3f p = b0 * mesh->p[idx[0]] + b1 * mesh->p[idx[1]] + b2 * mesh->p[idx[2]];
    Point2f uv;
    if (mesh->uv != nullptr)
        uv = b0 * mesh->uv[idx[0]] + b1 * mesh->uv[idx[1]] + b2 * mesh->uv[idx[2]];
    else
        uv = Point2f(b1 + b2, b2);

    OptixTraversableHandle handle = optixGetTransformListHandle(0);
    const float4 *tr =
        optixGetInstanceTransformFromHandle(handle);
    if (tr != nullptr) // ?????
         p = Point3f(tr[0].x * p.x + tr[0].y * p.y + tr[0].z * p.z + tr[0].w,
                     tr[1].x * p.x + tr[1].y * p.y + tr[1].z * p.z + tr[1].w,
                     tr[2].x * p.x + tr[2].y * p.y + tr[2].z * p.z + tr[2].w);

    Normal3f ng = Normal3f(Cross(vertices[idx[0]] - vertices[idx[1]],
                                 vertices[idx[2]] - vertices[idx[1]]));
    const float4 *trInv =
        optixGetInstanceInverseTransformFromHandle(handle);
    if (trInv != nullptr)
        // transpose...
        ng = Normal3f(trInv[0].x * ng.x + trInv[1].x * ng.y + trInv[2].x * ng.z,
                      trInv[0].y * ng.x + trInv[1].y * ng.y + trInv[2].y * ng.z,
                      trInv[0].z * ng.x + trInv[1].z * ng.y + trInv[2].z * ng.z);
    ng = Normalize(ng);

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    Vector3f dpdu, dpdv;
    CoordinateSystem(ng, &dpdu, &dpdv);
    SurfaceInteraction intr(Point3f(p), uv, wo, dpdu, dpdv,
                            Normal3f(0, 0, 0), Normal3f(0, 0, 0), 0.f /* time */,
                            false /*flip normal */);

    BasicTextureEvaluator eval;
    TextureEvalContext ctx(intr);
    Float alpha = eval(*rec.alphaTextureHandle, ctx);
    return alpha == 0;
}

struct Payload {
    SurfaceInteraction *intr;
    FloatTextureHandle *alphaTextureHandle;
};

extern "C" __global__ void __raygen__findClosest() {
    int rayIndex = optixGetLaunchIndex().x;

    if (rayIndex >= *optixLaunchParams.numActiveRays)
        return;

    int pixelIndex = optixLaunchParams.rayIndexToPixelIndex[rayIndex];
    uint32_t p0 = packPointer0(&optixLaunchParams.intersections[pixelIndex]);
    uint32_t p1 = packPointer1(&optixLaunchParams.intersections[pixelIndex]);

    Point3f rayo = optixLaunchParams.rayo->at(rayIndex);
    Vector3f rayd = optixLaunchParams.rayd->at(rayIndex);

    optixTrace(optixLaunchParams.traversable,
               make_float3(rayo.x, rayo.y, rayo.z),
               make_float3(rayd.x, rayd.y, rayd.z),
               0.f,    // tmin
               1e20f,  // tmax
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

extern "C" __global__ void __closesthit__triangle() {
    const MeshRecord &rec = *(const MeshRecord *)optixGetSbtDataPointer();

    const TriangleMesh *mesh = rec.mesh;

    float b1 = optixGetTriangleBarycentrics().x;
    float b2 = optixGetTriangleBarycentrics().y;
    float b0 = 1 - b1 - b2;

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    SurfaceInteraction intr;
    OptixTraversableHandle handle = optixGetTransformListHandle(0);
    const float4 *tr =
        optixGetInstanceTransformFromHandle(handle);
    // TODO: Does this come back null or should we check for identity, or....?????
    if (tr != nullptr) {
        const float4 *trInv =
            optixGetInstanceInverseTransformFromHandle(handle);
        SquareMatrix<4> trM(tr[0].x, tr[0].y, tr[0].z, tr[0].w,
                            tr[1].x, tr[1].y, tr[1].z, tr[1].w,
                            tr[2].x, tr[2].y, tr[2].z, tr[2].w,
                            0.f, 0.f, 0.f, 1.f);
        SquareMatrix<4> trInvM(trInv[0].x, trInv[0].y, trInv[0].z, trInv[0].w,
                               trInv[1].x, trInv[1].y, trInv[1].z, trInv[1].w,
                               trInv[2].x, trInv[2].y, trInv[2].z, trInv[2].w,
                               0.f, 0.f, 0.f, 1.f);

        Transform instToWorld(trM, trInvM);
        intr = mesh->InteractionFromIntersection(optixGetPrimitiveIndex(),
                                                 {b0, b1, b2}, 0. /* time */, wo,
                                                 instToWorld);
    } else
        intr = mesh->InteractionFromIntersection(optixGetPrimitiveIndex(),
                                                 {b0, b1, b2}, 0. /* time */, wo);

    getPayload<SurfaceInteraction>()->pi = intr.pi;
    getPayload<SurfaceInteraction>()->dpdu = intr.dpdu;
    getPayload<SurfaceInteraction>()->dpdv = intr.dpdv;
    getPayload<SurfaceInteraction>()->dndu = intr.dndu;
    getPayload<SurfaceInteraction>()->dndv = intr.dndv;
    getPayload<SurfaceInteraction>()->n = intr.n;
    getPayload<SurfaceInteraction>()->uv = intr.uv;
    getPayload<SurfaceInteraction>()->wo = intr.wo;
    getPayload<SurfaceInteraction>()->shading = intr.shading;
    getPayload<SurfaceInteraction>()->material = rec.materialHandle;
    getPayload<SurfaceInteraction>()->areaLight = rec.areaLights ?
        rec.areaLights[optixGetPrimitiveIndex()] : nullptr;
}

extern "C" __global__ void __anyhit__triangle() {
    const MeshRecord &mesh = *(const MeshRecord *)optixGetSbtDataPointer();

    if (alphaKilled(mesh))
        optixIgnoreIntersection();
}

extern "C" __global__ void __raygen__shadow() {
    int rayIndex = optixGetLaunchIndex().x;

    if (rayIndex >= *optixLaunchParams.numActiveRays)
        return;

    if (optixLaunchParams.occluded[rayIndex] == 1)
        return;

    int missed = 0;

    uint32_t p0 = packPointer0(&missed);
    uint32_t p1 = packPointer1(&missed);

    Point3f rayo = optixLaunchParams.rayo->at(rayIndex);
    Vector3f rayd = optixLaunchParams.rayd->at(rayIndex);

    optixTrace(optixLaunchParams.traversable,
               make_float3(rayo.x, rayo.y, rayo.z),
               make_float3(rayd.x, rayd.y, rayd.z),
               1e-5f,    // tmin
               optixLaunchParams.tMax[rayIndex],
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
        optixLaunchParams.occluded[rayIndex] = 1;
}

extern "C" __global__ void __miss__shadow() {
    *getPayload<int>() = 1;
}

extern "C" __global__ void __anyhit__shadow_triangle() {
    const MeshRecord &mesh = *(const MeshRecord *)optixGetSbtDataPointer();

    if (alphaKilled(mesh))
        optixIgnoreIntersection();
}

extern "C" __global__ void __closesthit__sphere() {
    ShapeRecord &shapeRecord = *((ShapeRecord *)optixGetSbtDataPointer());

    const Sphere *sphere = shapeRecord.shapeHandle->CastOrNullptr<Sphere>();

    QuadricIntersection si;
    si.pObj = Point3f(BitsToFloat(optixGetAttribute_0()),
                      BitsToFloat(optixGetAttribute_1()),
                      BitsToFloat(optixGetAttribute_2()));
    si.phi = BitsToFloat(optixGetAttribute_3());
    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    SurfaceInteraction intr = sphere->InteractionFromIntersection(si, wo, 0. /* time */);

    getPayload<SurfaceInteraction>()->pi = intr.pi;
    getPayload<SurfaceInteraction>()->dpdu = intr.dpdu;
    getPayload<SurfaceInteraction>()->dpdv = intr.dpdv;
    getPayload<SurfaceInteraction>()->dndu = intr.dndu;
    getPayload<SurfaceInteraction>()->dndv = intr.dndv;
    getPayload<SurfaceInteraction>()->n = intr.n;
    getPayload<SurfaceInteraction>()->uv = intr.uv;
    getPayload<SurfaceInteraction>()->wo = intr.wo;
    getPayload<SurfaceInteraction>()->shading = intr.shading;
    getPayload<SurfaceInteraction>()->material = shapeRecord.materialHandle;
    getPayload<SurfaceInteraction>()->areaLight = shapeRecord.areaLight;
}

extern "C" __global__ void __anyhit__sphere() {
}

extern "C" __global__ void __intersection__sphere() {
    ShapeRecord &shapeRecord = *((ShapeRecord *)optixGetSbtDataPointer());

    const Sphere *sphere = shapeRecord.shapeHandle->CastOrNullptr<Sphere>();
    if (!sphere)
        printf("not sphere!\n");

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();

    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));
    pstd::optional<QuadricIntersection> isect = sphere->BasicIntersect(ray, tMax);

    if (isect)
        optixReportIntersection(isect->tHit,
                                0 /* hit kind */,
                                FloatToBits(isect->pObj.x),
                                FloatToBits(isect->pObj.y),
                                FloatToBits(isect->pObj.z),
                                FloatToBits(isect->phi));
}
