// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/pbrt.h>

#include <pbrt/gpu/accel.h>
#include <pbrt/gpu/optix.h>
#include <pbrt/interaction.h>
#include <pbrt/materials.h>
#include <pbrt/media.h>
#include <pbrt/shapes.h>
#include <pbrt/textures.h>
#include <pbrt/util/float.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/transform.h>
#include <pbrt/util/vecmath.h>

#include <pbrt/util/color.cpp>          // :-(
#include <pbrt/util/colorspace.cpp>     // :-(
#include <pbrt/util/log.cpp>            // :-(
#include <pbrt/util/sobolmatrices.cpp>  // :-(
#include <pbrt/util/spectrum.cpp>       // :-(
#include <pbrt/util/transform.cpp>      // :-(

using namespace pbrt;

#include <optix_device.h>

extern "C" {
extern __constant__ pbrt::RayIntersectParameters params;
}

static __forceinline__ __device__ void *unpackPointer(uint32_t i0, uint32_t i1) {
    const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
    return reinterpret_cast<void *>(uptr);
}

static __forceinline__ __device__ uint32_t packPointer0(void *ptr) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return uptr >> 32;
}

static __forceinline__ __device__ uint32_t packPointer1(void *ptr) {
    uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
    return uptr & 0x00000000ffffffff;
}

template <typename T>
static __forceinline__ __device__ T *getPayload() {
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}

static __forceinline__ __device__ pstd::optional<SurfaceInteraction>
getTriangleIntersection() {
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
    SquareMatrix<4> worldFromObjM(worldFromObj[0], worldFromObj[1], worldFromObj[2],
                                  worldFromObj[3], worldFromObj[4], worldFromObj[5],
                                  worldFromObj[6], worldFromObj[7], worldFromObj[8],
                                  worldFromObj[9], worldFromObj[10], worldFromObj[11],
                                  0.f, 0.f, 0.f, 1.f);
    SquareMatrix<4> objFromWorldM(objFromWorld[0], objFromWorld[1], objFromWorld[2],
                                  objFromWorld[3], objFromWorld[4], objFromWorld[5],
                                  objFromWorld[6], objFromWorld[7], objFromWorld[8],
                                  objFromWorld[9], objFromWorld[10], objFromWorld[11],
                                  0.f, 0.f, 0.f, 1.f);

    Transform worldFromInstance(worldFromObjM, objFromWorldM);
    return Triangle::InteractionFromIntersection(mesh, optixGetPrimitiveIndex(),
                                                 {b0, b1, b2}, 0. /* time */, wo,
                                                 worldFromInstance);
}

static __forceinline__ __device__ bool alphaKilled(const TriangleMeshRecord &rec) {
    if (!rec.alphaTexture)
        return false;

    pstd::optional<SurfaceInteraction> intr = getTriangleIntersection();
    if (!intr)
        return true;

    BasicTextureEvaluator eval;
    TextureEvalContext ctx(*intr);
    Float alpha = eval(rec.alphaTexture, ctx);
    return alpha == 0;
}

extern "C" __global__ void __raygen__findClosest() {
    PathRayIndex rayIndex(optixGetLaunchIndex().x);

    if (rayIndex >= params.pathRays->Size())
        return;

    Float tMax;
    Ray ray = params.pathRays->GetRay(rayIndex, &tMax);

    PixelIndex pixelIndex = params.rayIndexToPixelIndex[rayIndex];
    SurfaceInteraction &isect = params.intersections[pixelIndex];
    uint32_t p0 = packPointer0(&isect);
    uint32_t p1 = packPointer1(&isect);

    uint32_t missed = 0;
    optixTrace(params.traversable, make_float3(ray.o.x, ray.o.y, ray.o.z),
               make_float3(ray.d.x, ray.d.y, ray.d.z),
               0.f,  // tmin
               tMax,
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               // OPTIX_RAY_FLAG_DISABLE_ANYHIT,
               OPTIX_RAY_FLAG_NONE, 0, /* ray type */
               1,                      /* total number of ray types */
               0 /* missSBTIndex */, p0, p1, missed);

    bool hit = !missed;
    if (hit && !isect.mediumInterface)
        isect.medium = ray.medium;

    params.interactionType[rayIndex] =
        hit ? InteractionType::Surface : InteractionType::None;
}

extern "C" __global__ void __miss__noop() {
    optixSetPayload_2(1);
}

///////////////////////////////////////////////////////////////////////////
// Triangles

extern "C" __global__ void __closesthit__triangle() {
    const TriangleMeshRecord &mesh =
        *(const TriangleMeshRecord *)optixGetSbtDataPointer();
    pstd::optional<SurfaceInteraction> intr = getTriangleIntersection();

    // It's slightly dicey to assume intr is valid. But invalid would
    // presumably mean that OptiX returned a hit with a degenerate
    // triangle...
    getPayload<SurfaceInteraction>()->pi = intr->pi;
    getPayload<SurfaceInteraction>()->dpdu = intr->dpdu;
    getPayload<SurfaceInteraction>()->dpdv = intr->dpdv;
    getPayload<SurfaceInteraction>()->dndu = intr->dndu;
    getPayload<SurfaceInteraction>()->dndv = intr->dndv;
    getPayload<SurfaceInteraction>()->n = intr->n;
    getPayload<SurfaceInteraction>()->uv = intr->uv;
    getPayload<SurfaceInteraction>()->wo = intr->wo;
    getPayload<SurfaceInteraction>()->shading = intr->shading;
    getPayload<SurfaceInteraction>()->material = mesh.material;
    getPayload<SurfaceInteraction>()->areaLight =
        mesh.areaLights ? mesh.areaLights[optixGetPrimitiveIndex()] : nullptr;
    getPayload<SurfaceInteraction>()->mediumInterface =
        (mesh.mediumInterface && mesh.mediumInterface->IsMediumTransition())
            ? mesh.mediumInterface
            : nullptr;

    if (params.pathRays) {
        // null for Tr rays...
        PathRayIndex rayIndex(optixGetLaunchIndex().x);
        params.pathRays->SetTMax(rayIndex, optixGetRayTmax());
    }
}

extern "C" __global__ void __anyhit__triangle() {
    const TriangleMeshRecord &mesh =
        *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (alphaKilled(mesh))
        optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__shadowTriangle() {
    const TriangleMeshRecord &mesh =
        *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    if (mesh.material && mesh.material.IsTransparent())
        optixIgnoreIntersection();

    if (alphaKilled(mesh))
        optixIgnoreIntersection();
}

///////////////////////////////////////////////////////////////////////////
// Shadows

extern "C" __global__ void __raygen__shadow() {
    ShadowRayIndex rayIndex(optixGetLaunchIndex().x);

    if (rayIndex >= params.shadowRays->Size())
        return;

    uint32_t missed = 0;

    Float tMax;
    Ray ray = params.shadowRays->GetRay(rayIndex, &tMax);

    optixTrace(params.traversable, make_float3(ray.o.x, ray.o.y, ray.o.z),
               make_float3(ray.d.x, ray.d.y, ray.d.z),
               1e-5f,  // tmin
               tMax,
               0.0f,  // rayTime
               OptixVisibilityMask(255),
               /* TODO: if no alpha mapped stuff, pass:
               OPTIX_RAY_FLAG_DISABLE_ANYHIT
               | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
               | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, */
               OPTIX_RAY_FLAG_NONE, 0, /* ray type */
               1,                      /* total number of ray types */
               0,                      /* missSBTIndex */
               missed);

    if (!missed)
        params.shadowRayLd[rayIndex] = SampledSpectrum(0.);
    else
        params.shadowRayLd[rayIndex] /= (params.shadowRayPDFUni[rayIndex] +
                                         params.shadowRayPDFLight[rayIndex]).Average();
}

extern "C" __global__ void __miss__shadow() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __raygen__shadow_Tr() {
    ShadowRayIndex rayIndex(optixGetLaunchIndex().x);

    if (rayIndex >= params.shadowRays->Size())
        return;

    PixelIndex pixelIndex = params.shadowRayIndexToPixelIndex[rayIndex];
    SampledWavelengths lambda = params.lambda[pixelIndex];
    RNG &rng = *params.rng[pixelIndex];
    SampledSpectrum &Ld = params.shadowRayLd[rayIndex];
    SampledSpectrum pdfUni = params.shadowRayPDFUni[rayIndex];
    SampledSpectrum pdfLight = params.shadowRayPDFLight[rayIndex];

    SurfaceInteraction intr;
    uint32_t p0 = packPointer0(&intr), p1 = packPointer1(&intr);

    Float tMax;
    Ray ray = params.shadowRays->GetRay(rayIndex, &tMax);
    Point3f pLight = ray(tMax);

    while (true) {
        uint32_t missed = 0;

        optixTrace(params.traversable, make_float3(ray.o.x, ray.o.y, ray.o.z),
                   make_float3(ray.d.x, ray.d.y, ray.d.z),
                   1e-5f,  // tmin
                   tMax,
                   0.0f,                                             // rayTime
                   OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, /* ray type */
                   1, /* total number of ray types */
                   0, /* missSBTIndex */
                   p0, p1, missed);

        if (!missed && intr.material) {
            // Hit opaque surface
            Ld = SampledSpectrum(0.f);
            return;
        }

        if (ray.medium) {
            Float tEnd = missed ? tMax : (Distance(ray.o, intr.p()) / Length(ray.d));
            Point3f pExit = ray(tEnd);
            ray.d = pExit - ray.o;

            while (ray.o != pExit) {
                Float u = rng.Uniform<Float>();
                MediumSample mediumSample =
                    ray.medium.Sample_Tmaj(ray, 1.f, u, lambda, nullptr);
                if (!mediumSample.intr)
                    // FIXME: include last Tmaj?
                    break;

                const SampledSpectrum &Tmaj = mediumSample.Tmaj;
                const MediumInteraction &intr = *mediumSample.intr;
                SampledSpectrum sigma_n = intr.sigma_n();

                // ratio-tracking: only evaluate null scattering
                Ld *= Tmaj * sigma_n;
                pdfLight *= Tmaj * intr.sigma_maj;
                pdfUni *= Tmaj * sigma_n;

                if (!Ld)
                    return;

                ray = intr.SpawnRayTo(pExit);

                if (Ld.MaxComponentValue() > 0x1p24f ||
                    pdfLight.MaxComponentValue() > 0x1p24f ||
                    pdfUni.MaxComponentValue() > 0x1p24f) {
                    Ld *= 1.f / 0x1p24f;
                    pdfLight *= 1.f / 0x1p24f;
                    pdfUni *= 1.f / 0x1p24f;
                }
            }
        }

        if (missed)
            // done
            break;

        ray = intr.SpawnRayTo(pLight);

        if (ray.d == Vector3f(0, 0, 0))
            break;
    }

    Ld /= (pdfUni + pdfLight).Average();
}

extern "C" __global__ void __miss__shadow_Tr() {
    optixSetPayload_2(1);
}

/////////////////////////////////////////////////////////////////////////////////////
// Quadric

static __device__ inline SurfaceInteraction getQuadricIntersection(
    const QuadricIntersection &si) {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    SurfaceInteraction intr;
    if (const Sphere *sphere = rec.shape.CastOrNullptr<Sphere>())
        intr = sphere->InteractionFromIntersection(si, wo, 0. /* time */);
    else if (const Cylinder *cylinder = rec.shape.CastOrNullptr<Cylinder>())
        intr = cylinder->InteractionFromIntersection(si, wo, 0. /* time */);
    else if (const Disk *disk = rec.shape.CastOrNullptr<Disk>())
        intr = disk->InteractionFromIntersection(si, wo, 0. /* time */);
    else
        assert(!"unexpected quadric");

    return intr;
}

extern "C" __global__ void __closesthit__quadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    QuadricIntersection qi;
    qi.pObj =
        Point3f(BitsToFloat(optixGetAttribute_0()), BitsToFloat(optixGetAttribute_1()),
                BitsToFloat(optixGetAttribute_2()));
    qi.phi = BitsToFloat(optixGetAttribute_3());

    SurfaceInteraction intr = getQuadricIntersection(qi);

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
    getPayload<SurfaceInteraction>()->areaLight = rec.areaLight;
    getPayload<SurfaceInteraction>()->mediumInterface =
        (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
            ? rec.mediumInterface
            : nullptr;

    if (params.pathRays) {
        // it's null for Tr rays...
        PathRayIndex rayIndex(optixGetLaunchIndex().x);
        params.pathRays->SetTMax(rayIndex, optixGetRayTmax());
    }
}

extern "C" __global__ void __anyhit__shadowQuadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    if (rec.material && rec.material.IsTransparent())
        optixIgnoreIntersection();
}

extern "C" __global__ void __intersection__quadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    float3 org = optixGetObjectRayOrigin();
    float3 dir = optixGetObjectRayDirection();
    Float tMax = optixGetRayTmax();
    Ray ray(Point3f(org.x, org.y, org.z), Vector3f(dir.x, dir.y, dir.z));
    pstd::optional<QuadricIntersection> isect;

    if (const Sphere *sphere = rec.shape.CastOrNullptr<Sphere>())
        isect = sphere->BasicIntersect(ray, tMax);
    else if (const Cylinder *cylinder = rec.shape.CastOrNullptr<Cylinder>())
        isect = cylinder->BasicIntersect(ray, tMax);
    else if (const Disk *disk = rec.shape.CastOrNullptr<Disk>())
        isect = disk->BasicIntersect(ray, tMax);

    if (isect) {
        if (rec.alphaTexture) {
            SurfaceInteraction intr = getQuadricIntersection(*isect);

            BasicTextureEvaluator eval;
            TextureEvalContext ctx(intr);
            Float alpha = eval(rec.alphaTexture, ctx);
            if (alpha == 0)
                // No hit
                return;
        }

        optixReportIntersection(isect->tHit, 0 /* hit kind */, FloatToBits(isect->pObj.x),
                                FloatToBits(isect->pObj.y), FloatToBits(isect->pObj.z),
                                FloatToBits(isect->phi));
    }
}

///////////////////////////////////////////////////////////////////////////
// Bilinear patches

static __forceinline__ __device__ SurfaceInteraction
getBilinearPatchIntersection(Point2f uv) {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    float3 rd = optixGetWorldRayDirection();
    Vector3f wo = -Vector3f(rd.x, rd.y, rd.z);

    return BilinearPatch::InteractionFromIntersection(rec.mesh, optixGetPrimitiveIndex(),
                                                      uv, 0.f /* time */, wo);
}

extern "C" __global__ void __closesthit__bilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    Point2f uv(BitsToFloat(optixGetAttribute_0()), BitsToFloat(optixGetAttribute_1()));
    SurfaceInteraction intr = getBilinearPatchIntersection(uv);

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
    getPayload<SurfaceInteraction>()->areaLight =
        rec.areaLights ? rec.areaLights[optixGetPrimitiveIndex()] : nullptr;
    getPayload<SurfaceInteraction>()->mediumInterface =
        (rec.mediumInterface && rec.mediumInterface->IsMediumTransition())
            ? rec.mediumInterface
            : nullptr;

    if (params.pathRays) {
        // it's null for Tr rays...
        PathRayIndex rayIndex(optixGetLaunchIndex().x);
        params.pathRays->SetTMax(rayIndex, optixGetRayTmax());
    }
}

extern "C" __global__ void __anyhit__shadowBilinearPatch() {
    BilinearMeshRecord &rec = *((BilinearMeshRecord *)optixGetSbtDataPointer());

    if (rec.material && rec.material.IsTransparent())
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

    if (isect) {
        if (rec.alphaTexture) {
            SurfaceInteraction intr = getBilinearPatchIntersection(isect->uv);
            BasicTextureEvaluator eval;
            TextureEvalContext ctx(intr);
            Float alpha = eval(rec.alphaTexture, ctx);
            if (alpha == 0)
                // No intersection
                return;
        }

        optixReportIntersection(isect->t, 0 /* hit kind */, FloatToBits(isect->uv[0]),
                                FloatToBits(isect->uv[1]));
    }
}

///////////////////////////////////////////////////////////////////////////
// Random hit

extern "C" __global__ void __raygen__randomHit() {
    // Keep as uint32_t so can pass directly to optixTrace.
    uint32_t rayIndex = optixGetLaunchIndex().x;

    if (rayIndex >= params.randomHitRays->Size())
        return;

    Float tMax;
    Ray ray = params.randomHitRays->GetRay(SSRayIndex(rayIndex), &tMax);

    optixTrace(params.traversable, make_float3(ray.o.x, ray.o.y, ray.o.z),
               make_float3(ray.d.x, ray.d.y, ray.d.z),
               0.f,  // tmin
               tMax,
               0.0f,                                             // rayTime
               OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, /* ray type */
               1, /* total number of ray types */
               0 /* missSBTIndex */, rayIndex);
}

extern "C" __global__ void __anyhit__randomHitTriangle() {
    const TriangleMeshRecord &rec = *(const TriangleMeshRecord *)optixGetSbtDataPointer();

    SSRayIndex rayIndex(optixGetPayload_0());
    if (rec.material == params.materials[rayIndex]) {
        // It's a candidate...
        params.reservoirSamplers[rayIndex].Add(
            [&] __device__() { return *getTriangleIntersection(); }, 1.f);
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__randomHitBilinearPatch() {
    BilinearMeshRecord &rec = *(BilinearMeshRecord *)optixGetSbtDataPointer();

    SSRayIndex rayIndex(optixGetPayload_0());
    if (rec.material == params.materials[rayIndex]) {
        params.reservoirSamplers[rayIndex].Add(
            [&] __device__() {
                Point2f uv(BitsToFloat(optixGetAttribute_0()),
                           BitsToFloat(optixGetAttribute_1()));
                return getBilinearPatchIntersection(uv);
            },
            1.f);
    }

    optixIgnoreIntersection();
}

extern "C" __global__ void __anyhit__randomHitQuadric() {
    QuadricRecord &rec = *((QuadricRecord *)optixGetSbtDataPointer());

    SSRayIndex rayIndex(optixGetPayload_0());
    if (rec.material == params.materials[rayIndex]) {
        params.reservoirSamplers[rayIndex].Add(
            [&] __device__() {
                QuadricIntersection qi;
                qi.pObj = Point3f(BitsToFloat(optixGetAttribute_0()),
                                  BitsToFloat(optixGetAttribute_1()),
                                  BitsToFloat(optixGetAttribute_2()));
                qi.phi = BitsToFloat(optixGetAttribute_3());

                return getQuadricIntersection(qi);
            },
            1.f);
    }

    optixIgnoreIntersection();
}
