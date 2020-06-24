
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


// shapes.cpp*
#include <pbrt/shapes.h>

#include <pbrt/base.h>
#include <pbrt/loopsubdiv.h>
#include <pbrt/plymesh.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/image.h>
#include <pbrt/interaction.h>
#include <pbrt/util/lowdiscrepancy.h>
#include <pbrt/util/math.h>
#include <pbrt/util/memory.h>
#include <pbrt/paramdict.h>
#include <pbrt/util/print.h>
#include <pbrt/util/profile.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/splines.h>
#include <pbrt/util/stats.h>

#include <rply/rply.h>

#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
#include <cuda.h>
#include <pbrt/gpu.h>
#endif

namespace pbrt {

// Sphere Method Definitions
Bounds3f Sphere::WorldBound() const {
    return (*worldFromObject)(Bounds3f(Point3f(-radius, -radius, zMin),
                                       Point3f(radius, radius, zMax)));
}

pstd::optional<ShapeSample> Sphere::Sample(const Point2f &u) const {
    Point3f pObj = Point3f(0, 0, 0) + radius * SampleUniformSphere(u);
    // Reproject _pObj_ to sphere surface and compute _pObjError_
    pObj *= radius / Distance(pObj, Point3f(0, 0, 0));
    Vector3f pObjError = gamma(5) * Abs((Vector3f)pObj);
    Point3fi pi = (*worldFromObject)(Point3fi(pObj, pObjError));

    Normal3f n = Normalize((*worldFromObject)(Normal3f(pObj.x, pObj.y, pObj.z)));
    if (reverseOrientation) n *= -1;

    return ShapeSample{Interaction(pi, n), 1 / Area()};
}

Float Sphere::SolidAngle(const Point3f &p, int nSamples) const {
    Point3f pCenter = (*worldFromObject)(Point3f(0, 0, 0));
    if (DistanceSquared(p, pCenter) <= radius * radius)
        return 4 * Pi;

    Float sinTheta2 = radius * radius / DistanceSquared(p, pCenter);
    if (sinTheta2 < 0.00068523f /* sin^2(1.5 deg) */)
        // 2 * Pi * sinTheta2 / 2;
        return Pi * sinTheta2;

    Float cosTheta = SafeSqrt(1 - sinTheta2);
    return 2 * Pi * (1 - cosTheta);
}

std::string Sphere::ToString() const {
    return StringPrintf("[ Sphere worldFromObject: %s objectFromWorld: %s reverseOrientation: %s "
                        "transformSwapsHandedness: %s radius: %f zMin: %f zMax: %f thetaMin: %f "
                        "thetaMax: %f phiMax: %f ]", *worldFromObject, *objectFromWorld,
                        reverseOrientation, transformSwapsHandedness, radius, zMin,
                        zMax, thetaMin, thetaMax, phiMax);
}

Sphere *Sphere::Create(const Transform *worldFromObject,
                       const Transform *objectFromWorld, bool reverseOrientation,
                       const ParameterDictionary &dict,
                       Allocator alloc) {
    Float radius = dict.GetOneFloat("radius", 1.f);
    Float zmin = dict.GetOneFloat("zmin", -radius);
    Float zmax = dict.GetOneFloat("zmax", radius);
    Float phimax = dict.GetOneFloat("phimax", 360.f);
    return alloc.new_object<Sphere>(worldFromObject, objectFromWorld, reverseOrientation,
                                    radius, zmin, zmax, phimax);
}


// Disk Method Definitions
Bounds3f Disk::WorldBound() const {
    return (*worldFromObject)(Bounds3f(Point3f(-radius, -radius, height),
                                       Point3f(radius, radius, height)));
}

DirectionCone Disk::NormalBounds() const {
    Normal3f n = (*worldFromObject)(Normal3f(0, 0, 1));
    return DirectionCone(Vector3f(n));
}

std::string Disk::ToString() const {
    return StringPrintf("[ Disk worldFromObject: %s objectFromWorld: %s reverseOrientation: %s "
                        "transformSwapsHandedness: %s height: %f radius: %f innerRadius: %f "
                        "phiMax: %f ]", *worldFromObject, *objectFromWorld, reverseOrientation,
                        transformSwapsHandedness, height, radius,
                        innerRadius, phiMax);
}

Disk *Disk::Create(const Transform *worldFromObject,
                   const Transform *objectFromWorld, bool reverseOrientation,
                   const ParameterDictionary &dict, Allocator alloc) {
    Float height = dict.GetOneFloat("height", 0.);
    Float radius = dict.GetOneFloat("radius", 1);
    Float innerRadius = dict.GetOneFloat("innerradius", 0);
    Float phimax = dict.GetOneFloat("phimax", 360);
    return alloc.new_object<Disk>(worldFromObject, objectFromWorld, reverseOrientation,
                             height, radius, innerRadius, phimax);
}

// Cylinder Method Definitions
Bounds3f Cylinder::WorldBound() const {
    return (*worldFromObject)(Bounds3f(Point3f(-radius, -radius, zMin),
                                       Point3f(radius, radius, zMax)));
}

std::string Cylinder::ToString() const {
    return StringPrintf("[ Cylinder worldFromObject: %s objectFromWorld: %s reverseOrientation: %s "
                        "transformSwapsHandedness: %s radius: %f zMin: %f zMax: %f "
                        "phiMax: %f ]", *worldFromObject, *objectFromWorld, reverseOrientation,
                        transformSwapsHandedness, radius, zMin, zMax,
                        phiMax);
}

Cylinder *Cylinder::Create(
    const Transform *worldFromObject,
    const Transform *objectFromWorld, bool reverseOrientation,
    const ParameterDictionary &dict,
    Allocator alloc) {
    Float radius = dict.GetOneFloat("radius", 1);
    Float zmin = dict.GetOneFloat("zmin", -1);
    Float zmax = dict.GetOneFloat("zmax", 1);
    Float phimax = dict.GetOneFloat("phimax", 360);
    return alloc.new_object<Cylinder>(worldFromObject, objectFromWorld, reverseOrientation,
                                      radius, zmin, zmax, phimax);
}

STAT_PIXEL_RATIO("Intersections/Ray-Triangle intersection tests", nTriHits, nTriTests);

// Triangle Local Definitions
std::string TriangleIntersection::ToString() const {
    return StringPrintf("[ TriangleIntersection b0: %f b1: %f b2: %f t: %f ]",
                        b0, b1, b2, t);
}

static void PlyErrorCallback(p_ply, const char *message) {
    Error("PLY writing error: %s", message);
}

pstd::vector<const TriangleMesh *> *TriangleMesh::allMeshes;
#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
__device__ pstd::vector<const TriangleMesh *> *allTriangleMeshesGPU;
#endif

void TriangleMesh::Init(Allocator alloc) {
    allMeshes = alloc.new_object<pstd::vector<const TriangleMesh *>>(alloc);
#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
    CUDA_CHECK(cudaMemcpyToSymbol(allTriangleMeshesGPU, &allMeshes, sizeof(allMeshes)));
#endif
}

std::string TriangleMesh::ToString() const {
    std::string np = "(nullptr)";
    return StringPrintf("[ TriangleMesh reverseOrientation: %s transformSwapsHandedness: %s "
                        "nTriangles: %d nVertices: %d vertexIndices: %s p: %s n: %s "
                        "s: %s uv: %s faceIndices: %s ]",
                        reverseOrientation, transformSwapsHandedness, nTriangles, nVertices,
                        vertexIndices ? StringPrintf("%s", pstd::MakeSpan(vertexIndices, 3 * nTriangles)) : np,
                        p ? StringPrintf("%s", pstd::MakeSpan(p, nVertices)) : nullptr,
                        n ? StringPrintf("%s", pstd::MakeSpan(n, nVertices)) : nullptr,
                        s ? StringPrintf("%s", pstd::MakeSpan(s, nVertices)) : nullptr,
                        uv ? StringPrintf("%s", pstd::MakeSpan(uv, nVertices)) : nullptr,
                        faceIndices ? StringPrintf("%s", pstd::MakeSpan(faceIndices, nTriangles)) : nullptr);
}

STAT_RATIO("Triangles/Triangles per mesh", nTris, nTriMeshes);

TriangleMesh::TriangleMesh(
    const Transform &worldFromObject, bool reverseOrientation,
    std::vector<int> indices, std::vector<Point3f> P,
    std::vector<Vector3f> S, std::vector<Normal3f> N, std::vector<Point2f> UV,
    std::vector<int> fIndices)
    : reverseOrientation(reverseOrientation),
      transformSwapsHandedness(worldFromObject.SwapsHandedness()),
      nTriangles(indices.size() / 3),
      nVertices(P.size()) {
    CHECK_EQ((indices.size() % 3), 0);
    ++nTriMeshes;
    nTris += nTriangles;

    // Make sure that we don't have too much stuff to be using integers to
    // index into things.
    CHECK_LE(P.size(), std::numeric_limits<int>::max());
    // We could be clever and check indices.size() / 3 if we were careful
    // to promote to a 64-bit int before multiplying by 3 when we look up
    // in the indices array...
    CHECK_LE(indices.size(), std::numeric_limits<int>::max());

    vertexIndices = ShapeHandle::indexBufferCache->LookupOrAdd(std::move(indices));

    triangleBytes += sizeof(*this);

    // Transform mesh vertices to world space
    for (Point3f &p : P)
        p = worldFromObject(p);
    p = ShapeHandle::pBufferCache->LookupOrAdd(std::move(P));

    // Copy _UV_, _N_, and _S_ vertex data, if present
    if (!UV.empty()) {
        CHECK_EQ(nVertices, UV.size());
        uv = ShapeHandle::uvBufferCache->LookupOrAdd(std::move(UV));
    }
    if (!N.empty()) {
        CHECK_EQ(nVertices, N.size());
        for (Normal3f &n : N)
            n = worldFromObject(n);
        n = ShapeHandle::nBufferCache->LookupOrAdd(std::move(N));
    }
    if (!S.empty()) {
        CHECK_EQ(nVertices, S.size());
        for (Vector3f &s : S)
            s = worldFromObject(s);
        s = ShapeHandle::sBufferCache->LookupOrAdd(std::move(S));
    }

    if (!fIndices.empty()) {
        CHECK_EQ(nTriangles, fIndices.size());
        faceIndices = ShapeHandle::faceIndexBufferCache->LookupOrAdd(std::move(fIndices));
    }
}

pstd::vector<ShapeHandle> TriangleMesh::CreateTriangles(Allocator alloc) {
    CHECK_LT(TriangleMesh::allMeshes->size(), 1 << 31);
    int meshIndex = int(TriangleMesh::allMeshes->size());
    allMeshes->push_back(this);

    pstd::vector<ShapeHandle> tris(nTriangles, alloc);
    Triangle *t = alloc.allocate_object<Triangle>(nTriangles);
    for (int i = 0; i < nTriangles; ++i) {
        alloc.construct(&t[i], meshIndex, i);
        tris[i] = &t[i];
    }

    return tris;
}

bool WritePlyFile(const std::string &filename, pstd::span<const int> vertexIndices,
                  pstd::span<const Point3f> P, pstd::span<const Vector3f> S,
                  pstd::span<const Normal3f> N, pstd::span<const Point2f> UV,
                  pstd::span<const int> faceIndices) {
    p_ply plyFile =
        ply_create(filename.c_str(), PLY_DEFAULT, PlyErrorCallback, 0, nullptr);
    if (plyFile == nullptr)
        return false;

    size_t nVertices = P.size();
    size_t nTriangles = vertexIndices.size() / 3;
    CHECK_EQ(vertexIndices.size() % 3, 0);

    ply_add_element(plyFile, "vertex", nVertices);
    ply_add_scalar_property(plyFile, "x", PLY_FLOAT);
    ply_add_scalar_property(plyFile, "y", PLY_FLOAT);
    ply_add_scalar_property(plyFile, "z", PLY_FLOAT);
    if (!N.empty()) {
        ply_add_scalar_property(plyFile, "nx", PLY_FLOAT);
        ply_add_scalar_property(plyFile, "ny", PLY_FLOAT);
        ply_add_scalar_property(plyFile, "nz", PLY_FLOAT);
    }
    if (!UV.empty()) {
        ply_add_scalar_property(plyFile, "u", PLY_FLOAT);
        ply_add_scalar_property(plyFile, "v", PLY_FLOAT);
    }
    if (!S.empty())
        Warning(R"(PLY mesh in "%s" will be missing tangent vectors "S".)",
                filename);

    ply_add_element(plyFile, "face", nTriangles);
    ply_add_list_property(plyFile, "vertex_indices", PLY_UINT8, PLY_INT);
    if (!faceIndices.empty())
        ply_add_scalar_property(plyFile, "face_indices", PLY_INT);

    ply_write_header(plyFile);

    for (int i = 0; i < nVertices; ++i) {
        ply_write(plyFile, P[i].x);
        ply_write(plyFile, P[i].y);
        ply_write(plyFile, P[i].z);
        if (!N.empty()) {
            ply_write(plyFile, N[i].x);
            ply_write(plyFile, N[i].y);
            ply_write(plyFile, N[i].z);
        }
        if (!UV.empty()) {
            ply_write(plyFile, UV[i].x);
            ply_write(plyFile, UV[i].y);
        }
    }

    for (int i = 0; i < nTriangles; ++i) {
        ply_write(plyFile, 3);
        ply_write(plyFile, vertexIndices[3 * i]);
        ply_write(plyFile, vertexIndices[3 * i + 1]);
        ply_write(plyFile, vertexIndices[3 * i + 2]);
        if (!faceIndices.empty())
            ply_write(plyFile, faceIndices[i]);
    }

    ply_close(plyFile);
    return true;
}

Bounds3f Triangle::WorldBound() const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];
    return Union(Bounds3f(p0, p1), p2);
}

pstd::optional<TriangleIntersection> Triangle::Intersect(const Ray &ray, Float tMax, const Point3f &p0,
                                                         const Point3f &p1, const Point3f &p2) {
    // Transform triangle vertices to ray coordinate space

    // Translate vertices based on ray origin
    Point3f p0t = p0 - Vector3f(ray.o);
    Point3f p1t = p1 - Vector3f(ray.o);
    Point3f p2t = p2 - Vector3f(ray.o);

    // Permute components of triangle vertices and ray direction
    int kz = MaxComponentIndex(Abs(ray.d));
    int kx = kz + 1;
    if (kx == 3) kx = 0;
    int ky = kx + 1;
    if (ky == 3) ky = 0;
    Vector3f d = Permute(ray.d, {kx, ky, kz});
    p0t = Permute(p0t, {kx, ky, kz});
    p1t = Permute(p1t, {kx, ky, kz});
    p2t = Permute(p2t, {kx, ky, kz});

    // Apply shear transformation to translated vertex positions
    Float Sx = -d.x / d.z;
    Float Sy = -d.y / d.z;
    Float Sz = 1.f / d.z;
    p0t.x += Sx * p0t.z;
    p0t.y += Sy * p0t.z;
    p1t.x += Sx * p1t.z;
    p1t.y += Sy * p1t.z;
    p2t.x += Sx * p2t.z;
    p2t.y += Sy * p2t.z;

    // Compute edge function coefficients _e0_, _e1_, and _e2_
    Float e0 = DifferenceOfProducts(p1t.x, p2t.y, p1t.y, p2t.x);
    Float e1 = DifferenceOfProducts(p2t.x, p0t.y, p2t.y, p0t.x);
    Float e2 = DifferenceOfProducts(p0t.x, p1t.y, p0t.y, p1t.x);

#ifndef __CUDA_ARCH__
    // Fall back to double precision test at triangle edges
    if (sizeof(Float) == sizeof(float) &&
        (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
        double p2txp1ty = (double)p2t.x * (double)p1t.y;
        double p2typ1tx = (double)p2t.y * (double)p1t.x;
        e0 = (float)(p2typ1tx - p2txp1ty);
        double p0txp2ty = (double)p0t.x * (double)p2t.y;
        double p0typ2tx = (double)p0t.y * (double)p2t.x;
        e1 = (float)(p0typ2tx - p0txp2ty);
        double p1txp0ty = (double)p1t.x * (double)p0t.y;
        double p1typ0tx = (double)p1t.y * (double)p0t.x;
        e2 = (float)(p1typ0tx - p1txp0ty);
    }
#endif

    // Perform triangle edge and determinant tests
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return {};
    Float det = e0 + e1 + e2;
    if (det == 0) return {};

    // Compute scaled hit distance to triangle and test against ray $t$ range
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < tMax * det))
        return {};
    else if (det > 0 && (tScaled <= 0 || tScaled > tMax * det))
        return {};

    // Compute barycentric coordinates and $t$ value for triangle intersection
    Float invDet = 1 / det;
    Float b0 = e0 * invDet;
    Float b1 = e1 * invDet;
    Float b2 = e2 * invDet;
    Float t = tScaled * invDet;

    // Ensure that computed triangle $t$ is conservatively greater than zero

    // Compute $\delta_z$ term for triangle $t$ error bounds
    Float maxZt = MaxComponentValue(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
    Float deltaZ = gamma(3) * maxZt;

    // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
    Float maxXt = MaxComponentValue(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
    Float maxYt = MaxComponentValue(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
    Float deltaX = gamma(5) * (maxXt + maxZt);
    Float deltaY = gamma(5) * (maxYt + maxZt);

    // Compute $\delta_e$ term for triangle $t$ error bounds
    Float deltaE =
        2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

    // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
    Float maxE = MaxComponentValue(Abs(Vector3f(e0, e1, e2)));
    Float deltaT = 3 *
                   (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
                   std::abs(invDet);
    if (t <= deltaT) return {};

    return TriangleIntersection{b0, b1, b2, t};
}

pstd::optional<ShapeIntersection> Triangle::Intersect(const Ray &ray, Float tMax) const {
    ProfilerScope p(ProfilePhase::TriIntersect);
    ++nTriTests;
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    // Perform ray--triangle intersection test

    pstd::optional<TriangleIntersection> triIsect = Intersect(ray, tMax, p0, p1, p2);
    if (!triIsect)
        return {};

    Float b0 = triIsect->b0, b1 = triIsect->b1, b2 = triIsect->b2;

    SurfaceInteraction intr = mesh->InteractionFromIntersection(triIndex, {b0, b1, b2},
                                                                ray.time, -ray.d);

    ++nTriHits;
    return ShapeIntersection{intr, triIsect->t};
}

bool Triangle::IntersectP(const Ray &ray, Float tMax) const {
    ProfilerScope p(ProfilePhase::TriIntersectP);
    ++nTriTests;
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    pstd::optional<TriangleIntersection> isect = Intersect(ray, tMax, p0, p1, p2);
    if (isect) {
        ++nTriHits;
        return true;
    } else
        return false;
}

Float Triangle::Area() const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];
    return 0.5f * Length(Cross(p1 - p0, p2 - p0));
}

pstd::optional<ShapeSample> Triangle::Sample(const Point2f &u) const {
    pstd::array<Float, 3> b = SampleUniformTriangle(u);
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    Point3f p = b[0] * p0 + b[1] * p1 + b[2] * p2;
    // Compute surface normal for sampled point on triangle
    Normal3f n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
    // Ensure correct orientation of the geometric normal; follow the same
    // approach as was used in Triangle::Intersect().
    if (mesh->n != nullptr) {
        Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                    (1 - b[0] - b[1]) * mesh->n[v[2]]);
        n = FaceForward(n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        n *= -1;

    // Compute error bounds for sampled point on triangle
    Point3f pAbsSum =
        Abs(b[0] * p0) + Abs(b[1] * p1) + Abs((1 - b[0] - b[1]) * p2);
    Vector3f pError = Vector3f(gamma(6) * pAbsSum);
    Point3fi pi = Point3fi(p, pError);

    return ShapeSample{Interaction(pi, n), 1 / Area()};
}

// The spherical sampling code has trouble with both very small and very
// large triangles (on the hemisphere); fall back to uniform area sampling
// in these cases. In the first case, there is presumably not a lot of
// contribution from the emitter due to its subtending a small solid angle.
// In the second, BSDF sampling should be the much better sampling strategy
// anyway.
static constexpr Float MinSphericalSampleArea = 1e-5;
static constexpr Float MaxSphericalSampleArea = 6.28;

// Note: much of this method---other than the call to the sampling function
// and the check about how to sample---is shared with the other
// Triangle::Sample() routine.
pstd::optional<ShapeSample> Triangle::Sample(const Interaction &ref, const Point2f &uo) const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    Float sa = SolidAngle(ref.p());
    if (sa < MinSphericalSampleArea || sa > MaxSphericalSampleArea) {
        // From Shape::Sample().
        pstd::optional<ShapeSample> ss = Sample(uo);
        if (!ss) return {};
        ss->intr.time = ref.time;
        Vector3f wi = ss->intr.p() - ref.p();
        if (LengthSquared(wi) == 0)
            return {};
        else {
            wi = Normalize(wi);
            // Convert from area measure, as returned by the Sample() call
            // above, to solid angle measure.
            ss->pdf *= DistanceSquared(ref.p(), ss->intr.p()) / AbsDot(ss->intr.n, -wi);
            if (std::isinf(ss->pdf)) return {};
        }
        return ss;
    }

    Float pdf = 1;
    Point2f u = uo;
    if (ref.IsSurfaceInteraction()) {
        Point3f rp = ref.p();
        Normal3f rnf = FaceForward(ref.AsSurface().shading.n, ref.wo);
        Vector3f wi[3] = { Normalize(p0 - rp), Normalize(p1 - rp),
                           Normalize(p2 - rp) };
        // (0,0) -> p1, (1,0) -> p1, (0,1) -> p0, (1,1) -> p2
        pstd::array<Float, 4> w;
        if (ref.AsSurface().bsdf && ref.AsSurface().bsdf->HasTransmission())
            w = pstd::array<Float, 4>{ std::max<Float>(0.01, AbsDot(rnf, wi[1])),
                                       std::max<Float>(0.01, AbsDot(rnf, wi[1])),
                                       std::max<Float>(0.01, AbsDot(rnf, wi[0])),
                                       std::max<Float>(0.01, AbsDot(rnf, wi[2])) };
        else
            w = pstd::array<Float, 4>{ std::max<Float>(0.01, Dot(rnf, wi[1])),
                                       std::max<Float>(0.01, Dot(rnf, wi[1])),
                                       std::max<Float>(0.01, Dot(rnf, wi[0])),
                                       std::max<Float>(0.01, Dot(rnf, wi[2])) };
        u = SampleBilinear(u, w);
        pdf *= BilinearPDF(u, w);
    }
    Float triPDF;
    pstd::array<Float, 3> b = SampleSphericalTriangle({p0, p1, p2}, ref.p(), u, &triPDF);
    if (triPDF == 0)
        return {};
    pdf *= triPDF;

    // Compute surface normal for sampled point on triangle
    Normal3f n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
    // Ensure correct orientation of the geometric normal; follow the same
    // approach as was used in Triangle::Intersect().
    if (mesh->n != nullptr) {
        Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                    b[2] * mesh->n[v[2]]);
        n = FaceForward(n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        n *= -1;

    // Compute error bounds for sampled point on triangle
    Point3f ps = b[0] * p0 + b[1] * p1 + b[2] * p2;
    Point3f pAbsSum =
        Abs(b[0] * p0) + Abs(b[1] * p1) + Abs(b[2] * p2);
    Vector3f pError = gamma(6) * Vector3f(pAbsSum.x, pAbsSum.y, pAbsSum.z);
    Point3fi pi = Point3fi(ps, pError);

    return ShapeSample{Interaction(pi, n, ref.time), pdf};
}

Float Triangle::PDF(const Interaction &ref, const Vector3f &wi) const {
    Float sa = SolidAngle(ref.p());
    if (sa < MinSphericalSampleArea || sa > MaxSphericalSampleArea) {
        // From Shape::PDF()
        // Intersect sample ray with area light geometry
        Ray ray = ref.SpawnRay(wi);
        pstd::optional<ShapeIntersection> si = Intersect(ray);
        if (!si) return 0;

        // Convert light sample weight to solid angle measure
        Float pdf = DistanceSquared(ref.p(), si->intr.p()) /
            (AbsDot(si->intr.n, -wi) * Area());
        if (std::isinf(pdf)) pdf = 0.f;
        return pdf;
    }

    if (!IntersectP(ref.SpawnRay(wi), Infinity))
        return 0;

    Float pdf = 1 / sa;
    if (ref.IsSurfaceInteraction()) {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        auto mesh = GetMesh();
        const int *v = &mesh->vertexIndices[3 * triIndex];
        const Point3f &p0 = mesh->p[v[0]];
        const Point3f &p1 = mesh->p[v[1]];
        const Point3f &p2 = mesh->p[v[2]];

        Point3f rp = ref.p();
        Normal3f rnf = FaceForward(ref.AsSurface().shading.n, ref.wo);
        Vector3f wit[3] = { Normalize(p0 - rp), Normalize(p1 - rp), Normalize(p2 - rp) };
        pstd::array<Float, 4> w;
        if (ref.AsSurface().bsdf->HasTransmission())
            w = pstd::array<Float, 4>{ std::max<Float>(0.01, AbsDot(rnf, wit[1])),
                                       std::max<Float>(0.01, AbsDot(rnf, wit[1])),
                                       std::max<Float>(0.01, AbsDot(rnf, wit[0])),
                                       std::max<Float>(0.01, AbsDot(rnf, wit[2])) };
        else
            w = pstd::array<Float, 4>{ std::max<Float>(0.01, Dot(rnf, wit[1])),
                                       std::max<Float>(0.01, Dot(rnf, wit[1])),
                                       std::max<Float>(0.01, Dot(rnf, wit[0])),
                                       std::max<Float>(0.01, Dot(rnf, wit[2])) };

        Point2f u = InvertSphericalTriangleSample({p0, p1, p2}, rp, wi);
        pdf *= BilinearPDF(u, w);
    }

    return pdf;
}

DirectionCone Triangle::NormalBounds() const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    // Compute surface normal for sampled point on triangle
    Normal3f n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
    // Ensure correct orientation of the geometric normal; follow the same
    // approach as was used in Triangle::Intersect().
    if (mesh->n != nullptr) {
        // TODO: um, can this be different at different points on the
        // triangle, and if so, what is the implication for NormalBounds()?
        Normal3f ns(mesh->n[v[0]] + mesh->n[v[1]] + mesh->n[v[2]]);
        n = FaceForward(n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        n *= -1;

    return DirectionCone(Vector3f(n));
}

std::string Triangle::ToString() const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    return StringPrintf("[ Triangle meshIndex: %d triIndex: %d -> p [ %s %s %s ] ]",
                        meshIndex, triIndex, p0, p1, p2);
}

TriangleMesh *TriangleMesh::Create(
    const Transform *worldFromObject, bool reverseOrientation,
    const ParameterDictionary &dict, Allocator alloc) {
    std::vector<int> vi = dict.GetIntArray("indices");
    std::vector<Point3f> P = dict.GetPoint3fArray("P");
    std::vector<Point2f> uvs = dict.GetPoint2fArray("uv");

    if (vi.empty()) {
        if (P.size() == 3)
            vi = { 0, 1, 2 };
        else {
            Error("Vertex indices \"indices\" must be provided with triangle mesh.");
            return {};
        }
    } else if ((vi.size() % 3) != 0u) {
        Error("Number of vertex indices %d not a multiple of 3. Discarding %d excess.",
              int(vi.size()), int(vi.size() % 3));
        while ((vi.size() % 3) != 0u) vi.pop_back();
    }

    if (P.empty()) {
        Error("Vertex positions \"P\" must be provided with triangle mesh.");
        return {};
    }

    if (!uvs.empty() && uvs.size() != P.size()) {
        Error("Number of \"uv\"s for triangle mesh must match \"P\"s. "
              "Discarding uvs.");
        uvs = {};
    }

    std::vector<Vector3f> S = dict.GetVector3fArray("S");
    if (!S.empty() && S.size() != P.size()) {
        Error("Number of \"S\"s for triangle mesh must match \"P\"s. "
              "Discarding \"S\"s.");
        S = {};
    }
    std::vector<Normal3f> N = dict.GetNormal3fArray("N");
    if (!N.empty() && N.size() != P.size()) {
        Error("Number of \"N\"s for triangle mesh must match \"P\"s. "
              "Discarding \"N\"s.");
        N = {};
    }

    for (size_t i = 0; i < vi.size(); ++i)
        if (vi[i] >= P.size()) {
            Error(
                "trianglemesh has out of-bounds vertex index %d (%d \"P\" "
                "values were given. Discarding this mesh.",
                vi[i], (int)P.size());
            return {};
        }

    std::vector<int> faceIndices = dict.GetIntArray("faceIndices");
    if (!faceIndices.empty() && faceIndices.size() != vi.size() / 3) {
        Error("Number of face indices %d does not match number of triangles %d. "
              "Discarding face indices.",
              int(faceIndices.size()), int(vi.size() / 3));
        faceIndices = {};
    }

    return alloc.new_object<TriangleMesh>(*worldFromObject, reverseOrientation,
                                          std::move(vi), std::move(P),
                                          std::move(S), std::move(N), std::move(uvs),
                                          std::move(faceIndices));
}


STAT_MEMORY_COUNTER("Memory/Curves", curveBytes);
STAT_PERCENT("Intersections/Ray-curve intersection tests", nCurveHits, nCurveTests);
STAT_INT_DISTRIBUTION("Intersections/Curve refinement level", refinementLevel);
STAT_COUNTER("Scene/Curves", nCurves);
STAT_COUNTER("Scene/Split curves", nSplitCurves);

// Curve Method Definitions
std::string ToString(CurveType type) {
    switch (type) {
    case CurveType::Flat: return "Flat";
    case CurveType::Cylinder: return "Cylinder";
    case CurveType::Ribbon: return "Ribbon";
    default: LOG_FATAL("Unhandled case"); return "";
    }
}

CurveCommon::CurveCommon(pstd::span<const Point3f> c, Float width0, Float width1,
                         CurveType type, pstd::span<const Normal3f> norm,
                         const Transform *worldFromObject,
                         const Transform *objectFromWorld,
                         bool reverseOrientation)
    : type(type),
      worldFromObject(worldFromObject),
      objectFromWorld(objectFromWorld),
      reverseOrientation(reverseOrientation),
      transformSwapsHandedness(worldFromObject->SwapsHandedness()) {
    width[0] = width0;
    width[1] = width1;
    CHECK_EQ(c.size(), 4);
    for (int i = 0; i < 4; ++i) cpObj[i] = c[i];
    if (norm.size() == 2) {
        n[0] = Normalize(norm[0]);
        n[1] = Normalize(norm[1]);
        normalAngle = AngleBetween(n[0], n[1]);
        invSinNormalAngle = 1 / std::sin(normalAngle);
    }
    ++nCurves;
}

std::string CurveCommon::ToString() const {
    return StringPrintf("[ CurveCommon type: %s cpObj: %s width: %s n: %s normalAngle: %f "
                        "invSinNormalAngle: %f worldFromObject: %s objectFromWorld: %s "
                        "reverseOrientation: %s transformSwapsHandedness: %s ]",
                        type, pstd::MakeSpan(cpObj), pstd::MakeSpan(width),
                        pstd::MakeSpan(n), normalAngle, invSinNormalAngle, *worldFromObject,
                        *objectFromWorld, reverseOrientation, transformSwapsHandedness);
}

pstd::vector<ShapeHandle> CreateCurve(
    const Transform *worldFromObject,
    const Transform *objectFromWorld, bool reverseOrientation,
    pstd::span<const Point3f> c, Float w0, Float w1, CurveType type,
    pstd::span<const Normal3f> norm, int splitDepth, Allocator alloc) {
    CurveCommon *common =
        alloc.new_object<CurveCommon>(c, w0, w1, type, norm, worldFromObject,
                                      objectFromWorld, reverseOrientation);

    const int nSegments = 1 << splitDepth;
    pstd::vector<ShapeHandle> segments(nSegments, alloc);
    Curve *curves = alloc.allocate_object<Curve>(nSegments);
    for (int i = 0; i < nSegments; ++i) {
        Float uMin = i / (Float)nSegments;
        Float uMax = (i + 1) / (Float)nSegments;
        alloc.construct(&curves[i], common, uMin, uMax);
        segments[i] = &curves[i];
        ++nSplitCurves;
    }

    curveBytes += sizeof(CurveCommon) + nSegments * sizeof(Curve);
    return segments;
}

Bounds3f Curve::WorldBound() const {
    Bounds3f b = BoundCubicBezier<Bounds3f>(pstd::MakeConstSpan(common->cpObj), uMin, uMax);
    Float width[2] = {Lerp(uMin, common->width[0], common->width[1]),
                      Lerp(uMax, common->width[0], common->width[1])};
    return (*common->worldFromObject)(
        Expand(b, std::max(width[0], width[1]) * 0.5f));
}

pstd::optional<ShapeIntersection> Curve::Intersect(const Ray &ray, Float tMax) const {
    ProfilerScope p(ProfilePhase::CurveIntersect);
    pstd::optional<ShapeIntersection> si;
    bool hit = intersect(ray, tMax, &si);
    if (hit) DCHECK(si.has_value());
    else DCHECK(!si.has_value());
    return si;
}

bool Curve::IntersectP(const Ray &ray, Float tMax) const {
    ProfilerScope p(ProfilePhase::CurveIntersectP);
    return intersect(ray, tMax, nullptr);
}

bool Curve::intersect(const Ray &r, Float tMax, pstd::optional<ShapeIntersection> *si) const {
    ++nCurveTests;
    // Transform _Ray_ to object space
    Point3fi oi = (*common->objectFromWorld)(Point3fi(r.o));
    Vector3fi di = (*common->objectFromWorld)(Vector3fi(r.d));
    Ray ray(Point3f(oi), Vector3f(di), r.time, r.medium);

    // Compute object-space control points for curve segment, _cpObj_
    pstd::array<Point3f, 4> cpObj =
        CubicBezierControlPoints(pstd::MakeConstSpan(common->cpObj), uMin, uMax);

    // Project curve control points to plane perpendicular to ray

    // Be careful to set the "up" direction passed to LookAt() to equal the
    // vector from the first to the last control points.  In turn, this
    // helps orient the curve to be roughly parallel to the x axis in the
    // ray coordinate system.
    //
    // In turn (especially for curves that are approaching stright lines),
    // we get curve bounds with minimal extent in y, which in turn lets us
    // early out more quickly in recursiveIntersect().
    Vector3f dx = Cross(ray.d, cpObj[3] - cpObj[0]);
    if (LengthSquared(dx) == 0) {
        // If the ray and the vector between the first and last control
        // points are parallel, dx will be zero.  Generate an arbitrary xy
        // orientation for the ray coordinate system so that intersection
        // tests can proceeed in this unusual case.
        Vector3f dy;
        CoordinateSystem(ray.d, &dx, &dy);
    }

    Transform RayFromObject = LookAt(ray.o, ray.o + ray.d, dx);
    pstd::array<Point3f, 4> cp = { RayFromObject(cpObj[0]), RayFromObject(cpObj[1]),
                                   RayFromObject(cpObj[2]), RayFromObject(cpObj[3]) };

    // Before going any further, see if the ray's bounding box intersects
    // the curve's bounding box. We start with the y dimension, since the y
    // extent is generally the smallest (and is often tiny) due to our
    // careful orientation of the ray coordinate system above.
    Float maxWidth = std::max(Lerp(uMin, common->width[0], common->width[1]),
                              Lerp(uMax, common->width[0], common->width[1]));
    if (std::max({cp[0].y, cp[1].y, cp[2].y, cp[3].y}) +
            0.5f * maxWidth < 0 ||
        std::min({cp[0].y, cp[1].y, cp[2].y, cp[3].y}) -
            0.5f * maxWidth > 0)
        return false;

    // Check for non-overlap in x.
    if (std::max({cp[0].x, cp[1].x, cp[2].x, cp[3].x}) +
            0.5f * maxWidth < 0 ||
        std::min({cp[0].x, cp[1].x, cp[2].x, cp[3].x}) -
            0.5f * maxWidth > 0)
        return false;

    // Check for non-overlap in z.
    Float rayLength = Length(ray.d);
    Float zMax = rayLength * tMax;
    if (std::max({cp[0].z, cp[1].z, cp[2].z, cp[3].z}) + 0.5f * maxWidth < 0 ||
        std::min({cp[0].z, cp[1].z, cp[2].z, cp[3].z}) - 0.5f * maxWidth > zMax)
        return false;

    // Compute refinement depth for curve, _maxDepth_
    Float L0 = 0;
    for (int i = 0; i < 2; ++i)
        L0 = std::max(
            L0, std::max(
                    std::max(std::abs(cp[i].x - 2 * cp[i + 1].x + cp[i + 2].x),
                             std::abs(cp[i].y - 2 * cp[i + 1].y + cp[i + 2].y)),
                    std::abs(cp[i].z - 2 * cp[i + 1].z + cp[i + 2].z)));

    Float eps =
        std::max(common->width[0], common->width[1]) * .05f;  // width / 20
    // Compute log base 4 by dividing log2 in half.
    int r0 = Log2Int(1.41421356237f * 6.f * L0 / (8.f * eps)) / 2;
    int maxDepth = Clamp(r0, 0, 10);
    ReportValue(refinementLevel, maxDepth);

    return recursiveIntersect(ray, tMax, pstd::MakeConstSpan(cp),
                              Inverse(RayFromObject), uMin, uMax, maxDepth, si);
}

bool Curve::recursiveIntersect(
        const Ray &ray, Float tMax, pstd::span<const Point3f> cp,
        const Transform &ObjectFromRay, Float u0, Float u1,
        int depth, pstd::optional<ShapeIntersection> *si) const {
    Float rayLength = Length(ray.d);

    if (depth > 0) {
        // Split curve segment into sub-segments and test for intersection
        pstd::array<Point3f, 7> cpSplit = SubdivideCubicBezier(cp);

        // For each of the two segments, see if the ray's bounding box
        // overlaps the segment before recursively checking for
        // intersection with it.
        Float u[3] = {u0, (u0 + u1) / 2.f, u1};
        for (int seg = 0; seg < 2; ++seg) {
            Float maxWidth =
                std::max(Lerp(u[seg], common->width[0], common->width[1]),
                         Lerp(u[seg + 1], common->width[0], common->width[1]));

            // As above, check y first, since it most commonly lets us exit
            // out early.
            pstd::span<const Point3f> cps = pstd::MakeConstSpan(&cpSplit[3 * seg], 4);
            if (std::max({cps[0].y, cps[1].y, cps[2].y, cps[3].y}) +
                        0.5f * maxWidth < 0 ||
                std::min({cps[0].y, cps[1].y, cps[2].y, cps[3].y}) -
                        0.5f * maxWidth > 0)
                continue;

            if (std::max({cps[0].x, cps[1].x, cps[2].x, cps[3].x}) +
                        0.5f * maxWidth < 0 ||
                std::min({cps[0].x, cps[1].x, cps[2].x, cps[3].x}) -
                        0.5f * maxWidth > 0)
                continue;

            Float zMax = rayLength * ((si && *si) ? (*si)->tHit : tMax);
            if (std::max({cps[0].z, cps[1].z, cps[2].z, cps[3].z}) +
                        0.5f * maxWidth < 0 ||
                std::min({cps[0].z, cps[1].z, cps[2].z, cps[3].z}) -
                        0.5f * maxWidth > zMax)
                continue;

            bool hit = recursiveIntersect(ray, tMax, cps, ObjectFromRay,
                                          u[seg], u[seg + 1], depth - 1, si);
            // If we found an intersection and this is a shadow ray,
            // we can exit out immediately.
            if (hit && si == nullptr) return true;
        }
        return si ? si->has_value() : false;
    } else {
        // Intersect ray with curve segment

        // Test ray against segment endpoint boundaries

        // Test sample point against tangent perpendicular at curve start
        Float edge =
            (cp[1].y - cp[0].y) * -cp[0].y + cp[0].x * (cp[0].x - cp[1].x);
        if (edge < 0) return false;

        // Test sample point against tangent perpendicular at curve end
        edge = (cp[2].y - cp[3].y) * -cp[3].y + cp[3].x * (cp[3].x - cp[2].x);
        if (edge < 0) return false;

        // Compute line $w$ that gives minimum distance to sample point
        Vector2f segmentDirection = Point2f(cp[3].x, cp[3].y) - Point2f(cp[0].x, cp[0].y);
        Float denom = LengthSquared(segmentDirection);
        if (denom == 0) return false;
        Float w = Dot(-Vector2f(cp[0].x, cp[0].y), segmentDirection) / denom;

        // Compute $u$ coordinate of curve intersection point and _hitWidth_
        Float u = Clamp(Lerp(w, u0, u1), u0, u1);
        Float hitWidth = Lerp(u, common->width[0], common->width[1]);
        Normal3f nHit;
        if (common->type == CurveType::Ribbon) {
            // Scale _hitWidth_ based on ribbon orientation
            Float sin0 = std::sin((1 - u) * common->normalAngle) *
                         common->invSinNormalAngle;
            Float sin1 =
                std::sin(u * common->normalAngle) * common->invSinNormalAngle;
            nHit = sin0 * common->n[0] + sin1 * common->n[1];
            hitWidth *= AbsDot(nHit, ray.d) / rayLength;
        }

        // Test intersection point against curve width
        Vector3f dpcdw;
        Point3f pc = EvaluateCubicBezier(pstd::MakeConstSpan(cp), Clamp(w, 0, 1), &dpcdw);
        Float ptCurveDist2 = pc.x * pc.x + pc.y * pc.y;
        if (ptCurveDist2 > hitWidth * hitWidth * .25) return false;
        Float zMax = rayLength * tMax;
        if (pc.z < 0 || pc.z > zMax) return false;

        // Compute hit _t_ and partial derivatives for curve intersection
        if (si != nullptr) {
            // Compute $v$ coordinate of curve intersection point
            Float ptCurveDist = std::sqrt(ptCurveDist2);
            Float edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y;
            Float v = (edgeFunc > 0) ? 0.5f + ptCurveDist / hitWidth
                : 0.5f - ptCurveDist / hitWidth;

            // FIXME: this tHit isn't quite right for ribbons...
            Float tHit = pc.z / rayLength;
            DCHECK_LT(tHit, 1.0001 * tMax);
            if (*si) DCHECK_LT(tHit, 1.001 * (*si)->tHit); // ???
            // Compute error bounds for curve intersection
            Vector3f pError(2 * hitWidth, 2 * hitWidth, 2 * hitWidth);

            // Compute $\dpdu$ and $\dpdv$ for curve intersection
            Vector3f dpdu, dpdv;
            EvaluateCubicBezier(pstd::MakeConstSpan(common->cpObj), u, &dpdu);
            CHECK_NE(Vector3f(0, 0, 0), dpdu);

            if (common->type == CurveType::Ribbon)
                dpdv = Normalize(Cross(nHit, dpdu)) * hitWidth;
            else {
                // Compute curve $\dpdv$ for flat and cylinder curves
                Vector3f dpduPlane = ObjectFromRay.ApplyInverse(dpdu);
                Vector3f dpdvPlane =
                    Normalize(Vector3f(-dpduPlane.y, dpduPlane.x, 0)) *
                    hitWidth;
                if (common->type == CurveType::Cylinder) {
                    // Rotate _dpdvPlane_ to give cylindrical appearance
                    Float theta = Lerp(v, -90., 90.);
                    Transform rot = Rotate(-theta, dpduPlane);
                    dpdvPlane = rot(dpdvPlane);
                }
                dpdv = ObjectFromRay(dpdvPlane);
            }
            Point3f pHit = ray(tHit);
            Point3fi pe(pHit, pError);
            *si = {{(*common->worldFromObject)(SurfaceInteraction(
                pe, Point2f(u, v), -ray.d, dpdu, dpdv,
                Normal3f(0, 0, 0), Normal3f(0, 0, 0), ray.time, OrientationIsReversed() ^ TransformSwapsHandedness())),
                    tHit}};
        }
        ++nCurveHits;
        return true;
    }
}

Float Curve::Area() const {
    pstd::array<Point3f, 4> cpObj =
        CubicBezierControlPoints(pstd::MakeConstSpan(common->cpObj), uMin, uMax);
    Float width0 = Lerp(uMin, common->width[0], common->width[1]);
    Float width1 = Lerp(uMax, common->width[0], common->width[1]);
    Float avgWidth = (width0 + width1) * 0.5f;
    Float approxLength = 0.f;
    for (int i = 0; i < 3; ++i)
        approxLength += Distance(cpObj[i], cpObj[i + 1]);
    return approxLength * avgWidth;
}

pstd::optional<ShapeSample> Curve::Sample(const Point2f &u) const {
    LOG_FATAL("Curve::Sample not implemented.");
    return {};
}

Float Curve::PDF(const Interaction &) const {
    LOG_FATAL("Curve::PDF not implemented.");
    return {};
}

pstd::optional<ShapeSample> Curve::Sample(const Interaction &ref,
                                          const Point2f &u) const {
    LOG_FATAL("Curve::Sample not implemented.");
    return {};
}

Float Curve::PDF(const Interaction &ref, const Vector3f &wi) const {
    LOG_FATAL("Curve::PDF not implemented.");
    return {};
}

std::string Curve::ToString() const {
    return StringPrintf("[ Curve common: %s uMin: %f uMax: %f ]", *common,
                        uMin, uMax);
}

pstd::vector<ShapeHandle> Curve::Create(const Transform *worldFromObject,
                                        const Transform *objectFromWorld,
                                        bool reverseOrientation,
                                        const ParameterDictionary &dict,
                                        Allocator alloc) {
    Float width = dict.GetOneFloat("width", 1.f);
    Float width0 = dict.GetOneFloat("width0", width);
    Float width1 = dict.GetOneFloat("width1", width);

    int degree = dict.GetOneInt("degree", 3);
    if (degree != 2 && degree != 3) {
        Error("Invalid degree %d: only degree 2 and 3 curves are supported.",
              degree);
        return {};
    }

    std::string basis = dict.GetOneString("basis", "bezier");
    if (basis != "bezier" && basis != "bspline") {
        Error("Invalid basis \"%s\": only \"bezier\" and \"bspline\" are "
              "supported.", basis);
        return {};
    }

    int nSegments;
    std::vector<Point3f> cp = dict.GetPoint3fArray("P");
    if (basis == "bezier") {
        // After the first segment, which uses degree+1 control points,
        // subsequent segments reuse the last control point of the previous
        // one and then use degree more control points.
        if (((cp.size() - 1 - degree) % degree) != 0) {
            Error("Invalid number of control points %d: for the degree %d "
                  "Bezier basis %d + n * %d are required, for n >= 0.",
                  (int)cp.size(), degree, degree + 1, degree);
            return {};
        }
        nSegments = (cp.size() - 1) / degree;
    } else {
        if (cp.size() < degree + 1) {
            Error("Invalid number of control points %d: for the degree %d "
                  "b-spline basis, must have >= %d.", int(cp.size()), degree,
                  degree + 1);
            return {};
        }
        nSegments = cp.size() - degree;
    }


    CurveType type;
    std::string curveType = dict.GetOneString("type", "flat");
    if (curveType == "flat")
        type = CurveType::Flat;
    else if (curveType == "ribbon")
        type = CurveType::Ribbon;
    else if (curveType == "cylinder")
        type = CurveType::Cylinder;
    else {
        Error(R"(Unknown curve type "%s".  Using "cylinder".)", curveType);
        type = CurveType::Cylinder;
    }

    std::vector<Normal3f> n = dict.GetNormal3fArray("N");
    if (!n.empty()) {
        if (type != CurveType::Ribbon) {
            Warning("Curve normals are only used with \"ribbon\" type curves.");
            n = {};
        } else if (n.size() != nSegments + 1) {
            Error(
                "Invalid number of normals %d: must provide %d normals for ribbon "
                "curves with %d segments.", int(n.size()), nSegments + 1, nSegments);
            return {};
        }
    } else if (type == CurveType::Ribbon) {
        Error(
            "Must provide normals \"N\" at curve endpoints with ribbon "
            "curves.");
        return {};
    }

    int sd = dict.GetOneInt("splitdepth", 3);

    if (type == CurveType::Ribbon && n.empty()) {
        Error(
            "Must provide normals \"N\" at curve endpoints with ribbon "
            "curves.");
        return {};
    }

    pstd::vector<ShapeHandle> curves(alloc);
    // Pointer to the first control point for the current segment. This is
    // updated after each loop iteration depending on the current basis.
    int cpOffset = 0;
    for (int seg = 0; seg < nSegments; ++seg) {
        pstd::array<Point3f, 4> segCpBezier;

        // First, compute the cubic Bezier control points for the current
        // segment and store them in segCpBezier. (It is admittedly
        // wasteful storage-wise to turn b-splines into Bezier segments and
        // wasteful computationally to turn quadratic curves into cubics,
        // but yolo.)
        if (basis == "bezier") {
            if (degree == 2) {
                // Elevate to degree 3.
                segCpBezier = ElevateQuadraticBezierToCubic(pstd::MakeConstSpan(cp).subspan(cpOffset, 3));
            } else {
                // All set.
                for (int i = 0; i < 4; ++i)
                    segCpBezier[i] = cp[cpOffset + i];
            }
            cpOffset += degree;
        } else {
            // Uniform b-spline.
            if (degree == 2) {
                pstd::array<Point3f, 3> bezCp = QuadraticBSplineToBezier(pstd::MakeConstSpan(cp).subspan(cpOffset, 3));
                segCpBezier = ElevateQuadraticBezierToCubic(pstd::MakeConstSpan(bezCp));
            } else {
                segCpBezier = CubicBSplineToBezier(pstd::MakeConstSpan(cp).subspan(cpOffset, 4));
            }
            ++cpOffset;
        }

        pstd::span<const Normal3f> nspan;
        if (!n.empty()) nspan = pstd::MakeSpan(&n[seg], 2);
        auto c = CreateCurve(worldFromObject, objectFromWorld, reverseOrientation,
                             segCpBezier,
                             Lerp(Float(seg) / Float(nSegments), width0, width1),
                             Lerp(Float(seg + 1) / Float(nSegments), width0, width1),
                             type, nspan, sd, alloc);
        curves.insert(curves.end(), c.begin(), c.end());
    }
    return curves;
}


STAT_PIXEL_RATIO("Intersections/Ray-bilinear patch intersection tests", nBLPHits, nBLPTests);
STAT_MEMORY_COUNTER("Memory/Bilinear patch indices", blpIndexBytes);
STAT_MEMORY_COUNTER("Memory/Bilinear patch vertex positions", blpPositionBytes);
STAT_MEMORY_COUNTER("Memory/Bilinear patch normals", blpNormalBytes);
STAT_MEMORY_COUNTER("Memory/Bilinear patch uvs", blpUVBytes);
STAT_MEMORY_COUNTER("Memory/Bilinear patch face indices", blpFaceIndexBytes);
STAT_RATIO("Bilinear patches/Patches per mesh", nBlps, nBilinearMeshes);
STAT_MEMORY_COUNTER("Memory/Bilinear patches", blpBytes);

// BilinearPatch Method Definitions
std::string BilinearIntersection::ToString() const {
    return StringPrintf("[ BilinearIntersection uv: %s t: %f", uv, t);
}

BilinearPatchMesh::BilinearPatchMesh(const Transform &worldFromObject, bool reverseOrientation,
                                     std::vector<int> indices, std::vector<Point3f> P,
                                     std::vector<Normal3f> N, std::vector<Point2f> UV,
                                     std::vector<int> fIndices, Distribution2D *imageDist)
    : reverseOrientation(reverseOrientation),
      transformSwapsHandedness(worldFromObject.SwapsHandedness()),
      nPatches(indices.size() / 4),
      nVertices(P.size()),
      imageDistribution(std::move(imageDist)) {
    CHECK_EQ((indices.size() % 4), 0);
    ++nBilinearMeshes;
    nBlps += nPatches;

    // Make sure that we don't have too much stuff to be using integers to
    // index into things.
    CHECK_LE(P.size(), std::numeric_limits<int>::max());
    CHECK_LE(indices.size(), std::numeric_limits<int>::max());

    vertexIndices = ShapeHandle::indexBufferCache->LookupOrAdd(std::move(indices));

    blpBytes += sizeof(*this);

    // Transform mesh vertices to world space
    for (Point3f &p : P)
        p = worldFromObject(p);
    p = ShapeHandle::pBufferCache->LookupOrAdd(std::move(P));

    // Copy _UV_ and _N_ vertex data, if present
    if (!UV.empty()) {
        CHECK_EQ(nVertices, UV.size());
        uv = ShapeHandle::uvBufferCache->LookupOrAdd(std::move(UV));
    }
    if (!N.empty()) {
        CHECK_EQ(nVertices, N.size());
        for (Normal3f &n : N)
            n = worldFromObject(n);
        n = ShapeHandle::nBufferCache->LookupOrAdd(std::move(N));
    }

    if (!fIndices.empty()) {
        CHECK_EQ(nPatches, fIndices.size());
        faceIndices = ShapeHandle::faceIndexBufferCache->LookupOrAdd(std::move(fIndices));
    }
}

std::string BilinearPatchMesh::ToString() const {
    std::string np = "(nullptr)";
    return StringPrintf("[ BilinearMatchMesh reverseOrientation: %s transformSwapsHandedness: %s "
                        "nPatches: %d nVertices: %d vertexIndices: %s p: %s n: %s "
                        "uv: %s faceIndices: %s ]",
                        reverseOrientation, transformSwapsHandedness, nPatches, nVertices,
                        vertexIndices ? StringPrintf("%s", pstd::MakeSpan(vertexIndices, 4 * nPatches)) : np,
                        p ? StringPrintf("%s", pstd::MakeSpan(p, nVertices)) : nullptr,
                        n ? StringPrintf("%s", pstd::MakeSpan(n, nVertices)) : nullptr,
                        uv ? StringPrintf("%s", pstd::MakeSpan(uv, nVertices)) : nullptr,
                        faceIndices ? StringPrintf("%s", pstd::MakeSpan(faceIndices, nPatches)) : nullptr);
}

pstd::vector<ShapeHandle> BilinearPatchMesh::Create(const Transform *worldFromObject,
                                                    bool reverseOrientation,
                                                    const ParameterDictionary &dict,
                                                    Allocator alloc) {
    std::vector<int> vertexIndices = dict.GetIntArray("indices");
    std::vector<Point3f> P = dict.GetPoint3fArray("P");
    std::vector<Point2f> uv = dict.GetPoint2fArray("uv");

    if (vertexIndices.empty()) {
        if (P.size() == 4)
            // single patch
            vertexIndices = { 0, 1, 2, 3 };
        else {
            Error("Vertex indices \"indices\" must be provided with bilinear patch mesh shape.");
            return {};
        }
    } else if ((vertexIndices.size() % 4) != 0u) {
        Error("Number of vertex indices %d not a multiple of 4. Discarding %d excess.",
              int(vertexIndices.size()), int(vertexIndices.size() % 4));
        while ((vertexIndices.size() % 4) != 0u) vertexIndices.pop_back();
    }

    if (P.empty()) {
        Error("Vertex positions \"P\" must be provided with bilinear patch mesh shape.");
        return {};
    }

    if (!uv.empty() && uv.size() != P.size()) {
        Error("Number of \"uv\"s for bilinear patch mesh must match \"P\"s. "
              "Discarding uvs.");
        uv = {};
    }

    std::vector<Normal3f> N = dict.GetNormal3fArray("N");
    if (!N.empty() && N.size() != P.size()) {
        Error("Number of \"N\"s for bilinear patch mesh must match \"P\"s. "
              "Discarding \"N\"s.");
        N = {};
    }

    for (size_t i = 0; i < vertexIndices.size(); ++i)
        if (vertexIndices[i] >= P.size()) {
            Error(
                "Bilinear patch mesh has out of-bounds vertex index %d (%d \"P\" "
                "values were given. Discarding this mesh.",
                vertexIndices[i], (int)P.size());
            return {};
        }

    std::vector<int> faceIndices = dict.GetIntArray("faceIndices");
    if (!faceIndices.empty() && faceIndices.size() != vertexIndices.size() / 4) {
        Error("Number of face indices %d does not match number of bilinear patches %d. "
              "Discarding face indices.",
              int(faceIndices.size()), int(vertexIndices.size() / 4));
        faceIndices = {};
    }

    // Grab this before the vertexIndices are std::moved...
    size_t nBlps = vertexIndices.size() / 4;

    std::string filename = ResolveFilename(dict.GetOneString("imagefile", ""));
    Distribution2D *imageDist = nullptr;
    if (!filename.empty()) {
        auto im = Image::Read(filename, alloc);
        CHECK(im);
        Bounds2f domain = Bounds2f(Point2f(0, 0), Point2f(1, 1));
        pstd::optional<ImageChannelDesc> desc = im->image.GetChannelDesc({ "R", "G", "B" });
        CHECK(desc);
        Array2D<Float> d = im->image.ComputeSamplingDistribution(*desc, 1, domain, Norm::LInfinity);
        imageDist = alloc.new_object<Distribution2D>(d, domain);
    }

    return Create(worldFromObject, reverseOrientation, std::move(vertexIndices), std::move(P),
                  std::move(N), std::move(uv), std::move(faceIndices), std::move(imageDist),
                  alloc);
}

pstd::vector<ShapeHandle>
BilinearPatchMesh::Create(const Transform *worldFromObject,
                          bool reverseOrientation,
                          std::vector<int> indices, std::vector<Point3f> p,
                          std::vector<Normal3f> n, std::vector<Point2f> uv,
                          std::vector<int> faceIndices,
                          Distribution2D *imageDist,
                          Allocator alloc) {
    CHECK_LT(allMeshes->size(), 1 << 31);
    int meshIndex = int(allMeshes->size());

    // Grab this before we std::move(indices)
    CHECK_EQ(0, indices.size() % 4);
    int nBlps = indices.size() / 4;

    // Create the mesh first so the BilinearPatch constructor can call GetMesh().
    BilinearPatchMesh *mesh = alloc.new_object<BilinearPatchMesh>(
        *worldFromObject, reverseOrientation, std::move(indices), std::move(p),
        std::move(n), std::move(uv), std::move(faceIndices), imageDist);
    allMeshes->push_back(mesh);

    pstd::vector<ShapeHandle> blps(nBlps, alloc);
    BilinearPatch *patches = alloc.allocate_object<BilinearPatch>(nBlps);
    for (int i = 0; i < nBlps; ++i) {
        alloc.construct(&patches[i], meshIndex, i);
        blps[i] = &patches[i];
    }

    return blps;
}

pstd::vector<const BilinearPatchMesh *> *BilinearPatchMesh::allMeshes;
#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
__device__ pstd::vector<const BilinearPatchMesh *> *allBilinearMeshesGPU;
#endif

void BilinearPatchMesh::Init(Allocator alloc) {
    allMeshes = alloc.new_object<pstd::vector<const BilinearPatchMesh *>>(alloc);
#if defined(PBRT_HAVE_OPTIX) && defined(__NVCC__)
    CUDA_CHECK(cudaMemcpyToSymbol(allBilinearMeshesGPU, &allMeshes, sizeof(allMeshes)));
#endif
}

BilinearPatch::BilinearPatch(int meshIndex, int blpIndex)
    : meshIndex(meshIndex), blpIndex(blpIndex) {
    blpBytes += sizeof(*this);

    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    if (IsQuad())
        area = Distance(p00, p01) * Distance(p00, p10);
    else {
        // FIXME: it would be good to skip this for flat patches, or to
        // be adaptive based on curvature in some manner
        constexpr int na = 8;
        Point3f p[na+1][na+1];
        for (int i = 0; i <= na; ++i) {
            Float u = Float(i) / Float(na);
            for (int j = 0; j <= na; ++j) {
                Float v = Float(j) / Float(na);
                p[i][j] = Lerp(u, Lerp(v, p00, p01), Lerp(v, p10, p11));
            }
        }

        area = 0;
        for (int i = 0; i < na; ++i)
            for (int j = 0; j < na; ++j)
                area += 0.5f * Length(Cross(p[i+1][j+1] - p[i][j],
                                           p[i+1][j] - p[i][j+1]));
#if 0
        fprintf(stderr, "area old a %f, old b %f, subsampled %f\n", 0.5 * Length(Cross(p11 - p00, p10 - p01)),
                0.5 * (Length(Cross(p01 - p00, p10 - p00)) +
                       Length(Cross(p01 - p11, p10 - p11))),
                area);
#endif
    }
}

Bounds3f BilinearPatch::WorldBound() const {
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    return Union(Bounds3f(p00, p01), Bounds3f(p10, p11));
}

pstd::optional<BilinearIntersection> BilinearPatch::Intersect(const Ray &ray, Float tMax,
                                                        const Point3f &p00, const Point3f &p10,
                                                        const Point3f &p01, const Point3f &p11) {
    ++nBLPTests;

    Vector3f qn = Cross(p10-p00, p01-p11);
    Vector3f e10 = p10 - p00; // p01------u--------p11
    Vector3f e11 = p11 - p10; // |                   |
    Vector3f e00 = p01 - p00; // v e00           e11 v
                              // |        e10        |
                              // p00------u--------p10
    Vector3f q00 = p00 - ray.o;
    Vector3f q10 = p10 - ray.o;
    Float a = Dot(Cross(q00, ray.d), e00); // the equation is
    Float c = Dot(qn, ray.d);              // a + b u + c u^2
    Float b = Dot(Cross(q10, ray.d), e11); // first compute a+b+c
    b -= a + c;                                    // and then b
    Float det = b*b - 4*a*c;
    if (det < 0) return {};
    det = std::sqrt(det);
    Float u1, u2;             // two roots (u parameter)
    Float t = tMax, u, v; // need solution for the smallest t > 0
    if (c == 0) {                        // if c == 0, it is a trapezoid
        u1  = -a/b; u2 = -1;              // and there is only one root
    } else {                             // (c != 0 in Stanford models)
        u1  = (-b - copysignf(det, b))/2; // numerically "stable" root
        u2  = a/u1;                       // Viete's formula for u1*u2
        u1 /= c;
    }
    if (0 <= u1 && u1 <= 1) {                // is it inside the patch?
        Vector3f pa = Lerp(u1, q00, q10);       // point on edge e10 (Figure 8.4)
        Vector3f pb = Lerp(u1, e00, e11);       // it is, actually, pb - pa
        Vector3f n  = Cross(ray.d, pb);
        det = Dot(n, n);
        n = Cross(n, pa);
        Float t1 = Dot(n, pb);
        Float v1 = Dot(n, ray.d);     // no need to check t1 < t
        if (t1 > 0 && 0 <= v1 && v1 <= det) { // if t1 > ray.tmax,
            t = t1/det; u = u1; v = v1/det;    // it will be rejected
        }                                     // in rtPotentialIntersection
    }
    if (0 <= u2 && u2 <= 1) {                // it is slightly different,
        Vector3f pa = Lerp(u2, q00, q10);       // since u1 might be good
        Vector3f pb = Lerp(u2, e00, e11);       // and we need 0 < t2 < t1
        Vector3f n  = Cross(ray.d, pb);
        det = Dot(n, n);
        n = Cross(n, pa);
        Float t2 = Dot(n, pb)/det;
        Float v2 = Dot(n, ray.d);
        if (0 <= v2 && v2 <= det && t > t2 && t2 > 0) {
            t = t2; u = u2; v = v2/det;
        }
    }

    // TODO: reject hits with sufficiently small t that we're not sure.

    if (t >= tMax)
        return {};

    ++nBLPHits;
    return BilinearIntersection{{u, v}, t};
}

pstd::optional<ShapeIntersection> BilinearPatch::Intersect(const Ray &ray, Float tMax) const {
    ProfilerScope _(ProfilePhase::ShapeIntersect);

    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    pstd::optional<BilinearIntersection> blpIsect =
        Intersect(ray, tMax, p00, p10, p01, p11);
    if (!blpIsect)
        return {};

    // Found a hit.
    Point2f uv = blpIsect->uv;
    Point3f pHit = Lerp(uv[0], Lerp(uv[1], p00, p01), Lerp(uv[1], p10, p11));

    Vector3f dpdu = Lerp(uv[1], p10, p11) - Lerp(uv[1], p00, p01);
    Vector3f dpdv = Lerp(uv[0], p01, p11) - Lerp(uv[0], p00, p10);

    // Interpolate texture coordinates, if provided
    if (mesh->uv != nullptr) {
        const Point2f &uv00 = mesh->uv[v[0]];
        const Point2f &uv10 = mesh->uv[v[1]];
        const Point2f &uv01 = mesh->uv[v[2]];
        const Point2f &uv11 = mesh->uv[v[3]];

        Float dsdu = -uv00[0] + uv10[0] + uv[1] * (uv00[0] - uv01[0] - uv10[0] + uv11[0]);
        Float dsdv = -uv00[0] + uv01[0] + uv[0] * (uv00[0] - uv01[0] - uv10[0] + uv11[0]);
        Float dtdu = -uv00[1] + uv10[1] + uv[1] * (uv00[1] - uv01[1] - uv10[1] + uv11[1]);
        Float dtdv = -uv00[1] + uv01[1] + uv[0] * (uv00[1] - uv01[1] - uv10[1] + uv11[1]);

        Float duds = std::abs(dsdu) < 1e-8f ? 0 : 1 / dsdu;
        Float dvds = std::abs(dsdv) < 1e-8f ? 0 : 1 / dsdv;
        Float dudt = std::abs(dtdu) < 1e-8f ? 0 : 1 / dtdu;
        Float dvdt = std::abs(dtdv) < 1e-8f ? 0 : 1 / dtdv;

        // actually this is st (and confusing)
        uv = Lerp(uv[0], Lerp(uv[1], uv00, uv01), Lerp(uv[1], uv10, uv11));

        // dpds = dpdu * duds + dpdv * dvds, etc
        // duds = 1/dsdu
        Vector3f dpds = dpdu * duds + dpdv * dvds;
        Vector3f dpdt = dpdu * dudt + dpdv * dvdt;

        // These end up as zero-vectors of the mapping is degenerate...
        if (Cross(dpds, dpdt) != Vector3f(0, 0, 0)) {
            dpdu = dpds;
            dpdv = dpdt;
        }
    }

    // Compute coefficients for fundamental forms
    Float E = Dot(dpdu, dpdu);
    Float F = Dot(dpdu, dpdv);
    Float G = Dot(dpdv, dpdv);
    Vector3f N = Normalize(Cross(dpdu, dpdv));
    Float e = 0;  // 2nd derivative d2p/du2 == 0
    Vector3f d2Pduv(p00.x - p01.x - p10.x + p11.x,
                    p00.y - p01.y - p10.y + p11.y,
                    p00.z - p01.z - p10.z + p11.z);
    Float f = Dot(N, d2Pduv);
    Float g = 0;  // samesies

    // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
    Float EGF2 = DifferenceOfProducts(E, G, F, F);
    Normal3f dndu, dndv;
    if (EGF2 != 0) {
        Float invEGF2 = 1 / EGF2;
        Normal3f dndu = Normal3f(DifferenceOfProducts(f, F, e, G) * invEGF2 * dpdu +
                                 DifferenceOfProducts(e, F, f, E) * invEGF2 * dpdv);
        Normal3f dndv = Normal3f(DifferenceOfProducts(g, F, f, G) * invEGF2 * dpdu +
                                 DifferenceOfProducts(f, F, g, E) * invEGF2 * dpdv);
    }

    // Two lerps; each is gamma(3). TODO: double check.
    Vector3f pError = gamma(6) *
        Vector3f(Max(Max(Abs(p00), Abs(p10)), Max(Abs(p01), Abs(p11))));

    // Initialize _SurfaceInteraction_ from parametric information
    int faceIndex = mesh->faceIndices != nullptr ? mesh->faceIndices[blpIndex] : 0;
    Point3fi pe(pHit, pError);
    SurfaceInteraction isect(pe, uv, -ray.d, dpdu, dpdv, dndu, dndv,
                             ray.time, OrientationIsReversed() ^ TransformSwapsHandedness(),
                             faceIndex);

    if (mesh->n != nullptr) {
        const Normal3f &n00 = mesh->n[v[0]];
        const Normal3f &n10 = mesh->n[v[1]];
        const Normal3f &n01 = mesh->n[v[2]];
        const Normal3f &n11 = mesh->n[v[3]];
        Normal3f dndu = Lerp(blpIsect->uv[1], n10, n11) - Lerp(blpIsect->uv[1], n00, n01);
        Normal3f dndv = Lerp(blpIsect->uv[0], n01, n11) - Lerp(blpIsect->uv[0], n00, n10);

        Normal3f ns = Lerp(blpIsect->uv[0],
                           Lerp(blpIsect->uv[1], n00, n01),
                           Lerp(blpIsect->uv[1], n10, n11));
        if (LengthSquared(ns) > 0) {
            ns = Normalize(ns);
            Normal3f n = Normal3f(Normalize(Cross(dpdu, dpdv)));
            Vector3f axis = Cross(Vector3f(n), Vector3f(ns));
            if (LengthSquared(axis) > .0001f) {
                axis = Normalize(axis);
                // The shading normal is different enough.
                //
                // Don't worry about if ns == -n; that, too, is handled
                // naturally by the following.
                //
                // Rotate dpdu and dpdv around the axis perpendicular to the
                // plane defined by n and ns by the angle between them -> their
                // dot product will equal ns.
                Float cosTheta = Dot(n, ns), sinTheta = SafeSqrt(1 - cosTheta * cosTheta);
                Transform r = Rotate(sinTheta, cosTheta, axis);
                Vector3f sdpdu = r(dpdu), sdpdv = r(dpdv);

                if (mesh->reverseOrientation) {
                    ns = -ns;
                    sdpdv = -sdpdv;
                }
                isect.SetShadingGeometry(ns, sdpdu, sdpdv, dndu, dndv, true);
            }
        }
    }

    return ShapeIntersection{isect, blpIsect->t};
}

bool BilinearPatch::IntersectP(const Ray &ray, Float tMax) const {
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    return Intersect(ray, tMax, p00, p10, p01, p11).has_value();
}

Float BilinearPatch::Area() const {
    return area;
}

bool BilinearPatch::IsQuad() const {
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    Point3f p11quad = p00 + (p10 - p00) + (p01 - p00);
    Float diag = std::max(Distance(p00, p11), Distance(p01, p11));
    return Distance(p11quad, p11) < .001f * diag;
}

pstd::optional<ShapeSample> BilinearPatch::Sample(const Interaction &ref,
                                            const Point2f &uo) const {
    if (IsQuad()) {
        // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
        auto mesh = GetMesh();
        const int *v = &mesh->vertexIndices[4 * blpIndex];
        const Point3f &p00 = mesh->p[v[0]];
        const Point3f &p10 = mesh->p[v[1]];
        const Point3f &p01 = mesh->p[v[2]];
        const Point3f &p11 = mesh->p[v[3]];

        Float pdf = 1;
        Point2f u = uo;
        if (mesh->imageDistribution) {
            if (u[0] < 0.5) {
                u[0] = std::min(2.f * u[0], OneMinusEpsilon);

                Float pdf;
                u = mesh->imageDistribution->SampleContinuous(u, &pdf);

                Point3f p = Lerp(u[1], Lerp(u[0], p00, p10), Lerp(u[0], p01, p11));
                Normal3f n = Normal3f(Normalize(Cross(p10 - p00, p01 - p00)));

                Interaction intr(p, n, ref.time, u);

                Vector3f wi = Normalize(p - ref.p());
                pdf *= DistanceSquared(ref.p(), p) / (area * AbsDot(n, wi));

                // MIS
                Vector3f v00 = Normalize(p00 - ref.p());
                Vector3f v10 = Normalize(p10 - ref.p());
                Vector3f v01 = Normalize(p01 - ref.p());
                Vector3f v11 = Normalize(p11 - ref.p());
                pdf = 0.5f * pdf + 0.5f / SphericalQuadArea(v00, v10, v11, v01); // note: arg ordering...
                return ShapeSample{intr, pdf};
            }
            else
                u[0] = std::min(2.f * (u[0] - 0.5f), OneMinusEpsilon);
                // and continue...
        } else if (ref.IsSurfaceInteraction()) {
            // Note: we either do MIS of image sampling + uniform spherical
            // -or- warp product cosine spherical, just to keep things
            // (somewhat) simple.
            Point3f rp = ref.p();
            Normal3f nf = FaceForward(ref.AsSurface().shading.n, ref.wo);
            pstd::array<Float, 4> w;
            if (ref.AsSurface().bsdf->HasTransmission())
                w = pstd::array<Float, 4>{ std::max<Float>(0.01, AbsDot(Normalize(p00 - rp), nf)),
                                           std::max<Float>(0.01, AbsDot(Normalize(p10 - rp), nf)),
                                           std::max<Float>(0.01, AbsDot(Normalize(p01 - rp), nf)),
                                           std::max<Float>(0.01, AbsDot(Normalize(p11 - rp), nf)) };
            else
                w = pstd::array<Float, 4>{ std::max<Float>(0.01, Dot(Normalize(p00 - rp), nf)),
                                           std::max<Float>(0.01, Dot(Normalize(p10 - rp), nf)),
                                           std::max<Float>(0.01, Dot(Normalize(p01 - rp), nf)),
                                           std::max<Float>(0.01, Dot(Normalize(p11 - rp), nf)) };

            u = SampleBilinear(u, w);
            pdf *= BilinearPDF(u, w);
        }

        Float quadPDF;
        Point3f p = SampleSphericalQuad(ref.p(), p00, p10 - p00, p01 - p00, u, &quadPDF);
        pdf *= quadPDF;
        Normal3f n = Normal3f(Normalize(Cross(p10 - p00, p01 - p00)));
        if (OrientationIsReversed()) n *= -1;
        Point2f uv(Dot(p - p00, p10 - p00) / DistanceSquared(p10, p00),
                   Dot(p - p00, p01 - p00) / DistanceSquared(p01, p00));

        if (mesh->imageDistribution) {
            Vector3f wi = Normalize(p - ref.p());
            Float imPDF = mesh->imageDistribution->ContinuousPDF(uv);
            imPDF *= DistanceSquared(ref.p(), p) / (area * AbsDot(n, wi));
            pdf = 0.5f * pdf + 0.5f * imPDF;
        }

        return ShapeSample{Interaction(p, n, ref.time, uv), pdf};
    }

    // From Shape::Sample().
    pstd::optional<ShapeSample> ss = Sample(uo);
    if (!ss) return {};
    ss->intr.time = ref.time;
    Vector3f wi = ss->intr.p() - ref.p();

    if (LengthSquared(wi) == 0)
        return {};
    else {
        wi = Normalize(wi);
        // Convert from area measure, as returned by the Sample() call
        // above, to solid angle measure.
        ss->pdf *= DistanceSquared(ref.p(), ss->intr.p()) / AbsDot(ss->intr.n, -wi);
        if (std::isinf(ss->pdf)) return {};
    }
    return ss;
}

Float BilinearPatch::PDF(const Interaction &ref, const Vector3f &wi) const {
    // From Shape::PDF()
    // Intersect sample ray with area light geometry
    Ray ray = ref.SpawnRay(wi);
    pstd::optional<ShapeIntersection> si = Intersect(ray);
    if (!si) return 0;

    if (IsQuad()) {
        // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
        auto mesh = GetMesh();
        const int *v = &mesh->vertexIndices[4 * blpIndex];
        const Point3f &p00 = mesh->p[v[0]];
        const Point3f &p10 = mesh->p[v[1]];
        const Point3f &p01 = mesh->p[v[2]];
        const Point3f &p11 = mesh->p[v[3]];

        Vector3f v00 = Normalize(p00 - ref.p());
        Vector3f v10 = Normalize(p10 - ref.p());
        Vector3f v01 = Normalize(p01 - ref.p());
        Vector3f v11 = Normalize(p11 - ref.p());
        Float pdf = 1 / SphericalQuadArea(v00, v10, v11, v01); // note: arg ordering...

        if (GetMesh()->imageDistribution) {
            Float imPDF = PDF(si->intr) * DistanceSquared(ref.p(), si->intr.p()) /
                AbsDot(si->intr.n, -wi);
            return 0.5f * pdf + 0.5f * imPDF;
        } else if (ref.IsSurfaceInteraction()) {
            Point3f rp = ref.p();
            Normal3f nf = FaceForward(ref.AsSurface().shading.n, ref.wo);
            pstd::array<Float, 4> w;
            if (ref.AsSurface().bsdf->HasTransmission())
                w = pstd::array<Float, 4>{ std::max<Float>(0.01, AbsDot(Normalize(p00 - rp), nf)),
                                           std::max<Float>(0.01, AbsDot(Normalize(p10 - rp), nf)),
                                           std::max<Float>(0.01, AbsDot(Normalize(p01 - rp), nf)),
                                           std::max<Float>(0.01, AbsDot(Normalize(p11 - rp), nf)) };
            else
                w = pstd::array<Float, 4>{ std::max<Float>(0.01, Dot(Normalize(p00 - rp), nf)),
                                           std::max<Float>(0.01, Dot(Normalize(p10 - rp), nf)),
                                           std::max<Float>(0.01, Dot(Normalize(p01 - rp), nf)),
                                           std::max<Float>(0.01, Dot(Normalize(p11 - rp), nf)) };

            Point2f u = InvertSphericalQuadSample(rp, p00, p10 - p00, p01 - p00, si->intr.p());
            return BilinearPDF(u, w) * pdf;
        } else
            return pdf;
    } else {
        // Convert light sample weight to solid angle measure
        Float pdf = PDF(si->intr) * DistanceSquared(ref.p(), si->intr.p()) /
            AbsDot(si->intr.n, -wi);
        if (std::isinf(pdf)) pdf = 0.f;
        return pdf;
    }
}

pstd::optional<ShapeSample> BilinearPatch::Sample(const Point2f &uo) const {
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    Point2f u = uo;
    Float pdf = 1;
    if (mesh->imageDistribution)
        u = mesh->imageDistribution->SampleContinuous(u, &pdf);
    else if (!IsQuad()) {
        // Approximate uniform area
        pstd::array<Float, 4> w = { Length(Cross(p10 - p00, p01 - p00)),
                                    Length(Cross(p10 - p00, p11 - p10)),
                                    Length(Cross(p01 - p00, p11 - p01)),
                                    Length(Cross(p11 - p10, p11 - p01)) };
        u = SampleBilinear(u, w);
        pdf = BilinearPDF(u, w);
    }

    Point3f pu0 = Lerp(u[1], p00, p01), pu1 = Lerp(u[1], p10, p11);
    Vector3f dpdu = pu1 - pu0;
    Vector3f dpdv = Lerp(u[0], p01, p11) - Lerp(u[0], p00, p10);
    if (LengthSquared(dpdu) == 0 || LengthSquared(dpdv) == 0)
        return {};

    Normal3f n = Normal3f(Normalize(Cross(dpdu, dpdv)));
    if (OrientationIsReversed()) n = -n;

    // TODO: double check pError
    Point3f p = Lerp(u[0], pu0, pu1);
    Vector3f pError = gamma(6) *
        Vector3f(Max(Max(Abs(p00), Abs(p10)), Max(Abs(p01), Abs(p11))));

    return ShapeSample{Interaction(Point3fi(p, pError), n, u),
            pdf / Length(Cross(dpdu, dpdv))};
}

Float BilinearPatch::PDF(const Interaction &intr) const {
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    Float pdf = 1;
    if (mesh->imageDistribution)
        pdf = mesh->imageDistribution->ContinuousPDF(intr.uv);
    else if (!IsQuad()) {
        pstd::array<Float, 4> w = { Length(Cross(p10 - p00, p01 - p00)),
                                    Length(Cross(p10 - p00, p11 - p10)),
                                    Length(Cross(p01 - p00, p11 - p01)),
                                    Length(Cross(p11 - p10, p11 - p01)) };
        pdf = BilinearPDF(intr.uv, w);
    }

    CHECK(!intr.uv.HasNaN());
    Point3f pu0 = Lerp(intr.uv[1], p00, p01), pu1 = Lerp(intr.uv[1], p10, p11);
    Vector3f dpdu = pu1 - pu0;
    Vector3f dpdv = Lerp(intr.uv[0], p01, p11) - Lerp(intr.uv[0], p00, p10);
    return pdf / Length(Cross(dpdu, dpdv));
}

Float BilinearPatch::SolidAngle(const Point3f &pref) const {
    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    return SphericalQuadArea(Normalize(p00 - pref), Normalize(p01 - pref),
                             Normalize(p10 - pref), Normalize(p11 - pref));
}

DirectionCone BilinearPatch::NormalBounds() const {
    // Ignore shading normal...

    // Get bilinear patch vertices in _p00_, _p01_, _p10_, and _p11_
    auto mesh = GetMesh();
    const int *v = &mesh->vertexIndices[4 * blpIndex];
    const Point3f &p00 = mesh->p[v[0]];
    const Point3f &p10 = mesh->p[v[1]];
    const Point3f &p01 = mesh->p[v[2]];
    const Point3f &p11 = mesh->p[v[3]];

    if (p00 == p10 || p10 == p11 || p11 == p01 || p01 == p00) {
        // it's a triangle. Evaluate the normal at the center so we don't
        // have to worry about the degeneracies.
        Vector3f dpdu = Lerp(0.5f, p10, p11) - Lerp(0.5f, p00, p01);
        Vector3f dpdv = Lerp(0.5f, p01, p11) - Lerp(0.5f, p00, p10);
        Vector3f n = Normalize(Cross(dpdu, dpdv));
        return DirectionCone(n);
    }

    Vector3f n00 = Normalize(Cross(p10 - p00, p01 - p00));
    Vector3f n10 = Normalize(Cross(p11 - p10, p00 - p10));
    Vector3f n01 = Normalize(Cross(p00 - p01, p11 - p01));
    Vector3f n11 = Normalize(Cross(p01 - p11, p10 - p11));

    Vector3f n = Normalize(n00 + n10 + n01 + n11);
    Float cosTheta = std::min({Dot(n, n00), Dot(n, n01), Dot(n, n10), Dot(n, n11)});
    //LOG(WARNING) << "NB " << n << ", cos " << cosTheta;
    return DirectionCone(n, Clamp(cosTheta, -1, 1));
}

std::string BilinearPatch::ToString() const {
    return StringPrintf("[ BilinearPatch meshIndex: %d blpIndex: %d area: %f ]",
                        meshIndex, blpIndex, area);
}

pstd::vector<ShapeHandle> ShapeHandle::Create(
        const std::string &name, const Transform *worldFromObject,
        const Transform *objectFromWorld, bool reverseOrientation,
        const ParameterDictionary &dict, Allocator alloc, FileLoc loc) {
    pstd::vector<ShapeHandle> shapes(alloc);
    if (name == "sphere")
        shapes = {Sphere::Create(worldFromObject, objectFromWorld, reverseOrientation,
                                 dict, alloc)};
    // Create remaining single _Shape_ types
    else if (name == "cylinder")
        shapes = {Cylinder::Create(worldFromObject, objectFromWorld, reverseOrientation,
                                   dict, alloc)};
    else if (name == "disk")
        shapes = {Disk::Create(worldFromObject, objectFromWorld, reverseOrientation,
                               dict, alloc)};
    else if (name == "bilinearmesh")
        shapes = BilinearPatchMesh::Create(worldFromObject, reverseOrientation,
                                           dict, alloc);
    // Create multiple-_Shape_ types
    else if (name == "curve")
        shapes = Curve::Create(worldFromObject, objectFromWorld, reverseOrientation,
                               dict, alloc);
    else if (name == "trianglemesh") {
        TriangleMesh *mesh = TriangleMesh::Create(worldFromObject, reverseOrientation, dict,
                                                  alloc);
        shapes = mesh->CreateTriangles(alloc);
    } else if (name == "plymesh")
        shapes = CreatePLYMesh(worldFromObject, reverseOrientation,
                               dict, alloc);
    else if (name == "loopsubdiv") {
        TriangleMesh *mesh = CreateLoopSubdivMesh(worldFromObject, reverseOrientation,
                                                  dict, alloc);
        shapes = mesh->CreateTriangles(alloc);
    } else
        ErrorExit(&loc, "%s: shape type unknown.", name);

    if (shapes.empty())
        ErrorExit(&loc, "%s: unable to create shape.", name);

    return shapes;
}

std::string ShapeHandle::ToString() const {
    if (Tag() == TypeIndex<Triangle>())
        return Cast<Triangle>()->ToString();
    else if (Tag() == TypeIndex<BilinearPatch>())
        return Cast<BilinearPatch>()->ToString();
    else if (Tag() == TypeIndex<Curve>())
        return Cast<Curve>()->ToString();
    else if (Tag() == TypeIndex<Sphere>())
        return Cast<Sphere>()->ToString();
    else if (Tag() == TypeIndex<Cylinder>())
        return Cast<Cylinder>()->ToString();
    else if (Tag() == TypeIndex<Disk>())
        return Cast<Disk>()->ToString();
    else {
        LOG_FATAL("Unhandled case");
        return {};
    }
}

}  // namespace pbrt
