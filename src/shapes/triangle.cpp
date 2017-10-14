
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

// shapes/triangle.cpp*
#include "shapes/triangle.h"

#include "paramset.h"
#include "error.h"
#include "interaction.h"
#include "sampling.h"
#include "util/efloat.h"
#include "ext/rply.h"

#include <array>


namespace pbrt {

STAT_PERCENT("Intersections/Ray-triangle intersection tests", nHits, nTests);

// Triangle Local Definitions
static void PlyErrorCallback(p_ply, const char *message) {
    Error("PLY writing error: %s", message);
}

// Triangle Method Definitions
STAT_RATIO("Scene/Triangles per triangle mesh", nTris, nMeshes);
TriangleMesh::TriangleMesh(
    const Transform &ObjectToWorld, bool reverseOrientation,
    absl::Span<const int> vertexIndices, absl::Span<const Point3f> P,
    absl::Span<const Vector3f> S, absl::Span<const Normal3f> N, absl::Span<const Point2f> UV,
    absl::Span<const int> faceIndices, const std::shared_ptr<const ParamSet> &attributes)
    : reverseOrientation(reverseOrientation),
      transformSwapsHandedness(ObjectToWorld.SwapsHandedness()),
      nTriangles(vertexIndices.size() / 3),
      nVertices(P.size()),
      vertexIndices(vertexIndices.begin(), vertexIndices.end()),
      faceIndices(faceIndices.begin(), faceIndices.end()),
      attributes(attributes) {
    CHECK_EQ((vertexIndices.size() % 3), 0);
    ++nMeshes;
    nTris += nTriangles;
    triMeshBytes +=
        sizeof(*this) + (3 * nTriangles * sizeof(int)) +
        nVertices * (sizeof(P[0]) + (N.size() > 0 ? sizeof(N[0]) : 0) +
                     (S.size() > 0 ? sizeof(S[0]) : 0) +
                     (UV.size() > 0 ? sizeof(UV[0]) : 0));

    // Transform mesh vertices to world space
    p.resize(nVertices);
    for (int i = 0; i < nVertices; ++i) p[i] = ObjectToWorld(P[i]);

    // Copy _UV_, _N_, and _S_ vertex data, if present
    if (UV.size() > 0) {
        CHECK_GE(UV.size(), nVertices);
        uv = {UV.begin(), UV.end()};
    }
    if (N.size() > 0) {
        CHECK_GE(N.size(), nVertices);
        n.resize(nVertices);
        for (int i = 0; i < nVertices; ++i) n[i] = ObjectToWorld(N[i]);
    }
    if (S.size() > 0) {
        s.resize(nVertices);
        for (int i = 0; i < nVertices; ++i) s[i] = ObjectToWorld(S[i]);
    }
}

std::vector<std::shared_ptr<Shape>> CreateTriangleMesh(
    const Transform &ObjectToWorld, const Transform &WorldToObject,
    bool reverseOrientation, absl::Span<const int> vertexIndices,
    absl::Span<const Point3f> p, absl::Span<const Vector3f> s, absl::Span<const Normal3f> n,
    absl::Span<const Point2f> uv, absl::Span<const int> faceIndices,
    const std::shared_ptr<const ParamSet> &attributes) {
    std::shared_ptr<TriangleMesh> mesh = std::make_shared<TriangleMesh>(
        ObjectToWorld, reverseOrientation, vertexIndices, p, s, n, uv,
        faceIndices, attributes);
    std::vector<std::shared_ptr<Shape>> tris;
    size_t nTriangles = vertexIndices.size() / 3;
    tris.reserve(nTriangles);
    for (int i = 0; i < nTriangles; ++i)
        tris.push_back(std::make_shared<Triangle>(mesh, i));
    return tris;
}

bool WritePlyFile(const std::string &filename, absl::Span<const int> vertexIndices,
                  absl::Span<const Point3f> P, absl::Span<const Vector3f> S,
                  absl::Span<const Normal3f> N, absl::Span<const Point2f> UV) {
    size_t nVertices = P.size();
    size_t nTriangles = vertexIndices.size() / 3;
    CHECK_EQ(vertexIndices.size() % 3, 0);
    p_ply plyFile =
        ply_create(filename.c_str(), PLY_DEFAULT, PlyErrorCallback, 0, nullptr);
    if (plyFile != nullptr) {
        ply_add_element(plyFile, "vertex", nVertices);
        ply_add_scalar_property(plyFile, "x", PLY_FLOAT);
        ply_add_scalar_property(plyFile, "y", PLY_FLOAT);
        ply_add_scalar_property(plyFile, "z", PLY_FLOAT);
        if (N.size() > 0) {
            ply_add_scalar_property(plyFile, "nx", PLY_FLOAT);
            ply_add_scalar_property(plyFile, "ny", PLY_FLOAT);
            ply_add_scalar_property(plyFile, "nz", PLY_FLOAT);
        }
        if (UV.size() > 0) {
            ply_add_scalar_property(plyFile, "u", PLY_FLOAT);
            ply_add_scalar_property(plyFile, "v", PLY_FLOAT);
        }
        if (S.size() > 0)
            Warning("PLY mesh in \"%s\" will be missing tangent vectors \"S\".",
                    filename.c_str());

        ply_add_element(plyFile, "face", nTriangles);
        ply_add_list_property(plyFile, "vertex_indices", PLY_UINT8, PLY_INT);
        ply_write_header(plyFile);

        for (int i = 0; i < nVertices; ++i) {
            ply_write(plyFile, P[i].x);
            ply_write(plyFile, P[i].y);
            ply_write(plyFile, P[i].z);
            if (N.size() > 0) {
                ply_write(plyFile, N[i].x);
                ply_write(plyFile, N[i].y);
                ply_write(plyFile, N[i].z);
            }
            if (UV.size() > 0) {
                ply_write(plyFile, UV[i].x);
                ply_write(plyFile, UV[i].y);
            }
        }

        for (int i = 0; i < nTriangles; ++i) {
            ply_write(plyFile, 3);
            ply_write(plyFile, vertexIndices[3 * i]);
            ply_write(plyFile, vertexIndices[3 * i + 1]);
            ply_write(plyFile, vertexIndices[3 * i + 2]);
        }
        ply_close(plyFile);
        return true;
    }
    return false;
}

Bounds3f Triangle::WorldBound() const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];
    return Union(Bounds3f(p0, p1), p2);
}

bool Triangle::Intersect(const Ray &ray, Float *tHit, SurfaceInteraction *isect) const {
    ProfilePhase p(Prof::TriIntersect);
    ++nTests;
    // Get triangle vertices in _p0_, _p1_, and _p2_
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    // Perform ray--triangle intersection test

    // Transform triangle vertices to ray coordinate space

    // Translate vertices based on ray origin
    Point3f p0t = p0 - Vector3f(ray.o);
    Point3f p1t = p1 - Vector3f(ray.o);
    Point3f p2t = p2 - Vector3f(ray.o);

    // Permute components of triangle vertices and ray direction
    int kz = MaxDimension(Abs(ray.d));
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
    Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

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

    // Perform triangle edge and determinant tests
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    Float det = e0 + e1 + e2;
    if (det == 0) return false;

    // Compute scaled hit distance to triangle and test against ray $t$ range
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
        return false;
    else if (det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
        return false;

    // Compute barycentric coordinates and $t$ value for triangle intersection
    Float invDet = 1 / det;
    Float b0 = e0 * invDet;
    Float b1 = e1 * invDet;
    Float b2 = e2 * invDet;
    Float t = tScaled * invDet;

    // Ensure that computed triangle $t$ is conservatively greater than zero

    // Compute $\delta_z$ term for triangle $t$ error bounds
    Float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
    Float deltaZ = gamma(3) * maxZt;

    // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
    Float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
    Float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
    Float deltaX = gamma(5) * (maxXt + maxZt);
    Float deltaY = gamma(5) * (maxYt + maxZt);

    // Compute $\delta_e$ term for triangle $t$ error bounds
    Float deltaE =
        2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

    // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
    Float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
    Float deltaT = 3 *
                   (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
                   std::abs(invDet);
    if (t <= deltaT) return false;

    // Compute triangle partial derivatives
    Vector3f dpdu, dpdv;
    std::array<Point2f, 3> uv = GetUVs();

    // Compute deltas for triangle partial derivatives
    Vector2f duv02 = uv[0] - uv[2], duv12 = uv[1] - uv[2];
    Vector3f dp02 = p0 - p2, dp12 = p1 - p2;
    Float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
    bool degenerateUV = std::abs(determinant) < 1e-8;
    if (!degenerateUV) {
        Float invdet = 1 / determinant;
        dpdu = (duv12[1] * dp02 - duv02[1] * dp12) * invdet;
        dpdv = (-duv12[0] * dp02 + duv02[0] * dp12) * invdet;
    }
    if (degenerateUV || LengthSquared(Cross(dpdu, dpdv)) == 0) {
        Vector3f n = Cross(p2 - p0, p1 - p0);
        if (LengthSquared(n) == 0) {
            // TODO: should these be eliminated from the start?
            LOG(WARNING) << "Intersected degenerate triangle; ignoring";
            return false;
        }
        // Handle zero determinant for triangle partial derivative matrix
        CoordinateSystem(Normalize(n), &dpdu, &dpdv);
    }

    // Compute error bounds for triangle intersection
    Float xAbsSum =
        (std::abs(b0 * p0.x) + std::abs(b1 * p1.x) + std::abs(b2 * p2.x));
    Float yAbsSum =
        (std::abs(b0 * p0.y) + std::abs(b1 * p1.y) + std::abs(b2 * p2.y));
    Float zAbsSum =
        (std::abs(b0 * p0.z) + std::abs(b1 * p1.z) + std::abs(b2 * p2.z));
    Vector3f pError = gamma(7) * Vector3f(xAbsSum, yAbsSum, zAbsSum);

    // Interpolate $(u,v)$ parametric coordinates and hit point
    Point3f pHit = b0 * p0 + b1 * p1 + b2 * p2;
    Point2f uvHit = b0 * uv[0] + b1 * uv[1] + b2 * uv[2];

    // Fill in _SurfaceInteraction_ from triangle hit
    int faceIndex = mesh->faceIndices.size() ? mesh->faceIndices[triIndex] : 0;
    *isect = SurfaceInteraction(pHit, pError, uvHit, -ray.d, dpdu, dpdv,
                                Normal3f(0, 0, 0), Normal3f(0, 0, 0), ray.time,
                                this, faceIndex);

    // Override surface normal in _isect_ for triangle
    isect->n = isect->shading.n = Normal3f(Normalize(Cross(dp02, dp12)));
    if (mesh->n.size() || mesh->s.size()) {
        // Initialize _Triangle_ shading geometry

        // Compute shading normal _ns_ for triangle
        Normal3f ns;
        if (mesh->n.size()) {
            ns = (b0 * mesh->n[v[0]] + b1 * mesh->n[v[1]] + b2 * mesh->n[v[2]]);
            if (LengthSquared(ns) > 0)
                ns = Normalize(ns);
            else
                ns = isect->n;
        } else
            ns = isect->n;

        // Compute shading tangent _ss_ for triangle
        Vector3f ss;
        if (mesh->s.size()) {
            ss = (b0 * mesh->s[v[0]] + b1 * mesh->s[v[1]] + b2 * mesh->s[v[2]]);
            if (LengthSquared(ss) > 0)
                ss = Normalize(ss);
            else
                ss = Normalize(isect->dpdu);
        } else
            ss = Normalize(isect->dpdu);

        // Compute shading bitangent _ts_ for triangle and adjust _ss_
        Vector3f ts = Cross(ss, ns);
        if (LengthSquared(ts) > 0) {
            ts = Normalize(ts);
            ss = Cross(ts, ns);
        } else
            CoordinateSystem((Vector3f)ns, &ss, &ts);

        // Compute $\dndu$ and $\dndv$ for triangle shading geometry
        Normal3f dndu, dndv;
        if (mesh->n.size()) {
            // Compute deltas for triangle partial derivatives of normal
            Vector2f duv02 = uv[0] - uv[2];
            Vector2f duv12 = uv[1] - uv[2];
            Normal3f dn1 = mesh->n[v[0]] - mesh->n[v[2]];
            Normal3f dn2 = mesh->n[v[1]] - mesh->n[v[2]];
            Float determinant = duv02[0] * duv12[1] - duv02[1] * duv12[0];
            bool degenerateUV = std::abs(determinant) < 1e-8;
            if (degenerateUV)
                dndu = dndv = Normal3f(0, 0, 0);
            else {
                Float invDet = 1 / determinant;
                dndu = (duv12[1] * dn1 - duv02[1] * dn2) * invDet;
                dndv = (-duv12[0] * dn1 + duv02[0] * dn2) * invDet;
            }
        } else
            dndu = dndv = Normal3f(0, 0, 0);
        isect->SetShadingGeometry(ss, ts, dndu, dndv, true);
    }

    // Ensure correct orientation of the geometric normal
    if (mesh->n.size())
        isect->n = Faceforward(isect->n, isect->shading.n);
    else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        isect->n = isect->shading.n = -isect->n;
    *tHit = t;
    ++nHits;
    return true;
}

bool Triangle::IntersectP(const Ray &ray) const {
    ProfilePhase p(Prof::TriIntersectP);
    ++nTests;
    // Get triangle vertices in _p0_, _p1_, and _p2_
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    // Perform ray--triangle intersection test

    // Transform triangle vertices to ray coordinate space

    // Translate vertices based on ray origin
    Point3f p0t = p0 - Vector3f(ray.o);
    Point3f p1t = p1 - Vector3f(ray.o);
    Point3f p2t = p2 - Vector3f(ray.o);

    // Permute components of triangle vertices and ray direction
    int kz = MaxDimension(Abs(ray.d));
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
    Float e0 = p1t.x * p2t.y - p1t.y * p2t.x;
    Float e1 = p2t.x * p0t.y - p2t.y * p0t.x;
    Float e2 = p0t.x * p1t.y - p0t.y * p1t.x;

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

    // Perform triangle edge and determinant tests
    if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0))
        return false;
    Float det = e0 + e1 + e2;
    if (det == 0) return false;

    // Compute scaled hit distance to triangle and test against ray $t$ range
    p0t.z *= Sz;
    p1t.z *= Sz;
    p2t.z *= Sz;
    Float tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
    if (det < 0 && (tScaled >= 0 || tScaled < ray.tMax * det))
        return false;
    else if (det > 0 && (tScaled <= 0 || tScaled > ray.tMax * det))
        return false;

    // Compute barycentric coordinates and $t$ value for triangle intersection
    Float invDet = 1 / det;
    Float b0 = e0 * invDet;
    Float b1 = e1 * invDet;
    Float b2 = e2 * invDet;
    Float t = tScaled * invDet;

    // Ensure that computed triangle $t$ is conservatively greater than zero

    // Compute $\delta_z$ term for triangle $t$ error bounds
    Float maxZt = MaxComponent(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));
    Float deltaZ = gamma(3) * maxZt;

    // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
    Float maxXt = MaxComponent(Abs(Vector3f(p0t.x, p1t.x, p2t.x)));
    Float maxYt = MaxComponent(Abs(Vector3f(p0t.y, p1t.y, p2t.y)));
    Float deltaX = gamma(5) * (maxXt + maxZt);
    Float deltaY = gamma(5) * (maxYt + maxZt);

    // Compute $\delta_e$ term for triangle $t$ error bounds
    Float deltaE =
        2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

    // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
    Float maxE = MaxComponent(Abs(Vector3f(e0, e1, e2)));
    Float deltaT = 3 *
                   (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) *
                   std::abs(invDet);
    if (t <= deltaT) return false;

    ++nHits;
    return true;
}

Float Triangle::Area() const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];
    return 0.5 * Length(Cross(p1 - p0, p2 - p0));
}

Interaction Triangle::Sample(const Point2f &u, Float *pdf) const {
    std::array<Float, 3> b = UniformSampleTriangle(u);
    // Get triangle vertices in _p0_, _p1_, and _p2_
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];
    Interaction it;
    it.p = b[0] * p0 + b[1] * p1 + b[2] * p2;
    // Compute surface normal for sampled point on triangle
    it.n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
    // Ensure correct orientation of the geometric normal; follow the same
    // approach as was used in Triangle::Intersect().
    if (mesh->n.size()) {
        Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                    (1 - b[0] - b[1]) * mesh->n[v[2]]);
        it.n = Faceforward(it.n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        it.n *= -1;

    // Compute error bounds for sampled point on triangle
    Point3f pAbsSum =
        Abs(b[0] * p0) + Abs(b[1] * p1) + Abs((1 - b[0] - b[1]) * p2);
    it.pError = gamma(6) * Vector3f(pAbsSum.x, pAbsSum.y, pAbsSum.z);
    *pdf = 1 / Area();
    return it;
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
Interaction Triangle::Sample(const Interaction &ref, const Point2f &u,
                             Float *pdf) const {
    // Get triangle vertices in _p0_, _p1_, and _p2_
    const int *v = &mesh->vertexIndices[3 * triIndex];
    const Point3f &p0 = mesh->p[v[0]];
    const Point3f &p1 = mesh->p[v[1]];
    const Point3f &p2 = mesh->p[v[2]];

    Float sa = SolidAngle(ref.p);
    if (sa < MinSphericalSampleArea || sa > MaxSphericalSampleArea)
        return Shape::Sample(ref, u, pdf);

    std::array<Float, 3> b = SphericalSampleTriangle({p0, p1, p2}, ref.p, u, pdf);
    if (*pdf == 0)
        return {};

    // Initialize Interaction for sampled point on triangle.
    Interaction it;
    it.p = b[0] * p0 + b[1] * p1 + b[2] * p2;

    // Compute surface normal for sampled point on triangle
    it.n = Normalize(Normal3f(Cross(p1 - p0, p2 - p0)));
    // Ensure correct orientation of the geometric normal; follow the same
    // approach as was used in Triangle::Intersect().
    if (!mesh->n.empty()) {
        Normal3f ns(b[0] * mesh->n[v[0]] + b[1] * mesh->n[v[1]] +
                    b[2] * mesh->n[v[2]]);
        it.n = Faceforward(it.n, ns);
    } else if (mesh->reverseOrientation ^ mesh->transformSwapsHandedness)
        it.n *= -1;

    // Compute error bounds for sampled point on triangle
    Point3f pAbsSum =
        Abs(b[0] * p0) + Abs(b[1] * p1) + Abs(b[2] * p2);
    it.pError = gamma(6) * Vector3f(pAbsSum.x, pAbsSum.y, pAbsSum.z);

    return it;
}

Float Triangle::Pdf(const Interaction &ref, const Vector3f &wi) const {
    Float sa = SolidAngle(ref.p);
    if (sa < MinSphericalSampleArea || sa > MaxSphericalSampleArea)
        return Shape::Pdf(ref, wi);

    return IntersectP(ref.SpawnRay(wi)) ? (1 / sa) : 0;
}

Float Triangle::SolidAngle(const Point3f &p, int nSamples) const {
    // Project the vertices into the unit sphere around p.
    const int *v = &mesh->vertexIndices[3 * triIndex];
    Vector3f a = mesh->p[v[0]] - p, b = mesh->p[v[1]] - p, c = mesh->p[v[2]] - p;

    // http://math.stackexchange.com/questions/9819/area-of-a-spherical-triangle
    // Girard's theorem: surface area of a spherical triangle on a unit
    // sphere is the 'excess angle' alpha+beta+gamma-pi, where
    // alpha/beta/gamma are the interior angles at the vertices.
    //
    // Given three vertices on the sphere, a, b, c, then we can compute,
    // for example, the angle c->a->b by
    //
    // cos theta =  Dot(Cross(c, a), Cross(b, a)) /
    //              (Length(Cross(c, a)) * Length(Cross(b, a))).
    //
    // We only need to do three cross products to evaluate the angles at
    // all three vertices, though, since we can take advantage of the fact
    // that Cross(a, b) = -Cross(b, a).
    Vector3f axb = Cross(a, b), bxc = Cross(b, c), cxa = Cross(c, a);
    if (LengthSquared(axb) == 0 || LengthSquared(bxc) == 0 || LengthSquared(cxa) == 0)
        return 0;
    axb = Normalize(axb);
    bxc = Normalize(bxc);
    cxa = Normalize(cxa);

    Float alpha = AngleBetween(cxa, -axb);
    Float beta = AngleBetween(axb, -bxc);
    Float gamma = AngleBetween(bxc, -cxa);

    return std::abs(alpha + beta + gamma - Pi);
}

std::vector<std::shared_ptr<Shape>> CreateTriangleMeshShape(
    std::shared_ptr<const Transform> ObjectToWorld,
    std::shared_ptr<const Transform> WorldToObject, bool reverseOrientation,
    const ParamSet &params, const std::shared_ptr<const ParamSet> &attributes) {
    absl::Span<const int> vi = params.GetIntArray("indices");
    absl::Span<const Point3f> P = params.GetPoint3fArray("P");
    absl::Span<const Point2f> uvs = params.GetPoint2fArray("uv");
    if (uvs.empty()) uvs = params.GetPoint2fArray("st");
    std::vector<Point2f> tempUVs;
    if (uvs.empty()) {
        absl::Span<const Float> fuv = params.GetFloatArray("uv");
        if (fuv.empty()) fuv = params.GetFloatArray("st");
        if (!fuv.empty()) {
            tempUVs.reserve(fuv.size() / 2);
            for (size_t i = 0; i < fuv.size() / 2; ++i)
                tempUVs.push_back(Point2f(fuv[2 * i], fuv[2 * i + 1]));
            uvs = tempUVs;
        }
    }

    if (!uvs.empty()) {
        if (uvs.size() < P.size()) {
            Error(
                "Not enough of \"uv\"s for triangle mesh.  Expected %d, "
                "found %d.  Discarding.",
                (int)P.size(), (int)uvs.size());
            uvs = {};
        } else if (uvs.size() > P.size())
            Warning(
                "More \"uv\"s provided than will be used for triangle "
                "mesh.  (%d expcted, %d found)",
                (int)P.size(), (int)uvs.size());
    }

    if (vi.empty()) {
        Error(
            "Vertex indices \"indices\" not provided with triangle mesh shape");
        return std::vector<std::shared_ptr<Shape>>();
    } else if (vi.size() % 3) {
        Error("Number of vertex indices %d not a multiple of 3.",
              int(vi.size()));
        return {};
    }

    if (P.empty()) {
        Error("Vertex positions \"P\" not provided with triangle mesh shape");
        return std::vector<std::shared_ptr<Shape>>();
    }
    absl::Span<const Vector3f> S = params.GetVector3fArray("S");
    if (!S.empty() && S.size() != P.size()) {
        Error("Number of \"S\"s for triangle mesh must match \"P\"s");
        S = {};
    }
    absl::Span<const Normal3f> N = params.GetNormal3fArray("N");
    if (!N.empty() && N.size() != P.size()) {
        Error("Number of \"N\"s for triangle mesh must match \"P\"s");
        N = {};
    }

    for (size_t i = 0; i < vi.size(); ++i)
        if (vi[i] >= P.size()) {
            Error(
                "trianglemesh has out of-bounds vertex index %d (%d \"P\" "
                "values were given",
                vi[i], (int)P.size());
            return std::vector<std::shared_ptr<Shape>>();
        }

    absl::Span<const int> faceIndices = params.GetIntArray("faceIndices");
    if (!faceIndices.empty() && faceIndices.size() != vi.size() / 3) {
        Error("Number of face indices %d != number of triangles %d",
              int(faceIndices.size()), int(vi.size() / 3));
        faceIndices = {};
    }

    return CreateTriangleMesh(*ObjectToWorld, *WorldToObject,
                              reverseOrientation, vi, P, S, N, uvs,
                              faceIndices, attributes);
}

}  // namespace pbrt
