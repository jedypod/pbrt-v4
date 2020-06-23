// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_UTIL_MESH_H
#define PBRT_UTIL_MESH_H 1

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <string>
#include <vector>

namespace pbrt {

void InitBufferCaches(Allocator alloc);
void FreeBufferCaches();

class TriangleMesh {
  public:
    // TriangleMesh Public Methods
    TriangleMesh(const Transform &worldFromObject, bool reverseOrientation,
                 std::vector<int> vertexIndices, std::vector<Point3f> p,
                 std::vector<Vector3f> S, std::vector<Normal3f> N,
                 std::vector<Point2f> uv, std::vector<int> faceIndices);

    std::string ToString() const;

    bool WritePLY(const std::string &filename) const;

    static void Init(Allocator alloc);

    // TriangleMesh Data
    bool reverseOrientation, transformSwapsHandedness;
    int nTriangles, nVertices;
    const int *vertexIndices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;
};

class BilinearPatchMesh {
  public:
    BilinearPatchMesh(const Transform &worldFromObject, bool reverseOrientation,
                      std::vector<int> vertexIndices, std::vector<Point3f> p,
                      std::vector<Normal3f> N, std::vector<Point2f> uv,
                      std::vector<int> faceIndices, PiecewiseConstant2D *imageDist);

    std::string ToString() const;

    static void Init(Allocator alloc);

    bool reverseOrientation, transformSwapsHandedness;
    int nPatches, nVertices;
    const int *vertexIndices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;
    PiecewiseConstant2D *imageDistribution;
};

struct TriQuadMesh {
    static pstd::optional<TriQuadMesh> ReadPLY(const std::string &filename);

    void ConvertToOnlyTriangles();
    std::string ToString() const;

    std::vector<Point3f> p;
    std::vector<Normal3f> n;
    std::vector<Point2f> uv;
    std::vector<int> faceIndices;
    std::vector<int> triIndices, quadIndices;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_MESH_H
