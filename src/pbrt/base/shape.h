// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_BASE_SHAPE_H
#define PBRT_BASE_SHAPE_H

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#include <pbrt/pbrt.h>

#include <pbrt/util/buffercache.h>
#include <pbrt/util/float.h>
#include <pbrt/util/taggedptr.h>
#include <pbrt/util/vecmath.h>

#include <string>

namespace pbrt {

// Shape Declarations
class Triangle;
class BilinearPatch;
class Curve;
class Sphere;
class Cylinder;
class Disk;

class ShapeHandle
    : public TaggedPointer<Triangle, BilinearPatch, Curve, Sphere, Cylinder, Disk> {
  public:
    using TaggedPointer::TaggedPointer;

    static pstd::vector<ShapeHandle> Create(const std::string &name,
                                            const Transform *worldFromObject,
                                            const Transform *objectFromWorld,
                                            bool reverseOrientation,
                                            const ParameterDictionary &parameters,
                                            const FileLoc *loc, Allocator alloc);

    PBRT_CPU_GPU inline Bounds3f Bounds() const;

    PBRT_CPU_GPU inline pstd::optional<ShapeIntersection> Intersect(
        const Ray &ray, Float tMax = Infinity) const;
    PBRT_CPU_GPU inline bool IntersectP(const Ray &ray, Float tMax = Infinity) const;

    PBRT_CPU_GPU inline Float Area() const;

    PBRT_CPU_GPU inline pstd::optional<ShapeSample> Sample(const Point2f &u) const;
    PBRT_CPU_GPU inline Float PDF(const Interaction &) const;

    PBRT_CPU_GPU inline pstd::optional<ShapeSample> Sample(const Interaction &ref,
                                                           const Point2f &u) const;
    PBRT_CPU_GPU inline Float PDF(const Interaction &ref, const Vector3f &wi) const;

    PBRT_CPU_GPU inline Float SolidAngle(const Point3f &p, int nSamples = 512) const;
    PBRT_CPU_GPU inline DirectionCone NormalBounds() const;

    PBRT_CPU_GPU inline bool OrientationIsReversed() const;
    PBRT_CPU_GPU inline bool TransformSwapsHandedness() const;

    std::string ToString() const;

    PBRT_CPU_GPU
    Float SampledSolidAngle(const Point3f &p, int nSamples = 512) const;

  private:
    friend class TriangleMesh;
    friend class BilinearPatchMesh;

    static BufferCache<int> *indexBufferCache;
    static BufferCache<Point3f> *pBufferCache;
    static BufferCache<Normal3f> *nBufferCache;
    static BufferCache<Point2f> *uvBufferCache;
    static BufferCache<Vector3f> *sBufferCache;
    static BufferCache<int> *faceIndexBufferCache;
};

}  // namespace pbrt

#endif  // PBRT_BASE_SHAPE_H
