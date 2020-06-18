// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/geometry/bounds.h>
#include <pbrt/geometry/frame.h>
#include <pbrt/geometry/octree.h>
#include <pbrt/geometry/quaternion.h>
#include <pbrt/geometry/ray.h>
#include <pbrt/geometry/spherical.h>
#include <pbrt/geometry/splines.h>
#include <pbrt/geometry/tuple.h>
#include <pbrt/geometry/util/transform.h>
#include <pbrt/geometry/vecmath.h>

#include <pbrt/core/interaction.h>
#include <pbrt/core/medium.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitGeometry(py::module &m) {
    // bounds.h
    py::class_<Bounds2f>(m, "Bounds2f")
        .def(py::init<>())
        .def(py::init<Point2f>())
        .def(py::init<Point2f, Point2f>())
        .def("Diagonal", &Bounds2f::Diagonal)
        .def("Area", &Bounds2f::Area)
        .def("IsEmpty", &Bounds2f::IsEmpty)
        .def("IsDegenerate", &Bounds2f::IsDegenerate)
        .def("MaxDimension", &Bounds2f::MaxDimension)
        .def("__getitem__", [](const Bounds2f &b, int i) { return b[i]; })
        .def("__setitem__", [](Bounds2f &p, int i, Point2f v) { p[i] = v; })
        .def("__eq__", &Bounds2f::operator==)
        .def("__str__", &Bounds2f::ToString)
        .def("Lerp", &Bounds2f::Lerp)
        .def("Offset", &Bounds2f::Offset)
        .def("BoundingSphere",
             [](const Bounds2f &b) {
                 Point2f c;
                 Float r;
                 b.BoundingSphere(&c, &r);
                 return std::make_tuple(c, r);
             })
        .def_readwrite("pMin", &Bounds2f::pMin)
        .def_readwrite("pMax", &Bounds2f::pMax);

    m.def("Union", (Bounds2f(*)(const Bounds2f &, const Point2f &)) & Union);
    m.def("Union", (Bounds2f(*)(const Bounds2f &, const Bounds2f &)) & Union);
    m.def("Intersect", (Bounds2f(*)(const Bounds2f &, const Bounds2f &)) & Intersect);
    m.def("Overlaps", (bool (*)(const Bounds2f &, const Bounds2f &)) & Overlaps);
    m.def("Inside", (bool (*)(const Point2f &, const Bounds2f &)) & Inside);
    m.def("InsideExclusive",
          (bool (*)(const Point2f &, const Bounds2f &)) & InsideExclusive);
    m.def("Expand", (Bounds2f(*)(const Bounds2f &, Float)) & Expand);

    py::class_<Bounds3f>(m, "Bounds3f")
        .def(py::init<>())
        .def(py::init<Point3f>())
        .def(py::init<Point3f, Point3f>())
        .def("Diagonal", &Bounds3f::Diagonal)
        .def("SurfaceArea", &Bounds3f::SurfaceArea)
        .def("Volume", &Bounds3f::Volume)
        .def("IsEmpty", &Bounds3f::IsEmpty)
        .def("IsDegenerate", &Bounds3f::IsDegenerate)
        .def("MaxDimension", &Bounds3f::MaxDimension)
        .def("__getitem__", [](const Bounds3f &b, int i) { return b[i]; })
        .def("__setitem__", [](Bounds3f &p, int i, Point3f v) { p[i] = v; })
        .def("__eq__", &Bounds3f::operator==)
        .def("__str__", &Bounds3f::ToString)
        .def("Lerp", &Bounds3f::Lerp)
        .def("Offset", &Bounds3f::Offset)
        .def("SurfaceArea", &Bounds3f::SurfaceArea)
        .def("BoundingSphere",
             [](const Bounds3f &b) {
                 Point3f c;
                 Float r;
                 b.BoundingSphere(&c, &r);
                 return std::make_tuple(c, r);
             })
        .def("IntersectP",
             (bool (Bounds3f::*)(const Ray &, Float, Float *, Float *) const) &
                 Bounds3f::IntersectP,
             "ray"_a, "tMax"_a = Infinity, "t0"_a = nullptr, "t1"_a = nullptr)
        .def("IntersectP",
             (bool (Bounds3f::*)(const Ray &, Float, const Vector3f &, const int[3])
                  const) &
                 Bounds3f::IntersectP,
             "ray"_a, "tMax"_a, "invDir"_a, "dirIsNegative"_a)
        .def_readwrite("pMin", &Bounds3f::pMin)
        .def_readwrite("pMax", &Bounds3f::pMax);

    m.def("Union", (Bounds3f(*)(const Bounds3f &, const Point3f &)) & Union);
    m.def("Union", (Bounds3f(*)(const Bounds3f &, const Bounds3f &)) & Union);
    m.def("Intersect", (Bounds3f(*)(const Bounds3f &, const Bounds3f &)) & Intersect);
    m.def("Overlaps", (bool (*)(const Bounds3f &, const Bounds3f &)) & Overlaps);
    m.def("Inside", (bool (*)(const Point3f &, const Bounds3f &)) & Inside);
    m.def("InsideExclusive",
          (bool (*)(const Point3f &, const Bounds3f &)) & InsideExclusive);
    m.def("Expand", (Bounds3f(*)(const Bounds3f &, Float)) & Expand);
    m.def("DistanceSquared",
          (Float(*)(const Point3f &, const Bounds3f &)) & DistanceSquared);
    m.def("Distance", (Float(*)(const Point3f &, const Bounds3f &)) & Distance);

    // frame.h
    py::class_<Frame>(m, "Frame")
        .def(py::init<>())
        .def(py::init<Vector3f, Vector3f, Vector3f>())
        .def("FromXZ", &Frame::FromXZ)
        .def("FromZ", (Frame(*)(const Vector3f &)) & Frame::FromZ)
        .def("FromZ", (Frame(*)(const Normal3f &)) & Frame::FromZ)
        .def("ToLocal", (Vector3f(Frame::*)(const Vector3f &) const) & Frame::ToLocal)
        .def("FromLocal", (Vector3f(Frame::*)(const Vector3f &) const) & Frame::FromLocal)
        .def("ToLocal", (Normal3f(Frame::*)(const Normal3f &) const) & Frame::ToLocal)
        .def("FromLocal", (Normal3f(Frame::*)(const Normal3f &) const) & Frame::FromLocal)
        .def("Transform", &Frame::operator Transform);

    // octree.h
    // TODO

    // quaternion.h
    py::class_<Quaternion>(m, "Quaternion")
        .def(py::init<>())
        .def(py::init<Vector3f, Float>())
        .def_readwrite("v", &Quaternion::v)
        .def_readwrite("w", &Quaternion::w)
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self -= py::self)
        .def(py::self - py::self)
        .def(-py::self)
        .def(py::self *= float())
        .def(py::self * float())
        .def(float() * py::self)
        .def(py::self /= float())
        .def(py::self / float())
        .def("__str__", &Quaternion::ToString);

    m.def("Slerp", Slerp);
    m.def("Dot", (Float(*)(const Quaternion &, const Quaternion &)) & Dot);
    m.def("Normalize", (Quaternion(*)(const Quaternion &)) & Normalize);

    // ray.h
    py::class_<Ray>(m, "Ray")
        .def(py::init<>())
        // FIXME: this doesn't seem to work
        .def(py::init<Point3f, Vector3f, Float, const Medium *>(), "o"_a, "d"_a,
             "time"_a = 0.f, "medium"_a = nullptr)
        .def("HasNan", &Ray::HasNaN)
        .def("__str__", &Ray::ToString)
        .def("__call__", &Ray::operator())
        .def_readwrite("o", &Ray::o)
        .def_readwrite("d", &Ray::d)
        .def_readwrite("time", &Ray::time)
        .def_readwrite("medium", &Ray::medium);

    py::class_<RayDifferential>(m, "RayDifferential")
        .def(py::init<>())
        .def(py::init<Point3f, Vector3f, Float, const Medium *>(), "o"_a, "d"_a,
             "time"_a = 0.f, "medium"_a = nullptr)
        .def(py::init<Ray>())
        .def("HasNaN", &RayDifferential::HasNaN)
        .def("ScaleDifferentials", &RayDifferential::ScaleDifferentials)
        .def("__str__", &RayDifferential::ToString)
        .def_readwrite("hasDifferentials", &RayDifferential::hasDifferentials)
        .def_readwrite("rxOrigin", &RayDifferential::rxOrigin)
        .def_readwrite("ryOrigin", &RayDifferential::ryOrigin)
        .def_readwrite("rxDirection", &RayDifferential::rxDirection)
        .def_readwrite("ryDirection", &RayDifferential::ryDirection);

    // spherical.h
    m.def("CosTheta", CosTheta);
    m.def("Cos2Theta", Cos2Theta);
    m.def("AbsCosTheta", AbsCosTheta);
    m.def("Sin2Theta", Sin2Theta);
    m.def("SinTheta", SinTheta);
    m.def("TanTheta", TanTheta);
    m.def("Tan2Theta", Tan2Theta);
    m.def("CosPhi", CosPhi);
    m.def("SinPhi", SinPhi);
    m.def("Cos2Phi", Cos2Phi);
    m.def("Sin2Phi", Sin2Phi);
    m.def("CosDPhi", CosDPhi);
    m.def("SameHemisphere",
          (bool (*)(const Vector3f &, const Vector3f &)) & SameHemisphere);
    m.def("SameHemisphere",
          (bool (*)(const Vector3f &, const Normal3f &)) & SameHemisphere);
    m.def("SphericalDirection", SphericalDirection, "sinTheta"_a, "cosTheta"_a, "phi"_a);
    m.def("SphericalTheta", SphericalTheta);
    m.def("SphericalPhi", SphericalPhi);

    py::class_<DirectionCone>(m, "DirectionCone")
        .def(py::init<>())
        .def(py::init<Vector3f, Float>(), "w"_a, "cosTheta"_a)
        .def("EntireSphere", &DirectionCone::EntireSphere)
        .def("__str__", &DirectionCone::ToString)
        .def_readwrite("w", &DirectionCone::w)
        .def_readwrite("cosTheta", &DirectionCone::cosTheta);
    m.def("Inside", (bool (*)(const DirectionCone &, const Vector3f &)) & Inside);
    m.def("Union",
          (DirectionCone(*)(const DirectionCone &, const DirectionCone &)) & Union);
    m.def(
        "BoundSubtendedDirections",
        (DirectionCone(*)(const Bounds3f &, const Point3f &)) & BoundSubtendedDirections);

    // splines.h
    m.def(
        "BlossomCubicBezier",
        [](std::array<Point3f, 4> p, Float u0, Float u1, Float u2) {
            return BlossomCubicBezier(MakeConstSpan(p), u0, u1, u2);
        },
        "p"_a, "u0"_a, "u1"_a, "u2"_a);
    m.def(
        "SubdivideCubicBezier",
        [](std::array<Point3f, 4> p) { return SubdivideCubicBezier(MakeConstSpan(p)); },
        "p"_a);
    m.def(
        "EvaluateCubicBezier",
        [](std::array<Point3f, 4> cp, Float u) -> std::tuple<Point3f, Vector3f> {
            Vector3f v;
            Point3f p = EvaluateCubicBezier(MakeConstSpan(cp), u, &v);
            return std::make_tuple(p, v);
        },
        "cp"_a, "u"_a);
    m.def(
        "CubicBezierControlPoints",
        [](std::array<Point3f, 4> p, Float uMin, Float uMax) {
            return CubicBezierControlPoints(MakeConstSpan(p), uMin, uMax);
        },
        "p"_a, "uMin"_a, "uMax"_a);
    m.def(
        "BoundCubicBezier",
        [](std::array<Point3f, 4> p, Float uMin, Float uMax) {
            return BoundCubicBezier<Bounds3f>(MakeConstSpan(p), uMin, uMax);
        },
        "p"_a, "uMin"_a, "uMax"_a);
    m.def(
        "ElevateQuadraticBezierToCubic",
        [](std::array<Point3f, 3> p) {
            return ElevateQuadraticBezierToCubic(MakeConstSpan(p));
        },
        "p"_a);
    m.def(
        "QuadraticBSplineToBezier",
        [](std::array<Point3f, 3> p) {
            return QuadraticBSplineToBezier(MakeConstSpan(p));
        },
        "p"_a);
    m.def(
        "CubicBSplineToBezier",
        [](std::array<Point3f, 4> p) { return CubicBSplineToBezier(MakeConstSpan(p)); },
        "p"_a);

    // transform.h
    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def(py::init<SquareMatrix<4>>())
        .def(py::init<SquareMatrix<4>, SquareMatrix<4>>())
        .def("__eq__", &Transform::operator==)
        .def("IsIdentity", &Transform::IsIdentity)
        .def("GetMatrix", &Transform::GetMatrix)
        .def("GetInverseMatrix", &Transform::GetInverseMatrix)
        .def("HasScale", &Transform::HasScale)
        .def("__call__",
             (Point3f(Transform::*)(const Point3f &) const) & Transform::operator())
        .def("__call__", (Point3f(Transform::*)(const Point3f &, Vector3f *) const) &
                             Transform::operator())
        .def("__call__", (Point3f(Transform::*)(const Point3f &, const Vector3f &,
                                                Vector3f *) const) &
                             Transform::operator())
        .def("__call__",
             (Vector3f(Transform::*)(const Vector3f &) const) & Transform::operator())
        .def("__call__", (Vector3f(Transform::*)(const Vector3f &, Vector3f *) const) &
                             Transform::operator())
        .def("__call__", (Vector3f(Transform::*)(const Vector3f &, const Vector3f &,
                                                 Vector3f *) const) &
                             Transform::operator())
        .def("__call__",
             (Normal3f(Transform::*)(const Normal3f &) const) & Transform::operator())
        .def("__call__",
             (Ray(Transform::*)(const Ray &, Float *) const) & Transform::operator())
        .def("__call__", (Ray(Transform::*)(const Ray &, Vector3f *, Vector3f *) const) &
                             Transform::operator())
        .def("__call__",
             (Ray(Transform::*)(const Ray &, const Vector3f &, const Vector3f &,
                                Vector3f *, Vector3f *) const) &
                 Transform::operator())
        .def("__call__",
             (RayDifferential(Transform::*)(const RayDifferential &, Float *) const) &
                 Transform::operator())
        .def("__call__",
             (Bounds3f(Transform::*)(const Bounds3f &) const) & Transform::operator())
        .def("__call__", (Interaction(Transform::*)(const Interaction &) const) &
                             Transform::operator())
        .def("ApplyInverse",
             (Point3f(Transform::*)(const Point3f &) const) & Transform::ApplyInverse)
        .def("ApplyInverse",
             (Vector3f(Transform::*)(const Vector3f &) const) & Transform::ApplyInverse)
        .def("ApplyInverse",
             (Normal3f(Transform::*)(const Normal3f &) const) & Transform::ApplyInverse)
        .def("ApplyInverse",
             (Ray(Transform::*)(const Ray &, Float *) const) & Transform::ApplyInverse)
        .def("ApplyInverse",
             (RayDifferential(Transform::*)(const RayDifferential &, Float *) const) &
                 Transform::ApplyInverse)
        .def("ApplyInverse", (Interaction(Transform::*)(const Interaction &) const) &
                                 Transform::ApplyInverse)
        .def(py::self * py::self)
        .def("SwapsHandedness", &Transform::SwapsHandedness)
        .def("Hash", &Transform::Hash)
        .def("Decompose", &Transform::Decompose)
        .def("__str__", &Transform::ToString);

    m.def("Translate", Translate);
    m.def("Scale", Scale);
    m.def("RotateX", RotateX);
    m.def("RotateY", RotateY);
    m.def("RotateZ", RotateZ);
    m.def("Rotate", Rotate);
    m.def("LookAt", LookAt);
    m.def("Orthographic", Orthographic);
    m.def("Perspective", Perspective);

    m.def("Inverse", (Transform(*)(const Transform &)) & Inverse);
    m.def("Transpose", (Transform(*)(const Transform &)) & Transpose);

    py::class_<AnimatedTransform>(m, "AnimatedTransform")
        .def(py::init<const Transform *>())
        .def(py::init<const Transform *, Float, const Transform *, Float>(),
             "startTransform"_a, "startTime"_a, "endTransform"_a, "endTime"_a)
        .def("Interpolate", &AnimatedTransform::Interpolate)
        .def("__call__", (Ray(AnimatedTransform::*)(const Ray &, Float *) const) &
                             AnimatedTransform::operator())
        .def("__call__", (RayDifferential(AnimatedTransform::*)(const RayDifferential &,
                                                                Float *) const) &
                             AnimatedTransform::operator())
        .def("__call__",
             (Point3f(AnimatedTransform::*)(const Point3f &, Float) const) &
                 AnimatedTransform::operator(),
             "p"_a, "time"_a)
        .def("__call__",
             (Vector3f(AnimatedTransform::*)(const Vector3f &, Float) const) &
                 AnimatedTransform::operator(),
             "v"_a, "time"_a)
        .def("__call__", (Interaction(AnimatedTransform::*)(const Interaction &) const) &
                             AnimatedTransform::operator())
        .def("ApplyInverse",
             (Point3f(AnimatedTransform::*)(const Point3f &, Float) const) &
                 AnimatedTransform::ApplyInverse,
             "p"_a, "time"_a)
        .def("ApplyInverse",
             (Vector3f(AnimatedTransform::*)(const Vector3f &, Float) const) &
                 AnimatedTransform::ApplyInverse,
             "v"_a, "time"_a)
        .def("ApplyInverse",
             (Interaction(AnimatedTransform::*)(const Interaction &) const) &
                 AnimatedTransform::ApplyInverse)
        .def("HasScale", &AnimatedTransform::HasScale)
        .def("IsAnimated", &AnimatedTransform::IsAnimated)
        .def("MotionBounds", &AnimatedTransform::MotionBounds)
        .def("BoundPointMotion", &AnimatedTransform::BoundPointMotion)
        .def_readonly("startTransform", &AnimatedTransform::startTransform)
        .def_readonly("startTime", &AnimatedTransform::startTime)
        .def_readonly("endTransform", &AnimatedTransform::endTransform)
        .def_readonly("endTime", &AnimatedTransform::endTime);

    // tuple.h / vecmath.h
    py::class_<Vector2f>(m, "Vector2f")
        .def(py::init<>())
        .def(py::init<Float, Float>())
        .def(py::init<Point2f>())
        .def(py::init<Vector2i>())
        .def_readwrite("x", &Vector2f::x)
        .def_readwrite("y", &Vector2f::y)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * float())
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self /= float())
        .def("__neg__", [](const Vector2f &p) { return -p; })
        .def("__getitem__",
             [](const Vector2f &p, int i) { return p[i]; })  // py::is_operator()??
        .def("__setitem__", [](Vector2f &p, int i, Float v) { p[i] = v; })
        .def("__str__", &Vector2f::ToString)
        .def("__repr__",
             [](const Vector2f &p) { return "<pbrt.Vector2f " + p.ToString() + ">"; });

    // tuple.h stuff
    m.def("Abs", [](const Vector2f &v) { return Abs(v); });
    m.def("Ceil", [](const Vector2f &v) { return Ceil(v); });
    m.def("Floor", [](const Vector2f &v) { return Floor(v); });
    m.def("Lerp", [](Float t, const Vector2f &a, const Vector2f &b) {
        return (1 - t) * a + t * b;
    });
    m.def("Min", [](const Vector2f &a, const Vector2f &b) { return Min(a, b); });
    m.def("Max", [](const Vector2f &a, const Vector2f &b) { return Max(a, b); });
    m.def("MinComponentIndex", [](const Vector2f &a) { return MinComponentIndex(a); });
    m.def("MaxComponentIndex", [](const Vector2f &a) { return MaxComponentIndex(a); });
    m.def("MinComponentValue", [](const Vector2f &a) { return MinComponentValue(a); });
    m.def("MaxComponentValue", [](const Vector2f &a) { return MaxComponentValue(a); });

    // vecmath.h stuff
    m.def("Dot", [](const Vector2f &a, const Vector2f &b) { return Dot(a, b); });
    m.def("AbsDot", [](const Vector2f &a, const Vector2f &b) { return AbsDot(a, b); });
    m.def("LengthSquared", [](const Vector2f &v) { return LengthSquared(v); });
    m.def("Length", [](const Vector2f &v) { return Length(v); });
    m.def("Normalize", [](const Vector2f &v) { return Normalize(v); });

    py::class_<Point2f>(m, "Point2f")
        .def(py::init<>())
        .def(py::init<Float, Float>())
        .def(py::init<Vector2f>())
        .def(py::init<Point2i>())
        .def_readwrite("x", &Point2f::x)
        .def_readwrite("y", &Point2f::y)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self + Vector2f())
        .def(py::self += Vector2f())
        .def(py::self - py::self)
        .def(py::self - Vector2f())
        .def(py::self -= Vector2f())
        .def(py::self * float())
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self /= float())
        .def("__neg__", [](const Point2f &p) { return -p; })
        .def("__getitem__",
             [](const Point2f &p, int i) { return p[i]; })  // py::is_operator()??
        .def("__setitem__", [](Point2f &p, int i, Float v) { p[i] = v; })
        .def("__str__", &Point2f::ToString)
        .def("__repr__",
             [](const Point2f &p) { return "<pbrt.Point2f " + p.ToString() + ">"; });

    // tuple.h stuff
    m.def("Abs", [](const Point2f &v) { return Abs(v); });
    m.def("Ceil", [](const Point2f &v) { return Ceil(v); });
    m.def("Floor", [](const Point2f &v) { return Floor(v); });
    m.def("Lerp", [](Float t, const Point2f &a, const Point2f &b) {
        return (1 - t) * a + t * b;
    });
    m.def("Min", [](const Point2f &a, const Point2f &b) { return Min(a, b); });
    m.def("Max", [](const Point2f &a, const Point2f &b) { return Max(a, b); });
    m.def("MinComponentIndex", [](const Point2f &a) { return MinComponentIndex(a); });
    m.def("MaxComponentIndex", [](const Point2f &a) { return MaxComponentIndex(a); });
    m.def("MinComponentValue", [](const Point2f &a) { return MinComponentValue(a); });
    m.def("MaxComponentValue", [](const Point2f &a) { return MaxComponentValue(a); });

    // vecmath.h stuff
    m.def("Distance", [](const Point2f &a, const Point2f &b) { return Distance(a, b); });
    m.def("DistanceSquared",
          [](const Point2f &a, const Point2f &b) { return DistanceSquared(a, b); });

    py::class_<Point2i>(m, "Point2i")
        .def(py::init<>())
        .def(py::init<Float, Float>())
        .def(py::init<Vector2i>())
        .def(py::init<Point2i>())
        .def_readwrite("x", &Point2i::x)
        .def_readwrite("y", &Point2i::y)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self + Vector2i())
        .def(py::self += Vector2i())
        .def(py::self - py::self)
        .def(py::self - Vector2i())
        .def(py::self -= Vector2i())
        .def(py::self * float())
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self /= float())
        .def("__neg__", [](const Point2i &p) { return -p; })
        .def("__getitem__",
             [](const Point2i &p, int i) { return p[i]; })  // py::is_operator()??
        .def("__setitem__", [](Point2i &p, int i, Float v) { p[i] = v; })
        .def("__str__", &Point2i::ToString)
        .def("__repr__",
             [](const Point2i &p) { return "<pbrt.Point2i " + p.ToString() + ">"; });

    // tuple.h stuff
    m.def("Abs", [](const Point2i &v) { return Abs(v); });
    m.def("Lerp", [](Float t, const Point2i &a, const Point2i &b) {
        return (1 - t) * a + t * b;
    });
    m.def("Min", [](const Point2i &a, const Point2i &b) { return Min(a, b); });
    m.def("Max", [](const Point2i &a, const Point2i &b) { return Max(a, b); });
    m.def("MinComponentIndex", [](const Point2i &a) { return MinComponentIndex(a); });
    m.def("MaxComponentIndex", [](const Point2i &a) { return MaxComponentIndex(a); });
    m.def("MinComponentValue", [](const Point2i &a) { return MinComponentValue(a); });
    m.def("MaxComponentValue", [](const Point2i &a) { return MaxComponentValue(a); });

    // vecmath.h stuff
    m.def("Distance", [](const Point2i &a, const Point2i &b) { return Distance(a, b); });
    m.def("DistanceSquared",
          [](const Point2i &a, const Point2i &b) { return DistanceSquared(a, b); });

    py::class_<Vector3f>(m, "Vector3f")
        .def(py::init<>())
        .def(py::init<Float, Float, Float>())
        .def(py::init<Point3f>())
        .def(py::init<Vector3i>())
        .def_readwrite("x", &Vector3f::x)
        .def_readwrite("y", &Vector3f::y)
        .def_readwrite("z", &Vector3f::z)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * float())
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self /= float())
        .def("__neg__", [](const Vector3f &p) { return -p; })
        .def("__getitem__",
             [](const Vector3f &p, int i) { return p[i]; })  // py::is_operator()??
        .def("__setitem__", [](Vector3f &p, int i, Float v) { p[i] = v; })
        .def("__str__", &Vector3f::ToString)
        .def("__repr__",
             [](const Vector3f &p) { return "<pbrt.Vector3f " + p.ToString() + ">"; });

    // tuple.h stuff
    m.def("Abs", [](const Vector3f &v) { return Abs(v); });
    m.def("Ceil", [](const Vector3f &v) { return Ceil(v); });
    m.def("Floor", [](const Vector3f &v) { return Floor(v); });
    m.def("Lerp", [](Float t, const Vector3f &a, const Vector3f &b) {
        return (1 - t) * a + t * b;
    });
    m.def("Min", [](const Vector3f &a, const Vector3f &b) { return Min(a, b); });
    m.def("Max", [](const Vector3f &a, const Vector3f &b) { return Max(a, b); });
    m.def("MinComponentIndex", [](const Vector3f &a) { return MinComponentIndex(a); });
    m.def("MaxComponentIndex", [](const Vector3f &a) { return MaxComponentIndex(a); });
    m.def("MinComponentValue", [](const Vector3f &a) { return MinComponentValue(a); });
    m.def("MaxComponentValue", [](const Vector3f &a) { return MaxComponentValue(a); });

    // vecmath.h stuff
    m.def("Dot", [](const Vector3f &a, const Vector3f &b) { return Dot(a, b); });
    m.def("AbsDot", [](const Vector3f &a, const Vector3f &b) { return AbsDot(a, b); });
    m.def("Cross", [](const Vector3f &a, const Vector3f &b) { return Cross(a, b); });
    m.def("LengthSquared", [](const Vector3f &v) { return LengthSquared(v); });
    m.def("Length", [](const Vector3f &v) { return Length(v); });
    m.def("Normalize", [](const Vector3f &v) { return Normalize(v); });
    m.def("AngleBetween",
          [](const Vector3f &a, const Vector3f &b) { return AngleBetween(a, b); });

    py::class_<Point3f>(m, "Point3f")
        .def(py::init<>())
        .def(py::init<Float, Float, Float>())
        .def(py::init<Vector3f>())
        .def(py::init<Point3i>())
        .def_readwrite("x", &Point3f::x)
        .def_readwrite("y", &Point3f::y)
        .def_readwrite("z", &Point3f::z)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self + Vector3f())
        .def(py::self += Vector3f())
        .def(py::self - py::self)
        .def(py::self - Vector3f())
        .def(py::self -= Vector3f())
        .def(py::self * float())
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self /= float())
        .def("__neg__", [](const Point3f &p) { return -p; })
        .def("__getitem__",
             [](const Point3f &p, int i) { return p[i]; })  // py::is_operator()??
        .def("__setitem__", [](Point3f &p, int i, Float v) { p[i] = v; })
        .def("__str__", &Point3f::ToString)
        .def("__repr__",
             [](const Point3f &p) { return "<pbrt.Point3f " + p.ToString() + ">"; });

    // tuple.h stuff
    m.def("Abs", [](const Point3f &v) { return Abs(v); });
    m.def("Ceil", [](const Point3f &v) { return Ceil(v); });
    m.def("Floor", [](const Point3f &v) { return Floor(v); });
    m.def("Lerp", [](Float t, const Point3f &a, const Point3f &b) {
        return (1 - t) * a + t * b;
    });
    m.def("Min", [](const Point3f &a, const Point3f &b) { return Min(a, b); });
    m.def("Max", [](const Point3f &a, const Point3f &b) { return Max(a, b); });
    m.def("MinComponentIndex", [](const Point3f &a) { return MinComponentIndex(a); });
    m.def("MaxComponentIndex", [](const Point3f &a) { return MaxComponentIndex(a); });
    m.def("MinComponentValue", [](const Point3f &a) { return MinComponentValue(a); });
    m.def("MaxComponentValue", [](const Point3f &a) { return MaxComponentValue(a); });

    // vecmath.h stuff
    m.def("Distance", [](const Point3f &a, const Point3f &b) { return Distance(a, b); });
    m.def("DistanceSquared",
          [](const Point3f &a, const Point3f &b) { return DistanceSquared(a, b); });
    m.def("CoordinateSystem", [](const Vector3f &a) {
        Vector3f b, c;
        CoordinateSystem(a, &b, &c);
        return std::make_tuple(a, b, c);
    });

    py::class_<Normal3f>(m, "Normal3f")
        .def(py::init<>())
        .def(py::init<Float, Float, Float>())
        .def(py::init<Vector3f>())
        .def_readwrite("x", &Normal3f::x)
        .def_readwrite("y", &Normal3f::y)
        .def_readwrite("z", &Normal3f::z)
        .def(py::self + py::self)
        .def(py::self += py::self)
        .def(py::self - py::self)
        .def(py::self -= py::self)
        .def(py::self * float())
        .def(py::self *= float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self /= float())
        .def("__neg__", [](const Normal3f &p) { return -p; })
        .def("__getitem__",
             [](const Normal3f &p, int i) { return p[i]; })  // py::is_operator()??
        .def("__setitem__", [](Normal3f &p, int i, Float v) { p[i] = v; })
        .def("__str__", &Normal3f::ToString)
        .def("__repr__",
             [](const Normal3f &p) { return "<pbrt.Normal3f " + p.ToString() + ">"; });

    // tuple.h stuff
    m.def("Abs", [](const Normal3f &v) { return Abs(v); });
    m.def("Ceil", [](const Normal3f &v) { return Ceil(v); });
    m.def("Floor", [](const Normal3f &v) { return Floor(v); });
    m.def("Lerp", [](Float t, const Normal3f &a, const Normal3f &b) {
        return (1 - t) * a + t * b;
    });
    m.def("Min", [](const Normal3f &a, const Normal3f &b) { return Min(a, b); });
    m.def("Max", [](const Normal3f &a, const Normal3f &b) { return Max(a, b); });
    m.def("MinComponentIndex", [](const Normal3f &a) { return MinComponentIndex(a); });
    m.def("MaxComponentIndex", [](const Normal3f &a) { return MaxComponentIndex(a); });
    m.def("MinComponentValue", [](const Normal3f &a) { return MinComponentValue(a); });
    m.def("MaxComponentValue", [](const Normal3f &a) { return MaxComponentValue(a); });

    // vecmath.h stuff
    m.def("Dot", [](const Normal3f &a, const Vector3f &b) { return Dot(a, b); });
    m.def("Dot", [](const Vector3f &a, const Normal3f &b) { return Dot(a, b); });
    m.def("AbsDot", [](const Normal3f &a, const Vector3f &b) { return AbsDot(a, b); });
    m.def("AbsDot", [](const Vector3f &a, const Normal3f &b) { return AbsDot(a, b); });
    m.def("LengthSquared", [](const Normal3f &v) { return LengthSquared(v); });
    m.def("Length", [](const Normal3f &v) { return Length(v); });
    m.def("Normalize", [](const Normal3f &v) { return Normalize(v); });
    m.def("FaceForward",
          [](const Normal3f &a, const Normal3f &b) { return FaceForward(a, b); });
    m.def("FaceForward",
          [](const Normal3f &a, const Vector3f &b) { return FaceForward(a, b); });
    m.def("FaceForward",
          [](const Vector3f &a, const Normal3f &b) { return FaceForward(a, b); });
    m.def("FaceForward",
          [](const Vector3f &a, const Vector3f &b) { return FaceForward(a, b); });
    m.def("CoordinateSystem", [](const Normal3f &a) {
        Vector3f b, c;
        CoordinateSystem(a, &b, &c);
        return std::make_tuple(a, b, c);
    });

    // TODO: OffsetRayOrigin
}

}  // namespace pbrt
