// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/core/interaction.h>
#include <pbrt/core/paramdict.h>
#include <pbrt/shapes/cone.h>
#include <pbrt/shapes/curve.h>
#include <pbrt/shapes/cylinder.h>
#include <pbrt/shapes/disk.h>
#include <pbrt/shapes/hyperboloid.h>
#include <pbrt/shapes/loopsubdiv.h>
#include <pbrt/shapes/nurbs.h>
#include <pbrt/shapes/paraboloid.h>
#include <pbrt/shapes/plymesh.h>
#include <pbrt/shapes/sphere.h>
#include <pbrt/shapes/triangle.h>
#include <pbrt/util/memory.h>

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitShapes(py::module &m) {
    py::class_<Cone, TransformedShape>(m, "Cone").def(
        py::init<const Transform *, const Transform *, bool, Float, Float, Float,
                 const std::shared_ptr<const ParameterDictionary> &>(),
        "worldFromObject"_a, "objectFromWorld"_a, "reverseOrientation"_a = false,
        "height"_a = 1, "radius"_a = 1, "phiMax"_a = 2 * Pi, "attributes"_a = nullptr);

    py::class_<Curve, Shape>(m, "Curve")
        .def(py::init<const CurveCommon *, Float, Float>(), "common"_a, "uMin"_a = 0,
             "uMax"_a = 1);
    py::class_<CurveCommon>(m, "CurveCommon")
        .def(py::init([](std::array<Point3f, 4> cp, Float w0, Float w1, CurveType type,
                         std::array<Normal3f, 2> norm, const Transform *worldFromObject,
                         const Transform *objectFromWorld, bool reverseOrientation,
                         const std::shared_ptr<const ParameterDictionary> &attributes) {
            return CurveCommon(MakeConstSpan(cp), w0, w1, type, MakeConstSpan(norm),
                               worldFromObject, objectFromWorld, reverseOrientation,
                               attributes);
        }));
    py::enum_<CurveType>(m, "CurveType")
        .value("Flat", CurveType::Flat)
        .value("Cylinder", CurveType::Cylinder)
        .value("Ribbon", CurveType::Ribbon);

    py::class_<Cylinder, TransformedShape>(m, "Cylinder")
        .def(py::init<const Transform *, const Transform *, bool, Float, Float, Float,
                      Float, const std::shared_ptr<const ParameterDictionary> &>(),
             "worldFromObject"_a, "objectFromWorld"_a, "reverseOrientation"_a = false,
             "radius"_a = 1, "zMin"_a = 0, "zMax"_a = 1, "phiMax"_a = 2 * Pi,
             "attributes"_a = nullptr);

    py::class_<Disk, TransformedShape>(m, "Disk").def(
        py::init<const Transform *, const Transform *, bool, Float, Float, Float, Float,
                 const std::shared_ptr<const ParameterDictionary> &>(),
        "worldFromObject"_a, "objectFromWorld"_a, "reverseOrientation"_a = false,
        "height"_a = 1, "radius"_a = 1, "innerRadius"_a = 0, "phiMax"_a = 2 * Pi,
        "attributes"_a = nullptr);

    py::class_<Hyperboloid, TransformedShape>(m, "Hyperboloid")
        .def(py::init<const Transform *, const Transform *, bool, const Point3f &,
                      const Point3f &, Float,
                      const std::shared_ptr<const ParameterDictionary> &>(),
             "worldFromObject"_a, "objectFromWorld"_a, "reverseOrientation"_a = false,
             "p1"_a = Point3f(0, 0, 0), "p2"_a = Point3f(1, 1, 1), "phiMax"_a = 2 * Pi,
             "attributes"_a = nullptr);

    py::class_<Paraboloid, TransformedShape>(m, "Paraboloid")
        .def(py::init<const Transform *, const Transform *, bool, Float, Float, Float,
                      Float, const std::shared_ptr<const ParameterDictionary> &>(),
             "worldFromObject"_a, "objectFromWorld"_a, "reverseOrientation"_a = false,
             "radius"_a = 1, "z0"_a = 0, "z1"_a = 1, "phiMax"_a = 2 * Pi,
             "attributes"_a = nullptr);

    py::class_<Sphere, TransformedShape>(m, "Sphere")
        .def(py::init<const Transform *, const Transform *, bool, Float, Float, Float,
                      Float, const std::shared_ptr<const ParameterDictionary> &>(),
             "worldFromObject"_a, "objectFromWorld"_a, "reverseOrientation"_a = false,
             "radius"_a = 1, "zMin"_a = 0, "zMax"_a = 1, "phiMax"_a = 2 * Pi,
             "attributes"_a = nullptr);

    m.def("CreateLoopSubdiv", CreateLoopSubdiv, "worldFromObject"_a,
          "reverseOrientation"_a = false, "params"_a, "attributes"_a, "arena"_a);
    m.def("CreateNURBS", CreateNURBS, "worldFromObject"_a, "reverseOrientation"_a = false,
          "params"_a, "attributes"_a, "arena"_a);
    m.def("CreatePLYMesh", CreatePLYMesh, "worldFromObject"_a,
          "reverseOrientation"_a = false, "params"_a, "attributes"_a, "arena"_a);

    py::class_<Triangle>(m, "Triangle")
        .def(py::init<int, int>(), "meshIndex"_a, "triIndex"_a)
        .def("CameraWorldBound", &Triangle::CameraWorldBound)
        .def("Intersect", &Triangle::Intersect)
        .def("IntersectP", &Triangle::IntersectP)
        .def("Area", &Triangle::Area)
        .def("Sample", (Optional<ShapeSample>(Triangle::*)(const Interaction &,
                                                           const Point2f &) const) &
                           Triangle::Sample)
        .def("Sample", (Optional<ShapeSample>(Triangle::*)(const Point2f &) const) &
                           Triangle::Sample)
        .def("Pdf", (Float(Triangle::*)(const Interaction &, const Vector3f &) const) &
                        Triangle::Pdf)
        .def("Pdf", (Float(Triangle::*)(const Interaction &) const) & Triangle::Pdf)
        .def("SolidAngle", &Triangle::SolidAngle)
        .def("OrientationIsReversed", &Triangle::OrientationIsReversed)
        .def("TransformSwapsHandedness", &Triangle::TransformSwapsHandedness)
        .def("GetAttributes", &Triangle::GetAttributes);
}

}  // namespace pbrt
