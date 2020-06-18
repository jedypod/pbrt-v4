// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pbrt/geometry/vecmath.h>
#include <pbrt/sampling/function.h>
#include <pbrt/sampling/geometry.h>
#include <pbrt/sampling/pmj02tables.h>
#include <pbrt/sampling/sampling.h>
#include <pbrt/util/sampling.hierwarp.h>

#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitSamplingSampling(py::module &m) {
    m.def("pmj02bnSamples", [](int set, int sample, int dim) {
        return pmj02bnSamples[set][sample][dim] * 0x1p-32;
    });
    m.def("SampleUniformHemisphere", &SampleUniformHemisphere,
          "Sample a hemisphere with uniform probability", "u"_a);
    m.def("UniformHemispherePdf", &UniformHemispherePdf,
          "PDF for uniform hemisphere sampling");
    m.def("SampleUniformSphere", &SampleUniformSphere,
          "Sample a sphere with uniform probability", "u"_a);
    m.def("UniformSpherePdf", &UniformSpherePdf, "PDF for uniform sphere sampling");
    m.def("SampleUniformCone",
          py::overload_cast<const Point2f &, Float>(&SampleUniformCone),
          "Sample a cone with uniform probability", "u"_a, "thetaMax"_a);
    m.def("UniformConePdf", &UniformConePdf, "PDF for uniform cone sampling",
          "thetaMax"_a);
    m.def("SampleUniformDiskPolar", &SampleUniformDiskPolar,
          "Sample a unit disk with uniform probability", "u"_a);
    m.def("SampleUniformDiskConcentric", &SampleUniformDiskConcentric,
          "Sample a unit disk with uniform probability using a concentric "
          "mapping",
          "u"_a);
    m.def("SampleUniformTriangle", &SampleUniformTriangle,
          "Sample a triangle with uniform probability", "u"_a);
    m.def("SampleSphericalTriangle", &SampleSphericalTriangle,
          "Sample a spherical triangle as seen from the given point with uniform "
          "probability",
          "vertices"_a, "p"_a, "u"_a, "pdf"_a);
    m.def("LowDiscrepancySampleTriangle", &LowDiscrepancySampleTriangle,
          "Sample a triangle using Owen's low discrepancy mapping probability", "u"_a);
    m.def("SampleCosineHemisphere", &SampleCosineHemisphere,
          "Cosine-weighted hemisphere sampling", "u"_a);
    m.def("CosineHemispherePdf", &CosineHemispherePdf,
          "PDF for cosine-weighted hemisphere sampling", "cosTheta"_a);
    m.def("BalanceHeuristic", &BalanceHeuristic,
          "Return MIS weight using the balance heuristic", "nf"_a, "fPdf"_a, "ng"_a,
          "gPdf"_a);
    m.def("PowerHeuristic", &PowerHeuristic,
          "Return MIS weight using the power heuristic", "nf"_a, "fPdf"_a, "ng"_a,
          "gPdf"_a);
    m.def("SampleSmoothstep", &SampleSmoothstep, "Sample smoothstep function", "u"_a,
          "start"_a, "end"_a);
    m.def("SmoothstepPdf", &SmoothstepPdf, "PDF for sampling smoothstep function", "x"_a,
          "start"_a, "end"_a);
    m.def(
        "SampleDiscrete",
        [](const std::vector<Float> &weights, Float u) {
            Float pdf, uRemapped;
            int index = SampleDiscrete(weights, u, &pdf, &uRemapped);
            return std::make_tuple(index, pdf, uRemapped);
        },
        "Sample discrete PDF given by |weights|. Returns tuple (index, pdf, "
        "uRemapped).",
        "weights"_a, "u"_a);
    m.def("SampleLinear", &SampleLinear,
          "Sample the linear function on [0,1] with values |a| at x=0 and |b| "
          "at x=1",
          "u"_a, "a"_a, "b"_a);
    m.def("LinearPdf", &LinearPdf,
          "Return PDF for sampling the offset |x| in [0,1] for a linear function "
          "on [0,1] with values |a| at x=0 and |b| at x=1",
          "x"_a, "a"_a, "b"_a);
    m.def(
        "SampleBilinear",
        [](Point2f u, std::array<Float, 4> v) { return SampleBilinear(u, v); },
        "Sample a bilinear function with given values at the corners of "
        "[0,1]^2",
        "u"_a, "v"_a);
    m.def(
        "BilinearPdf",
        [](Point2f p, std::array<Float, 4> v) { return BilinearPDF(p, v); },
        "Return PDF for sampling the given point in [0,1]^2 for the bilinear "
        "function specified by the given vertices.",
        "p"_a, "v"_a);
    m.def("SampleTent", &SampleTent, "Sample the tent function of given radius", "u"_a,
          "radius"_a);
    m.def("TentPdf", &TentPdf,
          "Return PDF for sampling the given tent function over +/- radius", "x"_a,
          "radius"_a);
    m.def("SampleXYZMatching", &SampleXYZMatching,
          "Sample a PDF that approximates the sum of the X, Y, and Z spectral "
          "matching curves.",
          "u"_a);
    m.def("XYZMatchingPdf", &XYZMatchingPdf,
          "Return PDF for sampling the given wavelength lambda using "
          "SampleXYZMatching.",
          "lambda"_a);
    m.def("SampleNormal", &SampleNormal, "Sample the normal distribution", "u"_a,
          "mu"_a = 0, "sigma"_a = 1);
    m.def("NormalPdf", &NormalPdf, "x"_a, "mu"_a = 0, "sigma"_a = 1);
    m.def("SampleTwoNormal", &SampleTwoNormal,
          "Return two independent samples of the given normal distribution", "u"_a,
          "mu"_a = 0, "sigma"_a = 1);
    m.def("SampleTrimmedLogistic", &SampleTrimmedLogistic, "u"_a, "s"_a, "a"_a, "b"_a);

    py::class_<VarianceEstimator<double>>(m, "VarianceEstimator")
        .def(py::init<>())
        .def(
            "__iadd__",
            [](VarianceEstimator<double> &est, Float v) {
                est.Add(v);
                return est;
            },
            "v"_a)
        .def(
            "__iadd__",
            [](VarianceEstimator<double> &e, const VarianceEstimator<double> &est) {
                e.Add(est);
                return est;
            },
            "est"_a)
        .def("Average", &VarianceEstimator<double>::Average)
        .def("Variance", &VarianceEstimator<double>::Variance)
        .def("RelativeVariance", &VarianceEstimator<double>::RelativeVariance);

    py::class_<Hierarchical2DWarp>(m, "Hierarchical2DWarp")
        .def(py::init<std::vector<Float>, int, int>())
        .def(
            "SampleDiscrete",
            [](const Hierarchical2DWarp &w, Point2f u) {
                Float pdf;
                Point2i p = w.SampleDiscrete(u, &pdf);
                return std::make_pair(p, pdf);
            },
            "u"_a)
        .def("DiscretePdf", &Hierarchical2DWarp::DiscretePdf)
        .def(
            "SampleContinuous",
            [](const Hierarchical2DWarp &w, Point2f u) {
                Float pdf;
                Point2f p = w.SampleContinuous(u, &pdf);
                return std::make_pair(p, pdf);
            },
            "u"_a)
        .def("ContinuousPdf", &Hierarchical2DWarp::ContinuousPdf)
        .def("__str__", &Hierarchical2DWarp::ToString);
}

}  // namespace pbrt
