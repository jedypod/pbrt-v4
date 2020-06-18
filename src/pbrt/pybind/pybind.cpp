// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace pbrt {

extern void PybindInitGeometry(py::module &m);
extern void PybindInitSamplingSampling(py::module &m);
extern void PybindInitSamplingLowDiscrepancy(py::module &m);
extern void PybindInitSamplingSobol(py::module &m);
extern void PybindInitSamplersSamplers(py::module &m);
extern void PybindInitSpectrum(py::module &m);
extern void PybindInitShapes(py::module &m);
extern void PybindInitUtil(py::module &m);

}  // namespace pbrt

PYBIND11_MODULE(pbrt, m) {
    m.doc() = "pbrt-v4 plugin";

    pbrt::PybindInitGeometry(m);
    pbrt::PybindInitSamplingSampling(m);
    pbrt::PybindInitSamplingLowDiscrepancy(m);
    pbrt::PybindInitSamplingSobol(m);
    pbrt::PybindInitSamplersSamplers(m);
    pbrt::PybindInitSpectrum(m);
    pbrt::PybindInitShapes(m);
    pbrt::PybindInitUtil(m);
}
