
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include <pbrt/sampling/lowdiscrepancy.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitSamplingLowDiscrepancy(py::module &m) {
    m.def("RadicalInverse", &RadicalInverse,
          "baseIndex"_a, "a"_a);
    m.def("ComputeRadicalInversePermutations", &ComputeRadicalInversePermutations,
          "rng"_a);
    m.def("ScrambledRadicalInverse",
          [](int baseIndex, uint64_t a, const std::vector<uint16_t> &perm) {
              return ScrambledRadicalInverse(baseIndex, a, perm);
          },
          "baseIndex"_a, "a"_a, "perm"_a);
    m.def("MultiplyGenerator",
          [](const std::vector<uint32_t> &C, uint32_t a) {
              return MultiplyGenerator(C, a);
          },
          "C"_a, "a"_a);
    m.def("SampleGeneratorMatrix",
          [](const std::vector<uint32_t> &C, uint32_t a,
             std::function<uint32_t(uint32_t)> randomizer) {
              return SampleGeneratorMatrix(C, a, randomizer);
          },
          "C"_a, "a"_a, "scramble"_a = 0);

    m.def("OwenScramble", &OwenScramble, "v"_a, "hash"_a);

    py::class_<NoRandomizer>(m, "NoRandomizer")
        .def(py::init<>())
        .def("__call__", [](const NoRandomizer &s, uint32_t v) { return s(v); });
    py::class_<CranleyPattersonRotator>(m, "CranleyPattersonRotator")
        .def(py::init<Float>())
        .def("__call__", [](const CranleyPattersonRotator &s, uint32_t v) { return s(v); });
    py::class_<XORScrambler>(m, "XORScrambler")
        .def(py::init<uint32_t>(), "scramble"_a)
        .def("__call__", [](const XORScrambler &s, uint32_t v) { return s(v); });
    py::class_<OwenScrambler>(m, "OwenScrambler")
        .def(py::init<uint32_t>(), "hash"_a)
        .def("__call__", [](const OwenScrambler &s, uint32_t v) { return s(v); });
}

} // namespace pbrt
