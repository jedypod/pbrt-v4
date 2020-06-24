
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/functional.h>

#include <pbrt/sampling/sobol.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitSamplingSobol(py::module &m) {
    m.def("SobolIntervalToIndex", &SobolIntervalToIndex,
          "log2Resolution"_a, "sampleNum"_a, "p"_a);
    m.def("SobolSampleFloat",
          [](int64_t index, int dimension, std::function<uint32_t(uint32_t)> randomizer) {
              return SobolSampleFloat(index, dimension, randomizer);
          },
          "index"_a, "dimension"_a, "randomizer"_a);
    m.def("SobolSampleDouble",
          [](int64_t index, int dimension, std::function<uint32_t(uint32_t)> randomizer) {
              return SobolSampleDouble(index, dimension, randomizer);
          },
          "index"_a, "dimension"_a, "randomizer"_a);
    m.def("SobolSample",
          [](int64_t index, int dimension, std::function<uint32_t(uint32_t)> randomizer) {
              return SobolSample(index, dimension, randomizer);
          },
          "index"_a, "dimension"_a, "randomizer"_a);
    m.def("SobolSampleBits32", &SobolSampleBits32,
          "a"_a, "dimension"_a);
    m.def("SobolSampleBits64", &SobolSampleBits64,
          "a"_a, "dimension"_a);
}

} // namespace pbrt
