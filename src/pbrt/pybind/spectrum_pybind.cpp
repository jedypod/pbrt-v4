// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pbrt/core/types.h>
#include <pbrt/spectrum/encoding.h>
#include <pbrt/spectrum/rgb.h>
#include <pbrt/spectrum/rgbcolorspace.h>
#include <pbrt/spectrum/rgbspectrum.h>
#include <pbrt/spectrum/sampled.h>
#include <pbrt/spectrum/spds.h>
#include <pbrt/spectrum/spectrum.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitSpectrum(py::module &m) {
    // Spectrum: also early...
    py::class_<Spectrum>(m, "Spectrum")
        .def("__call__", &Spectrum::operator())
        .def("Sample", &Spectrum::Sample)
        .def("MaxValue", &Spectrum::MaxValue)
        .def("__str__", &Spectrum::ToString)
        .def("ParameterType", &Spectrum::ParameterType)
        .def("ParameterString", &Spectrum::ParameterString);

    // rgb.h (do this early)
    py::class_<RGB>(m, "RGB")
        .def(py::init<>())
        .def(py::init<Float, Float, Float>())
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self -= py::self)
        .def(py::self - py::self)
        .def(py::self *= py::self)
        .def(py::self * py::self)
        .def(py::self /= py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(Float() - py::self)
        .def(Float() * py::self)
        .def(py::self * Float())
        .def(py::self *= Float())
        .def(py::self / Float())
        .def(py::self /= Float())
        .def("__eq__", &RGB::operator==)
        .def("__getitem__", [](const RGB &rgb, int i) { return rgb[i]; })
        .def("__setitem__", [](RGB &rgb, int i, Float v) { rgb[i] = v; })
        .def("__str__", &RGB::ToString);

    m.def("Lerp", (RGB(*)(Float, const RGB &, const RGB &))Lerp);

    // xyz.h (do this early)
    py::class_<XYZ>(m, "XYZ")
        .def(py::init<>())
        .def(py::init<Float, Float, Float>())
        .def("FromxyY", &XYZ::FromxyY)
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self -= py::self)
        .def(py::self - py::self)
        .def(py::self *= py::self)
        .def(py::self * py::self)
        .def(py::self /= py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(Float() - py::self)
        .def(Float() * py::self)
        .def(py::self * Float())
        .def(py::self *= Float())
        .def(py::self / Float())
        .def(py::self /= Float())
        .def("__eq__", &XYZ::operator==)
        .def("__getitem__", [](const XYZ &xyz, int i) { return xyz[i]; })
        .def("__setitem__", [](XYZ &xyz, int i, Float v) { xyz[i] = v; })
        .def("__str__", &XYZ::ToString);

    m.def("Lerp", (XYZ(*)(Float, const XYZ &, const XYZ &))Lerp);

    // rgbcolorspace.h
    py::class_<RGBColorSpace>(m, "RGBColorSpace")
        .def(py::init<Point2f, Point2f, Point2f, Point2f, const Spectrum *,
                      const RGBToSpectrumTable *>())
        .def("ToRGB", &RGBColorSpace::ToRGB)
        .def("ToXYZ", &RGBColorSpace::ToXYZ)
        .def("ToRGBCoeffs", &RGBColorSpace::ToRGBCoeffs)
        .def("GetNamed", &RGBColorSpace::GetNamed)
        .def("ColorCorrectionMatrixForxy", &RGBColorSpace::ColorCorrectionMatrixForxy)
        .def("ColorCorrectionMatrixForIlluminant",
             &RGBColorSpace::ColorCorrectionMatrixForIlluminant)
        .def("__eq__", &RGBColorSpace::operator==)
        .def("__str__", &RGBColorSpace::ToString)
        .def("Lookup", &RGBColorSpace::Lookup)
        .def_readonly_static("ACES2065_1", &RGBColorSpace::ACES2065_1)
        .def_readonly_static("Rec2020", &RGBColorSpace::Rec2020)
        .def_readonly_static("sRGB", &RGBColorSpace::sRGB)
        .def_readwrite("r", &RGBColorSpace::r)
        .def_readwrite("g", &RGBColorSpace::g)
        .def_readwrite("b", &RGBColorSpace::b)
        .def_readwrite("w", &RGBColorSpace::w);

    m.def("ConvertRGBColorSpace", &ConvertRGBColorSpace);

    // encoding.h
    py::class_<ColorEncoding>(m, "ColorEncoding")
        .def("ToLinear",
             [](const ColorEncoding &ce, std::vector<uint8_t> v) {
                 std::vector<Float> vout;
                 ce.ToLinear(MakeConstSpan(v), MakeSpan(vout));
                 return vout;
             })
        .def("ToFloatLinear", &ColorEncoding::ToFloatLinear)
        .def("FromLinear",
             [](const ColorEncoding &ce, std::vector<Float> v) {
                 std::vector<uint8_t> vout;
                 ce.FromLinear(MakeConstSpan(v), MakeSpan(vout));
                 return vout;
             })
        .def("__str__", &ColorEncoding::ToString)
        .def_readonly_static("Linear", ColorEncoding::Linear)
        .def_readonly_static("sRGB", ColorEncoding::sRGB)
        .def_static("Get", &ColorEncoding::Get);

    // rgbspectrum.h
    py::class_<RGBReflectanceSpectrum, Spectrum>(m, "RGBReflectanceSpectrum")
        .def(py::init<const RGBColorSpace &, const RGB &>());
    py::class_<RGBSpectrum, Spectrum>(m, "RGBSpectrum")
        .def(py::init<const RGBColorSpace &, const RGB &>());
    py::class_<RGBSigmoidPolynomial>(m, "RGBSigmoidPolynomial")
        .def(py::init())
        .def(py::init<Float, Float, Float>())
        .def("__call__", &RGBSigmoidPolynomial::operator())
        .def("MaxValue", &RGBSigmoidPolynomial::MaxValue)
        .def("__str__", &RGBSigmoidPolynomial::ToString);
    py::class_<RGBToSpectrumTable>(m, "RGBToSpectrumTable")
        .def("__call__", &RGBToSpectrumTable::operator())
        .def_readonly_static("sRGB", &RGBToSpectrumTable::sRGB)
        .def_readonly_static("Rec2020", &RGBToSpectrumTable::Rec2020)
        .def_readonly_static("ACES2065_1", &RGBToSpectrumTable::ACES2065_1);

    // sampled.h
    py::class_<SampledWavelengths>(m, "SampledWavelengths")
        .def("__getitem__", [](const SampledWavelengths &s, int i) { return s[i]; })
        .def("__eq__", &SampledWavelengths::operator==)
        .def("__str__", &SampledWavelengths::ToString)
        .def("TerminateSecondaryWavelengths",
             &SampledWavelengths::TerminateSecondaryWavelengths)
        .def_static("SampleEqui", &SampledWavelengths::SampleEqui, "u"_a,
                    "lambdaMin"_a = Spectrum::LambdaMin,
                    "lambdaMax"_a = Spectrum::LambdaMax);
    m.attr("NSpectrumSamples") = py::int_(NSpectrumSamples);

    py::class_<SampledSpectrum>(m, "SampledSpectrum")
        .def(py::init())
        .def(py::init<Float>())
        .def(py::self += py::self)
        .def(py::self + py::self)
        .def(py::self -= py::self)
        .def(py::self - py::self)
        .def(py::self *= py::self)
        .def(py::self * py::self)
        .def(py::self /= py::self)
        .def(py::self / py::self)
        .def(-py::self)
        .def(Float() - py::self)
        .def(py::self * Float())
        .def(py::self *= Float())
        .def(Float() * py::self)
        .def(py::self / Float())
        .def(py::self /= Float())
        .def("__eq__", &SampledSpectrum::operator==)
        .def("MinComponentValue", &SampledSpectrum::MinComponentValue)
        .def("MaxComponentValue", &SampledSpectrum::MaxComponentValue)
        .def("Average", &SampledSpectrum::Average)
        .def("HasNaNs", &SampledSpectrum::HasNaNs)
        .def("ToRGB", &SampledSpectrum::ToRGB)
        .def("ToXYZ", &SampledSpectrum::ToXYZ)
        .def("y", &SampledSpectrum::y)
        .def("__getitem__", [](const SampledSpectrum &s, int i) { return s[i]; })
        .def("__setitem__", [](SampledSpectrum &s, int i, Float v) { s[i] = v; });
    m.def("Sqrt", (SampledSpectrum(*)(const SampledSpectrum &))Sqrt);
    m.def("Pow", (SampledSpectrum(*)(const SampledSpectrum &, Float))Pow);
    m.def("Exp", (SampledSpectrum(*)(const SampledSpectrum &))Exp);
    m.def("SaveDiv",
          (SampledSpectrum(*)(const SampledSpectrum &, const SampledSpectrum &))SafeDiv);
    m.def("Clamp", [](const SampledSpectrum &s, Float low, Float high) {
        return Clamp(s, low, high);
    });

    // spds.h
    auto mspds = m.def_submodule("spds");
    mspds.attr("Zero") = SPDs::Zero();
    mspds.attr("One") = SPDs::One();
    mspds.attr("X") = SPDs::X();
    mspds.attr("Y") = SPDs::Y();
    mspds.attr("Z") = SPDs::Z();
    mspds.attr("CIE_Y_integral") = SPDs::CIE_Y_integral;
    mspds.attr("IllumA") = SPDs::IllumA();
    mspds.attr("IllumD50") = SPDs::IllumD50();
    mspds.attr("IllumACESD60") = SPDs::IllumACESD60();
    mspds.attr("IllumD65") = SPDs::IllumD65();
    mspds.attr("IllumF1") = SPDs::IllumF1();
    mspds.attr("IllumF2") = SPDs::IllumF2();
    mspds.attr("IllumF3") = SPDs::IllumF3();
    mspds.attr("IllumF4") = SPDs::IllumF4();
    mspds.attr("IllumF5") = SPDs::IllumF5();
    mspds.attr("IllumF6") = SPDs::IllumF6();
    mspds.attr("IllumF7") = SPDs::IllumF7();
    mspds.attr("IllumF8") = SPDs::IllumF8();
    mspds.attr("IllumF9") = SPDs::IllumF9();
    mspds.attr("IllumF10") = SPDs::IllumF10();
    mspds.attr("IllumF11") = SPDs::IllumF11();
    mspds.attr("IllumF12") = SPDs::IllumF12();
    mspds.attr("MetalAgEta") = SPDs::MetalAgEta();
    mspds.attr("MetalAgK") = SPDs::MetalAgK();
    mspds.attr("MetalAlEta") = SPDs::MetalAlEta();
    mspds.attr("MetalAlK") = SPDs::MetalAlK();
    mspds.attr("MetalAuEta") = SPDs::MetalAuEta();
    mspds.attr("MetalAuK") = SPDs::MetalAuK();
    mspds.attr("MetalCuEta") = SPDs::MetalCuEta();
    mspds.attr("MetalCuK") = SPDs::MetalCuK();
    mspds.attr("MetalMgOEta") = SPDs::MetalMgOEta();
    mspds.attr("MetalMgOK") = SPDs::MetalMgOK();
    mspds.attr("MetalTiO2Eta") = SPDs::MetalTiO2Eta();
    mspds.attr("MetalTiO2K") = SPDs::MetalTiO2K();
    mspds.attr("GlassBK7Eta") = SPDs::GlassBK7Eta();
    mspds.attr("GlassBAF10Eta") = SPDs::GlassBAF10Eta();
    mspds.attr("GlassFK51AEta") = SPDs::GlassFK51AEta();
    mspds.attr("GlassLASF9Eta") = SPDs::GlassLASF9Eta();
    mspds.attr("GlassSF5Eta") = SPDs::GlassSF5Eta();
    mspds.attr("GlassSF10Eta") = SPDs::GlassSF10Eta();
    mspds.attr("GlassSF11Eta") = SPDs::GlassSF11Eta();
    mspds.def("GetNamed", SPDs::GetNamed);
    mspds.def("FindMatchingNamed", SPDs::FindMatchingNamed);

    // spectrum.h
    py::class_<BlackbodySpectrum, Spectrum>(m, "BlackbodySpectrum")
        .def(py::init<Float>());
    py::class_<ConstantSpectrum, Spectrum>(m, "ConstantSpectrum").def(py::init<Float>());
    py::class_<ScaledSpectrum, Spectrum>(m, "ScaledSpectrum")
        .def(py::init<Float, const Spectrum *>());
    py::class_<PiecewiseLinearSpectrum>(m, "PiecewiseLienarSpectrum")
        .def(py::init([](std::vector<Float> lambda, std::vector<Float> values) {
            return new PiecewiseLinearSpectrum(MakeConstSpan(lambda),
                                               MakeConstSpan(values));
        }));
    py::class_<DenselySampledSpectrum, Spectrum>(m, "DenselySampledSpectrum")
        .def(py::init<const Spectrum &, int, int>(), "s"_a, "lambdaStart"_a = 400,
             "lambdaEnd"_a = 700);
}

}  // namespace pbrt
