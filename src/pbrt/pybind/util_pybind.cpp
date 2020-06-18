// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pbrt/util/bits.h>
#include <pbrt/util/efloat.h>
#include <pbrt/util/file.h>
#include <pbrt/util/float.h>
#include <pbrt/util/half.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/interpolation.h>
#include <pbrt/util/interval.h>
#include <pbrt/util/math.h>
#include <pbrt/util/matrix.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/primes.h>
#include <pbrt/util/print.h>
#include <pbrt/util/rng.h>
#include <pbrt/util/shuffle.h>

#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pbrt {

void PybindInitUtil(py::module &m) {
    // bits.h
    m.def("ReverseBits32", &ReverseBits32, "n"_a);
    m.def("ReverseBits64", &ReverseBits64, "n"_a);
    m.def("GrayCode", &GrayCode, "v"_a);
    m.def("CountTrailingZeros", &CountTrailingZeros, "v"_a);
    m.def("PopCount", (int (*)(uint64_t))(&PopCount), "v"_a);
    m.def("LeftShift3", &LeftShift3, "v"_a);
    m.def("EncodeMorton3", &EncodeMorton3, "x"_a, "y"_a, "z"_a);
    m.def("Compact1By1", &Compact1By1, "v"_a);
    m.def("DemuxFloat", &DemuxFloat, "f"_a);
    m.def("MixBits", &MixBits, "v"_a);

    py::class_<EFloat>(m, "EFloat")
        .def(py::init<>())
        .def(py::init<float>())
        .def(py::init<float, float>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)
        .def(-py::self)
        .def("Check", &EFloat::Check)
        .def("GetAbsoluteError", &EFloat::GetAbsoluteError)
        .def("UpperBound", &EFloat::UpperBound)
        .def("LowerBound", &EFloat::LowerBound)
#ifndef NDEBUG
        .def("GetRelativeError", &EFloat::GetRelativeError)
        .def("PreciseValue", &EFloat::PreciseValue)
#endif  // !NDEBUG
        .def("__eq__", &EFloat::operator==)
        .def("__float__", &EFloat::operator float)
        .def("__repr__", [](const EFloat &e) { return e.ToString(); });

    m.def("Sqrt", (EFloat(*)(EFloat))Sqrt, "e"_a);
    m.def("Abs", (EFloat(*)(EFloat))Abs, "e"_a);
    m.def("Quadratic", [](EFloat A, EFloat B, EFloat C) -> std::tuple<EFloat, EFloat> {
        EFloat t0, t1;
        if (!Quadratic(A, B, C, &t0, &t1))
            throw std::domain_error("No solutions to quadratic equation");
        return std::make_tuple(t0, t1);
    });

    // file.h
    m.def("IsAbsolutePath", &IsAbsolutePath, "filename"_a);
    m.def("AbsolutePath", &AbsolutePath, "filename"_a);
    m.def("ResolveFilename", &ResolveFilename, "filename"_a);
    m.def("DirectoryContaining", &DirectoryContaining, "filename"_a);
    m.def("SetSearchDirectory", &SetSearchDirectory, "dir"_a);

    // float.h
    m.attr("OneMinusEpsilon") = OneMinusEpsilon;
    m.attr("MaxFloat") = MaxFloat;
    m.attr("Infinity") = Infinity;
    m.attr("MachineEpsilon") = MachineEpsilon;
    m.def("FloatToBits", (uint32_t(*)(float))(&FloatToBits), "f"_a);
    m.def("BitsToFloat", (float (*)(uint32_t))(&BitsToFloat), "b"_a);
    m.def("NextFloatUp", (float (*)(float, int))(&NextFloatUp), "v"_a, "delta"_a = 1);
    m.def("NextFloatDown", (float (*)(float, int))(&NextFloatDown), "v"_a, "delta"_a = 1);
    m.def("FloatToBits", (uint64_t(*)(double))(&FloatToBits), "f"_a);
    m.def("BitsToFloat", (double (*)(uint64_t))(&BitsToFloat), "b"_a);
    m.def("NextFloatUp", (double (*)(double, int))(&NextFloatUp), "v"_a, "delta"_a = 1);
    m.def("NextFloatDown", (double (*)(double, int))(&NextFloatDown), "v"_a,
          "delta"_a = 1);
    m.def("Exponent", (int (*)(float))(&Exponent), "v"_a);
    m.def("Significand", (int (*)(float))(&Significand), "v"_a);
    m.def("Exponent", (int (*)(double))(&Exponent), "v"_a);
    m.def("Significand", (uint64_t(*)(double))(&Significand), "v"_a);
    m.def("gamma", &gamma, "n"_a);
    py::class_<KahanSum<double>>(m, "KahanSum")
        .def(py::init<>())
        .def(py::init<double>())
        .def(py::self += double())
        .def("__float__", &KahanSum<double>::operator double);

    // half.h
    py::class_<Half>(m, "Half")
        .def(py::init<>())
        .def(py::init<float>())
        .def("FromBits", Half::FromBits)
        .def("__eq__", [](const Half &a, const Half &b) { return a == b; })
        .def("__lt__", [](const Half &a, const Half &b) { return (float)a < (float)b; })
        .def("__le__", [](const Half &a, const Half &b) { return (float)a <= (float)b; })
        .def("__gt__", [](const Half &a, const Half &b) { return (float)a > (float)b; })
        .def("__ge__", [](const Half &a, const Half &b) { return (float)a >= (float)b; })
        .def("__neg__", [](const Half &h) { return -h; })
        .def("__float__", &Half::operator float)
        .def("Bits", &Half::Bits)
        .def("Sign", &Half::Sign)
        .def("IsInf", &Half::IsInf)
        .def("IsNan", &Half::IsNaN)
        .def("NextUp", &Half::NextUp)
        .def("NextDown", &Half::NextDown)
        .def("__repr__", [](const Half &h) {
            return StringPrintf("<pbrt.Half %f [0x%x]>", (float)h, h.Bits());
        });

    // hash.h
    // TODO: figure out what to do about hash buffer
    m.def("HashBuffer", (uint64_t(*)(const void *, size_t))(&HashBuffer), "ptr"_a,
          "size"_a);

    // interpolation.h
    m.def(
        "CatmullRom",
        [](std::vector<Float> nodes, std::vector<Float> values, Float x) {
            return CatmullRom(nodes, values, x);
        },
        "nodes"_a, "values"_a, "x"_a);
    m.def(
        "CatmullRomWeights",
        [](std::vector<Float> nodes, Float x) {
            std::array<Float, 4> weights;
            int offset;
            if (!CatmullRomWeights(nodes, x, &offset, MakeSpan(weights)))
                throw std::invalid_argument("\"x\" value outside of node range");
            return std::make_tuple(weights, offset);
        },
        "nodes"_a, "x"_a);
    m.def(
        "SampleCatmullRom",
        [](std::vector<Float> nodes, std::vector<Float> f, std::vector<Float> cdf,
           Float u) {
            Float fval, pdf;
            Float x = SampleCatmullRom(nodes, f, cdf, u, &fval, &pdf);
            return std::make_tuple(x, fval, pdf);
        },
        "nodes"_a, "f"_a, "cdf"_a, "u"_a);
    m.def(
        "SampleCatmullRom2D",
        [](std::vector<Float> nodes1, std::vector<Float> nodes2, std::vector<Float> f,
           std::vector<Float> cdf, Float alpha, Float u) {
            Float fval, pdf;
            Float x = SampleCatmullRom2D(nodes1, nodes2, f, cdf, alpha, u, &fval, &pdf);
            return std::make_tuple(x, fval, pdf);
        },
        "nodes1"_a, "nodes2"_a, "f"_a, "cdf"_a, "alpha"_a, "u"_a);
    m.def(
        "IntegrateCatmullRom",
        [](std::vector<Float> nodes, std::vector<Float> f) {
            std::vector<Float> cdf(nodes.size(), Float(0));
            Float i = IntegrateCatmullRom(nodes, f, MakeSpan(cdf));
            return std::make_tuple(i, cdf);
        },
        "nodes"_a, "f"_a);
    m.def(
        "InvertCatmullRom",
        [](std::vector<Float> nodes, std::vector<Float> f, Float u) {
            return InvertCatmullRom(nodes, f, u);
        },
        "nodes"_a, "f"_a, "u"_a);
    m.def(
        "Fourier", [](std::vector<Float> a, double cosPhi) { return Fourier(a, cosPhi); },
        "a"_a, "cosPhi"_a);
    m.def(
        "SampleFourier",
        [](std::vector<Float> ak, std::vector<Float> recip, Float u) {
            Float pdf, phi;
            Float v = SampleFourier(ak, recip, u, &pdf, &phi);
            return std::make_tuple(v, pdf, phi);
        },
        "ak"_a, "recip"_a, "u"_a);

    // interval.h
    py::class_<Interval>(m, "Interval")
        .def(py::init<>())
        .def(py::init<float>())
        .def(py::init<float, float>())
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def("__str__", [](const Interval &i) { return i.ToString(); });

    m.def("ACos", (Interval(*)(const Interval &))(&ACos));
    m.def("Sqrt", (Interval(*)(const Interval &))(&Sqrt));
    m.def("Cos", (Interval(*)(const Interval &))(&Cos));
    m.def("Sin", (Interval(*)(const Interval &))(&Sin));

    // math.h
    m.attr("ShadowEpsilon") = ShadowEpsilon;
    m.attr("Pi") = Pi;
    m.attr("InvPi") = InvPi;
    m.attr("Inv2Pi") = Inv2Pi;
    m.attr("Inv4Pi") = Inv4Pi;
    m.attr("PiOver2") = PiOver2;
    m.attr("PiOver4") = PiOver4;
    m.attr("Sqrt2") = Sqrt2;
    m.def("Clamp", [](Float v, Float low, Float high) { return Clamp(v, low, high); });
    m.def("Mod", [](int a, int b) { return Mod(a, b); });
    m.def("Mod", [](Float a, Float b) { return Mod(a, b); });
    m.def("Radians", Radians);
    m.def("Degrees", Degrees);
    m.def("NextPrime", NextPrime);
    m.def("Log2", Log2);
    m.def("Lerp", (Float(*)(Float, Float, Float))Lerp);
    m.def("Quadratic", [](Float a, Float b, Float c) -> std::pair<Float, Float> {
        Float t0, t1;
        if (!Quadratic(a, b, c, &t0, &t1))
            throw std::domain_error("No solutions to quadratic equation");
        return std::make_pair(t0, t1);
    });
    m.def("Bilerp",
          [](std::array<Float, 2> p, std::array<Float, 4> v) { return Bilerp(p, v); });

    m.def("ErfInv", ErfInv);
    m.def("Erf", Erf);
    m.def("Sinc", Sinc);
    m.def("WindowedSinc", WindowedSinc);
    m.def("Smoothstep", Smoothstep);
    m.def("I0", I0);
    m.def("LogI0", LogI0);
    m.def("Logistic", Logistic);
    m.def("LogisticCDF", LogisticCDF);
    m.def("TrimmedLogistic", TrimmedLogistic);

    m.def("Log2Int", (int (*)(float))Log2Int);
    m.def("Log2Int", (int (*)(double))Log2Int);
    m.def("Log2Int", (int (*)(int32_t))Log2Int);
    m.def("Log2Int", (int (*)(uint32_t))Log2Int);
    m.def("Log2Int", (int (*)(int64_t))Log2Int);
    m.def("Log2Int", (int (*)(uint64_t))Log2Int);
    m.def("IsPowerOf2", (bool (*)(int32_t))IsPowerOf2);
    m.def("IsPowerOf2", (bool (*)(uint32_t))IsPowerOf2);
    m.def("IsPowerOf2", (bool (*)(int64_t))IsPowerOf2);
    m.def("IsPowerOf2", (bool (*)(uint64_t))IsPowerOf2);
    m.def("RoundUpPow2", (int32_t(*)(int32_t))RoundUpPow2);
    m.def("RoundUpPow2", (int64_t(*)(int64_t))RoundUpPow2);

    m.def("Log4Int", (int (*)(float))Log4Int);
    m.def("Log4Int", (int (*)(double))Log4Int);
    m.def("Log4Int", (int (*)(int32_t))Log4Int);
    m.def("Log4Int", (int (*)(uint32_t))Log4Int);
    m.def("Log4Int", (int (*)(int64_t))Log4Int);
    m.def("Log4Int", (int (*)(uint64_t))Log4Int);
    m.def("IsPowerOf4", (bool (*)(int32_t))IsPowerOf4);
    m.def("IsPowerOf4", (bool (*)(uint32_t))IsPowerOf4);
    m.def("IsPowerOf4", (bool (*)(int64_t))IsPowerOf4);
    m.def("IsPowerOf4", (bool (*)(uint64_t))IsPowerOf4);
    m.def("RoundUpPow4", (int32_t(*)(int32_t))RoundUpPow4);
    m.def("RoundUpPow4", (int64_t(*)(int64_t))RoundUpPow4);

    m.def("SafeASin", (float (*)(float))SafeASin);
    m.def("SafeASin", (double (*)(double))SafeASin);
    m.def("SafeACos", (float (*)(float))SafeACos);
    m.def("SafeACos", (double (*)(double))SafeACos);
    m.def("SafeSqrt", (float (*)(float))SafeSqrt);
    m.def("SafeSqrt", (double (*)(double))SafeSqrt);

    m.def("EvaluatePolynomial",
          [](Float t, Float a) { return EvaluatePolynomial(t, a); });
    m.def("EvaluatePolynomial",
          [](Float t, Float a, Float b) { return EvaluatePolynomial(t, a, b); });
    m.def("EvaluatePolynomial", [](Float t, Float a, Float b, Float c) {
        return EvaluatePolynomial(t, a, b, c);
    });
    m.def("EvaluatePolynomial", [](Float t, Float a, Float b, Float c, Float d) {
        return EvaluatePolynomial(t, a, b, c, d);
    });
    m.def("EvaluatePolynomial", [](Float t, Float a, Float b, Float c, Float d, Float e) {
        return EvaluatePolynomial(t, a, b, c, d, e);
    });
    m.def("EvaluatePolynomial",
          [](Float t, Float a, Float b, Float c, Float d, Float e, Float f) {
              return EvaluatePolynomial(t, a, b, c, d, e, f);
          });
    m.def("EvaluatePolynomial",
          [](Float t, Float a, Float b, Float c, Float d, Float e, Float f, Float g) {
              return EvaluatePolynomial(t, a, b, c, d, e, f, g);
          });

    m.def("FindInterval", [](size_t size, std::function<Float(int)> pred) {
        return FindInterval(size, pred);
    });

    m.def(
        "NewtonBisection",
        [](Float x0, Float x1, std::function<std::pair<Float, Float>(Float)> f,
           Float xEps,
           Float fEps) -> Float { return NewtonBisection(x0, x1, f, xEps, fEps); },
        "x0"_a, "x1"_a, "func"_a, "xEps"_a = 1e-6f, "fEps"_a = 1e-6f);

    // matrix.h
    py::class_<SquareMatrix<2>>(m, "SquareMatrix2")
        .def(py::init<>())
        .def(py::init<std::array<Float, 4>>())
        .def(py::init<Float, Float, Float, Float>())
        .def(py::self * py::self)
        .def(py::self * std::array<Float, 2>())
        .def("Diag",
             [](std::array<Float, 2> v) { return SquareMatrix<2>::Diag(v[0], v[1]); })
        .def("IsIdentity", &SquareMatrix<2>::IsIdentity)
        .def("__eq__",
             [](const SquareMatrix<2> &a, const SquareMatrix<2> &b) { return a == b; })
        .def("__str__", &SquareMatrix<2>::ToString)
        .def("__getitem__",
             [](const SquareMatrix<2> &m, std::array<int, 2> index) {
                 return m[index[0]][index[1]];
             })
        .def("__setitem__", [](SquareMatrix<2> &m, std::array<int, 2> index, double v) {
            m[index[0]][index[1]] = v;
        });
    m.def("Transpose", (SquareMatrix<2>(*)(const SquareMatrix<2> &))(&Transpose));
    m.def("Inverse", [](const SquareMatrix<2> &m) {
        auto inv = Inverse(m);
        if (!inv.has_value())
            throw std::domain_error("Unable to invert matrix");
        return *inv;
    });
    py::class_<SquareMatrix<3>>(m, "SquareMatrix3")
        .def(py::init<>())
        .def(py::init<std::array<Float, 9>>())
        .def(py::init<Float, Float, Float, Float, Float, Float, Float, Float, Float>())
        .def(py::self * py::self)
        .def(py::self * std::array<Float, 3>())
        .def("Diag",
             [](std::array<Float, 3> v) {
                 return SquareMatrix<3>::Diag(v[0], v[1], v[2]);
             })
        .def("__eq__",
             [](const SquareMatrix<3> &a, const SquareMatrix<3> &b) { return a == b; })
        .def("IsIdentity", &SquareMatrix<3>::IsIdentity)
        .def("__str__", &SquareMatrix<3>::ToString)
        .def("__getitem__",
             [](const SquareMatrix<3> &m, std::array<int, 3> index) {
                 return m[index[0]][index[1]];
             })
        .def("__setitem__", [](SquareMatrix<3> &m, std::array<int, 3> index, double v) {
            m[index[0]][index[1]] = v;
        });
    m.def("Transpose", (SquareMatrix<3>(*)(const SquareMatrix<3> &))(&Transpose));
    m.def("Inverse", [](const SquareMatrix<3> &m) {
        auto inv = Inverse(m);
        if (!inv.has_value())
            throw std::domain_error("Unable to invert matrix");
        return *inv;
    });

    py::class_<SquareMatrix<4>>(m, "SquareMatrix4")
        .def(py::init<>())
        .def(py::init<std::array<Float, 16>>())
        .def(py::init<Float, Float, Float, Float, Float, Float, Float, Float, Float,
                      Float, Float, Float, Float, Float, Float, Float>())
        .def(py::self * py::self)
        .def(py::self * std::array<Float, 4>())
        .def("Diag",
             [](std::array<Float, 4> v) {
                 return SquareMatrix<4>::Diag(v[0], v[1], v[2], v[3]);
             })
        .def("IsIdentity", &SquareMatrix<4>::IsIdentity)
        .def("__eq__",
             [](const SquareMatrix<4> &a, const SquareMatrix<4> &b) { return a == b; })
        .def("__str__", &SquareMatrix<4>::ToString)
        .def("__getitem__",
             [](const SquareMatrix<4> &m, std::array<int, 4> index) {
                 return m[index[0]][index[1]];
             })
        .def("__setitem__", [](SquareMatrix<4> &m, std::array<int, 4> index, double v) {
            m[index[0]][index[1]] = v;
        });
    m.def("Transpose", (SquareMatrix<4>(*)(const SquareMatrix<4> &))(&Transpose));
    m.def("Inverse", [](const SquareMatrix<4> &m) {
        auto inv = Inverse(m);
        if (!inv.has_value())
            throw std::domain_error("Unable to invert matrix");
        return *inv;
    });

    // parallel.h
    py::class_<AtomicFloat>(m, "AtomicFloat")
        .def(py::init<float>(), "v"_a = 0)
        .def("__float__", &AtomicFloat::operator float)
        .def("Add", &AtomicFloat::Add, "v"_a);
    py::class_<AtomicDouble>(m, "AtomicDouble")
        .def(py::init<double>(), "v"_a = 0)
        .def("__float__", &AtomicDouble::operator double)
        .def("Add", &AtomicDouble::Add, "v"_a);

    // primes.h
    m.attr("Primes") = Primes;
    m.attr("PrimeSums") = PrimeSums;

    // print.h
    m.def(
        "FloatToString", [](float v) { return FloatToString(v); }, "v"_a);
    m.def(
        "DoubleToString", [](double v) { return DoubleToString(v); }, "v"_a);

    // rng.h
    py::class_<RNG>(m, "RNG")
        .def(py::init<>())
        .def(py::init<uint64_t>(), "sequenceIndex"_a)
        .def("SetSequence", &RNG::SetSequence, "sequenceIndex"_a)
        .def("Advance", &RNG::Advance, "steps"_a)
        .def(py::self - py::self)
        .def("UniformInt32", [](RNG &rng) { return rng.Uniform<int32_t>(); })
        .def("UniformInt32", [](RNG &rng, int n) { return rng.Uniform<int32_t>(n); })
        .def("UniformUInt32", [](RNG &rng) { return rng.Uniform<uint32_t>(); })
        .def("UniformUInt32", [](RNG &rng, int n) { return rng.Uniform<uint32_t>(n); })
        .def("UniformInt64", [](RNG &rng) { return rng.Uniform<int64_t>(); })
        .def("UniformInt64", [](RNG &rng, int n) { return rng.Uniform<int64_t>(n); })
        .def("UniformUInt64", [](RNG &rng) { return rng.Uniform<uint64_t>(); })
        .def("UniformUInt64", [](RNG &rng, int n) { return rng.Uniform<uint64_t>(n); })
        .def("UniformFloat", [](RNG &rng) { return rng.Uniform<float>(); })
        .def("UniformDouble", [](RNG &rng) { return rng.Uniform<double>(); })
        .def("__str__", [](const RNG &rng) { return rng.ToString(); });

    // shuffle.h
    m.def("PermutationElement", &PermutationElement,
          "Return i'th element of a permutation of length |l|", "i"_a, "l"_a, "hash"_a);
}

}  // namespace pbrt
