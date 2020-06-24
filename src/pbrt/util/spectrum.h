// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SPECTRUM_SPECTRUM_H
#define PBRT_SPECTRUM_SPECTRUM_H

// spectrum/sampled.h*
#include <pbrt/pbrt.h>

#include <pbrt/base/spectrum.h>
#include <pbrt/util/check.h>
#include <pbrt/util/color.h>
#include <pbrt/util/colorspace.h>
#include <pbrt/util/float.h>
#include <pbrt/util/math.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/taggedptr.h>

#include <cmath>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace pbrt {

enum class SpectrumType { Reflectance, General };

constexpr Float LambdaMin = 360;
constexpr Float LambdaMax = 830;

PBRT_CPU_GPU
Float Blackbody(Float lambda, Float T);

Float SpectrumToY(SpectrumHandle s);
XYZ SpectrumToXYZ(SpectrumHandle s);

// Spectrum Utility Declarations

// TODO: NSpectrumSamples ?

/* c-bdpt timings with varying # of spectrum samples:
   Very bad color aliasing with 1 and 2 wavelength samples; would like
   to better understand why 4+ is such a sweet spot.

   8: 345.5 user
   4: 324.9
   2: 315.7
   1: 306

path, 500x500, 16spp
8 451
4 372
2 332
1 302

*/
static constexpr int NSpectrumSamples = 4;

class alignas(8) SampledWavelengths {
  public:
    PBRT_CPU_GPU
    static SampledWavelengths SampleEqui(Float u, Float lambdaMin = LambdaMin,
                                         Float lambdaMax = LambdaMax) {
        SampledWavelengths swl;
        swl.lambda[0] = Lerp(u, lambdaMin, lambdaMax);
        Float delta = (lambdaMax - lambdaMin) / NSpectrumSamples;
        for (int i = 1; i < NSpectrumSamples; ++i) {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            if (swl.lambda[i] > lambdaMax)
                swl.lambda[i] = lambdaMin + (swl.lambda[i] - lambdaMax);
        }
        for (int i = 0; i < NSpectrumSamples; ++i)
            swl.pdf[i] = 1 / (lambdaMax - lambdaMin);
        return swl;
    }

    PBRT_CPU_GPU
    static SampledWavelengths SampleImportance(Float u) {
        SampledWavelengths swl;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            Float up = u + Float(i) / NSpectrumSamples;
            if (up > 1)
                up -= 1;
            swl.lambda[i] = SampleXYZMatching(up);
            swl.pdf[i] = XYZMatchingPDF(swl.lambda[i]);
        }
        return swl;
    }

    PBRT_CPU_GPU
    bool operator==(const SampledWavelengths &swl) const {
        return lambda == swl.lambda && pdf == swl.pdf;
    }
    PBRT_CPU_GPU
    bool operator!=(const SampledWavelengths &swl) const {
        return lambda != swl.lambda || pdf != swl.pdf;
    }
    PBRT_CPU_GPU
    Float operator[](int i) const { return lambda[i]; }

    std::string ToString() const;

    PBRT_CPU_GPU
    void TerminateSecondaryWavelengths() const {
        if (secondaryWavelengthsTerminated)
            return;
        secondaryWavelengthsTerminated = true;
        pdf[0] /= NSpectrumSamples;
        for (int i = 1; i < NSpectrumSamples; ++i)
            pdf[i] = 0;
    }

    mutable pstd::array<Float, NSpectrumSamples> pdf;

  private:
    mutable bool secondaryWavelengthsTerminated = false;
    pstd::array<Float, NSpectrumSamples> lambda;
};

class alignas(8) SampledSpectrum {
  public:
    PBRT_CPU_GPU
    SampledSpectrum() { v.fill(0); }
    PBRT_CPU_GPU
    explicit SampledSpectrum(Float c) { v.fill(c); }
    PBRT_CPU_GPU
    SampledSpectrum(pstd::array<Float, NSpectrumSamples> values) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] = values[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum &operator+=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] += s.v[i];
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret += s;
    }

    PBRT_CPU_GPU
    SampledSpectrum &operator-=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] -= s.v[i];
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator-(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret -= s;
    }
    PBRT_CPU_GPU
    friend SampledSpectrum operator-(Float a, const SampledSpectrum &s) {
        DCHECK(!std::isnan(a));
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.v[i] = a - s.v[i];
        return ret;
    }

    PBRT_CPU_GPU
    SampledSpectrum &operator*=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] *= s.v[i];
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator*(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret *= s;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator*(Float a) const {
        DCHECK(!std::isnan(a));
        SampledSpectrum ret = *this;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.v[i] *= a;
        return ret;
    }
    PBRT_CPU_GPU
    SampledSpectrum &operator*=(Float a) {
        DCHECK(!std::isnan(a));
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] *= a;
        return *this;
    }
    PBRT_CPU_GPU
    friend SampledSpectrum operator*(Float a, const SampledSpectrum &s) { return s * a; }

    PBRT_CPU_GPU
    SampledSpectrum &operator/=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) {
            DCHECK_NE(0, s.v[i]);
            v[i] /= s.v[i];
        }
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret /= s;
    }
    PBRT_CPU_GPU
    SampledSpectrum &operator/=(Float a) {
        DCHECK_NE(a, 0);
        DCHECK(!std::isnan(a));
        for (int i = 0; i < NSpectrumSamples; ++i)
            v[i] /= a;
        return *this;
    }
    PBRT_CPU_GPU
    SampledSpectrum operator/(Float a) const {
        SampledSpectrum ret = *this;
        return ret /= a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-() const {
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i)
            ret.v[i] = -v[i];
        return ret;
    }
    PBRT_CPU_GPU
    bool operator==(const SampledSpectrum &s) const { return v == s.v; }
    PBRT_CPU_GPU
    bool operator!=(const SampledSpectrum &s) const { return v != s.v; }
    // We don't need an explicit cast to bool in e.g. "if" tests or boolean
    // expressions, which is nice.
    // https://stackoverflow.com/questions/6242768/is-the-safe-bool-idiom-obsolete-in-c11
    PBRT_CPU_GPU
    explicit operator bool() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (v[i] != 0.)
                return true;
        return false;
    }

    PBRT_CPU_GPU
    friend SampledSpectrum Sqrt(const SampledSpectrum &s);
    PBRT_CPU_GPU
    friend SampledSpectrum Pow(const SampledSpectrum &s, Float e);
    PBRT_CPU_GPU
    friend SampledSpectrum Exp(const SampledSpectrum &s);
    PBRT_CPU_GPU
    friend SampledSpectrum FastExp(const SampledSpectrum &s);

    std::string ToString() const;

    PBRT_CPU_GPU
    Float MinComponentValue() const {
        Float m = v[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::min(m, v[i]);
        return m;
    }
    PBRT_CPU_GPU
    Float MaxComponentValue() const {
        Float m = v[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::max(m, v[i]);
        return m;
    }
    PBRT_CPU_GPU
    Float Average() const {
        Float sum = v[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            sum += v[i];
        return sum / NSpectrumSamples;
    }

    PBRT_CPU_GPU
    bool HasNaNs() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (std::isnan(v[i]))
                return true;
        return false;
    }

    PBRT_CPU_GPU
    Float &operator[](int i) {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return v[i];
    }
    PBRT_CPU_GPU
    Float operator[](int i) const {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return v[i];
    }

    PBRT_CPU_GPU
    XYZ ToXYZ(const SampledWavelengths &lambda) const;
    PBRT_CPU_GPU
    RGB ToRGB(const SampledWavelengths &lambda, const RGBColorSpace &cs) const;
    PBRT_CPU_GPU
    Float y(const SampledWavelengths &lambda) const;

  private:
    pstd::array<Float, NSpectrumSamples> v;
};

class alignas(8) BlackbodySpectrum {
  public:
    BlackbodySpectrum(Float T, Float scale)
        : T(T), scale(scale) {
        // Normalize _Le_ based on maximum blackbody radiance
        Float lambdaMax = Float(2.8977721e-3 / T * 1e9);
        normalizationFactor = 1 / Blackbody(lambdaMax, T);
    }

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        return scale * Blackbody(lambda, T) * normalizationFactor;
    }

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = scale * Blackbody(lambda[i], T) * normalizationFactor;
        return s;
    }

    PBRT_CPU_GPU
    Float MaxValue() const { return scale; }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

    Float T, scale, normalizationFactor;
};

class alignas(8) ConstantSpectrum {
  public:
    ConstantSpectrum(Float c) : c(c) {}

    PBRT_CPU_GPU
    Float operator()(Float lambda) const { return c; }
    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &) const;

    PBRT_CPU_GPU
    Float MaxValue() const { return c; }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

    Float c;
};

class alignas(8) ScaledSpectrum {
  public:
    ScaledSpectrum(Float scale, SpectrumHandle s) : scale(scale), s(s) {}

    PBRT_CPU_GPU
    Float operator()(Float lambda) const { return scale * s(lambda); }
    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float MaxValue() const { return scale * s.MaxValue(); }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    Float scale;
    SpectrumHandle s;
};

class alignas(8) PiecewiseLinearSpectrum {
  public:
    PiecewiseLinearSpectrum() = default;
    PiecewiseLinearSpectrum(pstd::span<const Float> l, pstd::span<const Float> values,
                            Allocator alloc = {});

    PBRT_CPU_GPU
    Float MaxValue() const;
    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = (*this)(lambda[i]);
        return s;
    }
    PBRT_CPU_GPU
    Float operator()(Float lambda) const;

    static pstd::optional<SpectrumHandle> Read(const std::string &filename,
                                               Allocator alloc);

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    pstd::vector<Float> lambda, v;
};

class alignas(8) DenselySampledSpectrum {
  public:
    DenselySampledSpectrum(int lambdaMin = LambdaMin,
                           int lambdaMax = LambdaMax, Allocator alloc = {})
        : lambdaMin(lambdaMin), lambdaMax(lambdaMax),
          v(lambdaMax - lambdaMin + 1, alloc) {}
    DenselySampledSpectrum(SpectrumHandle s, int lambdaMin = LambdaMin,
                           int lambdaMax = LambdaMax, Allocator alloc = {});
    DenselySampledSpectrum(SpectrumHandle s, Allocator alloc)
        : DenselySampledSpectrum(s, LambdaMin, LambdaMax, alloc) {}

    template <typename F>
    static DenselySampledSpectrum SampleFunction(F func, int lambdaMin = LambdaMin,
                                                 int lambdaMax = LambdaMax,
                                                 Allocator alloc = {}) {
        DenselySampledSpectrum s(lambdaMin, lambdaMax, alloc);
        for (int lambda = lambdaMin; lambda <= lambdaMax; ++lambda)
            s.v[lambda - lambdaMin] = func(lambda + 0.5f);
        return s;
    }

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        DCHECK_GT(lambda, 0);
        // Constant outside the defined range.
        // TODO: should it be zero instead?
        lambda = Clamp(lambda, lambdaMin, lambdaMax);
        return v[int(lambda) - lambdaMin];
    }

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            Float l = Clamp(lambda[i], lambdaMin, lambdaMax);
            int offset = int(l - lambdaMin);
            s[i] = v[offset];
        }
        return s;
    }

    PBRT_CPU_GPU
    Float MaxValue() const { return *std::max_element(v.begin(), v.end()); }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    int lambdaMin, lambdaMax;
    pstd::vector<Float> v;
};

class alignas(8) RGBReflectanceSpectrum {
  public:
    PBRT_CPU_GPU
    RGBReflectanceSpectrum(const RGBColorSpace &cs, const RGB &rgb)
        : rgb(rgb), rsp(cs.ToRGBCoeffs(rgb)) {}

    PBRT_CPU_GPU
    Float operator()(Float lambda) const { return rsp(lambda); }

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = rsp(lambda[i]);
        return s;
    }

    PBRT_CPU_GPU
    Float MaxValue() const { return rsp.MaxValue(); }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    RGB rgb;
    RGBSigmoidPolynomial rsp;
};

class alignas(8) RGBSpectrum {
  public:
    RGBSpectrum() = default;

    PBRT_CPU_GPU
    RGBSpectrum(const RGBColorSpace &cs, const RGB &rgb)
        : rgb(rgb), illuminant(cs.illuminant) {
        Float m = std::max({rgb.r, rgb.g, rgb.b});
        scale = m > 0 ? 0.5f / m : 0;
        rsp = cs.ToRGBCoeffs(rgb * scale);
    }

    PBRT_CPU_GPU
    Float operator()(Float lambda) const {
        return rsp(lambda) / (scale > 0 ? scale : 1) * illuminant(lambda);
    }

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = rsp(lambda[i]) / (scale > 0 ? scale : 1);
        return s * illuminant.Sample(lambda);
    }

    PBRT_CPU_GPU
    Float MaxValue() const {
        return rsp.MaxValue() / (scale > 0 ? scale : 1) * illuminant.MaxValue();
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    RGB rgb;
    Float scale;
    RGBSigmoidPolynomial rsp;
    SpectrumHandle illuminant;
};

// Spectrum Inline Functions
template <typename U, typename V>
PBRT_CPU_GPU inline SampledSpectrum Clamp(const SampledSpectrum &s, U low, V high) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = pbrt::Clamp(s[i], low, high);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum ClampZero(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::max<Float>(0, s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Sqrt(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret.v[i] = std::sqrt(s.v[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Pow(const SampledSpectrum &s, Float e) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret.v[i] = std::pow(s.v[i], e);
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum Exp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret.v[i] = std::exp(s.v[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum FastExp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret.v[i] = FastExp(s.v[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_CPU_GPU
inline SampledSpectrum SafeDiv(const SampledSpectrum &s1, const SampledSpectrum &s2) {
    SampledSpectrum r(0.);
    for (int i = 0; i < NSpectrumSamples; ++i)
        r[i] = (s2[i] != 0) ? s1[i] / s2[i] : 0.;
    return r;
}

PBRT_CPU_GPU
inline SampledSpectrum Lerp(Float t, const SampledSpectrum &s1,
                            const SampledSpectrum &s2) {
    return (1 - t) * s1 + t * s2;
}

PBRT_CPU_GPU
inline SampledSpectrum Bilerp(pstd::array<Float, 2> p,
                              pstd::span<const SampledSpectrum> v) {
    return ((1 - p[0]) * (1 - p[1]) * v[0] + p[0] * (1 - p[1]) * v[1] +
            (1 - p[0]) * p[1] * v[2] + p[0] * p[1] * v[3]);
}

namespace SPDs {

void Init(Allocator alloc);

SpectrumHandle Zero(), One();

static constexpr Float CIE_Y_integral = 106.856895;

PBRT_CPU_GPU
inline const DenselySampledSpectrum &X() {
#ifdef PBRT_IS_GPU_CODE
    extern PBRT_GPU DenselySampledSpectrum *xGPU;
    return *xGPU;
#else
    extern DenselySampledSpectrum *x;
    return *x;
#endif
}

PBRT_CPU_GPU
inline const DenselySampledSpectrum &Y() {
#ifdef PBRT_IS_GPU_CODE
    extern PBRT_GPU DenselySampledSpectrum *yGPU;
    return *yGPU;
#else
    extern DenselySampledSpectrum *y;
    return *y;
#endif
}

PBRT_CPU_GPU
inline const DenselySampledSpectrum &Z() {
#ifdef PBRT_IS_GPU_CODE
    extern PBRT_GPU DenselySampledSpectrum *zGPU;
    return *zGPU;
#else
    extern DenselySampledSpectrum *z;
    return *z;
#endif
}

SpectrumHandle IllumA();
SpectrumHandle IllumD50();
SpectrumHandle IllumACESD60();
SpectrumHandle IllumD65();
SpectrumHandle IllumF1(), IllumF2(), IllumF3(), IllumF4();
SpectrumHandle IllumF5(), IllumF6(), IllumF7(), IllumF8();
SpectrumHandle IllumF9(), IllumF10(), IllumF11(), IllumF12();

SpectrumHandle MetalAgEta(), MetalAgK();
SpectrumHandle MetalAlEta(), MetalAlK();
SpectrumHandle MetalAuEta(), MetalAuK();
SpectrumHandle MetalCuEta(), MetalCuK();
SpectrumHandle MetalMgOEta(), MetalMgOK();
SpectrumHandle MetalTiO2Eta(), MetalTiO2K();

SpectrumHandle GlassBK7Eta();
SpectrumHandle GlassBAF10Eta();
SpectrumHandle GlassFK51AEta();
SpectrumHandle GlassLASF9Eta();
SpectrumHandle GlassSF5Eta();
SpectrumHandle GlassSF10Eta();
SpectrumHandle GlassSF11Eta();

SpectrumHandle GetNamed(const std::string &name);
std::string FindMatchingNamed(SpectrumHandle s);

}  // namespace SPDs

inline Float SpectrumHandle::operator()(Float lambda) const {
    auto op = [&](auto ptr) { return (*ptr)(lambda); };
    return Apply<Float>(op);
}

inline SampledSpectrum SpectrumHandle::Sample(const SampledWavelengths &lambda) const {
    auto samp = [&](auto ptr) { return ptr->Sample(lambda); };
    return Apply<SampledSpectrum>(samp);
}

inline Float SpectrumHandle::MaxValue() const {
    auto max = [&](auto ptr) { return ptr->MaxValue(); };
    return Apply<Float>(max);
}

}  // namespace pbrt

#endif  // PBRT_SPECTRUM_H
