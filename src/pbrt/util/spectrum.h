
/*
    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SPECTRUM_SPECTRUM_H
#define PBRT_SPECTRUM_SPECTRUM_H

// spectrum/sampled.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/color.h>
#include <pbrt/util/check.h>
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

PBRT_HOST_DEVICE
Float Blackbody(Float lambda, Float T);

class BlackbodySpectrum;
class ConstantSpectrum;
class ProductSpectrum;
class ScaledSpectrum;
class PiecewiseLinearSpectrum;
class DenselySampledSpectrum;
class RGBReflectanceSpectrum;
class RGBSpectrum;

class SpectrumHandle : public TaggedPointer<BlackbodySpectrum, ConstantSpectrum, ProductSpectrum,
                                            ScaledSpectrum, PiecewiseLinearSpectrum, DenselySampledSpectrum,
                                            RGBReflectanceSpectrum, RGBSpectrum> {
public:
    using TaggedPointer::TaggedPointer;
    PBRT_HOST_DEVICE_INLINE
    SpectrumHandle(TaggedPointer<BlackbodySpectrum, ConstantSpectrum, ProductSpectrum,
                                 ScaledSpectrum, PiecewiseLinearSpectrum, DenselySampledSpectrum,
                                 RGBReflectanceSpectrum, RGBSpectrum> tp)
        : TaggedPointer(tp) { }

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const;

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const;

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;
};

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

class SampledWavelengths {
  public:
    PBRT_HOST_DEVICE_INLINE
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

    PBRT_HOST_DEVICE_INLINE
    static SampledWavelengths SampleImportance(Float u) {
        SampledWavelengths swl;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            Float up = u + Float(i) / NSpectrumSamples;
            if (up > 1) up -= 1;
            swl.lambda[i] = SampleXYZMatching(up);
            swl.pdf[i] = XYZMatchingPDF(swl.lambda[i]);
        }
        return swl;
    }

    PBRT_HOST_DEVICE_INLINE
    bool operator==(const SampledWavelengths &swl) const {
        return lambda == swl.lambda && pdf == swl.pdf;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const SampledWavelengths &swl) const {
        return lambda != swl.lambda || pdf != swl.pdf;
    }
    PBRT_HOST_DEVICE_INLINE
    Float operator[](int i) const { return lambda[i]; }

    std::string ToString() const;

    PBRT_HOST_DEVICE_INLINE
    void TerminateSecondaryWavelengths() const {
        if (secondaryWavelengthsTerminated) return;
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

class SampledSpectrum {
  public:
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum() { v.fill(0); }
    PBRT_HOST_DEVICE_INLINE
    explicit SampledSpectrum(Float c) { v.fill(c); }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum &operator+=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) v[i] += s.v[i];
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret += s;
    }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum &operator-=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) v[i] -= s.v[i];
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator-(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret -= s;
    }
    PBRT_HOST_DEVICE_INLINE
    friend SampledSpectrum operator-(Float a, const SampledSpectrum &s) {
        DCHECK(!std::isnan(a));
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i) ret.v[i] = a - s.v[i];
        return ret;
    }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum &operator*=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) v[i] *= s.v[i];
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator*(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret *= s;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator*(Float a) const {
        DCHECK(!std::isnan(a));
        SampledSpectrum ret = *this;
        for (int i = 0; i < NSpectrumSamples; ++i) ret.v[i] *= a;
        return ret;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum &operator*=(Float a) {
        DCHECK(!std::isnan(a));
        for (int i = 0; i < NSpectrumSamples; ++i) v[i] *= a;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    friend SampledSpectrum operator*(Float a, const SampledSpectrum &s) {
        return s * a;
    }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum &operator/=(const SampledSpectrum &s) {
        for (int i = 0; i < NSpectrumSamples; ++i) {
            DCHECK_NE(0, s.v[i]);
            v[i] /= s.v[i];
        }
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        SampledSpectrum ret = *this;
        return ret /= s;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum &operator/=(Float a) {
        DCHECK_NE(a, 0);
        DCHECK(!std::isnan(a));
        for (int i = 0; i < NSpectrumSamples; ++i) v[i] /= a;
        return *this;
    }
    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator/(Float a) const {
        SampledSpectrum ret = *this;
        return ret /= a;
    }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum operator-() const {
        SampledSpectrum ret;
        for (int i = 0; i < NSpectrumSamples; ++i) ret.v[i] = -v[i];
        return ret;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator==(const SampledSpectrum &s) const {
        return v == s.v;
    }
    PBRT_HOST_DEVICE_INLINE
    bool operator!=(const SampledSpectrum &s) const {
        return v != s.v;
    }
    // We don't need an explicit cast to bool in e.g. "if" tests or boolean
    // expressions, which is nice.
    // https://stackoverflow.com/questions/6242768/is-the-safe-bool-idiom-obsolete-in-c11
    PBRT_HOST_DEVICE_INLINE
    explicit operator bool() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (v[i] != 0.) return true;
        return false;
    }

    PBRT_HOST_DEVICE_INLINE
    friend SampledSpectrum Sqrt(const SampledSpectrum &s);
    PBRT_HOST_DEVICE_INLINE
    friend SampledSpectrum Pow(const SampledSpectrum &s, Float e);
    PBRT_HOST_DEVICE_INLINE
    friend SampledSpectrum Exp(const SampledSpectrum &s);

    std::string ToString() const;

    PBRT_HOST_DEVICE_INLINE
    Float MinComponentValue() const {
        Float m = v[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::min(m, v[i]);
        return m;
    }
    PBRT_HOST_DEVICE_INLINE
    Float MaxComponentValue() const {
        Float m = v[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            m = std::max(m, v[i]);
        return m;
    }
    PBRT_HOST_DEVICE_INLINE
    Float Average() const {
        Float sum = v[0];
        for (int i = 1; i < NSpectrumSamples; ++i)
            sum += v[i];
        return sum / NSpectrumSamples;
    }

    PBRT_HOST_DEVICE_INLINE
    bool HasNaNs() const {
        for (int i = 0; i < NSpectrumSamples; ++i)
            if (std::isnan(v[i])) return true;
        return false;
    }

    PBRT_HOST_DEVICE_INLINE
    Float &operator[](int i) {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return v[i];
    }
    PBRT_HOST_DEVICE_INLINE
    Float operator[](int i) const {
        DCHECK(i >= 0 && i < NSpectrumSamples);
        return v[i];
    }

    PBRT_HOST_DEVICE
    XYZ ToXYZ(const SampledWavelengths &lambda,
              const DenselySampledSpectrum &X,
              const DenselySampledSpectrum &Y,
              const DenselySampledSpectrum &Z) const;
    PBRT_HOST_DEVICE
    RGB ToRGB(const SampledWavelengths &lambda,
              const RGBColorSpace &cs) const;
    PBRT_HOST_DEVICE
    Float y(const SampledWavelengths &lambda,
            const DenselySampledSpectrum &Y) const;

  private:
    pstd::array<Float, NSpectrumSamples> v;
};

class BlackbodySpectrum {
  public:
    BlackbodySpectrum(Float T) : T(T) {
        // Normalize _Le_ based on maximum blackbody radiance
        Float lambdaMax = Float(2.8977721e-3 / T * 1e9);
        scale = 1 / Blackbody(lambdaMax, T);
    }

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        return scale * Blackbody(lambda, T);
    }
    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        return 1;
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

    Float T, scale;
};

class ConstantSpectrum {
  public:
    ConstantSpectrum(Float c) : c(c) {}

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const { return c; }
    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &) const;

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        return c;
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

    Float c;
};

class ScaledSpectrum {
  public:
    ScaledSpectrum(Float scale, SpectrumHandle s)
        : scale(scale), s(s) { }

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        return scale * s(lambda);
    }
    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        return scale * s.MaxValue();
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    Float scale;
    SpectrumHandle s;
};

class ProductSpectrum {
  public:
    ProductSpectrum(SpectrumHandle s1, SpectrumHandle s2)
        : s1(s1), s2(s2) {}

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        return s1(lambda) * s2(lambda);
    }
    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        // This is inaccurate, since the two may hit maximums at different
        // wavelengths.  At least it's conservative...
        return s1.MaxValue() * s2.MaxValue();
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    SpectrumHandle s1, s2;
};

class PiecewiseLinearSpectrum {
  public:
    PiecewiseLinearSpectrum() = default;
    PiecewiseLinearSpectrum(pstd::span<const Float> l,
                            pstd::span<const Float> values,
                            Allocator alloc = {});

    PBRT_HOST_DEVICE
    Float MaxValue() const;
    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = (*this)(lambda[i]);
        return s;
    }
    PBRT_HOST_DEVICE
    Float operator()(Float lambda) const;

    static pstd::optional<PiecewiseLinearSpectrum> Read(const std::string &filename);

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

  private:
    pstd::vector<Float> lambda, v;
};

class DenselySampledSpectrum {
 public:
    DenselySampledSpectrum() = default;
    DenselySampledSpectrum(SpectrumHandle s,
                           int lambdaMin = LambdaMin,
                           int lambdaMax = LambdaMax,
                           Allocator alloc = {});
    DenselySampledSpectrum(SpectrumHandle s, Allocator alloc)
        : DenselySampledSpectrum(s, LambdaMin, LambdaMax, alloc) { }

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        DCHECK_GT(lambda, 0);
        // Constant outside the defined range.
        // TODO: should it be zero instead?
        lambda = Clamp(lambda, lambdaMin, lambdaMax);
        return v[int(lambda) - lambdaMin];
    }

    PBRT_HOST_DEVICE_INLINE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            Float l = Clamp(lambda[i], lambdaMin, lambdaMax);
            int offset = int(l - lambdaMin);
            s[i] = v[offset];
        }
        return s;
    }

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        return *std::max_element(v.begin(), v.end());
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

 private:
    int lambdaMin, lambdaMax;
    pstd::vector<Float> v;
};


class RGBReflectanceSpectrum {
 public:
    PBRT_HOST_DEVICE
    RGBReflectanceSpectrum(const RGBColorSpace &cs, const RGB &rgb);

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        return rsp(lambda);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = rsp(lambda[i]);
        return s;
    }

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        return rsp.MaxValue();
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

 private:
    RGB rgb;
    RGBSigmoidPolynomial rsp;
};

class RGBSpectrum {
public:
    RGBSpectrum() = default;
    PBRT_HOST_DEVICE
    RGBSpectrum(const RGBColorSpace &cs, const RGB &rgb);

    PBRT_HOST_DEVICE_INLINE
    Float operator()(Float lambda) const {
        return rsp(lambda) / (scale > 0 ? scale : 1) * (*illuminant)(lambda);
    }

    PBRT_HOST_DEVICE
    SampledSpectrum Sample(const SampledWavelengths &lambda) const {
        SampledSpectrum s;
        for (int i = 0; i < NSpectrumSamples; ++i)
            s[i] = rsp(lambda[i]) / (scale > 0 ? scale : 1);
        return s * illuminant->Sample(lambda);
    }

    PBRT_HOST_DEVICE_INLINE
    Float MaxValue() const {
        return rsp.MaxValue() / (scale > 0 ? scale : 1) * illuminant->MaxValue();
    }

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;

private:
    RGB rgb;
    Float scale;
    RGBSigmoidPolynomial rsp;
    const DenselySampledSpectrum *illuminant;
};

// Spectrum Inline Functions
template <typename U, typename V>
PBRT_HOST_DEVICE_INLINE
SampledSpectrum Clamp(const SampledSpectrum &s, U low, V high) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = pbrt::Clamp(s[i], low, high);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum ClampZero(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret[i] = std::max<Float>(0, s[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum Sqrt(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i) ret.v[i] = std::sqrt(s.v[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum Pow(const SampledSpectrum &s, Float e) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i)
        ret.v[i] = std::pow(s.v[i], e);
    return ret;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum Exp(const SampledSpectrum &s) {
    SampledSpectrum ret;
    for (int i = 0; i < NSpectrumSamples; ++i) ret.v[i] = std::exp(s.v[i]);
    DCHECK(!ret.HasNaNs());
    return ret;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum SafeDiv(const SampledSpectrum &s1,
                        const SampledSpectrum &s2) {
    SampledSpectrum r(0.);
    for (int i = 0; i < NSpectrumSamples; ++i)
        r[i] = (s2[i] != 0) ? s1[i] / s2[i] : 0.;
    return r;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum Lerp(Float t, const SampledSpectrum &s1,
                     const SampledSpectrum &s2) {
    return (1 - t) * s1 + t * s2;
}

PBRT_HOST_DEVICE_INLINE
SampledSpectrum Bilerp(pstd::array<Float, 2> p,
                       pstd::span<const SampledSpectrum> v) {
    return ((1 - p[0]) * (1 - p[1]) * v[0] + p[0] * (1 - p[1]) * v[1] +
            (1 - p[0]) * p[1] * v[2] + p[0] * p[1] * v[3]);
}

namespace SPDs {

void Init(Allocator alloc);

SpectrumHandle Zero(), One();

static constexpr Float CIE_Y_integral = 106.856895;

SpectrumHandle X(), Y(), Z();

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
    switch (Tag()) {
    case TypeIndex<BlackbodySpectrum>():
        return (*Cast<BlackbodySpectrum>())(lambda);
    case TypeIndex<ConstantSpectrum>():
        return (*Cast<ConstantSpectrum>())(lambda);
    case TypeIndex<ProductSpectrum>():
        return (*Cast<ProductSpectrum>())(lambda);
    case TypeIndex<ScaledSpectrum>():
        return (*Cast<ScaledSpectrum>())(lambda);
    case TypeIndex<PiecewiseLinearSpectrum>():
        return (*Cast<PiecewiseLinearSpectrum>())(lambda);
    case TypeIndex<DenselySampledSpectrum>():
        return (*Cast<DenselySampledSpectrum>())(lambda);
    case TypeIndex<RGBReflectanceSpectrum>():
        return (*Cast<RGBReflectanceSpectrum>())(lambda);
    case TypeIndex<RGBSpectrum>():
        return (*Cast<RGBSpectrum>())(lambda);
    default:
        LOG_FATAL("Unhandled Spectrum type %d", Tag());
        return {};
    }
}

inline SampledSpectrum SpectrumHandle::Sample(const SampledWavelengths &lambda) const {
    switch (Tag()) {
    case TypeIndex<BlackbodySpectrum>():
        return Cast<BlackbodySpectrum>()->Sample(lambda);
    case TypeIndex<ConstantSpectrum>():
        return Cast<ConstantSpectrum>()->Sample(lambda);
    case TypeIndex<ProductSpectrum>():
        return Cast<ProductSpectrum>()->Sample(lambda);
    case TypeIndex<ScaledSpectrum>():
        return Cast<ScaledSpectrum>()->Sample(lambda);
    case TypeIndex<PiecewiseLinearSpectrum>():
        return Cast<PiecewiseLinearSpectrum>()->Sample(lambda);
    case TypeIndex<DenselySampledSpectrum>():
        return Cast<DenselySampledSpectrum>()->Sample(lambda);
    case TypeIndex<RGBReflectanceSpectrum>():
        return Cast<RGBReflectanceSpectrum>()->Sample(lambda);
    case TypeIndex<RGBSpectrum>():
        return Cast<RGBSpectrum>()->Sample(lambda);
    default:
        LOG_FATAL("Unhandled Spectrum type %d", Tag());
        return {};
    }
}

inline Float SpectrumHandle::MaxValue() const {
    switch (Tag()) {
    case TypeIndex<BlackbodySpectrum>():
        return Cast<BlackbodySpectrum>()->MaxValue();
    case TypeIndex<ConstantSpectrum>():
        return Cast<ConstantSpectrum>()->MaxValue();
    case TypeIndex<ProductSpectrum>():
        return Cast<ProductSpectrum>()->MaxValue();
    case TypeIndex<ScaledSpectrum>():
        return Cast<ScaledSpectrum>()->MaxValue();
    case TypeIndex<PiecewiseLinearSpectrum>():
        return Cast<PiecewiseLinearSpectrum>()->MaxValue();
    case TypeIndex<DenselySampledSpectrum>():
        return Cast<DenselySampledSpectrum>()->MaxValue();
    case TypeIndex<RGBReflectanceSpectrum>():
        return Cast<RGBReflectanceSpectrum>()->MaxValue();
    case TypeIndex<RGBSpectrum>():
        return Cast<RGBSpectrum>()->MaxValue();
    default:
        LOG_FATAL("Unhandled Spectrum type %d", Tag());
        return {};
    }
}

}  // namespace pbrt

#endif  // PBRT_SPECTRUM_H
