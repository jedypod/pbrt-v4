
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

#ifndef PBRT_CORE_SPECTRUM_H
#define PBRT_CORE_SPECTRUM_H

// core/spectrum.h*
#include "pbrt.h"

#include "mathutil.h"
#include "stringprint.h"
#include "ext/google/array_slice.h"
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <stdio.h>

namespace pbrt {

// Spectrum Utility Declarations
static const int sampledLambdaStart = 400;
static const int sampledLambdaEnd = 700;
static const int nSpectralSamples = 60;
extern bool SpectrumSamplesSorted(gtl::ArraySlice<Float> lambda,
                                  gtl::ArraySlice<Float> vals);
extern void SortSpectrumSamples(gtl::MutableArraySlice<Float> lambda,
                                gtl::MutableArraySlice<Float> vals);
extern Float AverageSpectrumSamples(gtl::ArraySlice<Float> lambda,
                                    gtl::ArraySlice<Float> vals,
                                    Float lambdaStart, Float lambdaEnd);
inline void XYZToRGB(const Float xyz[3], Float rgb[3]) {
    rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
    rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
    rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
}

inline void RGBToXYZ(const Float rgb[3], Float xyz[3]) {
    xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
    xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
    xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
}

enum class SpectrumType { Reflectance, Illuminant };
extern Float InterpolateSpectrumSamples(gtl::ArraySlice<Float> lambda,
                                        gtl::ArraySlice<Float> vals, Float l);
extern void Blackbody(gtl::ArraySlice<Float> lambda, Float T,
                      gtl::MutableArraySlice<Float> Le);
extern void BlackbodyNormalized(gtl::ArraySlice<Float> lambda, Float T,
                                gtl::MutableArraySlice<Float> vals);

// Spectral Data Declarations
static const int nCIESamples = 471;
extern const Float CIE_X[nCIESamples];
extern const Float CIE_Y[nCIESamples];
extern const Float CIE_Z[nCIESamples];
extern const Float CIE_lambda[nCIESamples];
static const Float CIE_Y_integral = 106.856895;
static const int nRGB2SpectSamples = 32;
extern const Float RGB2SpectLambda[nRGB2SpectSamples];
extern const Float RGBRefl2SpectWhite[nRGB2SpectSamples];
extern const Float RGBRefl2SpectCyan[nRGB2SpectSamples];
extern const Float RGBRefl2SpectMagenta[nRGB2SpectSamples];
extern const Float RGBRefl2SpectYellow[nRGB2SpectSamples];
extern const Float RGBRefl2SpectRed[nRGB2SpectSamples];
extern const Float RGBRefl2SpectGreen[nRGB2SpectSamples];
extern const Float RGBRefl2SpectBlue[nRGB2SpectSamples];
extern const Float RGBIllum2SpectWhite[nRGB2SpectSamples];
extern const Float RGBIllum2SpectCyan[nRGB2SpectSamples];
extern const Float RGBIllum2SpectMagenta[nRGB2SpectSamples];
extern const Float RGBIllum2SpectYellow[nRGB2SpectSamples];
extern const Float RGBIllum2SpectRed[nRGB2SpectSamples];
extern const Float RGBIllum2SpectGreen[nRGB2SpectSamples];
extern const Float RGBIllum2SpectBlue[nRGB2SpectSamples];

// Spectrum Declarations
template <typename Child, int nSpectrumSamples>
class CoefficientSpectrum {
  public:
    // CoefficientSpectrum Public Methods
    CoefficientSpectrum(Float v = 0.f) {
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] = v;
        DCHECK(!HasNaNs());
    }
#ifdef DEBUG
    CoefficientSpectrum(const Child &s) {
        DCHECK(!s.HasNaNs());
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] = s.c[i];
    }
    Child &operator=(const Child &s) {
        DCHECK(!s.HasNaNs());
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] = s.c[i];
        return static_cast<Child *>(*this);
    }
#endif  // DEBUG
    Child &operator+=(const Child &s2) {
        DCHECK(!s2.HasNaNs());
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] += s2.c[i];
        return static_cast<Child &>(*this);
    }
    Child operator+(const Child &s2) const {
        DCHECK(!s2.HasNaNs());
        Child ret = static_cast<const Child &>(*this);
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] += s2.c[i];
        return ret;
    }
    Child operator-(const Child &s2) const {
        DCHECK(!s2.HasNaNs());
        Child ret = static_cast<const Child &>(*this);
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] -= s2.c[i];
        return ret;
    }
    Child operator/(const Child &s2) const {
        DCHECK(!s2.HasNaNs());
        Child ret = static_cast<const Child &>(*this);
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] /= s2.c[i];
        return ret;
    }
    Child operator*(const Child &sp) const {
        DCHECK(!sp.HasNaNs());
        Child ret = static_cast<const Child &>(*this);
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] *= sp.c[i];
        return ret;
    }
    Child &operator*=(const Child &sp) {
        DCHECK(!sp.HasNaNs());
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] *= sp.c[i];
        return static_cast<Child &>(*this);
    }
    Child operator*(Float a) const {
        Child ret = static_cast<const Child &>(*this);
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] *= a;
        DCHECK(!ret.HasNaNs());
        return ret;
    }
    Child &operator*=(Float a) {
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] *= a;
        DCHECK(!HasNaNs());
        return static_cast<Child &>(*this);
    }
    friend inline Child operator*(Float a, const Child &s) {
        DCHECK(!std::isnan(a) && !s.HasNaNs());
        return s * a;
    }
    Child operator/(Float a) const {
        CHECK_NE(a, 0);
        DCHECK(!std::isnan(a));
        Child ret = static_cast<const Child &>(*this);
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] /= a;
        DCHECK(!ret.HasNaNs());
        return ret;
    }
    Child &operator/=(Float a) {
        CHECK_NE(a, 0);
        DCHECK(!std::isnan(a));
        for (int i = 0; i < nSpectrumSamples; ++i) c[i] /= a;
        return static_cast<Child &>(*this);
    }
    bool operator==(const Child &sp) const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != sp.c[i]) return false;
        return true;
    }
    bool operator!=(const Child &sp) const {
        return !(*this == sp);
    }
    bool IsBlack() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (c[i] != 0.) return false;
        return true;
    }
    friend Child Sqrt(const Child &s) {
        Child ret;
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = std::sqrt(s.c[i]);
        DCHECK(!ret.HasNaNs());
        return ret;
    }
    friend inline Child Pow(const Child &s, Float e) {
        CoefficientSpectrum<Child, nSpectrumSamples> ret;
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = std::pow(s.c[i], e);
        DCHECK(!ret.HasNaNs());
        return ret;
    }
    Child operator-() const {
        Child ret;
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = -c[i];
        return ret;
    }
    friend Child Exp(const Child &s) {
        Child ret;
        for (int i = 0; i < nSpectrumSamples; ++i) ret.c[i] = std::exp(s.c[i]);
        DCHECK(!ret.HasNaNs());
        return ret;
    }
    friend std::ostream &operator<<(std::ostream &os, const Child &s) {
        return os << s.ToString();
    }
    std::string ToString() const {
        std::string str = "[ ";
        for (int i = 0; i < nSpectrumSamples; ++i) {
            str += StringPrintf("%f", c[i]);
            if (i + 1 < nSpectrumSamples) str += ", ";
        }
        str += " ]";
        return str;
    }
    Child Clamp(Float low = 0, Float high = Infinity) const {
        Child ret;
        for (int i = 0; i < nSpectrumSamples; ++i)
            ret.c[i] = pbrt::Clamp(c[i], low, high);
        DCHECK(!ret.HasNaNs());
        return ret;
    }
    Float MaxComponentValue() const {
        Float m = c[0];
        for (int i = 1; i < nSpectrumSamples; ++i)
            m = std::max(m, c[i]);
        return m;
    }
    Float Average() const {
        Float sum = 0;
        for (int i = 0; i < nSpectrumSamples; ++i)
            sum += c[i];
        return sum / nSpectrumSamples;
    }

    bool HasNaNs() const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (std::isnan(c[i])) return true;
        return false;
    }
    bool Write(FILE *f) const {
        for (int i = 0; i < nSpectrumSamples; ++i)
            if (fprintf(f, "%.17g ", c[i]) < 0) return false;
        return true;
    }
    bool Read(FILE *f) {
        for (int i = 0; i < nSpectrumSamples; ++i) {
            double v;
            if (fscanf(f, "%lf ", &v) != 1) return false;
            c[i] = v;
        }
        return true;
    }
    Float &operator[](int i) {
        DCHECK(i >= 0 && i < nSpectrumSamples);
        return c[i];
    }
    Float operator[](int i) const {
        DCHECK(i >= 0 && i < nSpectrumSamples);
        return c[i];
    }

    // CoefficientSpectrum Public Data
    static const int nSamples = nSpectrumSamples;

  protected:
    // CoefficientSpectrum Protected Data
    Float c[nSpectrumSamples];
};

class SampledSpectrum :
    public CoefficientSpectrum<SampledSpectrum, nSpectralSamples> {
  public:
    using CoefficientSpectrum<SampledSpectrum, nSpectralSamples>::c;

    // SampledSpectrum Public Methods
    SampledSpectrum(Float v = 0.f) : CoefficientSpectrum(v) {}
/*CO    SampledSpectrum(const CoefficientSpectrum<nSpectralSamples> &v)*/
/*CO        : CoefficientSpectrum<nSpectralSamples>(v) {}*/
    static SampledSpectrum FromSampled(gtl::ArraySlice<Float> lambda,
                                       gtl::ArraySlice<Float> v) {
        // Sort samples if unordered, use sorted for returned spectrum
        CHECK_EQ(lambda.size(), v.size());
        if (!SpectrumSamplesSorted(lambda, v)) {
            std::vector<Float> slambda(lambda.begin(), lambda.end());
            std::vector<Float> sv(v.begin(), v.end());
            SortSpectrumSamples(&slambda, &sv);
            return FromSampled(slambda, sv);
        }
        SampledSpectrum r;
        for (int i = 0; i < nSpectralSamples; ++i) {
            // Compute average value of given SPD over $i$th sample's range
            Float lambda0 = Lerp(Float(i) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
            Float lambda1 = Lerp(Float(i + 1) / Float(nSpectralSamples),
                                 sampledLambdaStart, sampledLambdaEnd);
            r.c[i] = AverageSpectrumSamples(lambda, v, lambda0, lambda1);
        }
        return r;
    }
    static void Init() {
        // Compute XYZ matching functions for _SampledSpectrum_
        for (int i = 0; i < nSpectralSamples; ++i) {
            Float wl0 = Lerp(Float(i) / Float(nSpectralSamples),
                             sampledLambdaStart, sampledLambdaEnd);
            Float wl1 = Lerp(Float(i + 1) / Float(nSpectralSamples),
                             sampledLambdaStart, sampledLambdaEnd);
            X.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_X, wl0, wl1);
            Y.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Y, wl0, wl1);
            Z.c[i] = AverageSpectrumSamples(CIE_lambda, CIE_Z, wl0, wl1);
        }

        // Compute RGB to spectrum functions for _SampledSpectrum_
        for (int i = 0; i < nSpectralSamples; ++i) {
            Float wl0 = Lerp(Float(i) / Float(nSpectralSamples),
                             sampledLambdaStart, sampledLambdaEnd);
            Float wl1 = Lerp(Float(i + 1) / Float(nSpectralSamples),
                             sampledLambdaStart, sampledLambdaEnd);
            rgbRefl2SpectWhite.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectWhite, wl0, wl1);
            rgbRefl2SpectCyan.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectCyan, wl0, wl1);
            rgbRefl2SpectMagenta.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectMagenta, wl0, wl1);
            rgbRefl2SpectYellow.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectYellow, wl0, wl1);
            rgbRefl2SpectRed.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectRed, wl0, wl1);
            rgbRefl2SpectGreen.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectGreen, wl0, wl1);
            rgbRefl2SpectBlue.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBRefl2SpectBlue, wl0, wl1);

            rgbIllum2SpectWhite.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectWhite, wl0, wl1);
            rgbIllum2SpectCyan.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectCyan, wl0, wl1);
            rgbIllum2SpectMagenta.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectMagenta, wl0, wl1);
            rgbIllum2SpectYellow.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectYellow, wl0, wl1);
            rgbIllum2SpectRed.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectRed, wl0, wl1);
            rgbIllum2SpectGreen.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectGreen, wl0, wl1);
            rgbIllum2SpectBlue.c[i] = AverageSpectrumSamples(
                RGB2SpectLambda, RGBIllum2SpectBlue, wl0, wl1);
        }
    }
    void ToXYZ(Float xyz[3]) const {
        xyz[0] = xyz[1] = xyz[2] = 0.f;
        for (int i = 0; i < nSpectralSamples; ++i) {
            xyz[0] += X.c[i] * c[i];
            xyz[1] += Y.c[i] * c[i];
            xyz[2] += Z.c[i] * c[i];
        }
        Float scale = Float(sampledLambdaEnd - sampledLambdaStart) /
                      Float(CIE_Y_integral * nSpectralSamples);
        xyz[0] *= scale;
        xyz[1] *= scale;
        xyz[2] *= scale;
    }
    Float y() const {
        Float yy = 0.f;
        for (int i = 0; i < nSpectralSamples; ++i) yy += Y.c[i] * c[i];
        return yy * Float(sampledLambdaEnd - sampledLambdaStart) /
               Float(CIE_Y_integral * nSpectralSamples);
    }
    void ToRGB(Float rgb[3]) const {
        Float xyz[3];
        ToXYZ(xyz);
        XYZToRGB(xyz, rgb);
    }
    RGBSpectrum ToRGBSpectrum() const;
    static SampledSpectrum FromRGB(
        const Float rgb[3], SpectrumType type = SpectrumType::Illuminant);
    static SampledSpectrum FromXYZ(
        const Float xyz[3], SpectrumType type = SpectrumType::Reflectance) {
        Float rgb[3];
        XYZToRGB(xyz, rgb);
        return FromRGB(rgb, type);
    }
    SampledSpectrum(const RGBSpectrum &r,
                    SpectrumType type = SpectrumType::Reflectance);

  private:
    // SampledSpectrum Private Data
    static SampledSpectrum X, Y, Z;
    static SampledSpectrum rgbRefl2SpectWhite, rgbRefl2SpectCyan;
    static SampledSpectrum rgbRefl2SpectMagenta, rgbRefl2SpectYellow;
    static SampledSpectrum rgbRefl2SpectRed, rgbRefl2SpectGreen;
    static SampledSpectrum rgbRefl2SpectBlue;
    static SampledSpectrum rgbIllum2SpectWhite, rgbIllum2SpectCyan;
    static SampledSpectrum rgbIllum2SpectMagenta, rgbIllum2SpectYellow;
    static SampledSpectrum rgbIllum2SpectRed, rgbIllum2SpectGreen;
    static SampledSpectrum rgbIllum2SpectBlue;
};

class RGBSpectrum : public CoefficientSpectrum<RGBSpectrum, 3> {
  public:
    using CoefficientSpectrum<RGBSpectrum, 3>::c;

    // RGBSpectrum Public Methods
RGBSpectrum(Float v = 0.f) : CoefficientSpectrum<RGBSpectrum, 3>(v) {}
/*CO    RGBSpectrum(const CoefficientSpectrum<3> &v) : CoefficientSpectrum<RGBSpectrum 3>(v) {}*/
    RGBSpectrum(const RGBSpectrum &s,
                SpectrumType type = SpectrumType::Reflectance) {
        *this = s;
    }
    static RGBSpectrum FromRGB(const Float rgb[3],
                               SpectrumType type = SpectrumType::Reflectance) {
        RGBSpectrum s;
        s.c[0] = rgb[0];
        s.c[1] = rgb[1];
        s.c[2] = rgb[2];
        DCHECK(!s.HasNaNs());
        return s;
    }
    void ToRGB(Float *rgb) const {
        rgb[0] = c[0];
        rgb[1] = c[1];
        rgb[2] = c[2];
    }
    const RGBSpectrum &ToRGBSpectrum() const { return *this; }
    void ToXYZ(Float xyz[3]) const { RGBToXYZ(c, xyz); }
    static RGBSpectrum FromXYZ(const Float xyz[3],
                               SpectrumType type = SpectrumType::Reflectance) {
        RGBSpectrum r;
        XYZToRGB(xyz, r.c);
        return r;
    }
    Float y() const {
        const Float YWeight[3] = {0.212671f, 0.715160f, 0.072169f};
        return YWeight[0] * c[0] + YWeight[1] * c[1] + YWeight[2] * c[2];
    }
    static RGBSpectrum FromSampled(gtl::ArraySlice<Float> lambda,
                                   gtl::ArraySlice<Float> v) {
        // Sort samples if unordered, use sorted for returned spectrum
        if (!SpectrumSamplesSorted(lambda, v)) {
            std::vector<Float> slambda(lambda.begin(), lambda.end());
            std::vector<Float> sv(v.begin(), v.end());
            SortSpectrumSamples(&slambda, &sv);
            return FromSampled(slambda, sv);
        }
        Float xyz[3] = {0, 0, 0};
        for (int i = 0; i < nCIESamples; ++i) {
            Float val = InterpolateSpectrumSamples(lambda, v, CIE_lambda[i]);
            xyz[0] += val * CIE_X[i];
            xyz[1] += val * CIE_Y[i];
            xyz[2] += val * CIE_Z[i];
        }
        Float scale = Float(CIE_lambda[nCIESamples - 1] - CIE_lambda[0]) /
                      Float(CIE_Y_integral * nCIESamples);
        xyz[0] *= scale;
        xyz[1] *= scale;
        xyz[2] *= scale;
        return FromXYZ(xyz);
    }
};

// Spectrum Inline Functions
inline RGBSpectrum Lerp(Float t, const RGBSpectrum &s1, const RGBSpectrum &s2) {
    return (1 - t) * s1 + t * s2;
}

inline SampledSpectrum Lerp(Float t, const SampledSpectrum &s1,
                            const SampledSpectrum &s2) {
    return (1 - t) * s1 + t * s2;
}

void ResampleLinearSpectrum(const Float *lambdaIn, const Float *vIn, int nIn,
                            Float lambdaMin, Float lambdaMax, int nOut,
                            Float *vOut);

}  // namespace pbrt

#endif  // PBRT_CORE_SPECTRUM_H
