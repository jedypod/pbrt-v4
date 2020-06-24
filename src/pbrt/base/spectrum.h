// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_BASE_SPECTRUM_H
#define PBRT_BASE_SPECTRUM_H

#include <pbrt/pbrt.h>

#include <pbrt/util/taggedptr.h>

#include <string>

namespace pbrt {

class SampledWavelengths;

class BlackbodySpectrum;
class ConstantSpectrum;
class ProductSpectrum;
class ScaledSpectrum;
class PiecewiseLinearSpectrum;
class DenselySampledSpectrum;
class RGBReflectanceSpectrum;
class RGBSpectrum;

class SpectrumHandle
    : public TaggedPointer<BlackbodySpectrum, ConstantSpectrum, ProductSpectrum,
                           ScaledSpectrum, PiecewiseLinearSpectrum,
                           DenselySampledSpectrum, RGBReflectanceSpectrum, RGBSpectrum> {
  public:
    using TaggedPointer::TaggedPointer;

    PBRT_CPU_GPU
    Float operator()(Float lambda) const;

    PBRT_CPU_GPU
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    Float MaxValue() const;

    std::string ToString() const;
    std::string ParameterType() const;
    std::string ParameterString() const;
};

} // namespace pbrt

#endif
