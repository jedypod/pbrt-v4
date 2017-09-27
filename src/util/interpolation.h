
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

#ifndef PBRT_CORE_INTERPOLATION_H
#define PBRT_CORE_INTERPOLATION_H

// core/interpolation.h*
#include "pbrt.h"
#include <absl/types/span.h>

namespace pbrt {

// Spline Interpolation Declarations
Float CatmullRom(absl::Span<const Float> nodes, absl::Span<const Float> values,
                 Float x);
bool CatmullRomWeights(absl::Span<const Float> nodes, Float x, int *offset,
                       absl::Span<Float> weights);
Float SampleCatmullRom(absl::Span<const Float> nodes, absl::Span<const Float> f,
                       absl::Span<const Float> cdf, Float sample, Float *fval = nullptr,
                       Float *pdf = nullptr);
Float SampleCatmullRom2D(absl::Span<const Float> nodes1,
                         absl::Span<const Float> nodes2,
                         absl::Span<const Float> values,
                         absl::Span<const Float> cdf, Float alpha, Float sample,
                         Float *fval = nullptr, Float *pdf = nullptr);
Float IntegrateCatmullRom(absl::Span<const Float> nodes,
                          absl::Span<const Float> values,
                          absl::Span<Float> cdf);
Float InvertCatmullRom(absl::Span<const Float> x, absl::Span<const Float> values,
                       Float u);

// Fourier Interpolation Declarations
Float Fourier(absl::Span<const Float> a, double cosPhi);
Float SampleFourier(absl::Span<const Float> ak, absl::Span<const Float> recip,
                    Float u, Float *pdf, Float *phiPtr);

}  // namespace pbrt

#endif  // PBRT_CORE_INTERPOLATION_H
