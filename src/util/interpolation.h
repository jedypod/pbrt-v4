
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
#include "ext/google/array_slice.h"

namespace pbrt {

// Spline Interpolation Declarations
Float CatmullRom(gtl::ArraySlice<Float> nodes, gtl::ArraySlice<Float> values,
                 Float x);
bool CatmullRomWeights(gtl::ArraySlice<Float> nodes, Float x, int *offset,
                       gtl::MutableArraySlice<Float> weights);
Float SampleCatmullRom(gtl::ArraySlice<Float> nodes, gtl::ArraySlice<Float> f,
                       gtl::ArraySlice<Float> cdf, Float sample, Float *fval = nullptr,
                       Float *pdf = nullptr);
Float SampleCatmullRom2D(gtl::ArraySlice<Float> nodes1,
                         gtl::ArraySlice<Float> nodes2,
                         gtl::ArraySlice<Float> values,
                         gtl::ArraySlice<Float> cdf, Float alpha, Float sample,
                         Float *fval = nullptr, Float *pdf = nullptr);
Float IntegrateCatmullRom(gtl::ArraySlice<Float> nodes,
                          gtl::ArraySlice<Float> values,
                          gtl::MutableArraySlice<Float> cdf);
Float InvertCatmullRom(gtl::ArraySlice<Float> x, gtl::ArraySlice<Float> values,
                       Float u);

// Fourier Interpolation Declarations
Float Fourier(gtl::ArraySlice<Float> a, double cosPhi);
Float SampleFourier(gtl::ArraySlice<Float> ak, gtl::ArraySlice<Float> recip,
                    Float u, Float *pdf, Float *phiPtr);

}  // namespace pbrt

#endif  // PBRT_CORE_INTERPOLATION_H
