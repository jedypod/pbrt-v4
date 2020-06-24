
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

#ifndef PBRT_CORE_SOBOLMATRICES_H
#define PBRT_CORE_SOBOLMATRICES_H

// sampling/sobolmatrices.h*
#include <pbrt/pbrt.h>

#include <cstdint>

namespace pbrt {

// Sobol Matrix Declarations
static constexpr int NSobolDimensions = 1024;
static constexpr int SobolMatrixSize = 52;

extern PBRT_CONST uint32_t SobolMatrices32[NSobolDimensions * SobolMatrixSize];
extern PBRT_CONST uint64_t SobolMatrices64[NSobolDimensions * SobolMatrixSize];
extern PBRT_CONST uint64_t VdCSobolMatrices[][SobolMatrixSize];
extern PBRT_CONST uint64_t VdCSobolMatricesInv[][SobolMatrixSize];

// Define 2D Sobol$'$ generator matrices _CSobol[2]_
PBRT_CONST uint32_t CSobol[2][32] = {
    {0x80000000, 0x40000000, 0x20000000, 0x10000000, 0x8000000, 0x4000000,
     0x2000000, 0x1000000, 0x800000, 0x400000, 0x200000, 0x100000, 0x80000,
     0x40000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000, 0x1000, 0x800,
     0x400, 0x200, 0x100, 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1},
    {0x80000000, 0xc0000000, 0xa0000000, 0xf0000000, 0x88000000, 0xcc000000,
     0xaa000000, 0xff000000, 0x80800000, 0xc0c00000, 0xa0a00000, 0xf0f00000,
     0x88880000, 0xcccc0000, 0xaaaa0000, 0xffff0000, 0x80008000, 0xc000c000,
     0xa000a000, 0xf000f000, 0x88008800, 0xcc00cc00, 0xaa00aa00, 0xff00ff00,
     0x80808080, 0xc0c0c0c0, 0xa0a0a0a0, 0xf0f0f0f0, 0x88888888, 0xcccccccc,
     0xaaaaaaaa, 0xffffffff}};

}  // namespace pbrt

#endif  // PBRT_CORE_SOBOLMATRICES_H
