// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_SHAPES_LOOPSUBDIV_H
#define PBRT_SHAPES_LOOPSUBDIV_H

// shapes/loopsubdiv.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

namespace pbrt {

// LoopSubdiv Declarations
TriangleMesh *LoopSubdivide(const Transform *worldFromObject, bool reverseOrientation,
                            int nLevels, pstd::span<const int> vertexIndices,
                            pstd::span<const Point3f> p, Allocator alloc);

}  // namespace pbrt

#endif  // PBRT_SHAPES_LOOPSUBDIV_H
