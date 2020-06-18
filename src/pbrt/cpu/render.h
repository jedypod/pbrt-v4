// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_CPURENDER_H
#define PBRT_CPURENDER_H

#include <pbrt/pbrt.h>

namespace pbrt {

class ParsedScene;

void CPURender(ParsedScene &scene);

}  // namespace pbrt

#endif  // PBRT_CPURENDER_H
