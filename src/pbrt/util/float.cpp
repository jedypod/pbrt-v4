// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/util/float.h>

#include <pbrt/util/print.h>

namespace pbrt {

std::string Half::ToString() const {
    return StringPrintf("%f", (float)(*this));
}

}  // namespace pbrt
