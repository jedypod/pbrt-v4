// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/util/rng.h>

#include <pbrt/util/print.h>

#include <cinttypes>

namespace pbrt {

std::string RNG::ToString() const {
    return StringPrintf("[ RNG state: %" PRIu64 " inc: %" PRIu64 " ]", state, inc);
}

}  // namespace pbrt
