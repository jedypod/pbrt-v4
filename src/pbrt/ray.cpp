// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/ray.h>

#include <pbrt/util/print.h>

namespace pbrt {

std::string Ray::ToString() const {
    return StringPrintf("[ o: %s d: %s time: %f, medium: %s ]", o, d, time, medium);
}

std::string RayDifferential::ToString() const {
    return StringPrintf("[ ray: %s differentials: %s xo: %s xd: %s yo: %s yd: %s ]",
                        ((const Ray &)(*this)), hasDifferentials ? "true" : "false",
                        rxOrigin, rxDirection, ryOrigin, ryDirection);
}

}  // namespace pbrt
