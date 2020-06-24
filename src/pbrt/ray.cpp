
#include <pbrt/ray.h>

#include <pbrt/base.h>
#include <pbrt/util/print.h>

namespace pbrt {

std::string Ray::ToString() const {
    return StringPrintf("[ o: %s d: %s time: %f, medium:%s ]",
                        o, d, time, medium != nullptr ? medium->ToString().c_str() : "(none)");
}

std::string RayDifferential::ToString() const {
    return StringPrintf("[ ray: %s differentials: %s xo: %s xd: %s yo: %s yd: %s ]",
                        ((const Ray &)(*this)),
                        hasDifferentials ? "true" : "false", rxOrigin,
                        rxDirection, ryOrigin, ryDirection);
}

} // namespace pbrt
