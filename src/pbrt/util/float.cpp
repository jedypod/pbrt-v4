
#include <pbrt/util/float.h>

#include <pbrt/util/print.h>

namespace pbrt {

std::string Half::ToString() const {
    return StringPrintf("%f", (float)(*this));
}

} // namespace pbrt

