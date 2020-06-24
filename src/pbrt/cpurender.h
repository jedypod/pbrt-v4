
#ifndef PBRT_CPURENDER_H
#define PBRT_CPURENDER_H

#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/util/spectrum.h>

#include <map>
#include <memory>

namespace pbrt {

void CPURender(GeneralScene &scene);

} // namespace pbrt

#endif // PBRT_CPURENDER_H
