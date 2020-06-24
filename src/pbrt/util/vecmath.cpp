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

#include <pbrt/util/vecmath.h>
#include <pbrt/util/math.h>
#include <pbrt/util/print.h>
#include <pbrt/util/stats.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

namespace pbrt {

template<> std::string internal::ToString2<FloatInterval> (FloatInterval x, FloatInterval y) {
    return StringPrintf("[ %s %s ]", x, y);
}

template<> std::string internal::ToString3<FloatInterval> (FloatInterval x, FloatInterval y,
                                                           FloatInterval z) {
    return StringPrintf("[ %s %s %s ]", x, y, z);
}

template <typename T> std::string internal::ToString2(T x, T y) {
    if (std::is_floating_point<T>::value)
        return StringPrintf("[ %f, %f ]", x, y);
    else
        return StringPrintf("[ %d, %d ]", x, y);
}

template <typename T> std::string internal::ToString3(T x, T y, T z) {
    if (std::is_floating_point<T>::value)
        return StringPrintf("[ %f, %f, %f ]", x, y, z);
    else
        return StringPrintf("[ %d, %d, %d ]", x, y, z);
}

template std::string internal::ToString2(float, float);
template std::string internal::ToString2(double, double);
template std::string internal::ToString2(int, int);
template std::string internal::ToString3(float, float, float);
template std::string internal::ToString3(double, double, double);
template std::string internal::ToString3(int, int, int);


// Quaternion Method Definitions
std::string Quaternion::ToString() const {
    return StringPrintf("[ %f, %f, %f, %f ]", v.x, v.y, v.z, w);
}

std::string DirectionCone::ToString() const {
    return StringPrintf("[ DirectionCone w: %s cosTheta: %f ]", w, cosTheta);
}

}  // namespace pbrt
