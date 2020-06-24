

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

#ifndef PBRT_SAMPLING_BLUENOISE_H
#define PBRT_SAMPLING_BLUENOISE_H

#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>

namespace pbrt {

static constexpr int BlueNoiseResolution = 128;
static constexpr int NumBlueNoiseTextures = 48;

extern PBRT_CONST uint16_t
BlueNoiseTextures[NumBlueNoiseTextures][BlueNoiseResolution][BlueNoiseResolution];

// Returns a sample in [0,1].
PBRT_HOST_DEVICE_INLINE
float BlueNoise(int textureIndex, int px, int py) {
    CHECK(textureIndex >= 0 && px >= 0 && py >= 0);
    textureIndex %= NumBlueNoiseTextures;
    int x = px % BlueNoiseResolution, y = py % BlueNoiseResolution;
    return BlueNoiseTextures[textureIndex][x][y] / 65535.f;
}

}  // namespace pbrt

#endif // PBRT_SAMPLING_BLUENOISE_H
