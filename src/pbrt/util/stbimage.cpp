// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/util/check.h>

#define STBI_NO_PNG
// too old school
#define STBI_NO_PIC
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ASSERT CHECK
#include <stb/stb_image.h>
