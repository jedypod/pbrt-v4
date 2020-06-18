// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#ifndef PBRT_STRING_H
#define PBRT_STRING_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <string>
#include <string_view>
#include <vector>

namespace pbrt {

bool Atoi(std::string_view str, int *);
bool Atof(std::string_view str, float *);
bool Atod(std::string_view str, double *);

std::vector<std::string> SplitStringsFromWhitespace(std::string_view str);

std::vector<std::string> SplitString(std::string_view str, char ch);
pstd::optional<std::vector<int>> SplitStringToInts(std::string_view str, char ch);
pstd::optional<std::vector<Float>> SplitStringToFloats(std::string_view str, char ch);
pstd::optional<std::vector<double>> SplitStringToDoubles(std::string_view str, char ch);

}  // namespace pbrt

#endif  // PBRT_STRING_H
