
#ifndef PBRT_STRING_H
#define PBRT_STRING_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <string>
#include <vector>

namespace pbrt {

bool Atoi(pstd::string_view str, int *);
bool Atof(pstd::string_view str, float *);
bool Atod(pstd::string_view str, double *);

std::vector<std::string> SplitStringsFromWhitespace(pstd::string_view str);

std::vector<std::string> SplitString(pstd::string_view str, char ch);
pstd::optional<std::vector<int>> SplitStringToInts(pstd::string_view str, char ch);
pstd::optional<std::vector<Float>> SplitStringToFloats(pstd::string_view str, char ch);
pstd::optional<std::vector<double>> SplitStringToDoubles(pstd::string_view str, char ch);

} // namespace pbrt

#endif // PBRT_STRING_H
