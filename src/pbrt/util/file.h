// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_FILEUTIL_H
#define PBRT_UTIL_FILEUTIL_H

// core/file.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>

#include <string>
#include <vector>

namespace pbrt {

std::string ResolveFilename(const std::string &filename);
void SetSearchDirectory(const std::string &filename);

bool HasExtension(const std::string &filename, const std::string &ext);
std::string RemoveExtension(const std::string &filename);

std::vector<std::string> MatchingFilenames(const std::string &base);

pstd::optional<std::string> ReadFileContents(const std::string &filename);
pstd::optional<std::vector<float>> ReadFloatFile(const std::string &filename);
bool WriteFile(const std::string &filename, const std::string &contents);

}  // namespace pbrt

#endif  // PBRT_UTIL_FILEUTIL_H
