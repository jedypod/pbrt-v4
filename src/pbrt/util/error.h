// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_UTIL_ERROR_H
#define PBRT_UTIL_ERROR_H

// util/error.h*

#include <pbrt/pbrt.h>

#include <pbrt/util/print.h>
#include <pbrt/util/pstd.h>

#include <string>
#include <string_view>

namespace pbrt {

// FileLoc represents a position in a file being parsed.
struct FileLoc {
    FileLoc() = default;
    FileLoc(std::string_view filename) : filename(filename) {}

    std::string ToString() const;

    std::string_view filename;
    int line = 1, column = 0;
};

void SuppressErrorMessages();

void Warning(const FileLoc *loc, const char *message);

template <typename... Args>
inline void Warning(const char *fmt, Args &&... args) {
    Warning(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Warning(const FileLoc *loc, const char *fmt, Args &&... args) {
    Warning(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

void Error(const FileLoc *loc, const char *message);

template <typename... Args>
inline void Error(const char *fmt, Args &&... args) {
    Error(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Error(const FileLoc *loc, const char *fmt, Args &&... args) {
    Error(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

[[noreturn]] void ErrorExit(const FileLoc *loc, const char *message);

template <typename... Args>
[[noreturn]] inline void ErrorExit(const char *fmt, Args &&... args) {
    ErrorExit(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
[[noreturn]] inline void ErrorExit(const FileLoc *loc, const char *fmt, Args &&... args) {
    ErrorExit(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

int LastError();
std::string ErrorString(int errorId = LastError());

}  // namespace pbrt

#endif  // PBRT_UTIL_ERROR_H
