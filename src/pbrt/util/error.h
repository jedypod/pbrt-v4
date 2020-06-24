
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

namespace pbrt {

// FileLoc represents a position in a file being parsed.
struct FileLoc {
    FileLoc() = default;
    FileLoc(pstd::string_view filename) : filename(filename) {}

    std::string ToString() const;

    pstd::string_view filename;
    int line = 1, column = 0;
};

void Warning(const FileLoc *loc, const char *message);

template <typename... Args>
inline void Warning(const char *fmt, Args&&... args) {
    Warning(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Warning(const FileLoc *loc, const char *fmt, Args&&... args) {
    Warning(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

void Error(const FileLoc *loc, const char *message);

template <typename... Args>
inline void Error(const char *fmt, Args&&... args) {
    Error(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
inline void Error(const FileLoc *loc, const char *fmt, Args&&... args) {
    Error(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

[[noreturn]] void ErrorExit(const FileLoc *loc, const char *message);

template <typename... Args>
[[noreturn]] inline void ErrorExit(const char *fmt, Args&&... args) {
    ErrorExit(nullptr, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

template <typename... Args>
[[noreturn]] inline void ErrorExit(const FileLoc *loc, const char *fmt, Args&&... args) {
    ErrorExit(loc, StringPrintf(fmt, std::forward<Args>(args)...).c_str());
}

}  // namespace pbrt

#endif  // PBRT_UTIL_ERROR_H
