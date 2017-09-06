
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

#ifndef PBRT_UTIL_ARGS_H
#define PBRT_UTIL_ARGS_H

#include "pbrt.h"
#include "util/stringprint.h"

#include <cctype>
#include <cstring>
#include <functional>
#include <string>

namespace pbrt {
namespace {

// Downcase the string and remove any '-' or '_' characters; thus we can be
// a little flexible in what we match for argument names.
std::string normalize(const std::string &str) {
    std::string ret;
    for (unsigned char c : str) {
        if (c != '_' && c != '-') ret += std::tolower(c);
    }
    return ret;
}

bool initArg(const std::string &str, int *ptr) {
    if (str.empty() || (!std::isdigit(str[0]) && str[0] != '-')) return false;
    *ptr = std::stoi(str);
    return true;
}

bool initArg(const std::string &str, Float *ptr) {
    if (str.empty() ||
        (!std::isdigit(str[0]) && str[0] != '-' && str[0] != '.'))
        return false;
    *ptr = std::stof(str);
    return true;
}

bool initArg(const std::string &str, char **ptr) {
    if (str.empty()) return false;
    *ptr = new char[str.size() + 1];
    std::strcpy(*ptr, str.c_str());
    return true;
}

bool initArg(const std::string &str, std::string *ptr) {
    if (str.empty()) return false;
    *ptr = str;
    return true;
}

bool initArg(const std::string &str, bool *ptr) {
    if (normalize(str) == "false") {
        *ptr = false;
        return true;
    } else if (normalize(str) == "true") {
        *ptr = true;
        return true;
    }
    return false;
}

bool matchPrefix(const std::string &str, const std::string &prefix) {
    if (prefix.size() > str.size()) return false;
    for (size_t i = 0; i < prefix.size(); ++i)
        if (prefix[i] != str[i]) return false;
    return true;
}

template <typename T>
bool enable(T *ptr) {
    return false;
}

bool enable(bool *ptr) {
    *ptr = true;
    return true;
}

}  // namespace

template <typename T>
bool ParseArg(char ***argv, const std::string &name, T *ptr,
              std::function<void(std::string)> onError) {
    std::string arg = **argv;

    // Strip either one or two leading dashes.
    if (arg[1] == '-')
        arg = arg.substr(2);
    else
        arg = arg.substr(1);

    if (matchPrefix(normalize(arg), normalize(name + '='))) {
        // --arg=value
        *argv += 1;
        std::string value = arg.substr(name.size() + 1);
        if (!initArg(value, ptr)) {
            onError(StringPrintf("invalid value \"%s\" for %s argument",
                                 value.c_str(), name.c_str()));
            return false;
        }
        return true;
    } else if (normalize(arg) == normalize(name)) {
        // --arg <value>, except for bool arguments, which are set to true
        // without expecting another argument.
        *argv += 1;
        if (enable(ptr))
            return true;

        if (**argv == nullptr) {
            onError(StringPrintf("missing value after %s argument", **argv));
            return false;
        }
        initArg(**argv, ptr);
        *argv += 1;
        return true;
    } else
        return false;
}

}  // namespace pbrt

#endif  // PBRT_UTIL_ARGS_H
