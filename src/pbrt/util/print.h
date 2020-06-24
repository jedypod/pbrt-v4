
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

#ifndef PBRT_UTIL_PRINT_H
#define PBRT_UTIL_PRINT_H

#include <pbrt/pbrt.h>

#include <string>

// Hack: make util/log.h happy
namespace pbrt {
template <typename... Args>
inline std::string StringPrintf(const char *fmt, Args&&... args);
}

#include <pbrt/util/log.h>

#include <ctype.h>
#include <inttypes.h>
#include <stdio.h>
#include <memory>
#include <ostream>
#include <sstream>
#include <type_traits>

#ifdef __GNUG__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-security"
#endif  // __GNUG__

namespace pbrt {

// helpers, fwiw
template <typename T>
static auto operator<<(std::ostream &os, const T &v)
    -> decltype(v.ToString(), os) {
    return os << v.ToString();
}
template <typename T>
static auto operator<<(std::ostream &os, const T &v)
    -> decltype(ToString(v), os) {
    return os << ToString(v);
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::shared_ptr<T> &p) {
    if (p) return os << p->ToString();
    else return os << "(nullptr)";
}

template <typename T>
inline std::ostream &operator<<(std::ostream &os, const std::unique_ptr<T> &p) {
    if (p) return os << p->ToString();
    else return os << "(nullptr)";
}

namespace detail {

std::string FloatToString(float v);
std::string DoubleToString(double v);

template <typename T> struct IntegerFormatTrait {
    static constexpr const char *fmt() { return "ERROR"; }
};
template <> struct IntegerFormatTrait<int> {
    static constexpr const char *fmt() { return "d"; }
};
template <> struct IntegerFormatTrait<unsigned int> {
    static constexpr const char *fmt() { return "u"; }
};
template <> struct IntegerFormatTrait<short> {
    static constexpr const char *fmt() { return "d"; }
};
template <> struct IntegerFormatTrait<unsigned short> {
    static constexpr const char *fmt() { return "u"; }
};
template <> struct IntegerFormatTrait<unsigned long> {
    static constexpr const char *fmt() { return "lu"; }
};
template <> struct IntegerFormatTrait<int64_t> {
    static constexpr const char *fmt() { return PRId64; }
};
#ifdef PBRT_IS_OSX
template <> struct IntegerFormatTrait<uint64_t> {
    static constexpr const char *fmt() { return PRIu64; }
};
#endif

template <typename T>
using HasSize =
    std::is_integral<typename std::decay_t<decltype(std::declval<T&>().size())>>;

template <typename T>
using HasData =
    std::is_pointer<typename std::decay_t<decltype(std::declval<T&>().data())>>;

// Don't use size()/data()-based operator<< for std::string...
inline std::ostream &operator<<(std::ostream &os, const std::string &str) {
    return std::operator<<(os, str);
}

template <typename T>
inline std::enable_if_t<HasSize<T>::value && HasData<T>::value, std::ostream &>
operator<<(std::ostream &os, const T &v) {
    os << "[ ";
    auto ptr = v.data();
    for (size_t i = 0; i < v.size(); ++i) {
        os << ptr[i];
        if (i < v.size() - 1)
            os << ", ";
    }
    return os << " ]";
}

template <typename T> inline std::string BoolToString(const T &) {
    LOG_FATAL("error");
    return "";
}

template <typename T> inline std::string FloatToString(const T &) {
    LOG_FATAL("error");
    return "";
}

template <typename T> inline std::string DoubleToString(const T &) {
    LOG_FATAL("error");
    return "";
}

template <> inline std::string BoolToString(const bool &v) {
    return v ? "true" : "false";
}

template <> inline std::string
FloatToString(const float &v) {
    return detail::FloatToString(v);
}

template <> inline std::string
DoubleToString(const double &v) {
    return detail::DoubleToString(v);
}

// base case
void stringPrintfRecursive(std::string *s, const char *fmt);

// 1. Copy from fmt to *s, up to the next formatting directive.
// 2. Advance fmt past the next formatting directive and return the
//    formatting directive as a string.
std::string copyToFormatString(const char **fmt_ptr, std::string *s);

template <typename T>
inline typename std::enable_if_t<!std::is_class<typename std::decay_t<T>>::value,
                                 std::string>
formatOne(const char *fmt, T&& v) {
    // Figure out how much space we need to allocate; add an extra
    // character for the '\0'.
    size_t size = snprintf(nullptr, 0, fmt, v) + 1;
    std::string str;
    str.resize(size);
    snprintf(&str[0], size, fmt, v);
    str.pop_back();  // remove trailing NUL
    return str;
}

template <typename T>
inline typename std::enable_if_t<std::is_class<typename std::decay_t<T>>::value,
                                 std::string>
formatOne(const char *fmt, T&& v) {
    LOG_FATAL("MEH");
    return "ERROR";
}

template <typename T, typename... Args>
inline void stringPrintfRecursive(std::string *s, const char *fmt, T &&v,
                                  Args&&... args);

template <typename T, typename... Args>
inline void stringPrintfRecursiveWithPrecision(std::string *s, const char *fmt,
                                               const std::string &nextFmt,
                                               T &&v, Args&&... args) {
    LOG_FATAL("MEH");
}

template <typename T, typename... Args>
inline typename std::enable_if_t<std::is_class<typename std::decay_t<T>>::value,
                                 void>
stringPrintfRecursiveWithPrecision(std::string *s, const char *fmt,
                                   const std::string &nextFmt,
                                   int precision, T &&v, Args&&... args) {
    LOG_FATAL("MEH");
}

template <typename T, typename... Args>
inline typename std::enable_if_t<!std::is_class<typename std::decay_t<T>>::value,
                                 void>
stringPrintfRecursiveWithPrecision(std::string *s, const char *fmt,
                                   const std::string &nextFmt,
                                   int precision, T &&v, Args&&... args) {
    size_t size = snprintf(nullptr, 0, nextFmt.c_str(), precision, v) + 1;
    std::string str;
    str.resize(size);
    snprintf(&str[0], size, nextFmt.c_str(), precision, v);
    str.pop_back();  // remove trailing NUL
    *s += str;

    stringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
}

// General-purpose version of stringPrintfRecursive; add the formatted
// output for a single StringPrintf() argument to the final result string
// in *s.
template <typename T, typename... Args>
inline void stringPrintfRecursive(std::string *s, const char *fmt, T &&v,
                                  Args&&... args) {
    std::string nextFmt = copyToFormatString(&fmt, s);
    bool precisionViaArg = nextFmt.find('*') != std::string::npos;

    if (precisionViaArg) {
        if (!std::is_integral<std::decay_t<T>>::value)
            LOG_FATAL("Non integral type provided for %* format");
        stringPrintfRecursiveWithPrecision(s, fmt, nextFmt, v, std::forward<Args>(args)...);
        return;
    }

    bool isSFmt = nextFmt.find('s') != std::string::npos;
    bool isDFmt = nextFmt.find('d') != std::string::npos;
    if ((nextFmt == "%f" || nextFmt == "%s") && std::is_same<std::decay_t<T>, float>::value)
        *s += FloatToString(v);
    else if ((nextFmt == "%f" || nextFmt == "%s") && std::is_same<std::decay_t<T>, double>::value)
        // FIXME: do this and above with an overload
        *s += DoubleToString(v);
    else if (isSFmt && std::is_same<std::decay_t<T>, bool>::value) // FIXME: %-10s with booln
        *s += BoolToString(v);
    else if (isDFmt) {
        if (detail::IntegerFormatTrait<std::decay_t<T>>::fmt() == "ERROR")
            LOG_FATAL("Non-integral type passed to %d format");
        nextFmt.replace(nextFmt.find('d'), 1, detail::IntegerFormatTrait<std::decay_t<T>>::fmt());
        *s += formatOne(nextFmt.c_str(), std::forward<T>(v));
    } else if (isSFmt) {
        std::stringstream ss;
        ss << v;
        *s += formatOne(nextFmt.c_str(), ss.str().c_str());
    } else
        *s += formatOne(nextFmt.c_str(), std::forward<T>(v));
    stringPrintfRecursive(s, fmt, std::forward<Args>(args)...);
}

} // namespace detail

// StringPrintf() is a replacement for sprintf() (and the like) that
// returns the result as a std::string. This gives convenience/control
// of printf-style formatting in a more C++-ish way.
//
// Floating-point values with the formatting string "%f" are handled
// specially so that enough digits are always printed so that the original
// float/double can be reconstituted exactly from the printed digits.
template <typename... Args>
inline std::string StringPrintf(const char *fmt, Args&&... args) {
    std::string ret;
    detail::stringPrintfRecursive(&ret, fmt, std::forward<Args>(args)...);
    return ret;
}

template <typename... Args>
void Printf(const char *fmt, Args&&... args) {
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    fputs(s.c_str(), stdout);
}

#ifdef __GNUG__
#pragma GCC diagnostic pop
#endif  // __GNUG__

// https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
inline std::string Red(const std::string &s) {
    const char *red = "\033[1m\033[31m";  // bold red
    const char *reset = "\033[0m";
    return std::string(red) + s + std::string(reset);
}

inline std::string Yellow(const std::string &s) {
    // https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    const char *yellow = "\033[1m\033[38;5;100m";
    const char *reset = "\033[0m";
    return std::string(yellow) + s + std::string(reset);
}

inline std::string Green(const std::string &s) {
    // https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit
    const char *green = "\033[1m\033[38;5;22m";
    const char *reset = "\033[0m";
    return std::string(green) + s + std::string(reset);
}

}  // namespace pbrt

#endif  // PBRT_UTIL_PRINT_H
