
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

#ifndef PBRT_UTIL_CHECK_H
#define PBRT_UTIL_CHECK_H

#include <pbrt/pbrt.h>

#include <pbrt/util/log.h>
#include <pbrt/util/stats.h>

#include <functional>
#include <string>
#include <vector>

namespace pbrt {

#ifdef __CUDA_ARCH__

#define CHECK(x) assert(x)
#define CHECK_IMPL(a, b, op) assert((a) op (b))

#else

#define CHECK(x)                                \
    !(!(x) && (LOG_FATAL("Check failed: %s", #x), true))

#define CHECK_IMPL(a, b, op)                    \
    do {                                        \
        auto va = a;                            \
        auto vb = b;                            \
        if (!(va op vb)) { \
            LOG_FATAL("Check failed: %s " #op " %s with %s = %s, %s = %s", #a, #b, #a, va, #b, vb); \
        } \
    } while (false) /* swallow semicolon */

#endif // CUDA_ARCH

#define CHECK_EQ(a, b) CHECK_IMPL(a, b, ==)
#define CHECK_NE(a, b) CHECK_IMPL(a, b, !=)
#define CHECK_GT(a, b) CHECK_IMPL(a, b, >)
#define CHECK_GE(a, b) CHECK_IMPL(a, b, >=)
#define CHECK_LT(a, b) CHECK_IMPL(a, b, <)
#define CHECK_LE(a, b) CHECK_IMPL(a, b, <=)

#ifndef NDEBUG

#define DCHECK(x) CHECK(x)
#define DCHECK_EQ(a, b) CHECK_EQ(a, b)
#define DCHECK_NE(a, b) CHECK_NE(a, b)
#define DCHECK_GT(a, b) CHECK_GT(a, b)
#define DCHECK_GE(a, b) CHECK_GE(a, b)
#define DCHECK_LT(a, b) CHECK_LT(a, b)
#define DCHECK_LE(a, b) CHECK_LE(a, b)

#else


#define DCHECK(x)
#define DCHECK_EQ(a, b)
#define DCHECK_NE(a, b)
#define DCHECK_GT(a, b)
#define DCHECK_GE(a, b)
#define DCHECK_LT(a, b)
#define DCHECK_LE(a, b)

#endif

#define CHECK_RARE_TO_STRING(x) #x
#define CHECK_RARE_EXPAND_AND_TO_STRING(x) CHECK_RARE_TO_STRING(x)

#ifdef __CUDA_ARCH__

#define CHECK_RARE(freq, condition)
#define DCHECK_RARE(freq, condition)

#else

#define CHECK_RARE(freq, condition)                                     \
    static_assert(std::is_floating_point<decltype(freq)>::value,        \
                  "Expected floating-point frequency as first argument to CHECK_RARE"); \
    static_assert(std::is_integral<decltype(condition)>::value,         \
                  "Expected Boolean condition as second argument to CHECK_RARE"); \
    do {                                                                \
        static thread_local int64_t numTrue, total;                     \
        static StatRegisterer reg([](StatsAccumulator &accum) {         \
                accum.ReportRareCheck(__FILE__ " "                      \
                    CHECK_RARE_EXPAND_AND_TO_STRING(__LINE__) ": CHECK_RARE failed: " #condition, \
                                  freq, numTrue, total);               \
                numTrue = total = 0;                                   \
            });                                                        \
        ++total;                                                       \
        if (condition) ++numTrue;                                      \
    } while(0)

#ifdef NDEBUG
#define DCHECK_RARE(freq, condition)
#else
#define DCHECK_RARE(freq, condition) CHECK_RARE(freq, condition)
#endif  // NDEBUG

#endif // __CUDA_ARCH__

class CheckCallbackScope {
 public:
    CheckCallbackScope(std::function<std::string(void)> callback);
    ~CheckCallbackScope();

    CheckCallbackScope(const CheckCallbackScope &) = delete;
    CheckCallbackScope &operator=(const CheckCallbackScope &) = delete;

    static void Fail();

 private:
    static std::vector<std::function<std::string(void)>> callbacks;
};

}  // namespace pbrt

#endif  // PBRT_UTIL_CHECKSCOPE_H
