// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#include <pbrt/util/check.h>

#include <string.h>
#include <cstdlib>
#include <iostream>

#ifdef PBRT_IS_OSX
#include <cxxabi.h>
#include <execinfo.h>
#endif
#ifdef PBRT_IS_LINUX
#include <cxxabi.h>
#include <execinfo.h>
#endif

namespace pbrt {

void PrintStackTrace() {
#if defined(PBRT_IS_OSX)
    void* callstack[32];
    int frames = backtrace(callstack, PBRT_ARRAYSIZE(callstack));
    char** strs = backtrace_symbols(callstack, frames);
    for (int i = 0; i < frames; ++i) {
        // https://www.variadic.xyz/2013/04/14/generating-stack-trace-on-os-x/
        char functionSymbol[1024] = {};
        char moduleName[1024] = {};
        int offset = 0;
        char addr[48] = {};

        // split the string, take out chunks out of stack trace
        // we are primarily interested in module, function and address
        sscanf(strs[i], "%*s %s %s %s %*s %d", moduleName, addr, functionSymbol, &offset);

        if (strcmp(moduleName, "pbrt") != 0)
            // We've past the pbrt stack frames
            break;

        int validCppName = 0;
        char* functionName = abi::__cxa_demangle(functionSymbol, NULL, 0, &validCppName);

        char stackFrame[4096] = {};
        if (validCppName == 0)
            fprintf(stderr, "(%s)\t0x%s — %s + %d\n", moduleName, addr, functionName,
                    offset);
        else
            fprintf(stderr, "(%s)\t0x%s — %s + %d\n", moduleName, addr, functionSymbol,
                    offset);
    }
    free(strs);
#elif defined(PBRT_IS_LINUX)
    void *callstack[32];
    int frames = backtrace(callstack, PBRT_ARRAYSIZE(callstack));
    char **strs = backtrace_symbols(callstack, frames);
    for (int i = 0; i < frames; ++i) {
        char moduleName[1024] = {};
        char functionSymbol[1024] = {};
        char offset[64] = {};
        int validCppName = 0;
        char *demangledFunctionSymbol = nullptr;

        const char *ptr = strs[i];
        char *mptr = moduleName;
        while (*ptr && *ptr != '(')
            *mptr++ = *ptr++;
        *mptr = '\0';
        ++ptr;
        if (*ptr == '+') {
            strcpy(functionSymbol, "(unknown)");
            ++ptr;
        } else {
            // Copy mangled function name
            char *fptr = functionSymbol;
            while (*ptr && *ptr != '+')
                *fptr++ = *ptr++;
            *fptr = '\0';

            if (*ptr != '+') {
                fprintf(stderr, "Unable to decode frame: %s\n", strs[i]);
                continue;
            }
            ++ptr;
        }

        char *optr = offset;
        while (*ptr && *ptr != ')')
            *optr++ = *ptr++;
        *optr = '\0';

        validCppName = 0;
        demangledFunctionSymbol =
            abi::__cxa_demangle(functionSymbol, NULL, 0, &validCppName);

        if (validCppName == 0)
            fprintf(stderr, "(%-40s)\t0x%p — %s + %s\n", moduleName, callstack[i],
                    demangledFunctionSymbol, offset);
        else
            fprintf(stderr, "(%-40s)\t0x%p — %s + %s\n", moduleName, callstack[i],
                    functionSymbol, offset);
    }
    free(strs);
#else
#error "TODO: implement PrintStackTrace for target"
#endif
}

static std::vector<std::function<std::string(void)>> callbacks;

void CheckCallbackScope::Fail() {
    PrintStackTrace();

    std::string message;
    for (auto iter = callbacks.rbegin(); iter != callbacks.rend(); ++iter)
        message += (*iter)();
    fprintf(stderr, "%s\n\n", message.c_str());

#if defined(_DEBUG) && defined(_MSC_VER)
    // When debugging on windows, avoid the obnoxious dialog and make
    // it possible to continue past a LOG(FATAL) in the debugger
    __debugbreak();
#else
    abort();
#endif
}

std::vector<std::function<std::string(void)>> CheckCallbackScope::callbacks;

CheckCallbackScope::CheckCallbackScope(std::function<std::string(void)> callback) {
    callbacks.push_back(std::move(callback));
}

CheckCallbackScope::~CheckCallbackScope() {
    CHECK_GT(callbacks.size(), 0);
    callbacks.pop_back();
}

}  // namespace pbrt
