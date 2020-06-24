
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


#include <pbrt/util/check.h>

#include <cstdlib>
#include <iostream>

#ifdef PBRT_IS_OSX
#include <execinfo.h>
#include <cxxabi.h>
#endif

namespace pbrt {

static std::vector<std::function<std::string(void)>> callbacks;

void CheckCallbackScope::Fail() {
    // Print the stack trace
#ifdef PBRT_IS_OSX
    void *callstack[32];
    int frames = backtrace(callstack, PBRT_ARRAYSIZE(callstack));
    char** strs = backtrace_symbols(callstack, frames);
    for (int i = 0; i < frames; ++i) {
        // https://www.variadic.xyz/2013/04/14/generating-stack-trace-on-os-x/
        char functionSymbol[1024] = {};
        char moduleName[1024] = {};
        int  offset = 0;
        char addr[48] = {};

        // split the string, take out chunks out of stack trace
        // we are primarily interested in module, function and address
        sscanf(strs[i], "%*s %s %s %s %*s %d",
               moduleName, addr, functionSymbol, &offset);

        if (strcmp(moduleName, "pbrt") != 0)
            // We've past the pbrt stack frames
            break;

        int validCppName = 0;
        char* functionName = abi::__cxa_demangle(functionSymbol,
                                                 NULL, 0, &validCppName);

        char stackFrame[4096] = {};
        if (validCppName == 0)
            fprintf(stderr, "(%s)\t0x%s — %s + %d\n", moduleName, addr, functionName, offset);
        else
            fprintf(stderr, "(%s)\t0x%s — %s + %d\n", moduleName, addr, functionSymbol, offset);
    }
    free(strs);
#else
    fprintf(stderr, "TODO: print call stack in CheckCallbackScope::Fail\n");
#endif
    fprintf(stderr, "\n");

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
