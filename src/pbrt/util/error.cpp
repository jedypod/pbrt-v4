
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

// util/error.cpp*
#include <pbrt/util/error.h>

#include <pbrt/options.h>
#include <pbrt/util/check.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>

#include <cstdio>
#include <string>
#include <mutex>

namespace pbrt {

std::string FileLoc::ToString() const {
    return StringPrintf("%s:%d:%d", std::string(filename.data(), filename.size()),
                        line, column);
}

static void processError(const char *errorType, const FileLoc *loc, const char *message) {
    // Build up an entire formatted error string and print it all at once;
    // this way, if multiple threads are printing messages at once, they
    // don't get jumbled up...
    std::string errorString = Red(errorType);

    if (loc)
        errorString += ": " + loc->ToString();

    errorString += ": ";
    errorString += message;

    // Print the error message (but not more than one time).
    static std::string lastError;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (errorString != lastError) {
        fprintf(stderr, "%s\n", errorString.c_str());
        LOG_VERBOSE("%s", errorString);
        lastError = errorString;
    }
}

void Warning(const FileLoc *loc, const char *message) {
    if (PbrtOptions.quiet) return;
    processError("Warning", loc, message);
}

void Error(const FileLoc *loc, const char *message) {
    if (PbrtOptions.quiet) return;
    processError("Error", loc, message);
}

void ErrorExit(const FileLoc *loc, const char *message) {
    processError("Error", loc, message);
    // This is annoying, but gives a cleaner exit.
    ParallelCleanup();
    exit(1);
}

}  // namespace pbrt
