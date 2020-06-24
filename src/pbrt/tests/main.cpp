
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

#include <pbrt/pbrt.h>

#include <pbrt/api.h>
#include <pbrt/util/args.h>
#include <pbrt/options.h>
#include <pbrt/util/print.h>
#include <pbrt/util/error.h>

#include <gtest/gtest.h>
#include <string>

using namespace pbrt;

void usage(const std::string &msg = "") {
    if (!msg.empty())
        fprintf(stderr, "pbrt_test: %s\n\n", msg.c_str());

    fprintf(stderr, R"(

General pbrt_test arguments:
  --log-level <level>         Log messages at or above this level, where <level>
                              is "verbose", "error", or "fatal". Default: "error".
  --nthreads <num>            Use specified number of threads for rendering.
  --vlog-level <n>            Set VLOG verbosity. (Default: 0, disabled.)
)");

    pbrtCleanup();

    exit(msg.empty() ? 0 : 1);
}

int main(int argc, char **argv) {
    Options opt;
    opt.quiet = true;
    LogConfig logConfig;
    std::string logLevel = "error";

    pbrtInit(opt);

    testing::InitGoogleTest(&argc, argv);

    char **origArgv = argv;
    // Process command-line arguments
    ++argv;
    while (*argv != nullptr) {
        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        if (ParseArg(&argv, "log-level", &logLevel, onError) ||
            ParseArg(&argv, "nthreads", &opt.nThreads, onError) ||
            ParseArg(&argv, "vlog-level", &logConfig.vlogLevel, onError)) {
            // success
        } else if ((strcmp(*argv, "--help") == 0) || (strcmp(*argv, "-h") == 0)) {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *argv));
            return 1;
        }
    }

    logConfig.level = LogLevelFromString(logLevel);
    if (logConfig.level == LogLevel::Invalid)
        ErrorExit("%s: --log-level unknown", logLevel);

    InitLogging(logConfig, argv[0]);

    int ret = RUN_ALL_TESTS();

    pbrtCleanup();

    return ret;
}
