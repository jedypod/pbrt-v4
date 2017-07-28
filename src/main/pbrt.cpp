
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


// main/pbrt.cpp*
#include "pbrt.h"

#include "api.h"
#include "args.h"
#include "error.h"
#include "parser.h"
#include "parallel.h"
#include <glog/logging.h>

using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr, R"(usage: pbrt [<options>] <filename.pbrt...>
Rendering options:
  --help               Print this help text.
  --texcachemb <num>   Texture cache size in MB.
  --nthreads <num>     Use specified number of threads for rendering.
  --outfile <filename> Write the final image to the given filename.
  --quick              Automatically reduce a number of quality settings to
                       render more quickly.
  --quiet              Suppress all text output other than error messages.

Logging options:
  --logdir <dir>       Specify directory that log files should be written to.
                       Default: system temp directory (e.g. $TMPDIR or /tmp).
  --logtostderr        Print all logging messages to stderr.
  --minloglevel <num>  Log messages at or above this level (0 -> INFO,
                       1 -> WARNING, 2 -> ERROR, 3-> FATAL). Default: 0.
  --v <verbosity>      Set VLOG verbosity.

Reformatting options:
  --cat                Print a reformatted version of the input file(s) to
                       standard output. Does not render an image.
  --toply              Print a reformatted version of the input file(s) to
                       standard output and convert all triangle meshes to
                       PLY files. Does not render an image.
)");
}

// main program
int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_stderrthreshold = 1; // Warning and above.

    Options options;
    std::vector<std::string> filenames;
    // Process command-line arguments
    ++argv;
    while (*argv) {
        if ((*argv)[0] != '-') {
            filenames.push_back(*argv);
            ++argv;
            continue;
        }

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        if (ParseArg(&argv, "nthreads", &options.nThreads, onError) ||
            ParseArg(&argv, "texcachemb", &options.texCacheMB, onError) ||
            ParseArg(&argv, "texreadms", &options.texReadMinMS, onError) ||
            ParseArg(&argv, "minloglevel", &FLAGS_minloglevel, onError) ||
            ParseArg(&argv, "v", &FLAGS_v, onError) ||
            ParseArg(&argv, "outfile", &options.imageFile, onError) ||
            ParseArg(&argv, "logdir", &FLAGS_log_dir, onError) ||
            ParseArg(&argv, "quick", &options.quickRender, onError) ||
            ParseArg(&argv, "quiet", &options.quiet, onError) ||
            ParseArg(&argv, "cat", &options.cat, onError) ||
            ParseArg(&argv, "toply", &options.toPly, onError) ||
            ParseArg(&argv, "logtostderr", &FLAGS_logtostderr, onError)) {
            // success
        } else if (!strcmp(*argv, "--help") || !strcmp(*argv, "-help") ||
                   !strcmp(*argv, "-h")) {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *argv));
            return 1;
        }
    }

    // Print welcome banner
    if (!options.quiet && !options.cat && !options.toPly) {
        printf("pbrt version 3 (built %s at %s)\n", __DATE__, __TIME__);
#ifndef NDEBUG
        LOG(INFO) << "Running debug build";
        printf("*** DEBUG BUILD ***\n");
#endif // !NDEBUG
        printf(
            "Copyright (c)1998-2016 Matt Pharr, Greg Humphreys, and Wenzel "
            "Jakob.\n");
        printf(
            "The source code to pbrt (but *not* the book contents) is covered "
            "by the BSD License.\n");
        printf("See the file LICENSE.txt for the conditions of the license.\n");
        fflush(stdout);
    }
    pbrtInit(options);
    // Process scene description
    if (filenames.empty()) {
        // Parse scene from standard input
        ParseFile("-");
    } else {
        // Parse scene from input files
        for (const std::string &f : filenames)
            if (!ParseFile(f))
                Error("Couldn't open scene file \"%s\"", f.c_str());
    }
    pbrtCleanup();
    return 0;
}
