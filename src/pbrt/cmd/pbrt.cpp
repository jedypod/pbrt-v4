
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
#include <pbrt/pbrt.h>


#include <pbrt/cpurender.h>
#include <pbrt/genscene.h>
#include <pbrt/gpu.h>
#include <pbrt/options.h>
#include <pbrt/parser.h>
#include <pbrt/util/args.h>
#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/log.h>
#include <pbrt/util/parallel.h>
#include <pbrt/util/print.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/string.h>

using namespace pbrt;

static void usage(const std::string &msg = {}) {
    if (!msg.empty())
        fprintf(stderr, "pbrt: %s\n\n", msg.c_str());

    fprintf(stderr, R"(usage: pbrt [<options>] <filename.pbrt...>
Rendering options:
  --cropwindow <x0,x1,y0,y1>  Specify an image crop window w.r.t. [0,1]^2
  --debugstart <values>       Inform the Integrator where to start rendering for
                              faster debugging. (<values> are Integrator-specific
                              and come from error message text.)
  --disable-pixel-jitter      Always sample pixels at their centers.
  --disable-wavelength-jitter Always sample the same %d wavelengths of light.
  --force-diffuse             Convert all materials to be diffuse.
)"
#ifdef PBRT_HAVE_OPTIX
"  --gpu                       Use the GPU for rendering. (Default: disabled)"
#endif
R"(
  --help                      Print this help text.
  --mse-reference-image       Filename for reference image to use for MSE computation.
  --mse-reference-out         File to write MSE error vs spp results.
  --nthreads <num>            Use specified number of threads for rendering.
  --outfile <filename>        Write the final image to the given filename.
  --pixel <x,y>               Render just the specified pixel.
  --pixelbounds <x0,x1,y0,y1> Specify an image crop window w.r.t. pixel coordinates.
  --pixelstats                Record per-pixel statistics and write additional images
                              with their values.
  --quick                     Automatically reduce a number of quality settings
                              to render more quickly.
  --quiet                     Suppress all text output other than error messages.
  --seed <n>                  Set random number generator seed. Default: 0.
  --spp <n>                   Override number of pixel samples specified in scene
                              description file.

Logging options:
  --log-level <level>         Log messages at or above this level, where <level>
                              is "verbose", "error", or "fatal". Default: "error".
  --profile                   Profile the main phases of execution and print a
                              report at the end of rendering. Default: disabled.
  --vlog-level <n>            Set VLOG verbosity. (Default: 0, disabled.)

Reformatting options:
  --format                    Print a reformatted version of the input file(s) to
                              standard output. Does not render an image.
  --toply                     Print a reformatted version of the input file(s) to
                              standard output and convert all triangle meshes to
                              PLY files. Does not render an image.
  --upgrade                   Upgrade a pbrt-v3 file to pbrt-v4's format.
)", NSpectrumSamples);
    exit(msg.empty() ? 0 : 1);
}

// main program
int main(int argc, char *argv[]) {
    Options options;
    LogConfig logConfig;
    std::string logLevel = "error";
    std::vector<std::string> filenames;
    bool format = false, toPly = false, gpu = false;

    // Process command-line arguments
    ++argv;
    while (*argv != nullptr) {
        if ((*argv)[0] != '-') {
            filenames.push_back(*argv);
            ++argv;
            continue;
        }

        auto onError = [](const std::string &err) {
            usage(err);
            exit(1);
        };

        std::string cropWindow, pixelBounds, pixel;
        if (ParseArg(&argv, "cropwindow", &cropWindow, onError)) {
            pstd::optional<std::vector<Float>> c =
                SplitStringToFloats(cropWindow, ',');
            if (!c || c->size() != 4) {
                usage("Didn't find four values after --cropwindow");
                return 1;
            }
            options.cropWindow = Bounds2f(Point2f((*c)[0], (*c)[2]),
                                          Point2f((*c)[1], (*c)[3]));
        } else if (ParseArg(&argv, "pixel", &pixel, onError)) {
            pstd::optional<std::vector<int>> p = SplitStringToInts(pixel, ',');
            if (!p || p->size() != 2) {
                usage("Didn't find two values after --pixel");
                return 1;
            }
            options.pixelBounds = Bounds2i(Point2i((*p)[0], (*p)[1]),
                                           Point2i((*p)[0]+1, (*p)[1]+1));
        } else if (ParseArg(&argv, "pixelbounds", &pixelBounds, onError)) {
            pstd::optional<std::vector<int>> p = SplitStringToInts(pixelBounds, ',');
            if (!p || p->size() != 4) {
                usage("Didn't find four integer values after --pixelbounds");
                return 1;
            }
            options.pixelBounds = Bounds2i(Point2i((*p)[0], (*p)[2]),
                                           Point2i((*p)[1], (*p)[3]));
        } else if (
#ifdef PBRT_HAVE_OPTIX
                   ParseArg(&argv, "gpu", &gpu, onError) ||
#endif
                   ParseArg(&argv, "debugstart", &options.debugStart, onError) ||
                   ParseArg(&argv, "disable-pixel-jitter", &options.disablePixelJitter, onError) ||
                   ParseArg(&argv, "disable-wavelength-jitter", &options.disableWavelengthJitter, onError) ||
                   ParseArg(&argv, "force-diffuse", &options.forceDiffuse, onError) ||
                   ParseArg(&argv, "format", &format, onError) ||
                   ParseArg(&argv, "log-level", &logLevel, onError) ||
                   ParseArg(&argv, "mse-reference-image", &options.mseReferenceImage, onError) ||
                   ParseArg(&argv, "mse-reference-out", &options.mseReferenceOutput, onError) ||
                   ParseArg(&argv, "nthreads", &options.nThreads, onError) ||
                   ParseArg(&argv, "outfile", &options.imageFile, onError) ||
                   ParseArg(&argv, "pixelstats", &options.recordPixelStatistics, onError) ||
                   ParseArg(&argv, "profile", &options.profile, onError) ||
                   ParseArg(&argv, "quick", &options.quickRender, onError) ||
                   ParseArg(&argv, "quiet", &options.quiet, onError) ||
                   ParseArg(&argv, "seed", &options.seed, onError) ||
                   ParseArg(&argv, "spp", &options.pixelSamples, onError) ||
                   ParseArg(&argv, "toply", &toPly, onError) ||
                   ParseArg(&argv, "upgrade", &options.upgrade, onError) ||
                   ParseArg(&argv, "vlog-level", &logConfig.vlogLevel, onError)) {
            // success
        } else if ((strcmp(*argv, "--help") == 0) || (strcmp(*argv, "-help") == 0) ||
                   (strcmp(*argv, "-h") == 0)) {
            usage();
            return 0;
        } else {
            usage(StringPrintf("argument \"%s\" unknown", *argv));
            return 1;
        }
    }

    // Print welcome banner
    if (!options.quiet && !format && !toPly && !options.upgrade) {
        printf("pbrt version 4 (built %s at %s)\n", __DATE__, __TIME__);
        if (sizeof(void *) == 4)
            printf("*** WARNING: This is a 32-bit build of pbrt. It will crash "
                   "if used to render highly complex scenes. ***\n");
#ifndef NDEBUG
        LOG_VERBOSE("Running debug build");
        printf("*** DEBUG BUILD ***\n");
#endif // !NDEBUG
        printf(
            "Copyright (c)1998-2018 Matt Pharr, Greg Humphreys, and Wenzel "
            "Jakob.\n");
        printf(
            "The source code to pbrt (but *not* the book contents) is covered "
            "by the BSD License.\n");
        printf("See the file LICENSE.txt for the conditions of the license.\n");
        fflush(stdout);
    }

    if (options.mseReferenceImage && ! options.mseReferenceOutput)
        ErrorExit("Must provide MSE reference output filename via --mse-reference-out");
    if (options.mseReferenceOutput && ! options.mseReferenceImage)
        ErrorExit("Must provide MSE reference image via --mse-reference-image");

    options.renderFunction = CPURender;
#ifdef PBRT_HAVE_OPTIX
    if (gpu)
        options.renderFunction = GPURender;
#endif

    logConfig.level = LogLevelFromString(logLevel);

    InitPBRT(options, logConfig);

    std::unique_ptr<GeneralSceneBase> scene;
    if (format || toPly || options.upgrade)
        scene = std::make_unique<FormattingScene>(toPly, options.upgrade);
    else {
#ifdef PBRT_HAVE_OPTIX
        if (gpu) {
            static CUDAMemoryResource memoryResource;
            scene = std::make_unique<GeneralScene>(&memoryResource);
        } else
#endif
            scene = std::make_unique<GeneralScene>(pstd::pmr::get_default_resource());
    }

    // Process scene description
    if (filenames.empty()) {
        // Parse scene from standard input
        ParseFile(scene.get(), "-");
    } else {
        // Parse scene from input files
        for (const std::string &f : filenames)
            ParseFile(scene.get(), f);
    }

    CleanupPBRT();
    return 0;
}
