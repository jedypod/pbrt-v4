
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

#ifndef PBRT_UTIL_PROGRESSREPORTER_H
#define PBRT_UTIL_PROGRESSREPORTER_H

// core/progressreporter.h*
#include <pbrt/pbrt.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#ifdef PBRT_HAVE_OPTIX
  #include <cuda_runtime.h>
  #include <vector>
  #include <pbrt/util/pstd.h>
#endif

namespace pbrt {

// ProgressReporter Declarations
class Timer {
public:
    Timer() {
        start = clock::now();
    }
    double ElapsedSeconds() const {
        clock::time_point now = clock::now();
        int64_t elapseduS =
            std::chrono::duration_cast<std::chrono::microseconds>(now - start)
                .count();
        return elapseduS / 1000000.;
    }

    std::string ToString() const;

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start;
};

class ProgressReporter {
  public:
    // ProgressReporter Public Methods
    ProgressReporter()
        : quiet(true) {}
    ProgressReporter(int64_t totalWork, const std::string &title, bool gpu = false);
    ~ProgressReporter();

    void Update(int64_t num = 1) {
#ifdef PBRT_HAVE_OPTIX
        if (gpuEvents.size() > 0) {
            CHECK_LE(gpuEventsLaunchedOffset + num, gpuEvents.size());
            while (num-- > 0) {
                CHECK_EQ(cudaEventRecord(gpuEvents[gpuEventsLaunchedOffset]),
                         cudaSuccess);
                ++gpuEventsLaunchedOffset;
            }
            return;
        }
#endif
        if (num == 0 || quiet) return;
        workDone += num;
    }
    double ElapsedSeconds() const {
        return timer.ElapsedSeconds();
    }
    void Done();

    std::string ToString() const;

  private:
    // ProgressReporter Private Methods
    void launchThread();
    void printBar();

    // ProgressReporter Private Data
    int64_t totalWork;
    std::string title;
    bool quiet;
    Timer timer;
    std::atomic<int64_t> workDone;
    std::atomic<bool> exitThread;
    std::thread updateThread;

#ifdef PBRT_HAVE_OPTIX
    std::vector<cudaEvent_t> gpuEvents;
    std::atomic<int> gpuEventsLaunchedOffset;
    int gpuEventsFinishedOffset;
#endif
};

}  // namespace pbrt

#endif  // PBRT_UTIL_PROGRESSREPORTER_H
