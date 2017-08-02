
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


// core/parallel.cpp*
#include "parallel.h"

#include "memory.h"
#include "stats.h"

#include <tbb/tbb.h>

#include <list>
#include <thread>
#include <vector>

namespace pbrt {

static std::unique_ptr<tbb::task_scheduler_init> taskScheduler;

void Barrier::Wait() {
    std::unique_lock<std::mutex> lock(mutex);
    CHECK_GT(count, 0);
    if (--count == 0)
        // This is the last thread to reach the barrier; wake up all of the
        // other ones before exiting.
        cv.notify_all();
    else
        // Otherwise there are still threads that haven't reached it. Give
        // up the lock and wait to be notified.
        cv.wait(lock, [this] { return count == 0; });
}

// Parallel Definitions
void ParallelFor(int64_t start, int64_t end, int chunkSize,
                 std::function<void(int64_t)> func) {
    tbb::parallel_for(tbb::blocked_range<int64_t>(start, end, chunkSize),
                      [&func](const tbb::blocked_range<int64_t> &r) {
                          for (int64_t i = r.begin(); i < r.end(); ++i)
                              func(i);
                      });
}

static int AvailableCores() {
    return std::max(1u, std::thread::hardware_concurrency());
}

PBRT_THREAD_LOCAL int ThreadIndex = -1;

int MaxThreadIndex() {
    return PbrtOptions.nThreads == 0 ? AvailableCores() : PbrtOptions.nThreads;
}

void ParallelFor2D(const Bounds2i &extent, int chunkSize,
                   std::function<void(Bounds2i)> func) {
    CHECK_GE(extent.Area(), 0);  // or just return immediately?

    tbb::parallel_for(tbb::blocked_range2d<int>(extent.pMin.y, extent.pMax.y, chunkSize,
                                                extent.pMin.x, extent.pMax.x, chunkSize),
                      [&func](const tbb::blocked_range2d<int> &r) {
                          func(Bounds2i({r.cols().begin(), r.rows().begin()},
                                        {r.cols().end(), r.rows().end()}));
                      });
}

void ParallelInit() {
    int nThreads = MaxThreadIndex();

    // Create a barrier so that we can be sure all worker threads get past
    // their call to ProfilerWorkerThreadInit() before we return from this
    // function.  In turn, we can be sure that the profiling system isn't
    // started until after all worker threads have done that.
    Barrier barrier(nThreads);

    CHECK(!taskScheduler);
    taskScheduler = std::make_unique<tbb::task_scheduler_init>(nThreads);

    ParallelFor(0, nThreads, [&barrier,nThreads](int tIndex) {
        if (ThreadIndex != -1) {
            CHECK(ThreadIndex >= 0 && ThreadIndex < nThreads);
        } else
            ThreadIndex = tIndex;

        LOG(INFO) << "Started execution in worker thread " << ThreadIndex;

        // Give the profiler a chance to do per-thread initialization for
        // the worker thread before the profiling system actually stops
        // running.
        ProfilerWorkerThreadInit();

        // Waiting here ensures that each worker thread will get a single
        // loop iteration (and thus, that each worker thread will run this
        // loop), ensuring that all of the per-thread setup runs before
        // this ParallelFor returns.
        barrier.Wait();
    });
}

void MergeWorkerThreadStats() {
    int nThreads = MaxThreadIndex();

    // Same trick as in ParallelInit() to make sure all threads run exactly
    // once here...
    Barrier barrier(nThreads);

    ParallelFor(0, nThreads, [&barrier](int) {
        ReportThreadStats();
        barrier.Wait();
    });
}

void ParallelCleanup() {
    taskScheduler = nullptr;
}

}  // namespace pbrt
