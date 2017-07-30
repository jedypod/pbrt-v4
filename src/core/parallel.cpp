
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
#include <async++.h>

#include <array>
#include <list>
#include <thread>
#include <vector>

namespace pbrt {

// Parallel Local Definitions
static std::vector<std::thread> threads;
static bool shutdownThreads = false;
class ParallelForLoop;
static ParallelForLoop *workList = nullptr;
static std::mutex workListMutex;

// Bookkeeping variables to help with the implementation of
// MergeWorkerThreadStats().
static std::atomic<bool> reportWorkerStats{false};
// Number of workers that still need to report their stats.
static std::atomic<int> reporterCount;
// After kicking the workers to report their stats, the main thread waits
// on this condition variable until they've all done so.
static std::condition_variable reportDoneCondition;
static std::mutex reportDoneMutex;

class ParallelForLoop {
  public:
    // ParallelForLoop Public Methods
    ParallelForLoop(std::function<void(int64_t)> func1D, int64_t maxIndex,
                    int chunkSize, uint64_t profilerState)
        : func1D(std::move(func1D)),
          maxIndex(maxIndex),
          chunkSize(chunkSize),
          indexIncrement(chunkSize * threads.size()),
          profilerState(profilerState) {
        for (size_t i = 0; i < threads.size(); ++i)
            threadNextIndex[i].index.store(i * chunkSize, std::memory_order_release);
    }
    ParallelForLoop(const std::function<void(Point2i)> &f, const Point2i &count,
                    uint64_t profilerState)
        : func2D(f),
          maxIndex(count.x * count.y),
          chunkSize(1),
          indexIncrement(threads.size()),
          profilerState(profilerState) {
        nX = count.x;
        for (size_t i = 0; i < threads.size(); ++i)
            threadNextIndex[i].index.store(i * chunkSize, std::memory_order_release);
    }

  public:
    // ParallelForLoop Private Data
    std::function<void(int64_t)> func1D;
    std::function<void(Point2i)> func2D;

    struct alignas(PBRT_L1_CACHE_LINE_SIZE) ThreadWorkIndex {
        std::atomic<int64_t> index;
    };
    static constexpr int maxThreads = 128;
    std::array<ThreadWorkIndex, maxThreads> threadNextIndex;

    const int64_t maxIndex;
    const int chunkSize;
    const int indexIncrement;
    uint64_t profilerState;
    int activeWorkers = 0;
    int nX = -1;
};

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

static std::condition_variable workListCondition;

static void workerThreadFunc(int tIndex, std::shared_ptr<Barrier> barrier) {
    LOG(INFO) << "Started execution in worker thread " << tIndex;
    ThreadIndex = tIndex;

    // Give the profiler a chance to do per-thread initialization for
    // the worker thread before the profiling system actually stops running.
    ProfilerWorkerThreadInit();

    // The main thread sets up a barrier so that it can be sure that all
    // workers have called ProfilerWorkerThreadInit() before it continues
    // (and actually starts the profiling system).
    barrier->Wait();

    // Release our reference to the Barrier so that it's freed once all of
    // the threads have cleared it.
    barrier.reset();

    std::unique_lock<std::mutex> lock(workListMutex);
    while (!shutdownThreads) {
        if (reportWorkerStats) {
            ReportThreadStats();
            if (--reporterCount == 0)
                // Once all worker threads have merged their stats, wake up
                // the main thread.
                reportDoneCondition.notify_one();
            // Now sleep again.
            workListCondition.wait(lock);
        } else if (!workList) {
            // Sleep until there are more tasks to run
            workListCondition.wait(lock);
        } else {
            // Get work from _workList_ and run loop iterations
            ParallelForLoop &loop = *workList;

            loop.activeWorkers++;
            lock.unlock();

            // Lock-free until we're done...
            // For now, just do our own work list...
            // TODO: per-worker random ordering, or something smarter...
            for (int dt = 0; dt < threads.size(); ++dt) {
                while (true) {
                    int64_t indexStart = loop.threadNextIndex[tIndex + dt].index.
                        fetch_add(loop.indexIncrement, std::memory_order_acq_rel);
                    if (indexStart >= loop.maxIndex)
                        // Done.
                        // TODO: work stealing
                        break;
                    int64_t indexEnd =
                        std::min(indexStart + loop.chunkSize, loop.maxIndex);

                    // Run loop indices in _[indexStart, indexEnd)_
                    for (int64_t index = indexStart; index < indexEnd; ++index) {
                        uint64_t oldState = ProfilerState;
                        ProfilerState = loop.profilerState;
                        if (loop.func1D) {
                            loop.func1D(index);
                        }
                        // Handle other types of loops
                        else {
                            CHECK(loop.func2D);
                            loop.func2D(Point2i(index % loop.nX, index / loop.nX));
                        }
                        ProfilerState = oldState;
                    }
                }
            }

            // No more work.
            lock.lock();

            // Update _loop_ to reflect completion of iterations
            if (--loop.activeWorkers > 0) {
                // others are still working. hold up  for now...
                workListCondition.wait(lock);
            } else {
                // everyone is done.
                workList = nullptr;
                // wake them up to clear the wait() from a few lines above
                workListCondition.notify_all();
            }
        }
    }
    LOG(INFO) << "Exiting worker thread " << tIndex;
}

// Parallel Definitions
void ParallelFor(std::function<void(int64_t)> func, int64_t count,
                 int chunkSize) {
    if (threads.size() == 0 && MaxThreadIndex() > 1)
      LOG(WARNING) << "Threads not launched; ParallelFor will run serially";

    // Run iterations immediately if not using threads or if _count_ is small
    if (threads.empty() || count < chunkSize) {
        for (int64_t i = 0; i < count; ++i) func(i);
        return;
    }

    // Create and enqueue _ParallelForLoop_ for this loop
    ParallelForLoop loop(std::move(func), count, chunkSize,
                         CurrentProfilerState());
    workListMutex.lock();
    CHECK(workList == nullptr);
    workList = &loop;
    workListMutex.unlock();

    // Notify worker threads of work to be done
    std::unique_lock<std::mutex> lock(workListMutex);
    workListCondition.notify_all();

    // Help out with parallel loop iterations in the current thread
    int tIndex = ThreadIndex;

    // Get work from _workList_ and run loop iterations
    loop.activeWorkers++;
    lock.unlock();

    // Lock-free until we're done...
    // For now, just do our own work list...
    // TODO: chunksize... We want to add nThreads * chunkSize, but then set end according
    // to chunksize.
    for (int dt = 0; dt < threads.size(); ++dt) {
        while (true) {
            int64_t indexStart = loop.threadNextIndex[tIndex + dt].index.
                fetch_add(loop.indexIncrement, std::memory_order_acq_rel);
            if (indexStart >= loop.maxIndex)
                // Done.
                // TODO: work stealing
                break;
            int64_t indexEnd =
                std::min(indexStart + loop.chunkSize, loop.maxIndex);

            // Run loop indices in _[indexStart, indexEnd)_
            for (int64_t index = indexStart; index < indexEnd; ++index) {
                uint64_t oldState = ProfilerState;
                ProfilerState = loop.profilerState;
                if (loop.func1D) {
                    loop.func1D(index);
                }
                // Handle other types of loops
                else {
                    CHECK(loop.func2D);
                    loop.func2D(Point2i(index % loop.nX, index / loop.nX));
                }
                ProfilerState = oldState;
            }
        }
    }

    // No more work.
    lock.lock();

    // Update _loop_ to reflect completion of iterations
    if (--loop.activeWorkers > 0) {
        // others are still working. hold up  for now...
        workListCondition.wait(lock);
    } else {
        // everyone is done.
        workList = nullptr;
        workListCondition.notify_all();
    }
}

PBRT_THREAD_LOCAL int ThreadIndex;

static int AvailableCores() {
    return std::max(1u, std::thread::hardware_concurrency());
}

int MaxThreadIndex() {
    return PbrtOptions.nThreads == 0 ? AvailableCores() : PbrtOptions.nThreads;
}

void ParallelFor2D(std::function<void(Point2i)> func, const Point2i &count) {
    if (threads.size() == 0 && MaxThreadIndex() > 1)
        LOG(WARNING) << "Threads not launched; ParallelFor will run serially";

    if (threads.empty() || count.x * count.y <= 1) {
        for (int y = 0; y < count.y; ++y)
            for (int x = 0; x < count.x; ++x) func(Point2i(x, y));
        return;
    }
#if 1
    std::vector<Point2i> pos;
    pos.reserve(count.x * count.y);
    for (int y = 0; y < count.y; ++y)
        for (int x = 0; x < count.x; ++x)
            pos.push_back({x, y});
    async::parallel_for(pos, [&func](Point2i p) { func(p); });
    return;
#endif

    ParallelForLoop loop(std::move(func), count, CurrentProfilerState());
    {
        std::lock_guard<std::mutex> lock(workListMutex);
        CHECK(workList == nullptr);
        workList = &loop;
    }

    std::unique_lock<std::mutex> lock(workListMutex);
    workListCondition.notify_all();

    // Help out with parallel loop iterations in the current thread
    int tIndex = ThreadIndex;

    // Get work from _workList_ and run loop iterations
    loop.activeWorkers++;
    lock.unlock();

    // Lock-free until we're done...
    // For now, just do our own work list...
    // TODO: chunksize... We want to add nThreads * chunkSize, but then set end according
    // to chunksize.
    for (int dt = 0; dt < threads.size(); ++dt) {
        while (true) {
            int64_t indexStart = loop.threadNextIndex[tIndex + dt].index.
                fetch_add(loop.indexIncrement, std::memory_order_acq_rel);
            if (indexStart >= loop.maxIndex)
                // Done.
                // TODO: work stealing
                break;
            int64_t indexEnd =
                std::min(indexStart + loop.chunkSize, loop.maxIndex);

            // Run loop indices in _[indexStart, indexEnd)_
            for (int64_t index = indexStart; index < indexEnd; ++index) {
                uint64_t oldState = ProfilerState;
                ProfilerState = loop.profilerState;
                if (loop.func1D) {
                    loop.func1D(index);
                }
                // Handle other types of loops
                else {
                    CHECK(loop.func2D);
                    loop.func2D(Point2i(index % loop.nX, index / loop.nX));
                }
                ProfilerState = oldState;
            }
        }
    }

    // No more work.
    lock.lock();

    // Update _loop_ to reflect completion of iterations
    if (--loop.activeWorkers > 0) {
        // others are still working. hold up  for now...
        workListCondition.wait(lock);
    } else {
        // everyone is done.
        workList = nullptr;
        workListCondition.notify_all();
    }
}

void ParallelInit() {
    CHECK_EQ(threads.size(), 0);
    int nThreads = MaxThreadIndex();
    ThreadIndex = 0;

    // Create a barrier so that we can be sure all worker threads get past
    // their call to ProfilerWorkerThreadInit() before we return from this
    // function.  In turn, we can be sure that the profiling system isn't
    // started until after all worker threads have done that.
    std::shared_ptr<Barrier> barrier = std::make_shared<Barrier>(nThreads);

    // Launch one fewer worker thread than the total number we want doing
    // work, since the main thread helps out, too.
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(workerThreadFunc, i + 1, barrier));

    barrier->Wait();
}

void ParallelCleanup() {
    if (threads.empty()) return;

    {
        std::lock_guard<std::mutex> lock(workListMutex);
        shutdownThreads = true;
        workListCondition.notify_all();
    }

    for (std::thread &thread : threads) thread.join();
    threads.erase(threads.begin(), threads.end());
    shutdownThreads = false;
}

void MergeWorkerThreadStats() {
    std::unique_lock<std::mutex> lock(workListMutex);
    std::unique_lock<std::mutex> doneLock(reportDoneMutex);
    // Set up state so that the worker threads will know that we would like
    // them to report their thread-specific stats when they wake up.
    reportWorkerStats = true;
    reporterCount = threads.size();

    // Wake up the worker threads.
    workListCondition.notify_all();

    // Wait for all of them to merge their stats.
    reportDoneCondition.wait(lock, []() { return reporterCount == 0; });

    reportWorkerStats = false;
}

}  // namespace pbrt
