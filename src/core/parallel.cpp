
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
    virtual ~ParallelForLoop() { }
    // *lock should be locked going in and will be locked coming out.
    virtual void RunStep(std::unique_lock<std::mutex> *lock) = 0;
    virtual bool Finished() const = 0;

    ParallelForLoop *next = nullptr;
};

class ParallelForLoop1D : public ParallelForLoop {
  public:
    ParallelForLoop1D(int64_t start, int64_t end, int chunkSize,
                      std::function<void(int64_t)> func, uint64_t profilerState)
        : func(std::move(func)),
          nextIndex(start),
          maxIndex(end),
          chunkSize(chunkSize),
          profilerState(profilerState) {}

    bool Finished() const {
        return nextIndex >= maxIndex && activeWorkers == 0;
    }
    void RunStep(std::unique_lock<std::mutex> *lock);

private:
    std::function<void(int64_t)> func;
    int64_t nextIndex;
    const int64_t maxIndex;
    const int chunkSize;
    const uint64_t profilerState;
    int activeWorkers = 0;
};

class ParallelForLoop2D : public ParallelForLoop {
  public:
    ParallelForLoop2D(const Bounds2i &extent, int chunkSize,
                      const std::function<void(Bounds2i)> &f,
                      uint64_t profilerState)
        : func(f),
          extent(extent),
          nextStart(extent.pMin),
          chunkSize(chunkSize),
          profilerState(profilerState) {}

    bool Finished() const {
        return nextStart.y >= extent.pMax.y && activeWorkers == 0;
    }
    void RunStep(std::unique_lock<std::mutex> *lock);

  private:
    std::function<void(Bounds2i)> func;
    const Bounds2i extent;
    Point2i nextStart;
    const int chunkSize;
    const uint64_t profilerState;
    int activeWorkers = 0;
};

void ParallelForLoop1D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Run a chunk of loop iterations for _loop_
    DCHECK(lock->owns_lock());

    activeWorkers++;

    // Find the set of loop iterations to run next
    int64_t indexStart = nextIndex;
    int64_t indexEnd = std::min(indexStart + chunkSize, maxIndex);

    // Update _loop_ to reflect iterations this thread will run
    nextIndex = indexEnd;
    if (nextIndex == maxIndex) workList = next;

    lock->unlock();

    // Run loop indices in _[indexStart, indexEnd)_
    uint64_t oldState = ProfilerState;
    ProfilerState = profilerState;

    for (int64_t index = indexStart; index < indexEnd; ++index)
        func(index);

    ProfilerState = oldState;

    lock->lock();

    // Update _loop_ to reflect completion of iterations
    activeWorkers--;
}

void ParallelForLoop2D::RunStep(std::unique_lock<std::mutex> *lock) {
    DCHECK(lock->owns_lock());

    if (nextStart.y >= extent.pMax.y) {
        lock->unlock();
        lock->lock();
        return;
    }

    // Compute extent for this step
    Point2i end = nextStart + Vector2i(chunkSize, chunkSize);
    Bounds2i b = Intersect(Bounds2i(nextStart, end), extent);

    // Advance to be ready for the next extent.
    nextStart.x += chunkSize;
    if (nextStart.x >= extent.pMax.x) {
        nextStart.x = extent.pMin.x;
        nextStart.y += chunkSize;
    }
    if (nextStart.y >= extent.pMax.y) workList = next;

    activeWorkers++;

    lock->unlock();

    if (!b.Empty()) {
        // Run the loop iteration
        uint64_t oldState = ProfilerState;
        ProfilerState = profilerState;

        func(b);

        ProfilerState = oldState;
    }

    lock->lock();

    // Update _loop_ to reflect completion of iterations
    activeWorkers--;
}

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
            loop.RunStep(&lock);
            if (loop.Finished()) workListCondition.notify_all();
        }
    }
    LOG(INFO) << "Exiting worker thread " << tIndex;
}

// Parallel Definitions
void ParallelFor(int64_t start, int64_t end, int chunkSize,
                 std::function<void(int64_t)> func) {
    if (threads.size() == 0 && MaxThreadIndex() > 1)
        LOG(WARNING) << "Threads not launched; ParallelFor will run serially";

    // Create and enqueue _ParallelForLoop_ for this loop
    ParallelForLoop1D loop(start, end, chunkSize, std::move(func),
                           CurrentProfilerState());
    std::unique_lock<std::mutex> lock(workListMutex);
    loop.next = workList;
    workList = &loop;
    workListCondition.notify_all();

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        loop.RunStep(&lock);
}

PBRT_THREAD_LOCAL int ThreadIndex;

static int AvailableCores() {
    return std::max(1u, std::thread::hardware_concurrency());
}

int MaxThreadIndex() {
    return PbrtOptions.nThreads == 0 ? AvailableCores() : PbrtOptions.nThreads;
}

void ParallelFor2D(const Bounds2i &extent, int chunkSize,
                   std::function<void(Bounds2i)> func) {
    if (threads.size() == 0 && MaxThreadIndex() > 1)
        LOG(WARNING) << "Threads not launched; ParallelFor will run serially";

    CHECK_GE(extent.Area(), 0);  // or just return immediately?

    ParallelForLoop2D loop(extent, chunkSize, std::move(func), CurrentProfilerState());

    std::unique_lock<std::mutex> lock(workListMutex);
    loop.next = workList;
    workList = &loop;
    workListCondition.notify_all();

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        loop.RunStep(&lock);
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
