
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
#include <pbrt/util/parallel.h>

#include <pbrt/util/memory.h>
#include <pbrt/util/stats.h>

#include <absl/synchronization/barrier.h>
#include <list>
#include <thread>
#include <vector>

namespace pbrt {

class ParallelJob {
  public:
    static void StartThreads(int nThreads);
    static void TerminateThreads();

    virtual ~ParallelJob() { DCHECK(removed); }

    std::unique_lock<std::mutex> AddToJobList();
    // *lock should be locked going in and and unlocked coming out.
    virtual void RunStep(std::unique_lock<std::mutex> *lock) = 0;

    virtual bool HaveWork() const = 0;
    bool Finished() const { return !HaveWork() && activeWorkers == 0; }
    static size_t NumThreads() { return threads.size(); }

    static void DoWork(std::unique_lock<std::mutex> &lock);

protected:
    void RemoveFromJobList();

private:
    static void workerFunc(int tIndex, absl::Barrier *barrier);

    static ParallelJob *jobList;
    // Protects jobList
    static std::mutex jobListMutex;
    // Signaled both when a new job is added to the list and when a job has
    // finished.
    static std::condition_variable jobListCondition;

    static std::vector<std::thread> threads;
    static bool shutdownThreads;

    ParallelJob *prev = nullptr, *next = nullptr;
    int activeWorkers = 0;
    bool removed = false;
};

ParallelJob *ParallelJob::jobList = nullptr;
std::mutex ParallelJob::jobListMutex;
std::condition_variable ParallelJob::jobListCondition;

std::vector<std::thread> ParallelJob::threads;
bool ParallelJob::shutdownThreads = false;

void ParallelJob::DoWork(std::unique_lock<std::mutex> &lock) {
    DCHECK(lock.owns_lock());

    ParallelJob *job = jobList;
    while (job && !job->HaveWork())
        job = job->next;
    if (job) {
        // Run a chunk of loop iterations for _loop_
        CHECK(job->HaveWork());

        job->activeWorkers++;

        job->RunStep(&lock);

        DCHECK(!lock.owns_lock());
        lock.lock();

        // Update _loop_ to reflect completion of iterations
        job->activeWorkers--;

        if (job->Finished())
            jobListCondition.notify_all();
    } else
        // Wait for something to change (new work, or this loop being
        // finished).
        ParallelJob::jobListCondition.wait(lock);
}

std::unique_lock<std::mutex> ParallelJob::AddToJobList() {
    if (threads.size() == 0 && MaxThreadIndex() > 1)
        LOG(WARNING) << "Threads not launched; job will run serially";

    std::unique_lock<std::mutex> lock(jobListMutex);
    if (jobList) jobList->prev = this;
    next = jobList;
    jobList = this;
    jobListCondition.notify_all();
    return lock;
}

void ParallelJob::RemoveFromJobList() {
    if (prev) {
        prev->next = next;
    } else {
        DCHECK(jobList == this);
        jobList = next;
    }
    if (next)
        next->prev = prev;
    removed = true;
}

class ParallelForLoop1D : public ParallelJob {
  public:
    ParallelForLoop1D(int64_t start, int64_t end, int chunkSize,
                      std::function<void(int64_t, int64_t)> func,
                      uint64_t profilerState)
        : func(std::move(func)),
          nextIndex(start),
          maxIndex(end),
          chunkSize(chunkSize),
          profilerState(profilerState) {}

    bool HaveWork() const { return nextIndex < maxIndex; }
    void RunStep(std::unique_lock<std::mutex> *lock);

private:
    std::function<void(int64_t, int64_t)> func;
    int64_t nextIndex;
    const int64_t maxIndex;
    const int chunkSize;
    const uint64_t profilerState;
};

class ParallelForLoop2D : public ParallelJob {
  public:
    ParallelForLoop2D(const Bounds2i &extent, int chunkSize,
                      std::function<void(Bounds2i)> func,
                      uint64_t profilerState)
        : func(std::move(func)),
          extent(extent),
          nextStart(extent.pMin),
          chunkSize(chunkSize),
          profilerState(profilerState) {}

    bool HaveWork() const { return nextStart.y < extent.pMax.y; }
    void RunStep(std::unique_lock<std::mutex> *lock);

  private:
    std::function<void(Bounds2i)> func;
    const Bounds2i extent;
    Point2i nextStart;
    const int chunkSize;
    const uint64_t profilerState;
};

void ParallelForLoop1D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Find the set of loop iterations to run next
    int64_t indexStart = nextIndex;
    int64_t indexEnd = std::min(indexStart + chunkSize, maxIndex);

    // Update _loop_ to reflect iterations this thread will run
    nextIndex = indexEnd;

    if (!HaveWork())
        RemoveFromJobList();

    lock->unlock();

    // Run loop indices in _[indexStart, indexEnd)_
    uint64_t oldState = ProfilerState;
    ProfilerState = profilerState;

    func(indexStart, indexEnd);

    ProfilerState = oldState;
}

void ParallelForLoop2D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Compute extent for this step
    Point2i end = nextStart + Vector2i(chunkSize, chunkSize);
    Bounds2i b = Intersect(Bounds2i(nextStart, end), extent);
    CHECK(!b.Empty());

    // Advance to be ready for the next extent.
    nextStart.x += chunkSize;
    if (nextStart.x >= extent.pMax.x) {
        nextStart.x = extent.pMin.x;
        nextStart.y += chunkSize;
    }

    if (!HaveWork())
        RemoveFromJobList();

    lock->unlock();

    // Run the loop iteration
    uint64_t oldState = ProfilerState;
    ProfilerState = profilerState;

    func(b);

    ProfilerState = oldState;
}

void ParallelJob::workerFunc(int tIndex, absl::Barrier *barrier) {
    LOG(INFO) << "Started execution in worker thread " << tIndex;
    ThreadIndex = tIndex;

    // Give the profiler a chance to do per-thread initialization for
    // the worker thread before the profiling system actually stops running.
    ProfilerWorkerThreadInit();

    // The main thread sets up a barrier so that it can be sure that all
    // workers have called ProfilerWorkerThreadInit() before it continues
    // (and actually starts the profiling system).
    if (barrier->Block()) delete barrier;

    std::unique_lock<std::mutex> lock(jobListMutex);
    while (!shutdownThreads)
        DoWork(lock);

    LOG(INFO) << "Exiting worker thread " << tIndex;
}

void ParallelFor(int64_t start, int64_t end, int chunkSize,
                 std::function<void(int64_t, int64_t)> func) {
    if (end - start < chunkSize) {
        func(start, end);
        return;
    }

    // Create and enqueue _ParallelJob_ for this loop
    ParallelForLoop1D loop(start, end, chunkSize, std::move(func),
                           CurrentProfilerState());
    std::unique_lock<std::mutex> lock = loop.AddToJobList();

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        ParallelJob::DoWork(lock);
}

void ParallelFor2D(const Bounds2i &extent, int chunkSize, std::function<void(Bounds2i)> func) {
    if (extent.Empty())
        return;

    if (extent.Area() < chunkSize * chunkSize) {
        func(extent);
        return;
    }

    ParallelForLoop2D loop(extent, chunkSize, std::move(func),
                           CurrentProfilerState());
    std::unique_lock<std::mutex> lock = loop.AddToJobList();

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        ParallelJob::DoWork(lock);
}

///////////////////////////////////////////////////////////////////////////

thread_local int ThreadIndex;
static bool parallelInitialized = false;

int AvailableCores() {
    return std::max<int>(1, std::thread::hardware_concurrency());
}

int MaxThreadIndex() {
    return parallelInitialized ? (1 + ParallelJob::NumThreads()) : 1;
}

void ParallelInit(int nThreads) {
    CHECK(!parallelInitialized);
    parallelInitialized = true;
    if (nThreads <= 0)
        nThreads = AvailableCores();
    ParallelJob::StartThreads(nThreads);
}

void ParallelJob::StartThreads(int nThreads) {
    CHECK_EQ(threads.size(), 0);
    ThreadIndex = 0;

    // Create a barrier so that we can be sure all worker threads get past
    // their call to ProfilerWorkerThreadInit() before we return from this
    // function.  In turn, we can be sure that the profiling system isn't
    // started until after all worker threads have done that.
    absl::Barrier *barrier = new absl::Barrier(nThreads);

    // Launch one fewer worker thread than the total number we want doing
    // work, since the main thread helps out, too.
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(workerFunc, i + 1, barrier));

    if (barrier->Block()) delete barrier;
}

void ParallelCleanup() {
    ParallelJob::TerminateThreads();
    parallelInitialized = false;
}

void ParallelJob::TerminateThreads() {
    if (threads.empty()) return;

    {
        std::lock_guard<std::mutex> lock(jobListMutex);
        shutdownThreads = true;
        jobListCondition.notify_all();
    }

    for (std::thread &thread : threads) thread.join();
    threads.erase(threads.begin(), threads.end());
    shutdownThreads = false;
}

void ForEachWorkerThread(std::function<void(void)> func) {
    int nThreads = MaxThreadIndex();
    absl::Barrier *barrier = new absl::Barrier(nThreads);

    ParallelFor(0, nThreads,
                [barrier,&func](int64_t) {
                    func();
                    if (barrier->Block()) delete barrier;
                });
}

void MergeWorkerThreadStats() {
    ForEachWorkerThread(ReportThreadStats);
}

}  // namespace pbrt
