
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

#include <pbrt/util/check.h>
#include <pbrt/util/print.h>
#include <pbrt/util/progressreporter.h>
#include <pbrt/util/profile.h>

#include <iterator>
#include <list>
#include <thread>
#include <vector>

namespace pbrt {

std::string AtomicFloat::ToString() const {
    return StringPrintf("%f", float(*this));
}

std::string AtomicDouble::ToString() const {
    return StringPrintf("%f", double(*this));
}

bool Barrier::Block() {
    std::unique_lock<std::mutex> lock(mutex);

    --numToBlock;
    CHECK_GE(numToBlock, 0);

    if (numToBlock > 0) {
        cv.wait(lock, [this]() { return numToBlock == 0; });
    } else
        cv.notify_all();

    return --numToExit == 0;
}

class ParallelJob {
  public:
    virtual ~ParallelJob() { DCHECK(removed); }

    // *lock should be locked going in and and unlocked coming out.
    virtual void RunStep(std::unique_lock<std::mutex> *lock) = 0;
    virtual bool HaveWork() const = 0;

    bool Finished() const { return !HaveWork() && activeWorkers == 0; }

    virtual std::string ToString() const = 0;

protected:
    std::string BaseToString() const {
        return StringPrintf("activeWorkers: %d removed: %s", activeWorkers,
                            removed);
    }

private:
    friend class ThreadPool;

    ParallelJob *prev = nullptr, *next = nullptr;
    int activeWorkers = 0;
    bool removed = false;
};

class ThreadPool {
  public:
    explicit ThreadPool(int nThreads);
    ~ThreadPool();

    size_t size() const { return threads.size(); }

    std::unique_lock<std::mutex> AddToJobList(ParallelJob *job);
    void RemoveFromJobList(ParallelJob *job);

    void WorkOrWait(std::unique_lock<std::mutex> *lock);

    void ForEachThread(std::function<void(void)> func);

    std::string ToString() const;

  private:
    void workerFunc(int tIndex, Barrier *barrier);

    ParallelJob *jobList = nullptr;
    // Protects jobList
    mutable std::mutex jobListMutex;
    // Signaled both when a new job is added to the list and when a job has
    // finished.
    std::condition_variable jobListCondition;

    std::vector<std::thread> threads;
    bool shutdownThreads = false;
};

thread_local int ThreadIndex;

static std::unique_ptr<ThreadPool> threadPool;
static bool maxThreadIndexCalled = false;

ThreadPool::ThreadPool(int nThreads) {
    ThreadIndex = 0;

    // Create a barrier so that we can be sure all worker threads get past
    // their call to ProfilerWorkerThreadInit() before we return from this
    // function.  In turn, we can be sure that the profiling system isn't
    // started until after all worker threads have done that.
    Barrier *barrier = new Barrier(nThreads);

    // Launch one fewer worker thread than the total number we want doing
    // work, since the main thread helps out, too.
    for (int i = 0; i < nThreads - 1; ++i)
        threads.push_back(std::thread(&ThreadPool::workerFunc, this, i + 1,
                                      barrier));

    if (barrier->Block()) delete barrier;
}

ThreadPool::~ThreadPool() {
    if (threads.empty()) return;

    {
        std::lock_guard<std::mutex> lock(jobListMutex);
        shutdownThreads = true;
        jobListCondition.notify_all();
    }

    for (std::thread &thread : threads) thread.join();
}

std::unique_lock<std::mutex> ThreadPool::AddToJobList(ParallelJob *job) {
    std::unique_lock<std::mutex> lock(jobListMutex);
    if (jobList != nullptr) jobList->prev = job;
    job->next = jobList;
    jobList = job;
    jobListCondition.notify_all();
    return lock;
}

void ThreadPool::RemoveFromJobList(ParallelJob *job) {
    DCHECK(!job->removed);

    if (job->prev != nullptr) {
        job->prev->next = job->next;
    } else {
        DCHECK(jobList == job);
        jobList = job->next;
    }
    if (job->next != nullptr)
        job->next->prev = job->prev;

    job->removed = true;
}

void ThreadPool::workerFunc(int tIndex, Barrier *barrier) {
    LOG_VERBOSE("Started execution in worker thread %d", tIndex);
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
        WorkOrWait(&lock);

    LOG_VERBOSE("Exiting worker thread %d", tIndex);
}

void ThreadPool::WorkOrWait(std::unique_lock<std::mutex> *lock) {
    DCHECK(lock->owns_lock());

    ParallelJob *job = jobList;
    while ((job != nullptr) && !job->HaveWork())
        job = job->next;
    if (job != nullptr) {
        // Run a chunk of loop iterations for _loop_
        job->activeWorkers++;

        job->RunStep(lock);

        DCHECK(!lock->owns_lock());
        lock->lock();

        // Update _loop_ to reflect completion of iterations
        job->activeWorkers--;

        if (job->Finished())
            jobListCondition.notify_all();
    } else
        // Wait for something to change (new work, or this loop being
        // finished).
        jobListCondition.wait(*lock);
}

void ThreadPool::ForEachThread(std::function<void(void)> func) {
    Barrier *barrier = new Barrier(threads.size() + 1);

    ParallelFor(0, threads.size() + 1,
                [barrier,&func](int64_t) {
                    func();
                    if (barrier->Block()) delete barrier;
                });
}

std::string ThreadPool::ToString() const {
    std::string s = StringPrintf("[ ThreadPool threads.size(): %d shutdownThreads: %s ",
                                 threads.size(), shutdownThreads);
    if (jobListMutex.try_lock()) {
        s += "jobList: [ ";
        ParallelJob *job = jobList;
        while (job) {
            s += job->ToString() + " ";
            job = job->next;
        }
        s += "] ";
        jobListMutex.unlock();
    }
    else s += "(job list mutex locked) ";
    return s + "]";
}

///////////////////////////////////////////////////////////////////////////
// ParallelJob

class ParallelForLoop1D : public ParallelJob {
  public:
    ParallelForLoop1D(int64_t start, int64_t end, int chunkSize,
                      std::function<void(int64_t, int64_t)> func,
                      ProgressReporter *progressReporter,
                      uint64_t profilerState)
        : func(std::move(func)),
          nextIndex(start),
          maxIndex(end),
          chunkSize(chunkSize),
          progressReporter(progressReporter),
          profilerState(profilerState) {}

    bool HaveWork() const { return nextIndex < maxIndex; }
    void RunStep(std::unique_lock<std::mutex> *lock);

    std::string ToString() const {
        return StringPrintf("[ ParallelForLoop1D nextIndex: %d maxIndex: %d "
                            "chunkSize: %d progressReporter: %s profilerState: %d ]", nextIndex,
                            maxIndex, chunkSize,
                            progressReporter ? progressReporter->ToString() : "(nullptr)",
                            profilerState);
    }

private:
    std::function<void(int64_t, int64_t)> func;
    int64_t nextIndex;
    int64_t maxIndex;
    int chunkSize;
    ProgressReporter *progressReporter;
    uint64_t profilerState;
};

class ParallelForLoop2D : public ParallelJob {
  public:
    ParallelForLoop2D(const Bounds2i &extent, int chunkSize,
                      std::function<void(Bounds2i)> func,
                      ProgressReporter *progressReporter,
                      uint64_t profilerState)
        : func(std::move(func)),
          extent(extent),
          nextStart(extent.pMin),
          chunkSize(chunkSize),
          progressReporter(progressReporter),
          profilerState(profilerState) {}

    bool HaveWork() const { return nextStart.y < extent.pMax.y; }
    void RunStep(std::unique_lock<std::mutex> *lock);

    std::string ToString() const {
        return StringPrintf("[ ParallelForLoop2D extent: %s nextStart: %s "
                            "chunkSize: %d progressReporter: %s profilerState: %d ]", extent,
                            nextStart, chunkSize,
                            progressReporter ? progressReporter->ToString() : "(nullptr)",
                            profilerState);
    }

  private:
    std::function<void(Bounds2i)> func;
    const Bounds2i extent;
    Point2i nextStart;
    int chunkSize;
    ProgressReporter *progressReporter;
    uint64_t profilerState;
};

void ParallelForLoop1D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Find the set of loop iterations to run next
    int64_t indexStart = nextIndex;
    int64_t indexEnd = std::min(indexStart + chunkSize, maxIndex);

    // Update _loop_ to reflect iterations this thread will run
    nextIndex = indexEnd;

    if (!HaveWork())
        threadPool->RemoveFromJobList(this);

    lock->unlock();

    // Run loop indices in _[indexStart, indexEnd)_
    uint64_t oldState = ProfilerState;
    ProfilerState = profilerState;

    func(indexStart, indexEnd);

    ProfilerState = oldState;

    if (progressReporter) {
        if (!HaveWork())
            progressReporter->Done();
        else
            progressReporter->Update(indexEnd - indexStart);
    }
}

void ParallelForLoop2D::RunStep(std::unique_lock<std::mutex> *lock) {
    // Compute extent for this step
    Point2i end = nextStart + Vector2i(chunkSize, chunkSize);
    Bounds2i b = Intersect(Bounds2i(nextStart, end), extent);
    CHECK(!b.IsEmpty());

    // Advance to be ready for the next extent.
    nextStart.x += chunkSize;
    if (nextStart.x >= extent.pMax.x) {
        nextStart.x = extent.pMin.x;
        nextStart.y += chunkSize;
    }

    if (!HaveWork())
        threadPool->RemoveFromJobList(this);

    lock->unlock();

    // Run the loop iteration
    uint64_t oldState = ProfilerState;
    ProfilerState = profilerState;

    func(b);

    ProfilerState = oldState;

    if (progressReporter) {
        if (!HaveWork())
            progressReporter->Done();
        else
            progressReporter->Update(b.Area());
    }
}

void ParallelFor(int64_t start, int64_t end, std::function<void(int64_t, int64_t)> func,
                 const char *progressName) {
    CHECK(threadPool);

    // https://stackoverflow.com/a/23934764 :-(
    ProgressReporter &&progress = [&]() -> ProgressReporter {
        if (progressName)
            return {end - start, progressName};
        else
            return { };
    }();

    int64_t chunkSize = std::max<int64_t>(1, (end - start) / (8 * RunningThreads()));

    if (end - start < chunkSize) {
        func(start, end);
        return;
    }

    // Create and enqueue _ParallelJob_ for this loop
    ParallelForLoop1D loop(start, end, chunkSize, std::move(func),
                           &progress, CurrentProfilerState());
    std::unique_lock<std::mutex> lock = threadPool->AddToJobList(&loop);

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        threadPool->WorkOrWait(&lock);
}

void ParallelFor2D(const Bounds2i &extent, std::function<void(Bounds2i)> func,
                   const char *progressName) {
    CHECK(threadPool);

    // https://stackoverflow.com/a/23934764 :-(
    ProgressReporter &&progress = [&]() -> ProgressReporter {
        if (progressName)
            return {extent.Area(), progressName};
        else
            return { };
    }();

    if (extent.IsEmpty())
        return;
    if (extent.Area() == 1) {
        func(extent);
        return;
    }

    // Want at least 8 tiles per thread, subject to not too big and not too small.
    // TODO: should we do non-square?
    int tileSize = Clamp(int(std::sqrt(extent.Diagonal().x * extent.Diagonal().y /
                                       (8 * RunningThreads()))),
                         1, 32);

    ParallelForLoop2D loop(extent, tileSize, std::move(func), &progress,
                           CurrentProfilerState());
    std::unique_lock<std::mutex> lock = threadPool->AddToJobList(&loop);

    // Help out with parallel loop iterations in the current thread
    while (!loop.Finished())
        threadPool->WorkOrWait(&lock);
}

///////////////////////////////////////////////////////////////////////////

int AvailableCores() {
    return std::max<int>(1, std::thread::hardware_concurrency());
}

int RunningThreads() {
    return threadPool ? (1 + threadPool->size()) : 1;
}

int MaxThreadIndex() {
    maxThreadIndexCalled = true;
    return threadPool ? (1 + threadPool->size()) : 1;
}

void ParallelInit(int nThreads) {
    // This is risky: if the caller has allocated per-thread data
    // structures before calling ParallelInit(), then we may end up having
    // them accessed with a higher ThreadIndex than the caller expects.
    CHECK(!maxThreadIndexCalled);

    CHECK(!threadPool);
    if (nThreads <= 0)
        nThreads = AvailableCores();
    threadPool = std::make_unique<ThreadPool>(nThreads);
}

void ParallelCleanup() {
    threadPool.reset();
    maxThreadIndexCalled = false;
}

void ForEachThread(std::function<void(void)> func) {
    threadPool->ForEachThread(std::move(func));
}

}  // namespace pbrt
