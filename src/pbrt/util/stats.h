
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

#ifndef PBRT_UTIL_STATS_H
#define PBRT_UTIL_STATS_H

// util/stats.h*

#include <pbrt/pbrt.h>

#include <cstdio>
#include <limits>

namespace pbrt {

// Statistics Declarations
class StatsAccumulator;
class PixelStatsAccumulator;
class StatRegisterer {
  public:
    // StatRegisterer Public Methods
    using InitFunc = void(*)();
    using AccumFunc = void(*)(StatsAccumulator &);
    using PixelAccumFunc = void(*)(const Point2i &p, int counterIndex,
                                   PixelStatsAccumulator &);
    StatRegisterer(InitFunc func);
    StatRegisterer(AccumFunc func, PixelAccumFunc = {});

    static void CallInitializationCallbacks();
    static void CallCallbacks(StatsAccumulator &accum);
    static void CallPixelCallbacks(const Point2i &p, PixelStatsAccumulator &accum);
};

void StatsSetImageResolution(const Bounds2i &b);
void StatsSetPixelStatsBaseName(const char *base);
void StatsReportPixelStart(const Point2i &p);
void StatsReportPixelEnd(const Point2i &p);

void PrintStats(FILE *dest);
void StatsWritePixelImages();
bool PrintCheckRare(FILE *dest);
void ClearStats();
void ReportThreadStats();

class StatsAccumulator {
  public:
    // StatsAccumulator Public Methods
    StatsAccumulator();

    void ReportCounter(const char *name, int64_t val);
    void ReportMemoryCounter(const char *name, int64_t val);
    void ReportPercentage(const char *name, int64_t num, int64_t denom);
    void ReportRatio(const char *name, int64_t num, int64_t denom);
    void ReportRareCheck(const char *condition, float maxFrequency,
                         int64_t numTrue, int64_t total);

    void ReportIntDistribution(const char *name, int64_t sum,
                               int64_t count, int64_t min, int64_t max);
    void ReportFloatDistribution(const char *name, double sum,
                                 int64_t count, double min, double max);

    void AccumulatePixelStats(const PixelStatsAccumulator &accum);
    void WritePixelImages() const;

    void Print(FILE *file);
    bool PrintCheckRare(FILE *dest);
    void Clear();

  private:
    // StatsAccumulator Private Data
    struct Stats;
    Stats *stats = nullptr;
};

class PixelStatsAccumulator {
public:
    PixelStatsAccumulator();

    void ReportPixelMS(const Point2i &p, float ms);
    void ReportCounter(const Point2i &p, int counterIndex,
                       const char *name, int64_t val);
    void ReportRatio(const Point2i &p, int counterIndex,
                     const char *name, int64_t num, int64_t denom);

private:
    friend class StatsAccumulator;
    struct PixelStats;
    PixelStats *stats = nullptr;
};

template <typename T> struct StatCounter {
    using value_type = T;

    StatCounter() = default;
    PBRT_HOST_DEVICE_INLINE
    explicit StatCounter(T value) : value(value) { }

    PBRT_HOST_DEVICE_INLINE
    void operator++() {
#ifdef __CUDA_ARCH__
        atomicAdd(&value, T(1));
#else
        ++value;
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    void operator++(int) {
#ifdef __CUDA_ARCH__
        atomicAdd(&value, T(1));
#else
        ++value;
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    void operator+=(T v) {
#ifdef __CUDA_ARCH__
        atomicAdd(&value, v);
#else
        value += v;
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    void Min(T v) {
#ifdef __CUDA_ARCH__
        atomicMin(&value, v);
#else
        value = v < value ? v : value;
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    void Max(T v) {
#ifdef __CUDA_ARCH__
        atomicMax(&value, v);
#else
        value = v > value ? v : value;
#endif
    }

    PBRT_HOST_DEVICE_INLINE
    explicit operator T() const { return value; }
    PBRT_HOST_DEVICE_INLINE
    void Reset() { value = 0; }

private:
    T value{};
};

#ifdef __CUDA_ARCH__
using CounterT = unsigned long long int;
#else
using CounterT = int64_t;
#endif

// Statistics Macros
#ifdef __CUDA_ARCH__

#define STAT_PERCENT(title, numVar, denomVar)                   \
    static __device__ StatCounter<CounterT> numVar, denomVar

#define STAT_COUNTER(title, var)                \
    static __device__ StatCounter<CounterT> var

#define STAT_PIXEL_COUNTER(title, var)                          \
    static __device__ StatCounter<CounterT> var, var##Sum

#define STAT_MEMORY_COUNTER(title, var)         \
    static __device__ StatCounter<CounterT> var

#define STAT_INT_DISTRIBUTION(title, var)                               \
    static __device__ StatCounter<CounterT> var##sum;                   \
    static __device__ StatCounter<CounterT> var##count;                 \
    static __device__ StatCounter<CounterT> var##min; /*FIXME*/         \
    static __device__ StatCounter<CounterT> var##max /* FIXME */

#define STAT_FLOAT_DISTRIBUTION(title, var)                             \
    static __device__ StatCounter<double> var##sum;                     \
    static __device__ StatCounter<CounterT> var##count;                 \
    static __device__ StatCounter<double> var##min(1e708); \
    static __device__ StatCounter<double> var##max(-1e708)

#define STAT_RATIO(title, numVar, denomVar)                             \
    static __device__ StatCounter<CounterT> numVar, denomVar

#define STAT_PIXEL_RATIO(title, numVar, denomVar) \
    static __device__ StatCounter<CounterT> numVar, numVar##Sum, denomVar, denomVar##Sum

#else

#ifdef __CUDACC__

#define STAT_COUNTER(title, var)                                       \
    static thread_local StatCounter<CounterT> var;                               \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) { \
        accum.ReportCounter(title, CounterT(var));                         \
        var.Reset();                                                    \
        CounterT varHost;                                               \
        cudaMemcpyFromSymbol(&varHost, var, sizeof(var), 0, cudaMemcpyDeviceToHost); \
        accum.ReportCounter(title, CounterT(varHost));                  \
        varHost = 0;                                                    \
        cudaMemcpyToSymbol(var, &varHost, sizeof(var), 0, cudaMemcpyHostToDevice); \
    }); \
    static StatRegisterer STATS_INIT##var([]() { \
        CounterT varHost(0);                                            \
        if (cudaMemcpyToSymbol(var, &varHost, sizeof(var), 0, cudaMemcpyHostToDevice) != cudaSuccess) { \
            cudaError_t error = cudaGetLastError();                         \
            fprintf(stderr, "%s(%d): CUDA error: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        } \
    });

#else

#define STAT_COUNTER(title, var)                                       \
    static thread_local StatCounter<CounterT> var;                               \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) { \
        accum.ReportCounter(title, CounterT(var)); \
        var.Reset();                                               \
    });

#endif


#define STAT_PIXEL_COUNTER(title, var)                                 \
    static thread_local StatCounter<CounterT> var, var##Sum;                     \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) { \
    /* report sum, since if disabled, it all just goes into var... */  \
        accum.ReportCounter(title, CounterT(var) + CounterT(var##Sum)); \
        var##Sum.Reset();                                               \
        var.Reset();                                                    \
    }, [](const Point2i &p, int counterIndex, PixelStatsAccumulator &accum) { \
           accum.ReportCounter(p, counterIndex, title, CounterT(var));   \
           var##Sum += CounterT(var);                                    \
           var.Reset();                                                 \
       });

#define STAT_MEMORY_COUNTER(title, var)                                \
    static thread_local StatCounter<CounterT> var;                               \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) { \
        accum.ReportMemoryCounter(title, CounterT(var)); \
        var.Reset();                                                   \
    });

#define STAT_INT_DISTRIBUTION(title, var)                               \
    static thread_local StatCounter<CounterT> var##sum;                  \
    static thread_local StatCounter<CounterT> var##count;                \
    static thread_local StatCounter<CounterT> var##min(std::numeric_limits<CounterT>::max()); \
    static thread_local StatCounter<CounterT> var##max(std::numeric_limits<CounterT>::lowest()); \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) {  \
        accum.ReportIntDistribution(title, CounterT(var##sum),           \
                                    CounterT(var##count), CounterT(var##min), \
                                    CounterT(var##max));                 \
        var##sum.Reset();                                               \
        var##count.Reset();                                             \
        var##min = StatCounter<CounterT>(std::numeric_limits<CounterT>::max()); \
        var##max = StatCounter<CounterT>(std::numeric_limits<CounterT>::lowest()); \
    });

#define STAT_FLOAT_DISTRIBUTION(title, var)                             \
    static thread_local StatCounter<double> var##sum;                   \
    static thread_local StatCounter<CounterT> var##count;                \
    static thread_local StatCounter<double> var##min(std::numeric_limits<double>::max()); \
    static thread_local StatCounter<double> var##max(std::numeric_limits<double>::lowest()); \
    static StatRegisterer STATS_REG##var([](StatsAccumulator &accum) {  \
        accum.ReportFloatDistribution(title, double(var##sum),         \
                                      CounterT(var##count), double(var##min), \
                                      double(var##max));                \
        var##sum.Reset();                                               \
        var##count.Reset();                                             \
        var##min = StatCounter<double>(std::numeric_limits<double>::max()); \
        var##max = StatCounter<double>(std::numeric_limits<double>::lowest()); \
    });

#define STAT_PERCENT(title, numVar, denomVar)                           \
    static thread_local StatCounter<CounterT> numVar, denomVar;         \
    static StatRegisterer STATS_REG##numVar([](StatsAccumulator &accum) { \
        accum.ReportPercentage(title, CounterT(numVar), CounterT(denomVar)); \
        numVar.Reset();                                                 \
        denomVar.Reset();                                               \
    });

#define STAT_RATIO(title, numVar, denomVar)                             \
    static thread_local StatCounter<CounterT> numVar, denomVar;          \
    static StatRegisterer STATS_REG##numVar([](StatsAccumulator &accum) { \
        accum.ReportRatio(title, CounterT(numVar), CounterT(denomVar)); \
        numVar.Reset();                                                 \
        denomVar.Reset();                                               \
    });

#define STAT_PIXEL_RATIO(title, numVar, denomVar) \
    static thread_local StatCounter<CounterT> numVar, numVar##Sum, denomVar, denomVar##Sum; \
    static StatRegisterer STATS_REG##numVar##denomVar([](StatsAccumulator &accum) {  \
        /* report sum, since if disabled, it all just goes into var... */ \
        accum.ReportRatio(title, CounterT(numVar) + CounterT(numVar##Sum), \
                          CounterT(denomVar) + CounterT(denomVar##Sum));  \
        numVar.Reset();                                                 \
        numVar##Sum.Reset();                                            \
        denomVar.Reset();                                               \
        denomVar##Sum.Reset();                                          \
    }, [](const Point2i &p, int counterIndex, PixelStatsAccumulator &accum) { \
           accum.ReportRatio(p, counterIndex, title, CounterT(numVar),   \
                             CounterT(denomVar));                        \
           numVar##Sum += CounterT(numVar);                              \
           denomVar##Sum += CounterT(denomVar);                          \
           numVar.Reset();                                              \
           denomVar.Reset();                                            \
       });

#endif

#ifdef __CUDA_ARCH__

#define ReportValue(var, value)

#else

#define ReportValue(var, value)                                         \
    do {                                                                \
        var##sum += value;                                              \
        var##count += 1;                                                \
        var##min.Min(value);                                            \
        var##max.Max(value);                                            \
    } while (0)

#endif

}  // namespace pbrt

#endif  // PBRT_UTIL_STATS_H
