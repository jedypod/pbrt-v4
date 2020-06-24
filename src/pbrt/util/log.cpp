
#include <pbrt/util/log.h>

#include <pbrt/gpu.h>
#include <pbrt/util/check.h>
#include <pbrt/util/parallel.h>

#include <iostream>
#include <stdio.h>
#include <mutex>
#include <iomanip>
#include <ctime>
#include <sstream>

#ifdef PBRT_IS_OSX
  #include <sys/syscall.h>
  #include <unistd.h>
#endif
#ifdef PBRT_IS_LINUX
  #include <sys/types.h>
  #include <unistd.h>
#endif

namespace pbrt {

namespace {

std::string TimeNow() {
    std::time_t t =  std::time(NULL);
    std::tm tm    = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y%m%d.%H%M%S");
    return ss.str();
}

#define LOG_BASE_FMT "%d.%03d %s"
#define LOG_BASE_ARGS getpid(), ThreadIndex, TimeNow().c_str()

}

LogConfig LOGGING_logConfig;

void InitLogging(LogConfig config, const char *argv0) {
    LOGGING_logConfig = config;
}

#ifdef __CUDACC__

__constant__ LogConfig LOGGING_logConfigGPU;

#define MAX_LOG_ITEMS 128
__device__ GPULogItem rawLogItems[MAX_LOG_ITEMS];
__device__ int nRawLogItems;

void InitGPULogging() {
    CUDA_CHECK(cudaMemcpyToSymbol(LOGGING_logConfigGPU, &LOGGING_logConfig,
                                  sizeof(LOGGING_logConfig)));
}

std::vector<GPULogItem> ReadGPULogs() {
    int nItems;
    CUDA_CHECK(cudaMemcpyFromSymbol(&nItems, nRawLogItems, sizeof(nItems)));

    std::vector<GPULogItem> items(nItems);
    CUDA_CHECK(cudaMemcpyFromSymbol(items.data(), rawLogItems,
                          nItems * sizeof(GPULogItem)));

    return items;
}
#endif

LogLevel LogLevelFromString(const std::string &s) {
    if (s == "verbose")
        return LogLevel::Verbose;
    else if (s == "error")
        return LogLevel::Error;
    else if (s == "fatal")
        return LogLevel::Fatal;
    return LogLevel::Invalid;
}

std::string ToString(LogLevel level) {
    switch (level) {
    case LogLevel::Verbose:
        return "VERBOSE";
    case LogLevel::Error:
        return "ERROR";
    case LogLevel::Fatal:
        return "FATAL";
    default:
        return "UNKNOWN";
    }
}

void Log(LogLevel level, const char *file, int line, const char *s) {
#ifdef __CUDA_ARCH__
    int offset = atomicAdd(&nRawLogItems, 1);

    GPULogItem &item = rawLogItems[offset];
    item.level = level;

    const char *ptr = file;
    while (*ptr)
        ++ptr;
    int len = ptr - file;
    int start = len - sizeof(item.file) + 1;
    if (start < 0) start = 0;
    for (int i = 0; start + i <= len; ++i)
        item.file[i] = file[start + i];

    item.line = line;

    int i;
    for (i = 0; i < sizeof(item.message) - 1 && *s; ++i, ++s)
        item.message[i] = *s;
    item.message[i] = '\0';
#else
    if (strlen(s) == 0) return;
    fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS,
            file, line, ToString(level).c_str(), s);
#endif
}

void LogFatal(LogLevel level, const char *file, int line,
              const char *s) {
#ifdef __CUDA_ARCH__
    Log(LogLevel::Fatal, file, line, s);
    __threadfence();
    asm("trap;");
#else
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    fprintf(stderr, "[ " LOG_BASE_FMT " %s:%d ] %s %s\n", LOG_BASE_ARGS,
            file, line, ToString(level).c_str(), s);

    CheckCallbackScope::Fail();
    abort();
#endif
}

} // namespace pbrt
