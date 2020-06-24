
#ifndef PBRT_LOG_H
#define PBRT_LOG_H

#include <pbrt/pbrt.h>

#include <string>
#include <vector>

namespace pbrt {

enum class LogLevel { Verbose, Error, Fatal, Invalid };

std::string ToString(LogLevel level);
LogLevel LogLevelFromString(const std::string &s);

struct LogConfig {
    LogLevel level = LogLevel::Error;
    int vlogLevel = 0;
};

void InitLogging(LogConfig config, const char *argv0);

#ifdef PBRT_HAVE_OPTIX
void InitGPULogging();

struct GPULogItem {
    LogLevel level;
    char file[32];
    int line;
    char message[64];
};

std::vector<GPULogItem> ReadGPULogs();
#endif

extern LogConfig LOGGING_logConfig;

PBRT_HOST_DEVICE
void Log(LogLevel level, const char *file, int line, const char *s);

PBRT_HOST_DEVICE
[[noreturn]]
void LogFatal(LogLevel level, const char *file, int line, const char *s);

template <typename... Args>
PBRT_HOST_DEVICE
inline void Log(LogLevel level, const char *file, int line,
                const char *fmt, Args&&... args);

template <typename... Args>
PBRT_HOST_DEVICE
[[noreturn]]
inline void LogFatal(LogLevel level, const char *file, int line,
                     const char *fmt, Args&&... args);

#define TO_STRING(x) TO_STRING2(x)
#define TO_STRING2(x) #x

#ifdef __CUDA_ARCH__

extern __constant__ LogConfig LOGGING_logConfigGPU;

#if 1

#define LOG_VERBOSE(...)
#define LOG_ERROR(...)
#define LOG_FATAL(...)
#define VLOG(...)

#else

#define LOG_VERBOSE(...)                                                \
    (pbrt::LogLevel::Verbose >= LOGGING_logConfigGPU.level &&              \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                                                  \
    (pbrt::LogLevel::Error >= LOGGING_logConfigGPU.level &&                \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...)                                                  \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#define VLOG(level, ...)                                                \
    (level <= LOGGING_logConfigGPU.vlogLevel &&                            \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#endif

#else

#define LOG_VERBOSE(...)                                                \
    (pbrt::LogLevel::Verbose >= LOGGING_logConfig.level &&              \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_ERROR(...)                                                  \
    (pbrt::LogLevel::Error >= LOGGING_logConfig.level &&                \
     (pbrt::Log(LogLevel::Error, __FILE__, __LINE__, __VA_ARGS__), true))

#define LOG_FATAL(...)                                                  \
    pbrt::LogFatal(pbrt::LogLevel::Fatal, __FILE__, __LINE__, __VA_ARGS__)

#define VLOG(level, ...)                                                \
    (level <= LOGGING_logConfig.vlogLevel &&                            \
     (pbrt::Log(LogLevel::Verbose, __FILE__, __LINE__, __VA_ARGS__), true))

#endif

} // namespace pbrt

#include <pbrt/util/print.h>

namespace pbrt {

template <typename... Args>
inline void Log(LogLevel level, const char *file, int line,
                const char *fmt, Args&&... args) {
#ifdef __CUDA_ARCH__
    Log(level, file, line, fmt); // just the format string #yolo
#else
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    Log(level, file, line, s.c_str());
#endif
}

template <typename... Args>
inline void LogFatal(LogLevel level, const char *file, int line,
                     const char *fmt, Args&&... args) {
#ifdef __CUDA_ARCH__
    LogFatal(level, file, line, fmt); // just the format string #yolo
#else
    std::string s = StringPrintf(fmt, std::forward<Args>(args)...);
    LogFatal(level, file, line, s.c_str());
#endif
}

} // namespace pbrt

#endif // PBRT_LOG_H
