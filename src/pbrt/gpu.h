
#ifndef PBRT_GPU_H
#define PBRT_GPU_H

#include <pbrt/pbrt.h>

#include <pbrt/base.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#ifdef __NVCC__
#include <cuda.h>
#endif

namespace pbrt {

#ifdef PBRT_HAVE_OPTIX

void GPUInit();
void GPURender(GeneralScene &scene);

class CUDAMemoryResource : public pstd::pmr::memory_resource {
    void *do_allocate(size_t size, size_t alignment);
    void do_deallocate(void *p, size_t bytes, size_t alignment);

    bool do_is_equal(const memory_resource &other) const noexcept {
        return this == &other;
    }
};

class Point3fSOA {
public:
    Point3fSOA(Allocator alloc, size_t n)
        : alloc(alloc), n(n) {
        x = alloc.allocate_object<Float>(n);
        y = alloc.allocate_object<Float>(n);
        z = alloc.allocate_object<Float>(n);
    }
    ~Point3fSOA() {
        alloc.deallocate_object(x, n);
        alloc.deallocate_object(y, n);
        alloc.deallocate_object(z, n);
    }

    Point3fSOA() = delete;
    Point3fSOA(const Point3fSOA &) = delete;
    Point3fSOA &operator=(const Point3fSOA &) = delete;

    PBRT_HOST_DEVICE_INLINE
    Point3f at(size_t offset) const {
        DCHECK_LT(offset, n);
        return Point3f(x[offset], y[offset], z[offset]);
    }

    class Point3fRef {
    public:
        PBRT_HOST_DEVICE_INLINE
        Point3fRef(Float *x, Float *y, Float *z) : x(x), y(y), z(z) { }

        PBRT_HOST_DEVICE_INLINE
        operator Point3f() const { return {*x, *y, *z}; }
        PBRT_HOST_DEVICE_INLINE
        void operator=(const Point3f &p) { *x = p.x; *y = p.y; *z = p.z; }

    private:
        Float *x, *y, *z;
    };

    PBRT_HOST_DEVICE_INLINE
    Point3fRef at(size_t offset) {
        DCHECK_LT(offset, n);
        return Point3fRef(&x[offset], &y[offset], &z[offset]);
    }

private:
    Allocator alloc;
    Float *x, *y, *z;
    size_t n;
};

class Vector3fSOA {
public:
    Vector3fSOA(Allocator alloc, size_t n)
        : alloc(alloc), n(n) {
        x = alloc.allocate_object<Float>(n);
        y = alloc.allocate_object<Float>(n);
        z = alloc.allocate_object<Float>(n);
    }
    ~Vector3fSOA() {
        alloc.deallocate_object(x, n);
        alloc.deallocate_object(y, n);
        alloc.deallocate_object(z, n);
    }

    Vector3fSOA() = delete;
    Vector3fSOA(const Vector3fSOA &) = delete;
    Vector3fSOA &operator=(const Vector3fSOA &) = delete;

    PBRT_HOST_DEVICE_INLINE
    Vector3f at(size_t offset) const {
        DCHECK_LT(offset, n);
        return Vector3f(x[offset], y[offset], z[offset]);
    }

    class Vector3fRef {
    public:
        PBRT_HOST_DEVICE_INLINE
        Vector3fRef(Float *x, Float *y, Float *z) : x(x), y(y), z(z) { }

        PBRT_HOST_DEVICE_INLINE
        operator Vector3f() const { return {*x, *y, *z}; }
        PBRT_HOST_DEVICE_INLINE
        void operator=(const Vector3f &p) { *x = p.x; *y = p.y; *z = p.z; }

    private:
        Float *x, *y, *z;
    };

    PBRT_HOST_DEVICE_INLINE
    Vector3fRef at(size_t offset) {
        DCHECK_LT(offset, n);
        return Vector3fRef(&x[offset], &y[offset], &z[offset]);
    }

private:
    Allocator alloc;
    Float *x, *y, *z;
    size_t n;
};

#define CUDA_CHECK(EXPR)                                                \
    if (EXPR != cudaSuccess) {                                          \
        cudaError_t error = cudaGetLastError();                         \
        LOG_FATAL("CUDA error: %s", cudaGetErrorString(error));        \
    } else /* eat semicolon */

#define CU_CHECK(EXPR)                                                \
    do {                                                              \
        CUresult result = EXPR;                                       \
        if (result != CUDA_SUCCESS) {                                 \
            const char *str;                                          \
            CHECK_EQ(CUDA_SUCCESS, cuGetErrorString(result, &str));   \
            LOG_FATAL("CUDA error: %s", str);                         \
        }                                                             \
    } while (false) /* eat semicolon */

#define CUDA_LAUNCH(...)                                                \
    do {                                                                \
      __VA_ARGS__;                                                      \
      cudaError_t error = cudaGetLastError();                           \
      if (error != cudaSuccess)                                         \
          LOG_FATAL("CUDA error: %s", cudaGetErrorString(error));       \
    } while (false) /* eat semicolon */

#define CUDA_LAUNCH_PROFILE(...)                                        \
    do {                                                                \
        cudaEvent_t start, stop;                                        \
        cudaEventCreate(&start);                                        \
        cudaEventCreate(&stop);                                        \
        cudaEventRecord(start);                                         \
        __VA_ARGS__;                                                    \
        cudaError_t error = cudaGetLastError();                         \
        if (error != cudaSuccess)                                       \
            LOG_FATAL("CUDA error: %s", cudaGetErrorString(error));     \
        cudaEventRecord(stop);                                          \
        cudaEventSynchronize(stop);                                     \
        float milliseconds = 0;                                         \
        cudaEventElapsedTime(&milliseconds, start, stop);               \
        LOG_VERBOSE("%s -> %.2f ms on GPU", #__VA_ARGS__, milliseconds); \
    } while (false) /* eat semicolon */

#endif // PBRT_HAVE_OPTIX

}  // namespace pbrt

#endif // PBRT_GPU_H
