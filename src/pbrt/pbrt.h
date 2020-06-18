// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_TYPES_H
#define PBRT_CORE_TYPES_H

// core/types.h*

#include <stdint.h>

#if defined(__CUDA_ARCH__)
#define PBRT_IS_GPU_CODE
#endif

#if defined(PBRT_BUILD_GPU_RENDERER) && defined(__CUDACC__)
#ifndef PBRT_NOINLINE
#define PBRT_NOINLINE __attribute__((noinline))
#endif
#define PBRT_CPU_GPU __host__ __device__
#define PBRT_GPU __device__
#if defined(PBRT_IS_GPU_CODE)
#define PBRT_CONST __device__ const
#else
#define PBRT_CONST const
#endif
#else
#define PBRT_CONST const
#define PBRT_CPU_GPU
#define PBRT_GPU
#endif

#ifdef PBRT_BUILD_GPU_RENDERER
#define PBRT_L1_CACHE_LINE_SIZE 128
#else
#define PBRT_L1_CACHE_LINE_SIZE 64
#endif

// From ABSL_ARRAYSIZE
#define PBRT_ARRAYSIZE(array) (sizeof(::pbrt::detail::ArraySizeHelper(array)))

namespace pbrt {
namespace detail {

template <typename T, uint64_t N>
auto ArraySizeHelper(const T (&array)[N]) -> char (&)[N];

}  // namespace detail
}  // namespace pbrt

namespace pstd {

enum class byte : unsigned char {};

PBRT_CPU_GPU
inline bool operator==(byte a, byte b) {
    return (unsigned char)a == (unsigned char)b;
}
PBRT_CPU_GPU
inline bool operator!=(byte a, byte b) {
    return !(a == b);
}

namespace pmr {
template <typename T>
class polymorphic_allocator;
}

}  // namespace pstd

namespace pbrt {

#ifdef PBRT_FLOAT_AS_DOUBLE
using Float = double;
using FloatBits = uint64_t;
#else
using Float = float;
using FloatBits = uint32_t;
#endif  // PBRT_FLOAT_AS_DOUBLE
static_assert(sizeof(Float) == sizeof(FloatBits),
              "Float and FloatBits must have the same size");

// Global Forward Declarations
class Integrator;
template <typename T>
class Vector2;
template <typename T>
class Vector3;
template <typename T>
class Point3;
template <typename T>
class Point2;
template <typename T>
class Normal3;
using Point2f = Point2<Float>;
using Point2i = Point2<int>;
using Point3f = Point3<Float>;
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;
class Ray;
struct MediumSample;
class RayDifferential;
template <typename T>
class Bounds2;
using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
template <typename T>
class Bounds3;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;
class AnimatedTransform;
class Transform;
class Interaction;
class SurfaceInteraction;
class Shape;
class GeometricPrimitive;
class Image;
struct ImageMetadata;
struct ImageChannelDesc;
class RGBColorSpace;
class RGB;
class XYZ;
enum class SpectrumType;
class SpectrumWavelengths;
class DenselySampledSpectrum;
class SampledSpectrum;
class SpectrumHandle;
struct CameraTransform;
class CameraHandle;
struct CameraSample;
class CameraWiSample;
class FilterHandle;
class FilterSampler;
class FilmHandle;
class BxDFHandle;
class BSDF;
enum class BxDFFlags;
class FresnelHandle;
class MicrofacetDistributionHandle;
class MaterialHandle;
class ScratchBuffer;
class FloatTextureHandle;
class TextureEvalContext;
class Triangle;
class TriangleMesh;
class BilinearPatch;
class SampledWavelengths;
class SpectrumTextureHandle;
class MediumInteraction;
struct MediumInterface;
class BSSRDFHandle;
class MeasuredBRDFData;
class TabulatedBSSRDF;
struct BSSRDFTable;
class LightHandle;
struct LightBounds;
class VisibilityTester;
class PiecewiseConstant1D;
class PiecewiseConstant2D;
class ShapeHandle;
template <typename Float>
class Interval;
struct FileLoc;
class RGBToSpectrumTable;
class RGBSigmoidPolynomial;
class RGBSpectrum;
class ShapeHandle;
class MediumHandle;
struct ShapeSample;
class LightLiSample;
class LightLeSample;
struct PhaseFunctionSample;
struct BSDFSample;
struct BSSRDFSample;
struct ShapeIntersection;
template <typename T>
class Array2D;
class SummedAreaTable;
class Image;
class RNG;
class ProgressReporter;
class Matrix4x4;
class ParameterDictionary;
class TextureParameterDictionary;
class DirectionCone;
struct ExtendedOptions;
struct LogConfig;

using Allocator = pstd::pmr::polymorphic_allocator<pstd::byte>;

void InitPBRT(const ExtendedOptions &opt, const LogConfig &logConfig);
void CleanupPBRT();

}  // namespace pbrt

#endif  // PBRT_CORE_TYPES_H
