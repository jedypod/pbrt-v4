
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

#ifndef PBRT_CORE_TYPES_H
#define PBRT_CORE_TYPES_H

// core/types.h*

#include <stdint.h>

#if defined(PBRT_HAVE_OPTIX) && defined(__CUDACC__)
  #ifndef PBRT_NOINLINE
    #define PBRT_NOINLINE __attribute__((noinline))
  #endif
  #define PBRT_HOST __host__
  #define PBRT_HOST_DEVICE __host__ __device__
  #define PBRT_HOST_DEVICE_INLINE __host__ __device__ inline
  #define PBRT_DEVICE_INLINE __device__ inline
  #if defined(__CUDA_ARCH__)
    #define PBRT_CONST __device__ const
  #else
    #define PBRT_CONST const
  #endif
#else
  #define PBRT_CONST const
  #define PBRT_HOST
  #define PBRT_HOST_DEVICE
  #define PBRT_HOST_DEVICE_INLINE inline
  #define PBRT_DEVICE_INLINE inline
#endif

// From ABSL_ARRAYSIZE
#define PBRT_ARRAYSIZE(array) \
  (sizeof(::pbrt::detail::ArraySizeHelper(array)))

namespace pbrt {
namespace detail {

template <typename T, uint64_t N>
auto ArraySizeHelper(const T (&array)[N]) -> char (&)[N];

} // namespace detail
} // namespace pbrt


namespace pstd {

enum class byte : unsigned char {};

PBRT_HOST_DEVICE_INLINE
bool operator==(byte a, byte b) { return (unsigned char)a == (unsigned char)b; }
PBRT_HOST_DEVICE_INLINE
bool operator!=(byte a, byte b) { return !(a == b); }

namespace pmr { template <typename T> class polymorphic_allocator; }

} // namespace pstd

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
class Scene;
class GeneralSceneBase;
class GeneralScene;
class Integrator;
class RayIntegrator;
template <typename T> class Vector2;
template <typename T> class Vector3;
template <typename T> class Point3;
template <typename T> class Point2;
template <typename T> class Normal3;
using Point2f = Point2<Float>;
using Point2i = Point2<int>;
using Point3f = Point3<Float>;
using Vector2f = Vector2<Float>;
using Vector2i = Vector2<int>;
using Vector3f = Vector3<Float>;
class Ray;
class RayDifferential;
template <typename T> class Bounds2;
using Bounds2f = Bounds2<Float>;
using Bounds2i = Bounds2<int>;
template <typename T> class Bounds3;
using Bounds3f = Bounds3<Float>;
using Bounds3i = Bounds3<int>;
class AnimatedTransform;
class Transform;
class Interaction;
class SurfaceInteraction;
class Shape;
class Primitive;
class GeometricPrimitive;
class Image;
struct ImageMetadata;
class RGBColorSpace;
class ColorEncoding;
class RGB;
class XYZ;
enum class SpectrumType;
class SpectrumWavelengths;
class DenselySampledSpectrum;
class SampledSpectrum;
class SpectrumHandle;
class Camera;
struct CameraSample;
class ProjectiveCamera;
class PerspectiveCamera;
class RealisticCamera;
class Sampler;
class Filter;
class FilterSampler;
class Film;
class RGBFilm;
class BxDFHandle;
class BSDF;
enum class BxDFFlags;
class FresnelHandle;
class MicrofacetDistributionHandle;
class PhaseFunction;
class MaterialHandle;
class MaterialBuffer;
class FloatTextureHandle;
class TextureEvalContext;
class Triangle;
class BilinearPatch;
class SampledWavelengths;
class SpectrumTextureHandle;
class Medium;
class MediumInteraction;
struct MediumInterface;
class BSSRDF;
class SeparableBSSRDF;
class MeasuredBRDFData;
class TabulatedBSSRDF;
struct BSSRDFTable;
class Light;
class LightHandle;
struct LightBounds;
class DiffuseAreaLight;
class VisibilityTester;
class AreaLight;
class LightSampler;
class FixedLightSampler;
class Distribution1D;
class Distribution2D;
class ShapeHandle;
template <typename Float> class Interval;
struct FileLoc;
class RGBToSpectrumTable;
class RGBSigmoidPolynomial;
class RGBSpectrum;
class ShapeHandle;
struct ShapeSample;
class LightLiSample;
class LightLeSample;
struct PhaseFunctionSample;
struct BSDFSample;
struct BSSRDFSample;
struct ShapeIntersection;
template <typename T> class Array2D;
class SummedAreaTable;
class Image;
class RNG;
class ProgressReporter;
class MemoryArena;
class Matrix4x4;
class ParameterDictionary;
class TextureParameterDictionary;
class DirectionCone;
struct Options;

using Allocator = pstd::pmr::polymorphic_allocator<pstd::byte>;

 // TransportMode Declarations
enum class TransportMode { Radiance, Importance };

}  // namespace pbrt

#endif  // PBRT_CORE_TYPES_H
