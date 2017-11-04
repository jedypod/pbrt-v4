
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

#ifndef PBRT_CORE_PARAMSET_H
#define PBRT_CORE_PARAMSET_H

// core/paramset.h*
#include <pbrt/core/pbrt.h>

#include <absl/container/inlined_vector.h>
#include <absl/types/span.h>
#include <pbrt/util/geometry.h>
#include <pbrt/core/spectrum.h>
#include <pbrt/core/texture.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pbrt {

// Note the std::string *s. We assume that the caller will handle
// allocating and deallocating these (e.g. via a MemoryPool);
// NamedValues doesn't do anything to free them itself.
class NamedValues {
 public:
    void AddNumber(double d);
    void AddString(const std::string *str);
    void AddBool(bool v);

    std::string ToString() const;

    std::string *name = nullptr;
    NamedValues *next = nullptr;

    std::vector<double> numbers;
    std::vector<const std::string *> strings;
    std::vector<bool> bools;
};

template <typename T>
struct ParamSetItem {
    // ParamSetItem Public Methods
    ParamSetItem(const std::string &name, std::vector<T> values)
       : name(name), values(std::move(values)) {}

    // ParamSetItem Data
    std::string name;
    std::vector<T> values;
    mutable bool lookedUp = false;
};

// ParamSet Declarations
class ParamSet {
  public:
    // ParamSet Public Methods
    void Parse(const NamedValues *namedValuesList, SpectrumType spectrumType);

    void AddFloat(const std::string &, std::vector<Float> v);
    void AddInt(const std::string &, std::vector<int> v);
    // Use uint8_t for bool rather than an actual bool so that the
    // std::vector<bool> specialization doesn't kick in (which in turn
    // prevents us from being able to return an ArraySlice from
    // GetBoolArray()).
    void AddBool(const std::string &, std::vector<uint8_t> v);
    void AddPoint2f(const std::string &, std::vector<Point2f> v);
    void AddVector2f(const std::string &, std::vector<Vector2f> v);
    void AddPoint3f(const std::string &, std::vector<Point3f> v);
    void AddVector3f(const std::string &, std::vector<Vector3f> v);
    void AddNormal3f(const std::string &, std::vector<Normal3f> v);
    void AddString(const std::string &, std::vector<std::string> v);
    void AddTexture(const std::string &, const std::string &);
    void AddSpectrum(const std::string &, std::vector<Spectrum> v);

    Float GetOneFloat(const std::string &name, Float def) const;
    int GetOneInt(const std::string &name, int def) const;
    bool GetOneBool(const std::string &name, bool def) const;
    Point2f GetOnePoint2f(const std::string &name, const Point2f &def) const;
    Vector2f GetOneVector2f(const std::string &name, const Vector2f &def) const;
    Point3f GetOnePoint3f(const std::string &name, const Point3f &def) const;
    Vector3f GetOneVector3f(const std::string &name, const Vector3f &def) const;
    Normal3f GetOneNormal3f(const std::string &name, const Normal3f &def) const;
    Spectrum GetOneSpectrum(const std::string &name, const Spectrum &def) const;
    std::string GetOneString(const std::string &name,
                             const std::string &def) const;
    std::string FindTexture(const std::string &) const;

    absl::Span<const Float> GetFloatArray(const std::string &name) const;
    absl::Span<const int> GetIntArray(const std::string &name) const;
    absl::Span<const uint8_t> GetBoolArray(const std::string &name) const;
    absl::Span<const Point2f> GetPoint2fArray(const std::string &name) const;
    absl::Span<const Vector2f> GetVector2fArray(const std::string &name) const;
    absl::Span<const Point3f> GetPoint3fArray(const std::string &name) const;
    absl::Span<const Vector3f> GetVector3fArray(const std::string &name) const;
    absl::Span<const Normal3f> GetNormal3fArray(const std::string &name) const;
    absl::Span<const Spectrum> GetSpectrumArray(const std::string &name) const;
    absl::Span<const std::string> GetStringArray(const std::string &name) const;

    void ReportUnused() const;
    std::string ToString(int indent = 0) const;

  private:
    // ParamSet Private Data
    friend class TextureParams;
    friend bool shapeMaySetMaterialParameters(const ParamSet &ps);

    std::vector<ParamSetItem<uint8_t>> bools;
    std::vector<ParamSetItem<int>> ints;
    std::vector<ParamSetItem<Float>> floats;
    std::vector<ParamSetItem<Point2f>> point2fs;
    std::vector<ParamSetItem<Vector2f>> vector2fs;
    std::vector<ParamSetItem<Point3f>> point3fs;
    std::vector<ParamSetItem<Vector3f>> vector3fs;
    std::vector<ParamSetItem<Normal3f>> normals;
    std::vector<ParamSetItem<Spectrum>> spectra;
    std::vector<ParamSetItem<std::string>> strings;
    std::vector<ParamSetItem<std::string>> textures;
};

// TextureParams Declarations
class TextureParams {
  public:
    // TextureParams Public Methods
    TextureParams(
        std::initializer_list<const ParamSet *> params,
        std::map<std::string, std::shared_ptr<Texture<Float>>> &fTex,
        std::map<std::string, std::shared_ptr<Texture<Spectrum>>> &sTex)
        : paramSets(params),
          floatTextures(fTex),
          spectrumTextures(sTex) {}

    Float GetOneFloat(const std::string &name, Float def) const;
    int GetOneInt(const std::string &name, int def) const;
    bool GetOneBool(const std::string &name, bool def) const;
    Point2f GetOnePoint2f(const std::string &name, const Point2f &def) const;
    Vector2f GetOneVector2f(const std::string &name, const Vector2f &def) const;
    Point3f GetOnePoint3f(const std::string &name, const Point3f &def) const;
    Vector3f GetOneVector3f(const std::string &name, const Vector3f &def) const;
    Normal3f GetOneNormal3f(const std::string &name, const Normal3f &def) const;
    Spectrum GetOneSpectrum(const std::string &name, const Spectrum &def) const;
    std::string GetOneString(const std::string &name,
                             const std::string &def) const;
    std::string FindTexture(const std::string &) const;

    absl::Span<const Float> GetFloatArray(const std::string &name) const;
    absl::Span<const int> GetIntArray(const std::string &name) const;
    absl::Span<const uint8_t> GetBoolArray(const std::string &name) const;
    absl::Span<const Point2f> GetPoint2fArray(const std::string &name) const;
    absl::Span<const Vector2f> GetVector2fArray(const std::string &name) const;
    absl::Span<const Point3f> GetPoint3fArray(const std::string &name) const;
    absl::Span<const Vector3f> GetVector3fArray(const std::string &name) const;
    absl::Span<const Normal3f> GetNormal3fArray(const std::string &name) const;
    absl::Span<const Spectrum> GetSpectrumArray(const std::string &name) const;
    absl::Span<const std::string> GetStringArray(const std::string &name) const;

    std::shared_ptr<Texture<Spectrum>> GetSpectrumTexture(
        const std::string &name, const Spectrum &def) const;
    std::shared_ptr<Texture<Spectrum>> GetSpectrumTextureOrNull(
        const std::string &name) const;
    std::shared_ptr<Texture<Float>> GetFloatTexture(const std::string &name,
                                                    Float def) const;
    std::shared_ptr<Texture<Float>> GetFloatTextureOrNull(
        const std::string &name) const;

    void ReportUnused() const;

 private:
    // TextureParams Private Data
    absl::InlinedVector<const ParamSet *, 4> paramSets;
    std::map<std::string, std::shared_ptr<Texture<Float>>> &floatTextures;
    std::map<std::string, std::shared_ptr<Texture<Spectrum>>> &spectrumTextures;
};

}  // namespace pbrt

#endif  // PBRT_CORE_PARAMSET_H
