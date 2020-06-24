
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

#ifndef PBRT_CORE_PARAMDICT_H
#define PBRT_CORE_PARAMDICT_H

// core/paramdict.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/error.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/parser.h>
#include <pbrt/util/vecmath.h>
#include <pbrt/util/containers.h>

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace pbrt {

class ParsedParameter {
public:
    ParsedParameter(MemoryArena *arena, FileLoc loc)
        : loc(loc),
          numbers(ArenaAllocator<double>(arena)),
          strings(ArenaAllocator<std::string>(arena)),
          bools(ArenaAllocator<uint8_t>(arena)) {}

    void AddNumber(double d);
    void AddString(pstd::string_view str);
    void AddBool(bool v);

    std::string ToString() const;

    // TODO: We're trusting the short string optimization. Could also use
    // pstd::string_views...
    std::string type, name;

    FileLoc loc;
    mutable bool lookedUp = false;
    // If set, overrides the one given to the ParameterDictionary constructor...
    mutable const RGBColorSpace *colorSpace = nullptr;
    bool mayBeUnused = false;

    std::vector<double, ArenaAllocator<double>> numbers;
    std::vector<std::string, ArenaAllocator<std::string>> strings;
    std::vector<uint8_t, ArenaAllocator<uint8_t>> bools;
};

using ParsedParameterVector = InlinedVector<ParsedParameter *, 8>;

enum class ParameterType {
    Boolean, Float, Integer, Point2f, Vector2f, Point3f, Vector3f, Normal3f,
    Spectrum, String, Texture
};

template <ParameterType PT> struct ParameterTypeTraits {};

class ParameterDictionary {
public:
    ParameterDictionary() = default;
    ParameterDictionary(ParsedParameterVector params,
                        const RGBColorSpace *colorSpace);
    ParameterDictionary(ParsedParameterVector params0, const ParsedParameterVector &params1,
                        const RGBColorSpace *colorSpace);

    Float GetOneFloat(const std::string &name, Float def) const;
    int GetOneInt(const std::string &name, int def) const;
    bool GetOneBool(const std::string &name, bool def) const;
    Point2f GetOnePoint2f(const std::string &name, const Point2f &def) const;
    Vector2f GetOneVector2f(const std::string &name, const Vector2f &def) const;
    Point3f GetOnePoint3f(const std::string &name, const Point3f &def) const;
    Vector3f GetOneVector3f(const std::string &name, const Vector3f &def) const;
    Normal3f GetOneNormal3f(const std::string &name, const Normal3f &def) const;
    SpectrumHandle GetOneSpectrum(const std::string &name, SpectrumHandle def,
                                  SpectrumType spectrumType, Allocator alloc) const;
    std::string GetOneString(const std::string &name, const std::string &def) const;
    std::string GetTexture(const std::string &name) const;

    std::vector<Float> GetFloatArray(const std::string &name) const;
    std::vector<int> GetIntArray(const std::string &name) const;
    std::vector<uint8_t> GetBoolArray(const std::string &name) const;
    std::vector<Point2f> GetPoint2fArray(const std::string &name) const;
    std::vector<Vector2f> GetVector2fArray(const std::string &name) const;
    std::vector<Point3f> GetPoint3fArray(const std::string &name) const;
    std::vector<Vector3f> GetVector3fArray(const std::string &name) const;
    std::vector<Normal3f> GetNormal3fArray(const std::string &name) const;
    std::vector<SpectrumHandle> GetSpectrumArray(const std::string &name,
                                                 SpectrumType spectrumType,
                                                 Allocator alloc) const;
    std::vector<std::string> GetStringArray(const std::string &name) const;


    // For --upgrade only
    pstd::optional<RGB> GetOneRGB(const std::string &name) const;
    void RemoveFloat(const std::string &);
    void RemoveInt(const std::string &);
    void RemoveBool(const std::string &);
    void RemovePoint2f(const std::string &);
    void RemoveVector2f(const std::string &);
    void RemovePoint3f(const std::string &);
    void RemoveVector3f(const std::string &);
    void RemoveNormal3f(const std::string &);
    void RemoveString(const std::string &);
    void RemoveTexture(const std::string &);
    void RemoveSpectrum(const std::string &);

    void RenameParameter(const std::string &before, const std::string &after);
    void RenameUsedTextures(const std::map<std::string, std::string> &m);

    void ReportUnused() const;

    const RGBColorSpace *ColorSpace() const { return colorSpace; }

    std::string ToParameterList(int indent = 0) const;
    std::string ToParameterDefinition(const std::string &) const;
    std::string ToString() const;

    // Returns true if this ParameterDictionary shadows anything represented in
    // ps::requestedParameters.
    bool ShadowsAny(const ParameterDictionary &dict) const;

    const FileLoc *loc(const std::string &) const;

private:
    friend class TextureParameterDictionary;

    ParsedParameterVector params;
    const RGBColorSpace *colorSpace = nullptr;

    static std::mutex requestedParametersMutex;
    mutable InlinedVector<std::pair<ParameterType, std::string>, 16> requestedParameters;

    template <ParameterType PT>
    typename ParameterTypeTraits<PT>::ReturnType lookupSingle(
        const std::string &name, typename ParameterTypeTraits<PT>::ReturnType defaultValue) const;

    template <ParameterType PT>
    std::vector<typename ParameterTypeTraits<PT>::ReturnType> lookupArray(const std::string &name) const;

    template <typename ReturnType, typename G, typename C>
    std::vector<ReturnType> lookupArray(const std::string &name, ParameterType type,
                                        const char *typeName, int nPerItem, G getValues,
                                        C convert) const;

    std::vector<SpectrumHandle> extractSpectrumArray(const ParsedParameter &param,
                                                     SpectrumType spectrumType, Allocator alloc) const;

    void remove(const std::string &name, const char *typeName);
    void checkParameterTypes();
    static std::string ToParameterDefinition(const ParsedParameter *p, int indentCount);
};

class TextureParameterDictionary {
public:
    TextureParameterDictionary(const ParameterDictionary *dict,
                               const std::map<std::string, FloatTextureHandle> *floatTextures,
                               const std::map<std::string, SpectrumTextureHandle> *spectrumTextures)
        : dict(dict), floatTextures(floatTextures), spectrumTextures(spectrumTextures) { }

    operator const ParameterDictionary &() const { return *dict; }

    Float GetOneFloat(const std::string &name, Float def) const;
    int GetOneInt(const std::string &name, int def) const;
    bool GetOneBool(const std::string &name, bool def) const;
    Point2f GetOnePoint2f(const std::string &name, const Point2f &def) const;
    Vector2f GetOneVector2f(const std::string &name, const Vector2f &def) const;
    Point3f GetOnePoint3f(const std::string &name, const Point3f &def) const;
    Vector3f GetOneVector3f(const std::string &name, const Vector3f &def) const;
    Normal3f GetOneNormal3f(const std::string &name, const Normal3f &def) const;
    SpectrumHandle GetOneSpectrum(const std::string &name, SpectrumHandle def,
                                   SpectrumType spectrumType, Allocator alloc) const;
    std::string GetOneString(const std::string &name, const std::string &def) const;

    std::vector<Float> GetFloatArray(const std::string &name) const;
    std::vector<int> GetIntArray(const std::string &name) const;
    std::vector<uint8_t> GetBoolArray(const std::string &name) const;
    std::vector<Point2f> GetPoint2fArray(const std::string &name) const;
    std::vector<Vector2f> GetVector2fArray(const std::string &name) const;
    std::vector<Point3f> GetPoint3fArray(const std::string &name) const;
    std::vector<Vector3f> GetVector3fArray(const std::string &name) const;
    std::vector<Normal3f> GetNormal3fArray(const std::string &name) const;
    std::vector<SpectrumHandle> GetSpectrumArray(const std::string &name,
                                                 SpectrumType spectrumType,
                                                 Allocator alloc) const;
    std::vector<std::string> GetStringArray(const std::string &name) const;

    SpectrumTextureHandle GetSpectrumTexture(const std::string &name,
                                             SpectrumHandle defaultValue,
                                             SpectrumType spectrumType,
                                             Allocator alloc) const;
    SpectrumTextureHandle GetSpectrumTextureOrNull(const std::string &name,
                                                   SpectrumType spectrumType,
                                                   Allocator alloc) const;
    FloatTextureHandle GetFloatTexture(const std::string &name,
                                       Float defaultValue, Allocator alloc) const;
    FloatTextureHandle GetFloatTextureOrNull(const std::string &name, Allocator alloc) const;

    void ReportUnused() const;

private:
    const ParameterDictionary *dict;
    const std::map<std::string, FloatTextureHandle> *floatTextures;
    const std::map<std::string, SpectrumTextureHandle> *spectrumTextures;
};

}  // namespace pbrt

#endif  // PBRT_CORE_PARAMDICT_H
