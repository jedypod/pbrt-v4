
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


// core/paramset.cpp*
#include "paramset.h"

#include "error.h"
#include "fileutil.h"
#include "floatfile.h"
#include "textures/constant.h"
#include <glog/logging.h>

using gtl::ArraySlice;

namespace pbrt {

template <typename T> static void
addParam(const std::string &name, std::unique_ptr<T[]> values, int nValues,
         std::vector<ParamSetItem<T>> &vec) {
    for (auto &v : vec) {
        if (v.name == name) {
            Warning("%s: parameter redefined", name.c_str());
            v.values = std::move(values);
            v.nValues = nValues;
            return;
        }
    }
    vec.push_back(ParamSetItem<T>(name, std::move(values), nValues));
}

template <typename T> static ArraySlice<T>
lookupPtr(const std::string &name,
          const std::vector<ParamSetItem<T>> &vec) {
    for (const auto &v : vec)
        if (v.name == name) {
            v.lookedUp = true;
            return ArraySlice<T>(v.values.get(), v.nValues);
        }
    return {};
}

template <typename T> static T
lookupOne(const std::string &name, T def,
          const std::vector<ParamSetItem<T>> &vec) {
    for (const auto &v : vec)
        if (v.name == name && v.nValues == 1) {
            v.lookedUp = true;
            return v.values[0];
        }
    return def;
}

// ParamSet Methods
void ParamSet::AddFloat(const std::string &name,
                        std::unique_ptr<Float[]> values, int nValues) {
    addParam(name, std::move(values), nValues, floats);
}

void ParamSet::AddInt(const std::string &name, std::unique_ptr<int[]> values,
                      int nValues) {
    addParam(name, std::move(values), nValues, ints);
}

void ParamSet::AddBool(const std::string &name, std::unique_ptr<bool[]> values,
                       int nValues) {
    addParam(name, std::move(values), nValues, bools);
}

void ParamSet::AddPoint2f(const std::string &name,
                          std::unique_ptr<Point2f[]> values, int nValues) {
    addParam(name, std::move(values), nValues, point2fs);
}

void ParamSet::AddVector2f(const std::string &name,
                           std::unique_ptr<Vector2f[]> values, int nValues) {
    addParam(name, std::move(values), nValues, vector2fs);
}

void ParamSet::AddPoint3f(const std::string &name,
                          std::unique_ptr<Point3f[]> values, int nValues) {
    addParam(name, std::move(values), nValues, point3fs);
}

void ParamSet::AddVector3f(const std::string &name,
                           std::unique_ptr<Vector3f[]> values, int nValues) {
    addParam(name, std::move(values), nValues, vector3fs);
}

void ParamSet::AddNormal3f(const std::string &name,
                           std::unique_ptr<Normal3f[]> values, int nValues) {
    addParam(name, std::move(values), nValues, normals);
}

void ParamSet::AddSpectrum(const std::string &name,
                           std::unique_ptr<Spectrum[]> values, int nValues) {
    addParam(name, std::move(values), nValues, spectra);
}

void ParamSet::AddString(const std::string &name,
                         std::unique_ptr<std::string[]> values, int nValues) {
    addParam(name, std::move(values), nValues, strings);
}

void ParamSet::AddTexture(const std::string &name, const std::string &value) {
    std::unique_ptr<std::string[]> str(new std::string[1]);
    str[0] = value;

    addParam(name, std::move(str), 1, textures);
}

Float ParamSet::FindOneFloat(const std::string &name, Float def) const {
    return lookupOne(name, def, floats);
}

ArraySlice<Float> ParamSet::FindFloat(const std::string &name) const {
    return lookupPtr(name, floats);
}

ArraySlice<int> ParamSet::FindInt(const std::string &name) const {
    return lookupPtr(name, ints);
}

ArraySlice<bool> ParamSet::FindBool(const std::string &name) const {
    return lookupPtr(name, bools);
}

int ParamSet::FindOneInt(const std::string &name, int def) const {
    return lookupOne(name, def, ints);
}

bool ParamSet::FindOneBool(const std::string &name, bool def) const {
    return lookupOne(name, def, bools);
}

ArraySlice<Point2f> ParamSet::FindPoint2f(const std::string &name) const {
    return lookupPtr(name, point2fs);
}

Point2f ParamSet::FindOnePoint2f(const std::string &name,
                                 const Point2f &def) const {
    return lookupOne(name, def, point2fs);
}

ArraySlice<Vector2f> ParamSet::FindVector2f(const std::string &name) const {
    return lookupPtr(name, vector2fs);
}

Vector2f ParamSet::FindOneVector2f(const std::string &name,
                                   const Vector2f &def) const {
    return lookupOne(name, def, vector2fs);
}

ArraySlice<Point3f> ParamSet::FindPoint3f(const std::string &name) const {
    return lookupPtr(name, point3fs);
}

Point3f ParamSet::FindOnePoint3f(const std::string &name,
                                 const Point3f &def) const {
    return lookupOne(name, def, point3fs);
}

ArraySlice<Vector3f> ParamSet::FindVector3f(const std::string &name) const {
    return lookupPtr(name, vector3fs);
}

Vector3f ParamSet::FindOneVector3f(const std::string &name,
                                   const Vector3f &def) const {
    return lookupOne(name, def, vector3fs);
}

ArraySlice<Normal3f> ParamSet::FindNormal3f(const std::string &name) const {
    return lookupPtr(name, normals);
}

Normal3f ParamSet::FindOneNormal3f(const std::string &name,
                                   const Normal3f &def) const {
    return lookupOne(name, def, normals);
}

ArraySlice<Spectrum> ParamSet::FindSpectrum(const std::string &name) const {
    return lookupPtr(name, spectra);
}

Spectrum ParamSet::FindOneSpectrum(const std::string &name,
                                   const Spectrum &def) const {
    return lookupOne(name, def, spectra);
}

ArraySlice<std::string> ParamSet::FindString(const std::string &name) const {
    return lookupPtr(name, strings);
}

std::string ParamSet::FindOneString(const std::string &name,
                                    const std::string &def) const {
    return lookupOne(name, def, strings);
}

std::string ParamSet::FindOneFilename(const std::string &name,
                                      const std::string &def) const {
    std::string filename = FindOneString(name, "");
    if (filename == "") return def;
    return AbsolutePath(ResolveFilename(filename));
}

std::string ParamSet::FindTexture(const std::string &name) const {
    return lookupOne(name, std::string(""), textures);
}

template <typename T> static void
checkUnused(const std::vector<ParamSetItem<T>> &vec) {
    for (const auto &v : vec)
        if (!v.lookedUp)
            Warning("Parameter \"%s\" not used", v.name.c_str());
}

void ParamSet::ReportUnused() const {
    checkUnused(ints);
    checkUnused(bools);
    checkUnused(floats);
    checkUnused(point2fs);
    checkUnused(vector2fs);
    checkUnused(point3fs);
    checkUnused(vector3fs);
    checkUnused(normals);
    checkUnused(spectra);
    checkUnused(strings);
    checkUnused(textures);
}

static std::string toString(int v) { return std::to_string(v); }

static std::string toString(bool v) { return v ? "\"true\"" : "\"false\""; }

static std::string toString(Float f) {
    if ((int)f == f) return toString(int(f));
    return StringPrintf("%f", f);
}

static std::string toString(const Point2f &p) {
    return StringPrintf("%f %f", p.x, p.y);
}

static std::string toString(const Vector2f &v) {
    return StringPrintf("%f %f", v.x, v.y);
}

static std::string toString(const Point3f &p) {
    return StringPrintf("%f %f %f", p.x, p.y, p.z);
}

static std::string toString(const Vector3f &v) {
    return StringPrintf("%f %f %f", v.x, v.y, v.z);
}

static std::string toString(const Normal3f &n) {
    return StringPrintf("%f %f %f", n.x, n.y, n.z);
}

static std::string toString(const std::string &s) {
    return '\"' + s + "\"";
}

static std::string toString(const Spectrum &s) {
    Float rgb[3];
    s.ToRGB(rgb);
    return StringPrintf("%f %f %f", rgb[0], rgb[1], rgb[2]);
}


template <typename T> static std::string
toString(const char *type, int indent, bool first,
         const std::vector<ParamSetItem<T>> &vec) {
    std::string ret;
    for (const auto &item : vec) {
        if (first) first = false;
        else {
            ret += '\n';
            ret.append(indent + 4, ' ');
        }

        ret += StringPrintf("\"%s %s\" [ ", type, item.name.c_str());
        for (int i = 0; i < item.nValues; ++i) {
            ret += toString(item.values[i]) + ' ';
        }
        ret += "] ";
    }
    return ret;
}

std::string ParamSet::ToString(int indent) const {
    std::string ret;
    ret += toString("integer", indent, ret.empty(), ints);
    ret += toString("bool", indent, ret.empty(), bools);
    ret += toString("float", indent, ret.empty(), floats);
    ret += toString("point2", indent, ret.empty(), point2fs);
    ret += toString("vector2", indent, ret.empty(), vector2fs);
    ret += toString("point3", indent, ret.empty(), point3fs);
    ret += toString("vector3", indent, ret.empty(), vector3fs);
    ret += toString("normal", indent, ret.empty(), normals);
    ret += toString("string", indent, ret.empty(), strings);
    ret += toString("texture", indent, ret.empty(), textures);
    // FIXME: this downsamples everything into RGB
    ret += toString("rgb", indent, ret.empty(), spectra);
    return ret;
}

// TextureParams Method Definitions
std::shared_ptr<Texture<Spectrum>> TextureParams::GetSpectrumTexture(
    const std::string &name, const Spectrum &def) const {
    auto tex = GetSpectrumTextureOrNull(name);
    if (tex) return tex;

    Spectrum val = FindOneSpectrum(name, def);
    return std::make_shared<ConstantTexture<Spectrum>>(val);
}

std::shared_ptr<Texture<Spectrum>> TextureParams::GetSpectrumTextureOrNull(
    const std::string &n) const {
    std::string name = FindTexture(n);
    if (name != "") {
        if (spectrumTextures.find(name) != spectrumTextures.end())
            return spectrumTextures[name];
        else {
            Error(
                "Couldn't find spectrum texture named \"%s\" for parameter \"%s\"",
                name.c_str(), n.c_str());
            return nullptr;
        }
    }
    ArraySlice<Spectrum> val = FindSpectrum(n);
    if (!val.empty()) return std::make_shared<ConstantTexture<Spectrum>>(val[0]);
    return nullptr;
}
std::shared_ptr<Texture<Float>> TextureParams::GetFloatTexture(
    const std::string &name, Float def) const {
    auto tex = GetFloatTextureOrNull(name);
    if (tex) return tex;

    Float val = FindOneFloat(name, def);
    return std::make_shared<ConstantTexture<Float>>(val);
}

std::shared_ptr<Texture<Float>> TextureParams::GetFloatTextureOrNull(
    const std::string &n) const {
    std::string name = FindTexture(n);
    if (name != "") {
        if (floatTextures.find(name) != floatTextures.end())
            return floatTextures[name];
        else {
            Error(
                "Couldn't find float texture named \"%s\" for parameter \"%s\"",
                name.c_str(), n.c_str());
            return nullptr;
        }
    }

    ArraySlice<Float> val = FindFloat(n);
    if (!val.empty()) return std::make_shared<ConstantTexture<Float>>(val[0]);
    return nullptr;
}

}  // namespace pbrt
