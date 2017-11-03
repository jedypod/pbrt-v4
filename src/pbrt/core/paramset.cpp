
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
#include <pbrt/core/paramset.h>

#include <pbrt/core/error.h>
#include <pbrt/util/fileutil.h>
#include <pbrt/util/floatfile.h>
#include <pbrt/util/stringprint.h>
#include <pbrt/textures/constant.h>

#include <glog/logging.h>


namespace pbrt {

///////////////////////////////////////////////////////////////////////////
// NamedValues

void NamedValues::AddNumber(double d) {
    if (strings.size() || bools.size())
        Error("Ignoring number \"%f\" in non-numeric parameter list", d);
    else
        numbers.push_back(d);
}

void NamedValues::AddString(const std::string *str) {
    if (numbers.size() || bools.size())
        Error("Ignoring string \"%s\" in non-string parameter list",
              str->c_str());
    else
        strings.push_back(str);
}

void NamedValues::AddBool(bool v) {
    if (numbers.size() || strings.size())
        Error("Ignoring bool \"%s\" in non-bool parameter list",
              v ? "true" : "false");
    else
        bools.push_back(v);
}


std::string NamedValues::ToString() const {
    CHECK_NOTNULL(name);
    std::string str = std::string("\"") + *name + std::string("\" [");
    if (!numbers.empty())
        for (double d : numbers)
            str += StringPrintf("%f ", d);
    else if (!strings.empty())
        for (const std::string *s : strings)
            str += *s + " ";
    else if (!bools.empty())
        for (bool b : bools)
            str += b ? "true " : "false ";
    str += "] ";

    if (next) str += " " + next->ToString();
    return str;
}

/////////////////////////////////////////////////////////////////////////////////////
// ParamSet

template <typename T>
static void addParam(const std::string &name, std::vector<T> values,
                     std::vector<ParamSetItem<T>> &vec) {
    for (auto &v : vec) {
        if (v.name == name) {
            //Warning("%s: parameter redefined", name.c_str());
            v.values = std::move(values);
            return;
        }
    }
    vec.push_back(ParamSetItem<T>(name, std::move(values)));
}

template <typename T>
static absl::Span<const T> lookupPtr(const std::string &name,
                               const std::vector<ParamSetItem<T>> &vec) {
    for (const auto &v : vec)
        if (v.name == name) {
            v.lookedUp = true;
            return absl::Span<const T>(v.values);
        }
    return {};
}

template <typename T>
static T lookupOne(const std::string &name, T def,
                   const std::vector<ParamSetItem<T>> &vec) {
    for (const auto &v : vec)
        if (v.name == name && v.values.size() == 1) {
            v.lookedUp = true;
            return v.values[0];
        }
    return def;
}

// ParamSet Methods
void ParamSet::AddFloat(const std::string &name,
                        std::vector<Float> values) {
    addParam(name, std::move(values), floats);
}

void ParamSet::AddInt(const std::string &name, std::vector<int> values) {
    addParam(name, std::move(values), ints);
}

void ParamSet::AddBool(const std::string &name, std::vector<uint8_t> values) {
    addParam(name, std::move(values), bools);
}

void ParamSet::AddPoint2f(const std::string &name,
                          std::vector<Point2f> values) {
    addParam(name, std::move(values), point2fs);
}

void ParamSet::AddVector2f(const std::string &name,
                           std::vector<Vector2f> values) {
    addParam(name, std::move(values), vector2fs);
}

void ParamSet::AddPoint3f(const std::string &name,
                          std::vector<Point3f> values) {
    addParam(name, std::move(values), point3fs);
}

void ParamSet::AddVector3f(const std::string &name,
                           std::vector<Vector3f> values) {
    addParam(name, std::move(values), vector3fs);
}

void ParamSet::AddNormal3f(const std::string &name,
                           std::vector<Normal3f> values) {
    addParam(name, std::move(values), normals);
}

void ParamSet::AddSpectrum(const std::string &name,
                           std::vector<Spectrum> values) {
    addParam(name, std::move(values), spectra);
}

void ParamSet::AddString(const std::string &name,
                         std::vector<std::string> values) {
    addParam(name, std::move(values), strings);
}

void ParamSet::AddTexture(const std::string &name, const std::string &value) {
    std::vector<std::string> str(1);
    str[0] = value;

    addParam(name, std::move(str), textures);
}

Float ParamSet::GetOneFloat(const std::string &name, Float def) const {
    return lookupOne(name, def, floats);
}

absl::Span<const Float> ParamSet::GetFloatArray(const std::string &name) const {
    return lookupPtr(name, floats);
}

absl::Span<const int> ParamSet::GetIntArray(const std::string &name) const {
    return lookupPtr(name, ints);
}

absl::Span<const uint8_t> ParamSet::GetBoolArray(const std::string &name) const {
    return lookupPtr(name, bools);
}

int ParamSet::GetOneInt(const std::string &name, int def) const {
    return lookupOne(name, def, ints);
}

bool ParamSet::GetOneBool(const std::string &name, bool def) const {
    return lookupOne(name, uint8_t(def), bools);
}

absl::Span<const Point2f> ParamSet::GetPoint2fArray(const std::string &name) const {
    return lookupPtr(name, point2fs);
}

Point2f ParamSet::GetOnePoint2f(const std::string &name,
                                const Point2f &def) const {
    return lookupOne(name, def, point2fs);
}

absl::Span<const Vector2f> ParamSet::GetVector2fArray(const std::string &name) const {
    return lookupPtr(name, vector2fs);
}

Vector2f ParamSet::GetOneVector2f(const std::string &name,
                                  const Vector2f &def) const {
    return lookupOne(name, def, vector2fs);
}

absl::Span<const Point3f> ParamSet::GetPoint3fArray(const std::string &name) const {
    return lookupPtr(name, point3fs);
}

Point3f ParamSet::GetOnePoint3f(const std::string &name,
                                const Point3f &def) const {
    return lookupOne(name, def, point3fs);
}

absl::Span<const Vector3f> ParamSet::GetVector3fArray(const std::string &name) const {
    return lookupPtr(name, vector3fs);
}

Vector3f ParamSet::GetOneVector3f(const std::string &name,
                                  const Vector3f &def) const {
    return lookupOne(name, def, vector3fs);
}

absl::Span<const Normal3f> ParamSet::GetNormal3fArray(const std::string &name) const {
    return lookupPtr(name, normals);
}

Normal3f ParamSet::GetOneNormal3f(const std::string &name,
                                  const Normal3f &def) const {
    return lookupOne(name, def, normals);
}

absl::Span<const Spectrum> ParamSet::GetSpectrumArray(const std::string &name) const {
    return lookupPtr(name, spectra);
}

Spectrum ParamSet::GetOneSpectrum(const std::string &name,
                                  const Spectrum &def) const {
    return lookupOne(name, def, spectra);
}

absl::Span<const std::string> ParamSet::GetStringArray(
    const std::string &name) const {
    return lookupPtr(name, strings);
}

std::string ParamSet::GetOneString(const std::string &name,
                                   const std::string &def) const {
    return lookupOne(name, def, strings);
}

std::string ParamSet::GetOneFilename(const std::string &name,
                                     const std::string &def) const {
    std::string filename = GetOneString(name, "");
    if (filename == "") return def;
    return AbsolutePath(ResolveFilename(filename));
}

std::string ParamSet::FindTexture(const std::string &name) const {
    return lookupOne(name, std::string(""), textures);
}

template <typename T>
static void checkUnused(const std::vector<ParamSetItem<T>> &vec) {
    for (const auto &v : vec)
        if (!v.lookedUp) Warning("Parameter \"%s\" not used", v.name.c_str());
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

///////////////////////////////////////////////////////////////////////////

static void initBoolParameter(const std::vector<bool> &bools,
                              const std::vector<const std::string *> &strings,
                              const std::string &fullname,
                              const std::string &name, ParamSet *ps) {
    if (!bools.empty()) {
        std::vector<uint8_t> values(bools.size());
        std::copy(bools.begin(), bools.end(), values.begin());
        ps->AddBool(name, std::move(values));
    } else if (!strings.empty()) {
        // Legacy string-based bools
        std::vector<uint8_t> values;
        for (const std::string *s : strings) {
            if (*s == "true")
                values.push_back(true);
            else if (*s == "false")
                values.push_back(false);
            else
                Warning("Ignoring non \"true\"/\"false\" bool value \"%s\"",
                        s->c_str());
        }
        ps->AddBool(name, std::move(values));
    } else
        Error("No bool values provided for \"%s\"", fullname.c_str());
}

// General numeric parameter array extraction function.  Given the vector
// of values, |numbers|, provide pointers to every |n|th of them in turn to
// the given |convert| function, which is responsible for converting them
// to the desired type T.  These T values are accumulated into a vector,
// which is then passed to the |add| function after all of the original
// array values have been processed.
template <typename T, typename C, typename A>
static void initNumericParameter(int n, const std::vector<double> &numbers,
                                 const std::string &name, C convert, A add) {
    if (numbers.empty()) {
        Error("No numeric values provided for \"%s\"", name.c_str());
        return;
    }
    if (numbers.size() % n != 0) {
        Error("Number of values provided for \"%s\" not a multiple of %d",
              name.c_str(), n);
        return;
    }

    std::vector<T> values(numbers.size() / n);
    for (size_t i = 0; i < values.size(); ++i)
        values[i] = convert(&numbers[n * i]);

    add(std::move(values));
}

static std::map<std::string, Spectrum> cachedSpectra;

// TODO: move this functionality (but not the caching?) to a Spectrum method.
Spectrum readSpectrumFromFile(const std::string &filename) {
    std::string fn = AbsolutePath(ResolveFilename(filename));
    if (cachedSpectra.find(fn) != cachedSpectra.end()) return cachedSpectra[fn];

    std::vector<Float> vals;
    Spectrum s;
    if (!ReadFloatFile(fn.c_str(), &vals)) {
        Warning("Unable to read SPD file \"%s\".  Using black distribution.",
                fn.c_str());
        s = Spectrum(0.);
    } else {
        if (vals.size() % 2 != 0) {
            Warning("Extra value found in spectrum file \"%s\". Ignoring it.",
                    fn.c_str());
        }
        std::vector<Float> lambda, v;
        for (size_t i = 0; i < vals.size() / 2; ++i) {
            lambda.push_back(vals[2 * i]);
            v.push_back(vals[2 * i + 1]);
        }
        s = Spectrum::FromSampled(lambda, v);
    }
    cachedSpectra[fn] = s;
    return s;
}

void ParamSet::Parse(const NamedValues *namedValues,
                     SpectrumType spectrumType) {
    for (const NamedValues *nv = namedValues; nv; nv = nv->next) {
        if (nv->name->find(' ') == std::string::npos) {
            Error("Syntax error in parameter name \"%s\"", nv->name->c_str());
            continue;
        }
        std::string type = nv->name->substr(0, nv->name->find(' '));
        std::string name = nv->name->substr(nv->name->rfind(' ') + 1);

        if (type == "integer") {
            initNumericParameter<int>(
                1, nv->numbers, *nv->name,
                [](const double *v) {
                    if (*v > std::numeric_limits<int>::max())
                        Warning(
                            "Numeric value %f too large to represent as an "
                            "integer. Clamping to %d",
                            *v, std::numeric_limits<int>::max());
                    else if (*v < std::numeric_limits<int>::lowest())
                        Warning(
                            "Numeric value %f too low to represent as an "
                            "integer. Clamping to %d",
                            *v, std::numeric_limits<int>::lowest());
                    else if (double(int(*v)) != *v)
                        Warning(
                            "Floating-point value %f will be rounded to an "
                            "integer",
                            *v);

                    return int(Clamp(*v, std::numeric_limits<int>::lowest(),
                                     std::numeric_limits<int>::max()));
                },
                [this, &name](std::vector<int> v) {
                    AddInt(name, std::move(v));
                });
        } else if (type == "float") {
            initNumericParameter<Float>(
                1, nv->numbers, *nv->name,
                [](const double *v) { return Float(*v); },
                [this, &name](std::vector<Float> v) {
                    AddFloat(name, std::move(v));
                });
        } else if (type == "bool") {
            initBoolParameter(nv->bools, nv->strings, *nv->name, name, this);
        } else if (type == "point2") {
            initNumericParameter<Point2f>(
                2, nv->numbers, *nv->name,
                [](const double *v) { return Point2f(v[0], v[1]); },
                [this, &name](std::vector<Point2f> v) {
                    AddPoint2f(name, std::move(v));
                });
        } else if (type == "vector2") {
            initNumericParameter<Vector2f>(
                2, nv->numbers, *nv->name,
                [](const double *v) { return Vector2f(v[0], v[1]); },
                [this, &name](std::vector<Vector2f> v) {
                    AddVector2f(name, std::move(v));
                });
        } else if (type == "point" || type == "point3") {
            initNumericParameter<Point3f>(
                3, nv->numbers, *nv->name,
                [](const double *v) { return Point3f(v[0], v[1], v[2]); },
                [this, &name](std::vector<Point3f> v) {
                    AddPoint3f(name, std::move(v));
                });
        } else if (type == "vector" || type == "vector3") {
            initNumericParameter<Vector3f>(
                3, nv->numbers, *nv->name,
                [](const double *v) { return Vector3f(v[0], v[1], v[2]); },
                [this, &name](std::vector<Vector3f> v) {
                    AddVector3f(name, std::move(v));
                });
        } else if (type == "normal") {
            initNumericParameter<Normal3f>(
                3, nv->numbers, *nv->name,
                [](const double *v) { return Normal3f(v[0], v[1], v[2]); },
                [this, &name](std::vector<Normal3f> v) {
                    AddNormal3f(name, std::move(v));
                });
        } else if (type == "color" || type == "rgb") {
            initNumericParameter<Spectrum>(
                3, nv->numbers, *nv->name,
                [spectrumType](const double *v) -> Spectrum {
                    Float rgb[3] = {Float(v[0]), Float(v[1]), Float(v[2])};
                    return Spectrum::FromRGB(rgb, spectrumType);
                },
                [this, &name](std::vector<Spectrum> v) {
                    AddSpectrum(name, std::move(v));
                });
        } else if (type == "xyz") {
            initNumericParameter<Spectrum>(
                3, nv->numbers, *nv->name,
                [spectrumType](const double *v) -> Spectrum {
                    Float xyz[3] = {Float(v[0]), Float(v[1]), Float(v[2])};
                    return Spectrum::FromXYZ(xyz, spectrumType);
                },
                [this, &name](std::vector<Spectrum> v) {
                    AddSpectrum(name, std::move(v));
                });
        } else if (type == "blackbody") {
            std::vector<Float> values(nCIESamples);
            initNumericParameter<Spectrum>(
                2, nv->numbers, *nv->name,
                [&values](const double *v) -> Spectrum {
                    Float T = v[0], scale = v[1];
                    BlackbodyNormalized(CIE_lambda, T, absl::MakeSpan(values));
                    return scale * Spectrum::FromSampled(CIE_lambda, values);
                },
                [this, &name](std::vector<Spectrum> v) {
                    AddSpectrum(name, std::move(v));
                });
        } else if (type == "spectrum") {
            if (nv->numbers.size()) {
                if (nv->numbers.size() % 2 != 0) {
                    Error("Found odd number of values for \"%s\"",
                          nv->name->c_str());
                    continue;
                }
                int nSamples = nv->numbers.size() / 2;
                std::vector<Float> lambda(nSamples), value(nSamples);
                for (int i = 0; i < nSamples; ++i) {
                    lambda[i] = nv->numbers[2 * i];
                    value[i] = nv->numbers[2 * i + 1];
                }
                std::vector<Spectrum> spectrum(1);
                spectrum[0] = Spectrum::FromSampled(lambda, value);
                AddSpectrum(name, std::move(spectrum));
            } else if (nv->strings.size()) {
                std::vector<Spectrum> values;
                for (const std::string *filename : nv->strings)
                    values.push_back(readSpectrumFromFile(*filename));
                AddSpectrum(name, std::move(values));
            } else {
                Error("No values found for \"%s\"", nv->name->c_str());
            }
        } else if (type == "string") {
            if (nv->strings.empty())
                Error("No strings provided for \"%s\"", nv->name->c_str());
            else {
                std::vector<std::string> strings;
                for (const std::string *s : nv->strings)
                    strings.push_back(*s);
                AddString(name, std::move(strings));
            }
        } else if (type == "texture") {
            if (nv->strings.size() != 1)
                Error("Expecting a single string for \"%s\"",
                      nv->name->c_str());
            else
                AddTexture(name, *nv->strings[0]);
        } else
            Error("Unexpected parameter type \"%s\"", type.c_str());
    }
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

static std::string toString(const std::string &s) { return '\"' + s + "\""; }

static std::string toString(const Spectrum &s) {
    std::array<Float, 3> rgb = s.ToRGB();
    return StringPrintf("%f %f %f", rgb[0], rgb[1], rgb[2]);
}

template <typename T>
static std::string toString(const char *type, int indent, bool first,
                            const std::vector<ParamSetItem<T>> &vec) {
    std::string ret;
    for (const auto &item : vec) {
        if (first)
            first = false;
        else {
            ret += '\n';
            ret.append(indent + 4, ' ');
        }

        ret += StringPrintf("\"%s %s\" [ ", type, item.name.c_str());
        for (const auto &val : item.values)
            ret += toString(val) + ' ';
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

///////////////////////////////////////////////////////////////////////////
// TextureParams Method Definitions

std::shared_ptr<Texture<Spectrum>> TextureParams::GetSpectrumTexture(
    const std::string &name, const Spectrum &def) const {
    auto tex = GetSpectrumTextureOrNull(name);
    if (tex) return tex;
    return std::make_shared<ConstantTexture<Spectrum>>(def);
}

std::shared_ptr<Texture<Spectrum>> TextureParams::GetSpectrumTextureOrNull(
    const std::string &n) const {
    std::string name = FindTexture(n);
    if (name != "") {
        if (spectrumTextures.find(name) != spectrumTextures.end())
            return spectrumTextures[name];
        else {
            Error(
                "Couldn't find spectrum texture named \"%s\" for parameter "
                "\"%s\"",
                name.c_str(), n.c_str());
            return nullptr;
        }
    }
    absl::Span<const Spectrum> val = GetSpectrumArray(n);
    if (!val.empty())
        return std::make_shared<ConstantTexture<Spectrum>>(val[0]);
    return nullptr;
}

std::shared_ptr<Texture<Float>> TextureParams::GetFloatTexture(
    const std::string &name, Float def) const {
    auto tex = GetFloatTextureOrNull(name);
    if (tex) return tex;
    return std::make_shared<ConstantTexture<Float>>(def);
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

    absl::Span<const Float> val = GetFloatArray(n);
    if (!val.empty()) return std::make_shared<ConstantTexture<Float>>(val[0]);
    return nullptr;
}

}  // namespace pbrt
