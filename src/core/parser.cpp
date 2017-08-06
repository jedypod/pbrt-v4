
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

// core/parser.cpp*
#include "parser.h"

#include "error.h"
#include "fileutil.h"
#include "floatfile.h"
#include "paramset.h"

#include <glog/logging.h>

#include <stdio.h>
#include <functional>
#include <memory>

extern FILE *yyin;
extern int yyparse(void);
extern int yydebug;

namespace pbrt {

namespace parse {

std::unique_ptr<MemoryPool<std::string>> stringPool;
std::unique_ptr<MemoryPool<ParameterAndValues>> paramArrayPool;

} // namespace parse

// Parsing Global Interface
bool ParseFile(const std::string &filename) {
    CHECK(!parse::stringPool && !parse::paramArrayPool);
    parse::stringPool = std::make_unique<MemoryPool<std::string>>(
        [](std::string *str) { str->clear(); });
    parse::paramArrayPool = std::make_unique<MemoryPool<parse::ParameterAndValues>>(
        [](parse::ParameterAndValues *pa) {
            pa->name = nullptr;
            pa->next = nullptr;

            size_t cap = pa->numbers.capacity();
            pa->numbers.clear();
            // Part of why we want to reuse these is to reuse the internal
            // allocations done by the vectors. The C++ standard doesn't
            // require that clear() maintain the current capacity, but
            // current implementations seem to do this. Verify that this
            // remains so (and figure out what to do about it if this ever
            // starts to hit...)
            CHECK_EQ(cap, pa->numbers.capacity());

            pa->strings.clear();
            pa->bools.clear();
        });

    LOG(INFO) << "Starting to parse input file " << filename;

    if (getenv("PBRT_YYDEBUG") != nullptr) yydebug = 1;

    if (filename == "-")
        yyin = stdin;
    else {
        yyin = fopen(filename.c_str(), "r");
        SetSearchDirectory(DirectoryContaining(filename));
    }
    if (yyin != nullptr) {
        parse::currentFilename = filename;
        if (yyin == stdin) parse::currentFilename = "<standard input>";
        parse::currentLineNumber = 1;
        yyparse();
        if (yyin != stdin) fclose(yyin);
    }
    parse::currentFilename = "";
    parse::currentLineNumber = 0;
    LOG(INFO) << "Done parsing input file " << filename;

    parse::stringPool = nullptr;
    parse::paramArrayPool = nullptr;

    return (yyin != nullptr);
}

///////////////////////////////////////////////////////////////////////////
// ParameterAndValues Method Definitions

namespace parse {

void ParameterAndValues::AddNumber(double d) {
    if (strings.size() || bools.size())
        Error("Ignoring number \"%f\" in non-numeric parameter list", d);
    else
        numbers.push_back(d);
}

void ParameterAndValues::AddString(std::string *str) {
    if (numbers.size() || bools.size())
        Error("Ignoring string \"%s\" in non-string parameter list",
              str->c_str());
    else
        strings.push_back(str);
}

void ParameterAndValues::AddBool(bool v) {
    if (numbers.size() || strings.size())
        Error("Ignoring bool \"%s\" in non-bool parameter list",
              v ? "true" : "false");
    else
        bools.push_back(v);
}

} // namespace parse

///////////////////////////////////////////////////////////////////////////

static void initBoolParameter(const std::vector<bool> &bools,
                              const std::vector<std::string *> &strings,
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
template <typename T>
static void initNumericParameter(int n, const std::vector<double> &numbers,
                                 const std::string &name,
                                 std::function<T(const double *)> convert,
                                 std::function<void(std::vector<T>)> add) {
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

ParamSet ParseParameters(const parse::ParameterAndValues *parameters,
                         SpectrumType spectrumType) {
    ParamSet ps;
    for (const parse::ParameterAndValues *param = parameters; param; param = param->next) {
        if (param->name->find(' ') == std::string::npos) {
            Error("Syntax error in parameter name \"%s\"", param->name->c_str());
            continue;
        }
        std::string type = param->name->substr(0, param->name->find(' '));
        std::string name = param->name->substr(param->name->rfind(' ') + 1);

        if (type == "integer") {
            initNumericParameter<int>(
                1, param->numbers, *param->name,
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
                            "Floating-point value %f will be rounted to an "
                            "integer",
                            *v);

                    return int(Clamp(*v, std::numeric_limits<int>::lowest(),
                                     std::numeric_limits<int>::max()));
                },
                [&ps, &name](std::vector<int> v) {
                    ps.AddInt(name, std::move(v));
                });
        } else if (type == "float") {
            initNumericParameter<Float>(
                1, param->numbers, *param->name,
                [](const double *v) { return Float(*v); },
                [&ps, &name](std::vector<Float> v) {
                    ps.AddFloat(name, std::move(v));
                });
        } else if (type == "bool") {
            initBoolParameter(param->bools, param->strings, *param->name, name, &ps);
        } else if (type == "point2") {
            initNumericParameter<Point2f>(
                2, param->numbers, *param->name,
                [](const double *v) { return Point2f(v[0], v[1]); },
                [&ps, &name](std::vector<Point2f> v) {
                    ps.AddPoint2f(name, std::move(v));
                });
        } else if (type == "vector2") {
            initNumericParameter<Vector2f>(
                2, param->numbers, *param->name,
                [](const double *v) { return Vector2f(v[0], v[1]); },
                [&ps, &name](std::vector<Vector2f> v) {
                    ps.AddVector2f(name, std::move(v));
                });
        } else if (type == "point" || type == "point3") {
            initNumericParameter<Point3f>(
                3, param->numbers, *param->name,
                [](const double *v) { return Point3f(v[0], v[1], v[2]); },
                [&ps, &name](std::vector<Point3f> v) {
                    ps.AddPoint3f(name, std::move(v));
                });
        } else if (type == "vector" || type == "vector3") {
            initNumericParameter<Vector3f>(
                3, param->numbers, *param->name,
                [](const double *v) { return Vector3f(v[0], v[1], v[2]); },
                [&ps, &name](std::vector<Vector3f> v) {
                    ps.AddVector3f(name, std::move(v));
                });
        } else if (type == "normal") {
            initNumericParameter<Normal3f>(
                3, param->numbers, *param->name,
                [](const double *v) { return Normal3f(v[0], v[1], v[2]); },
                [&ps, &name](std::vector<Normal3f> v) {
                    ps.AddNormal3f(name, std::move(v));
                });
        } else if (type == "color" || type == "rgb") {
            initNumericParameter<Spectrum>(
                3, param->numbers, *param->name,
                [spectrumType](const double *v) -> Spectrum {
                    Float rgb[3] = {Float(v[0]), Float(v[1]), Float(v[2])};
                    return Spectrum::FromRGB(rgb, spectrumType);
                },
                [&ps, &name](std::vector<Spectrum> v) {
                    ps.AddSpectrum(name, std::move(v));
                });
        } else if (type == "xyz") {
            initNumericParameter<Spectrum>(
                3, param->numbers, *param->name,
                [spectrumType](const double *v) -> Spectrum {
                    Float xyz[3] = {Float(v[0]), Float(v[1]), Float(v[2])};
                    return Spectrum::FromXYZ(xyz, spectrumType);
                },
                [&ps, &name](std::vector<Spectrum> v) {
                    ps.AddSpectrum(name, std::move(v));
                });
        } else if (type == "blackbody") {
            std::vector<Float> values(nCIESamples);
            initNumericParameter<Spectrum>(
                2, param->numbers, *param->name,
                [&values](const double *v) -> Spectrum {
                    Float T = v[0], scale = v[1];
                    BlackbodyNormalized(CIE_lambda, T, &values);
                    return scale * Spectrum::FromSampled(CIE_lambda, values);
                },
                [&ps, &name](std::vector<Spectrum> v) {
                    ps.AddSpectrum(name, std::move(v));
                });
        } else if (type == "spectrum") {
            if (param->numbers.size()) {
                if (param->numbers.size() % 2 != 0) {
                    Error("Found odd number of values for \"%s\"",
                          param->name->c_str());
                    continue;
                }
                int nSamples = param->numbers.size() / 2;
                std::vector<Float> lambda(nSamples), value(nSamples);
                for (int i = 0; i < nSamples; ++i) {
                    lambda[i] = param->numbers[2 * i];
                    value[i] = param->numbers[2 * i + 1];
                }
                std::vector<Spectrum> spectrum(1);
                spectrum[0] = Spectrum::FromSampled(lambda, value);
                ps.AddSpectrum(name, std::move(spectrum));
            } else if (param->strings.size()) {
                std::vector<Spectrum> values;
                for (const std::string *filename : param->strings)
                    values.push_back(readSpectrumFromFile(*filename));
                ps.AddSpectrum(name, std::move(values));
            } else {
                Error("No values found for \"%s\"", param->name->c_str());
            }
        } else if (type == "string") {
            if (param->strings.empty())
                Error("No strings provided for \"%s\"", param->name->c_str());
            else {
                std::vector<std::string> strings;
                for (const std::string *s : param->strings)
                    strings.push_back(*s);
                ps.AddString(name, std::move(strings));
            }
        } else if (type == "texture") {
            if (param->strings.size() != 1)
                Error("Expecting a single string for \"%s\"",
                      param->name->c_str());
            else
                ps.AddTexture(name, *param->strings[0]);
        } else
            Error("Unexpected parameter type \"%s\"", type.c_str());
    }
    return ps;
}

}  // namespace pbrt
