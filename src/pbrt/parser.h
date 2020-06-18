// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// It is licensed under the BSD license; see the file LICENSE.txt
// SPDX: BSD-3-Clause

#if defined(_MSC_VER)
#define NOMINMAX
#pragma once
#endif

#ifndef PBRT_CORE_PARSER_H
#define PBRT_CORE_PARSER_H

// core/parser.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/containers.h>
#include <pbrt/util/error.h>
#include <pbrt/util/pstd.h>

#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace pbrt {

class ParsedParameter {
  public:
    ParsedParameter(Allocator alloc, FileLoc loc)
        : loc(loc), numbers(alloc), strings(alloc), bools(alloc) {}

    void AddNumber(double d);
    void AddString(std::string_view str);
    void AddBool(bool v);

    std::string ToString() const;

    // TODO: We're trusting the short string optimization. Could also use
    // std::string_views...
    std::string type, name;

    FileLoc loc;
    mutable bool lookedUp = false;
    // If set, overrides the one given to the ParameterDictionary constructor...
    mutable const RGBColorSpace *colorSpace = nullptr;
    bool mayBeUnused = false;

    pstd::vector<double> numbers;
    pstd::vector<std::string> strings;
    pstd::vector<uint8_t> bools;
};

using ParsedParameterVector = InlinedVector<ParsedParameter *, 8>;

class SceneRepresentation {
  public:
    virtual ~SceneRepresentation();

    virtual void Option(const std::string &name, const std::string &value,
                        FileLoc loc) = 0;
    virtual void Identity(FileLoc loc) = 0;
    virtual void Translate(Float dx, Float dy, Float dz, FileLoc loc) = 0;
    virtual void Rotate(Float angle, Float ax, Float ay, Float az, FileLoc loc) = 0;
    virtual void Scale(Float sx, Float sy, Float sz, FileLoc loc) = 0;
    virtual void LookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz,
                        Float ux, Float uy, Float uz, FileLoc loc) = 0;
    virtual void ConcatTransform(Float transform[16], FileLoc loc) = 0;
    virtual void Transform(Float transform[16], FileLoc loc) = 0;
    virtual void CoordinateSystem(const std::string &, FileLoc loc) = 0;
    virtual void CoordSysTransform(const std::string &, FileLoc loc) = 0;
    virtual void ActiveTransformAll(FileLoc loc) = 0;
    virtual void ActiveTransformEndTime(FileLoc loc) = 0;
    virtual void ActiveTransformStartTime(FileLoc loc) = 0;
    virtual void TransformTimes(Float start, Float end, FileLoc loc) = 0;
    virtual void ColorSpace(const std::string &n, FileLoc loc) = 0;
    virtual void PixelFilter(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) = 0;
    virtual void Film(const std::string &type, ParsedParameterVector params,
                      FileLoc loc) = 0;
    virtual void Sampler(const std::string &name, ParsedParameterVector params,
                         FileLoc loc) = 0;
    virtual void Accelerator(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) = 0;
    virtual void Integrator(const std::string &name, ParsedParameterVector params,
                            FileLoc loc) = 0;
    virtual void Camera(const std::string &, ParsedParameterVector params,
                        FileLoc loc) = 0;
    virtual void MakeNamedMedium(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) = 0;
    virtual void MediumInterface(const std::string &insideName,
                                 const std::string &outsideName, FileLoc loc) = 0;
    virtual void WorldBegin(FileLoc loc) = 0;
    virtual void AttributeBegin(FileLoc loc) = 0;
    virtual void AttributeEnd(FileLoc loc) = 0;
    virtual void Attribute(const std::string &target, ParsedParameterVector params,
                           FileLoc loc) = 0;
    virtual void TransformBegin(FileLoc loc) = 0;
    virtual void TransformEnd(FileLoc loc) = 0;
    virtual void Texture(const std::string &name, const std::string &type,
                         const std::string &texname, ParsedParameterVector params,
                         FileLoc loc) = 0;
    virtual void Material(const std::string &name, ParsedParameterVector params,
                          FileLoc loc) = 0;
    virtual void MakeNamedMaterial(const std::string &name, ParsedParameterVector params,
                                   FileLoc loc) = 0;
    virtual void NamedMaterial(const std::string &name, FileLoc loc) = 0;
    virtual void LightSource(const std::string &name, ParsedParameterVector params,
                             FileLoc loc) = 0;
    virtual void AreaLightSource(const std::string &name, ParsedParameterVector params,
                                 FileLoc loc) = 0;
    virtual void Shape(const std::string &name, ParsedParameterVector params,
                       FileLoc loc) = 0;
    virtual void ReverseOrientation(FileLoc loc) = 0;
    virtual void ObjectBegin(const std::string &name, FileLoc loc) = 0;
    virtual void ObjectEnd(FileLoc loc) = 0;
    virtual void ObjectInstance(const std::string &name, FileLoc loc) = 0;

    virtual void EndOfFiles() = 0;

  protected:
    template <typename... Args>
    void ErrorExitDeferred(const char *fmt, Args &&... args) const {
        errorExit = true;
        Error(fmt, std::forward<Args>(args)...);
    }
    template <typename... Args>
    void ErrorExitDeferred(const FileLoc *loc, const char *fmt, Args &&... args) const {
        errorExit = true;
        Error(loc, fmt, std::forward<Args>(args)...);
    }

    mutable bool errorExit = false;
};

void ParseFiles(SceneRepresentation *scene, pstd::span<const std::string> filenames);
void ParseString(SceneRepresentation *scene, std::string str);

struct Token {
    Token() = default;
    Token(std::string_view token, FileLoc loc) : token(token), loc(loc) {}

    std::string ToString() const;

    std::string_view token;
    FileLoc loc;
};

// Tokenizer converts a single pbrt scene file into a series of tokens.
class Tokenizer {
  public:
    Tokenizer(std::string str,
              std::function<void(const char *, const FileLoc *)> errorCallback);
#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
    Tokenizer(void *ptr, size_t len, std::string filename,
              std::function<void(const char *, const FileLoc *)> errorCallback);
#endif

    static std::unique_ptr<Tokenizer> CreateFromFile(
        const std::string &filename,
        std::function<void(const char *, const FileLoc *)> errorCallback);
    static std::unique_ptr<Tokenizer> CreateFromString(
        std::string str,
        std::function<void(const char *, const FileLoc *)> errorCallback);

    ~Tokenizer();

    // Note that the returned std::string_view in the Token is not
    // guaranteed to be valid after a subsequent call to Next().
    pstd::optional<Token> Next();

    // Just for parse().
    // TODO? Have a method to set this?
    FileLoc loc;

  private:
    int getChar() {
        if (pos == end)
            return EOF;
        int ch = *pos++;
        if (ch == '\n') {
            ++loc.line;
            loc.column = 0;
        } else
            ++loc.column;
        return ch;
    }
    void ungetChar() {
        --pos;
        if (*pos == '\n')
            // Don't worry about the column; we'll be going to the start of
            // the next line again shortly...
            --loc.line;
    }

    // This function is called if there is an error during lexing.
    std::function<void(const char *, const FileLoc *)> errorCallback;

#if defined(PBRT_HAVE_MMAP) || defined(PBRT_IS_WINDOWS)
    // Scene files on disk are mapped into memory for lexing.  We need to
    // hold on to the starting pointer and total length so they can be
    // unmapped in the destructor.
    void *unmapPtr = nullptr;
    size_t unmapLength = 0;
#endif

    // If the input is stdin, then we copy everything until EOF into this
    // string and then start lexing.  This is a little wasteful (versus
    // tokenizing directly from stdin), but makes the implementation
    // simpler.
    std::string contents;

    // Pointers to the current position in the file and one past the end of
    // the file.
    const char *pos, *end;

    // If there are escaped characters in the string, we can't just return
    // a std::string_view into the mapped file. In that case, we handle the
    // escaped characters and return a std::string_view to sEscaped.  (And
    // thence, std::string_views from previous calls to Next() must be invalid
    // after a subsequent call, since we may reuse sEscaped.)
    std::string sEscaped;
};

}  // namespace pbrt

#endif  // PBRT_CORE_PARSER_H
