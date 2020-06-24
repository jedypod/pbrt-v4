
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

#ifndef PBRT_CORE_PARSER_H
#define PBRT_CORE_PARSER_H

// core/parser.h*
#include <pbrt/pbrt.h>

#include <pbrt/util/check.h>
#include <pbrt/util/error.h>
#include <pbrt/util/pstd.h>

#include <functional>
#include <memory>
#include <string>

namespace pbrt {

void ParseFile(GeneralSceneBase *scene, std::string filename);
void ParseString(GeneralSceneBase *scene, std::string str);

struct Token {
    Token() = default;
    Token(pstd::string_view token, FileLoc loc)
        : token(token), loc(loc) {}

    std::string ToString() const;

    pstd::string_view token;
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
        std::string str, std::function<void(const char *, const FileLoc *)> errorCallback);

    ~Tokenizer();

    // Note that the returned pstd::string_view in the Token is not
    // guaranteed to be valid after a subsequent call to Next().
    pstd::optional<Token> Next();

    // Just for parse().
    // TODO? Have a method to set this?
    FileLoc loc;

  private:
    int getChar() {
        if (pos == end) return EOF;
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
    // a pstd::string_view into the mapped file. In that case, we handle the
    // escaped characters and return a pstd::string_view to sEscaped.  (And
    // thence, pstd::string_views from previous calls to Next() must be invalid
    // after a subsequent call, since we may reuse sEscaped.)
    std::string sEscaped;
};

}  // namespace pbrt

#endif  // PBRT_CORE_PARSER_H
