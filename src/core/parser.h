
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
#include "pbrt.h"

#include "spectrum.h"

#include <string>
#include <vector>

namespace pbrt {

bool ParseFile(const std::string &filename);

namespace parse {

extern std::string currentFilename;
extern int currentLineNumber;

}  // namespace parse

struct ParamArray {
    void AddNumber(double d);
    void AddString(const std::string &str);
    void AddBool(bool v);

    std::vector<double> numbers;
    std::vector<std::string> strings;
    std::vector<bool> bools;
};

struct ParamListItem {
    ParamListItem(const std::string &name, std::unique_ptr<ParamArray> array)
        : name(name), array(std::move(array)) {}

    std::string name;
    std::unique_ptr<ParamArray> array;
};

ParamSet ParseParameters(const std::vector<ParamListItem> &paramList,
                         SpectrumType spectrumType);

}  // namespace pbrt

#endif  // PBRT_CORE_PARSER_H
