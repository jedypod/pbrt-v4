
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
#include <pbrt/core/parser.h>

#include <pbrt/core/error.h>
#include <pbrt/util/fileutil.h>
#include <pbrt/util/floatfile.h>
#include <pbrt/core/paramset.h>

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
std::unique_ptr<MemoryPool<NamedValues>> namedValuesPool;

} // namespace parse

// Parsing Global Interface
bool ParseFile(const std::string &filename) {
    CHECK(!parse::stringPool && !parse::namedValuesPool);
    parse::stringPool = std::make_unique<MemoryPool<std::string>>(
        [](std::string *str) { str->clear(); });
    parse::namedValuesPool = std::make_unique<MemoryPool<NamedValues>>(
        [](NamedValues *nv) {
            nv->name = nullptr;
            nv->next = nullptr;

            size_t cap = nv->numbers.capacity();
            nv->numbers.clear();
            // Part of why we want to reuse these is to reuse the internal
            // allocations done by the vectors. The C++ standard doesn't
            // require that clear() maintain the current capacity, but
            // current implementations seem to do this. Verify that this
            // remains so (and figure out what to do about it if this ever
            // starts to hit...)
            CHECK_EQ(cap, nv->numbers.capacity());

            nv->strings.clear();
            nv->bools.clear();
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
    parse::namedValuesPool = nullptr;

    return (yyin != nullptr);
}

}  // namespace pbrt
