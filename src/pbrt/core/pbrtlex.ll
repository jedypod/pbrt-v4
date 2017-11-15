
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

%option nounistd

%{

#include <pbrt/core/pbrt.h>

#include <pbrt/core/error.h>
#include <pbrt/core/options.h>
#include <pbrt/core/parser.h>
#include <pbrt/util/fileutil.h>

#include <string>
#include <vector>
#include <stdio.h>

#if defined(PBRT_IS_MSVC)
#include <io.h>
#pragma warning(disable:4244)
#pragma warning(disable:4065)
#pragma warning(disable:4018)
#pragma warning(disable:4996)
int isatty(int fd) { return _isatty(fd); }
#else
#include <unistd.h>
#endif  // PBRT_IS_MSVC

#include "pbrtparse.h"

namespace pbrt {

extern int catIndentCount;

namespace parse {

struct IncludeInfo {
    std::string filename;
    YY_BUFFER_STATE bufState;
    int lineNum;
};

static std::vector<IncludeInfo> includeStack;

void includePush(const std::string &filename) {
    if (includeStack.size() > 32) {
        ErrorExit("Only 32 levels of nested Include allowed in scene files.");
    }

    std::string newFilename = AbsolutePath(ResolveFilename(filename));

    FILE *f = fopen(newFilename.c_str(), "r");
    if (!f)
        ErrorExit("Unable to open included scene file \"%s\"", newFilename.c_str());
    else {
        IncludeInfo ii;
        ii.filename = currentFilename;
        ii.bufState = YY_CURRENT_BUFFER;
        ii.lineNum = currentLineNumber;
        includeStack.push_back(ii);

        yyin = f;
        currentFilename = newFilename;
        currentLineNumber = 1;

        yy_switch_to_buffer(yy_create_buffer(yyin, YY_BUF_SIZE));
    }
}

void includePop() {
    fclose(yyin);
    yy_delete_buffer(YY_CURRENT_BUFFER);
    yy_switch_to_buffer(includeStack.back().bufState);
    currentFilename = includeStack.back().filename;
    currentLineNumber = includeStack.back().lineNum;
    includeStack.pop_back();
}

}  // namespace pbrt
}  // namespace parse

%}

%option nounput
WHITESPACE [ \t\r]+
NUMBER [-+]?([0-9]+|(([0-9]+\.[0-9]*)|(\.[0-9]+)))([eE][-+]?[0-9]+)?
IDENT [a-zA-Z_][a-zA-Z_0-9]*
%x STR COMMENT INCL INCL_FILE

%%
"#" {
    BEGIN COMMENT;
    if (pbrt::PbrtOptions.cat || pbrt::PbrtOptions.toPly)
        printf("%*s#", pbrt::catIndentCount, "");
}
<COMMENT>. {
    /* eat it up */
    if (pbrt::PbrtOptions.cat || pbrt::PbrtOptions.toPly)
        putchar(yytext[0]);
}
<COMMENT>\n {
    pbrt::parse::currentLineNumber++;
    if (pbrt::PbrtOptions.cat || pbrt::PbrtOptions.toPly) putchar('\n');
    BEGIN INITIAL;
}
Accelerator             { return ACCELERATOR; }
ActiveTransform         { return ACTIVETRANSFORM; }
All                     { return ALL; }
AreaLightSource         { return AREALIGHTSOURCE; }
Attribute               { return ATTRIBUTE; }
AttributeBegin          { return ATTRIBUTEBEGIN; }
AttributeEnd            { return ATTRIBUTEEND; }
Camera                  { return CAMERA; }
ConcatTransform         { return CONCATTRANSFORM; }
CoordinateSystem        { return COORDINATESYSTEM; }
CoordSysTransform       { return COORDSYSTRANSFORM; }
EndTime                 { return ENDTIME; }
false                   { return FALSE; }
Film                    { return FILM; }
Identity                { return IDENTITY; }
Include                 { return INCLUDE; }
LightSource             { return LIGHTSOURCE; }
LookAt                  { return LOOKAT; }
MakeNamedMedium         { return MAKENAMEDMEDIUM; }
MakeNamedMaterial       { return MAKENAMEDMATERIAL; }
Material                { return MATERIAL; }
MediumInterface         { return MEDIUMINTERFACE; }
NamedMaterial           { return NAMEDMATERIAL; }
ObjectBegin             { return OBJECTBEGIN; }
ObjectEnd               { return OBJECTEND; }
ObjectInstance          { return OBJECTINSTANCE; }
PixelFilter             { return PIXELFILTER; }
ReverseOrientation      { return REVERSEORIENTATION; }
Rotate                  { return ROTATE; }
Sampler                 { return SAMPLER; }
Scale                   { return SCALE; }
Shape                   { return SHAPE; }
StartTime               { return STARTTIME; }
Integrator              { return INTEGRATOR; }
Texture                 { return TEXTURE; }
TransformBegin          { return TRANSFORMBEGIN; }
TransformEnd            { return TRANSFORMEND; }
TransformTimes          { return TRANSFORMTIMES; }
Transform               { return TRANSFORM; }
Translate               { return TRANSLATE; }
true                    { return TRUE; }
WorldBegin              { return WORLDBEGIN; }
WorldEnd                { return WORLDEND; }

{WHITESPACE} /* do nothing */
\n { pbrt::parse::currentLineNumber++; }

{NUMBER} {
    yylval.num = atof(yytext);
    return NUM;
}

{IDENT} {
    yylval.string = pbrt::parse::stringPool->Alloc();
    *yylval.string = yytext;
    return ID;
}

"[" { return LBRACK; }
"]" { return RBRACK; }

\" {
  BEGIN STR;
  yylval.string = pbrt::parse::stringPool->Alloc();
}
<STR>\\n { *yylval.string += '\n'; }
<STR>\\t { *yylval.string += '\t'; }
<STR>\\r { *yylval.string += '\r'; }
<STR>\\b { *yylval.string += '\b'; }
<STR>\\f { *yylval.string += '\f'; }
<STR>\\\" { *yylval.string += '\"'; }
<STR>\\\\ { *yylval.string += '\\'; }
<STR>\\[0-9]{3} {
    int val = atoi(yytext+1);
    while (val > 256)
        val -= 256;
    *yylval.string += val; 
}
<STR>\\\n { pbrt::parse::currentLineNumber++; }
<STR>\\. {  *yylval.string += yytext[1]; }
<STR>\" {
    BEGIN INITIAL;
    CHECK(yylval.string != nullptr);
    return STRING;
}
<STR>. { *yylval.string += yytext[0]; }
<STR>\n { pbrt::Error("Unterminated string!"); }

. {
    pbrt::Error("Illegal character: %c (0x%x)", yytext[0], int(yytext[0]));
}

%%

int yywrap() {
    if (pbrt::parse::includeStack.size() == 0) return 1;
    pbrt::parse::includePop();
    return 0;
}
