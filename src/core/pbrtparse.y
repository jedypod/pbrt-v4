
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

%{
#include "pbrt.h"

#include "api.h"
#include "error.h"
#include "paramset.h"
#include "parser.h"

#include <string>
#include <vector>

#ifdef PBRT_IS_MSVC
#pragma warning(disable:4065)
#pragma warning(disable:4996)
#pragma warning(disable:4018)
#endif // PBRT_IS_MSVC

void yyerror(const char *str) {
    pbrt::Error("Parsing error: %s", str);
    exit(1);
}
extern int yylex();

#define YYMAXDEPTH 100000000

namespace pbrt {
namespace parse {

int currentLineNumber = 0;
std::string currentFilename;

extern void includePush(const std::string &filename);

static std::vector<ParamListItem> currentParamList;

static ParamArray *currentArray = nullptr;

}  // namespace parse
}  // namespace pbrt

using namespace pbrt;
using namespace pbrt::parse;

%}

%union {
std::string *string;
double num;
pbrt::ParamArray *paramArray;
}

%token <string> STRING ID
%token <num> NUM

%token LBRACK RBRACK

%token ACCELERATOR ACTIVETRANSFORM ALL AREALIGHTSOURCE ATTRIBUTEBEGIN
%token ATTRIBUTEEND CAMERA CONCATTRANSFORM COORDINATESYSTEM COORDSYSTRANSFORM
%token ENDTIME FALSE FILM IDENTITY INCLUDE INTEGRATOR LIGHTSOURCE LOOKAT
%token MAKENAMEDMATERIAL MAKENAMEDMEDIUM MATERIAL MEDIUMINTERFACE NAMEDMATERIAL
%token OBJECTBEGIN OBJECTEND OBJECTINSTANCE PIXELFILTER REVERSEORIENTATION
%token ROTATE SAMPLER SCALE SHAPE STARTTIME TEXTURE TRANSFORMBEGIN TRANSFORMEND
%token TRANSFORMTIMES TRANSFORM TRANSLATE TRUE WORLDBEGIN WORLDEND

%token HIGH_PRECEDENCE

%type<paramArray> array num_array string_array bool_array

%%
start: pbrt_stmt_list
{
};

pbrt_stmt_list: pbrt_stmt_list pbrt_stmt
{
}
| pbrt_stmt
{
};

paramlist: paramlist_init paramlist_contents
{
};

paramlist_init: %prec HIGH_PRECEDENCE
{
    currentParamList.clear();
};

paramlist_contents: paramlist_entry paramlist_contents
{
}
|
{
};

paramlist_entry: STRING array
{
    currentParamList.push_back(ParamListItem(*$1, std::unique_ptr<ParamArray>($2)));
    delete $1;
};

array: string_array
{
    $$ = $1;
}
| num_array
{
    $$ = $1;
}
| bool_array
{
    $$ = $1;
};

string_array: array_init LBRACK string_list RBRACK
{
    $$ = currentArray;
    currentArray = nullptr;
}
| array_init string_list_entry  /* single string param */
{
    $$ = currentArray;
    currentArray = nullptr;
};

array_init: %prec HIGH_PRECEDENCE
{
    CHECK(currentArray == nullptr);
    currentArray = new ParamArray;
};

string_array_init: %prec HIGH_PRECEDENCE
{
};

num_array_init: %prec HIGH_PRECEDENCE
{
};

bool_array_init: %prec HIGH_PRECEDENCE
{
};

string_list: string_list string_list_entry
{
}
| string_list_entry
{
};

string_list_entry: string_array_init STRING
{
    currentArray->AddString(*$2);
    delete $2;
};

num_array: array_init LBRACK num_list RBRACK
{
    $$ = currentArray;
    currentArray = nullptr;
}
| array_init num_list_entry /* single number */
{
    $$ = currentArray;
    currentArray = nullptr;
};

num_list: num_list num_list_entry
{
}
| num_list_entry
{
};

num_list_entry: num_array_init NUM
{
    currentArray->AddNumber($2);
};

bool_array: array_init LBRACK bool_list RBRACK
{
    $$ = currentArray;
    currentArray = nullptr;
}
| array_init bool_list_entry /* single number */
{
    $$ = currentArray;
    currentArray = nullptr;
};

bool_list: bool_list bool_list_entry
{
}
| bool_list_entry
{
};

bool_list_entry: bool_array_init TRUE
{
    currentArray->AddBool(true);
}
| bool_array_init FALSE
{
    currentArray->AddBool(false);
};


pbrt_stmt: ACCELERATOR STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtAccelerator(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| ACTIVETRANSFORM ALL
{
    pbrtActiveTransformAll();
}
| ACTIVETRANSFORM ENDTIME
{
    pbrtActiveTransformEndTime();
}
| ACTIVETRANSFORM STARTTIME
{
    pbrtActiveTransformStartTime();
}
| AREALIGHTSOURCE STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Illuminant);
    pbrtAreaLightSource(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| ATTRIBUTEBEGIN
{
    pbrtAttributeBegin();
}
| ATTRIBUTEEND
{
    pbrtAttributeEnd();
}
| CAMERA STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtCamera(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| CONCATTRANSFORM num_array
{
    if ($2->numbers.size() != 16)
        Error("ConcatTransform expects 16 numeric values. Given %d",
              int($2->numbers.size()));
    else {
        Float m[16];
        std::copy($2->numbers.begin(), $2->numbers.end(), m);
        pbrtConcatTransform(m);
    }
    delete $2;
}
| COORDINATESYSTEM STRING
{
    pbrtCoordinateSystem(*$2);
    delete $2;
}
| COORDSYSTRANSFORM STRING
{
    pbrtCoordSysTransform(*$2);
    delete $2;
}
| FILM STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtFilm(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| IDENTITY
{
    pbrtIdentity();
}
| INCLUDE STRING
{
    includePush(*$2);
    delete $2;
}
| INTEGRATOR STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtIntegrator(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| LIGHTSOURCE STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Illuminant);
    pbrtLightSource(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| LOOKAT NUM NUM NUM NUM NUM NUM NUM NUM NUM
{
    pbrtLookAt($2, $3, $4, $5, $6, $7, $8, $9, $10);
}
| MAKENAMEDMATERIAL STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtMakeNamedMaterial(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| MAKENAMEDMEDIUM STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtMakeNamedMedium(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| MATERIAL STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtMaterial(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| MEDIUMINTERFACE STRING
{
    pbrtMediumInterface(*$2, *$2);
    delete $2;
}
| MEDIUMINTERFACE STRING STRING
{
    pbrtMediumInterface(*$2, *$3);
    delete $2;
    delete $3;
}
| NAMEDMATERIAL STRING
{
    pbrtNamedMaterial(*$2);
    delete $2;
}
| OBJECTBEGIN STRING
{
    pbrtObjectBegin(*$2);
    delete $2;
}
| OBJECTEND
{
    pbrtObjectEnd();
}
| OBJECTINSTANCE STRING
{
    pbrtObjectInstance(*$2);
    delete $2;
}
| PIXELFILTER STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtPixelFilter(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| REVERSEORIENTATION
{
    pbrtReverseOrientation();
}
| ROTATE NUM NUM NUM NUM
{
    pbrtRotate($2, $3, $4, $5);
}
| SAMPLER STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtSampler(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| SCALE NUM NUM NUM
{
    pbrtScale($2, $3, $4);
}
| SHAPE STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtShape(*$2, std::move(params));
    delete $2;
    currentParamList.clear();
}
| TEXTURE STRING STRING STRING paramlist
{
    ParamSet params = ParseParameters(currentParamList, SpectrumType::Reflectance);
    pbrtTexture(*$2, *$3, *$4, std::move(params));
    delete $2;
    delete $3;
    delete $4;
    currentParamList.clear();
}
| TRANSFORMBEGIN
{
    pbrtTransformBegin();
}
| TRANSFORMEND
{
    pbrtTransformEnd();
}
| TRANSFORMTIMES NUM NUM
{
    pbrtTransformTimes($2, $3);
}
| TRANSFORM num_array
{
    if ($2->numbers.size() != 16)
        Error("Transform expects 16 numeric values. Given %d",
              int($2->numbers.size()));
    else {
        Float m[16];
        std::copy($2->numbers.begin(), $2->numbers.end(), m);
        pbrtTransform(m);
    }
    delete $2;
}
| TRANSLATE NUM NUM NUM
{
    pbrtTranslate($2, $3, $4);
}
| WORLDBEGIN
{
    pbrtWorldBegin();
}
| WORLDEND
{
    pbrtWorldEnd();
};

%%
