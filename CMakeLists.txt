# TODO: many of the best practices described here
# (https://www.slideshare.net/DanielPfeifer1/cmake-48475415) are violated
# in this file. Would be nice to address some of these.

CMAKE_MINIMUM_REQUIRED ( VERSION 2.8 )

# For sanitizers
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

PROJECT ( PBRT-V3 )

ENABLE_TESTING()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/openexr/OpenEXR")
  message(FATAL_ERROR "The OpenEXR submodule directory is missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by running \"git submodule update --init --recursive\"")
endif()

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/glog")
  message(FATAL_ERROR "The glog submodule directory is missing! "
    "You probably did not clone the project with --recursive, or first checked out "
    "pbrt before it was added. It is possible to recover by running "
    "\"git submodule update --init --recursive\"")
endif()

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/ptex")
  message(FATAL_ERROR "The ptex submodule directory is missing! "
    "You probably did not clone the project with --recursive, or first checked out "
    "pbrt before it was added. It is possible to recover by running "
    "\"git submodule update --init --recursive\"")
endif()

if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/abseil-cpp")
  message(FATAL_ERROR "The abseil-cpp submodule directory is missing! "
    "You probably did not clone the project with --recursive, or first checked out "
    "pbrt before it was added. It is possible to recover by running "
    "\"git submodule update --init --recursive\"")
endif()

FIND_PACKAGE ( Sanitizers )
FIND_PACKAGE ( Threads )

IF(CMAKE_BUILD_TYPE MATCHES RELEASE)
  ADD_DEFINITIONS (-DNDEBUG)
ENDIF()

###########################################################################
# Annoying compiler-specific details

IF(CMAKE_COMPILER_IS_GNUCXX)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-conversion-null")
ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  FIND_PROGRAM(XIAR xiar)
  IF(XIAR)
    SET(CMAKE_AR "${XIAR}")
  ENDIF(XIAR)
  MARK_AS_ADVANCED(XIAR)

  FIND_PROGRAM(XILD xild)
  IF(XILD)
    SET(CMAKE_LINKER "${XILD}")
  ENDIF(XILD)
  MARK_AS_ADVANCED(XILD)

  # ICC will default to -fp-model fast=1, which performs value-unsafe optimizations which will
  # cause pbrt_test to fail. For safety, -fp-model precise is explicitly set here by default.
  set(FP_MODEL "precise" CACHE STRING "The floating point model to compile with.")
  set_property(CACHE FP_MODEL PROPERTY STRINGS "precise" "fast=1" "fast=2")

  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model ${FP_MODEL}")
ENDIF()

IF(MSVC)
  ADD_DEFINITIONS (/D _CRT_SECURE_NO_WARNINGS)
ENDIF()

INCLUDE (CheckIncludeFiles)

CHECK_INCLUDE_FILES ( memory.h HAVE_MEMORY_H )
IF ( HAVE_MEMORY_H )
  ADD_DEFINITIONS ( -D PBRT_HAVE_MEMORY_H )
ENDIF ()

CHECK_INCLUDE_FILES ( unistd.h HAVE_UNISTD_H )
IF ( NOT HAVE_UNISTD_H )
  ADD_DEFINITIONS ( -D YY_NO_UNISTD_H )
ENDIF ()

###########################################################################
# Check for various C++11 features and set preprocessor variables or
# define workarounds.

INCLUDE (CheckCXXSourceCompiles)

CHECK_CXX_SOURCE_COMPILES (
  "int main() { float x = 0x1p-32f; }"
  HAVE_HEX_FP_CONSTANTS )
IF ( HAVE_HEX_FP_CONSTANTS )
  ADD_DEFINITIONS ( -D PBRT_HAVE_HEX_FP_CONSTANTS )
ENDIF ()

CHECK_CXX_SOURCE_COMPILES ( "
#include <signal.h>
#include <string.h>
#include <sys/time.h>
void ReportProfileSample(int, siginfo_t *, void *) { }
int main() {
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = ReportProfileSample;
    sa.sa_flags = SA_RESTART | SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGPROF, &sa, NULL);
    static struct itimerval timer;
    setitimer(ITIMER_PROF, &timer, NULL);
}
" HAVE_ITIMER )
IF ( HAVE_ITIMER )
  ADD_DEFINITIONS ( -D PBRT_HAVE_ITIMER )
ENDIF()

########################################
# noinline

CHECK_CXX_SOURCE_COMPILES (
"__declspec(noinline) void foo() { }
int main() { }"
HAVE_DECLSPEC_NOINLINE )

CHECK_CXX_SOURCE_COMPILES (
"__attribute__((noinline)) void foo() { }
int main() { }"
HAVE_ATTRIBUTE_NOINLINE )

IF ( HAVE_DECLSPEC_NOINLINE )
  ADD_DEFINITIONS ( -D "PBRT_NOINLINE=__declspec(noinline)" )
ELSEIF ( HAVE_ATTRIBUTE_NOINLINE )
  ADD_DEFINITIONS ( -D "PBRT_NOINLINE=__attribute__\\(\\(noinline\\)\\)" )
ELSE ()
  ADD_DEFINITIONS ( -D PBRT_NOINLINE )
ENDIF ()

########################################
# Aligned memory allocation

CHECK_CXX_SOURCE_COMPILES ( "
#include <malloc.h>
int main() { void * ptr = _aligned_malloc(1024, 32); }
" HAVE__ALIGNED_MALLOC )

CHECK_CXX_SOURCE_COMPILES ( "
#include <stdlib.h>
int main() {
  void *ptr;
  posix_memalign(&ptr, 32, 1024);
} " HAVE_POSIX_MEMALIGN )

CHECK_CXX_SOURCE_COMPILES ( "
#include <malloc.h>
int main() {
    void *ptr = memalign(32, 1024);
} " HAVE_MEMALIGN )

IF ( HAVE__ALIGNED_MALLOC )
  ADD_DEFINITIONS ( -D PBRT_HAVE__ALIGNED_MALLOC )
ELSEIF ( HAVE_POSIX_MEMALIGN )
  ADD_DEFINITIONS ( -D PBRT_HAVE_POSIX_MEMALIGN )
ELSEIF ( HAVE_MEMALIGN )
  ADD_DEFINITIONS ( -D PBRTHAVE_MEMALIGN )
ELSE ()
  MESSAGE ( SEND_ERROR "Unable to find a way to allocate aligned memory" )
ENDIF ()

########################################
# thread-local variables

CHECK_CXX_SOURCE_COMPILES ( "
thread_local int x; int main() { }
" HAVE_THREAD_LOCAL )

CHECK_CXX_SOURCE_COMPILES ( "
__declspec(thread) int x; int main() { }
" HAVE_DECLSPEC_THREAD )

CHECK_CXX_SOURCE_COMPILES ( "
__thread int x; int main() { }
" HAVE___THREAD )

IF ( HAVE_THREAD_LOCAL )
  ADD_DEFINITIONS ( -D PBRT_THREAD_LOCAL=thread_local )
ELSEIF ( HAVE_DECLSPEC_THREAD )
  ADD_DEFINITIONS ( -D "PBRT_THREAD_LOCAL=__declspec(thread)" )
ELSEIF ( HAVE___THREAD )
  ADD_DEFINITIONS ( -D PBRT_THREAD_LOCAL=__thread )
ELSE ()
  MESSAGE ( SEND_ERROR "Unable to find a way to declare a thread-local variable")
ENDIF ()

###########################################################################
# flex / bison

# Optionally use Bison and Flex to regenerate parser files
# Use pregenerated files otherwise (may be outdated)
FIND_PACKAGE ( BISON )
FIND_PACKAGE ( FLEX )
IF(BISON_FOUND AND FLEX_FOUND)
  SET(BisonOutput ${CMAKE_BINARY_DIR}/pbrtparse.cpp)
  ADD_CUSTOM_COMMAND(
    OUTPUT ${BisonOutput} ${CMAKE_BINARY_DIR}/pbrtparse.h
    DEPENDS ${CMAKE_SOURCE_DIR}/src/pbrt/core/pbrtparse.y
    COMMAND ${BISON_EXECUTABLE} -d -v -t
            --output=${CMAKE_BINARY_DIR}/pbrtparse.c
            ${CMAKE_SOURCE_DIR}/src/pbrt/core/pbrtparse.y
    COMMAND ${CMAKE_COMMAND} -E rename ${CMAKE_BINARY_DIR}/pbrtparse.c ${CMAKE_BINARY_DIR}/pbrtparse.cpp
    COMMENT "Generating pbrtparse.cpp and pbrtparse.h"
  )

  SET(FlexOutput ${CMAKE_BINARY_DIR}/pbrtlex.cpp)
  ADD_CUSTOM_COMMAND(
    OUTPUT ${FlexOutput}
    DEPENDS ${CMAKE_BINARY_DIR}/pbrtparse.h ${CMAKE_SOURCE_DIR}/src/pbrt/core/pbrtlex.ll
    COMMAND ${FLEX_EXECUTABLE}
            -o${FlexOutput}
            ${CMAKE_SOURCE_DIR}/src/pbrt/core/pbrtlex.ll
    COMMENT "Generating pbrtlex.cpp"
  )

  SET ( PBRT_YACC_LEX_SOURCE
    ${BisonOutput}
    ${FlexOutput}
    )
ELSE()
  SET ( PBRT_YACC_LEX_SOURCE
    src/pbrt/core/pbrtlex.cpp
    src/pbrt/core/pbrtparse.cpp
    )
ENDIF()

###########################################################################
# abseil

ADD_SUBDIRECTORY(src/ext/abseil-cpp)

INCLUDE_DIRECTORIES(
    src/ext/abseil-cpp
    src/ext/abseil-cpp/cctz/include
)

###########################################################################
# zlib

FIND_PACKAGE ( ZLIB )
IF(NOT ZLIB_FOUND)
  # Build zlib
  SET(ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  SET(ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  ADD_SUBDIRECTORY(src/ext/zlib)

  SET(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/zlib")
  SET(ZLIB_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/src/ext/zlib/$<CONFIGURATION>/zlibstatic.lib")
  SET(ZLIB_LIBRARIES zlibstatic)
  SET_PROPERTY(TARGET zlibstatic PROPERTY FOLDER "dependencies")
  INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/src/ext/zlib")
ENDIF()
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})

###########################################################################
# OpenEXR

SET(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
SET(ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)

ADD_SUBDIRECTORY(src/ext/openexr)

SET_PROPERTY(TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups CopyIlmBaseLibs IlmThread Half Iex Imath IlmImf PROPERTY FOLDER "dependencies")

INCLUDE_DIRECTORIES (
  src/ext/openexr/IlmBase/Imath
  src/ext/openexr/IlmBase/Half
  src/ext/openexr/IlmBase/Iex
  src/ext/openexr/OpenEXR/IlmImf
  ${CMAKE_BINARY_DIR}/src/ext/openexr/IlmBase/config
  ${CMAKE_BINARY_DIR}/src/ext/openexr/OpenEXR/config
)
IF(WIN32)
  SET(OPENEXR_LIBS IlmImf Imath Half ${ZLIB_LIBRARIES})
ELSE()
  SET(OPENEXR_LIBS IlmImf Imath Half)
ENDIF()

###########################################################################
# glog

SET(WITH_GFLAGS OFF CACHE BOOL "Use gflags")
SET(BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
ADD_SUBDIRECTORY(src/ext/glog)
SET_PROPERTY(TARGET glog PROPERTY FOLDER "dependencies")
INCLUDE_DIRECTORIES (
  src/ext/glog/src
  ${CMAKE_BINARY_DIR}/src/ext/glog
)

###########################################################################
# ptex

# work around https://github.com/wdas/ptex/issues/28
IF ( CMAKE_BUILD_TYPE )
  STRING ( TOLOWER ${CMAKE_BUILD_TYPE} LOWER_BUILD_TYPE )
  SET ( ENV{FLAVOR} ${LOWER_BUILD_TYPE} )
ENDIF ()

SET(CMAKE_MACOSX_RPATH 1)
IF ( WIN32 )
  ADD_DEFINITIONS ( /D PTEX_STATIC)
ENDIF ()
ADD_SUBDIRECTORY(src/ext/ptex)
SET_PROPERTY(TARGET Ptex_static PROPERTY FOLDER "dependencies")
INCLUDE_DIRECTORIES ( src/ext/ptex/src/ptex )

###########################################################################
# On to pbrt...

SET ( PBRT_CORE_SOURCE
  src/pbrt/core/api.cpp
  src/pbrt/core/bssrdf.cpp
  src/pbrt/core/camera.cpp
  src/pbrt/core/error.cpp
  src/pbrt/core/film.cpp
  src/pbrt/core/filter.cpp
  src/pbrt/core/image.cpp
  src/pbrt/core/integrator.cpp
  src/pbrt/core/interaction.cpp
  src/pbrt/core/light.cpp
  src/pbrt/core/lightdistrib.cpp
  src/pbrt/core/lowdiscrepancy.cpp
  src/pbrt/core/material.cpp
  src/pbrt/core/medium.cpp
  src/pbrt/core/microfacet.cpp
  src/pbrt/core/mipmap.cpp
  src/pbrt/core/paramset.cpp
  src/pbrt/core/parser.cpp
  src/pbrt/core/primitive.cpp
  src/pbrt/core/reflection.cpp
  src/pbrt/core/sampler.cpp
  src/pbrt/core/sampling.cpp
  src/pbrt/core/scene.cpp
  src/pbrt/core/shape.cpp
  src/pbrt/core/sobolmatrices.cpp
  src/pbrt/core/spectrum.cpp
  src/pbrt/core/texcache.cpp
  src/pbrt/core/texture.cpp
  )

SET ( PBRT_CORE_HEADERS
  src/pbrt/core/api.h
  src/pbrt/core/bssrdf.h
  src/pbrt/core/camera.h
  src/pbrt/core/error.h
  src/pbrt/core/film.h
  src/pbrt/core/filter.h
  src/pbrt/core/integrator.h
  src/pbrt/core/interaction.h
  src/pbrt/core/light.h
  src/pbrt/core/lowdiscrepancy.h
  src/pbrt/core/material.h
  src/pbrt/core/medium.h
  src/pbrt/core/microfacet.h
  src/pbrt/core/mipmap.h
  src/pbrt/core/paramset.h
  src/pbrt/core/parser.h
  src/pbrt/core/pbrt.h
  src/pbrt/core/primitive.h
  src/pbrt/core/reflection.h
  src/pbrt/core/sampler.h
  src/pbrt/core/sampling.h
  src/pbrt/core/scene.h
  src/pbrt/core/shape.h
  src/pbrt/core/sobolmatrices.h
  src/pbrt/core/spectrum.h
  src/pbrt/core/texture.h
  )

FILE ( GLOB PBRT_SOURCE
  src/pbrt/accelerators/*.cpp
  src/pbrt/cameras/*.cpp
  src/pbrt/filters/*.cpp
  src/pbrt/integrators/*.cpp
  src/pbrt/lights/*.cpp
  src/pbrt/materials/*.cpp
  src/pbrt/media/*.cpp
  src/pbrt/samplers/*.cpp
  src/pbrt/shapes/*.cpp
  src/pbrt/textures/*.cpp
  src/pbrt/util/*.cpp

  src/ext/gtest/gtest-all.cc
  src/ext/gtest/gtest_main.cc
  src/ext/lodepng/lodepng.cpp
  src/ext/rply/rply.cpp
  src/ext/skymodel/ArHosekSkyModel.c
  )

INCLUDE_DIRECTORIES ( src )
INCLUDE_DIRECTORIES ( src/ext )

###########################################################################
# pbrt libraries and executables

SET ( PBRT_CXX14_NEEDS
  cxx_aggregate_default_initializers
  cxx_alignas
  cxx_alignof
  cxx_auto_type
  cxx_binary_literals
  cxx_decltype
  cxx_delegating_constructors
  cxx_explicit_conversions
  cxx_generalized_initializers
  cxx_lambdas
  cxx_nullptr
  cxx_range_for
  cxx_raw_string_literals
  cxx_relaxed_constexpr
  cxx_right_angle_brackets
  cxx_static_assert
  cxx_variadic_templates
)

ADD_LIBRARY ( pbrt STATIC
  ${PBRT_YACC_LEX_SOURCE}
  ${PBRT_CORE_SOURCE}
  ${PBRT_CORE_HEADERS}
  ${PBRT_SOURCE}
  )
ADD_SANITIZERS ( pbrt )
TARGET_COMPILE_FEATURES ( pbrt PRIVATE ${PBRT_CXX14_NEEDS} )

IF (WIN32)
  # Avoid a name clash when building on Visual Studio
  SET_TARGET_PROPERTIES ( pbrt PROPERTIES OUTPUT_NAME libpbrt )
ENDIF()

SET(ALL_PBRT_LIBS
  pbrt
  ${CMAKE_THREAD_LIBS_INIT}
  ${OPENEXR_LIBS}
  glog
  Ptex_static
  ${ZLIB_LIBRARY}
  absl
  cctz
)

# Main renderer
ADD_EXECUTABLE ( pbrt_exe src/pbrt/cmd/pbrt.cpp )
ADD_SANITIZERS ( pbrt_exe )
TARGET_COMPILE_FEATURES ( pbrt_exe PRIVATE ${PBRT_CXX14_NEEDS} )
SET_TARGET_PROPERTIES ( pbrt_exe PROPERTIES OUTPUT_NAME pbrt )
TARGET_LINK_LIBRARIES ( pbrt_exe ${ALL_PBRT_LIBS} )

# Tools
ADD_EXECUTABLE ( bsdftest src/pbrt/cmd/bsdftest.cpp )
ADD_SANITIZERS ( bsdftest )
TARGET_COMPILE_FEATURES ( bsdftest PRIVATE ${PBRT_CXX14_NEEDS} )
TARGET_LINK_LIBRARIES ( bsdftest ${ALL_PBRT_LIBS} )

ADD_EXECUTABLE ( imgtool src/pbrt/cmd/imgtool.cpp )
ADD_SANITIZERS ( imgtool )
TARGET_COMPILE_FEATURES ( imgtool PRIVATE ${PBRT_CXX14_NEEDS} )
TARGET_LINK_LIBRARIES ( imgtool ${ALL_PBRT_LIBS} )

ADD_EXECUTABLE ( obj2pbrt src/pbrt/cmd/obj2pbrt.cpp )
ADD_SANITIZERS ( obj2pbrt )
TARGET_COMPILE_FEATURES ( obj2pbrt PRIVATE ${PBRT_CXX14_NEEDS} )

ADD_EXECUTABLE ( cyhair2pbrt src/pbrt/cmd/cyhair2pbrt.cpp )
ADD_SANITIZERS ( cyhair2pbrt )
TARGET_COMPILE_FEATURES ( cyhair2pbrt PRIVATE ${PBRT_CXX14_NEEDS} )

# Unit test

FILE ( GLOB PBRT_TEST_SOURCE
  src/pbrt/tests/*.cpp
  )

ADD_EXECUTABLE ( pbrt_test ${PBRT_TEST_SOURCE} )
ADD_SANITIZERS ( pbrt_test )
TARGET_COMPILE_FEATURES ( pbrt_test PRIVATE ${PBRT_CXX14_NEEDS} )
TARGET_LINK_LIBRARIES ( pbrt_test ${ALL_PBRT_LIBS} )

ADD_TEST ( pbrt_unit_test pbrt_test )

# Installation

INSTALL ( TARGETS
  pbrt_exe
  bsdftest
  imgtool
  obj2pbrt
  cyhair2pbrt
  DESTINATION
  bin
  )

INSTALL ( TARGETS
  pbrt
  DESTINATION
  lib
  )
