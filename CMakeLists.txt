# TODO: many of the best practices described here
# (https://www.slideshare.net/DanielPfeifer1/cmake-48475415) are violated
# in this file. Would be nice to address some of these.

cmake_minimum_required(VERSION 3.12)

PROJECT ( PBRT-V4 LANGUAGES CXX C )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For sanitizers
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Configuration options

OPTION(PBRT_FLOAT_AS_DOUBLE "Use 64-bit floats" OFF)
OPTION(PBRT_BUILD_PYBIND "Build pybind libraries" OFF)
OPTION(PBRT_BUILD_NATIVE_EXECUTABLE "Build executable optimized for CPU architecture of system pbrt was built on" ON)
SET(PBRT_OPTIX7_PATH "" CACHE STRING "Path to OptiX 7 SDK")

IF (PBRT_FLOAT_AS_DOUBLE)
  ADD_DEFINITIONS ( -D PBRT_FLOAT_AS_DOUBLE )
ENDIF()

ENABLE_TESTING()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

function(CHECK_EXT NAME DIR HASH)
  if(NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/${DIR}")
    message(FATAL_ERROR "The ${NAME} submodule directory is missing! "
      "You probably did not clone the project with --recursive. It is possible to recover by running:\n"
      "  \"git submodule update --init --recursive\"")
  endif()

  FIND_PACKAGE(Git)
  IF(GIT_FOUND)
    EXECUTE_PROCESS(
      COMMAND ${GIT_EXECUTABLE} branch --contains ${HASH} HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/${DIR}"
      RESULT_VARIABLE "git_return"
      ERROR_QUIET
      OUTPUT_QUIET)
    IF(NOT ${git_return} EQUAL 0)
      MESSAGE(FATAL_ERROR "The ${CMAKE_CURRENT_SOURCE_DIR}/src/ext/${DIR} "
        "submodule isn't up to date. Please run:\n"
        "  \"git submodule update --recursive\"")
    ELSE()
      #MESSAGE( STATUS "${NAME}: includes git commit: ${HASH}" )
    ENDIF()
  ELSE(GIT_FOUND)
    MESSAGE( STATUS "git not found: unable to verify revisions in submodules" )
  ENDIF(GIT_FOUND)

endfunction()

CHECK_EXT("OpenEXR" "openexr/OpenEXR" d951c04)
CHECK_EXT("Ptex" "ptex/src" 82bd326)
CHECK_EXT("double-conversion" "double-conversion/cmake" 9a8e518)
CHECK_EXT("filesystem" "filesystem/filesystem" 4efd2628)
CHECK_EXT("pybind11" "pybind11/pybind11" f7bc18f)
CHECK_EXT("stb" "stb/tools" 1034f5)
CHECK_EXT("zlib" "zlib/doc" 54d591e)

IF(CMAKE_BUILD_TYPE MATCHES Release)
  ADD_DEFINITIONS (-DNDEBUG)
ENDIF()
# To build a release build with CHECKs enabled, comment-out the above
# 3 lines and un-comment out this one:
# SET(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(CheckLanguage)
CHECK_LANGUAGE(CUDA)
if (CMAKE_CUDA_COMPILER)
    find_package(CUDA REQUIRED)

    set(CMAKE_CXX_STANDARD 17)

    message(STATUS "Found CUDA: ${CMAKE_CUDA_COMPILER_VERSION}")
    if ("${PBRT_OPTIX7_PATH}" STREQUAL "")
        message(WARNING "Found CUDA but PBRT_OPTIX7_PATH is not set. Disabling GPU compilation.")
    else()
        enable_language(CUDA)
        ADD_DEFINITIONS( -D PBRT_BUILD_GPU_RENDERER )
        set(PBRT_CUDA_ENABLED ON)
        INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})  # for regular c++ compiles

        # http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
        SET(PBRT_CUDA_DIAG_FLAGS "")
        #SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} -Xptxas --warn-on-double-precision-use")
        SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=partial_override")
        SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=virtual_function_decl_hidden")
        SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=integer_sign_change")
        SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=declared_but_not_referenced")
        SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} --expt-relaxed-constexpr")
        SET(PBRT_CUDA_DIAG_FLAGS "${PBRT_CUDA_DIAG_FLAGS} --extended-lambda")
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${PBRT_CUDA_DIAG_FLAGS}")

        # https://wagonhelm.github.io/articles/2018-03/detecting-cuda-capability-with-cmake
        #Get CUDA compute capability
        set(OUTPUTFILE ${CMAKE_BINARY_DIR}/checkcuda)
        execute_process(COMMAND nvcc -lcuda ${CMAKE_SOURCE_DIR}/cmake/checkcuda.cu -o ${OUTPUTFILE})
        execute_process(COMMAND ${OUTPUTFILE}
                        RESULT_VARIABLE CUDA_RETURN_CODE
                        OUTPUT_VARIABLE ARCH)

        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")
        IF(CMAKE_BUILD_TYPE MATCHES Release)
          SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
        ELSE()
          SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -G -g")
        ENDIF()

        if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
            MESSAGE(SEND_ERROR "Unable to determine GPU's compute capability")
        else()
            message(STATUS "CUDA Architecture: ${ARCH}")
            SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=${ARCH}")
        endif()

        SET(PBRT_CUDA_LIB cuda)

        # optix
        INCLUDE_DIRECTORIES(${PBRT_OPTIX7_PATH}/include)

        # from Ingo's configure_optix.cmake (Apache licensed)
        find_program(BIN2C bin2c DOC "Path to the CUDA SDK bin2c executable.")

        # this macro defines cmake rules that execute the following four steps:
        # 1) compile the given cuda file ${cuda_file} to an intermediary PTX file
        # 2) use the 'bin2c' tool (that comes with CUDA) to
        #    create a second intermediary (.c-)file which defines a const string variable
        #    (named '${c_var_name}') whose (constant) value is the PTX output
        #    from the previous step.
        # 3) compile the given .c file to an intermediary object file (why thus has
        #    that PTX string 'embedded' as a global constant.
        # 4) assign the name of the intermediary .o file to the cmake variable
        #    'output_var', which can then be added to cmake targets.
        macro(cuda_compile_and_embed output_var cuda_file)
          set(c_var_name ${output_var})
          CUDA_COMPILE_PTX(ptx_files ${cuda_file} OPTIONS "--std=c++17 -O3" ${PBRT_CUDA_DIAG_FLAGS} "--gpu-architecture=${ARCH} -D PBRT_BUILD_GPU_RENDERER")
          list(GET ptx_files 0 ptx_file)
          set(embedded_file ${ptx_file}_embedded.c)
        #  message("adding rule to compile and embed ${cuda_file} to \"const char ${var_name}[];\"")
          add_custom_command(
            OUTPUT ${embedded_file}
            COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
            DEPENDS ${ptx_file}
            COMMENT "compiling (and embedding ptx from) ${cuda_file}"
            )
          set(${output_var} ${embedded_file})
        endmacro()
    endif()
else()
    message(STATUS "CUDA not found")
endif()

FIND_PACKAGE ( Sanitizers )
FIND_PACKAGE ( Threads )

SET_PROPERTY(GLOBAL PROPERTY USE_FOLDERS ON)

# libgoogle-perftools-dev
FIND_LIBRARY(PROFILE_LIB profiler)
if (NOT PROFILE_LIB)
  MESSAGE(STATUS "Unable to find -lprofiler")
else()
  MESSAGE(STATUS "Found -lprofiler: ${PROFILE_LIB}")
endif()

###########################################################################
# Annoying compiler-specific details

INCLUDE(CheckCXXCompilerFlag)

CHECK_CXX_COMPILER_FLAG("-std=gnu++17" COMPILER_SUPPORTS_STD_GNU17)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_STD_CXX17)
IF (COMPILER_SUPPORTS_STD_GNU17)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
ELSEIF (COMPILER_SUPPORTS_STD_CXX17)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
ENDIF()

CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
IF (COMPILER_SUPPORTS_MARCH_NATIVE AND PBRT_BUILD_NATIVE_EXECUTABLE)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ENDIF()
# TODO: how to specify this on wondows?

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

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
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4305") # double constant assigned to float
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4244") # int -> float conversion
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4843") # double -> float conversion
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267") # size_t -> int conversion
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4838") # double -> int conversion
ENDIF()

INCLUDE (CheckIncludeFiles)

CHECK_INCLUDE_FILES ( memory.h HAVE_MEMORY_H )
IF ( HAVE_MEMORY_H )
  ADD_DEFINITIONS ( -D PBRT_HAVE_MEMORY_H )
ENDIF ()

###########################################################################
# Check for various C++ features and set preprocessor variables or
# define workarounds.

INCLUDE (CheckCXXSourceCompiles)
INCLUDE (CheckCXXSourceRuns)

CHECK_CXX_SOURCE_COMPILES (
  "int main() { float x = 0x1p-32f; }"
  HAVE_HEX_FP_CONSTANTS )
IF ( HAVE_HEX_FP_CONSTANTS )
  ADD_DEFINITIONS ( -D PBRT_HAVE_HEX_FP_CONSTANTS )
ENDIF ()

CHECK_CXX_SOURCE_COMPILES ( "
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
int main() {
   int fd = open(\"foo\", O_RDONLY);
   struct stat s;
   fstat(fd, &s);
   size_t len = s.st_size;
   void *ptr = mmap(0, len, PROT_READ, MAP_FILE | MAP_SHARED, fd, 0);
   munmap(ptr, len);   
}
" HAVE_MMAP )
if ( HAVE_MMAP )
  ADD_DEFINITIONS ( -D PBRT_HAVE_MMAP )
ENDIF ()

CHECK_CXX_SOURCE_COMPILES ( "
template <template<typename> class Child, typename T>
struct Tuple2 {
    T x, y;
};
template <typename T>
class Vector2 : public Tuple2<Vector2, T> {
  public:
    using Tuple2<Vector2, T>::x;
    using Tuple2<Vector2, T>::y;
};
int main() { Vector2<float> v; }
" CRTP_USING_IS_SANE )
IF (NOT CRTP_USING_IS_SANE)
  ADD_DEFINITIONS ( -D PBRT_CRTP_USING_WORKAROUND )
ENDIF()

CHECK_CXX_SOURCE_COMPILES ( "
#include <intrin.h>
int main() {
    unsigned long lz = 0, v = 1234;
    if (_BitScanReverse(&lz, v)) return lz;
    return 0;
} " HAS_BITSCAN )
IF ( HAS_BITSCAN )
  ADD_DEFINITIONS ( -D PBRT_HAS_INTRIN_H )
ENDIF()

########################################
# os/compiler-specific stuff

IF ( CMAKE_SYSTEM_NAME STREQUAL Windows )
  ADD_DEFINITIONS ( -D PBRT_IS_WINDOWS )
ELSEIF ( CMAKE_SYSTEM_NAME STREQUAL Darwin )
  ADD_DEFINITIONS ( -D PBRT_IS_OSX )
ELSEIF (CMAKE_SYSTEM_NAME STREQUAL Linux )
  ADD_DEFINITIONS ( -D PBRT_IS_LINUX )
  # -rdynamic so we can get backtrace symbols...
  # --no-as-needed so libprofiler sticks around
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic -Wl,--no-as-needed")
ELSE()
  MESSAGE ( SEND_ERROR "Unknown system name: " + CMAKE_SYSTEM_NAME )
ENDIF ()

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

IF ( HAVE_ATTRIBUTE_NOINLINE )
  ADD_DEFINITIONS ( -D "PBRT_NOINLINE=__attribute__\\(\\(noinline\\)\\)" )
ELSEIF ( HAVE_DECLSPEC_NOINLINE )
  ADD_DEFINITIONS ( -D "PBRT_NOINLINE=__declspec(noinline)" )
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
# are long and int64_t the same

CHECK_CXX_SOURCE_COMPILES ( "
#include <cstdint>
#include <type_traits>
static_assert(!std::is_same<long, int64_t>::value && !std::is_same<long long, int64_t>::value);
int main() { }
" INT64_IS_OWN_TYPE)

IF (INT64_IS_OWN_TYPE)
  ADD_DEFINITIONS( -D PBRT_INT64_IS_OWN_TYPE )
ENDIF()

###########################################################################
# stb

INCLUDE_DIRECTORIES(
  src/ext/stb
)

###########################################################################
# filesystem

INCLUDE_DIRECTORIES(
  src/ext/filesystem
)

###########################################################################
# trove

INCLUDE_DIRECTORIES(
  src/ext/trove
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
  SET(ZLIB_LIBRARY zlibstatic)
  SET_PROPERTY(TARGET zlibstatic PROPERTY FOLDER "ext")
  INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/src/ext/zlib")
ENDIF()
INCLUDE_DIRECTORIES(${ZLIB_INCLUDE_DIRS})

###########################################################################
# OpenEXR

SET(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
SET(ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
SET(PYILMBASE_ENABLE             OFF CACHE BOOL " " FORCE)
SET(OPENEXR_VIEWERS_ENABLE       OFF CACHE BOOL " " FORCE)

ADD_SUBDIRECTORY(src/ext/openexr)

SET_PROPERTY(TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups IlmThread Half Iex Imath IlmImf PROPERTY FOLDER "ext/OpenEXR")

INCLUDE_DIRECTORIES (
  src/ext/openexr/IlmBase/Imath
  src/ext/openexr/IlmBase/Half
  src/ext/openexr/IlmBase/Iex
  src/ext/openexr/OpenEXR/IlmImf
  ${CMAKE_BINARY_DIR}/src/ext/openexr/IlmBase/config
  ${CMAKE_BINARY_DIR}/src/ext/openexr/OpenEXR/config
)

IF(WIN32)
  SET(OPENEXR_LIBS IlmImf Imath Half ${ZLIB_LIBRARY})
ELSE()
  SET(OPENEXR_LIBS IlmImf Imath Half)
ENDIF()

###########################################################################
# ptex

SET(PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)

SET(CMAKE_MACOSX_RPATH 1)
IF ( WIN32 )
  ADD_DEFINITIONS ( /D PTEX_STATIC)
ENDIF ()

ADD_SUBDIRECTORY(src/ext/ptex)

SET_PROPERTY(TARGET Ptex_static ptxinfo halftest ftest rtest wtest PROPERTY FOLDER "ext/ptex")

INCLUDE_DIRECTORIES ( src/ext/ptex/src/ptex )

###########################################################################
# double-conversion

ADD_SUBDIRECTORY(src/ext/double-conversion)
INCLUDE_DIRECTORIES(src/ext/double-conversion)

SET_PROPERTY(TARGET double-conversion PROPERTY FOLDER "ext")

###########################################################################
# pybind11

if (PBRT_BUILD_PYBIND)
  find_package(PythonLibs)

  if (PYTHONLIBS_FOUND)
    ADD_SUBDIRECTORY(src/ext/pybind11)
    include_directories(src/ext/pybind11)
  endif ()
endif ()

###########################################################################
# On to pbrt...

INCLUDE_DIRECTORIES ( src )
INCLUDE_DIRECTORIES ( src/ext )

SET ( PBRT_SOURCE
  src/pbrt/bsdf.cpp
  src/pbrt/bssrdf.cpp
  src/pbrt/bxdfs.cpp
  src/pbrt/cameras.cpp
  src/pbrt/film.cpp
  src/pbrt/filters.cpp
  src/pbrt/interaction.cpp
  src/pbrt/lights.cpp
  src/pbrt/lightsamplers.cpp
  src/pbrt/materials.cpp
  src/pbrt/media.cpp
  src/pbrt/options.cpp
  src/pbrt/paramdict.cpp
  src/pbrt/parsedscene.cpp
  src/pbrt/parser.cpp
  src/pbrt/pbrt.cpp
  src/pbrt/ray.cpp
  src/pbrt/samplers.cpp
  src/pbrt/shapes.cpp
  src/pbrt/textures.cpp

  src/pbrt/cpu/accelerators.cpp
  src/pbrt/cpu/integrators.cpp
  src/pbrt/cpu/primitive.cpp
  src/pbrt/cpu/render.cpp
  )

SET ( PBRT_UTIL_SOURCE
  src/pbrt/util/bluenoise.cpp
  src/pbrt/util/check.cpp
  src/pbrt/util/color.cpp
  src/pbrt/util/colorspace.cpp
  src/pbrt/util/display.cpp
  src/pbrt/util/error.cpp
  src/pbrt/util/file.cpp
  src/pbrt/util/float.cpp
  src/pbrt/util/image.cpp
  src/pbrt/util/log.cpp
  src/pbrt/util/loopsubdiv.cpp
  src/pbrt/util/lowdiscrepancy.cpp
  src/pbrt/util/math.cpp
  src/pbrt/util/memory.cpp
  src/pbrt/util/mesh.cpp
  src/pbrt/util/mipmap.cpp
  src/pbrt/util/parallel.cpp
  src/pbrt/util/pmj02tables.cpp
  src/pbrt/util/primes.cpp
  src/pbrt/util/print.cpp
  src/pbrt/util/progressreporter.cpp
  src/pbrt/util/pstd.cpp
  src/pbrt/util/rng.cpp
  src/pbrt/util/sampling.cpp
  src/pbrt/util/scattering.cpp
  src/pbrt/util/sobolmatrices.cpp
  src/pbrt/util/spectrum.cpp
  src/pbrt/util/stats.cpp
  src/pbrt/util/stbimage.cpp
  src/pbrt/util/string.cpp
  src/pbrt/util/transform.cpp
  src/pbrt/util/vecmath.cpp
)

if (PBRT_CUDA_ENABLED)
 SET ( PBRT_GPU_SOURCE
    src/pbrt/gpu/accel.cpp
    src/pbrt/gpu/camera.cpp
    src/pbrt/gpu/direct.cpp
    src/pbrt/gpu/film.cpp
    src/pbrt/gpu/indirect.cpp
    src/pbrt/gpu/init.cpp
    src/pbrt/gpu/launch.cpp
    src/pbrt/gpu/media.cpp
    src/pbrt/gpu/pathintegrator.cpp
    src/pbrt/gpu/subsurface.cpp
 )

 SET_SOURCE_FILES_PROPERTIES (
  src/pbrt/bsdf.cpp
  src/pbrt/bssrdf.cpp
  src/pbrt/bxdfs.cpp
  src/pbrt/cameras.cpp
  src/pbrt/film.cpp
  src/pbrt/filters.cpp
#  src/pbrt/genscene.cpp
  src/pbrt/interaction.cpp
  src/pbrt/lights.cpp
  src/pbrt/lightsamplers.cpp
  src/pbrt/materials.cpp
#  src/pbrt/media.cpp
  src/pbrt/options.cpp
#  src/pbrt/paramdict.cpp
#  src/pbrt/parser.cpp
  src/pbrt/pbrt.cpp
  src/pbrt/samplers.cpp
  src/pbrt/shapes.cpp
  src/pbrt/textures.cpp

  src/pbrt/util/bluenoise.cpp
  src/pbrt/util/check.cpp
  src/pbrt/util/color.cpp
  src/pbrt/util/colorspace.cpp
  src/pbrt/util/error.cpp
#  src/pbrt/util/file.cpp
#  src/pbrt/util/float.cpp
#  src/pbrt/util/image.cpp
  src/pbrt/util/log.cpp
#  src/pbrt/util/loopsubdiv.cpp
  src/pbrt/util/lowdiscrepancy.cpp
  src/pbrt/util/math.cpp
#  src/pbrt/util/memory.cpp
  src/pbrt/util/mesh.cpp
#  src/pbrt/util/mipmap.cpp
#  src/pbrt/util/parallel.cpp
  src/pbrt/util/pmj02tables.cpp
  src/pbrt/util/primes.cpp
#  src/pbrt/util/print.cpp
#  src/pbrt/util/progressreporter.cpp
  src/pbrt/util/pstd.cpp
  src/pbrt/util/rng.cpp
  src/pbrt/util/sampling.cpp
  src/pbrt/util/scattering.cpp
  src/pbrt/util/sobolmatrices.cpp
  src/pbrt/util/spectrum.cpp
  src/pbrt/util/stats.cpp
#  src/pbrt/util/stbimage.cpp
#  src/pbrt/util/string.cpp
  src/pbrt/util/transform.cpp
  src/pbrt/util/vecmath.cpp

   ${PBRT_GPU_SOURCE}

   PROPERTIES LANGUAGE CUDA
 )
 cuda_compile_and_embed(PBRT_EMBEDDED_PTX src/pbrt/gpu/optix.cu)
endif()

###########################################################################
# pbrt libraries and executables

ADD_LIBRARY ( pbrt_lib STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_aces.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_rec2020.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_srgb.cpp
  ${PBRT_SOURCE}
  ${PBRT_UTIL_SOURCE}
  ${PBRT_GPU_SOURCE}
  ${PBRT_EMBEDDED_PTX}

  src/ext/gtest/gtest-all.cc
  src/ext/lodepng/lodepng.cpp
  src/ext/rply/rply.cpp
  )
ADD_SANITIZERS ( pbrt_lib )

IF (WIN32)
  # Avoid a name clash when building on Visual Studio
  SET_TARGET_PROPERTIES ( pbrt_lib PROPERTIES OUTPUT_NAME libpbrt )
ENDIF()

if (PBRT_CUDA_ENABLED)
  SET_PROPERTY ( TARGET pbrt_lib PROPERTY CUDA_SEPARABLE_COMPILATION ON )
endif()

SET(ALL_PBRT_LIBS
  pbrt_lib
  ${CMAKE_THREAD_LIBS_INIT}
  ${OPENEXR_LIBS}
  Ptex_static
  ${ZLIB_LIBRARY}
  double-conversion
  ${PBRT_CUDA_LIB}
)

IF (WIN32)
  SET(ALL_PBRT_LIBS ${ALL_PBTR_LIBS} wsock32 ws2_32)
ENDIF()

if (PROFILE_LIB)
  SET(ALL_PBRT_LIBS ${ALL_PBRT_LIBS} ${PROFILE_LIB})
endif()

ADD_EXECUTABLE( rgb2spec_opt src/pbrt/cmd/rgb2spec_opt.cpp )
TARGET_LINK_LIBRARIES (rgb2spec_opt ${CMAKE_THREAD_LIBS_INIT})

ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_aces.cpp
    COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_aces.cpp ACES2065_1 
    DEPENDS rgb2spec_opt)

ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_rec2020.cpp
    COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_rec2020.cpp REC2020
    DEPENDS rgb2spec_opt)

ADD_CUSTOM_COMMAND(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_srgb.cpp
    COMMAND rgb2spec_opt 64 ${CMAKE_CURRENT_BINARY_DIR}/rgbspectrum_srgb.cpp sRGB
    DEPENDS rgb2spec_opt)

# Main renderer
ADD_EXECUTABLE ( pbrt_exe src/pbrt/cmd/pbrt.cpp )
ADD_SANITIZERS ( pbrt_exe )
SET_TARGET_PROPERTIES ( pbrt_exe PROPERTIES OUTPUT_NAME pbrt )
TARGET_LINK_LIBRARIES ( pbrt_exe ${ALL_PBRT_LIBS} )

# Tools
ADD_EXECUTABLE ( imgtool src/pbrt/cmd/imgtool.cpp src/ext/skymodel/ArHosekSkyModel.c )
ADD_SANITIZERS ( imgtool )
TARGET_LINK_LIBRARIES ( imgtool ${ALL_PBRT_LIBS} )

ADD_EXECUTABLE ( obj2pbrt src/pbrt/cmd/obj2pbrt.cpp )
ADD_SANITIZERS ( obj2pbrt )

ADD_EXECUTABLE ( cyhair2pbrt src/pbrt/cmd/cyhair2pbrt.cpp )
ADD_SANITIZERS ( cyhair2pbrt )

if (PBRT_BUILD_PYBIND AND PYTHONLIBS_FOUND)
  pybind11_add_module(pbrt
          src/pbrt/core/pybind.cpp
          src/pbrt/geometry/geometry_pybind.cpp
          src/pbrt/sampling/lowdiscrepancy_pybind.cpp
          src/pbrt/sampling/sampling_pybind.cpp
          src/pbrt/sampling/sobol_pybind.cpp
          src/pbrt/samplers/samplers_pybind.cpp
          src/pbrt/shapes/shapes_pybind.cpp
          src/pbrt/spectrum/spectrum_pybind.cpp
          src/pbrt/util/util_pybind.cpp
  )
  target_link_libraries(pbrt PUBLIC ${ALL_PBRT_LIBS} )
endif ()

# Unit test

SET ( PBRT_TEST_SOURCE
  src/pbrt/bsdfs_test.cpp
  src/pbrt/filters_test.cpp
  src/pbrt/lights_test.cpp
  src/pbrt/lightsamplers_test.cpp
  src/pbrt/media_test.cpp
  src/pbrt/parser_test.cpp
  src/pbrt/samplers_test.cpp
  src/pbrt/shapes_test.cpp

  src/pbrt/cpu/integrators_test.cpp

  src/pbrt/util/args_test.cpp
  src/pbrt/util/bits_test.cpp
  src/pbrt/util/color_test.cpp
  src/pbrt/util/containers_test.cpp
  src/pbrt/util/file_test.cpp
  src/pbrt/util/float_test.cpp
  src/pbrt/util/hash_test.cpp
  src/pbrt/util/image_test.cpp
  src/pbrt/util/math_test.cpp
  src/pbrt/util/octree_test.cpp
  src/pbrt/util/parallel_test.cpp
  src/pbrt/util/print_test.cpp
  src/pbrt/util/pstd_test.cpp
  src/pbrt/util/rng_test.cpp
  src/pbrt/util/sampling_test.cpp
  src/pbrt/util/spectrum_test.cpp
  src/pbrt/util/splines_test.cpp
  src/pbrt/util/transform_test.cpp
  src/pbrt/util/vecmath_test.cpp
  )

# FIXME HACK FOR NOW
ADD_EXECUTABLE ( pbrt_test src/pbrt/cmd/pbrt_test.cpp ${PBRT_TEST_SOURCE} )
ADD_SANITIZERS ( pbrt_test )
TARGET_LINK_LIBRARIES ( pbrt_test ${ALL_PBRT_LIBS} )

ADD_TEST ( pbrt_unit_test pbrt_test )

# Installation

INSTALL ( TARGETS
  pbrt_exe
  imgtool
  obj2pbrt
  cyhair2pbrt
  DESTINATION
  bin
  )

INSTALL ( TARGETS
  pbrt_lib
  DESTINATION
  lib
  )
