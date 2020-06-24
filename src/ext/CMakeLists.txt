
cmake_minimum_required (VERSION 3.12)

###########################################################################
# stb

set (STB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)

###########################################################################
# filesystem

set (FILESYSTEM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/filesystem PARENT_SCOPE)

###########################################################################
# zlib

find_package ( ZLIB )
if (NOT ZLIB_FOUND)
  # Build zlib
  set (ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  set (ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  add_subdirectory (src/ext/zlib)

  set (ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/ext/zlib")
  set (ZLIB_LIBRARY zlibstatic)
  set_property (TARGET zlibstatic PROPERTY FOLDER "ext")
  set (ZLIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/src/ext/zlib" PARENT_SCOPE)
endif ()
set (ZLIB_INCLUDE ${ZLIB_INCLUDE_DIRS} PARENT_SCOPE)

###########################################################################
# OpenEXR

set (ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set (OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set (OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
set (ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
set (PYILMBASE_ENABLE             OFF CACHE BOOL " " FORCE)
set (OPENEXR_VIEWERS_ENABLE       OFF CACHE BOOL " " FORCE)

add_subdirectory (openexr)

set_property (TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups IlmThread Half Iex Imath IlmImf PROPERTY FOLDER "ext/OpenEXR")

set (OPENEXR_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Imath
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Half
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Iex
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/OpenEXR/IlmImf
  ${CMAKE_BINARY_DIR}/src/ext/openexr/IlmBase/config
  ${CMAKE_BINARY_DIR}/src/ext/openexr/OpenEXR/config
PARENT_SCOPE
)

if (WIN32)
  set (OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half ${ZLIB_LIBRARY} PARENT_SCOPE)
else ()
  set (OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half PARENT_SCOPE)
endif ()

###########################################################################
# ptex

set (PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)

set (CMAKE_MACOSX_RPATH 1)
if ( WIN32 )
  add_definitions ( /D PTEX_STATIC)
endif ()

add_subdirectory (ptex)

set_property (TARGET Ptex_static ptxinfo halftest ftest rtest wtest PROPERTY FOLDER "ext/ptex")

set (PTEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex PARENT_SCOPE)

###########################################################################
# double-conversion

add_subdirectory (double-conversion)

set (DOUBLE_CONVERSION_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/double-conversion PARENT_SCOPE)

set_property (TARGET double-conversion PROPERTY FOLDER "ext")

