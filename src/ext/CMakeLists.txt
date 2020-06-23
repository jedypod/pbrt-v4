
cmake_minimum_required(VERSION 3.12)

###########################################################################
# stb

set(STB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/stb PARENT_SCOPE)

###########################################################################
# filesystem

set(FILESYSTEM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/filesystem PARENT_SCOPE)

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
  set(ZLIB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${ZLIB_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/src/ext/zlib" PARENT_SCOPE)
ENDIF()
set(ZLIB_INCLUDE ${ZLIB_INCLUDE_DIRS} PARENT_SCOPE)

###########################################################################
# OpenEXR

SET(ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
SET(OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
SET(ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
SET(PYILMBASE_ENABLE             OFF CACHE BOOL " " FORCE)
SET(OPENEXR_VIEWERS_ENABLE       OFF CACHE BOOL " " FORCE)

ADD_SUBDIRECTORY(openexr)

SET_PROPERTY(TARGET IexMath eLut toFloat b44ExpLogTable dwaLookups IlmThread Half Iex Imath IlmImf PROPERTY FOLDER "ext/OpenEXR")

set(OPENEXR_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Imath
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Half
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/IlmBase/Iex
  ${CMAKE_CURRENT_SOURCE_DIR}/openexr/OpenEXR/IlmImf
  ${CMAKE_BINARY_DIR}/src/ext/openexr/IlmBase/config
  ${CMAKE_BINARY_DIR}/src/ext/openexr/OpenEXR/config
PARENT_SCOPE
)

IF(WIN32)
  SET(OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half ${ZLIB_LIBRARY} PARENT_SCOPE)
ELSE()
  SET(OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half PARENT_SCOPE)
ENDIF()

###########################################################################
# ptex

SET(PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)

SET(CMAKE_MACOSX_RPATH 1)
IF ( WIN32 )
  ADD_DEFINITIONS ( /D PTEX_STATIC)
ENDIF ()

ADD_SUBDIRECTORY(ptex)

SET_PROPERTY(TARGET Ptex_static ptxinfo halftest ftest rtest wtest PROPERTY FOLDER "ext/ptex")

SET(PTEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex PARENT_SCOPE)

###########################################################################
# double-conversion

ADD_SUBDIRECTORY(double-conversion)
set(DOUBLE_CONVERSION_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/double-conversion PARENT_SCOPE)

SET_PROPERTY(TARGET double-conversion PROPERTY FOLDER "ext")

