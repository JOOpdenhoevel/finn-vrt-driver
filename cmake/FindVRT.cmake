# First, find JsonCpp, and LibXml2
find_package(jsoncpp REQUIRED)
find_package(LibXml2 REQUIRED)

set(VRT_POSSIBLE_ROOT_DIRS
  "$ENV{XILINX_VRT}"
  "/usr/local/"
  "/usr/"
)

MESSAGE(STATUS "VRT search path: ${VRT_POSSIBLE_ROOT_DIRS}")

set(VRT_INCDIR_SUFFIXES
  "vrt/include"
)

set(VRT_LIBDIR_SUFFIXES
  "lib"
)

find_path(VRT_INCLUDE_DIR "api/buffer.hpp" PATHS ${VRT_POSSIBLE_ROOT_DIRS} PATH_SUFFIXES ${VRT_INCDIR_SUFFIXES} NO_CMAKE_FIND_ROOT_PATH)
find_library(VRT_LIBRARY NAMES vrt PATHS ${VRT_POSSIBLE_ROOT_DIRS} PATH_SUFFIXES ${VRT_LIBDIR_SUFFIXES})

message(STATUS "VRT_INCLUDE_DIR      = ${VRT_INCLUDE_DIR}")
message(STATUS "VRT_LIBRARY          = ${VRT_LIBRARY}")

if(EXISTS ${VRT_INCLUDE_DIR} AND EXISTS ${VRT_LIBRARY})
  MESSAGE(STATUS "VRT found")
  set(VRT_FOUND ON)
else()
  message(WARNING "VRT NOT found")
  set(VRT_FOUND OFF)
endif()

add_library(vrt SHARED IMPORTED GLOBAL)
set_target_properties(vrt PROPERTIES IMPORTED_LOCATION "${VRT_LIBRARY}")
target_include_directories(vrt INTERFACE ${VRT_INCLUDE_DIR})
target_link_libraries(vrt INTERFACE zmq jsoncpp_lib LibXml2::LibXml2)
