cmake_minimum_required(VERSION 3.5)
project(CompareSparseLibs)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# GSL lib
FIND_PACKAGE(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS} ${GSLCBLAS_INCLUDE_DIRS})
add_definitions(${GSL_DEFINITIONS})
# EIGEN
include_directories( "/usr/include/eigen3/" )


set(SOURCE_FILES main.cpp)
add_executable(CompareSparseLibs ${SOURCE_FILES})
target_link_libraries(CompareSparseLibs ${GSL_LIBRARIES})
