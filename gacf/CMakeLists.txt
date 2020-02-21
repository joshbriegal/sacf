cmake_minimum_required(VERSION 3.8)
project(gacf)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_REQUIRED_FLAGS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(PYBIND11_CPP_STANDARD -std=c++14)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/include/pybind11)

file(GLOB DATASTRUCTURE ${CMAKE_SOURCE_DIR}/datastructure/*.cpp)
file(GLOB CORRELATOR ${CMAKE_SOURCE_DIR}/correlator/*.cpp)

message(STATUS "source directory ${CMAKE_SOURCE_DIR} yeet")

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
			   message(FATAL_ERROR "Require at least gcc-4.7, version used:" ${CMAKE_CXX_COMPILER})

endif()

pybind11_add_module(datastructure ${DATASTRUCTURE})
pybind11_add_module(correlator ${CORRELATOR})
