cmake_minimum_required(VERSION 3.15)
project(sacf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB CORRELATOR_SOURCES cpp/correlator/*.cpp)
file(GLOB DATASTRUCTURE_SOURCES cpp/datastructure/*.cpp)

# === Python extension modules ===

# Build correlator module -> sacf.correlator
pybind11_add_module(correlator ${CORRELATOR_SOURCES})
target_include_directories(correlator PRIVATE include)
install(TARGETS correlator DESTINATION sacf)

# Build datastructure module -> sacf.datastructure
pybind11_add_module(datastructure ${DATASTRUCTURE_SOURCES})
target_include_directories(datastructure PRIVATE include)
install(TARGETS datastructure DESTINATION sacf)
