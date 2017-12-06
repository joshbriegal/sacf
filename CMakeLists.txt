cmake_minimum_required(VERSION 3.8)
project(GACF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")

if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
			   message(FATAL_ERROR "Require at least gcc-4.7, version used:" ${CMAKE_CXX_COMPILER})

endif()

add_executable(GACF ${SOURCES})
