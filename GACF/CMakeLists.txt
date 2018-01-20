cmake_minimum_required(VERSION 3.8)
project(GACF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)
find_package(pybind11 REQUIRED)

file(GLOB DATASTRUCTURE datastructure/*.cpp)
file(GLOB CORRELATOR correlator/*.cpp)


if(CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.7)
			   message(FATAL_ERROR "Require at least gcc-4.7, version used:" ${CMAKE_CXX_COMPILER})

endif()

#add_library(_datastructure ${DATASTRUCTURE})
#add_library(correlator MODULE ${CORRELATOR})

#target_link_libraries(correlator _datastructure)

#pybind11_add_module(datastructure ${DATASTRUCTURE})

#set_target_properties(datastructure PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")
#set_target_properties(correlator PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}")


pybind11_add_module(datastructure ${DATASTRUCTURE})
pybind11_add_module(correlator ${CORRELATOR})
