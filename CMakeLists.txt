cmake_minimum_required(VERSION 3.18)
project(ndarray CXX)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

add_subdirectory(src)
add_library(GREEN::NDARRAY ALIAS ndarray)

option(Build_Tests "Build tests" ON)
if (Build_Tests)
    enable_testing()
    add_subdirectory(test)
endif (Build_Tests)



