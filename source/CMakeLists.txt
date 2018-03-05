cmake_minimum_required(VERSION 3.5.0)

include_directories(${CMAKE_SOURCE_DIR}/include/)

file(GLOB_RECURSE TARGET_SRC "*.cpp")
file(GLOB_RECURSE TARGET_INC "*.h")
set(TARGET_SRC ${TARGET_SRC} ${TARGET_INC})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/libs)
add_library(openifem SHARED ${TARGET_SRC})
DEAL_II_SETUP_TARGET(openifem)
