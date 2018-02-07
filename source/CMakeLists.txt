cmake_minimum_required(VERSION 2.8.12)

include_directories(${CMAKE_SOURCE_DIR}/include/)

file(GLOB_RECURSE TARGET_SRC "*.cpp")
file(GLOB_RECURSE TARGET_INC "*.h")
set(TARGET_SRC ${TARGET_SRC} ${TARGET_INC})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(ifem ${TARGET_SRC})
DEAL_II_SETUP_TARGET(ifem)
