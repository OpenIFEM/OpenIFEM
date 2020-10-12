cmake_minimum_required(VERSION 2.8.12)

SET(CMAKE_CXX_STANDARD 17)

find_package(deal.II QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
if(NOT ${deal.II_FOUND})
  message(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
endif()

#
# Are all dependencies fulfilled?
#
if(NOT DEAL_II_WITH_UMFPACK) # keep in one line
  message(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_UMFPACK = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_UMFPACK = ${DEAL_II_WITH_UMFPACK}
which conflict with the requirements."
    )
endif()

if(NOT (DEAL_II_WITH_PETSC OR DEAL_II_WITH_TRILINOS) OR NOT DEAL_II_WITH_P4EST OR DEAL_II_PETSC_WITH_COMPLEX) # keep in one line
  message(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_PETSC = ON
    DEAL_II_PETSC_WITH_COMPLEX = OFF
    DEAL_II_WITH_P4EST = ON
or
    DEAL_II_WITH_TRILINOS = ON
    DEAL_II_WITH_P4EST = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_PETSC = ${DEAL_II_WITH_PETSC}
    DEAL_II_PETSC_WITH_COMPLEX = ${DEAL_II_PETSC_WITH_COMPLEX}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
which conflict with the requirements.
One or both of the aforementioned combinations of prerequisites are not met by your installation, but at least one is required for this tutorial step."
    )
endif()

if(NOT DEAL_II_WITH_METIS)
  message(FATAL_ERROR "
Error! OpenIFEM requires a deal.ii libray that was configured with the following options:
    DEAL_II_WITH_METIS = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_METIS = ${DEAL_II_WITH_METIS}
which conflict with the requirements.
The prerequisite are not met by your installation, but this is required for OpenIFEM."
   )
endif()

deal_ii_initialize_cached_variables()

project(OpenIFEM)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

option(OPENIFEM_WITH_rkpm-rk4 "Build with rkpm-rk4" OFF)
set(EIGEN3_INCLUDE_DIR "" CACHE PATH "Path to Eigen3 include directory")
if(OPENIFEM_WITH_rkpm-rk4)
  set(rkpm-rk4_DIR "" CACHE PATH "Path to rkpm-rk4 build directory")
  find_package(rkpm-rk4 REQUIRED)
endif()

option(OPENIFEM_WITH_shell-element "Build with shell-element" OFF)
set(LIBMESH_INCLUDE_DIR "" CACHE PATH "Path to LibMesh include directory")
if (OPENIFEM_WITH_shell-element)
  set(shell-element_DIR "" CACHE PATH "Path to shell-element build directory")
  find_package(shell-element REQUIRED)
endif()

if(OPENIFEM_WITH_rkpm-rk4)
  if (NOT rkpm-rk4_FOUND)
    message(FATAL_ERROR "Error! Cannot find rkpm-rk4!")
  endif()
  if (NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Error! EIGEN3_INCLUDE_DIR not set! "
      "Ideally rkpm-rk4 should export Eigen3 but I don't know how to do that!")
  endif()
endif()

if (OPENIFEM_WITH_shell-element)
  if (NOT shell-element_FOUND)
    message(FATAL_ERROR "Error! Cannot find shell-element!")
  endif()
endif()

option(OPENIFEM_BUILD_TESTS "Build ctests along with OpenIFEM library" ON)
if (OPENIFEM_BUILD_TESTS)
  enable_testing()
endif()
add_subdirectory(source)
if (OPENIFEM_BUILD_TESTS)
  add_subdirectory(tests)
endif()
