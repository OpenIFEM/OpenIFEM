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

option(OPENIFEM_WITH_MFREE_IWF "Build with mfree_iwf" OFF)
if(OPENIFEM_WITH_MFREE_IWF)
  set(MFREE_IWF_DIR "" CACHE PATH "Path to mfree_iwf build directory")
  find_package(MFREE_IWF REQUIRED)
  # We should not need find glm because it is exported by mfree_iwf,
  # but for some reason DRP cannot find glm include directory without it.
  find_package(glm REQUIRED)
endif()

if(OPENIFEM_WITH_MFREE_IWF AND NOT MFREE_IWF_FOUND)
  message(FATAL_ERROR "Error! Cannot find mfree_iwf!")
endif()

enable_testing()
add_subdirectory(source)
add_subdirectory(tests)
