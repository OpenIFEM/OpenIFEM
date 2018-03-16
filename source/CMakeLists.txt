# List all the source files here
set(TARGET_SRC fsi.cpp
               hyperelasticMaterial.cpp
               hyperelasticSolver.cpp
               linearElasticMaterial.cpp
               linearElasticSolver.cpp
               mpi_linearelasticity.cpp
               mpi_navierstokes.cpp
               navierstokes.cpp
               parameters.cpp
               utilities.cpp)

# List all the header files here
set(headers fsi.h
            hyperelasticMaterial.h
            hyperelasticSolver.h
            linearElasticMaterial.h
            linearElasticSolver.h
            material.h
            mpi_linearelasticity.h
            mpi_navierstokes.h
            navierstokes.h
            neoHookean.h
            parameters.h
            utilities.h)

# Prefix the header files with the path to include directory
foreach(header ${headers})
  list(APPEND TARGET_INC ${CMAKE_SOURCE_DIR}/include/${header})
endforeach()

# Build the library
add_library(openifem ${TARGET_SRC})
target_include_directories(openifem PUBLIC "${CMAKE_SOURCE_DIR}/include")
deal_ii_setup_target(openifem)
