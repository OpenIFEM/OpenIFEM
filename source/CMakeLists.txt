# List all the source files here
set(TARGET_SRC
               hyperelasticMaterial.cpp
               hyperelasticSolver.cpp
               linearElasticMaterial.cpp
               linearElasticSolver.cpp
               mpi_linearelasticity.cpp
               navierstokes.cpp
               parameters.cpp
               utilities.cpp)

# List all the header files here
set(headers 
            hyperelasticMaterial.h
            hyperelasticSolver.h
            linearElasticMaterial.h
            linearElasticSolver.h
	    mpi_linearelasticity.h
            material.h
            navierstokes.h
            neoHookean.h
	    mooneyrivlin.h
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
