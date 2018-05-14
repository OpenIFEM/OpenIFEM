# List all the source files here
set(TARGET_SRC fluidSolver.cpp
               fsi.cpp
               hyperelasticMaterial.cpp
               hyperelasticSolver.cpp
               insim.cpp
               insimex.cpp
               linearElasticMaterial.cpp
               linearElasticSolver.cpp
               mpi_insimex.cpp
               mpi_linearelasticity.cpp
               mpi_navierstokes.cpp
               parameters.cpp
	       preconditionerPilut.cpp
               scnsim.cpp
               solidSolver.cpp
               utilities.cpp)

# List all the header files here
set(headers fluidSolver.h
            fsi.h
            hyperelasticMaterial.h
            hyperelasticSolver.h
            insim.h
            insimex.h
            linearElasticMaterial.h
            linearElasticSolver.h
            material.h
            mpi_insimex.h
            mpi_linearelasticity.h
            mpi_navierstokes.h
            neoHookean.h
            parameters.h
            preconditionerPilut.h
            scnsim.h
            solidSolver.h
            utilities.h)

# Prefix the header files with the path to include directory
foreach(header ${headers})
  list(APPEND TARGET_INC ${CMAKE_SOURCE_DIR}/include/${header})
endforeach()

# Build the library
add_library(openifem ${TARGET_SRC})
target_include_directories(openifem PUBLIC "${CMAKE_SOURCE_DIR}/include")
deal_ii_setup_target(openifem)
