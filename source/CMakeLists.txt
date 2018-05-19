# List all the source files here
set(TARGET_SRC fluidSolver.cpp
               fsi.cpp
               hyperelasticMaterial.cpp
               hyperelasticSolver.cpp
               insim.cpp
               insimex.cpp
               linearElasticMaterial.cpp
               linearElasticSolver.cpp
               mpi_fluid_solver.cpp
               mpi_hyper_elasticity.cpp
               mpi_insimex.cpp
               mpi_linear_elasticity.cpp
               mpi_insim.cpp
               mpi_scnsim.cpp
               mpi_solid_solver.cpp
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
            mpi_fluid_solver.h
            mpi_hyper_elasticity.h
            mpi_insimex.h
            mpi_linear_elasticity.h
            mpi_insim.h
            mpi_scnsim.h
            mpi_solid_solver.h
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
