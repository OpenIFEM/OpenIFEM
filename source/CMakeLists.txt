# List all the source files here
set(TARGET_SRC fluid_solver.cpp
               fsi.cpp
               hyper_elastic_material.cpp
               hyper_elasticity.cpp
               insim.cpp
               insimex.cpp
               linear_elastic_material.cpp
               linear_elasticity.cpp
               mpi_fluid_solver.cpp
               mpi_fsi.cpp
               mpi_hyper_elasticity.cpp
               mpi_insimex.cpp
               mpi_linear_elasticity.cpp
               mpi_insim.cpp
               mpi_scnsim.cpp
               mpi_shared_hyper_elasticity.cpp
               mpi_shared_linear_elasticity.cpp
               mpi_shared_solid_solver.cpp
               mpi_solid_solver.cpp
               parameters.cpp
               preconditioner_pilut.cpp
               scnsim.cpp
               solid_solver.cpp
               utilities.cpp)

# List all the header files here
set(headers fluid_solver.h
            fsi.h
            hyper_elastic_material.h
            hyper_elasticity.h
            insim.h
            insimex.h
            linear_elastic_material.h
            linear_elasticity.h
            material.h
            mpi_fluid_solver.h
            mpi_fsi.h
            mpi_hyper_elasticity.h
            mpi_insimex.h
            mpi_linear_elasticity.h
            mpi_insim.h
            mpi_scnsim.h
            mpi_shared_hyper_elasticity.h
            mpi_shared_linear_elasticity.h
            mpi_shared_solid_solver.h
            mpi_solid_solver.h
            neoHookean.h
            parameters.h
            preconditioner_pilut.h
            scnsim.h
            solid_solver.h
            utilities.h)

# Prefix the header files with the path to include directory
foreach(header ${headers})
  list(APPEND TARGET_INC ${CMAKE_SOURCE_DIR}/include/${header})
endforeach()

# Build the library
add_library(openifem ${TARGET_SRC})
target_include_directories(openifem PUBLIC "${CMAKE_SOURCE_DIR}/include")
deal_ii_setup_target(openifem)
