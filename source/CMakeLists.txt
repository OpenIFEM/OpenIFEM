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
               mpi_shared_hypo_elasticity.cpp
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
            mpi_shared_hypo_elasticity.h
            mpi_shared_linear_elasticity.h
            mpi_shared_solid_solver.h
            mpi_solid_solver.h
            neoHookean.h
            parameters.h
            preconditioner_pilut.h
            scnsim.h
            solid_solver.h
            utilities.h)

if(OPENIFEM_WITH_MFREE_IWF)
  list(APPEND TARGET_SRC "hypo_elasticity.cpp")
  list(APPEND headers "hypo_elasticity.h")
endif()

# Build the library
add_library(openifem ${TARGET_SRC})
target_include_directories(openifem PUBLIC "${CMAKE_SOURCE_DIR}/include")
if(OPENIFEM_WITH_MFREE_IWF)
  target_include_directories(openifem PRIVATE ${MFREE_IWF_INCLUDE_DIR})
  target_include_directories(openifem PRIVATE ${GLM_INCLUDE_DIRS})
  target_link_libraries(openifem ${MFREE_IWF_LIBRARY})
endif()
deal_ii_setup_target(openifem)
