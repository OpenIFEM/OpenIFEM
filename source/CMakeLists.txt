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
               mpi_fluid_solver_extractor.cpp
               mpi_fsi.cpp
               mpi_hyper_elasticity.cpp
               mpi_insimex.cpp
               mpi_linear_elasticity.cpp
               mpi_insim.cpp
               mpi_scnsex.cpp
               mpi_scnsim.cpp
               mpi_shared_hyper_elasticity.cpp
               mpi_shared_linear_elasticity.cpp
               mpi_shared_solid_solver.cpp
               mpi_solid_solver.cpp
               mpi_spalart_allmaras.cpp
               mpi_turbulence_model.cpp
               parameters.cpp
               preconditioner_pilut.cpp
               scnsim.cpp
               solid_solver.cpp
               utilities.cpp
               cv_fsi.cpp)

# List all the header files here
set(headers fluid_solver.h
            fsi.h
            hyper_elastic_material.h
            hyper_elasticity.h
            inheritance_macros.h
            insim.h
            insimex.h
            linear_elastic_material.h
            linear_elasticity.h
            material.h
            mpi_fluid_solver.h
            mpi_fluid_solver_extractor.h
            mpi_fsi.h
            mpi_hyper_elasticity.h
            mpi_insimex.h
            mpi_linear_elasticity.h
            mpi_insim.h
            mpi_scnsex.h
            mpi_scnsim.h
            mpi_shared_hyper_elasticity.h
            mpi_shared_linear_elasticity.h
            mpi_shared_solid_solver.h
            mpi_solid_solver.h
            mpi_spalart_allmaras.h
            mpi_turbulence_model.h
            neoHookean.h
            parameters.h
            preconditioner_pilut.h
            scnsim.h
            solid_solver.h
            utilities.h
            cv_fsi.h)

if(OPENIFEM_WITH_rkpm-rk4)
  list(APPEND TARGET_SRC "hypo_elasticity.cpp")
  list(APPEND headers "hypo_elasticity.h")
  list(APPEND TARGET_SRC "mpi_shared_hypo_elasticity.cpp")
  list(APPEND headers "mpi_shared_hypo_elasticity.h")
endif()

if (OPENIFEM_WITH_shell-element)
  list(APPEND TARGET_SRC "shell_solid_solver.cpp")
  list(APPEND headers "shell_solid_solver.h")
endif()

# Build the library
add_library(openifem ${TARGET_SRC})
target_include_directories(openifem PUBLIC "${CMAKE_SOURCE_DIR}/include")
if(OPENIFEM_WITH_rkpm-rk4)
  target_include_directories(openifem PUBLIC ${rkpm-rk4_INCLUDE_DIR})
  target_include_directories(openifem PUBLIC ${EIGEN3_INCLUDE_DIR})
  target_link_libraries(openifem ${rkpm-rk4_LIBRARY})
endif()
if(OPENIFEM_WITH_shell-element)
  target_include_directories(openifem PUBLIC ${shell-element_INCLUDE_DIR})
  target_include_directories(openifem PUBLIC ${LIBMESH_INCLUDE_DIR})
  target_link_libraries(openifem ${shell-element_LIBRARY} ${libmesh_LIBRARY})
endif()
deal_ii_setup_target(openifem)
