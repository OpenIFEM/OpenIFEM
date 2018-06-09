# List all the serial tests
set(serial_tests acoustic_duct_wave
                 acoustic_pml
                 fluid_cylinder
                 fluid_cylinder_insimex
                 fluid_gravity
                 fluid_pipe
                 fluid_pressure_driven
                 fsi_gravity
                 fsi_leaflet
                 solid_beam_bending_linearelastic
                 solid_beam_bending_NeoHookean
                 solid_gravity_hyperelastic
                 solid_gravity_linearelastic)

# List all the mpi tests
set(mpi_tests acoustic_duct_wave_mpi
              acoustic_pml_mpi
              fluid_cylinder_mpi
              fluid_cylinder_mpi_insimex
              fluid_pipe_mpi
              solid_beam_bending_mpi_linearelastic
              solid_beam_bending_mpi_NeoHookean)

set(tests ${serial_tests} ${mpi_tests})

# Create a subdirectory and add an executible for each test
foreach(test ${tests})
  set(input ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.prm)
  set(output ${CMAKE_CURRENT_BINARY_DIR}/${test})
  file(MAKE_DIRECTORY ${output})
  add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.cpp)
  target_include_directories(${test} PUBLIC "${CMAKE_SOURCE_DIR}/include")
  deal_ii_setup_target(${test})
  target_link_libraries(${test} openifem)
  list(FIND mpi_tests ${test} index)
  if(${index} GREATER -1)
    # FIXME: it is not good practice to specify the number of processors in this way
    add_test(NAME ${test} COMMAND mpirun -n 2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test} ${input} WORKING_DIRECTORY ${output})
  else()
    add_test(NAME ${test} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test} ${input} WORKING_DIRECTORY ${output})
  endif()
endforeach()
