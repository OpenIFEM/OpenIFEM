# List all the serial tests
set(serial_tests fluid_cylinder
                 fluid_pipe
                 fluid_pressure_driven
                 fsi_gravity
                 solid_bending_beam
                 solid_gravity
                 beam_bending_NeoHookean)

# List all the mpi tests
set(mpi_tests fluid_mpi_cylinder)

set(tests ${serial_tests})
list(APPEND tests ${mpi_tests})

# Create a subdirectory and add an executible for each test
foreach(test ${tests})
  set(input ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.prm)
  set(output ${CMAKE_CURRENT_BINARY_DIR}/${test})
  file(MAKE_DIRECTORY ${output})
  add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.cpp)
  target_include_directories(${test} PUBLIC "${CMAKE_SOURCE_DIR}/include")
  deal_ii_setup_target(${test})
  target_link_libraries(${test} openifem)
  list(FIND test ${mpi_tests} index)
  if(${index} GREATER -1)
    # FIXME: it is not good practice to specify the number of processors in this way
    add_test(NAME ${test} COMMAND mpirun -n 2 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test} ${input} WORKING_DIRECTORY ${output})
  else()
    add_test(NAME ${test} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test} ${input} WORKING_DIRECTORY ${output})
  endif()
endforeach()
