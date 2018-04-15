# List all the tests
set(tests beam_bending_NeoHookean
	  beam_bending_MooneyRivlin
          )

# Create a subdirectory and add an executible for each test
foreach(test ${tests})
  set(input ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.prm)
  set(output ${CMAKE_CURRENT_BINARY_DIR}/${test})
  file(MAKE_DIRECTORY ${output})
  add_executable(${test} ${CMAKE_CURRENT_SOURCE_DIR}/${test}/${test}.cpp)
  target_include_directories(${test} PUBLIC "${CMAKE_SOURCE_DIR}/include")
  deal_ii_setup_target(${test})
  target_link_libraries(${test} openifem)
  add_test(NAME ${test} COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test} ${input} WORKING_DIRECTORY ${output})
endforeach()
