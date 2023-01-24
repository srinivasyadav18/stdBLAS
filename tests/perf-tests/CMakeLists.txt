enable_testing()

set (perf_tests 
    add
)

foreach(perf_test ${perf_tests})
    set(target ${perf_test}_perf_test)
    add_executable(${target} ${perf_test}.cpp)

    target_link_libraries(${target} linalg HPX::hpx)
    add_test(NAME ${target} COMMAND ${CMAKE_CROSSCOMPILING_CMD} ${target})
endforeach()