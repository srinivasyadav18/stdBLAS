enable_testing()

set (perf_tests 
    add
    dot
    gemv
    gemv_blocked
    gemm
    gemm_all
    gemm_hpx
    gemm_blocked
    gemm_blocked_mdspan
    gemm_blocked_mdspan_hpx
)

foreach(perf_test ${perf_tests})
    set(target ${perf_test}_perf_test)
    add_executable(${target} ${perf_test}.cpp)

    target_link_libraries(${target} linalg HPX::hpx)
    add_test(NAME ${target} COMMAND ${CMAKE_CROSSCOMPILING_CMD} ${target})
endforeach()