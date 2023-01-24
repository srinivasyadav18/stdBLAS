//  Copyright (c) 2023 Srinivas Yadav

#include <complex>
#include <cstddef>

#include <iostream>

#include <experimental/linalg>
#include <experimental/mdspan>

#include "gtest/gtest.h"
#include "gtest_fixtures.hpp"

#include "helpers.hpp"

namespace {

template <class x_t, class y_t, class z_t>
void add_gold_solution(x_t x, y_t y, z_t z)
{
    for (std::size_t i = 0; i < x.extent(0); ++i)
    {
        for (std::size_t j = 0; j < x.extent(1); ++j)
        {
            z(i, j) = x(i, j) + y(i, j);
        }
    }
}

template <class ExPolicy, class x_t, class y_t, class z_t>
void hpx_blas_add_test_impl(ExPolicy policy, x_t x, y_t y, z_t z)
{
    namespace stdla = std::experimental::linalg;

    using value_type = typename x_t::value_type;
    const std::size_t extent0 = x.extent(0);
    const std::size_t extent1 = x.extent(1);

    // compute gold
    std::vector<value_type> gold(extent0 * extent1);
    using mdspan_t = std::experimental::mdspan<value_type,
        std::experimental::extents<::std::size_t, dynamic_extent, dynamic_extent>>;
    mdspan_t z_gold(gold.data(), extent0, extent1);
    add_gold_solution(x, y, z_gold);

    stdla::add(policy, x, y, z);

    if constexpr (std::is_same_v<value_type, float>)
    {
        for (std::size_t i = 0; i < extent0; ++i)
        {
            for (std::size_t j = 0; j < extent1; ++j)
            {
                EXPECT_FLOAT_EQ(z(i, j), z_gold(i, j));
            }
        }
    }

    if constexpr (std::is_same_v<value_type, double>)
    {
        for (std::size_t i = 0; i < extent0; ++i)
        {
            for (std::size_t j = 0; j < extent1; ++j)
            {
                EXPECT_DOUBLE_EQ(z(i, j), z_gold(i, j));
            }
        }
    }

    if constexpr (std::is_same_v<value_type, std::complex<double>>)
    {
        for (std::size_t i = 0; i < extent0; ++i)
        {
            for (std::size_t j = 0; j < extent1; ++j)
            {
                EXPECT_DOUBLE_EQ(z(i, j).real(), z_gold(i, j).real());
                EXPECT_DOUBLE_EQ(z(i, j).imag(), z_gold(i, j).imag());
            }
        }
    }
}
}    // namespace

TEST_F(blas2_signed_float_fixture, hpx_add)
{
    hpx_blas_add_test_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, B_e0e1, C_e0e1);
    hpx_blas_add_test_impl(hpx::execution::par, A_e0e1, B_e0e1, C_e0e1);
    hpx_blas_add_test_impl(hpx::execution::par_unseq, A_e0e1, B_e0e1, C_e0e1);
#if defined(HPX_HAVE_DATAPAR)
    hpx_blas_add_test_impl(hpx::execution::simd, A_e0e1, B_e0e1, C_e0e1);
    hpx_blas_add_test_impl(hpx::execution::par_simd, A_e0e1, B_e0e1, C_e0e1);
#endif
}

TEST_F(blas2_signed_double_fixture, hpx_add)
{
    hpx_blas_add_test_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, B_e0e1, C_e0e1);
    hpx_blas_add_test_impl(hpx::execution::par, A_e0e1, B_e0e1, C_e0e1);
    hpx_blas_add_test_impl(hpx::execution::par_unseq, A_e0e1, B_e0e1, C_e0e1);
    #if defined(HPX_HAVE_DATAPAR)
        hpx_blas_add_test_impl(hpx::execution::simd, A_e0e1, B_e0e1, C_e0e1);
        hpx_blas_add_test_impl(hpx::execution::par_simd, A_e0e1, B_e0e1, C_e0e1);
    #endif
}

TEST_F(blas2_signed_complex_double_fixture, hpx_add)
{
    using kc_t = std::complex<double>;
    using stdc_t = value_type;
    if (alignof(value_type) == alignof(kc_t))
    {
        hpx_blas_add_test_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, B_e0e1, C_e0e1);
        hpx_blas_add_test_impl(hpx::execution::par, A_e0e1, B_e0e1, C_e0e1);
        hpx_blas_add_test_impl(hpx::execution::par_unseq, A_e0e1, B_e0e1, C_e0e1);
#if defined(HPX_HAVE_DATAPAR)
        hpx_blas_add_test_impl(hpx::execution::simd, A_e0e1, B_e0e1, C_e0e1);
        hpx_blas_add_test_impl(hpx::execution::par_simd, A_e0e1, B_e0e1, C_e0e1);
#endif
    }
}
