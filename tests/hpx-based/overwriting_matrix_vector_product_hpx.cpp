
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class A_t, class x_t, class y_t>
void gemv_gold_solution(A_t A, x_t x, y_t y)
{
  for (std::size_t i=0; i<A.extent(0); ++i){
    y(i) = typename y_t::value_type{};
    for (std::size_t j=0; j<A.extent(1); ++j){
      y(i) += A(i,j) * x(j);
    }
  }
}

template<class ExPolicy, class A_t, class x_t, class y_t>
void hpx_blas_overwriting_gemv_impl(ExPolicy policy, A_t A, x_t x, y_t y)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename A_t::value_type;
  const std::size_t extent0 = A.extent(0);
  const std::size_t extent1 = A.extent(1);

  // copy operands before running the kernel
  auto A_preKernel = hpxtesting::create_stdvector_and_copy_rowwise(A);
  auto x_preKernel = hpxtesting::create_stdvector_and_copy(x);
  auto y_preKernel = hpxtesting::create_stdvector_and_copy(y);

  // compute y gold gemv
  std::vector<value_type> gold(y.extent(0));
  using mdspan_t = mdspan<value_type, extents<std::size_t, dynamic_extent>>;
  mdspan_t y_gold(gold.data(), y.extent(0));
  gemv_gold_solution(A, x, y_gold);

  stdla::matrix_vector_product(policy, A, x, y);

  // after kernel, A,x should be unchanged, y should be equal to y_gold.
  // note that for A we need to visit all elements rowwise
  // since that is how we stored above the preKernel values

  if constexpr(std::is_same_v<value_type, float>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_FLOAT_EQ(x(j), x_preKernel[j]);
    }

    // check A and y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_NEAR(y(i), y_gold(i), 1e-2);
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_FLOAT_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, double>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_DOUBLE_EQ(x(j), x_preKernel[j]);
    }

    // check A and y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_NEAR(y(i), y_gold(i), 1e-9);
      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j), A_preKernel[count++]);
      }
    }
  }

  else if constexpr(std::is_same_v<value_type, std::complex<double>>){
    // check x
    for (std::size_t j=0; j<extent1; ++j){
      EXPECT_DOUBLE_EQ(x(j).real(), x_preKernel[j].real());
      EXPECT_DOUBLE_EQ(x(j).imag(), x_preKernel[j].imag());
    }

    // check A and y
    std::size_t count=0;
    for (std::size_t i=0; i<extent0; ++i){
      EXPECT_NEAR(y(i).real(), y_gold(i).real(), 1e-9);
      EXPECT_NEAR(y(i).imag(), y_gold(i).imag(), 1e-9);

      for (std::size_t j=0; j<extent1; ++j){
	EXPECT_DOUBLE_EQ(A(i,j).real(), A_preKernel[count].real());
	EXPECT_DOUBLE_EQ(A(i,j).imag(), A_preKernel[count++].imag());
      }
    }
  }

}
}//end anonym namespace

TEST_F(blas2_signed_float_fixture, kokkos_overwriting_matrix_vector_product)
{
  hpx_blas_overwriting_gemv_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, x_e1, x_e0);
  hpx_blas_overwriting_gemv_impl(hpx::execution::par, A_e0e1, x_e1, x_e0);
  hpx_blas_overwriting_gemv_impl(hpx::execution::par_unseq, A_e0e1, x_e1, x_e0);
  // hpx_blas_overwriting_gemv_impl(A_e0e1, x_e1, x_e0);
}

TEST_F(blas2_signed_double_fixture, hpx_overwriting_matrix_vector_product)
{
  hpx_blas_overwriting_gemv_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, x_e1, x_e0);
  hpx_blas_overwriting_gemv_impl(hpx::execution::par, A_e0e1, x_e1, x_e0);
  hpx_blas_overwriting_gemv_impl(hpx::execution::par_unseq, A_e0e1, x_e1, x_e0);
}

TEST_F(blas2_signed_complex_double_fixture, kokkos_overwriting_matrix_vector_product)
{
  using kc_t = std::complex<double>;
  using stdc_t = value_type;
  if constexpr (alignof(value_type) == alignof(kc_t)){
    hpx_blas_overwriting_gemv_impl(HPXKernelsSTD::hpx_exec<>(), A_e0e1, x_e1, x_e0);
    hpx_blas_overwriting_gemv_impl(hpx::execution::par, A_e0e1, x_e1, x_e0);
    hpx_blas_overwriting_gemv_impl(hpx::execution::par_unseq, A_e0e1, x_e1, x_e0);
  }
}
