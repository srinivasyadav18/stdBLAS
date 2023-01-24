
#include "gtest_fixtures.hpp"
#include "helpers.hpp"

namespace
{

template<class x_t, class y_t, class T>
auto dot_gold_solution(x_t x, y_t y, T initValue, bool useInit)
{

  T result = {};
  for (std::size_t i=0; i<x.extent(0); ++i){
    result += x(i) * y(i);
  }

  if (useInit) result += initValue;
  return result;
}

template<class ExPolicy, class x_t, class y_t, class T>
void hpx_blas1_dot_test_impl(ExPolicy policy, x_t x, y_t y, T initValue, bool useInit)
{
  namespace stdla = std::experimental::linalg;

  using value_type = typename x_t::value_type;
  const std::size_t extent = x.extent(0);

  // copy x and y to verify they are not changed after kernel
  auto x_preKernel = hpxtesting::create_stdvector_and_copy(x);
  auto y_preKernel = hpxtesting::create_stdvector_and_copy(y);

  // compute gold
  const T gold = dot_gold_solution(x, y, initValue, useInit);

  T result = {};
  if (useInit){
    result = stdla::dot(policy,
			x, y, initValue);
  }else{
    result = stdla::dot(policy,
			x, y);
  }

  if constexpr(std::is_same_v<value_type, float>){
    // cannot use EXPECT_FLOAT_EQ because
    // in some cases that fails on third digit or similr
    EXPECT_NEAR(result, gold, 1e-2);
  }

  if constexpr(std::is_same_v<value_type, double>){
    // similarly to float
    EXPECT_NEAR(result, gold, 1e-9);
  }

  if constexpr(std::is_same_v<value_type, std::complex<double>>){
    EXPECT_NEAR(result.real(), gold.real(), 1e-9);
    EXPECT_NEAR(result.imag(), gold.imag(), 1e-9);
  }

  // x,y should not change after kernel
  for (std::size_t i=0; i<extent; ++i){
    EXPECT_TRUE(x(i) == x_preKernel[i]);
    EXPECT_TRUE(y(i) == y_preKernel[i]);
  }
}
}//end anonym namespace

TEST_F(blas1_signed_float_fixture, hpx_dot_noinitvalue)
{
  // hpx_blas1_dot_test_impl(HPXKernelsSTD::hpx_exec<>(), x, y, static_cast<float>(0), false);
  hpx_blas1_dot_test_impl(hpx::execution::par, x, y, static_cast<float>(0), false);
  hpx_blas1_dot_test_impl(hpx::execution::par_unseq, x, y, static_cast<float>(0), false);
#if defined(HPX_HAVE_DATAPAR)
  hpx_blas1_dot_test_impl(hpx::execution::simd, x, y, static_cast<float>(0), false);
  hpx_blas1_dot_test_impl(hpx::execution::par_simd, x, y, static_cast<float>(0), false);
#endif
}

// TEST_F(blas1_signed_float_fixture, hpx_dot_initvalue)
// {
//   hpx_blas1_dot_test_impl(x, y, static_cast<float>(3), true);
// }

// TEST_F(blas1_signed_double_fixture, hpx_dot_noinitvalue)
// {
//   hpx_blas1_dot_test_impl(x, y, static_cast<double>(0), false);
// }

// TEST_F(blas1_signed_double_fixture, hpx_dot_initvalue)
// {
//   hpx_blas1_dot_test_impl(x, y, static_cast<double>(5), true);
// }

// TEST_F(blas1_signed_complex_double_fixture, hpx_dot_noinitvalue)
// {
//   using kc_t   = HPX::complex<double>;
//   using stdc_t = value_type;
//   if constexpr (alignof(value_type) == alignof(kc_t)){
//     const value_type init{0., 0.};
//     hpx_blas1_dot_test_impl(x, y, init, false);
//   }
// }

// TEST_F(blas1_signed_complex_double_fixture, hpx_dot_initvalue)
// {
//   using kc_t   = HPX::complex<double>;
//   using stdc_t = value_type;
//   if constexpr (alignof(value_type) == alignof(kc_t)){
//     const value_type init{-2., 4.};
//     hpx_blas1_dot_test_impl(x, y, init, true);
//   }
// }
