
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_OVERWRITING_GEMM_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_OVERWRITING_GEMM_HPP_

#include <hpx/algorithm.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include "signal_hpx_impl_called.hpp"
#include "static_extent_match.hpp"

namespace HPXKernelsSTD {

void initalize_block(auto C);;
void multiply_blocks(auto C, auto const A, auto const B);

//
// overwriting gemm: C = alpha*A*B
//
template<class ExPolicy,
         class ElementType_A,
         class SizeType_A, ::std::size_t numRows_A, ::std::size_t numCols_A,
         class Layout_A,
         class Accessor_A,
         class ElementType_B,
         class SizeType_B, ::std::size_t numRows_B, ::std::size_t numCols_B,
         class Layout_B,
         class Accessor_B,
         class ElementType_C,
         class SizeType_C, ::std::size_t numRows_C, ::std::size_t numCols_C,
         class Layout_C,
         class Accessor_C>
void matrix_product(
  hpx_exec<ExPolicy>&& policy_,
  std::experimental::mdspan<ElementType_A, std::experimental::extents<SizeType_A, numRows_A, numCols_A>, Layout_A, Accessor_A> A,
  std::experimental::mdspan<ElementType_B, std::experimental::extents<SizeType_B, numRows_B, numCols_B>, Layout_B, Accessor_B> B,
  std::experimental::mdspan<ElementType_C, std::experimental::extents<SizeType_C, numRows_C, numCols_C>, Layout_C, Accessor_C> C)
{
  auto policy = policy_.policy_;
  // preconditions
  if ( A.extent(1) != B.extent(0) ){
    throw std::runtime_error("HPXBlas: gemm_C_AB_product: A.extent(1) != B.extent(0) ");
  }
  if ( A.extent(0) != C.extent(0) ){
    throw std::runtime_error("HPXBlas: gemm_C_AB_product: A.extent(0) != C.extent(0) ");
  }
  if ( B.extent(1) != C.extent(1) ){
    throw std::runtime_error("HPXBlas: gemm_C_AB_product: B.extent(1) != C.extent(1) ");
  }

  // mandates
  Impl::static_extent_match(A.static_extent(1), B.static_extent(0));
  Impl::static_extent_match(A.static_extent(0), C.static_extent(0));
  Impl::static_extent_match(B.static_extent(1), C.static_extent(1));

  std::cout << "called\n";
  Impl::signal_hpx_impl_called("gemm_C_AB_product");

  // hpx::experimental::for_loop(
  //   hpx::execution::experimental::to_non_simd(policy),
  //   SizeType_C(0), C.extent(0), [&](auto i){
  //         for (std::size_t j = 0; j < C.extent(1); ++j)
  //         {
  //           C(i,j) = ElementType_C{};
  //           for (std::size_t k = 0; k < A.extent(1); ++k) {
  //             C(i,j) += A(i,k) * B(k,j);
  //           }
  //         }
  //   });

  // for (std::size_t i = 0; i < C.extent(0); ++i) {
  //   for (std::size_t j = 0; j < C.extent(1); ++j) {
  //     C(i,j) = ElementType_C{};
  //     // for (std::size_t k = 0; k < A.extent(1); ++k) {
  //     //   C(i,j) += A(i,k) * B(k,j);
  //     // }
  //   }
  // }

  // for (std::size_t i = 0; i < C.extent(0); i += 8) {
  //   for (std::size_t j = 0; j < C.extent(1); j += 8) {
  //       for (std::size_t ii = i; ii < i + 8; ii++) 
  //         for (std::size_t jj = j; jj < j + 8; jj++) 
  //           C(ii, jj) = ElementType_C{};
  //     for (std::size_t k = 0; k < A.extent(1); k += 8) {

  //       for (std::size_t ii = i; ii < i + 8; ii++) {
  //         for (std::size_t jj = j; jj < j + 8; jj++) {
  //           for (std::size_t kk = k; kk < k + 8; kk++) {
  //             C(ii,jj) += A(ii,kk) * B(kk,jj);
  //           }
  //         }
  //       }
  //     }

  //   }
  // }

  const int block_size = 8;
  // for (std::size_t i = 0; i < C.extent(0); i += block_size)
  // {
  //   for (std::size_t j = 0; j < C.extent(1); j += block_size)
  //   {
  //     auto C_ij_block = std::experimental::submdspan(C, std::pair{i, i+block_size}, std::pair{j, j+block_size});
  //     initalize_block(C_ij_block);
  //     for (std::size_t k = 0; k < A.extent(1); k += block_size)
  //     {
  //       auto A_ik_block = std::experimental::submdspan(A, std::pair{i, i+block_size}, std::pair{k, k+block_size});
  //       auto B_kj_block = std::experimental::submdspan(B, std::pair{k, k+block_size}, std::pair{j, j+block_size});
  //       multiply_blocks(C_ij_block, A_ik_block, B_kj_block);
  //     }
  //   }
  // }

  hpx::experimental::for_loop_strided(policy, std::size_t(0), C.extent(0), block_size, [&](std::size_t i){
    for (std::size_t j = 0; j < C.extent(1); j += block_size)
    {
    auto C_ij_block = std::experimental::submdspan(C, std::pair{i, i+block_size}, std::pair{j, j+block_size});
    initalize_block(C_ij_block);
    for (std::size_t k = 0; k < A.extent(1); k += block_size)
    {
        auto A_ik_block = std::experimental::submdspan(A, std::pair{i, i+block_size}, std::pair{k, k+block_size});
        auto B_kj_block = std::experimental::submdspan(B, std::pair{k, k+block_size}, std::pair{j, j+block_size});
        multiply_blocks(C_ij_block, A_ik_block, B_kj_block);
    }
    }
  });
}

void initalize_block(auto C)
{
  for (int i = 0; i < C.extent(0); i++)
  {
    for (int j = 0; j < C.extent(1); j++)
    {
      C(i, j) = {};
    }
  }
}

void multiply_blocks(auto C, auto const A, auto const B)
{
  for (std::size_t i = 0; i < C.extent(0); ++i) {
    for (std::size_t j = 0; j < C.extent(1); ++j) {
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
}

} 
#endif