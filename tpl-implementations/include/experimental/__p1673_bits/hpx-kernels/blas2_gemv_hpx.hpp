
#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_GEMV_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_GEMV_HPP_

#include "signal_hpx_impl_called.hpp"
#include "static_extent_match.hpp"

namespace HPXKernelsSTD {

// overwriting gemv: y = Ax
template <class ExPolicy, 
    class ElementType_A, class SizeType_A,
    ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A, class Accessor_A,
    class ElementType_x, class SizeType_x,
    ::std::size_t ext_x, class Layout_x, class Accessor_x, 
    class ElementType_y, class SizeType_y,
    ::std::size_t ext_y, class Layout_y, class Accessor_y>
void matrix_vector_product(hpx_exec<ExPolicy>&& policy,
    std::experimental::mdspan<ElementType_A,
        std::experimental::extents<SizeType_A, numRows_A, numCols_A>,
        Layout_A, Accessor_A> A,
        std::experimental::mdspan<ElementType_x,
        std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y,
        std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y)
{

  // preconditions
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("HPXBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("HPXBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }

  // mandates
  Impl::static_extent_match(A.static_extent(1), x.static_extent(0));
  Impl::static_extent_match(A.static_extent(0), y.static_extent(0));

  std::cout << "called3\n";

  Impl::signal_hpx_impl_called("overwriting_matrix_vector_product");

  for (std::size_t i = 0; i < A.extent(0); ++i) {
    y(i) = ElementType_y{};
    for (std::size_t j = 0; j < A.extent(1); ++j) {
      y(i) += A(i,j) * x(j);
    }
  }
}

// updating gemv: z = y + Ax
template <class ExPolicy, 
    class ElementType_A, class SizeType_A,
    ::std::size_t numRows_A, ::std::size_t numCols_A, class Layout_A, class Accessor_A,
    class ElementType_x, class SizeType_x,
    ::std::size_t ext_x, class Layout_x, class Accessor_x, 
    class ElementType_y, class SizeType_y,
    ::std::size_t ext_y, class Layout_y, class Accessor_y,
    class ElementType_z, class SizeType_z,
    ::std::size_t ext_z, class Layout_z, class Accessor_z>
void matrix_vector_product(hpx_exec<ExPolicy>&& policy,
    std::experimental::mdspan<ElementType_A,
        std::experimental::extents<SizeType_A, numRows_A, numCols_A>,
        Layout_A, Accessor_A> A,
        std::experimental::mdspan<ElementType_x,
        std::experimental::extents<SizeType_x, ext_x>, Layout_x, Accessor_x> x,
        std::experimental::mdspan<ElementType_y,
        std::experimental::extents<SizeType_y, ext_y>, Layout_y, Accessor_y> y,
        std::experimental::mdspan<ElementType_z,
        std::experimental::extents<SizeType_z, ext_z>, Layout_z, Accessor_z> z)
{

  // preconditions
  if ( A.extent(1) != x.extent(0) ){
    throw std::runtime_error("HPXBlas: matrix_vector_product: A.extent(1) != x.extent(0) ");
  }
  if ( A.extent(0) != y.extent(0) ){
    throw std::runtime_error("HPXBlas: matrix_vector_product: A.extent(0) != y.extent(0) ");
  }
  if ( A.extent(0) != z.extent(0) ){
    throw std::runtime_error("HPXBlas: matrix_vector_product: A.extent(0) != z.extent(0) ");
  }

  std::cout << "called2\n";

  // mandates
  Impl::static_extent_match(A.static_extent(1), x.static_extent(0));
  Impl::static_extent_match(A.static_extent(0), y.static_extent(0));
  Impl::static_extent_match(y.static_extent(0), z.static_extent(0));

  Impl::signal_hpx_impl_called("updating_matrix_vector_product");

  for (std::size_t i = 0; i < A.extent(0); ++i) {
    z(i) = ElementType_z{};

    for (std::size_t j = 0; j < A.extent(1); ++j) {
      // z(i) += A(i,j) * x(j);
      z(i) += y(i) + A(i,j) * x(j);
    }
  }
}

} // namespace HPXKernelsSTD
#endif
