/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_HPXKERNELS_DOT_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_HPXKERNELS_DOT_HPP_

#include <hpx/algorithm.hpp>
#include <hpx/parallel/algorithms/transform_reduce.hpp>
#include <hpx/iterator_support/zip_iterator.hpp>
// #include <hpx/tuple.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include <cstddef>
#include <type_traits>

#include "exec_policy_wrapper_hpx.hpp"
#include "signal_hpx_impl_called.hpp"

#include "mditerator.hpp"

namespace HPXKernelsSTD {

template<class ExPolicy,
         class ElementType1,
	 class SizeType1,	 
         ::std::size_t ext1,
         class Layout1,
         class Accessor1,
         class ElementType2,
	 class SizeType2,
         ::std::size_t ext2,
         class Layout2,
         class Accessor2,
         class Scalar>
Scalar dot(
  hpx_exec<ExPolicy>&& policy,
  std::experimental::mdspan<ElementType1, std::experimental::extents<SizeType1, ext1>, Layout1, Accessor1> x,
  std::experimental::mdspan<ElementType2, std::experimental::extents<SizeType2, ext2>, Layout2, Accessor2> y,
  Scalar init)
{
    Impl::signal_hpx_impl_called("dot");
    return hpx::transform_reduce(
            policy.policy_,
            hpx::util::zip_iterator(mditerator_begin(x), mditerator_begin(y)),
            hpx::util::zip_iterator(mditerator_end(x), mditerator_end(y)), 
            init,
            std::plus<>{},
            [](auto r) { return hpx::get<0>(r) * hpx::get<1>(r); });
}

}    // namespace HPXKernelsSTD

#endif    //LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_HPXKERNELS_DOT_HPP_
