//  Copyright (c) 2022 Hartmut Kaiser

#ifndef LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_SCALE_HPP_
#define LINALG_TPLIMPLEMENTATIONS_INCLUDE_EXPERIMENTAL___P1673_BITS_HPXKERNELS_SCALE_HPP_

#include <hpx/algorithm.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>

#include <cstddef>
#include <type_traits>

#include "exec_policy_wrapper_hpx.hpp"
#include "signal_hpx_impl_called.hpp"

#include "mditerator.hpp"

namespace HPXKernelsSTD {

namespace {

template <class ExPolicy, class Scalar, class ElementType, class SizeType,
    ::std::size_t ext0, class Layout, class Accessor>
void linalg_scale_rank_1(ExPolicy&& policy, const Scalar alpha,
    std::experimental::mdspan<ElementType,
        std::experimental::extents<SizeType, ext0>, Layout, Accessor>
        x)
{
#if defined(HPX_HAVE_DATAPAR)
    if constexpr (supports_vectorization_v<ExPolicy>)
    {
        using mdspan_t = std::experimental::mdspan<ElementType,
            std::experimental::extents<SizeType, ext0>, Layout, Accessor>;

        if constexpr (allow_vectorization_v<mdspan_t>)
        {
            // vectorize only if the array is contiguous and not strided
            if (x.is_contiguous() && x.stride(0) == 1)
            {
                hpx::for_each(policy, mditerator_begin(x), mditerator_end(x), 
                    [&](auto &i) { i *= alpha; });
            }
            else
            {
            hpx::for_each(hpx::execution::experimental::to_non_simd(policy),
                mditerator_begin(x), mditerator_end(x), 
                [&](auto &i) { i *= alpha; });
            }
        }
        else
        {
            hpx::for_each(hpx::execution::experimental::to_non_simd(policy),
                mditerator_begin(x), mditerator_end(x), 
                [&](auto &i) { i *= alpha; });
        }
    }
    else
#endif
    {
        hpx::for_each(policy, mditerator_begin(x), mditerator_end(x), 
                [&](auto &i) { i *= alpha; });
    }
}

template <class ExPolicy, class Scalar, class ElementType, class SizeType,
    ::std::size_t numRows, ::std::size_t numCols, class Layout, class Accessor>
void linalg_scale_rank_2(ExPolicy&& policy, const Scalar alpha,
    std::experimental::mdspan<ElementType,
        std::experimental::extents<SizeType, numRows, numCols>, Layout,
        Accessor>
        A)
{
#if defined(HPX_HAVE_DATAPAR)
    if constexpr (supports_vectorization_v<ExPolicy>)
    {
        hpx::for_each(hpx::execution::experimental::to_non_simd(policy),
            mditerator_begin(A), mditerator_end(A), [&](auto j)
        {
            linalg_scale_rank_1(hpx::execution::experimental::to_non_par(policy), alpha, j);
        });
    }
    else
#endif
    {
        hpx::for_each(policy, mditerator_begin(A), mditerator_end(A), [&](auto j)
        {
            hpx::for_each(mditerator_begin(j), mditerator_end(j), [&](auto &i){
                i *= alpha;
            });
        });
    }
}

}    // namespace

MDSPAN_TEMPLATE_REQUIRES(class ExPolicy, class Scalar, class ElementType,
    class SizeType, ::std::size_t... ext, class Layout, class Accessor,
    /* requires */ (sizeof...(ext) <= 2))
void scale(hpx_exec<ExPolicy>&& policy, const Scalar alpha,
    std::experimental::mdspan<ElementType,
        std::experimental::extents<SizeType, ext...>, Layout, Accessor>
        x)
{
    Impl::signal_hpx_impl_called("scale");
    if constexpr (x.rank() == 1)
    {
        linalg_scale_rank_1(policy.policy_, alpha, x);
    }
    else if constexpr (x.rank() == 2)
    {
        linalg_scale_rank_2(policy.policy_, alpha, x);
    }
}

}    // namespace HPXKernelsSTD

#endif
