// Copyright (c) 2022 Hartmut Kaiser
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/iterator_support/iterator_facade.hpp>

#include <experimental/mdspan>
#include <type_traits>
#include <utility>

#pragma once

template <typename MdSpan, typename Enable = void>
struct mditerator;

template <typename MdSpan>
constexpr mditerator<MdSpan> mditerator_begin(MdSpan& span) noexcept;

template <typename MdSpan>
constexpr mditerator<MdSpan> mditerator_end(MdSpan& span) noexcept;

// One dimensional specialization
template <typename ElementType, typename Extents, typename LayoutPolicy,
    typename AccessorPolicy>
struct mditerator<std::experimental::mdspan<ElementType, Extents, LayoutPolicy,
                      AccessorPolicy>,
    std::enable_if_t<Extents::rank() == 1>>
  : hpx::util::iterator_facade<mditerator<std::experimental::mdspan<ElementType,
                                   Extents, LayoutPolicy, AccessorPolicy>>,
        ElementType, std::random_access_iterator_tag, ElementType&,
        std::ptrdiff_t>
{
private:
    using mdspan_type = std::experimental::mdspan<ElementType, Extents,
        LayoutPolicy, AccessorPolicy>;
    using size_type = std::ptrdiff_t;
    using base_type = hpx::util::iterator_facade<mditerator<mdspan_type>,
        ElementType, std::random_access_iterator_tag, ElementType&, size_type>;

public:
    mditerator() = default;

private:
    friend class hpx::util::iterator_core_access;

    constexpr void increment() noexcept
    {
        ++idx_;
    }
    constexpr void decrement() noexcept
    {
        --idx_;
    }

    constexpr typename base_type::reference dereference() const noexcept
    {
        return (*span_)(idx_);
    }

    constexpr void advance(size_type n) noexcept
    {
        idx_ += n;
    }

    constexpr bool equal(mditerator rhs) const noexcept
    {
        return (span_ == rhs.span_) && (span_ == nullptr || idx_ == rhs.idx_);
    }

    constexpr size_type distance_to(mditerator rhs) const noexcept
    {
        return rhs.idx_ - idx_;
    }

private:
    constexpr mditerator(mdspan_type& span, size_type idx = 0) noexcept
      : span_(&span), idx_(idx)
    {}

    template <typename MdSpan>
    friend constexpr mditerator<MdSpan> mditerator_begin(MdSpan& span) noexcept;

    template <typename MdSpan>
    friend constexpr mditerator<MdSpan> mditerator_end(MdSpan& span) noexcept;

private:
    mdspan_type* span_ = nullptr;
    size_type idx_ = 0;
};

// higher dimensional mditerator

template <typename ElementType, typename Extents, typename LayoutPolicy,
    typename AccessorPolicy>
struct mditerator_2d_base
{
    using mdspan_type = std::experimental::mdspan<ElementType, Extents,
        LayoutPolicy, AccessorPolicy>;
    using size_type = std::ptrdiff_t;
    using element_type = decltype(std::experimental::submdspan(
        std::declval<mdspan_type const&>(), std::declval<size_type>(),
        std::experimental::full_extent));

    using type = hpx::util::iterator_facade<mditerator<mdspan_type>,
        element_type, std::random_access_iterator_tag, element_type, size_type>;
};

template <typename ElementType, typename Extents, typename LayoutPolicy,
    typename AccessorPolicy>
struct mditerator<std::experimental::mdspan<ElementType, Extents, LayoutPolicy,
                      AccessorPolicy>,
    std::enable_if_t<Extents::rank() == 2>>
  : mditerator_2d_base<ElementType, Extents, LayoutPolicy, AccessorPolicy>::type
{
private:
    using mdspan_type = std::experimental::mdspan<ElementType, Extents,
        LayoutPolicy, AccessorPolicy>;
    using size_type = std::ptrdiff_t;
    using base_type = typename mditerator_2d_base<ElementType, Extents,
        LayoutPolicy, AccessorPolicy>::type;

public:
    mditerator() = default;

private:
    friend class hpx::util::iterator_core_access;

    constexpr void increment() noexcept
    {
        ++idx_;
    }
    constexpr void decrement() noexcept
    {
        --idx_;
    }

    constexpr typename base_type::reference dereference() const noexcept
    {
        return std::experimental::submdspan(
            *span_, idx_, std::experimental::full_extent);
    }

    constexpr void advance(size_type n) noexcept
    {
        idx_ += n;
    }

    constexpr bool equal(mditerator rhs) const noexcept
    {
        return (span_ == rhs.span_) && (span_ == nullptr || idx_ == rhs.idx_);
    }

    constexpr size_type distance_to(mditerator rhs) const noexcept
    {
        return rhs.idx_ - idx_;
    }

private:
    constexpr mditerator(mdspan_type& span, size_type idx = 0) noexcept
      : span_(&span), idx_(idx)
    {}

    template <typename MdSpan>
    friend constexpr mditerator<MdSpan> mditerator_begin(MdSpan& span) noexcept;

    template <typename MdSpan>
    friend constexpr mditerator<MdSpan> mditerator_end(MdSpan& span) noexcept;

private:
    mdspan_type* span_ = nullptr;
    size_type idx_ = 0;
};

// construct mditerator instances from a given mdspan
template <typename MdSpan>
constexpr mditerator<MdSpan> mditerator_begin(MdSpan& span) noexcept
{
    return mditerator<MdSpan>(span);
}

template <typename MdSpan>
constexpr mditerator<MdSpan> mditerator_end(MdSpan& span) noexcept
{
    return mditerator<MdSpan>(span, span.extent(0));
}
