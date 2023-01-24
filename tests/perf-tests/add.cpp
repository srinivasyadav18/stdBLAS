
#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>

template <typename ExPolicy, typename T, typename Cond>
auto test(ExPolicy policy, T n, Cond c)
{  
    _blas1_signed_fixture<T> full_data(n);

    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (c)
    {
        std::experimental::linalg::add(full_data.x, full_data.y, full_data.z);
    }
    else
    {
        std::experimental::linalg::add(policy, full_data.x, full_data.y, full_data.z);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

template <typename T>
void test2(T n)
{
    for (int i = 5; i <= 26; i++)
    {
        T curr_n = T(std::pow(2, i));
        std::cout << curr_n << ',';
        std::cout << test(std::true_type{}, T(n), std::true_type{}) << ',';
        std::cout << test(hpx::execution::seq, T(n), std::false_type{}) << ',';
        std::cout << test(hpx::execution::simd, T(n), std::false_type{}) << ',';
        std::cout << test(hpx::execution::par, T(n), std::false_type{}) << ',';
        std::cout << test(hpx::execution::par_simd, T(n), std::false_type{}) << '\n';
        // std::cout << endl;
    }
}

int hpx_main()
{
    test2(static_cast<float>(1 << 20));
    return hpx::finalize();
}

int main()
{
    return hpx::init();
}