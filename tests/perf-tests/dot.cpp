
#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <fstream>
#include <numeric>

void no_optimize(const auto& md)
{
    if constexpr (md.rank() == 1)
        std::cout << md(rand()%4) << '\n';
    else
        std::cout << md(rand()%3, rand()%2) << '\n';
}

template <typename ExPolicy, typename T, typename Cond>
auto test(ExPolicy&& policy, T n, Cond c)
{  
    using value_type = T;
    using mdspan_t = mdspan<T, extents<::std::size_t, dynamic_extent>>;
    std::vector<value_type> x_data(n);
    std::vector<value_type> y_data(n);

    std::iota(x_data.begin(), x_data.end(), 0);
    std::iota(y_data.begin(), y_data.end(), 0);

    mdspan_t x(x_data.data(), n);
    mdspan_t y(y_data.data(), n);

    double opt = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (c)
    {
        opt += std::experimental::linalg::dot(x, y);
    }
    else
    {
        opt += std::experimental::linalg::dot(policy, x, y);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << opt << "\n";

    std::chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

template <typename ExPolicy, typename T, typename Cond>
auto test2(ExPolicy&& policy, T n, Cond c, int iters = 10)
{
    double d = 0;
    for (int i = 0; i < iters; i++)
    {
        d += test(std::forward(policy), n, c);
    }
    d /= double(iters);
    return d;
}

template <typename T>
void test3(T n)
{
    std::string file_name = std::string("plots/") +
                            std::string(typeid(T).name()) + 
                            std::string("_dot.csv");

    std::ofstream fout(file_name.c_str());

    for (int i = 5; i <= 26; i++)
    {
        T curr_n = T(std::pow(2, i));
        // T curr_n = T(i);
        fout << i << ',';
        fout << test(std::true_type{}, T(curr_n), std::true_type{}) << ',';
        fout << test(hpx::execution::seq, T(curr_n), std::false_type{}) << ',';
        fout << test(hpx::execution::simd, T(curr_n), std::false_type{}) << ',';
        fout << test(hpx::execution::par, T(curr_n), std::false_type{}) << ',';
        fout << test(hpx::execution::par_simd, T(curr_n), std::false_type{}) << '\n';
        fout.flush();
    }
    fout.close();
}

int hpx_main()
{
    std::filesystem::create_directory("plots");
    test3(static_cast<float>(28));
    return hpx::finalize();
}

int main()
{
    return hpx::init();
}