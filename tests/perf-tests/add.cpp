
#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <fstream>

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
    // std::cout << n << '\n';
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

    // no_optimize(full_data.x);
    // no_optimize(full_data.y);
    // no_optimize(full_data.z);

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
                            std::string("_add.csv");

    std::ofstream fout(file_name.c_str());

    for (int i = 5; i <= n; i++)
    {
        T curr_n = T(std::pow(2, i));
        fout << i << ',';
        fout << test(std::true_type{}, T(curr_n), std::true_type{}) << ',';
        fout << test(hpx::execution::seq, T(curr_n), std::false_type{}) << ',';
        fout << test(hpx::execution::simd, T(curr_n), std::false_type{}) << ',';
        fout << test(hpx::execution::par, T(curr_n), std::false_type{}) << ',';
        fout << test(hpx::execution::par_simd, T(curr_n), std::false_type{}) << '\n';
        fout.flush();
        // std::cout << endl;
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