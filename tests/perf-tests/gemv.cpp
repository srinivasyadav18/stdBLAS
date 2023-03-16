
#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <fstream>
#include <numeric>

std::string csv_name = "gemv.csv";

template <typename ExPolicy, typename T>
auto test(ExPolicy&& policy, T n)
{  
    using value_type = T;
    using matrix_t = std::experimental::mdspan<T, extents<::std::size_t, dynamic_extent, dynamic_extent>>;
    using vector_t = std::experimental::mdspan<T, extents<::std::size_t, dynamic_extent>>;

    std::vector<value_type> x_data(n*n);
    std::vector<value_type> y_data(n);
    std::vector<value_type> z_data(n);

    std::iota(x_data.begin(), x_data.end(), 0);
    std::iota(y_data.begin(), y_data.end(), 0);
    std::iota(z_data.begin(), z_data.end(), 0);

    matrix_t A(x_data.data(), n, n);
    vector_t x(y_data.data(), n);
    vector_t y(y_data.data(), n);

    auto t1 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < A.extent(0); ++i) {
        y(i) = {};
        for (std::size_t j = 0; j < A.extent(1); ++j) {
            y(i) += A(i,j) * x(j);
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = t2 - t1;
    return diff.count();
}

template <typename ExPolicy, typename T>
auto test2(ExPolicy&& policy, T n, int iters = 3)
{
    double d = 0;
    for (int i = 0; i < iters; i++)
    {
        d += test(std::forward(policy), n);
    }
    d /= double(iters);
    return d;
}

template <typename T>
void test3(T n)
{
    std::string file_name = std::string("plots/") +
                            std::string(typeid(T).name()) + 
                            csv_name;

    std::ofstream fout(file_name.c_str());

    auto loop_iter = [&](int i, int bs)
    {
        T curr_n = T(std::pow(2, i));
        double gflops = (curr_n * curr_n)/1e9;
        double ts = test(hpx::execution::seq, T(curr_n));
        double tp = test(hpx::execution::par, T(curr_n));
        fout << i << ',';
        fout << gflops << ',';
        fout << bs << ",";
        fout << ts << ',';
        fout << tp << ',';
        fout << gflops/ts << ',';
        fout << gflops/tp << '\n';
        fout.flush();
    };

    fout << "N,glops,bs,ts,tp,gf_s,gf_p\n";
    for (int i = 10; i <= 20; i++)
    {
        loop_iter(i, i);
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