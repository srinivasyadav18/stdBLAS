
#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <fstream>
#include <numeric>

template <typename ExPolicy, typename T>
auto test(ExPolicy&& policy, T n)
{  
    using value_type = T;
    using mdspan_t = mdspan<T, extents<::std::size_t, dynamic_extent, dynamic_extent>>;

    std::vector<value_type> x_data(n*n);
    std::vector<value_type> y_data(n*n);
    std::vector<value_type> z_data(n*n);

    std::iota(x_data.begin(), x_data.end(), 0);
    std::iota(y_data.begin(), y_data.end(), 0);
    std::iota(z_data.begin(), z_data.end(), 0);

    mdspan_t A(x_data.data(), n, n);
    mdspan_t B(y_data.data(), n, n);
    mdspan_t C(y_data.data(), n, n);

    auto t1 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < C.extent(0); ++i) {
            for (std::size_t j = 0; j < C.extent(1); ++j) {
                C(i,j) = value_type{};
                for (std::size_t k = 0; k < A.extent(1); ++k) {
                    C(i,j) += A(i,k) * B(k,j);
                }
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
void test3(T n_)
{
    std::string file_name = std::string("plots/") +
                            std::string(typeid(T).name()) + 
                            std::string("_gemm.csv");

    std::ofstream fout(file_name.c_str());

    fout << "N,GFLOPs,GBW,time_seq,time_par,gf_s,gf_p,bw_s,bw_p\n";

    for (int i = 10; i <= 15; i++)
    {
        T n = T(std::pow(2, i));
        double gflops = (n * n * n)/1e9;    // n^3 madds
        double dram_bw = (n*n*n + 3*n*n)/1e9; // n^3 + 3n^2
        double ts = test(hpx::execution::seq, T(n));
        double tp = test(hpx::execution::par, T(n));
        fout << i << ',';
        fout << gflops << ',';
        fout << dram_bw << ',';
        fout << ts << ',';
        fout << tp << ',';
        fout << gflops/ts << ',';
        fout << gflops/tp << ',';
        fout << dram_bw/ts << ',';
        fout << dram_bw/tp << '\n';

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