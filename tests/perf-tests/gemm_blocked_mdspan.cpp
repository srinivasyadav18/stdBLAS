
#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <fstream>
#include <numeric>

void initalize_block(auto& C)
{
  for (int i = 0; i < C.extent(0); i++)
  {
    for (int j = 0; j < C.extent(1); j++)
    {
      C(i, j) = {};
    }
  }
}

void multiply_blocks(auto& C, auto const A, auto const B)
{
  for (std::size_t i = 0; i < C.extent(0); ++i) {
    for (std::size_t j = 0; j < C.extent(1); ++j) {
      for (std::size_t k = 0; k < A.extent(1); ++k) {
        C(i,j) += A(i,k) * B(k,j);
      }
    }
  }
}

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

    const int block_size=64;
    auto t1 = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < C.extent(0); i += block_size)
        {
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
                            std::string("_gemm_blocked_mdspan.csv");

    std::ofstream fout(file_name.c_str());

    for (int i = 10; i <= 15; i++)
    {
        T curr_n = T(std::pow(2, i));
        double gflops = (curr_n * curr_n * curr_n)/1e9;
        double ts = test(hpx::execution::seq, T(curr_n));
        double tp = test(hpx::execution::par, T(curr_n));
        fout << i << ',';
        fout << gflops << ',';
        fout << ts << ',';
        fout << tp << ',';
        fout << gflops/ts << ',';
        fout << gflops/tp << '\n';

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