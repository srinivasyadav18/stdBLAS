#include "helpers.hpp"
#include <hpx/hpx.hpp>
#include <hpx/init.hpp>

#include <experimental/linalg>
#include <experimental/mdspan>
#include <type_traits>
#include <fstream>
#include <numeric>
#include <sstream>

void print_md(auto const& C)
{
    for (std::size_t i=0; i<C.extent(0); ++i){
        for (std::size_t j=0; j<C.extent(1); ++j){
            std::cout << C(i,j) << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

template <typename T, typename F>
struct gemm_perf_test
{
    using mdspan_t = mdspan<T, extents<::std::size_t, dynamic_extent, dynamic_extent>>;

    std::string file_name;
    std::ofstream fout;
    F f;
    int start, end, iters;

    gemm_perf_test(std::string test_name, F f_, int start_, int end_, int iters_) : f(f_), start(start_), end(end_), iters(iters_)
    {
        file_name = std::string("plots/") +
                                test_name + 
                                std::string("_") + 
                                std::string(typeid(T).name()) + 
                                std::string(".csv");
        std::cout << file_name << "\n";
        fout = std::ofstream(file_name.c_str());

        fout << "N,GFLOPs,time_seq,time_par,spd_up,flops_seq,flops_par\n";
        test_range();
    }

    ~gemm_perf_test()
    {
        fout.close();
    }

    template <typename ExPolicy>
    double test_single_iteration(ExPolicy policy, std::size_t n, int& i_iter)
    {  
        std::vector<T> x_data(n*n);
        std::vector<T> y_data(n*n);
        std::vector<T> z_data(n*n);

        // TODO : change the matrix inits to use `fill_random_mdspan`
        // std::iota(x_data.begin(), x_data.end(), 0);
        // std::iota(y_data.begin(), y_data.end(), 0);
        // std::iota(z_data.begin(), z_data.end(), 0);

        mdspan_t A(x_data.data(), n, n);
        mdspan_t B(y_data.data(), n, n);
        mdspan_t C(z_data.data(), n, n);

        const auto a = static_cast<T>(-42);
        const auto b = static_cast<T>(42);
        UnifDist<T> randObj(a, b);

        fill_random_mdspan(randObj, A);
        fill_random_mdspan(randObj, B);
        fill_random_mdspan(randObj, C);

        // print_md(A);
        // print_md(B);
        // print_md(C);

        auto t1 = std::chrono::high_resolution_clock::now();
            f(policy, A, B, C);
        auto t2 = std::chrono::high_resolution_clock::now();

        // print_md(A);
        // print_md(B);
        // print_md(C);
        if (i_iter == 0)
        {
            check_results(A, B, C, n);
            // std::cout << "passed : " << n << '\n';
        }

        std::chrono::duration<double> diff = t2 - t1;
        return diff.count();
    }

    void check_results(auto const& A, auto const& B, auto const&C, std::size_t n)
    {
        std::vector<T> z1_data(n*n);
        std::iota(z1_data.begin(), z1_data.end(), 0);
        mdspan_t C1(z1_data.data(), n, n);

        std::experimental::linalg::matrix_product(A, B, C1);
        
        for (std::size_t i=0; i<C.extent(0); ++i){
            for (std::size_t j=0; j<C.extent(1); ++j){
                if(abs(C(i,j)-C1(i,j)) > 1e-3){
                    std::cerr << "Failed at " << i << " " << j << "\n";
                    std::cerr << C(i,j) << " " << C1(i, j) << "\n";
                    exit(-1);
                }
            }
        }   
    }

    template <typename ExPolicy>
    double test_iters(ExPolicy policy, std::size_t n)
    {
        double avg = 0.0;
        for (int i = 0; i < iters; i++)
        {
            avg += test_single_iteration(policy, n, i);
        }
        avg /= double(iters);
        return avg;
    }

    void test_range()
    {
        for (int i = start; i <= end; i++)
        {
            int n = std::pow(2, i);
            double s = test_iters(hpx::execution::seq, n);
            double p = test_iters(hpx::execution::par, n);
            print(s, p, i);
        }
    }

    void print(const double ts, const double tp, const double i)
    {
        std::stringstream sout;
        double n = std::pow(2, i);
        double gflops = (n * n * n)/1e9;    // n^3 madds
        double dram_bw = (n*n*n + 3*n*n)/1e9; // n^3 + 3n^2
        sout << i << ',';
        sout << gflops << ',';
        sout << ts << ',';
        sout << tp << ',';
        sout << ts/tp << ',';
        sout << gflops/ts << ',';
        sout << gflops/tp << '\n';

        std::cout << sout.str();
        fout << sout.str();
        fout.flush();
    }
};

struct gemm_naive
{
    void operator()(auto policy, auto const& A, auto const&B, auto& C)
    {
        for (std::size_t i = 0; i < C.extent(0); ++i) {
            for (std::size_t j = 0; j < C.extent(1); ++j) {
                C(i,j) = {};
                for (std::size_t k = 0; k < A.extent(1); ++k) {
                    C(i,j) += A(i,k) * B(k,j);
                }
            }
        }
    }
};

struct gemm_hpx
{
    void operator()(auto policy, auto const& A, auto const&B, auto& C)
    {
        hpx::experimental::for_loop(
            hpx::execution::experimental::to_non_simd(policy),
            std::size_t(0), C.extent(0), [&](auto i){
                for (std::size_t j = 0; j < C.extent(1); ++j) {
                    C(i,j) = {};
                    for (std::size_t k = 0; k < A.extent(1); ++k) {
                        C(i,j) += A(i,k) * B(k,j);
                    }
                }
        });
    }
};

template <int bs=32>
struct gemm_L1
{
    void operator()(auto policy, auto const& A, auto const&B, auto& C)
    {
        const int BLOCK_SIZE=std::min(int(A.extent(0)), bs);

        for (std::size_t i = 0; i < C.extent(0); i += BLOCK_SIZE) {
            for (std::size_t j = 0; j < C.extent(1); j += BLOCK_SIZE) {
                for (std::size_t ii = i; ii < i + BLOCK_SIZE; ii++) 
                    for (std::size_t jj = j; jj < j + BLOCK_SIZE; jj++) 
                        C(ii, jj) = {};
                for (std::size_t k = 0; k < A.extent(1); k += BLOCK_SIZE) {
                    for (std::size_t ii = i; ii < i + BLOCK_SIZE; ii++) {
                        for (std::size_t jj = j; jj < j + BLOCK_SIZE; jj++) {
                            for (std::size_t kk = k; kk < k + BLOCK_SIZE; kk++) {
                                C(ii,jj) += A(ii,kk) * B(kk,jj);
                            }
                        }
                    }
                }
            }
        }
    }
};

template <int bs=32>
struct gemm_L1_hpx
{
    void operator()(auto policy, auto const& A, auto const&B, auto& C)
    {
        const int BLOCK_SIZE=std::min(int(A.extent(0)), bs);

        hpx::experimental::for_loop_strided(
            hpx::execution::experimental::to_non_simd(policy),
            std::size_t(0), C.extent(0), BLOCK_SIZE, [&](auto i){
                for (std::size_t j = 0; j < C.extent(1); j += BLOCK_SIZE) {
                    for (std::size_t ii = i; ii < i + BLOCK_SIZE; ii++) 
                        for (std::size_t jj = j; jj < j + BLOCK_SIZE; jj++) 
                            C(ii, jj) = {};
                    for (std::size_t k = 0; k < A.extent(1); k += BLOCK_SIZE) {
                        for (std::size_t ii = i; ii < i + BLOCK_SIZE; ii++) {
                            for (std::size_t jj = j; jj < j + BLOCK_SIZE; jj++) {
                                for (std::size_t kk = k; kk < k + BLOCK_SIZE; kk++) {
                                    C(ii,jj) += A(ii,kk) * B(kk,jj);
                                }
                            }
                        }
                    }
                }
        });
    }
};

int hpx_main()
{
    int start = 3, end = 12, iters=5; 
    gemm_perf_test<float, gemm_L1_hpx<16>>("gemm_L1_hpx<16>", gemm_L1_hpx<16>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1_hpx<24>>("gemm_L1_hpx<24>", gemm_L1_hpx<24>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1_hpx<32>>("gemm_L1_hpx<32>", gemm_L1_hpx<32>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1_hpx<40>>("gemm_L1_hpx<40>", gemm_L1_hpx<40>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1_hpx<64>>("gemm_L1_hpx<64>", gemm_L1_hpx<64>{}, start, end, iters);
    
    gemm_perf_test<float, gemm_L1<16>>("gemm_L1<16>", gemm_L1<16>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1<24>>("gemm_L1<24>", gemm_L1<24>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1<32>>("gemm_L1<32>", gemm_L1<32>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1<40>>("gemm_L1<40>", gemm_L1<40>{}, start, end, iters);
    gemm_perf_test<float, gemm_L1<64>>("gemm_L1<64>", gemm_L1<64>{}, start, end, iters);  

    gemm_perf_test<float, gemm_naive>("gemm_naive", gemm_naive{}, start, end, iters);
    
    gemm_perf_test<float, gemm_hpx>("gemm_hpx", gemm_hpx{}, start, end, iters);
  
    return hpx::finalize();
}

int main()
{
    return hpx::init();
}