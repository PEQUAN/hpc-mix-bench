#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

#include "floatx.hpp"

using namespace std;
using namespace std::chrono;
using namespace flx;


template <typename T>
void init_matrix(vector<T>& M, int N) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = T( (i % 100) * 0.01 );
    }
}


template <typename T>
double matmul(int N) {
    vector<T> A(N * N), B(N * N), C(N * N);

    init_matrix(A, N);
    init_matrix(B, N);

    auto t0 = high_resolution_clock::now();

    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            T aik = A[i * N + k];
            for (int j = 0; j < N; ++j) {
                C[i * N + j] += aik * B[k * N + j];
            }
        }
    }

    auto t1 = high_resolution_clock::now();

    // 防止优化
    volatile double checksum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        checksum += double(C[i]);
    }

    duration<double> dt = t1 - t0;
    return dt.count();
}


template <typename T>
void run_test(const string& name, int N) {
    cout << "Precision " << name
         << ", N = " << N << " ... " << flush;
    double t = matmul<T>(N);

    cout << t << " sec" << endl;
}


int main() {
    vector<int> sizes = {1000, 5000};

    for (int N : sizes) {
        cout << "\n=== Matrix size: " << N << " x " << N << " ===\n";
        run_test<floatx<2, 2, double>>("(4, 2)", N);
        run_test<floatx<4, 2, double>>("(5, 2)", N);
        run_test<floatx<6, 2, double>>("(6, 2)", N);
        run_test<floatx<8, 2, double>>("(8, 2)", N);
        run_test<floatx<10, 2, double>>("(10, 2)", N);
        run_test<floatx<12, 2, double>>("(12, 2)", N);
        run_test<floatx<14, 2, double>>("(14, 2)", N);
        run_test<floatx<16, 2, double>>("(16, 2)", N);
        run_test<floatx<18, 2, double>>("(18, 2)", N);
        run_test<floatx<20, 2, double>>("(20, 2)", N);
        
        run_test<floatx<2, 2, double>>("(2, 2)", N);
        run_test<floatx<2, 4, double>>("(2, 4)", N);
        run_test<floatx<2, 6, double>>("(2, 6)", N);
        run_test<floatx<2, 8, double>>("(2, 8)", N);
        run_test<floatx<2, 10, double>>("(2, 10)", N);
        run_test<floatx<2, 12, double>>("(2, 12)", N);
        run_test<floatx<2, 14, double>>("(2, 14)", N);
        run_test<floatx<2, 16, double>>("(2, 16)", N);
        run_test<floatx<2, 18, double>>("(2, 18)", N);
        run_test<floatx<2, 20, double>>("(2, 20)", N);
    }

    return 0;
}
