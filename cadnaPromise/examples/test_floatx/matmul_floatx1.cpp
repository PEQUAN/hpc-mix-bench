#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>

#include "floatx.hpp"

using namespace std;
using namespace std::chrono;
using namespace flx;

using fx_c = floatx<4, 3, double>;
using fx_w = floatx<5, 2, double>;

using fx_b = floatx<8, 7, double>;
using fx_p = floatx<5, 10, double>;

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

// ---------- 跑一组 ----------
template <typename T>
void run_test(const string& name, int N) {
    cout << "Precision " << name
         << ", N = " << N << " ... " << flush;

    double t = matmul<T>(N);

    cout << t << " sec" << endl;
}

int main() {
    vector<int> sizes = {500, 1000, 2000};

    for (int N : sizes) {
        cout << "\n=== Matrix size: " << N << " x " << N << " ===\n";

        run_test<fx_c>("c (4,3)", N);
        run_test<fx_w>("w (5,2)", N);
        run_test<fx_b>("b (8,7)", N);
        run_test<fx_p>("p (5,10)", N);
    }

    return 0;
}
