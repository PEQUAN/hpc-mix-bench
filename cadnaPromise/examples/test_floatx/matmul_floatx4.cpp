#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

#include "floatx.hpp"

using namespace std;
using namespace std::chrono;
using namespace flx;

// Precision aliases
using fx_c = floatx<4, 3, double>;
using fx_w = floatx<5, 2, double>;
using fx_b = floatx<8, 7, double>;
using fx_p = floatx<5, 10, double>;

template <typename T>
void init_matrix(vector<T>& M, int N) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = T((i % 100) * 0.01);
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

    // Prevent optimization
    volatile double checksum = 0.0;
    for (int i = 0; i < N * N; ++i) {
        checksum += double(C[i]);
    }

    duration<double> dt = t1 - t0;
    return dt.count();
}

// ---------- run 3 times and average ----------
template <typename T>
double run_test_avg(const string& name, int N) {
    const int repeats = 3;
    double total = 0.0;

    cout << "Precision " << name << ", N = " << N << " ... " << flush;

    for (int i = 0; i < repeats; ++i) {
        total += matmul<T>(N);
    }

    double avg = total / repeats;
    cout << avg << " sec (avg of 3)" << endl;
    return avg;
}

int main() {
    vector<int> sizes = {500, 1000};

    ofstream outfile("results_bar.csv");
    outfile << "MatrixSize,Precision,AvgTime\n";

    for (int N : sizes) {
        cout << "\n=== Matrix size: " << N << " x " << N << " ===\n";

        double t_c = run_test_avg<fx_c>("c (4,3)", N);
        double t_w = run_test_avg<fx_w>("w (5,2)", N);
        double t_b = run_test_avg<fx_b>("b (8,7)", N);
        double t_p = run_test_avg<fx_p>("p (5,10)", N);
        double t_s = run_test_avg<float>("s (8,23)", N);
        double t_d = run_test_avg<double>("d (11,52)", N);

        outfile << N << ",c," << t_c << "\n";
        outfile << N << ",w," << t_w << "\n";
        outfile << N << ",b," << t_b << "\n";
        outfile << N << ",p," << t_p << "\n";
        outfile << N << ",s," << t_s << "\n";
        outfile << N << ",d," << t_d << "\n";
    }

    outfile.close();
    return 0;
}
