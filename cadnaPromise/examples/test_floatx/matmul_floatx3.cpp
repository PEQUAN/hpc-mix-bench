#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>

#include "floatx.hpp"

using namespace std;
using namespace std::chrono;
using namespace flx;

template <typename T>
void init_matrix(vector<T>& M, int N) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = T((i % 100) * 0.01);
    }
}

template <typename T>
double matmul(int N) {
    vector<T> A(N * N), B(N * N), C(N * N, T(0)); // ✅ 初始化 C

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

template <typename T>
double run_test_average(int N) {
    const int repeats = 1;
    double total_time = 0.0;

    // warm-up
    matmul<T>(N);

    for (int r = 0; r < repeats; ++r) {
        total_time += matmul<T>(N);
    }
    return total_time / repeats;
}

int main() {
    vector<int> sizes = {1000};

    ofstream outfile("results.csv");
    outfile << "MatrixSize,Type,ExpBits,SigBits,AvgTime\n";

    for (int N : sizes) {
        cout << "\n=== Matrix size: " << N << " x " << N << " ===\n";

        /* ===============================
           Exponent sweep
           fixed significand = {3, 7}
        =============================== */
        vector<int> fixed_sigs = {3, 7};

        for (int s : fixed_sigs) {
            for (int e = 3; e <= 11; e += 2) {
                double avg_time = 0.0;

                if (s == 3) {
                    if (e == 3) avg_time = run_test_average<floatx<3,3,double>>(N);
                    else if (e == 5) avg_time = run_test_average<floatx<5,3,double>>(N);
                    else if (e == 7) avg_time = run_test_average<floatx<7,3,double>>(N);
                    else if (e == 9) avg_time = run_test_average<floatx<9,3,double>>(N);
                    else if (e == 11) avg_time = run_test_average<floatx<11,3,double>>(N);
                }
                else if (s == 7) {
                    if (e == 3) avg_time = run_test_average<floatx<3,7,double>>(N);
                    else if (e == 5) avg_time = run_test_average<floatx<5,7,double>>(N);
                    else if (e == 7) avg_time = run_test_average<floatx<7,7,double>>(N);
                    else if (e == 9) avg_time = run_test_average<floatx<9,7,double>>(N);
                    else if (e == 11) avg_time = run_test_average<floatx<11,7,double>>(N);
                }


                cout << "Exp sweep: (e=" << e << ", s=" << s
                     << "), AvgTime = " << avg_time << " sec\n";

                outfile << N << ",exp," << e << "," << s << "," << avg_time << "\n";
            }
        }


        /* ===============================
           Significand sweep
           fixed exponent = {3, 7}
        =============================== */
        vector<int> fixed_exps = {3, 7};

        for (int e : fixed_exps) {
            for (int s = 3; s <= 53; s += 2) {
                double avg_time = 0.0;

                if (e == 3) {
                    if (s == 3) avg_time = run_test_average<floatx<3,3,double>>(N);
                    else if (s == 5) avg_time = run_test_average<floatx<3,5,double>>(N);
                    else if (s == 7) avg_time = run_test_average<floatx<3,7,double>>(N);
                    else if (s == 9) avg_time = run_test_average<floatx<3,9,double>>(N);
                    else if (s == 11) avg_time = run_test_average<floatx<3,11,double>>(N);
                    else if (s == 13) avg_time = run_test_average<floatx<3,13,double>>(N);
                    else if (s == 15) avg_time = run_test_average<floatx<3,15,double>>(N);
                    else if (s == 17) avg_time = run_test_average<floatx<3,17,double>>(N);
                    else if (s == 19) avg_time = run_test_average<floatx<3,19,double>>(N);
                    else if (s == 21) avg_time = run_test_average<floatx<3,21,double>>(N);
                    else if (s == 23) avg_time = run_test_average<floatx<3,23,double>>(N);
                    else if (s == 25) avg_time = run_test_average<floatx<3,25,double>>(N);
                    else if (s == 27) avg_time = run_test_average<floatx<3,27,double>>(N);
                    else if (s == 29) avg_time = run_test_average<floatx<3,29,double>>(N);
                    else if (s == 31) avg_time = run_test_average<floatx<3,31,double>>(N);
                    else if (s == 33) avg_time = run_test_average<floatx<3,33,double>>(N);
                    else if (s == 35) avg_time = run_test_average<floatx<3,35,double>>(N);
                    else if (s == 37) avg_time = run_test_average<floatx<3,37,double>>(N);
                    else if (s == 39) avg_time = run_test_average<floatx<3,39,double>>(N);
                    else if (s == 41) avg_time = run_test_average<floatx<3,41,double>>(N);
                    else if (s == 43) avg_time = run_test_average<floatx<3,43,double>>(N);
                    else if (s == 45) avg_time = run_test_average<floatx<3,45,double>>(N);
                    else if (s == 47) avg_time = run_test_average<floatx<3,47,double>>(N);
                    else if (s == 49) avg_time = run_test_average<floatx<3,49,double>>(N);
                    else if (s == 51) avg_time = run_test_average<floatx<3,51,double>>(N);
                    else if (s == 53) avg_time = run_test_average<floatx<3,53,double>>(N);
                }
                else if (e == 7) {
                    if (s == 3) avg_time = run_test_average<floatx<7,3,double>>(N);
                    else if (s == 5) avg_time = run_test_average<floatx<7,5,double>>(N);
                    else if (s == 7) avg_time = run_test_average<floatx<7,7,double>>(N);
                    else if (s == 9) avg_time = run_test_average<floatx<7,9,double>>(N);
                    else if (s == 11) avg_time = run_test_average<floatx<7,11,double>>(N);
                    else if (s == 13) avg_time = run_test_average<floatx<7,13,double>>(N);
                    else if (s == 15) avg_time = run_test_average<floatx<7,15,double>>(N);
                    else if (s == 17) avg_time = run_test_average<floatx<7,17,double>>(N);
                    else if (s == 19) avg_time = run_test_average<floatx<7,19,double>>(N);
                    else if (s == 21) avg_time = run_test_average<floatx<7,21,double>>(N);
                    else if (s == 23) avg_time = run_test_average<floatx<7,23,double>>(N);
                    else if (s == 25) avg_time = run_test_average<floatx<7,25,double>>(N);
                    else if (s == 27) avg_time = run_test_average<floatx<7,27,double>>(N);
                    else if (s == 29) avg_time = run_test_average<floatx<7,29,double>>(N);
                    else if (s == 31) avg_time = run_test_average<floatx<7,31,double>>(N);
                    else if (s == 33) avg_time = run_test_average<floatx<7,33,double>>(N);
                    else if (s == 35) avg_time = run_test_average<floatx<7,35,double>>(N);
                    else if (s == 37) avg_time = run_test_average<floatx<7,37,double>>(N);
                    else if (s == 39) avg_time = run_test_average<floatx<7,39,double>>(N);
                    else if (s == 41) avg_time = run_test_average<floatx<7,41,double>>(N);
                    else if (s == 43) avg_time = run_test_average<floatx<7,43,double>>(N);
                    else if (s == 45) avg_time = run_test_average<floatx<7,45,double>>(N);
                    else if (s == 47) avg_time = run_test_average<floatx<7,47,double>>(N);
                    else if (s == 49) avg_time = run_test_average<floatx<7,49,double>>(N);
                    else if (s == 51) avg_time = run_test_average<floatx<7,51,double>>(N);
                    else if (s == 53) avg_time = run_test_average<floatx<7,53,double>>(N);
                }

                cout << "Sig sweep: (e=" << e << ", s=" << s
                     << "), AvgTime = " << avg_time << " sec\n";

                outfile << N << ",sig," << e << "," << s << "," << avg_time << "\n";
            }
        }
    }

    outfile.close();
    return 0;
}
