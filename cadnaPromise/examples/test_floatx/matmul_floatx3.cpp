#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <cmath>

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

template <typename T>
double run_test_average(int N) {
    double total_time = 0.0;
    const int repeats = 1;
    for (int r = 0; r < repeats; ++r) {
        total_time += matmul<T>(N);
    }
    return total_time / repeats;
}

int main() {
    vector<int> sizes = {500, 1000};

    ofstream outfile("results.csv");
    outfile << "MatrixSize,Type,ExpBits,SigBits,AvgTime\n";

    for (int N : sizes) {
        cout << "\n=== Matrix size: " << N << " x " << N << " ===\n";

        // Test varying exponent bits
        for (int e = 4; e <= 20; e += 2) {
            int s = 3; // fixed significand
            string name = "(" + to_string(e) + ", " + to_string(s) + ")";
            double avg_time = 0.0;

            if constexpr (true) {
                if (e == 4) avg_time = run_test_average<floatx<4,3,double>>(N);
                else if (e == 6) avg_time = run_test_average<floatx<6,3,double>>(N);
                else if (e == 8) avg_time = run_test_average<floatx<8,3,double>>(N);
                else if (e == 10) avg_time = run_test_average<floatx<10,3,double>>(N);
                else if (e == 12) avg_time = run_test_average<floatx<12,3,double>>(N);
                else if (e == 14) avg_time = run_test_average<floatx<14,3,double>>(N);
                else if (e == 16) avg_time = run_test_average<floatx<16,3,double>>(N);
                else if (e == 18) avg_time = run_test_average<floatx<18,3,double>>(N);
                else if (e == 20) avg_time = run_test_average<floatx<20,3,double>>(N);
            }

            cout << "Precision " << name << ", AvgTime = " << avg_time << " sec" << endl;
            outfile << N << ",exp," << e << "," << s << "," << avg_time << "\n";
        }

        // Test varying significand bits
        for (int s = 4; s <= 20; s += 2) {
            int e = 3; // fixed exponent
            string name = "(" + to_string(e) + ", " + to_string(s) + ")";
            double avg_time = 0.0;

            if constexpr (true) {
                if (s == 4) avg_time = run_test_average<floatx<3,4,double>>(N);
                else if (s == 6) avg_time = run_test_average<floatx<3,6,double>>(N);
                else if (s == 8) avg_time = run_test_average<floatx<3,8,double>>(N);
                else if (s == 10) avg_time = run_test_average<floatx<3,10,double>>(N);
                else if (s == 12) avg_time = run_test_average<floatx<3,12,double>>(N);
                else if (s == 14) avg_time = run_test_average<floatx<3,14,double>>(N);
                else if (s == 16) avg_time = run_test_average<floatx<3,16,double>>(N);
                else if (s == 18) avg_time = run_test_average<floatx<3,18,double>>(N);
                else if (s == 20) avg_time = run_test_average<floatx<3,20,double>>(N);
            }

            cout << "Precision " << name << ", AvgTime = " << avg_time << " sec" << endl;
            outfile << N << ",sig," << e << "," << s << "," << avg_time << "\n";
        }
    }

    outfile.close();
    return 0;
}
