#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../cuda_common/h100_mixed_precision.cuh"

#ifndef DLU_RUN_LABEL
#define DLU_RUN_LABEL "H100 CUDA dense LU mixed-precision run"
#endif

#ifndef DLU_PIVOT_E
#define DLU_PIVOT_E 5
#define DLU_PIVOT_T 2
#endif

#ifndef DLU_AKK_E
#define DLU_AKK_E 8
#define DLU_AKK_T 23
#endif

namespace dense_lu_h100 {

using matrix_t = mp_cuda::storage<DLU_AKK_E, DLU_AKK_T>;

#define M(A, i, j, n) (A)[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)]

static std::vector<double> alloc_matrix(int n) {
    return std::vector<double>(static_cast<size_t>(n) * static_cast<size_t>(n), 0.0);
}

static void mat_vec(const std::vector<double>& A,
                    const std::vector<double>& x,
                    std::vector<double>& b,
                    int n) {
    std::fill(b.begin(), b.end(), 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += M(A, i, j, n) * x[j];
        }
        b[i] = sum;
    }
}

static double norm2(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) {
        s += x * x;
    }
    return std::sqrt(s);
}

static void random_orthogonal(std::vector<double>& Q, int n, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    for (double& x : Q) {
        x = nd(rng);
    }

    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < n; ++i) dot += M(Q, i, k, n) * M(Q, i, j, n);
            for (int i = 0; i < n; ++i) M(Q, i, j, n) -= dot * M(Q, i, k, n);
        }
        double len = 0.0;
        for (int i = 0; i < n; ++i) len += M(Q, i, j, n) * M(Q, i, j, n);
        len = std::sqrt(len);
        for (int i = 0; i < n; ++i) M(Q, i, j, n) /= len;
    }
}

static std::vector<double> make_matrix_with_cond(int n, double kappa, std::mt19937_64& rng) {
    std::vector<double> U = alloc_matrix(n);
    std::vector<double> V = alloc_matrix(n);
    random_orthogonal(U, n, rng);
    random_orthogonal(V, n, rng);

    std::vector<double> sigma(n);
    for (int i = 0; i < n; ++i) {
        sigma[i] = std::pow(kappa, static_cast<double>(i) / static_cast<double>(n - 1));
    }

    std::vector<double> A = alloc_matrix(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += M(U, i, k, n) * sigma[k] * M(V, j, k, n);
            }
            M(A, i, j, n) = sum;
        }
    }
    return A;
}

static std::vector<double> make_scaled_dense_matrix(int n) {
    std::vector<double> A = alloc_matrix(n);
    const double diag = 2.0;
    const double off_scale = 0.25 / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double wave = std::sin(0.013 * static_cast<double>((i + 1) * (j + 1))) +
                          std::cos(0.017 * static_cast<double>((i + 3) + (j + 5)));
            M(A, i, j, n) = (i == j) ? diag + 0.001 * static_cast<double>((i % 17) + 1)
                                     : off_scale * wave;
        }
    }
    return A;
}

__global__ void init_pivots_kernel(int* piv, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) piv[i] = i;
}

__global__ void pivot_kernel(const matrix_t* __restrict__ A,
                             int n,
                             int k,
                             int* __restrict__ pivot_row,
                             double* __restrict__ pivot_abs,
                             int* __restrict__ ok) {
    extern __shared__ unsigned char shared_raw[];
    double* vals = reinterpret_cast<double*>(shared_raw);
    int* idx = reinterpret_cast<int*>(vals + blockDim.x);
    int tid = threadIdx.x;
    int row = k + tid;

    double best = -1.0;
    int best_idx = k;
    while (row < n) {
        double v = fabs(static_cast<double>(A[static_cast<size_t>(row) * n + k]));
        v = fabs(mp_cuda::round_to<DLU_PIVOT_E, DLU_PIVOT_T>(v));
        if (v > best) {
            best = v;
            best_idx = row;
        }
        row += blockDim.x;
    }

    vals[tid] = best;
    idx[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && vals[tid + stride] > vals[tid]) {
            vals[tid] = vals[tid + stride];
            idx[tid] = idx[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *pivot_row = idx[0];
        *pivot_abs = vals[0];
        if (vals[0] < 1e-15) *ok = 0;
    }
}

__global__ void swap_rows_kernel(matrix_t* A,
                                 int* piv,
                                 int n,
                                 int k,
                                 const int* __restrict__ pivot_row,
                                 const int* __restrict__ ok) {
    if (*ok == 0) return;
    int p = *pivot_row;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n && p != k) {
        matrix_t tmp = A[static_cast<size_t>(k) * n + j];
        A[static_cast<size_t>(k) * n + j] = A[static_cast<size_t>(p) * n + j];
        A[static_cast<size_t>(p) * n + j] = tmp;
    }
    if (j == 0 && p != k) {
        int tmp = piv[k];
        piv[k] = piv[p];
        piv[p] = tmp;
    }
}

__global__ void scale_column_kernel(matrix_t* A,
                                    int n,
                                    int k,
                                    const int* __restrict__ ok) {
    if (*ok == 0) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    if (i >= n) return;
    double akk = mp_cuda::round_to<DLU_AKK_E, DLU_AKK_T>(
        static_cast<double>(A[static_cast<size_t>(k) * n + k]));
    double lik = static_cast<double>(A[static_cast<size_t>(i) * n + k]) / akk;
    A[static_cast<size_t>(i) * n + k] = matrix_t(lik);
}

__global__ void eliminate_kernel(matrix_t* A,
                                 int n,
                                 int k,
                                 const int* __restrict__ ok) {
    if (*ok == 0) return;
    int j = blockIdx.x * blockDim.x + threadIdx.x + k + 1;
    int i = blockIdx.y * blockDim.y + threadIdx.y + k + 1;
    if (i >= n || j >= n) return;

    double lik = static_cast<double>(A[static_cast<size_t>(i) * n + k]);
    double akj = static_cast<double>(A[static_cast<size_t>(k) * n + j]);
    double aij = static_cast<double>(A[static_cast<size_t>(i) * n + j]);
    A[static_cast<size_t>(i) * n + j] = matrix_t(aij - lik * akj);
}

static bool gpu_lu_factorize(matrix_t* d_A, int* d_piv, int n, double& elapsed_ms) {
    init_pivots_kernel<<<(n + 255) / 256, 256>>>(d_piv, n);
    MP_CUDA_CHECK(cudaGetLastError());

    int* d_pivot_row = nullptr;
    double* d_pivot_abs = nullptr;
    int* d_ok = nullptr;
    MP_CUDA_CHECK(cudaMalloc(&d_pivot_row, sizeof(int)));
    MP_CUDA_CHECK(cudaMalloc(&d_pivot_abs, sizeof(double)));
    MP_CUDA_CHECK(cudaMalloc(&d_ok, sizeof(int)));
    int ok_value = 1;
    MP_CUDA_CHECK(cudaMemcpy(d_ok, &ok_value, sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    MP_CUDA_CHECK(cudaEventCreate(&start));
    MP_CUDA_CHECK(cudaEventCreate(&stop));
    MP_CUDA_CHECK(cudaEventRecord(start));

    bool ok = true;
    for (int k = 0; k < n; ++k) {
        size_t shmem = 1024 * sizeof(double) + 1024 * sizeof(int);
        pivot_kernel<<<1, 1024, shmem>>>(d_A, n, k, d_pivot_row, d_pivot_abs, d_ok);
        MP_CUDA_CHECK(cudaGetLastError());

        swap_rows_kernel<<<(n + 255) / 256, 256>>>(d_A, d_piv, n, k, d_pivot_row, d_ok);
        MP_CUDA_CHECK(cudaGetLastError());

        int trailing = n - k - 1;
        if (trailing > 0) {
            scale_column_kernel<<<(trailing + 255) / 256, 256>>>(d_A, n, k, d_ok);
            MP_CUDA_CHECK(cudaGetLastError());
        }

        dim3 block(16, 16);
        dim3 grid((trailing + block.x - 1) / block.x,
                  (trailing + block.y - 1) / block.y);
        if (grid.x > 0 && grid.y > 0) {
            eliminate_kernel<<<grid, block>>>(d_A, n, k, d_ok);
            MP_CUDA_CHECK(cudaGetLastError());
        }
    }

    MP_CUDA_CHECK(cudaEventRecord(stop));
    MP_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    MP_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    elapsed_ms = static_cast<double>(ms);

    MP_CUDA_CHECK(cudaMemcpy(&ok_value, d_ok, sizeof(int), cudaMemcpyDeviceToHost));
    ok = (ok_value != 0);

    cudaFree(d_pivot_row);
    cudaFree(d_pivot_abs);
    cudaFree(d_ok);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ok;
}

static std::vector<double> forward_sub(const std::vector<double>& LU,
                                       const std::vector<int>& piv,
                                       const std::vector<double>& b,
                                       int n) {
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) y[i] = b[piv[i]];
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            y[i] -= M(LU, i, j, n) * y[j];
        }
    }
    return y;
}

static std::vector<double> back_sub(const std::vector<double>& LU,
                                    const std::vector<double>& y,
                                    int n) {
    std::vector<double> x = y;
    for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) {
            x[i] -= M(LU, i, j, n) * x[j];
        }
        x[i] /= M(LU, i, i, n);
    }
    return x;
}

static std::vector<double> lu_solve(const std::vector<double>& LU,
                                    const std::vector<int>& piv,
                                    const std::vector<double>& b,
                                    int n) {
    std::vector<double> y = forward_sub(LU, piv, b, n);
    return back_sub(LU, y, n);
}

static bool write_solution_vector(const char* path, const std::vector<double>& x) {
    if (path == nullptr || path[0] == '\0') return true;
    std::ofstream out(path);
    if (!out) return false;
    out << std::scientific << std::setprecision(17);
    for (double v : x) {
        out << v << '\n';
    }
    return static_cast<bool>(out);
}

int run(int argc, char** argv) {
    int n = 5000;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    const char* solution_output_path = (argc > 2) ? argv[2] : nullptr;
    if (n <= 1) {
        std::fprintf(stderr, "usage: dense_lu_cuda [matrix_size] [solution_output_path]\n");
        return EXIT_FAILURE;
    }

    std::vector<double> A = make_scaled_dense_matrix(n);
    std::vector<double> x_true(n, 1.0);
    std::vector<double> b(n, 0.0);
    mat_vec(A, x_true, b, n);

    std::vector<matrix_t> LU_dev(A.size());
    for (size_t i = 0; i < A.size(); ++i) {
        LU_dev[i] = matrix_t(A[i]);
    }

    matrix_t* d_LU = nullptr;
    int* d_piv = nullptr;
    size_t device_bytes = LU_dev.size() * sizeof(matrix_t) +
                          static_cast<size_t>(n) * sizeof(int) +
                          sizeof(int) + sizeof(double) + sizeof(int);
    MP_CUDA_CHECK(cudaMalloc(&d_LU, LU_dev.size() * sizeof(matrix_t)));
    MP_CUDA_CHECK(cudaMalloc(&d_piv, n * sizeof(int)));
    MP_CUDA_CHECK(cudaMemcpy(d_LU, LU_dev.data(), LU_dev.size() * sizeof(matrix_t), cudaMemcpyHostToDevice));

    double factor_ms = 0.0;
    bool ok = gpu_lu_factorize(d_LU, d_piv, n, factor_ms);
    if (!ok) {
        std::fprintf(stderr, "GPU LU factorization failed: singular matrix to working precision\n");
        cudaFree(d_LU);
        cudaFree(d_piv);
        return EXIT_FAILURE;
    }

    std::vector<int> piv(n);
    MP_CUDA_CHECK(cudaMemcpy(LU_dev.data(), d_LU, LU_dev.size() * sizeof(matrix_t), cudaMemcpyDeviceToHost));
    MP_CUDA_CHECK(cudaMemcpy(piv.data(), d_piv, n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<double> LU(A.size());
    for (size_t i = 0; i < LU_dev.size(); ++i) {
        LU[i] = static_cast<double>(LU_dev[i]);
    }

    std::vector<double> x_comp = lu_solve(LU, piv, b, n);
    std::vector<double> Ax(n), res(n), err(n);
    mat_vec(A, x_comp, Ax, n);
    for (int i = 0; i < n; ++i) {
        res[i] = Ax[i] - b[i];
        err[i] = x_comp[i] - x_true[i];
    }
    double rel_res = norm2(res) / norm2(b);
    double rel_err = norm2(err) / norm2(x_true);

    if (!write_solution_vector(solution_output_path, x_comp)) {
        std::fprintf(stderr, "failed to write dense LU solution vector: %s\n", solution_output_path);
        cudaFree(d_LU);
        cudaFree(d_piv);
        return EXIT_FAILURE;
    }

    std::cout << DLU_RUN_LABEL << "\n";
    std::cout << "N=" << n << " matrix=scaled_diagonally_dominant\n";
    std::cout << std::fixed << std::setprecision(6)
              << "factorization_time_ms=" << factor_ms << "\n";
    std::cout << "device_allocation_bytes=" << device_bytes << "\n";
    std::cout << std::fixed << std::setprecision(6)
              << "device_allocation_mib=" << static_cast<double>(device_bytes) / (1024.0 * 1024.0) << "\n";
    std::cout << "storage_bytes_matrix=" << sizeof(matrix_t) << " pivot=" << sizeof(int) << "\n";
    std::cout << std::scientific << std::setprecision(6)
              << "relative_residual=" << rel_res << "\n"
              << "relative_error=" << rel_err << "\n";

    cudaFree(d_LU);
    cudaFree(d_piv);
    return EXIT_SUCCESS;
}

#undef M

}  // namespace dense_lu_h100

int main(int argc, char** argv) {
    return dense_lu_h100::run(argc, argv);
}
