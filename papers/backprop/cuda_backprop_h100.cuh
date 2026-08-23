#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cuda_common/h100_mixed_precision.cuh"

#ifndef BP_RUN_LABEL
#define BP_RUN_LABEL "H100 CUDA backprop mixed-precision run"
#endif

#ifndef BP_SQUASH_E
#define BP_SQUASH_E 11
#define BP_SQUASH_T 52
#endif

#ifndef BP_OUTPUT_O_E
#define BP_OUTPUT_O_E 11
#define BP_OUTPUT_O_T 52
#endif

#ifndef BP_OUTPUT_T_E
#define BP_OUTPUT_T_E 11
#define BP_OUTPUT_T_T 52
#endif

#ifndef BP_HIDDEN_H_E
#define BP_HIDDEN_H_E 5
#define BP_HIDDEN_H_T 2
#endif

#ifndef BP_HIDDEN_SUM_E
#define BP_HIDDEN_SUM_E 5
#define BP_HIDDEN_SUM_T 2
#endif

#ifndef BP_ADJUST_E
#define BP_ADJUST_E 5
#define BP_ADJUST_T 2
#endif

namespace bp_h100 {

constexpr double kEta = 0.3;
constexpr double kMomentum = 0.3;

using hidden_t = mp_cuda::storage<BP_HIDDEN_H_E, BP_HIDDEN_H_T>;
using output_t = mp_cuda::storage<BP_OUTPUT_O_E, BP_OUTPUT_O_T>;
using target_t = mp_cuda::storage<BP_OUTPUT_T_E, BP_OUTPUT_T_T>;
using old_weight_t = mp_cuda::storage<BP_ADJUST_E, BP_ADJUST_T>;

template <typename T>
static void cuda_malloc_tracked(T** ptr, size_t count, size_t& bytes) {
    bytes += count * sizeof(T);
    MP_CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
}

template <typename InT, typename OutT>
__global__ void layer_forward_kernel(const InT* __restrict__ l1,
                                     OutT* __restrict__ l2,
                                     const double* __restrict__ weights,
                                     int n1,
                                     int n2,
                                     int weight_cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > n2) return;

    double sum = 0.0;
    for (int k = 0; k <= n1; ++k) {
        sum += weights[k * weight_cols + j] * static_cast<double>(l1[k]);
    }

    double x = mp_cuda::round_to<BP_SQUASH_E, BP_SQUASH_T>(sum);
    float y = 1.0f / (1.0f + expf(-static_cast<float>(x)));
    l2[j] = OutT(mp_cuda::round_to<BP_SQUASH_E, BP_SQUASH_T>(y));
}

__global__ void output_error_kernel(double* __restrict__ delta,
                                    const target_t* __restrict__ target,
                                    const output_t* __restrict__ output,
                                    int nj,
                                    double* __restrict__ err) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > nj) return;

    double o = mp_cuda::round_to<BP_OUTPUT_O_E, BP_OUTPUT_O_T>(static_cast<double>(output[j]));
    double t = mp_cuda::round_to<BP_OUTPUT_T_E, BP_OUTPUT_T_T>(static_cast<double>(target[j]));
    double d = o * (1.0 - o) * (t - o);
    delta[j] = d;
    atomicAdd(err, fabs(d));
}

__global__ void hidden_error_kernel(double* __restrict__ delta_h,
                                    int nh,
                                    const double* __restrict__ delta_o,
                                    int no,
                                    const double* __restrict__ who,
                                    int who_cols,
                                    const hidden_t* __restrict__ hidden,
                                    double* __restrict__ err) {
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (j > nh) return;

    double h = mp_cuda::round_to<BP_HIDDEN_H_E, BP_HIDDEN_H_T>(static_cast<double>(hidden[j]));
    double sum = mp_cuda::round_to<BP_HIDDEN_SUM_E, BP_HIDDEN_SUM_T>(0.0);
    for (int k = 1; k <= no; ++k) {
        double term = delta_o[k] * who[j * who_cols + k];
        sum = mp_cuda::round_to<BP_HIDDEN_SUM_E, BP_HIDDEN_SUM_T>(sum + term);
    }

    double d = h * (1.0 - h) * sum;
    delta_h[j] = d;
    atomicAdd(err, fabs(d));
}

template <typename LyT>
__global__ void adjust_weights_kernel(const double* __restrict__ delta,
                                      int ndelta,
                                      const LyT* __restrict__ ly,
                                      int nly,
                                      double* __restrict__ w,
                                      old_weight_t* __restrict__ oldw,
                                      int weight_cols) {
    int linear = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ndelta * (nly + 1);
    if (linear >= total) return;

    int j = linear / (nly + 1) + 1;
    int k = linear % (nly + 1);
    int idx = k * weight_cols + j;
    double new_dw = kEta * delta[j] * static_cast<double>(ly[k]) +
                    kMomentum * static_cast<double>(oldw[idx]);
    new_dw = mp_cuda::round_to<BP_ADJUST_E, BP_ADJUST_T>(new_dw);
    w[idx] += new_dw;
    oldw[idx] = old_weight_t(new_dw);
}

static void fill_weights(std::vector<double>& w, int rows, int cols) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = dist(rng);
    }
}

static void fill_input(std::vector<double>& input, int n) {
    std::srand(1234);
    input[0] = 1.0;
    for (int i = 1; i <= n; ++i) {
        input[i] = static_cast<double>(std::rand()) / RAND_MAX;
    }
}

static double elapsed_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    MP_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return static_cast<double>(ms);
}

static void write_output_delta(const std::vector<double>& output_delta,
                               const std::string& file) {
    if (file.empty() || file == "-") {
        return;
    }
    std::ofstream out(file);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to open output-delta file: " + file);
    }
    out << std::scientific << std::setprecision(17);
    for (double value : output_delta) {
        out << value << "\n";
    }
}

int run(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "usage: backprop_cuda <num of input elements> [output_delta_file]\n");
        return EXIT_FAILURE;
    }

    int input_n = std::atoi(argv[1]);
    if (input_n <= 0 || input_n % 16 != 0) {
        std::fprintf(stderr, "The number of input points must be positive and divisible by 16\n");
        return EXIT_FAILURE;
    }

    constexpr int hidden_n = 16;
    constexpr int output_n = 1;
    int input_units_n = input_n + 1;
    int hidden_units_n = hidden_n + 1;
    int output_units_n = output_n + 1;
    int input_weight_cols = hidden_n + 1;
    int hidden_weight_cols = output_n + 1;

    std::vector<double> input_units(input_units_n, 0.0);
    std::vector<hidden_t> hidden_units(hidden_units_n);
    std::vector<output_t> output_units(output_units_n);
    std::vector<double> hidden_delta(hidden_units_n, 0.0);
    std::vector<double> output_delta(output_units_n, 0.0);
    std::vector<target_t> target(output_units_n);
    std::vector<double> input_weights(input_units_n * input_weight_cols);
    std::vector<double> hidden_weights(hidden_units_n * hidden_weight_cols);
    std::vector<old_weight_t> input_prev_weights(input_units_n * input_weight_cols);
    std::vector<old_weight_t> hidden_prev_weights(hidden_units_n * hidden_weight_cols);

    fill_input(input_units, input_n);
    hidden_units[0] = hidden_t(1.0);
    for (int i = 0; i < output_units_n; ++i) target[i] = target_t(0.1);
    fill_weights(input_weights, input_units_n, input_weight_cols);
    fill_weights(hidden_weights, hidden_units_n, hidden_weight_cols);

    double* d_input = nullptr;
    hidden_t* d_hidden = nullptr;
    output_t* d_output = nullptr;
    double *d_hidden_delta = nullptr, *d_output_delta = nullptr;
    target_t* d_target = nullptr;
    double *d_input_w = nullptr, *d_hidden_w = nullptr;
    old_weight_t *d_input_oldw = nullptr, *d_hidden_oldw = nullptr;
    double *d_out_err = nullptr, *d_hid_err = nullptr;
    size_t device_bytes = 0;

    cuda_malloc_tracked(&d_input, input_units.size(), device_bytes);
    cuda_malloc_tracked(&d_hidden, hidden_units.size(), device_bytes);
    cuda_malloc_tracked(&d_output, output_units.size(), device_bytes);
    cuda_malloc_tracked(&d_hidden_delta, hidden_delta.size(), device_bytes);
    cuda_malloc_tracked(&d_output_delta, output_delta.size(), device_bytes);
    cuda_malloc_tracked(&d_target, target.size(), device_bytes);
    cuda_malloc_tracked(&d_input_w, input_weights.size(), device_bytes);
    cuda_malloc_tracked(&d_hidden_w, hidden_weights.size(), device_bytes);
    cuda_malloc_tracked(&d_input_oldw, input_prev_weights.size(), device_bytes);
    cuda_malloc_tracked(&d_hidden_oldw, hidden_prev_weights.size(), device_bytes);
    cuda_malloc_tracked(&d_out_err, 1, device_bytes);
    cuda_malloc_tracked(&d_hid_err, 1, device_bytes);

    MP_CUDA_CHECK(cudaMemcpy(d_input, input_units.data(), input_units.size() * sizeof(double), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_hidden, hidden_units.data(), hidden_units.size() * sizeof(hidden_t), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_output, output_units.data(), output_units.size() * sizeof(output_t), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_hidden_delta, hidden_delta.data(), hidden_delta.size() * sizeof(double), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_output_delta, output_delta.data(), output_delta.size() * sizeof(double), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_target, target.data(), target.size() * sizeof(target_t), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_input_w, input_weights.data(), input_weights.size() * sizeof(double), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_hidden_w, hidden_weights.data(), hidden_weights.size() * sizeof(double), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_input_oldw, input_prev_weights.data(), input_prev_weights.size() * sizeof(old_weight_t), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_hidden_oldw, hidden_prev_weights.data(), hidden_prev_weights.size() * sizeof(old_weight_t), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemset(d_out_err, 0, sizeof(double)));
    MP_CUDA_CHECK(cudaMemset(d_hid_err, 0, sizeof(double)));

    cudaEvent_t start, stop;
    MP_CUDA_CHECK(cudaEventCreate(&start));
    MP_CUDA_CHECK(cudaEventCreate(&stop));
    MP_CUDA_CHECK(cudaEventRecord(start));

    layer_forward_kernel<<<1, 128>>>(d_input, d_hidden, d_input_w, input_n, hidden_n, input_weight_cols);
    layer_forward_kernel<<<1, 128>>>(d_hidden, d_output, d_hidden_w, hidden_n, output_n, hidden_weight_cols);
    output_error_kernel<<<1, 128>>>(d_output_delta, d_target, d_output, output_n, d_out_err);
    hidden_error_kernel<<<1, 128>>>(d_hidden_delta, hidden_n, d_output_delta, output_n,
                                    d_hidden_w, hidden_weight_cols, d_hidden, d_hid_err);
    int hidden_weight_updates = output_n * (hidden_n + 1);
    adjust_weights_kernel<<<(hidden_weight_updates + 255) / 256, 256>>>(
        d_output_delta, output_n, d_hidden, hidden_n, d_hidden_w, d_hidden_oldw, hidden_weight_cols);
    int input_weight_updates = hidden_n * (input_n + 1);
    adjust_weights_kernel<<<(input_weight_updates + 255) / 256, 256>>>(
        d_hidden_delta, hidden_n, d_input, input_n, d_input_w, d_input_oldw, input_weight_cols);

    MP_CUDA_CHECK(cudaGetLastError());
    MP_CUDA_CHECK(cudaEventRecord(stop));
    MP_CUDA_CHECK(cudaEventSynchronize(stop));
    double kernel_ms = elapsed_ms(start, stop);

    MP_CUDA_CHECK(cudaMemcpy(output_delta.data(), d_output_delta, output_delta.size() * sizeof(double), cudaMemcpyDeviceToHost));
    if (argc == 3) {
        write_output_delta(output_delta, argv[2]);
    }

    std::printf("%s\n", BP_RUN_LABEL);
    std::printf("input=%d hidden=%d output=%d\n", input_n, hidden_n, output_n);
    std::printf("kernel_time_ms=%.6f\n", kernel_ms);
    std::printf("device_allocation_bytes=%zu\n", device_bytes);
    std::printf("device_allocation_mib=%.6f\n", static_cast<double>(device_bytes) / (1024.0 * 1024.0));
    std::printf("storage_bytes_hidden=%zu output=%zu target=%zu old_weight=%zu\n",
                sizeof(hidden_t), sizeof(output_t), sizeof(target_t), sizeof(old_weight_t));
    std::printf("output_delta[1]=%.17g\n", output_delta[1]);

    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_hidden_delta);
    cudaFree(d_output_delta);
    cudaFree(d_target);
    cudaFree(d_input_w);
    cudaFree(d_hidden_w);
    cudaFree(d_input_oldw);
    cudaFree(d_hidden_oldw);
    cudaFree(d_out_err);
    cudaFree(d_hid_err);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return EXIT_SUCCESS;
}

}  // namespace bp_h100

int main(int argc, char** argv) {
    return bp_h100::run(argc, argv);
}
