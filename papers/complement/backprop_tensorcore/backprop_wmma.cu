// Backprop with WMMA Tensor Core acceleration
// Uses WMMA API for matrix multiply in backpropagation

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

#include "../../cuda_common/h100_mixed_precision.cuh"
#include "../../cuda_common/h100_profiling.cuh"

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

namespace backprop_tc {

// WMMA-based matrix multiply: C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void wmma_gemm_kernel(__half* __restrict__ C,
                                  const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    if (warpM >= (M + WMMA_M - 1) / WMMA_M || warpN >= (N + WMMA_N - 1) / WMMA_N)
        return;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
    
    wmma::fill_fragment(c_frag, __float2half(0.0f));
    
    // Accumulate over K dimension
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;
        
        if (aRow < M && aCol < K) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
        }
        if (bRow < K && bCol < N) {
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
        }
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store result
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// Backprop layer computation
struct BackpropResult {
    double forward_ms;
    double backward_ms;
    double total_ms;
    size_t device_bytes;
};

BackpropResult run_backprop_layer(int layer_size, int batch_size, int num_iterations) {
    BackpropResult result = {0.0, 0.0, 0.0, 0};
    
    // Allocate host memory
    size_t weight_size = layer_size * layer_size;
    size_t activation_size = layer_size * batch_size;
    
    std::vector<float> weights_fp32(weight_size);
    std::vector<float> input_fp32(activation_size);
    std::vector<float> grad_output_fp32(activation_size);
    
    // Initialize with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (auto& w : weights_fp32) w = dist(rng);
    for (auto& i : input_fp32) i = dist(rng);
    for (auto& g : grad_output_fp32) g = dist(rng);
    
    // Convert to FP16
    PROFILE_START("fp32_to_fp16_conversion", mp_profiling::Category::TYPE_CONVERSION);
    std::vector<__half> weights(weight_size);
    std::vector<__half> input(activation_size);
    std::vector<__half> grad_output(activation_size);
    
    for (size_t i = 0; i < weight_size; ++i) {
        weights[i] = __float2half(weights_fp32[i]);
    }
    for (size_t i = 0; i < activation_size; ++i) {
        input[i] = __float2half(input_fp32[i]);
        grad_output[i] = __float2half(grad_output_fp32[i]);
    }
    PROFILE_STOP(0, weight_size + 2 * activation_size);
    
    // Allocate device memory
    __half *d_weights, *d_input, *d_output, *d_grad_output, *d_grad_input;
    
    PROFILE_START("device_alloc", mp_profiling::Category::MEMORY_TRANSFER);
    MP_CUDA_CHECK(cudaMalloc(&d_weights, weight_size * sizeof(__half)));
    MP_CUDA_CHECK(cudaMalloc(&d_input, activation_size * sizeof(__half)));
    MP_CUDA_CHECK(cudaMalloc(&d_output, activation_size * sizeof(__half)));
    MP_CUDA_CHECK(cudaMalloc(&d_grad_output, activation_size * sizeof(__half)));
    MP_CUDA_CHECK(cudaMalloc(&d_grad_input, activation_size * sizeof(__half)));
    result.device_bytes = weight_size * sizeof(__half) + 4 * activation_size * sizeof(__half);
    PROFILE_STOP(result.device_bytes, 0);
    
    // Copy to device
    PROFILE_START("H2D_data", mp_profiling::Category::MEMORY_TRANSFER);
    MP_CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), weight_size * sizeof(__half), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_input, input.data(), activation_size * sizeof(__half), cudaMemcpyHostToDevice));
    MP_CUDA_CHECK(cudaMemcpy(d_grad_output, grad_output.data(), activation_size * sizeof(__half), cudaMemcpyHostToDevice));
    PROFILE_STOP((weight_size + 2 * activation_size) * sizeof(__half), 0);
    
    // Setup WMMA kernel launch params
    dim3 block(128, 1);
    dim3 grid_forward((layer_size + WMMA_M - 1) / WMMA_M, (batch_size + WMMA_N - 1) / WMMA_N);
    dim3 grid_backward((batch_size + WMMA_M - 1) / WMMA_M, (layer_size + WMMA_N - 1) / WMMA_N);
    
    // Run forward and backward passes
    cudaEvent_t start, stop;
    MP_CUDA_CHECK(cudaEventCreate(&start));
    MP_CUDA_CHECK(cudaEventCreate(&stop));
    
    PROFILE_START("wmma_forward_pass", mp_profiling::Category::COMPUTATION);
    MP_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        // Forward: output = weights * input
        wmma_gemm_kernel<<<grid_forward, block>>>(d_output, d_weights, d_input, 
                                                   layer_size, batch_size, layer_size);
    }
    MP_CUDA_CHECK(cudaEventRecord(stop));
    MP_CUDA_CHECK(cudaEventSynchronize(stop));
    float forward_ms = 0.0f;
    MP_CUDA_CHECK(cudaEventElapsedTime(&forward_ms, start, stop));
    result.forward_ms = forward_ms / num_iterations;
    PROFILE_STOP(0, 0);
    
    PROFILE_START("wmma_backward_pass", mp_profiling::Category::COMPUTATION);
    MP_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        // Backward: grad_input = weights^T * grad_output
        wmma_gemm_kernel<<<grid_backward, block>>>(d_grad_input, d_grad_output, d_weights,
                                                    batch_size, layer_size, layer_size);
    }
    MP_CUDA_CHECK(cudaEventRecord(stop));
    MP_CUDA_CHECK(cudaEventSynchronize(stop));
    float backward_ms = 0.0f;
    MP_CUDA_CHECK(cudaEventElapsedTime(&backward_ms, start, stop));
    result.backward_ms = backward_ms / num_iterations;
    PROFILE_STOP(0, 0);
    
    result.total_ms = result.forward_ms + result.backward_ms;
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_weights);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    
    return result;
}

int run(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <layer_size> <batch_size> <iterations>\n", argv[0]);
        return EXIT_FAILURE;
    }
    
    int layer_size = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
    int iterations = atoi(argv[3]);
    
    if (layer_size <= 0 || batch_size <= 0 || iterations <= 0 ||
        layer_size % 16 != 0 || batch_size % 16 != 0) {
        fprintf(stderr, "Arguments must be positive; layer_size and batch_size must be divisible by 16\n");
        return EXIT_FAILURE;
    }
    
    printf("Backprop WMMA Tensor Core Benchmark\n");
    printf("Layer size: %d, Batch size: %d, Iterations: %d\n\n", 
           layer_size, batch_size, iterations);
    
    BackpropResult result = run_backprop_layer(layer_size, batch_size, iterations);
    
    printf("Results (per iteration):\n");
    printf("  Forward pass:  %.4f ms\n", result.forward_ms);
    printf("  Backward pass: %.4f ms\n", result.backward_ms);
    printf("  Total:         %.4f ms\n", result.total_ms);
    printf("kernel_time_ms=%.9g\n", result.total_ms);
    printf("forward_time_ms=%.9g\n", result.forward_ms);
    printf("backward_time_ms=%.9g\n", result.backward_ms);
    printf("device_allocation_bytes=%zu\n", result.device_bytes);
    printf("device_allocation_mib=%.9g\n", result.device_bytes / (1024.0 * 1024.0));
    printf("uses_tensor_core_candidate=1\n");
    
    double forward_gflops = (2.0 * layer_size * layer_size * batch_size) / 
                            (result.forward_ms / 1000.0) / 1e9;
    double backward_gflops = (2.0 * layer_size * layer_size * batch_size) / 
                             (result.backward_ms / 1000.0) / 1e9;
    
    printf("\nPerformance:\n");
    printf("  Forward:  %.2f GFLOPS\n", forward_gflops);
    printf("  Backward: %.2f GFLOPS\n\n", backward_gflops);
    
    PROFILE_SUMMARY();
    PROFILE_EXPORT("backprop_wmma_profile.csv");
    
    return EXIT_SUCCESS;
}

}  // namespace backprop_tc

int main(int argc, char** argv) {
    return backprop_tc::run(argc, argv);
}
