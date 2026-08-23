// Tensor Core complement for the dense-LU trailing update.
//
// The direct CUDA dense-LU port follows the PROMISE-transformed scalar
// factorization.  This complement isolates the blocked trailing update
// C := C - L21 * U12, which is the part of a blocked LU factorization that
// can be mapped cleanly to GEMM/Tensor Cores without changing the numerical
// role of the update.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,       \
                         __LINE__, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__,     \
                         __LINE__, static_cast<int>(status));                  \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace {

struct Stats {
    double mean_ms = 0.0;
    double stddev_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
};

struct AccuracyStats {
    double relative_l2 = 0.0;
    double relative_linf = 0.0;
};

template <typename T>
static Stats summarize(const std::vector<T>& values) {
    Stats s;
    if (values.empty()) return s;
    s.min_ms = static_cast<double>(*std::min_element(values.begin(), values.end()));
    s.max_ms = static_cast<double>(*std::max_element(values.begin(), values.end()));
    for (T v : values) s.mean_ms += static_cast<double>(v);
    s.mean_ms /= static_cast<double>(values.size());
    if (values.size() > 1) {
        double ss = 0.0;
        for (T v : values) {
            double d = static_cast<double>(v) - s.mean_ms;
            ss += d * d;
        }
        s.stddev_ms = std::sqrt(ss / static_cast<double>(values.size() - 1));
    }
    return s;
}

static std::vector<float> random_float_data(size_t n) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n);
    for (float& x : data) x = dist(rng);
    return data;
}

template <typename T>
static double to_double(T x) {
    return static_cast<double>(x);
}

static double to_double(__half x) {
    return static_cast<double>(__half2float(x));
}

static double to_double(__nv_bfloat16 x) {
    return static_cast<double>(__bfloat162float(x));
}

template <typename T>
static AccuracyStats compare_with_fp64_reference(cublasHandle_t handle,
                                                 const std::vector<float>& l_src,
                                                 const std::vector<float>& u_src,
                                                 const std::vector<float>& c_src,
                                                 const T* d_c_result,
                                                 int m,
                                                 int k,
                                                 int cols) {
    const size_t l_elems = static_cast<size_t>(m) * k;
    const size_t u_elems = static_cast<size_t>(k) * cols;
    const size_t c_elems = static_cast<size_t>(m) * cols;

    std::vector<double> h_l(l_elems), h_u(u_elems), h_c(c_elems);
    for (size_t i = 0; i < l_elems; ++i) h_l[i] = static_cast<double>(l_src[i]);
    for (size_t i = 0; i < u_elems; ++i) h_u[i] = static_cast<double>(u_src[i]);
    for (size_t i = 0; i < c_elems; ++i) h_c[i] = static_cast<double>(c_src[i]);

    double* d_l_ref = nullptr;
    double* d_u_ref = nullptr;
    double* d_c_ref = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l_ref, l_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_u_ref, u_elems * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_c_ref, c_elems * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_l_ref, h_l.data(), l_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u_ref, h_u.data(), u_elems * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_ref, h_c.data(), c_elems * sizeof(double), cudaMemcpyHostToDevice));

    const double alpha = -1.0;
    const double beta = 1.0;
    CUBLAS_CHECK(cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m,
        cols,
        k,
        &alpha,
        d_l_ref,
        m,
        d_u_ref,
        k,
        &beta,
        d_c_ref,
        m));

    std::vector<double> ref(c_elems);
    std::vector<T> result(c_elems);
    CUDA_CHECK(cudaMemcpy(ref.data(), d_c_ref, c_elems * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(result.data(), d_c_result, c_elems * sizeof(T), cudaMemcpyDeviceToHost));

    double err_l2 = 0.0;
    double ref_l2 = 0.0;
    double err_linf = 0.0;
    double ref_linf = 0.0;
    for (size_t i = 0; i < c_elems; ++i) {
        double r = ref[i];
        double v = to_double(result[i]);
        double diff = v - r;
        err_l2 += diff * diff;
        ref_l2 += r * r;
        err_linf = std::max(err_linf, std::abs(diff));
        ref_linf = std::max(ref_linf, std::abs(r));
    }

    cudaFree(d_l_ref);
    cudaFree(d_u_ref);
    cudaFree(d_c_ref);

    AccuracyStats stats;
    stats.relative_l2 = (ref_l2 > 0.0) ? std::sqrt(err_l2) / std::sqrt(ref_l2) : std::sqrt(err_l2);
    stats.relative_linf = (ref_linf > 0.0) ? err_linf / ref_linf : err_linf;
    return stats;
}

template <typename T>
struct TypeTraits;

template <>
struct TypeTraits<double> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_64F;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_64F;
    static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
    static constexpr const char* name = "fp64";
    using Alpha = double;
    static double convert(float x) { return static_cast<double>(x); }
    static Alpha alpha(float x) { return static_cast<double>(x); }
};

template <>
struct TypeTraits<float> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_32F;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_TF32;
    static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    static constexpr const char* name = "tf32";
    using Alpha = float;
    static float convert(float x) { return x; }
    static Alpha alpha(float x) { return x; }
};

template <>
struct TypeTraits<__half> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16F;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    static constexpr const char* name = "fp16";
    using Alpha = float;
    static __half convert(float x) { return __float2half(x); }
    static Alpha alpha(float x) { return x; }
};

template <>
struct TypeTraits<__nv_bfloat16> {
    static constexpr cudaDataType_t cuda_type = CUDA_R_16BF;
    static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    static constexpr const char* name = "bf16";
    using Alpha = float;
    static __nv_bfloat16 convert(float x) { return __float2bfloat16(x); }
    static Alpha alpha(float x) { return x; }
};

template <typename T>
static int run_update(int n, int panel, int warmup_runs, int measured_runs) {
    if (panel <= 0 || panel > n) {
        std::fprintf(stderr, "panel must be in [1, n]\n");
        return EXIT_FAILURE;
    }

    const int m = n - panel;
    const int k = panel;
    const int cols = n - panel;
    const size_t l_elems = static_cast<size_t>(m) * k;
    const size_t u_elems = static_cast<size_t>(k) * cols;
    const size_t c_elems = static_cast<size_t>(m) * cols;

    std::vector<float> l_src = random_float_data(l_elems);
    std::vector<float> u_src = random_float_data(u_elems);
    std::vector<float> c_src = random_float_data(c_elems);
    std::vector<T> h_l(l_elems), h_u(u_elems), h_c(c_elems);
    for (size_t i = 0; i < l_elems; ++i) h_l[i] = TypeTraits<T>::convert(l_src[i]);
    for (size_t i = 0; i < u_elems; ++i) h_u[i] = TypeTraits<T>::convert(u_src[i]);
    for (size_t i = 0; i < c_elems; ++i) h_c[i] = TypeTraits<T>::convert(c_src[i]);

    T* d_l = nullptr;
    T* d_u = nullptr;
    T* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l, l_elems * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_u, u_elems * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_c, c_elems * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_l, h_l.data(), l_elems * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u, h_u.data(), u_elems * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), c_elems * sizeof(T), cudaMemcpyHostToDevice));

    cublasHandle_t handle = nullptr;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    typename TypeTraits<T>::Alpha alpha = TypeTraits<T>::alpha(-1.0f);
    typename TypeTraits<T>::Alpha beta = TypeTraits<T>::alpha(1.0f);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto one_run = [&]() {
        CUDA_CHECK(cudaMemcpy(d_c, h_c.data(), c_elems * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(start));
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            cols,
            k,
            &alpha,
            d_l,
            TypeTraits<T>::cuda_type,
            m,
            d_u,
            TypeTraits<T>::cuda_type,
            k,
            &beta,
            d_c,
            TypeTraits<T>::cuda_type,
            m,
            TypeTraits<T>::compute_type,
            TypeTraits<T>::algo));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    };

    for (int i = 0; i < warmup_runs; ++i) {
        (void)one_run();
    }

    std::vector<float> runs;
    runs.reserve(static_cast<size_t>(measured_runs));
    for (int i = 0; i < measured_runs; ++i) {
        runs.push_back(one_run());
    }
    Stats stats = summarize(runs);
    AccuracyStats accuracy = compare_with_fp64_reference(handle, l_src, u_src, c_src, d_c, m, k, cols);

    const double flops = 2.0 * static_cast<double>(m) * cols * k;
    const size_t device_bytes = (l_elems + u_elems + c_elems) * sizeof(T);

    std::printf("benchmark=dense_lu_panel_update\n");
    std::printf("mode=%s\n", TypeTraits<T>::name);
    std::printf("n=%d\n", n);
    std::printf("panel=%d\n", panel);
    std::printf("m=%d\n", m);
    std::printf("cols=%d\n", cols);
    std::printf("warmup_runs=%d\n", warmup_runs);
    std::printf("measured_runs=%d\n", measured_runs);
    std::printf("kernel_time_ms=%.9g\n", stats.mean_ms);
    std::printf("kernel_time_ms_stddev=%.9g\n", stats.stddev_ms);
    std::printf("kernel_time_ms_min=%.9g\n", stats.min_ms);
    std::printf("kernel_time_ms_max=%.9g\n", stats.max_ms);
    std::printf("device_allocation_bytes=%zu\n", device_bytes);
    std::printf("device_allocation_mib=%.9g\n",
                static_cast<double>(device_bytes) / (1024.0 * 1024.0));
    std::printf("relative_l2_error_vs_fp64=%.9g\n", accuracy.relative_l2);
    std::printf("relative_linf_error_vs_fp64=%.9g\n", accuracy.relative_linf);
    std::printf("gflops=%.9g\n", (flops / (stats.mean_ms / 1000.0)) / 1.0e9);
    std::printf("uses_tensor_core_candidate=%d\n",
                std::strcmp(TypeTraits<T>::name, "fp64") == 0 ? 0 : 1);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cublasDestroy(handle);
    cudaFree(d_l);
    cudaFree(d_u);
    cudaFree(d_c);
    return EXIT_SUCCESS;
}

static void usage(const char* prog) {
    std::fprintf(stderr,
                 "usage: %s <n> <panel> <fp64|tf32|fp16|bf16> [warmup_runs] [measured_runs]\n",
                 prog);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 6) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }
    int n = std::atoi(argv[1]);
    int panel = std::atoi(argv[2]);
    std::string mode = argv[3];
    int warmup_runs = (argc >= 5) ? std::atoi(argv[4]) : 1;
    int measured_runs = (argc >= 6) ? std::atoi(argv[5]) : 5;

    if (n <= 1 || panel <= 0 || warmup_runs < 0 || measured_runs <= 0) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (mode == "fp64") return run_update<double>(n, panel, warmup_runs, measured_runs);
    if (mode == "tf32" || mode == "fp32") return run_update<float>(n, panel, warmup_runs, measured_runs);
    if (mode == "fp16") return run_update<__half>(n, panel, warmup_runs, measured_runs);
    if (mode == "bf16") return run_update<__nv_bfloat16>(n, panel, warmup_runs, measured_runs);

    usage(argv[0]);
    return EXIT_FAILURE;
}
