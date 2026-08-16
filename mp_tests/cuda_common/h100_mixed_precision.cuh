#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if defined(__has_include)
#  if __has_include(<cuda_fp8.h>)
#    include <cuda_fp8.h>
#    define MP_CUDA_HAS_FP8 1
#  endif
#endif

#ifndef MP_CUDA_HAS_FP8
#  define MP_CUDA_HAS_FP8 0
#endif

#define MP_CUDA_CHECK(expr)                                                   \
    do {                                                                      \
        cudaError_t _mp_cuda_err = (expr);                                    \
        if (_mp_cuda_err != cudaSuccess) {                                    \
            std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",     \
                         __FILE__, __LINE__, #expr,                           \
                         cudaGetErrorString(_mp_cuda_err));                   \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

namespace mp_cuda {

template <int E, int T>
struct storage {
    float value;

    __host__ __device__ storage() : value(0.0f) {}
    __host__ __device__ explicit storage(float x) : value(quantize(x)) {}
    __host__ __device__ explicit storage(double x)
        : value(quantize(static_cast<float>(x))) {}

    __host__ __device__ operator float() const { return value; }
    __host__ __device__ operator double() const {
        return static_cast<double>(value);
    }

    __host__ __device__ static float quantize(float x) {
        if (!::isfinite(x) || x == 0.0f) return x;
        int exp2 = 0;
        float mant = ::frexpf(::fabsf(x), &exp2);
        const float levels = ::ldexpf(1.0f, T);
        float qmant = ::floorf(mant * levels + 0.5f) / levels;
        if (qmant >= 1.0f) {
            qmant = 0.5f;
            ++exp2;
        }
        const int max_exp = (1 << (E - 1)) - 1;
        const int min_exp = 2 - (1 << (E - 1));
        if (exp2 > max_exp) return ::copysignf(INFINITY, x);
        if (exp2 < min_exp) return ::copysignf(0.0f, x);
        return ::copysignf(::ldexpf(qmant, exp2), x);
    }
};

template <>
struct storage<8, 23> {
    float value;
    __host__ __device__ storage() : value(0.0f) {}
    __host__ __device__ explicit storage(float x) : value(x) {}
    __host__ __device__ explicit storage(double x) : value(static_cast<float>(x)) {}
    __host__ __device__ operator float() const { return value; }
    __host__ __device__ operator double() const { return static_cast<double>(value); }
};

template <>
struct storage<11, 52> {
    double value;
    __host__ __device__ storage() : value(0.0) {}
    __host__ __device__ explicit storage(float x) : value(static_cast<double>(x)) {}
    __host__ __device__ explicit storage(double x) : value(x) {}
    __host__ __device__ operator float() const { return static_cast<float>(value); }
    __host__ __device__ operator double() const { return value; }
};

template <>
struct storage<5, 10> {
    __half value;
    __host__ __device__ storage() : value(__float2half(0.0f)) {}
    __host__ __device__ explicit storage(float x) : value(__float2half(x)) {}
    __host__ __device__ explicit storage(double x)
        : value(__float2half(static_cast<float>(x))) {}
    __host__ __device__ operator float() const { return __half2float(value); }
    __host__ __device__ operator double() const {
        return static_cast<double>(__half2float(value));
    }
};

template <>
struct storage<8, 7> {
    __nv_bfloat16 value;
    __host__ __device__ storage() : value(__float2bfloat16(0.0f)) {}
    __host__ __device__ explicit storage(float x) : value(__float2bfloat16(x)) {}
    __host__ __device__ explicit storage(double x)
        : value(__float2bfloat16(static_cast<float>(x))) {}
    __host__ __device__ operator float() const { return __bfloat162float(value); }
    __host__ __device__ operator double() const {
        return static_cast<double>(__bfloat162float(value));
    }
};

#if MP_CUDA_HAS_FP8
template <>
struct storage<5, 2> {
    __nv_fp8_e5m2 value;
    __host__ __device__ storage() : value(__nv_fp8_e5m2(0.0f)) {}
    __host__ __device__ explicit storage(float x) : value(__nv_fp8_e5m2(x)) {}
    __host__ __device__ explicit storage(double x)
        : value(__nv_fp8_e5m2(static_cast<float>(x))) {}
    __host__ __device__ operator float() const { return static_cast<float>(value); }
    __host__ __device__ operator double() const {
        return static_cast<double>(static_cast<float>(value));
    }
};

template <>
struct storage<4, 3> {
    __nv_fp8_e4m3 value;
    __host__ __device__ storage() : value(__nv_fp8_e4m3(0.0f)) {}
    __host__ __device__ explicit storage(float x) : value(__nv_fp8_e4m3(x)) {}
    __host__ __device__ explicit storage(double x)
        : value(__nv_fp8_e4m3(static_cast<float>(x))) {}
    __host__ __device__ operator float() const { return static_cast<float>(value); }
    __host__ __device__ operator double() const {
        return static_cast<double>(static_cast<float>(value));
    }
};
#endif

template <int E, int T>
__host__ __device__ inline double round_to(double x) {
    return static_cast<double>(storage<E, T>(x));
}

template <int E, int T>
__host__ __device__ inline float round_to_float(float x) {
    return static_cast<float>(storage<E, T>(x));
}

}  // namespace mp_cuda
