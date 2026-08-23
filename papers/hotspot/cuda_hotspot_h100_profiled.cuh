#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cuda_common/h100_mixed_precision.cuh"
#include "../cuda_common/h100_profiling.cuh"

#ifndef HS_RUN_LABEL
#define HS_RUN_LABEL "H100 CUDA hotspot mixed-precision run (PROFILED)"
#endif

#ifndef HS_GRID_HEIGHT_E
#define HS_GRID_HEIGHT_E 11
#define HS_GRID_HEIGHT_T 52
#endif

#ifndef HS_GRID_WIDTH_E
#define HS_GRID_WIDTH_E 11
#define HS_GRID_WIDTH_T 52
#endif

#ifndef HS_CAP_E
#define HS_CAP_E 11
#define HS_CAP_T 52
#endif

#ifndef HS_RX_E
#define HS_RX_E 11
#define HS_RX_T 52
#endif

#ifndef HS_RY_E
#define HS_RY_E 11
#define HS_RY_T 52
#endif

#ifndef HS_RZ_E
#define HS_RZ_E 11
#define HS_RZ_T 52
#endif

#ifndef HS_MAX_SLOPE_E
#define HS_MAX_SLOPE_E 11
#define HS_MAX_SLOPE_T 52
#endif

#ifndef HS_STEP_E
#define HS_STEP_E 11
#define HS_STEP_T 52
#endif

#ifndef HS_RX_1_E
#define HS_RX_1_E 11
#define HS_RX_1_T 52
#endif

#ifndef HS_RY_1_E
#define HS_RY_1_E 11
#define HS_RY_1_T 52
#endif

#ifndef HS_RZ_1_E
#define HS_RZ_1_E 11
#define HS_RZ_1_T 52
#endif

#ifndef HS_CAP_1_E
#define HS_CAP_1_E 11
#define HS_CAP_1_T 52
#endif

#ifndef HS_DELTA_E
#define HS_DELTA_E 11
#define HS_DELTA_T 52
#endif

#ifndef HS_FIELD_E
#define HS_FIELD_E HS_DELTA_E
#endif

#ifndef HS_FIELD_T
#define HS_FIELD_T HS_DELTA_T
#endif

namespace hotspot_h100_profiled {

constexpr double kMaxPd = 3.0e6;
constexpr double kPrecision = 0.001;
constexpr double kSpecHeatSi = 1.75e6;
constexpr double kKSi = 100.0;
constexpr double kFactorChip = 0.5;
constexpr double kTChip = 0.0005;
constexpr double kChipHeight = 0.016;
constexpr double kChipWidth = 0.016;
constexpr double kAmbTemp = 80.0;

using grid_height_t = mp_cuda::storage<HS_GRID_HEIGHT_E, HS_GRID_HEIGHT_T>;
using grid_width_t = mp_cuda::storage<HS_GRID_WIDTH_E, HS_GRID_WIDTH_T>;
using cap_t = mp_cuda::storage<HS_CAP_E, HS_CAP_T>;
using rx_t = mp_cuda::storage<HS_RX_E, HS_RX_T>;
using ry_t = mp_cuda::storage<HS_RY_E, HS_RY_T>;
using rz_t = mp_cuda::storage<HS_RZ_E, HS_RZ_T>;
using max_slope_t = mp_cuda::storage<HS_MAX_SLOPE_E, HS_MAX_SLOPE_T>;
using step_t = mp_cuda::storage<HS_STEP_E, HS_STEP_T>;
using rx_1_t = mp_cuda::storage<HS_RX_1_E, HS_RX_1_T>;
using ry_1_t = mp_cuda::storage<HS_RY_1_E, HS_RY_1_T>;
using rz_1_t = mp_cuda::storage<HS_RZ_1_E, HS_RZ_1_T>;
using cap_1_t = mp_cuda::storage<HS_CAP_1_E, HS_CAP_1_T>;
using delta_t = mp_cuda::storage<HS_DELTA_E, HS_DELTA_T>;
using field_t = mp_cuda::storage<HS_FIELD_E, HS_FIELD_T>;

template <typename T>
static void cuda_malloc_tracked(T** ptr, size_t count, size_t& bytes) {
    bytes += count * sizeof(T);
    MP_CUDA_CHECK(cudaMalloc(ptr, count * sizeof(T)));
}

static std::vector<double> read_input(int grid_rows, int grid_cols, const std::string& file) {
    std::ifstream fp(file);
    if (!fp.is_open()) {
        throw std::runtime_error("Unable to open input file: " + file);
    }

    std::vector<double> vect(static_cast<size_t>(grid_rows) * grid_cols);
    std::string line;
    for (size_t i = 0; i < vect.size(); ++i) {
        if (!std::getline(fp, line)) {
            throw std::runtime_error("Not enough lines in file: " + file);
        }
        vect[i] = std::stod(line);
    }
    return vect;
}

static std::vector<field_t> to_field_vector(const std::vector<double>& input) {
    std::vector<field_t> output(input.size());
    
    PROFILE_START("host_type_conversion", mp_profiling::Category::TYPE_CONVERSION);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = field_t(input[i]);
    }
    PROFILE_STOP(0, input.size());
    
    return output;
}

static std::vector<double> to_double_vector(const std::vector<field_t>& input) {
    std::vector<double> output(input.size());
    
    PROFILE_START("host_type_conversion_back", mp_profiling::Category::TYPE_CONVERSION);
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = static_cast<double>(input[i]);
    }
    PROFILE_STOP(0, input.size());
    
    return output;
}

static void write_output(const std::vector<double>& vect,
                         int grid_rows,
                         int grid_cols,
                         const std::string& file) {
    if (file.empty() || file == "-") {
        return;
    }
    std::ofstream fp(file);
    if (!fp.is_open()) {
        throw std::runtime_error("Unable to open output file: " + file);
    }

    int index = 0;
    for (int i = 0; i < grid_rows; ++i) {
        for (int j = 0; j < grid_cols; ++j) {
            fp << index << "\t" << vect[static_cast<size_t>(i) * grid_cols + j] << "\n";
            ++index;
        }
    }
}

__device__ inline double hotspot_delta(double power,
                                       double center,
                                       double north,
                                       double south,
                                       double west,
                                       double east,
                                       double rx_1,
                                       double ry_1,
                                       double rz_1,
                                       double cap_1) {
    double delta = cap_1 * (power +
                           (south + north - 2.0 * center) * ry_1 +
                           (east + west - 2.0 * center) * rx_1 +
                           (kAmbTemp - center) * rz_1);
    return static_cast<double>(delta_t(delta));
}

__global__ void hotspot_iteration_kernel(field_t* __restrict__ result,
                                         const field_t* __restrict__ temp,
                                         const field_t* __restrict__ power,
                                         int row,
                                         int col,
                                         double cap_1,
                                         double rx_1,
                                         double ry_1,
                                         double rz_1) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= row || c >= col) return;

    size_t idx = static_cast<size_t>(r) * col + c;
    
    // Type conversion: field_t -> double (happens on every access)
    double center = static_cast<double>(temp[idx]);
    double north = (r == 0) ? center : static_cast<double>(temp[idx - col]);
    double south = (r == row - 1) ? center : static_cast<double>(temp[idx + col]);
    double west = (c == 0) ? center : static_cast<double>(temp[idx - 1]);
    double east = (c == col - 1) ? center : static_cast<double>(temp[idx + 1]);

    if (r == 0) north = center;
    if (r == row - 1) south = center;
    if (c == 0) west = center;
    if (c == col - 1) east = center;

    double delta;
    if (r > 0 && r < row - 1 && c > 0 && c < col - 1) {
        delta = hotspot_delta(static_cast<double>(power[idx]), center, north, south, west, east,
                              rx_1, ry_1, rz_1, cap_1);
    } else if (r == 0 && c == 0) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[0]) +
            (static_cast<double>(temp[1]) - static_cast<double>(temp[0])) * rx_1 +
            (static_cast<double>(temp[col]) - static_cast<double>(temp[0])) * ry_1 +
            (kAmbTemp - static_cast<double>(temp[0])) * rz_1)));
    } else if (r == 0 && c == col - 1) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[c]) +
            (static_cast<double>(temp[c - 1]) - static_cast<double>(temp[c])) * rx_1 +
            (static_cast<double>(temp[c + col]) - static_cast<double>(temp[c])) * ry_1 +
            (kAmbTemp - static_cast<double>(temp[c])) * rz_1)));
    } else if (r == row - 1 && c == col - 1) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[idx]) +
            (static_cast<double>(temp[idx - 1]) - static_cast<double>(temp[idx])) * rx_1 +
            (static_cast<double>(temp[idx - col]) - static_cast<double>(temp[idx])) * ry_1 +
            (kAmbTemp - static_cast<double>(temp[idx])) * rz_1)));
    } else if (r == row - 1 && c == 0) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[idx]) +
            (static_cast<double>(temp[idx + 1]) - static_cast<double>(temp[idx])) * rx_1 +
            (static_cast<double>(temp[idx - col]) - static_cast<double>(temp[idx])) * rz_1 +
            (kAmbTemp - static_cast<double>(temp[idx])) * rz_1)));
    } else if (r == 0) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[c]) +
            (static_cast<double>(temp[c + 1]) + static_cast<double>(temp[c - 1]) - 2.0 * static_cast<double>(temp[c])) * rx_1 +
            (static_cast<double>(temp[col + c]) - static_cast<double>(temp[c])) * ry_1 +
            (kAmbTemp - static_cast<double>(temp[c])) * rz_1)));
    } else if (c == col - 1) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[idx]) +
            (static_cast<double>(temp[idx + col]) + static_cast<double>(temp[idx - col]) - 2.0 * static_cast<double>(temp[idx])) * ry_1 +
            (static_cast<double>(temp[idx - 1]) - static_cast<double>(temp[idx])) * rx_1 +
            (kAmbTemp - static_cast<double>(temp[idx])) * rz_1)));
    } else if (r == row - 1) {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[idx]) +
            (static_cast<double>(temp[idx + 1]) + static_cast<double>(temp[idx - 1]) - 2.0 * static_cast<double>(temp[idx])) * rx_1 +
            (static_cast<double>(temp[idx - col]) - static_cast<double>(temp[idx])) * ry_1 +
            (kAmbTemp - static_cast<double>(temp[idx])) * rz_1)));
    } else {
        delta = static_cast<double>(delta_t(cap_1 * (static_cast<double>(power[idx]) +
            (static_cast<double>(temp[idx + col]) + static_cast<double>(temp[idx - col]) - 2.0 * static_cast<double>(temp[idx])) * ry_1 +
            (static_cast<double>(temp[idx + 1]) - static_cast<double>(temp[idx])) * rx_1 +
            (kAmbTemp - static_cast<double>(temp[idx])) * rz_1)));
    }

    // Type conversion: double -> field_t
    result[idx] = field_t(center + delta);
}

static double compute_tran_temp(field_t* d_result,
                                int num_iterations,
                                field_t* d_temp,
                                const field_t* d_power,
                                int row,
                                int col) {
    double grid_height = static_cast<double>(grid_height_t(kChipHeight / row));
    double grid_width = static_cast<double>(grid_width_t(kChipWidth / col));

    double cap = static_cast<double>(cap_t(kFactorChip * kSpecHeatSi * kTChip *
                                           grid_width * grid_height));
    double rx = static_cast<double>(rx_t(grid_width / (2.0 * kKSi * kTChip * grid_height)));
    double ry = static_cast<double>(ry_t(grid_height / (2.0 * kKSi * kTChip * grid_width)));
    double rz = static_cast<double>(rz_t(kTChip / (kKSi * grid_height * grid_width)));

    double max_slope = static_cast<double>(max_slope_t(
        kMaxPd / (kFactorChip * kTChip * kSpecHeatSi)));
    double step = static_cast<double>(step_t(kPrecision / max_slope / 1000.0));

    double rx_1 = static_cast<double>(rx_1_t(1.0 / rx));
    double ry_1 = static_cast<double>(ry_1_t(1.0 / ry));
    double rz_1 = static_cast<double>(rz_1_t(1.0 / rz));
    double cap_1 = static_cast<double>(cap_1_t(step / cap));

    dim3 block(16, 16);
    dim3 grid_dim((col + block.x - 1) / block.x, (row + block.y - 1) / block.y);
    field_t* current = d_temp;
    field_t* next = d_result;
    
    size_t total_elements = static_cast<size_t>(row) * col;
    
    PROFILE_START("hotspot_kernel_total", mp_profiling::Category::COMPUTATION);
    for (int i = 0; i < num_iterations; ++i) {
        hotspot_iteration_kernel<<<grid_dim, block>>>(next, current, d_power, row, col,
                                                  cap_1, rx_1, ry_1, rz_1);
        MP_CUDA_CHECK(cudaGetLastError());
        std::swap(current, next);
    }
    MP_CUDA_CHECK(cudaDeviceSynchronize());
    PROFILE_STOP(0, total_elements * num_iterations * 10);  // ~10 conversions per element

    return 0.0;  // Timing handled by profiler
}

static void usage(const char* program) {
    std::fprintf(stderr,
                 "usage: %s <grid_rows> <grid_cols> <sim_time> <temp_file> <power_file> <output_file>\n",
                 program);
}

int run(int argc, char** argv) {
    try {
        if (argc != 7) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }

        int grid_rows = std::atoi(argv[1]);
        int grid_cols = std::atoi(argv[2]);
        int sim_time = std::atoi(argv[3]);
        if (grid_rows <= 1 || grid_cols <= 1 || sim_time <= 0) {
            usage(argv[0]);
            return EXIT_FAILURE;
        }

        std::string tfile = argv[4];
        std::string pfile = argv[5];
        std::string ofile = argv[6];
        size_t size = static_cast<size_t>(grid_rows) * grid_cols;

        std::vector<double> temp = read_input(grid_rows, grid_cols, tfile);
        std::vector<double> power = read_input(grid_rows, grid_cols, pfile);
        
        std::vector<field_t> temp_dev = to_field_vector(temp);
        std::vector<field_t> power_dev = to_field_vector(power);
        std::vector<field_t> result_dev(size);
        std::vector<double> result(size);

        field_t* d_temp = nullptr;
        field_t* d_power = nullptr;
        field_t* d_result = nullptr;
        size_t device_bytes = 0;
        cuda_malloc_tracked(&d_temp, size, device_bytes);
        cuda_malloc_tracked(&d_power, size, device_bytes);
        cuda_malloc_tracked(&d_result, size, device_bytes);

        PROFILE_START("H2D_temp", mp_profiling::Category::MEMORY_TRANSFER);
        MP_CUDA_CHECK(cudaMemcpy(d_temp, temp_dev.data(), size * sizeof(field_t), cudaMemcpyHostToDevice));
        PROFILE_STOP(size * sizeof(field_t), 0);
        
        PROFILE_START("H2D_power", mp_profiling::Category::MEMORY_TRANSFER);
        MP_CUDA_CHECK(cudaMemcpy(d_power, power_dev.data(), size * sizeof(field_t), cudaMemcpyHostToDevice));
        PROFILE_STOP(size * sizeof(field_t), 0);

        compute_tran_temp(d_result, sim_time, d_temp, d_power, grid_rows, grid_cols);

        field_t* final_ptr = (sim_time & 1) ? d_result : d_temp;
        
        PROFILE_START("D2H_result", mp_profiling::Category::MEMORY_TRANSFER);
        MP_CUDA_CHECK(cudaMemcpy(result_dev.data(), final_ptr, size * sizeof(field_t), cudaMemcpyDeviceToHost));
        PROFILE_STOP(size * sizeof(field_t), 0);
        
        result = to_double_vector(result_dev);
        write_output(result, grid_rows, grid_cols, ofile);

        std::cout << HS_RUN_LABEL << "\n";
        std::cout << "rows=" << grid_rows << " cols=" << grid_cols
                  << " iterations=" << sim_time << "\n";
        std::cout << "device_allocation_bytes=" << device_bytes << "\n";
        std::cout << std::fixed << std::setprecision(6)
                  << "device_allocation_mib=" << static_cast<double>(device_bytes) / (1024.0 * 1024.0) << "\n";
        std::cout << "storage_bytes_temp=" << sizeof(field_t)
                  << " power=" << sizeof(field_t)
                  << " result=" << sizeof(field_t)
                  << " delta=" << sizeof(delta_t) << "\n";
        std::cout << std::setprecision(17) << "sample_result[0]=" << result[0] << "\n";

        PROFILE_SUMMARY();
        PROFILE_EXPORT("hotspot_profile.csv");

        cudaFree(d_temp);
        cudaFree(d_power);
        cudaFree(d_result);
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}

}  // namespace hotspot_h100_profiled

int main(int argc, char** argv) {
    return hotspot_h100_profiled::run(argc, argv);
}
