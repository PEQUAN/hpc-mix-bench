#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

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

namespace mp_profiling {

/// Profiling categories
enum class Category {
    MEMORY_TRANSFER,     // Host ↔ Device transfers
    TYPE_CONVERSION,     // Low-precision ↔ double conversions
    COMPUTATION,         // Actual arithmetic operations
    TOTAL               // Total kernel time
};

/// Single timing record
struct TimingRecord {
    Category category;
    std::string label;
    double time_ms;
    size_t bytes_transferred = 0;  // For memory ops
    size_t num_conversions = 0;    // For conversion ops
};

/// Profiler class
class Profiler {
public:
    Profiler() = default;
    ~Profiler() {
        for (auto& ev_pair : events_) {
            cudaEventDestroy(ev_pair.first);
            cudaEventDestroy(ev_pair.second);
        }
    }

    /// Start timing a region
    void start(const std::string& label, Category cat = Category::COMPUTATION) {
        cudaEvent_t start, stop;
        MP_CUDA_CHECK(cudaEventCreate(&start));
        MP_CUDA_CHECK(cudaEventCreate(&stop));
        MP_CUDA_CHECK(cudaEventRecord(start));
        
        active_timers_.push_back({label, cat, start, stop});
    }

    /// Stop timing the most recent region
    void stop(size_t bytes = 0, size_t conversions = 0) {
        if (active_timers_.empty()) {
            std::fprintf(stderr, "Profiler::stop() called without matching start()\n");
            return;
        }
        
        auto timer = active_timers_.back();
        active_timers_.pop_back();
        
        MP_CUDA_CHECK(cudaEventRecord(timer.stop));
        MP_CUDA_CHECK(cudaEventSynchronize(timer.stop));
        
        float ms = 0.0f;
        MP_CUDA_CHECK(cudaEventElapsedTime(&ms, timer.start, timer.stop));
        
        records_.push_back({
            timer.category,
            timer.label,
            static_cast<double>(ms),
            bytes,
            conversions
        });
        
        events_.push_back({timer.start, timer.stop});
    }

    /// Print profiling summary
    void print_summary(bool detailed = false) const {
        std::printf("\n=== Profiling Summary ===\n");
        
        double total_time = 0.0;
        double mem_time = 0.0;
        double conv_time = 0.0;
        double comp_time = 0.0;
        
        size_t total_bytes = 0;
        size_t total_conversions = 0;
        
        // Aggregate by category
        for (const auto& rec : records_) {
            total_time += rec.time_ms;
            switch (rec.category) {
                case Category::MEMORY_TRANSFER:
                    mem_time += rec.time_ms;
                    total_bytes += rec.bytes_transferred;
                    break;
                case Category::TYPE_CONVERSION:
                    conv_time += rec.time_ms;
                    total_conversions += rec.num_conversions;
                    break;
                case Category::COMPUTATION:
                    comp_time += rec.time_ms;
                    break;
                default:
                    break;
            }
        }
        
        std::printf("Category Breakdown:\n");
        std::printf("  Memory Transfer:   %8.4f ms (%5.2f%%)\n", 
                   mem_time, 100.0 * mem_time / total_time);
        if (total_bytes > 0) {
            std::printf("    Total Bytes:     %zu (%.2f MiB)\n",
                       total_bytes, total_bytes / (1024.0 * 1024.0));
            std::printf("    Bandwidth:       %.2f GiB/s\n",
                       (total_bytes / 1e9) / (mem_time / 1000.0));
        }
        
        std::printf("  Type Conversion:   %8.4f ms (%5.2f%%)\n", 
                   conv_time, 100.0 * conv_time / total_time);
        if (total_conversions > 0) {
            std::printf("    Total Conv:      %zu\n", total_conversions);
            std::printf("    Throughput:      %.2f M/s\n",
                       (total_conversions / 1e6) / (conv_time / 1000.0));
        }
        
        std::printf("  Computation:       %8.4f ms (%5.2f%%)\n", 
                   comp_time, 100.0 * comp_time / total_time);
        std::printf("  Total:             %8.4f ms\n", total_time);
        
        if (detailed) {
            std::printf("\nDetailed Breakdown:\n");
            for (const auto& rec : records_) {
                std::printf("  %-30s %8.4f ms", rec.label.c_str(), rec.time_ms);
                if (rec.bytes_transferred > 0) {
                    std::printf("  [%zu bytes]", rec.bytes_transferred);
                }
                if (rec.num_conversions > 0) {
                    std::printf("  [%zu conv]", rec.num_conversions);
                }
                std::printf("\n");
            }
        }
        std::printf("========================\n\n");
    }

    /// Export results to CSV
    void export_csv(const std::string& filename) const {
        FILE* fp = std::fopen(filename.c_str(), "w");
        if (!fp) {
            std::fprintf(stderr, "Failed to open %s for writing\n", filename.c_str());
            return;
        }
        
        std::fprintf(fp, "category,label,time_ms,bytes,conversions\n");
        for (const auto& rec : records_) {
            const char* cat_name = "unknown";
            switch (rec.category) {
                case Category::MEMORY_TRANSFER: cat_name = "memory"; break;
                case Category::TYPE_CONVERSION: cat_name = "conversion"; break;
                case Category::COMPUTATION: cat_name = "computation"; break;
                case Category::TOTAL: cat_name = "total"; break;
            }
            std::fprintf(fp, "%s,%s,%.6f,%zu,%zu\n",
                        cat_name, rec.label.c_str(), rec.time_ms,
                        rec.bytes_transferred, rec.num_conversions);
        }
        
        std::fclose(fp);
        std::printf("Profiling data exported to %s\n", filename.c_str());
    }

private:
    struct ActiveTimer {
        std::string label;
        Category category;
        cudaEvent_t start;
        cudaEvent_t stop;
    };
    
    std::vector<ActiveTimer> active_timers_;
    std::vector<TimingRecord> records_;
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> events_;
};

/// Global profiler instance
static Profiler g_profiler;

/// Convenience macros
#define PROFILE_START(label, cat) mp_profiling::g_profiler.start(label, cat)
#define PROFILE_STOP(bytes, conv) mp_profiling::g_profiler.stop(bytes, conv)
#define PROFILE_SUMMARY() mp_profiling::g_profiler.print_summary(true)
#define PROFILE_EXPORT(file) mp_profiling::g_profiler.export_csv(file)

}  // namespace mp_profiling
