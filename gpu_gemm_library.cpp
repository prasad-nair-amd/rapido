// Library-achieved GEMM throughput for AMD GPUs, via rocBLAS.
// Compile: hipcc -O3 -o gpu_gemm_library gpu_gemm_library.cpp -lrocblas
//
// Why this exists alongside gpu_kernel_benchmarks.cpp: that file's GEMM is a naive
// three-loop kernel that does not touch the matrix cores at all. On MI300X it reports
// roughly 5 TFLOPS against a 163 TFLOPS FP32 peak -- a 28x understatement. It is a
// useful uniformity probe (every GPU should be equally slow at it) but it cannot
// support any statement about how well the hardware performs, because it measures the
// kernel, not the GPU. rocBLAS is the tuned library a real workload would call, so the
// number it produces is the one an acceptance test should be judging against peak.
//
// FP8 is deliberately absent: rocblas_gemm_ex does not expose it, and the gfx942 path
// requires hipBLASLt with AMD-specific FNUZ types. Claiming an FP8 figure here would
// mean measuring something other than what the label says.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>

#define HIP_CHECK(cmd)                                                      \
    do {                                                                      \
        hipError_t error = (cmd);                                             \
        if (error != hipSuccess) {                                            \
            std::cerr << "HIP error: " << hipGetErrorString(error)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define ROCBLAS_CHECK(cmd)                                                    \
    do {                                                                      \
        rocblas_status status = (cmd);                                        \
        if (status != rocblas_status_success) {                               \
            std::cerr << "rocBLAS error " << status                           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace {

// Square problem sizes. 8192 is large enough that the matrix cores reach steady
// state; 4096 is kept because a partitioned GPU (CPX) has proportionally less
// memory and the larger case may not fit.
const int kSizes[] = {4096, 8192};

// Enough iterations to average out clock ramp without making a full run tedious.
const int kWarmupIterations = 3;
const int kTimedIterations = 10;

struct Measurement {
    std::string precision;
    int size = 0;
    double tflops = 0.0;
    double milliseconds = 0.0;
    bool ok = false;
};

// GEMM does 2*N^3 floating-point operations for a square N.
double tflops_for(int n, double seconds) {
    if (seconds <= 0.0) {
        return 0.0;
    }
    const double operations = 2.0 * static_cast<double>(n) * n * n;
    return operations / seconds / 1e12;
}

// Times a callable over kTimedIterations, returning average seconds per call.
template <typename Fn>
bool time_gemm(Fn&& launch, double* seconds_out) {
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < kWarmupIterations; ++i) {
        if (!launch()) {
            return false;
        }
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < kTimedIterations; ++i) {
        if (!launch()) {
            return false;
        }
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    *seconds_out = (elapsed_ms / 1000.0) / kTimedIterations;
    return true;
}

bool run_fp32(rocblas_handle handle, int n, Measurement* out) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    if (hipMalloc(&a, bytes) != hipSuccess ||
        hipMalloc(&b, bytes) != hipSuccess ||
        hipMalloc(&c, bytes) != hipSuccess) {
        static_cast<void>(hipFree(a));
        static_cast<void>(hipFree(b));
        static_cast<void>(hipFree(c));
        return false;  // Not enough memory for this size; the caller skips it.
    }
    HIP_CHECK(hipMemset(a, 0, bytes));
    HIP_CHECK(hipMemset(b, 0, bytes));
    HIP_CHECK(hipMemset(c, 0, bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    double seconds = 0.0;
    const bool timed = time_gemm(
        [&]() {
            return rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                 n, n, n, &alpha, a, n, b, n, &beta, c, n) ==
                   rocblas_status_success;
        },
        &seconds);

    static_cast<void>(hipFree(a));
    static_cast<void>(hipFree(b));
    static_cast<void>(hipFree(c));
    if (!timed) {
        return false;
    }

    out->precision = "FP32";
    out->size = n;
    out->milliseconds = seconds * 1000.0;
    out->tflops = tflops_for(n, seconds);
    out->ok = true;
    return true;
}

// FP16 and BF16 share a code path: both accumulate in FP32, which is what a real
// mixed-precision workload does and what the matrix cores are rated for.
bool run_reduced(rocblas_handle handle, int n, rocblas_datatype input_type,
                 const char* label, Measurement* out) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t input_bytes = elements * 2;  // 2 bytes per FP16/BF16 element.
    const size_t output_bytes = elements * sizeof(float);

    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    if (hipMalloc(&a, input_bytes) != hipSuccess ||
        hipMalloc(&b, input_bytes) != hipSuccess ||
        hipMalloc(&c, output_bytes) != hipSuccess) {
        static_cast<void>(hipFree(a));
        static_cast<void>(hipFree(b));
        static_cast<void>(hipFree(c));
        return false;
    }
    HIP_CHECK(hipMemset(a, 0, input_bytes));
    HIP_CHECK(hipMemset(b, 0, input_bytes));
    HIP_CHECK(hipMemset(c, 0, output_bytes));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    double seconds = 0.0;
    const bool timed = time_gemm(
        [&]() {
            return rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none,
                                   n, n, n, &alpha,
                                   a, input_type, n,
                                   b, input_type, n,
                                   &beta,
                                   c, rocblas_datatype_f32_r, n,
                                   c, rocblas_datatype_f32_r, n,
                                   rocblas_datatype_f32_r,
                                   rocblas_gemm_algo_standard, 0, 0) == rocblas_status_success;
        },
        &seconds);

    static_cast<void>(hipFree(a));
    static_cast<void>(hipFree(b));
    static_cast<void>(hipFree(c));
    if (!timed) {
        return false;
    }

    out->precision = label;
    out->size = n;
    out->milliseconds = seconds * 1000.0;
    out->tflops = tflops_for(n, seconds);
    out->ok = true;
    return true;
}

void emit_json(const std::vector<std::string>& devices) {
    std::cout << "{\n  \"results\": [\n";
    for (size_t i = 0; i < devices.size(); ++i) {
        std::cout << devices[i];
        if (i + 1 < devices.size()) {
            std::cout << ",";
        }
        std::cout << "\n";
    }
    std::cout << "  ]\n}" << std::endl;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::cout << "{\"error\": \"No HIP devices found\"}" << std::endl;
        return 0;
    }

    std::vector<std::string> device_blocks;

    for (int device = 0; device < device_count; ++device) {
        if (hipSetDevice(device) != hipSuccess) {
            continue;
        }
        hipDeviceProp_t props;
        if (hipGetDeviceProperties(&props, device) != hipSuccess) {
            continue;
        }

        rocblas_handle handle = nullptr;
        if (rocblas_create_handle(&handle) != rocblas_status_success) {
            continue;
        }

        std::vector<Measurement> measurements;
        for (int size_index = 0; size_index < 2; ++size_index) {
            const int n = kSizes[size_index];
            Measurement fp32;
            if (run_fp32(handle, n, &fp32)) {
                measurements.push_back(fp32);
            }
            Measurement fp16;
            if (run_reduced(handle, n, rocblas_datatype_f16_r, "FP16", &fp16)) {
                measurements.push_back(fp16);
            }
            Measurement bf16;
            if (run_reduced(handle, n, rocblas_datatype_bf16_r, "BF16", &bf16)) {
                measurements.push_back(bf16);
            }
        }

        rocblas_destroy_handle(handle);

        if (measurements.empty()) {
            continue;
        }

        std::ostringstream block;
        block << std::fixed << std::setprecision(2);
        block << "    {\n";
        block << "      \"gpu_id\": " << device << ",\n";
        block << "      \"gpu_name\": \"" << props.gcnArchName << "\",\n";
        block << "      \"library\": \"rocBLAS\",\n";
        block << "      \"measurements\": [\n";
        for (size_t i = 0; i < measurements.size(); ++i) {
            const Measurement& m = measurements[i];
            block << "        {\"precision\": \"" << m.precision << "\", "
                  << "\"size\": " << m.size << ", "
                  << "\"tflops\": " << m.tflops << ", "
                  << "\"time_ms\": " << m.milliseconds << "}";
            if (i + 1 < measurements.size()) {
                block << ",";
            }
            block << "\n";
        }
        block << "      ]\n";
        block << "    }";
        device_blocks.push_back(block.str());
    }

    if (device_blocks.empty()) {
        std::cout << "{\"error\": \"rocBLAS GEMM produced no measurements\"}" << std::endl;
        return 0;
    }

    emit_json(device_blocks);
    return 0;
}
