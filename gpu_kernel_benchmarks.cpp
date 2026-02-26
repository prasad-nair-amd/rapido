// GPU Kernel Benchmarks for AMD GPUs
// Measures real-world kernel performance: GEMM, Memory Bandwidth, Vector Operations
// Compile: hipcc -O3 -o gpu_kernel_benchmarks gpu_kernel_benchmarks.cpp

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cmath>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

// GEMM Kernel (Matrix Multiplication): C = A * B
__global__ void gemm_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Memory Bandwidth Kernel (Copy)
__global__ void memory_copy_kernel(const float* src, float* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Vector Addition Kernel
__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// FMA (Fused Multiply-Add) intensive kernel
__global__ void fma_kernel(const float* a, const float* b, const float* c, float* d, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = a[idx];
        for (int i = 0; i < 100; i++) {
            val = fmaf(val, b[idx], c[idx]);
        }
        d[idx] = val;
    }
}

// Double precision GEMM for FP64 testing
__global__ void gemm_fp64_kernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        double sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 1D Convolution kernel (simplified)
__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, 
                               int signal_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_kernel = kernel_size / 2;
    
    if (idx < signal_size) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            int input_idx = idx - half_kernel + k;
            if (input_idx >= 0 && input_idx < signal_size) {
                sum += input[input_idx] * kernel[k];
            }
        }
        output[idx] = sum;
    }
}

// Benchmark GEMM (Matrix Multiplication)
double benchmark_gemm(int gpu_id, int matrix_size, bool use_fp64 = false) {
    hipSetDevice(gpu_id);
    
    size_t total_elements = matrix_size * matrix_size;
    
    if (use_fp64) {
        size_t bytes = total_elements * sizeof(double);
        double *d_A, *d_B, *d_C;
        
        HIP_CHECK(hipMalloc(&d_A, bytes));
        HIP_CHECK(hipMalloc(&d_B, bytes));
        HIP_CHECK(hipMalloc(&d_C, bytes));
        
        std::vector<double> h_A(total_elements, 1.0);
        std::vector<double> h_B(total_elements, 2.0);
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);
        
        for (int i = 0; i < 3; i++) {
            hipLaunchKernelGGL(gemm_fp64_kernel, grid, block, 0, 0, d_A, d_B, d_C, matrix_size);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(gemm_fp64_kernel, grid, block, 0, 0, d_A, d_B, d_C, matrix_size);
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        double ops = 2.0 * matrix_size * matrix_size * matrix_size * iterations;
        double gflops = (ops / elapsed.count()) / 1e9;
        
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
        
        return gflops;
    } else {
        size_t bytes = total_elements * sizeof(float);
        float *d_A, *d_B, *d_C;
        
        HIP_CHECK(hipMalloc(&d_A, bytes));
        HIP_CHECK(hipMalloc(&d_B, bytes));
        HIP_CHECK(hipMalloc(&d_C, bytes));
        
        std::vector<float> h_A(total_elements, 1.0f);
        std::vector<float> h_B(total_elements, 2.0f);
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));
        
        dim3 block(16, 16);
        dim3 grid((matrix_size + block.x - 1) / block.x, (matrix_size + block.y - 1) / block.y);
        
        for (int i = 0; i < 3; i++) {
            hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0, d_A, d_B, d_C, matrix_size);
        }
        HIP_CHECK(hipDeviceSynchronize());
        
        const int iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            hipLaunchKernelGGL(gemm_kernel, grid, block, 0, 0, d_A, d_B, d_C, matrix_size);
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        std::chrono::duration<double> elapsed = end - start;
        double ops = 2.0 * matrix_size * matrix_size * matrix_size * iterations;
        double gflops = (ops / elapsed.count()) / 1e9;
        
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
        
        return gflops;
    }
}

// Benchmark Memory Bandwidth
double benchmark_memory_bandwidth(int gpu_id, size_t size_mb) {
    hipSetDevice(gpu_id);
    
    size_t num_elements = (size_mb * 1024 * 1024) / sizeof(float);
    size_t bytes = num_elements * sizeof(float);
    
    float *d_src, *d_dst;
    HIP_CHECK(hipMalloc(&d_src, bytes));
    HIP_CHECK(hipMalloc(&d_dst, bytes));
    
    std::vector<float> h_src(num_elements, 1.0f);
    HIP_CHECK(hipMemcpy(d_src, h_src.data(), bytes, hipMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;
    
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(memory_copy_kernel, gridSize, blockSize, 0, 0, d_src, d_dst, num_elements);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    const int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(memory_copy_kernel, gridSize, blockSize, 0, 0, d_src, d_dst, num_elements);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double total_bytes = bytes * iterations * 2;
    double bandwidth_gbps = (total_bytes / elapsed.count()) / 1e9;
    
    HIP_CHECK(hipFree(d_src));
    HIP_CHECK(hipFree(d_dst));
    
    return bandwidth_gbps;
}

// Benchmark Vector Operations
double benchmark_vector_add(int gpu_id, size_t size_mb) {
    hipSetDevice(gpu_id);
    
    size_t num_elements = (size_mb * 1024 * 1024) / sizeof(float);
    size_t bytes = num_elements * sizeof(float);
    
    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    
    std::vector<float> h_a(num_elements, 1.0f);
    std::vector<float> h_b(num_elements, 2.0f);
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;
    
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(vector_add_kernel, gridSize, blockSize, 0, 0, d_a, d_b, d_c, num_elements);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    const int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(vector_add_kernel, gridSize, blockSize, 0, 0, d_a, d_b, d_c, num_elements);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double ops = num_elements * iterations;
    double gflops = (ops / elapsed.count()) / 1e9;
    
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    
    return gflops;
}

// Benchmark FMA throughput
double benchmark_fma(int gpu_id, size_t size_mb) {
    hipSetDevice(gpu_id);
    
    size_t num_elements = (size_mb * 1024 * 1024) / sizeof(float);
    size_t bytes = num_elements * sizeof(float);
    
    float *d_a, *d_b, *d_c, *d_d;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMalloc(&d_d, bytes));
    
    std::vector<float> h_a(num_elements, 1.1f);
    std::vector<float> h_b(num_elements, 2.2f);
    std::vector<float> h_c(num_elements, 3.3f);
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_c, h_c.data(), bytes, hipMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (num_elements + blockSize - 1) / blockSize;
    
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(fma_kernel, gridSize, blockSize, 0, 0, d_a, d_b, d_c, d_d, num_elements);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    const int iterations = 10;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(fma_kernel, gridSize, blockSize, 0, 0, d_a, d_b, d_c, d_d, num_elements);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double ops = num_elements * 100 * 2 * iterations;
    double tflops = (ops / elapsed.count()) / 1e12;
    
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    HIP_CHECK(hipFree(d_d));
    
    return tflops;
}

// Benchmark Convolution
double benchmark_conv1d(int gpu_id, size_t signal_size, int kernel_size) {
    hipSetDevice(gpu_id);
    
    size_t input_bytes = signal_size * sizeof(float);
    size_t kernel_bytes = kernel_size * sizeof(float);
    
    float *d_input, *d_kernel, *d_output;
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_kernel, kernel_bytes));
    HIP_CHECK(hipMalloc(&d_output, input_bytes));
    
    std::vector<float> h_input(signal_size, 1.0f);
    std::vector<float> h_kernel(kernel_size, 0.1f);
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kernel, h_kernel.data(), kernel_bytes, hipMemcpyHostToDevice));
    
    int blockSize = 256;
    int gridSize = (signal_size + blockSize - 1) / blockSize;
    
    for (int i = 0; i < 3; i++) {
        hipLaunchKernelGGL(conv1d_kernel, gridSize, blockSize, 0, 0, 
                          d_input, d_kernel, d_output, signal_size, kernel_size);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    const int iterations = 20;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        hipLaunchKernelGGL(conv1d_kernel, gridSize, blockSize, 0, 0, 
                          d_input, d_kernel, d_output, signal_size, kernel_size);
    }
    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    double ops = signal_size * kernel_size * iterations;
    double gflops = (ops / elapsed.count()) / 1e9;
    
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_kernel));
    HIP_CHECK(hipFree(d_output));
    
    return gflops;
}

int main() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    
    if (deviceCount == 0) {
        std::cout << "GPU_KERNEL_BENCHMARK_RESULT: {\"error\": \"No GPUs found\"}" << std::endl;
        return 0;
    }
    
    std::cout << "GPU_KERNEL_BENCHMARK_RESULT: {" << std::endl;
    std::cout << "  \"gpu_count\": " << deviceCount << "," << std::endl;
    std::cout << "  \"results\": [" << std::endl;
    
    bool firstGpu = true;
    
    for (int gpu_id = 0; gpu_id < deviceCount; gpu_id++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, gpu_id));
        
        std::string gpu_name = prop.gcnArchName;
        
        if (!firstGpu) {
            std::cout << "," << std::endl;
        }
        firstGpu = false;
        
        std::cout << "    {" << std::endl;
        std::cout << "      \"gpu_id\": " << gpu_id << "," << std::endl;
        std::cout << "      \"gpu_name\": \"" << gpu_name << "\"," << std::endl;
        
        std::cout << "      \"memory_bandwidth_test\": {" << std::endl;
        double mem_bw = benchmark_memory_bandwidth(gpu_id, 512);
        std::cout << "        \"bandwidth_gbps\": " << std::fixed << std::setprecision(2) << mem_bw << "," << std::endl;
        std::cout << "        \"test_size_mb\": 512" << std::endl;
        std::cout << "      }," << std::endl;
        
        std::cout << "      \"gemm_fp32_test\": {" << std::endl;
        double gemm_fp32 = benchmark_gemm(gpu_id, 2048, false);
        std::cout << "        \"gflops\": " << std::fixed << std::setprecision(2) << gemm_fp32 << "," << std::endl;
        std::cout << "        \"matrix_size\": 2048" << std::endl;
        std::cout << "      }," << std::endl;
        
        std::cout << "      \"gemm_fp64_test\": {" << std::endl;
        double gemm_fp64 = benchmark_gemm(gpu_id, 1024, true);
        std::cout << "        \"gflops\": " << std::fixed << std::setprecision(2) << gemm_fp64 << "," << std::endl;
        std::cout << "        \"matrix_size\": 1024" << std::endl;
        std::cout << "      }," << std::endl;
        
        std::cout << "      \"vector_add_test\": {" << std::endl;
        double vec_add = benchmark_vector_add(gpu_id, 256);
        std::cout << "        \"gflops\": " << std::fixed << std::setprecision(2) << vec_add << "," << std::endl;
        std::cout << "        \"test_size_mb\": 256" << std::endl;
        std::cout << "      }," << std::endl;
        
        std::cout << "      \"fma_throughput_test\": {" << std::endl;
        double fma_tflops = benchmark_fma(gpu_id, 128);
        std::cout << "        \"tflops\": " << std::fixed << std::setprecision(2) << fma_tflops << "," << std::endl;
        std::cout << "        \"test_size_mb\": 128" << std::endl;
        std::cout << "      }," << std::endl;
        
        std::cout << "      \"convolution_test\": {" << std::endl;
        double conv_gflops = benchmark_conv1d(gpu_id, 16*1024*1024, 32);
        std::cout << "        \"gflops\": " << std::fixed << std::setprecision(2) << conv_gflops << "," << std::endl;
        std::cout << "        \"signal_size\": " << (16*1024*1024) << "," << std::endl;
        std::cout << "        \"kernel_size\": 32" << std::endl;
        std::cout << "      }" << std::endl;
        
        std::cout << "    }";
    }
    
    std::cout << std::endl << "  ]" << std::endl;
    std::cout << "}" << std::endl;
    
    return 0;
}
