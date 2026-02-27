// GPU-CPU (Host) Bandwidth Benchmark
// Measures host-to-device and device-to-host transfer bandwidth
// Compile: hipcc -o gpu_host_bandwidth gpu_host_bandwidth.cpp

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstring>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

// Test sizes: 256 MB for bandwidth measurement
const size_t TEST_SIZE = 256 * 1024 * 1024;
const int NUM_ITERATIONS = 10;

struct HostBandwidthResult {
    int gpu;
    double h2dBandwidthGBps;  // Host to Device
    double d2hBandwidthGBps;  // Device to Host
    double h2dPinnedGBps;     // Host to Device (pinned memory)
    double d2hPinnedGBps;     // Device to Host (pinned memory)
};

double measureH2D(int gpu, size_t size, bool usePinned) {
    void *hostBuffer = nullptr;
    void *deviceBuffer = nullptr;

    HIP_CHECK(hipSetDevice(gpu));

    // Allocate host memory
    if (usePinned) {
        HIP_CHECK(hipHostMalloc(&hostBuffer, size));
    } else {
        hostBuffer = malloc(size);
        if (!hostBuffer) {
            std::cerr << "Failed to allocate host memory" << std::endl;
            return 0.0;
        }
    }

    // Initialize host buffer
    memset(hostBuffer, 0xAB, size);

    // Allocate device memory
    HIP_CHECK(hipMalloc(&deviceBuffer, size));

    // Warm up
    for (int i = 0; i < 3; i++) {
        HIP_CHECK(hipMemcpy(deviceBuffer, hostBuffer, size, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measure bandwidth
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIP_CHECK(hipMemcpy(deviceBuffer, hostBuffer, size, hipMemcpyHostToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate bandwidth in GB/s
    double totalBytes = static_cast<double>(size) * NUM_ITERATIONS;
    double bandwidthGBps = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / elapsed.count();

    // Cleanup
    HIP_CHECK(hipFree(deviceBuffer));
    if (usePinned) {
        HIP_CHECK(hipHostFree(hostBuffer));
    } else {
        free(hostBuffer);
    }

    return bandwidthGBps;
}

double measureD2H(int gpu, size_t size, bool usePinned) {
    void *hostBuffer = nullptr;
    void *deviceBuffer = nullptr;

    HIP_CHECK(hipSetDevice(gpu));

    // Allocate host memory
    if (usePinned) {
        HIP_CHECK(hipHostMalloc(&hostBuffer, size));
    } else {
        hostBuffer = malloc(size);
        if (!hostBuffer) {
            std::cerr << "Failed to allocate host memory" << std::endl;
            return 0.0;
        }
    }

    // Allocate and initialize device memory
    HIP_CHECK(hipMalloc(&deviceBuffer, size));
    HIP_CHECK(hipMemset(deviceBuffer, 0xCD, size));

    // Warm up
    for (int i = 0; i < 3; i++) {
        HIP_CHECK(hipMemcpy(hostBuffer, deviceBuffer, size, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measure bandwidth
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIP_CHECK(hipMemcpy(hostBuffer, deviceBuffer, size, hipMemcpyDeviceToHost));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate bandwidth in GB/s
    double totalBytes = static_cast<double>(size) * NUM_ITERATIONS;
    double bandwidthGBps = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / elapsed.count();

    // Cleanup
    HIP_CHECK(hipFree(deviceBuffer));
    if (usePinned) {
        HIP_CHECK(hipHostFree(hostBuffer));
    } else {
        free(hostBuffer);
    }

    return bandwidthGBps;
}

int main() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "GPU_HOST_BANDWIDTH_RESULT: {\"error\": \"No GPUs found\"}" << std::endl;
        return 0;
    }

    // Get GPU names
    std::vector<std::string> gpuNames(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        gpuNames[i] = prop.gcnArchName;
    }

    std::cout << "GPU_HOST_BANDWIDTH_RESULT: {" << std::endl;
    std::cout << "  \"gpu_count\": " << deviceCount << "," << std::endl;
    std::cout << "  \"test_size_mb\": " << (TEST_SIZE / (1024 * 1024)) << "," << std::endl;
    std::cout << "  \"iterations\": " << NUM_ITERATIONS << "," << std::endl;
    std::cout << "  \"results\": [" << std::endl;

    // Test each GPU
    for (int gpu = 0; gpu < deviceCount; gpu++) {
        double h2dPageable = measureH2D(gpu, TEST_SIZE, false);
        double d2hPageable = measureD2H(gpu, TEST_SIZE, false);
        double h2dPinned = measureH2D(gpu, TEST_SIZE, true);
        double d2hPinned = measureD2H(gpu, TEST_SIZE, true);

        if (gpu > 0) {
            std::cout << "," << std::endl;
        }

        std::cout << "    {" << std::endl;
        std::cout << "      \"gpu\": " << gpu << "," << std::endl;
        std::cout << "      \"gpu_name\": \"" << gpuNames[gpu] << "\"," << std::endl;
        std::cout << "      \"h2d_pageable_gbps\": " << std::fixed << std::setprecision(2) << h2dPageable << "," << std::endl;
        std::cout << "      \"d2h_pageable_gbps\": " << std::fixed << std::setprecision(2) << d2hPageable << "," << std::endl;
        std::cout << "      \"h2d_pinned_gbps\": " << std::fixed << std::setprecision(2) << h2dPinned << "," << std::endl;
        std::cout << "      \"d2h_pinned_gbps\": " << std::fixed << std::setprecision(2) << d2hPinned << std::endl;
        std::cout << "    }";
    }

    std::cout << std::endl << "  ]" << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}
