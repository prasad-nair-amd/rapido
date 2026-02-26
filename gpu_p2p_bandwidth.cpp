// GPU Peer-to-Peer Bandwidth Benchmark
// Measures bandwidth between AMD GPUs using HIP
// Compile: hipcc -o gpu_p2p_bandwidth gpu_p2p_bandwidth.cpp

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

// Bandwidth test size (256 MB)
const size_t TEST_SIZE = 256 * 1024 * 1024;
const int NUM_ITERATIONS = 10;

struct BandwidthResult {
    int srcGpu;
    int dstGpu;
    double bandwidthGBps;
    bool p2pEnabled;
};

double measureBandwidth(int srcGpu, int dstGpu, size_t size) {
    void *srcBuffer = nullptr;
    void *dstBuffer = nullptr;

    // Set source GPU
    HIP_CHECK(hipSetDevice(srcGpu));
    HIP_CHECK(hipMalloc(&srcBuffer, size));

    // Initialize source buffer
    HIP_CHECK(hipMemset(srcBuffer, 0xAB, size));

    // Set destination GPU
    HIP_CHECK(hipSetDevice(dstGpu));
    HIP_CHECK(hipMalloc(&dstBuffer, size));

    // Enable peer access if possible
    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dstGpu, srcGpu));

    if (canAccess) {
        hipError_t err = hipDeviceEnablePeerAccess(srcGpu, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
            HIP_CHECK(err);
        }
    }

    // Warm up
    for (int i = 0; i < 3; i++) {
        HIP_CHECK(hipMemcpy(dstBuffer, srcBuffer, size, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measure bandwidth
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_ITERATIONS; i++) {
        HIP_CHECK(hipMemcpy(dstBuffer, srcBuffer, size, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate bandwidth in GB/s
    double totalBytes = static_cast<double>(size) * NUM_ITERATIONS;
    double bandwidthGBps = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / elapsed.count();

    // Cleanup
    HIP_CHECK(hipSetDevice(srcGpu));
    HIP_CHECK(hipFree(srcBuffer));
    HIP_CHECK(hipSetDevice(dstGpu));
    HIP_CHECK(hipFree(dstBuffer));

    return bandwidthGBps;
}

int main() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        std::cout << "GPU_P2P_BANDWIDTH_RESULT: {\"error\": \"Need at least 2 GPUs for P2P testing\"}" << std::endl;
        return 0;
    }

    std::vector<BandwidthResult> results;

    // Get GPU names
    std::vector<std::string> gpuNames(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        gpuNames[i] = prop.gcnArchName;
    }

    std::cout << "GPU_P2P_BANDWIDTH_RESULT: {" << std::endl;
    std::cout << "  \"gpu_count\": " << deviceCount << "," << std::endl;
    std::cout << "  \"test_size_mb\": " << (TEST_SIZE / (1024 * 1024)) << "," << std::endl;
    std::cout << "  \"iterations\": " << NUM_ITERATIONS << "," << std::endl;
    std::cout << "  \"results\": [" << std::endl;

    bool firstResult = true;

    // Test all GPU pairs
    for (int srcGpu = 0; srcGpu < deviceCount; srcGpu++) {
        for (int dstGpu = 0; dstGpu < deviceCount; dstGpu++) {
            if (srcGpu == dstGpu) continue;

            // Check if P2P is supported
            int canAccess = 0;
            HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dstGpu, srcGpu));

            double bandwidth = 0.0;
            if (canAccess) {
                bandwidth = measureBandwidth(srcGpu, dstGpu, TEST_SIZE);
            }

            if (!firstResult) {
                std::cout << "," << std::endl;
            }
            firstResult = false;

            std::cout << "    {" << std::endl;
            std::cout << "      \"src_gpu\": " << srcGpu << "," << std::endl;
            std::cout << "      \"dst_gpu\": " << dstGpu << "," << std::endl;
            std::cout << "      \"src_name\": \"" << gpuNames[srcGpu] << "\"," << std::endl;
            std::cout << "      \"dst_name\": \"" << gpuNames[dstGpu] << "\"," << std::endl;
            std::cout << "      \"p2p_enabled\": " << (canAccess ? "true" : "false") << "," << std::endl;
            std::cout << "      \"bandwidth_gbps\": " << std::fixed << std::setprecision(2) << bandwidth << std::endl;
            std::cout << "    }";
        }
    }

    std::cout << std::endl << "  ]" << std::endl;
    std::cout << "}" << std::endl;

    return 0;
}
