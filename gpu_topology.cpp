// GPU Topology Analysis - XGMI/Infinity Fabric Bandwidth Matrix
// Analyzes AMD GPU interconnect topology (equivalent to NVIDIA NVLink analysis)
// Compile: hipcc -o gpu_topology gpu_topology.cpp

#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = (cmd); \
        if (error != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return -1; \
        } \
    } while(0)

// Read AMD SMI topology information
std::string getAmdSmiTopology() {
    std::string result;
    FILE* pipe = popen("amd-smi topology --json 2>/dev/null", "r");
    if (pipe) {
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        pclose(pipe);
    }
    return result;
}

// Read link type from sysfs (XGMI hops information)
int getXgmiHops(int gpu1, int gpu2) {
    std::stringstream path;
    path << "/sys/class/drm/card" << gpu1 << "/device/xgmi_hive_info/node_" << gpu2 << "_hops";
    
    std::ifstream file(path.str());
    if (file.is_open()) {
        int hops;
        file >> hops;
        file.close();
        return hops;
    }
    return -1;
}

// Get link type string based on P2P capability and XGMI hops
std::string getLinkType(int gpu1, int gpu2, int canAccess) {
    if (gpu1 == gpu2) {
        return "Self";
    }
    
    if (!canAccess) {
        return "No P2P";
    }
    
    // Try to get XGMI hops information
    int hops = getXgmiHops(gpu1, gpu2);
    
    if (hops == 1) {
        return "XGMI";  // Direct XGMI link
    } else if (hops == 2) {
        return "XGMI-2hop";  // Two XGMI hops
    } else if (hops > 2) {
        std::stringstream ss;
        ss << "XGMI-" << hops << "hop";
        return ss.str();
    }
    
    // If no XGMI info, likely PCIe
    return "PCIe";
}

// Measure actual bandwidth between GPUs (smaller test for topology)
double measureTopologyBandwidth(int srcGpu, int dstGpu, size_t size) {
    void *srcBuffer = nullptr;
    void *dstBuffer = nullptr;

    HIP_CHECK(hipSetDevice(srcGpu));
    HIP_CHECK(hipMalloc(&srcBuffer, size));
    HIP_CHECK(hipMemset(srcBuffer, 0xAB, size));

    HIP_CHECK(hipSetDevice(dstGpu));
    HIP_CHECK(hipMalloc(&dstBuffer, size));

    int canAccess = 0;
    HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dstGpu, srcGpu));

    if (canAccess) {
        hipError_t err = hipDeviceEnablePeerAccess(srcGpu, 0);
        if (err != hipSuccess && err != hipErrorPeerAccessAlreadyEnabled) {
            HIP_CHECK(err);
        }
    }

    // Warm up
    for (int i = 0; i < 2; i++) {
        HIP_CHECK(hipMemcpy(dstBuffer, srcBuffer, size, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measure bandwidth (5 iterations for topology)
    auto start = std::chrono::high_resolution_clock::now();
    const int iterations = 5;
    
    for (int i = 0; i < iterations; i++) {
        HIP_CHECK(hipMemcpy(dstBuffer, srcBuffer, size, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double totalBytes = static_cast<double>(size) * iterations;
    double bandwidthGBps = (totalBytes / (1024.0 * 1024.0 * 1024.0)) / elapsed.count();

    HIP_CHECK(hipSetDevice(srcGpu));
    HIP_CHECK(hipFree(srcBuffer));
    HIP_CHECK(hipSetDevice(dstGpu));
    HIP_CHECK(hipFree(dstBuffer));

    return bandwidthGBps;
}

int main() {
    int deviceCount = 0;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cout << "GPU_TOPOLOGY_RESULT: {\"error\": \"No GPUs found\"}" << std::endl;
        return 0;
    }

    // Test size for topology: 64 MB (smaller, faster)
    const size_t TOPOLOGY_TEST_SIZE = 64 * 1024 * 1024;

    // Get GPU information
    std::vector<std::string> gpuNames(deviceCount);
    std::vector<std::string> gpuPciIds(deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        gpuNames[i] = prop.gcnArchName;
        
        std::stringstream pciId;
        pciId << std::hex << std::setfill('0') 
              << std::setw(4) << prop.pciDomainID << ":"
              << std::setw(2) << prop.pciBusID << ":"
              << std::setw(2) << prop.pciDeviceID << "."
              << prop.pciDomainID;
        gpuPciIds[i] = pciId.str();
    }

    // Try to get AMD SMI topology information
    std::string amdSmiTopology = getAmdSmiTopology();

    std::cout << "GPU_TOPOLOGY_RESULT: {" << std::endl;
    std::cout << "  \"gpu_count\": " << deviceCount << "," << std::endl;
    std::cout << "  \"test_size_mb\": " << (TOPOLOGY_TEST_SIZE / (1024 * 1024)) << "," << std::endl;
    
    // Output GPU information
    std::cout << "  \"gpus\": [" << std::endl;
    for (int i = 0; i < deviceCount; i++) {
        if (i > 0) std::cout << "," << std::endl;
        std::cout << "    {" << std::endl;
        std::cout << "      \"gpu_id\": " << i << "," << std::endl;
        std::cout << "      \"name\": \"" << gpuNames[i] << "\"," << std::endl;
        std::cout << "      \"pci_id\": \"" << gpuPciIds[i] << "\"" << std::endl;
        std::cout << "    }";
    }
    std::cout << std::endl << "  ]," << std::endl;

    // Output bandwidth matrix
    std::cout << "  \"bandwidth_matrix\": [" << std::endl;
    bool firstRow = true;
    
    for (int srcGpu = 0; srcGpu < deviceCount; srcGpu++) {
        if (!firstRow) std::cout << "," << std::endl;
        firstRow = false;
        
        std::cout << "    [" << std::endl;
        bool firstCol = true;
        
        for (int dstGpu = 0; dstGpu < deviceCount; dstGpu++) {
            if (!firstCol) std::cout << "," << std::endl;
            firstCol = false;
            
            int canAccess = 0;
            double bandwidth = 0.0;
            std::string linkType;
            int hops = -1;
            
            if (srcGpu == dstGpu) {
                linkType = "Self";
                bandwidth = 0.0;
                hops = 0;
            } else {
                HIP_CHECK(hipDeviceCanAccessPeer(&canAccess, dstGpu, srcGpu));
                linkType = getLinkType(srcGpu, dstGpu, canAccess);
                hops = getXgmiHops(srcGpu, dstGpu);
                
                if (canAccess) {
                    bandwidth = measureTopologyBandwidth(srcGpu, dstGpu, TOPOLOGY_TEST_SIZE);
                }
            }
            
            std::cout << "      {" << std::endl;
            std::cout << "        \"src\": " << srcGpu << "," << std::endl;
            std::cout << "        \"dst\": " << dstGpu << "," << std::endl;
            std::cout << "        \"link_type\": \"" << linkType << "\"," << std::endl;
            std::cout << "        \"hops\": " << hops << "," << std::endl;
            std::cout << "        \"bandwidth_gbps\": " << std::fixed << std::setprecision(2) << bandwidth << "," << std::endl;
            std::cout << "        \"p2p_enabled\": " << (canAccess ? "true" : "false") << std::endl;
            std::cout << "      }";
        }
        
        std::cout << std::endl << "    ]";
    }
    
    std::cout << std::endl << "  ]";
    
    // Include AMD SMI topology data if available
    if (!amdSmiTopology.empty() && amdSmiTopology.find("gpu") != std::string::npos) {
        std::cout << "," << std::endl;
        std::cout << "  \"amd_smi_topology\": " << amdSmiTopology << std::endl;
    } else {
        std::cout << std::endl;
    }
    
    std::cout << "}" << std::endl;

    return 0;
}
