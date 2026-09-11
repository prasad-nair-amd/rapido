// RCCL collective communication benchmark for AMD GPUs.
// Compile: hipcc -O3 -o gpu_rccl_collectives gpu_rccl_collectives.cpp -lrccl
//
// The existing P2P benchmark measures one GPU talking to one other GPU. Real
// distributed training never does that -- it runs all-reduce across every GPU at
// once, and the bandwidth that collective achieves is not predictable from the
// pairwise numbers, because it depends on the ring or tree the library builds over
// the fabric. A node can have healthy pairwise XGMI links and still collective badly.
//
// This uses ncclCommInitAll, which sets up one communicator across all local GPUs
// from a single process. No MPI and no rccl-tests install is needed, which matters
// because rapido has to run on nodes where neither is present.

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                        \
    do {                                                                      \
        hipError_t error = (cmd);                                             \
        if (error != hipSuccess) {                                            \
            std::cerr << "HIP error: " << hipGetErrorString(error)            \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            return false;                                                     \
        }                                                                     \
    } while (0)

#define NCCL_CHECK(cmd)                                                       \
    do {                                                                      \
        ncclResult_t result = (cmd);                                          \
        if (result != ncclSuccess) {                                          \
            std::cerr << "RCCL error: " << ncclGetErrorString(result)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;  \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace {

// Message sizes in bytes. The small case exposes latency and the large case
// exposes steady-state bandwidth; a node can look fine on one and bad on the other.
const size_t kMessageSizes[] = {
    16UL * 1024 * 1024,
    256UL * 1024 * 1024,
};

const int kWarmupIterations = 5;
const int kTimedIterations = 20;

enum CollectiveKind {
    kAllReduce,
    kAllGather,
    kReduceScatter,
};

struct Result {
    std::string name;
    size_t bytes = 0;
    double algorithm_gbps = 0.0;
    double bus_gbps = 0.0;
    double milliseconds = 0.0;
};

// Bus bandwidth converts the user-visible ("algorithm") bandwidth into the traffic
// actually crossing the fabric, which is what should be compared against the link
// rate. The correction factors are the standard nccl-tests ones: all-reduce moves
// each byte 2(n-1)/n times, all-gather and reduce-scatter (n-1)/n.
double bus_factor(CollectiveKind kind, int ranks) {
    if (ranks <= 1) {
        return 1.0;
    }
    const double n = static_cast<double>(ranks);
    if (kind == kAllReduce) {
        return 2.0 * (n - 1.0) / n;
    }
    return (n - 1.0) / n;
}

const char* kind_name(CollectiveKind kind) {
    switch (kind) {
        case kAllReduce:
            return "AllReduce";
        case kAllGather:
            return "AllGather";
        case kReduceScatter:
            return "ReduceScatter";
    }
    return "Unknown";
}

// Runs one collective at one message size across every rank and times it.
// send/recv buffers are sized for the worst case by the caller, so this only has to
// pick the right element counts for the collective in question.
bool run_collective(CollectiveKind kind, size_t bytes, int ranks,
                    const std::vector<int>& devices,
                    const std::vector<ncclComm_t>& comms,
                    const std::vector<hipStream_t>& streams,
                    const std::vector<float*>& send_buffers,
                    const std::vector<float*>& recv_buffers,
                    Result* out) {
    const size_t elements = bytes / sizeof(float);
    if (elements == 0) {
        return false;
    }
    // All-gather and reduce-scatter define their counts per-rank, so the
    // transferred total matches the requested message size in every case.
    const size_t per_rank = elements / static_cast<size_t>(ranks);
    if ((kind == kAllGather || kind == kReduceScatter) && per_rank == 0) {
        return false;
    }

    auto launch = [&]() -> bool {
        NCCL_CHECK(ncclGroupStart());
        for (int i = 0; i < ranks; ++i) {
            switch (kind) {
                case kAllReduce:
                    NCCL_CHECK(ncclAllReduce(send_buffers[i], recv_buffers[i], elements,
                                             ncclFloat, ncclSum, comms[i], streams[i]));
                    break;
                case kAllGather:
                    NCCL_CHECK(ncclAllGather(send_buffers[i], recv_buffers[i], per_rank,
                                             ncclFloat, comms[i], streams[i]));
                    break;
                case kReduceScatter:
                    NCCL_CHECK(ncclReduceScatter(send_buffers[i], recv_buffers[i], per_rank,
                                                 ncclFloat, ncclSum, comms[i], streams[i]));
                    break;
            }
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
        if (!launch()) {
            return false;
        }
    }
    for (int i = 0; i < ranks; ++i) {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    // Wall-clock across all ranks: the collective is only complete when the
    // slowest rank is, so per-stream event timing would understate it.
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kTimedIterations; ++i) {
        if (!launch()) {
            return false;
        }
    }
    for (int i = 0; i < ranks; ++i) {
        HIP_CHECK(hipSetDevice(devices[i]));
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }
    const auto stop = std::chrono::high_resolution_clock::now();

    const double seconds =
        std::chrono::duration<double>(stop - start).count() / kTimedIterations;
    if (seconds <= 0.0) {
        return false;
    }

    out->name = kind_name(kind);
    out->bytes = bytes;
    out->milliseconds = seconds * 1000.0;
    out->algorithm_gbps = (static_cast<double>(bytes) / seconds) / 1e9;
    out->bus_gbps = out->algorithm_gbps * bus_factor(kind, ranks);
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess || device_count == 0) {
        std::cout << "{\"error\": \"No HIP devices found\"}" << std::endl;
        return 0;
    }
    if (device_count < 2) {
        // A single-GPU host has no collective to measure. This is a normal
        // configuration, not a failure, so it reports a skip rather than an error.
        std::cout << "{\"skipped\": \"Collective benchmarks require at least 2 GPUs; "
                  << "this host has " << device_count << "\"}" << std::endl;
        return 0;
    }

    const int ranks = device_count;
    std::vector<int> devices(ranks);
    for (int i = 0; i < ranks; ++i) {
        devices[i] = i;
    }

    std::vector<ncclComm_t> comms(ranks, nullptr);
    if (ncclCommInitAll(comms.data(), ranks, devices.data()) != ncclSuccess) {
        std::cout << "{\"error\": \"ncclCommInitAll failed; RCCL could not build a "
                  << "communicator across the " << ranks << " visible GPUs\"}" << std::endl;
        return 0;
    }

    // Allocate for the largest message once. All-gather's receive buffer is the
    // biggest consumer at ranks x the message size.
    size_t largest = 0;
    for (size_t i = 0; i < sizeof(kMessageSizes) / sizeof(kMessageSizes[0]); ++i) {
        if (kMessageSizes[i] > largest) {
            largest = kMessageSizes[i];
        }
    }

    std::vector<hipStream_t> streams(ranks);
    std::vector<float*> send_buffers(ranks, nullptr);
    std::vector<float*> recv_buffers(ranks, nullptr);
    bool allocated = true;

    for (int i = 0; i < ranks && allocated; ++i) {
        if (hipSetDevice(devices[i]) != hipSuccess ||
            hipStreamCreate(&streams[i]) != hipSuccess ||
            hipMalloc(&send_buffers[i], largest) != hipSuccess ||
            hipMalloc(&recv_buffers[i], largest) != hipSuccess) {
            allocated = false;
            break;
        }
        hipMemset(send_buffers[i], 0, largest);
        hipMemset(recv_buffers[i], 0, largest);
    }

    std::vector<Result> results;
    if (allocated) {
        const CollectiveKind kinds[] = {kAllReduce, kAllGather, kReduceScatter};
        for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); ++k) {
            for (size_t s = 0; s < sizeof(kMessageSizes) / sizeof(kMessageSizes[0]); ++s) {
                Result result;
                if (run_collective(kinds[k], kMessageSizes[s], ranks, devices, comms,
                                   streams, send_buffers, recv_buffers, &result)) {
                    results.push_back(result);
                }
            }
        }
    }

    for (int i = 0; i < ranks; ++i) {
        if (send_buffers[i]) {
            hipFree(send_buffers[i]);
        }
        if (recv_buffers[i]) {
            hipFree(recv_buffers[i]);
        }
        if (streams[i]) {
            hipStreamDestroy(streams[i]);
        }
    }
    for (int i = 0; i < ranks; ++i) {
        if (comms[i]) {
            ncclCommDestroy(comms[i]);
        }
    }

    if (results.empty()) {
        std::cout << "{\"error\": \"RCCL collectives produced no measurements\"}" << std::endl;
        return 0;
    }

    int version = 0;
    ncclGetVersion(&version);

    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    out << "{\n";
    out << "  \"ranks\": " << ranks << ",\n";
    out << "  \"rccl_version\": " << version << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        out << "    {\"collective\": \"" << r.name << "\", "
            << "\"bytes\": " << r.bytes << ", "
            << "\"algorithm_gbps\": " << r.algorithm_gbps << ", "
            << "\"bus_gbps\": " << r.bus_gbps << ", "
            << "\"time_ms\": " << r.milliseconds << "}";
        if (i + 1 < results.size()) {
            out << ",";
        }
        out << "\n";
    }
    out << "  ]\n}";
    std::cout << out.str() << std::endl;
    return 0;
}
