# ReadMe.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AMD Rapido is a Python-based hardware profiling toolkit for AMD GPU-equipped servers. It consists of:
- **rapido-collect.py** - Collects comprehensive hardware information (CPU, GPU, network, BMC, ROCm) and exports to JSON
- **rapido-report.py** - Generates interactive HTML comparison reports from JSON files
- **gpu_p2p_bandwidth.cpp** - HIP-based GPU-to-GPU peer-to-peer bandwidth benchmark
- **gpu_kernel_benchmarks.cpp** - HIP-based compute kernel benchmarks (GEMM, convolution, memory bandwidth)
- **gpu_host_bandwidth.cpp** - HIP-based GPU-CPU transfer bandwidth benchmark (H2D/D2H)
- **gpu_topology.cpp** - HIP-based XGMI/Infinity Fabric topology analysis (NVLink equivalent)
- **storage_benchmark.py** - Storage I/O profiling (device detection, NVMe metrics, RAID config, GDS capability)
- **network_benchmark.py** - Network performance testing (bandwidth tools, RDMA/RoCE detection, topology, MPI benchmarks)

The tool is designed for Linux systems with AMD GPUs and ROCm, but includes basic support for Windows and macOS.

## Key Commands

### Tool Availability Check
When rapido-collect.py runs, it automatically checks for all required and optional tools at the beginning and reports:
- Which tools are available/missing
- Impact on each report section if tools are missing
- Special checks for ROCm installation and benchmark source files
- Summary of affected functionality

### Data Collection
```bash
# Basic collection - all sections (CPU, GPU, Network, BMC, ROCm) - quiet mode
# NOTE: Microbenchmarks are NOT included with -a, must use -m explicitly
python3 rapido-collect.py
# or explicitly
python3 rapido-collect.py -a

# Collect specific sections only
python3 rapido-collect.py -c           # CPU only
python3 rapido-collect.py -g           # GPU only
python3 rapido-collect.py -n           # Network only
python3 rapido-collect.py -b           # BMC only (may require sudo)
python3 rapido-collect.py -r           # ROCm only
python3 rapido-collect.py -m           # Microbenchmarks only (automatically includes ROCm)
python3 rapido-collect.py -t           # Host platform only (BIOS/DMI, EDAC, NUMA, tuning, affinity)

# Combine multiple sections
python3 rapido-collect.py -c -g        # CPU and GPU only
python3 rapido-collect.py -c -g -n     # CPU, GPU, and Network only
python3 rapido-collect.py -g -r        # GPU and ROCm only

# With GPU P2P bandwidth testing (requires -m)
python3 rapido-collect.py -m -p

# With the heavy library benchmarks as well: rocBLAS achieved GEMM and RCCL
# collectives. Adds a few minutes; the default -m run is unchanged without it.
python3 rapido-collect.py -m --full

# Verbose mode - shows tool availability check and progress messages
python3 rapido-collect.py -v

# Control where the JSON is written
python3 rapido-collect.py -o /tmp/myrun.json          # exact path
python3 rapido-collect.py --output-dir ./runs         # serverinfo_<hostname>.json in ./runs
python3 rapido-collect.py --tag before                # serverinfo_<hostname>_before.json

# Collect the same host twice without overwriting, then compare the two runs
python3 rapido-collect.py -m -p --tag before
# ... make a change ...
python3 rapido-collect.py -m -p --tag after
python3 rapido-report.py -f1 serverinfo_$(hostname)_before.json \
                         -f2 serverinfo_$(hostname)_after.json

# Full collection with all features and verbose output
sudo python3 rapido-collect.py -a -m -p --full -v
```

### Report Generation
```bash
# Single server report
python3 rapido-report.py -f1 serverinfo_server1.json -o report.html

# Side-by-side comparison of two servers (differences highlighted in yellow)
python3 rapido-report.py -f1 serverinfo_server1.json -f2 serverinfo_server2.json -o comparison.html

# Legacy single file mode
python3 rapido-report.py -i serverinfo_server1.json -o report.html
```

**Comparison Report Features**:
- Side-by-side layout for easy visual comparison
- **Yellow highlighting** automatically applied to fields with different values
- **Differences only** toggle hides every matching row so only the deltas remain
- Intelligent matching of corresponding components (GPUs, network interfaces, etc.)
- Works across all sections: CPU, GPU, ROCm, Network, BMC, and Microbenchmarks

**Report Features** (single and comparison mode):
- **GPU health banner** above the tabs: uncorrectable ECC errors, bad pages, XGMI errors,
  degraded PCIe links and active throttling. Renders a green "OK" when clean, because
  "no uncorrectable errors" is a result an acceptance audit needs stated, not inferred
- **GPU Health tab** (RAS counters and live telemetry) and **Platform tab** (BIOS, EDAC,
  NUMA, kernel tuning, device affinity)
- **Microbenchmark dashboard**: P2P bandwidth heatmap and per-GPU bar charts, with
  automatic flagging of any GPU or link deviating more than 15% from the node median;
  under `--full`, also achieved rocBLAS GEMM (absolute and as a percentage of the
  theoretical peak) and RCCL collective bus bandwidth
- **Search** across all cards, **collapsible cards**, **dark mode**, deep-linkable tabs
- Still a single self-contained HTML file - no external assets, renders offline

### GPU Benchmarks
```bash
# Compile all benchmarks (requires ROCm and hipcc)
hipcc -o gpu_p2p_bandwidth gpu_p2p_bandwidth.cpp
hipcc -O3 -o gpu_kernel_benchmarks gpu_kernel_benchmarks.cpp
hipcc -o gpu_host_bandwidth gpu_host_bandwidth.cpp
hipcc -o gpu_topology gpu_topology.cpp
hipcc -O3 -o gpu_gemm_library gpu_gemm_library.cpp -lrocblas
hipcc -O3 -o gpu_rccl_collectives gpu_rccl_collectives.cpp -lrccl

# Run directly (optional - rapido-collect.py does this automatically)
./gpu_p2p_bandwidth
./gpu_kernel_benchmarks
./gpu_host_bandwidth
./gpu_topology
./gpu_gemm_library        # only run by the collector under --full
./gpu_rccl_collectives    # only run by the collector under --full
```

## Architecture

### rapido-collect.py (main collection script)
**Platform detection**: Uses `platform.system()` to detect OS and call appropriate collection functions

**Tool availability checking**: `check_tool_availability(verbose: bool)` function runs at startup
- Checks OS-specific tools (lscpu, amd-smi, ethtool, ipmitool, hipcc, etc.)
- Displays tool name alongside command name for clarity (e.g., "GPU Information (amd-smi)")
- Verifies benchmark source files exist
- Checks for ROCm installation directory
- Reports impact of missing tools on each report section
- Groups missing tools by category (CPU, GPU, Network, BMC, ROCm, Microbenchmarks)
- Shows both friendly name and actual command in output for easy troubleshooting
- Only displays when `verbose=True` (controlled by `-v` flag)

**Verbose mode**: Controlled by `-v` or `--verbose` flag
- When disabled (default): Only shows final "Server information saved to: {filename}" message
- When enabled: Shows tool availability check, compilation messages, and all progress indicators
- All debug print statements in collection functions are suppressed by default

**Selective collection flags**: Control which sections to collect
- `-c` or `--cpu`: Collect CPU information only
- `-g` or `--gpu`: Collect GPU information only
- `-n` or `--network`: Collect network information only
- `-b` or `--bmc`: Collect BMC information only
- `-r` or `--rocm`: Collect ROCm information only
- `-m` or `--microbenchmarks`: Collect microbenchmarks only (automatically includes ROCm)
- `-t` or `--platform`: Collect host platform information only (BIOS/DMI, EDAC, NUMA, kernel tuning, GPU/NIC affinity)
- `-a` or `--all`: Collect all basic sections: CPU, GPU, Network, BMC, ROCm, Platform (NOT microbenchmarks)
- `--full`: With `-m`, also run the heavy library benchmarks (rocBLAS GEMM, RCCL collectives)
- Flags can be combined: `-c -g -n` collects CPU, GPU, and Network only
- When any specific flag is used, only those sections are collected
- When no flags or `-a` is used, all basic sections are collected (CPU, GPU, Network, BMC, ROCm, Platform)
- RAS/ECC health and live telemetry ride along with `-g`: they are GPU state, not benchmarks, and are cheap
- **Important**: Microbenchmarks are ONLY collected when `-m` flag is explicitly specified
- Note: `-m` flag automatically enables ROCm collection (microbenchmarks need ROCm info)

**Output location flags**: Control where the JSON is written
- `-o` or `--output FILE`: Write to this exact path, overriding the default filename
- `--output-dir DIR`: Write the default-named file into this directory (created if missing)
- `--tag TAG`: Append a suffix, e.g. `--tag before` -> `serverinfo_<hostname>_before.json`
- Default remains `serverinfo_<hostname>.json` in the current directory
- `-o` takes precedence over `--output-dir` and `--tag`
- Useful for collecting the same host before and after a change without overwriting

**Command timeouts**: Every external command runs under a wall-clock limit
- Default 60s; 600s for HIP compilation; 1800s for the GPU benchmarks
- A hung tool (e.g. `amd-smi` against a wedged GPU, `ipmitool` against an unresponsive BMC)
  no longer blocks the entire collection
- Failures are recorded in `_metadata.command_failures` with a reason: `not_found`,
  `timeout`, or `error` - so an empty section can be explained rather than being
  mistaken for "this machine has no such hardware"
- Timeouts are always reported at the end of the run; errors are shown with `-v`

**GPU health (`ras` and `telemetry` sections, collected with `-g`)**:
- `gather_ras_health()`: ECC correctable/uncorrectable/deferred counts per GPU and per block,
  retired/pending/unreservable memory pages, and XGMI link errors, plus a node summary.
  Blocks the ASIC does not instrument (`amd-smi` reports them as the *string* `"N/A"`)
  are listed as "Not Instrumented" rather than being counted as zero.
- `gather_gpu_telemetry()`: a point-in-time snapshot of power, temperature, clocks and
  utilisation, and two derived verdicts:
  - **PCIe link**: the currently trained width/speed compared against the static maximum
    already collected, so a card that trained to x8 in an x16 slot is called out as
    `DEGRADED` instead of hiding behind a plausible-looking "x8".
  - **Throttling**: only the present-tense `*_violation_status` fields are flagged. The
    `*_accumulated` counters are lifetime residency totals that are nonzero on any healthy
    board that has ever touched its power limit, so they are reported as context only.
- One targeted `amd-smi` invocation per metric group, no polling loop.

**Partition mode** (folded into the GPU section): accelerator mode (SPX/DPX/CPX) and memory
mode (NPS1/2/4) per device, with a summary that flags non-uniform configurations. Recorded as
context, not as a correction: each partition is enumerated as its own device, so the reported
CU counts and peak figures are already per-partition.

**Host platform (`platform` section, `-t`)**: `gather_platform_details()` reads unprivileged
sysfs only - no `sudo`, no prompts, nothing that can hang an unattended run:
- BIOS vendor/version/date and board model from `/sys/class/dmi/id`
- Host memory controllers and their EDAC correctable/uncorrectable counts
- NUMA node CPU lists, memory, and the distance matrix
- Kernel tuning: IOMMU boot parameters, transparent hugepages, cpufreq governor and driver,
  boost, cpuidle driver, SMT, automatic NUMA balancing
- GPU/NIC NUMA affinity map and a locality verdict
- Deliberately **descriptive, not prescriptive**: settings are reported as found, because the
  right values depend on the deployment (an SR-IOV host legitimately needs full IOMMU
  translation). Per-DIMM population needs root and its absence is recorded explicitly.

**Error handling and crash protection**:
- Each section has individual error handling - if one section fails, others continue
- Ctrl+C (KeyboardInterrupt) is caught and partial data is saved
- Collection errors are logged in `_metadata.errors` array
- Collection status is tracked: "complete" or "partial"
- JSON validation occurs after writing to detect corruption
- Helpful error messages guide users to re-run collection if needed

**Data gathering functions** (return OrderedDict structures):
- `gather_cpu_details()` - Routes to OS-specific CPU collection
  - `linux_cpu_info()`: Uses `lscpu --json` or `/proc/cpuinfo`
  - `windows_cpu_info()`: Uses WMIC or PowerShell
  - `mac_cpu_info()`: Uses `sysctl`
  
- `gather_gpu_details()` - Routes to OS-specific GPU collection
  - `linux_gpu_info()`: Uses `amd-smi static --json`, `amd-smi version`, `amd-smi list`, `amd-smi topology`, `amd-smi firmware`
  - `windows_gpu_info()`: Uses WMIC or PowerShell (basic info only)
  - `mac_gpu_info()`: Uses `system_profiler SPDisplaysDataType`
  
- `gather_network_details()` - Collects detailed NIC information
  - `linux_network_info()`: Uses `ip -json link/addr` and `ethtool` for driver, firmware, speed, duplex, statistics
  - `windows_network_info()`: Uses PowerShell `Get-NetAdapter`
  - `mac_network_info()`: Uses `ifconfig` and `netstat`
  
- `gather_bmc_info()` - Linux-only BMC data via IPMI
  - Uses `ipmitool` commands: `bmc info`, `lan print`, `sdr list`, `fru print`, `sel info/list`, `chassis status`
  - Groups sensor data by type (temperature, voltage, fan, power)
  
- `gather_rocm_details()` - Linux-only ROCm runtime information
  - Checks `/opt/rocm/.info/version`
  - Runs `hipcc --version`, `rocm-smi --version`, `amd-smi version`
  - Collects environment variables (ROCM_PATH, HIP_*, etc.)
  - Lists installed packages via `dpkg` or `rpm`
  - Uses `rocminfo`, `clinfo`, `hipconfig`, `lsmod`, `modinfo`
  
- `gather_gpu_microbenchmarks(include_p2p)` - Linux-only GPU benchmarking
  - Parses `rocminfo` to extract GPU specs (CUs, clock, memory)
  - Calculates peak performance with separate Dense and Sparse cards:
    - **Dense**: FP64, FP32, TF32 (CDNA2+), FP16/BF16, FP8 (CDNA3), INT8
    - **Sparse**: 2:1 sparse matrix operations (50% sparsity, 2x dense for CDNA), FP8 shown as range
  - Compiles and runs `gpu_kernel_benchmarks.cpp` for real-world kernel performance
  - If `include_p2p=True`: compiles and runs additional benchmarks:
    - `gpu_p2p_bandwidth.cpp` - GPU-to-GPU communication bandwidth
    - `gpu_host_bandwidth.cpp` - GPU-CPU transfer bandwidth (H2D/D2H, pageable/pinned)
    - `gpu_topology.cpp` - XGMI/Infinity Fabric topology analysis with bandwidth matrix
    - `storage_benchmark.py` - Storage I/O profiling (always runs with microbenchmarks)
    - `network_benchmark.py` - Network performance testing (always runs with microbenchmarks)

**Output structure**: JSON file `serverinfo_<hostname>.json` with sections:
- `_metadata`: Collection metadata (date, status, errors) - added in v2.1
  - `collection_date`: ISO format timestamp
  - `collection_status`: "complete" or "partial"
  - `errors`: Array of error messages if any section failed
- `cpu`: CPU information
- `gpu`: GPU information  
- `network`: Network interface information
- `bmc`: BMC/IPMI information
- `rocm`: ROCm software stack information
- `microbenchmarks`: GPU performance benchmarks

**Helper function**: `run_command(cmd)` - Executes subprocess, returns stdout or None on error

### rapido-report.py (HTML report generator)
**Input processing**: Loads 1-2 JSON files via argparse (`-f1`, `-f2`, or legacy `-i`)

**Data extraction**: `_extract_section_data(data, section)` - Flattens nested OS-specific data structures

**Difference detection**: `_values_differ(value1, value2)` - Intelligently compares values
- Handles None values and type differences
- Extracts and compares numeric values from formatted strings (e.g., "16 GB", "3.5 GHz")
- Falls back to normalized string comparison

**Rendering functions**:
- `_render_dict_as_table(data, comparison_data)`: Converts dict to HTML table with optional difference highlighting
- `_render_list_as_cards(items, comparison_items)`: Creates card UI with intelligent component matching
- `_render_comparison_section()`: Side-by-side layout with cross-comparison highlighting
- `_render_single_section()`: Single file layout
- `_value_to_html(value, comparison_value)`: Recursively converts data structures, preserving comparison context

**Comparison logic**:
- Matches corresponding cards by identifier (Section, Name, Interface, Adapter fields)
- Falls back to index-based matching when identifier matching fails
- Applies `highlight-diff` CSS class (yellow background) to differing values
- Works recursively through nested dictionaries and lists

**Microbenchmark dashboard**: Charts rendered above the Microbenchmarks cards
- `_render_p2p_heatmap()`: The N×N P2P bandwidth matrix as a colour-scaled grid, replacing
  56 unreadable per-pair cards on an 8-GPU node with one glance. Hover for exact GB/s.
- `_render_bar_chart()`: One bar per GPU per benchmark, with a dashed median marker, so a
  single degraded GPU in a node of 8 is visually obvious
- `_uniformity_warnings()`: Flags any GPU deviating more than `OUTLIER_THRESHOLD` (15%)
  from the node median - the most common defect an acceptance audit is looking for
- Charts are **inline SVG generated in Python** - no JS charting library, so the report
  remains a single self-contained file that renders with no network access
- Consumes the `_raw` numeric blocks; JSON files collected before those existed still
  render normally, just without charts

**Toolbar**: Controls above the tab content
- **Live search** filters cards by any text they contain
- **Differences only** (comparison mode) hides every card and row whose values match
- **Collapse all / Expand all**, plus click any card header to collapse that card
- **Dark mode**, persisted in localStorage
- Tabs are deep-linkable: `report.html#gpu` opens the GPU tab directly

**HTML output**: 
- Tabbed interface (CPU, GPU, ROCm, Network, BMC*, Microbenchmarks*)
- Tabs are draggable/reorderable with localStorage persistence
- Conditional BMC and Microbenchmarks tabs based on data availability
- Gradient purple theme with responsive design
- **Yellow highlighting** (`#fff9c4` background) for differences in comparison mode
- Underscore-prefixed JSON keys (e.g. `_raw`) are treated as machine-readable side-channel
  data and are never rendered as display rows

### gpu_p2p_bandwidth.cpp (HIP benchmark)
**Purpose**: Measures GPU-to-GPU communication bandwidth

**Methodology**:
- Tests all GPU pairs (N² tests for N GPUs)
- 256 MB transfers × 10 iterations per pair
- Checks P2P access capability with `hipDeviceCanAccessPeer()`
- Enables peer access with `hipDeviceEnablePeerAccess()` if available
- Uses `hipMemcpy(DeviceToDevice)` for transfers
- Times with `std::chrono::high_resolution_clock`

**Output**: JSON to stdout with structure:
```json
{
  "gpu_count": N,
  "test_size_mb": 256,
  "iterations": 10,
  "results": [
    {
      "src_gpu": 0,
      "dst_gpu": 1,
      "src_name": "gfx950:sramecc+:xnack-",
      "dst_name": "gfx950:sramecc+:xnack-",
      "p2p_enabled": true,
      "bandwidth_gbps": 56.85
    }
  ]
}
```

### gpu_kernel_benchmarks.cpp (HIP compute benchmarks)
**Purpose**: Measures real-world kernel performance across various compute patterns

**Benchmarks included**:
1. **Memory Bandwidth Test** (512 MB)
   - Device-to-device memory copy
   - Measures achievable memory bandwidth vs theoretical max
   
2. **GEMM FP32** (2048×2048 matrix multiplication)
   - Tests single-precision matrix multiply performance
   - Core operation for deep learning and HPC
   
3. **GEMM FP64** (1024×1024 matrix multiplication)
   - Tests double-precision compute capability
   - Important for scientific computing workloads
   
4. **Vector Add** (256 MB)
   - Simple element-wise addition
   - Tests memory-bound operation performance
   
5. **FMA Throughput** (128 MB, 100 FMA ops/element)
   - Fused multiply-add intensive kernel
   - Tests peak compute throughput
   
6. **1D Convolution** (16M elements, kernel size 32)
   - Stencil-based computation pattern
   - Representative of signal processing and ML workloads

**Output**: JSON with per-GPU results including GFLOPS/TFLOPS and bandwidth measurements

### gpu_host_bandwidth.cpp (HIP GPU-CPU bandwidth benchmark)
**Purpose**: Measures data transfer speeds between GPU and CPU (host) memory

**Benchmarks included**:
1. **Host-to-Device (H2D)** - Pageable memory
   - Standard malloc() host memory to GPU device memory
   - Tests typical CPU→GPU data upload performance
   
2. **Device-to-Host (D2H)** - Pageable memory
   - GPU device memory to standard malloc() host memory
   - Tests typical GPU→CPU data download performance
   
3. **Host-to-Device (H2D)** - Pinned memory
   - Page-locked (hipHostMalloc) memory to GPU device memory
   - Tests optimized CPU→GPU transfer performance
   
4. **Device-to-Host (D2H)** - Pinned memory
   - GPU device memory to page-locked host memory
   - Tests optimized GPU→CPU transfer performance

**Methodology**:
- 256 MB transfers × 10 iterations per test
- Warm-up runs to eliminate one-time costs
- Compares pageable vs pinned memory performance
- Identifies PCIe bandwidth bottlenecks

**Output**: JSON with per-GPU H2D/D2H bandwidth results in GB/s for both pageable and pinned memory

### gpu_topology.cpp (HIP topology analyzer)
**Purpose**: Analyzes AMD GPU interconnect topology - equivalent to NVIDIA's NVLink analysis

**Features**:
1. **Bandwidth Matrix** - Measures actual transfer speeds between all GPU pairs
2. **Link Type Detection**:
   - **XGMI** - AMD Infinity Fabric direct GPU-GPU links (high-speed)
   - **XGMI-2hop/3hop** - Multi-hop XGMI connections
   - **PCIe** - PCIe-based GPU communication (lower speed)
   - **No P2P** - No peer-to-peer access available
   
3. **Topology Information**:
   - Reads `/sys/class/drm/card*/device/xgmi_hive_info/node_*_hops` for hop counts
   - Uses `amd-smi topology --json` if available
   - Measures real bandwidth for each link
   - Identifies NUMA domains and socket topology
   
4. **Link Performance Analysis**:
   - Tests smaller transfers (64 MB) for faster topology mapping
   - 5 iterations per GPU pair
   - Identifies bottlenecks in multi-GPU configurations

**Output**: JSON with:
- GPU list with names and PCI IDs
- Full bandwidth matrix (N×N for N GPUs)
- Link types and hop counts for each connection
- Summary of XGMI vs PCIe links

### gpu_gemm_library.cpp (rocBLAS achieved GEMM)
**Purpose**: Measures the throughput a *tuned library* reaches, which is the only
honest basis for an efficiency verdict

`gpu_kernel_benchmarks.cpp` runs a naive hand-written GEMM. That is useful as a
smoke test but it reaches single-digit percentages of the hardware peak, so
dividing it by the theoretical peak understates the GPU by more than an order of
magnitude. This benchmark calls rocBLAS instead and reports both the achieved
TFLOPS and the ratio against the Tier 1 theoretical peak.

**Features**:
1. **FP32** via `rocblas_sgemm`
2. **FP16 and BF16** via `rocblas_gemm_ex` with an FP32 compute type, which is
   what exercises the MFMA matrix cores
3. **Two square sizes** (4096 and 8192) - 8192 is large enough to reach steady
   state, 4096 shows whether the smaller problem is already saturating
4. **Every visible GPU** is measured, so a single slow device is visible rather
   than averaged away
5. 3 warmup plus 10 timed iterations per case

FP8 is deliberately out of scope: `rocblas_gemm_ex` cannot express it, and the
gfx942 path needs hipBLASLt with AMD-specific FNUZ types.

**Output**: JSON with per-GPU achieved TFLOPS per precision and size, plus the
efficiency percentage against the theoretical peak

Only compiled and run when the collector is given `--full`, because it adds
minutes to a run.

### gpu_rccl_collectives.cpp (RCCL collective bandwidth)
**Purpose**: Measures multi-GPU collective bandwidth, the number that actually
predicts distributed training scaling

**Features**:
1. **All-reduce, all-gather, reduce-scatter** across every visible GPU
2. **Single process, no MPI and no rccl-tests** - `ncclCommInitAll` builds one
   communicator over all local devices, so there is nothing extra to install
3. **Bus bandwidth as well as algorithmic bandwidth**, applying the standard
   `2(n-1)/n` correction for all-reduce and `(n-1)/n` for the other two. Bus
   bandwidth is the figure comparable against the interconnect's rated speed
4. **Two message sizes** (16 MB and 256 MB): the small case exposes latency, the
   large case exposes steady-state bandwidth, and a node can look fine on one
   while being bad on the other

Detection is by library presence (`librccl.so` / `rccl.h`), not by probing for an
executable - RCCL is a library and has no binary to run. On a single-GPU host the
benchmark reports a skip rather than a meaningless one-rank result.

**Output**: JSON with per-collective, per-size algorithmic and bus bandwidth in
GB/s, the rank count, and the RCCL version

Only compiled and run when the collector is given `--full`.

### storage_benchmark.py (Storage I/O profiling)
**Purpose**: Comprehensive storage system analysis for HPC/AI workloads

**Features**:
1. **Storage Device Detection**:
   - Uses `lsblk -J` to enumerate all block devices
   - Identifies SSD vs HDD (rotational flag)
   - Detects transport type (SATA, NVMe, SAS, etc.)
   - Reports model, size, and device path
   
2. **NVMe Performance Metrics**:
   - Uses `nvme list -o json` to get NVMe-specific details
   - Reports model, serial number, firmware version
   - Shows namespace information
   - Identifies NVMe device capabilities
   
3. **RAID Configuration Detection**:
   - Uses `mdadm` to detect hardware/software RAID arrays
   - Shows RAID level (RAID0, RAID1, RAID5, RAID6, RAID10)
   - Reports array size and device count
   - Detects LVM RAID configurations
   - Shows RAID state (active, degraded, etc.)
   
4. **GPU Direct Storage (GDS) Capability**:
   - Checks for GDS kernel modules (`nvidia_fs`, `gdrdrv`)
   - Detects cuFile library installation
   - Verifies GDS configuration files
   - Reports overall GDS capability status
   - Note: GDS is primarily NVIDIA technology; AMD equivalent may vary
   
5. **Optional Disk Benchmarks** (disabled by default):
   - Sequential read/write tests using `dd`
   - Direct I/O to bypass page cache
   - Configurable test size
   - Can be enabled by uncommenting code in `storage_benchmark.py`

**Methodology**:
- Non-destructive testing (no existing data modified)
- Uses temporary directory for benchmark tests
- Requires root/sudo for some operations (RAID detection, cache clearing)
- Fast device enumeration (~1-2 seconds)
- Optional benchmarks can add 30-60 seconds per device

**Output**: JSON with:
- `storage_devices`: List of all detected storage devices with type/model/size
- `nvme_devices`: NVMe-specific details (if NVMe present)
- `raid_configs`: RAID array configurations (if RAID detected)
- `gds_capability`: GPU Direct Storage capability flags
- `benchmark_results`: Optional disk speed test results

### network_benchmark.py (Network performance testing)
**Purpose**: Comprehensive network performance analysis for HPC/AI multi-node clusters

**Features**:
1. **RDMA/InfiniBand Device Detection**:
   - Uses `ibstat` to detect InfiniBand devices
   - Uses `rdma link show` for RoCE devices
   - Reports device type, state, firmware version
   - Shows link rate and physical state
   - Displays LID (Local Identifier) for IB fabric
   
2. **RoCE (RDMA over Converged Ethernet) Capability**:
   - Detects RDMA kernel modules (rdma_*, ib_*, mlx*)
   - Checks `/sys/class/infiniband` for InfiniBand devices
   - Verifies rdma-core package installation
   - Checks for RDMA performance tools (ib_send_bw, etc.)
   - Reports overall RoCE capability status
   
3. **Multi-Node Network Topology Mapping**:
   - Reports hostname and active network interfaces
   - Shows interface state, MTU, and IP addresses
   - Detects MPI installation (mpirun/mpiexec)
   - Reports MPI version (OpenMPI, MPICH, Intel MPI)
   - Identifies interfaces suitable for cluster communication
   
4. **Network Bandwidth Testing Tools**:
   - Detects iperf3 installation
   - Provides usage notes for multi-node testing
   - Reports capability for bandwidth measurements
   - Note: Actual bandwidth tests require running iperf3 server on remote node
   
5. **MPI Benchmark Tools Detection**:
   - Detects OSU Micro-Benchmarks (osu_bw, osu_latency, osu_bibw)
   - Checks for Intel MPI Benchmarks (IMB-MPI1)
   - Reports installation paths and available tests
   - Provides usage examples for multi-node MPI testing
   - Note: MPI benchmarks require multi-node cluster setup

**Methodology**:
- Detection-only (no actual network traffic generated)
- Fast execution (~1-2 seconds)
- Safe to run on production systems
- Provides readiness assessment for network performance testing
- Actual performance tests require multi-node cluster environment

**Multi-Node Testing Notes**:
- For iperf3: Run `iperf3 -s` on one node, `iperf3 -c <host>` on another
- For OSU: `mpirun -np 2 -host node1,node2 osu_bw`
- For IMB: `mpirun -np 2 -host node1,node2 IMB-MPI1 PingPong`
- Requires passwordless SSH between nodes
- Requires MPI installation with network fabric support (IB, RoCE)

**Output**: JSON with:
- `rdma_devices`: List of detected RDMA/InfiniBand devices
- `roce_capability`: RoCE capability flags and module status
- `network_topology`: Hostname, interfaces, MPI availability
- `bandwidth_tools`: iperf3 availability and usage notes
- `mpi_benchmarks`: MPI benchmark tool availability and paths

## Important Data Structures

**Section cards**: All data is organized into OrderedDict entries with a "Section" key that becomes the card title in the HTML report. This is used for GPU per-device data, ROCm subsystems, BMC sensor groups, network interfaces, etc.

**Value/unit handling**: Some amd-smi fields return `{"value": X, "unit": Y}` - code handles both this format and plain values

**GPU architectures**: Code recognizes GFX versions (gfx90a, gfx940-942 = CDNA, gfx900/906/908 = Vega/MI100, gfx11xx = RDNA) to calculate correct FP64 ratios and advanced precisions (FP8, FP4)

## Dependencies

**Required**:
- Python 3.6+
- ROCm installation with `amd-smi` (for AMD GPU features on Linux)

**Optional** (Linux):
- `ipmitool` - BMC information collection
- `ethtool` - Enhanced network details
- `hipcc` - Compiling GPU P2P benchmark
- `numactl` - NUMA distance matrix in the platform section (the rest of that
  section comes from sysfs and needs no extra tools)
- `rocblas` and `rccl` development headers/libraries - only needed for `--full`;
  both ship with a standard ROCm install

**System-specific tools**:
- Linux: `lscpu`, `ip`, `rocminfo`, `clinfo`, `lsmod`, `modinfo`
- Windows: PowerShell, WMIC (deprecated but supported)
- macOS: `sysctl`, `system_profiler`, `ifconfig`, `netstat`

## Development Notes

**Error handling**: Most collection functions use try/except or check `run_command()` return for None - missing tools result in empty sections rather than failures

**OS compatibility**: Core collection works on all platforms, but advanced features (ROCm, BMC, P2P) are Linux-only

**Data format**: JSON output uses nested structure: `{section: {os_type: [items]}}` where items are OrderedDict with "Section" key

**HTML generation**: Template uses f-strings with embedded Python expressions, escapes user data via `html.escape()`

**Tab ordering**: JavaScript drag-and-drop allows users to reorder tabs, persists to localStorage per-browser

**Performance**: Full collection with `-m -p` on 8-GPU system takes ~1-2 minutes due to P2P tests (56 pairs × benchmark time)
