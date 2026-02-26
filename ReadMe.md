# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AMD Rapido is a Python-based hardware profiling toolkit for AMD GPU-equipped servers. It consists of:
- **rapido-collect.py** - Collects comprehensive hardware information (CPU, GPU, network, BMC, ROCm) and exports to JSON
- **rapido-report.py** - Generates interactive HTML comparison reports from JSON files
- **gpu_p2p_bandwidth.cpp** - HIP-based GPU-to-GPU peer-to-peer bandwidth benchmark
- **gpu_kernel_benchmarks.cpp** - HIP-based compute kernel benchmarks (GEMM, convolution, memory bandwidth)

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

# Combine multiple sections
python3 rapido-collect.py -c -g        # CPU and GPU only
python3 rapido-collect.py -c -g -n     # CPU, GPU, and Network only
python3 rapido-collect.py -g -r        # GPU and ROCm only

# With GPU P2P bandwidth testing (requires -m)
python3 rapido-collect.py -m -p

# Verbose mode - shows tool availability check and progress messages
python3 rapido-collect.py -v

# Full collection with all features and verbose output
sudo python3 rapido-collect.py -a -p -v
```

### Report Generation
```bash
# Single server report
python3 rapido-report.py -f1 serverinfo_server1.json -o report.html

# Side-by-side comparison of two servers
python3 rapido-report.py -f1 serverinfo_server1.json -f2 serverinfo_server2.json -o comparison.html

# Legacy single file mode
python3 rapido-report.py -i serverinfo_server1.json -o report.html
```

### GPU Benchmarks
```bash
# Compile the P2P benchmark (requires ROCm and hipcc)
hipcc -o gpu_p2p_bandwidth gpu_p2p_bandwidth.cpp

# Compile the kernel benchmarks
hipcc -O3 -o gpu_kernel_benchmarks gpu_kernel_benchmarks.cpp

# Run directly (optional - rapido-collect.py does this automatically)
./gpu_p2p_bandwidth
./gpu_kernel_benchmarks
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
- `-a` or `--all`: Collect all sections (default if no specific flags provided)
- Flags can be combined: `-c -g -n` collects CPU, GPU, and Network only
- When any specific flag is used, only those sections are collected
- When no flags or `-a` is used, all sections are collected
- Note: `-m` flag automatically enables ROCm collection (microbenchmarks need ROCm info)

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
  - Calculates peak performance: FP32, FP64, FP16, BF16, INT8, FP8, FP4 based on architecture
  - If `include_p2p=True`: compiles and runs `gpu_p2p_bandwidth.cpp`, parses JSON output

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

**Rendering functions**:
- `_render_dict_as_table(data)`: Converts dict to HTML table
- `_render_list_as_cards(items)`: Creates card UI for each item (uses "Section" field for title)
- `_render_comparison_section()`: Side-by-side layout for two files
- `_render_single_section()`: Single file layout

**HTML output**: 
- Tabbed interface (CPU, GPU, ROCm, Network, BMC*, Microbenchmarks*)
- Tabs are draggable/reorderable with localStorage persistence
- Conditional BMC and Microbenchmarks tabs based on data availability
- Gradient purple theme with responsive design

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
