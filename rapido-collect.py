#!/usr/bin/env python3
import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from collections import OrderedDict
from typing import Dict, List, Optional

# Default wall-clock limit for a single external command. Without this a single hung tool
# (amd-smi against a wedged GPU, ipmitool against an unresponsive BMC) blocks the whole run.
DEFAULT_COMMAND_TIMEOUT = 60

# Compiling a HIP source and running the GPU benchmarks legitimately take minutes
# (P2P alone is N^2 GPU pairs), so those calls pass these larger limits explicitly.
COMPILE_TIMEOUT = 600
BENCHMARK_TIMEOUT = 1800

# Records why each command failed, so the report can distinguish "tool missing" from
# "tool timed out" from "tool returned an error". Keyed by the command name.
COMMAND_FAILURES: "OrderedDict[str, Dict[str, str]]" = OrderedDict()


def _safe_hostname() -> str:
    """This machine's hostname, or "" if it cannot be determined."""
    try:
        return socket.gethostname() or ""
    except Exception:
        return ""


def _hip_runtime_env() -> Dict[str, str]:
    """Environment for running a locally compiled HIP binary.

    hipcc links against libamdhip64.so from the ROCm tree, but many installs never add
    /opt/rocm/lib to the loader search path (no ld.so.conf.d entry, no LD_LIBRARY_PATH in
    a non-login shell). The binary then dies with "libamdhip64.so.N: cannot open shared
    object file" and every GPU benchmark silently reports no output. Prepending the ROCm
    library directories here makes the benchmarks run regardless of shell setup.
    """
    env = dict(os.environ)
    rocm_path = env.get("ROCM_PATH") or "/opt/rocm"
    candidates = [os.path.join(rocm_path, "lib"), os.path.join(rocm_path, "lib64")]
    libdirs = [d for d in candidates if os.path.isdir(d)]
    if libdirs:
        existing = env.get("LD_LIBRARY_PATH", "")
        parts = libdirs + ([existing] if existing else [])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


def run_command(cmd: List[str], timeout: Optional[int] = DEFAULT_COMMAND_TIMEOUT,
                record: bool = True, env: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Run an external command and return its stdout, or None on any failure.

    Failures are recorded in COMMAND_FAILURES with a reason so the caller (and ultimately the
    report) can tell a missing tool apart from a hung or erroring one. Pass a larger timeout
    for long-running work such as compiling or running the GPU benchmarks.

    Pass record=False for speculative probes (does this tool exist? does it accept --version?)
    whose failure is expected and not worth surfacing in the report. Pass env to override the
    child environment, e.g. _hip_runtime_env() for the compiled HIP benchmarks.
    """
    key = " ".join(cmd)

    def fail(reason: str, detail: str) -> None:
        if record:
            COMMAND_FAILURES[key] = {"reason": reason, "detail": detail}

    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # text=True equivalent for Python 3.6 compatibility
            timeout=timeout,
            env=env,
        )
        # A command that succeeds now supersedes any earlier failure of the same
        # command, so a stale record cannot outlive the condition it described.
        COMMAND_FAILURES.pop(key, None)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        fail("timeout", f"No response after {timeout}s")
        return None
    except FileNotFoundError:
        fail("not_found", f"Command not found: {cmd[0]}")
        return None
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        fail("error", f"Exit code {e.returncode}"
                      + (f": {stderr.splitlines()[0][:200]}" if stderr else ""))
        return None
    except subprocess.SubprocessError as e:
        fail("error", str(e)[:200])
        return None
    except OSError as e:
        # PermissionError, ENOEXEC and friends are OSError but not SubprocessError,
        # and would otherwise escape and abort the whole collection.
        fail("error", f"{type(e).__name__}: {str(e)[:200]}")
        return None


def _amd_smi_json(args: List[str], timeout: int = DEFAULT_COMMAND_TIMEOUT,
                  record: bool = True) -> Optional[object]:
    """Run `amd-smi <args> --json` and return the parsed result, or None.

    amd-smi's JSON output is the only stable contract it offers; the human-readable
    form reflows between releases. Everything that reads amd-smi structurally goes
    through here so the parse and its failure mode live in one place.
    """
    output = run_command(["amd-smi"] + list(args) + ["--json"], timeout=timeout, record=record)
    if not output:
        return None
    try:
        return json.loads(output)
    except (ValueError, TypeError):
        if record:
            COMMAND_FAILURES["amd-smi " + " ".join(args)] = {
                "reason": "error", "detail": "Output was not valid JSON"}
        return None


def _num(value: object) -> Optional[float]:
    """Return value as a number, or None if it is not one.

    amd-smi reports unsupported fields as the *string* "N/A" mixed in among real
    integers (PCIE_BIF and HDP ECC blocks on MI300X, several violation sub-fields).
    Feeding those to int() raises, and testing them for truthiness counts them as
    present-and-zero, which would turn "this block has no ECC support" into "this
    block reports no errors". Both are wrong in ways that matter for an RMA call.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _int(value: object) -> Optional[int]:
    """_num, narrowed to an int. Returns None for non-numeric values."""
    number = _num(value)
    return None if number is None else int(number)


def _read_sysfs(path: str) -> Optional[str]:
    """Contents of a sysfs/procfs file, stripped, or None if it cannot be read.

    Unreadable is the normal case for much of what the platform collector looks at
    (attributes vary by kernel, driver and privilege level), so this never records a
    failure -- the absence of a field is the signal.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def check_tool_availability(verbose: bool = True) -> None:
    """Check and report availability of all required and optional tools."""
    system = platform.system().lower()
    
    if verbose:
        print("=" * 80)
        print("AMD Rapido - Tool Availability Check")
        print("=" * 80)
        print(f"Operating System: {platform.system()} {platform.release()}")
        print(f"Python Version: {platform.python_version()}")
        print()
    
    # Define tools and their impact
    tools_info = {
        "linux": {
            "required": [
                ("python3", "Python 3", "All functionality will fail"),
            ],
            "cpu": [
                ("lscpu", "CPU Information", "Will fallback to /proc/cpuinfo"),
            ],
            "gpu": [
                ("amd-smi", "GPU Information", "GPU details will be incomplete or missing"),
                ("rocminfo", "ROCm GPU Details", "GPU architecture details will be missing"),
            ],
            "network": [
                ("ip", "Network Interfaces", "Network information will be incomplete"),
                ("ethtool", "Network Details", "Driver, speed, and firmware info will be missing"),
            ],
            "bmc": [
                ("ipmitool", "BMC Information", "BMC section will be empty"),
            ],
            "rocm": [
                ("hipcc", "HIP Compiler", "ROCm version info will be incomplete"),
                ("rocm-smi", "ROCm SMI", "ROCm monitoring info will be missing"),
                ("clinfo", "OpenCL Info", "OpenCL details will be missing"),
                ("dpkg", "Package Info (Debian)", "Installed packages list will be incomplete"),
                ("rpm", "Package Info (RHEL)", "Installed packages list will be incomplete"),
            ],
            "microbenchmarks": [
                ("hipcc", "HIP Compiler", "Kernel benchmarks and P2P tests will be skipped"),
            ],
        },
        "windows": {
            "required": [
                ("python", "Python", "All functionality will fail"),
            ],
            "cpu": [
                ("wmic", "CPU Information", "Will fallback to PowerShell"),
                ("powershell", "PowerShell", "CPU information will be limited"),
            ],
            "gpu": [
                ("wmic", "GPU Information", "Will fallback to PowerShell"),
                ("powershell", "PowerShell", "GPU information will be limited"),
            ],
            "network": [
                ("powershell", "PowerShell", "Network information will be missing"),
            ],
        },
        "darwin": {
            "required": [
                ("python3", "Python 3", "All functionality will fail"),
            ],
            "cpu": [
                ("sysctl", "System Info", "CPU information will be missing"),
            ],
            "gpu": [
                ("system_profiler", "System Profiler", "GPU information will be missing"),
            ],
            "network": [
                ("ifconfig", "Network Config", "Network information will be missing"),
                ("netstat", "Network Stats", "Network statistics will be missing"),
            ],
        },
    }
    
    # Get tools for current OS
    os_tools = tools_info.get(system, tools_info.get("linux", {}))
    
    available_tools = []
    missing_tools = []
    
    # Check each category
    for category, tools in os_tools.items():
        if not tools:
            continue
            
        if verbose:
            print(f"\n{category.upper()} Tools:")
            print("-" * 80)
        
        for tool_cmd, tool_name, impact in tools:
            # Check if tool exists. These are probes, not data collection: a tool
            # that is simply absent, or that does not accept --version, is an
            # expected outcome and must not land in COMMAND_FAILURES.
            result = (run_command([tool_cmd, "--version"], record=False) if system != "windows"
                      else run_command([tool_cmd, "/?"], record=False))

            # Some tools don't support --version, try different approaches
            if result is None:
                if tool_cmd in ["ip", "ethtool", "ipmitool"]:
                    result = run_command([tool_cmd], record=False)
                elif tool_cmd == "system_profiler":
                    result = run_command(["which", tool_cmd], record=False)
                elif tool_cmd in ["dpkg", "rpm"]:
                    result = run_command(["which", tool_cmd], record=False)
            
            if verbose:
                status = "[+]" if result is not None else "[x]"
                status_text = "AVAILABLE" if result is not None else "MISSING"
                print(f"  {status} {tool_name:30} [{status_text:9}]  ({tool_cmd})")
            
            if result is None:
                if verbose:
                    print(f"    Impact: {impact}")
                missing_tools.append((category, tool_name, impact, tool_cmd))
            else:
                available_tools.append((category, tool_name, tool_cmd))
    
    # Special checks for files
    if system == "linux" and verbose:
        print(f"\n{'SPECIAL CHECKS'}")
        print("-" * 80)
        
        # Check for GPU benchmark source files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        p2p_cpp = os.path.join(script_dir, "gpu_p2p_bandwidth.cpp")
        kernel_cpp = os.path.join(script_dir, "gpu_kernel_benchmarks.cpp")
        host_cpp = os.path.join(script_dir, "gpu_host_bandwidth.cpp")
        topology_cpp = os.path.join(script_dir, "gpu_topology.cpp")
        
        p2p_exists = os.path.exists(p2p_cpp)
        kernel_exists = os.path.exists(kernel_cpp)
        host_exists = os.path.exists(host_cpp)
        topology_exists = os.path.exists(topology_cpp)
        
        print(f"  {'[+]' if p2p_exists else '[x]'} GPU P2P Benchmark Source    [{'FOUND' if p2p_exists else 'MISSING'}]")
        if not p2p_exists:
            print(f"    Impact: P2P bandwidth tests will be skipped (requires -p flag)")
        
        print(f"  {'[+]' if kernel_exists else '[x]'} Kernel Benchmark Source     [{'FOUND' if kernel_exists else 'MISSING'}]")
        if not kernel_exists:
            print(f"    Impact: Kernel benchmarks will be skipped (requires -m flag)")
        
        print(f"  {'[+]' if host_exists else '[x]'} GPU-CPU Bandwidth Source    [{'FOUND' if host_exists else 'MISSING'}]")
        if not host_exists:
            print(f"    Impact: GPU-CPU transfer bandwidth tests will be skipped (requires -p flag)")
        
        print(f"  {'[+]' if topology_exists else '[x]'} GPU Topology Source        [{'FOUND' if topology_exists else 'MISSING'}]")
        if not topology_exists:
            print(f"    Impact: XGMI/Infinity Fabric topology analysis will be skipped (requires -p flag)")
        
        # Check ROCm installation
        rocm_path_exists = os.path.exists("/opt/rocm")
        print(f"  {'[+]' if rocm_path_exists else '[x]'} ROCm Installation         [{'FOUND' if rocm_path_exists else 'MISSING'}]")
        if not rocm_path_exists:
            print(f"    Impact: ROCm section and GPU benchmarks will be incomplete")
    
    # Summary
    if verbose:
        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Available Tools: {len(available_tools)}")
        print(f"Missing Tools:   {len(missing_tools)}")
        print()
        
        if missing_tools:
            print("AFFECTED REPORT SECTIONS:")
            print("-" * 80)
            
            # Group by category
            category_impacts = {}
            for category, tool_name, impact, tool_cmd in missing_tools:
                if category not in category_impacts:
                    category_impacts[category] = []
                category_impacts[category].append(f"  - {tool_name} ({tool_cmd}): {impact}")
            
            for category, impacts in category_impacts.items():
                print(f"\n{category.upper()}:")
                for impact in impacts:
                    print(impact)
        else:
            print("[+] All tools are available! Full functionality enabled.")
        
        print()
        print("=" * 80)
        print()

def windows_cpu_info() -> Dict[str, str]:
    info = OrderedDict()
    # WMIC is deprecated but still present on many systems; fallback to PowerShell if missing.
    wmic = run_command(["wmic", "cpu", "get", "Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors", "/format:list"])
    if wmic:
        for line in wmic.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                info[key.strip()] = value.strip()
    else:
        ps = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors) "
            "| ConvertTo-Json -Compress"
        ])
        if ps:
            data = json.loads(ps)
            if isinstance(data, list):
                data = data[0]
            for key, value in data.items():
                info[key] = str(value)
    return info

def linux_cpu_info() -> Dict[str, str]:
    info = OrderedDict()
    lscpu = run_command(["lscpu", "--json"])
    if lscpu:
        data = json.loads(lscpu)
        for field in data.get("lscpu", []):
            label = field.get("field", "").rstrip(":")
            value = field.get("data", "").strip()
            if label and value:
                info[label] = value
        return info

    # Fallback to /proc/cpuinfo
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    if key not in info:
                        info[key] = value
    except FileNotFoundError:
        pass
    return info

def mac_cpu_info() -> Dict[str, str]:
    info = OrderedDict()
    sysctl = run_command(["sysctl", "-a"])
    if sysctl:
        for line in sysctl.splitlines():
            if line.startswith(("machdep.cpu.", "hw.phy", "hw.log", "hw.cpufrequency")):
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info

def windows_gpu_info() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    wmic = run_command([
        "wmic",
        "path",
        "win32_VideoController",
        "get",
        "Name,AdapterCompatibility,DriverVersion,AdapterRAM,VideoProcessor,PNPDeviceID",
        "/format:list",
    ])
    if wmic:
        current = OrderedDict()
        for line in wmic.splitlines():
            line = line.strip()
            if not line:
                if current:
                    entries.append(current)
                    current = OrderedDict()
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                current[key.strip()] = value.strip()
        if current:
            entries.append(current)
    else:
        ps = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_VideoController | Select-Object Name,AdapterCompatibility,DriverVersion,AdapterRAM,VideoProcessor,PNPDeviceID) | ConvertTo-Json -Compress",
        ])
        if ps:
            data = json.loads(ps)
            if isinstance(data, dict):
                data = [data]
            for adapter in data:
                record = OrderedDict()
                for key in ("Name", "AdapterCompatibility", "DriverVersion", "AdapterRAM", "VideoProcessor", "PNPDeviceID"):
                    value = adapter.get(key)
                    if value not in (None, ""):
                        record[key] = str(value)
                if record:
                    entries.append(record)
    return entries

def linux_gpu_info() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []

    # Get AMD GPU driver/library version first
    version_info = OrderedDict()
    version_info["Section"] = "AMD Driver and Library Versions"
    amd_smi_version = run_command(["amd-smi", "version", "--json"])
    if amd_smi_version:
        try:
            version_data = json.loads(amd_smi_version)
            if isinstance(version_data, dict):
                for key, value in version_data.items():
                    if value and str(value).strip():
                        version_info[key] = str(value)
        except json.JSONDecodeError as e:
            pass

    if len(version_info) > 1:
        entries.append(version_info)

    # List all GPUs, VFs, and NICs
    device_list_info = OrderedDict()
    device_list_info["Section"] = "Device List (GPUs, VFs, NICs)"
    amd_smi_list = run_command(["amd-smi", "list", "--json"])
    if amd_smi_list:
        try:
            list_data = json.loads(amd_smi_list)
            if isinstance(list_data, dict):
                device_num = 1
                for device_key, device_info in list_data.items():
                    if isinstance(device_info, dict):
                        device_str = f"Device {device_num}: "
                        details = []
                        if "bdf" in device_info:
                            details.append(f"BDF={device_info['bdf']}")
                        if "uuid" in device_info:
                            details.append(f"UUID={device_info['uuid']}")
                        if "device_name" in device_info:
                            details.append(f"Name={device_info['device_name']}")
                        device_list_info[device_str] = ", ".join(details) if details else str(device_info)
                        device_num += 1
        except json.JSONDecodeError:
            pass

    if len(device_list_info) > 1:
        entries.append(device_list_info)

    # AMD GPU detailed information using amd-smi static
    amd_smi_static = run_command(["amd-smi", "static", "--json"])
    parse_error_details = None
    
    if amd_smi_static:
        try:
            data = json.loads(amd_smi_static)

            # Handle both JSON formats:
            # 1. Newer format: direct array of GPU objects
            # 2. Older format: dict with "gpu_data" key containing array
            gpu_list = None
            
            if isinstance(data, list):
                # Newer amd-smi format: JSON is directly an array of GPUs
                gpu_list = data
            elif isinstance(data, dict) and "gpu_data" in data:
                # Older amd-smi format: JSON has "gpu_data" key
                gpu_list = data["gpu_data"]
            elif isinstance(data, dict):
                parse_error_details = f"amd-smi JSON missing 'gpu_data' key. Available keys: {list(data.keys())}"
            else:
                parse_error_details = f"amd-smi returned unexpected JSON type: {type(data).__name__}"
            
            if gpu_list is not None:
                if not gpu_list:
                    parse_error_details = "amd-smi returned empty gpu_data array (no GPUs found)"

                for gpu_data in gpu_list:
                    if not isinstance(gpu_data, dict):
                        continue

                    record = OrderedDict()

                    # Get GPU ID from the gpu field
                    gpu_id = gpu_data.get("gpu", "Unknown")

                    # Section header with GPU ID
                    gpu_name = "Unknown GPU"
                    if "asic" in gpu_data and isinstance(gpu_data["asic"], dict):
                        if "market_name" in gpu_data["asic"]:
                            gpu_name = gpu_data["asic"]["market_name"]

                    record["Section"] = f"{gpu_name} (GPU {gpu_id})"

                    # === ASIC Details ===
                    if "asic" in gpu_data and isinstance(gpu_data["asic"], dict):
                        asic = gpu_data["asic"]
                        if "market_name" in asic:
                            record["GPU Name"] = asic["market_name"]
                        if "vendor_name" in asic:
                            record["Vendor"] = asic["vendor_name"]
                        if "asic_serial" in asic:
                            record["Serial Number"] = asic["asic_serial"]
                        if "target_graphics_version" in asic:
                            record["GFX Architecture"] = asic["target_graphics_version"]
                        if "device_id" in asic:
                            record["Device ID"] = asic["device_id"]
                        if "vendor_id" in asic:
                            record["Vendor ID"] = asic["vendor_id"]
                        if "subsystem_id" in asic:
                            record["Subsystem ID"] = asic["subsystem_id"]
                        if "revision_id" in asic:
                            record["Revision ID"] = asic["revision_id"]
                        if "rev_id" in asic:
                            record["Revision ID"] = asic["rev_id"]
                        if "oam_id" in asic:
                            record["OAM ID"] = str(asic["oam_id"])
                        if "subvendor_id" in asic:
                            record["Subvendor ID"] = asic["subvendor_id"]
                        if "num_compute_units" in asic:
                            record["Compute Units"] = str(asic["num_compute_units"])
                        if "num_shader_engines" in asic:
                            record["Shader Engines"] = str(asic["num_shader_engines"])
                        if "num_shader_arrays_per_engine" in asic:
                            record["Shader Arrays per Engine"] = str(asic["num_shader_arrays_per_engine"])

                    # === Driver Information ===
                    if "driver" in gpu_data and isinstance(gpu_data["driver"], dict):
                        driver = gpu_data["driver"]
                        if "name" in driver:
                            record["Driver Name"] = driver["name"]
                        if "version" in driver:
                            record["Driver Version"] = driver["version"]

                    # === Bus/PCIe Information ===
                    if "bus" in gpu_data and isinstance(gpu_data["bus"], dict):
                        bus = gpu_data["bus"]
                        if "bdf" in bus:
                            record["PCI BDF"] = bus["bdf"]
                        if "max_pcie_width" in bus:
                            record["Max PCIe Link Width"] = f"{bus['max_pcie_width']} lanes"
                        if "max_pcie_speed" in bus:
                            # Handle value/unit structure
                            if isinstance(bus["max_pcie_speed"], dict):
                                value = bus["max_pcie_speed"].get("value", "")
                                unit = bus["max_pcie_speed"].get("unit", "")
                                record["Max PCIe Speed"] = f"{value} {unit}".strip()
                            else:
                                record["Max PCIe Speed"] = str(bus["max_pcie_speed"])
                        if "pcie_interface_version" in bus:
                            record["PCIe Generation"] = bus["pcie_interface_version"]
                        if "slot_type" in bus:
                            record["Slot Type"] = bus["slot_type"]

                    # === VBIOS Information ===
                    if "vbios" in gpu_data and isinstance(gpu_data["vbios"], dict):
                        vbios = gpu_data["vbios"]
                        if "version" in vbios:
                            record["VBIOS Version"] = vbios["version"]
                        if "part_number" in vbios:
                            record["VBIOS Part Number"] = vbios["part_number"]
                        if "build_date" in vbios:
                            record["VBIOS Build Date"] = vbios["build_date"]

                    # === Board Information ===
                    if "board" in gpu_data and isinstance(gpu_data["board"], dict):
                        board = gpu_data["board"]
                        if "model_number" in board:
                            record["Board Model"] = board["model_number"]
                        if "product_serial" in board:
                            record["Board Serial"] = board["product_serial"]
                        if "product_name" in board:
                            record["Board Name"] = board["product_name"]
                        if "manufacturer_name" in board:
                            record["Board Manufacturer"] = board["manufacturer_name"]

                    # === VRAM Information ===
                    if "vram" in gpu_data and isinstance(gpu_data["vram"], dict):
                        vram = gpu_data["vram"]
                        if "size" in vram:
                            # Handle value/unit structure
                            if isinstance(vram["size"], dict):
                                value = vram["size"].get("value", "")
                                unit = vram["size"].get("unit", "")
                                record["VRAM Size"] = f"{value} {unit}".strip()
                            else:
                                try:
                                    vram_mb = int(vram["size"])
                                    record["VRAM Size"] = f"{vram_mb / 1024:.2f} GB"
                                except (ValueError, TypeError):
                                    record["VRAM Size"] = str(vram["size"])
                        if "type" in vram:
                            record["VRAM Type"] = vram["type"]
                        if "vendor" in vram:
                            record["VRAM Vendor"] = vram["vendor"]
                        if "bit_width" in vram:
                            record["VRAM Bit Width"] = f"{vram['bit_width']}-bit"
                        if "max_bandwidth" in vram:
                            # Handle value/unit structure
                            if isinstance(vram["max_bandwidth"], dict):
                                value = vram["max_bandwidth"].get("value", "")
                                unit = vram["max_bandwidth"].get("unit", "")
                                record["VRAM Max Bandwidth"] = f"{value} {unit}".strip()
                            else:
                                record["VRAM Max Bandwidth"] = str(vram["max_bandwidth"])

                    # === Cache Information ===
                    if "cache_info" in gpu_data and isinstance(gpu_data["cache_info"], list):
                        for cache_data in gpu_data["cache_info"]:
                            if isinstance(cache_data, dict):
                                cache_level = cache_data.get("cache_level", "")
                                cache_id = cache_data.get("cache", "")
                                cache_props = cache_data.get("cache_properties", [])

                                # Build cache label
                                if cache_props and isinstance(cache_props, list):
                                    props_str = ", ".join(cache_props)
                                    cache_label = f"L{cache_level} Cache ({props_str})"
                                else:
                                    cache_label = f"L{cache_level} Cache {cache_id}"

                                # Handle cache size with value/unit structure
                                if "cache_size" in cache_data:
                                    if isinstance(cache_data["cache_size"], dict):
                                        value = cache_data["cache_size"].get("value", "")
                                        unit = cache_data["cache_size"].get("unit", "")
                                        record[f"{cache_label} Size"] = f"{value} {unit}".strip()
                                    else:
                                        record[f"{cache_label} Size"] = str(cache_data["cache_size"])

                                if "num_cache_instance" in cache_data:
                                    record[f"{cache_label} Instances"] = str(cache_data["num_cache_instance"])

                    # === Power and Thermal Limits ===
                    if "limit" in gpu_data and isinstance(gpu_data["limit"], dict):
                        limit = gpu_data["limit"]

                        # Power limits
                        for power_field, label in [
                            ("max_power", "Max Power"),
                            ("min_power", "Min Power"),
                            ("socket_power", "Socket Power")
                        ]:
                            if power_field in limit:
                                if isinstance(limit[power_field], dict):
                                    value = limit[power_field].get("value", "")
                                    unit = limit[power_field].get("unit", "")
                                    record[label] = f"{value} {unit}".strip()
                                else:
                                    record[label] = str(limit[power_field])

                        # Temperature limits
                        for temp_field, label in [
                            ("slowdown_edge_temperature", "Slowdown Edge Temperature"),
                            ("slowdown_hotspot_temperature", "Slowdown Hotspot Temperature"),
                            ("slowdown_vram_temperature", "Slowdown VRAM Temperature"),
                            ("shutdown_edge_temperature", "Shutdown Edge Temperature"),
                            ("shutdown_hotspot_temperature", "Shutdown Hotspot Temperature"),
                            ("shutdown_vram_temperature", "Shutdown VRAM Temperature")
                        ]:
                            if temp_field in limit:
                                if isinstance(limit[temp_field], dict):
                                    value = limit[temp_field].get("value", "")
                                    unit = limit[temp_field].get("unit", "")
                                    record[label] = f"{value}°{unit}".strip("°")
                                else:
                                    temp_val = limit[temp_field]
                                    if temp_val != "N/A":
                                        record[label] = str(temp_val)

                    # === NUMA Information ===
                    if "numa" in gpu_data and isinstance(gpu_data["numa"], dict):
                        numa = gpu_data["numa"]
                        if "node" in numa:
                            record["NUMA Node"] = str(numa["node"])
                        if "affinity" in numa:
                            record["NUMA Affinity"] = str(numa["affinity"])
                        if "cpu_affinity" in numa and isinstance(numa["cpu_affinity"], dict):
                            cpu_cores = []
                            for cpu_list_data in numa["cpu_affinity"].values():
                                if isinstance(cpu_list_data, dict) and "cpu_cores_affinity" in cpu_list_data:
                                    cores = cpu_list_data["cpu_cores_affinity"]
                                    if cores != "N/A":
                                        cpu_cores.append(cores)
                            if cpu_cores:
                                record["CPU Cores Affinity"] = ", ".join(cpu_cores)

                    # === Partition Information (SR-IOV) ===
                    if "partition" in gpu_data and isinstance(gpu_data["partition"], dict):
                        partition = gpu_data["partition"]
                        if "partition_id" in partition:
                            record["Partition ID"] = str(partition["partition_id"])
                        if "partition_type" in partition:
                            record["Partition Type"] = partition["partition_type"]
                        if "num_partitions" in partition:
                            record["Number of Partitions"] = str(partition["num_partitions"])

                    # === Firmware Versions ===
                    if "fw_version" in gpu_data and isinstance(gpu_data["fw_version"], dict):
                        fw = gpu_data["fw_version"]
                        for fw_component, fw_version in fw.items():
                            if fw_version and str(fw_version).strip():
                                record[f"FW {fw_component}"] = str(fw_version)

                    # === RAS Information ===
                    if "ras" in gpu_data and isinstance(gpu_data["ras"], dict):
                        ras = gpu_data["ras"]
                        if "eeprom_version" in ras:
                            record["RAS EEPROM Version"] = str(ras["eeprom_version"])
                        if "parity_schema" in ras:
                            record["RAS Parity Schema"] = ras["parity_schema"]
                        if "single_bit_schema" in ras:
                            record["RAS Single-Bit Schema"] = ras["single_bit_schema"]
                        if "double_bit_schema" in ras:
                            record["RAS Double-Bit Schema"] = ras["double_bit_schema"]
                        if "poison_schema" in ras:
                            record["RAS Poison Schema"] = ras["poison_schema"]

                        # ECC block states
                        if "ecc_block_state" in ras and isinstance(ras["ecc_block_state"], dict):
                            ecc_enabled = []
                            ecc_disabled = []
                            for block_name, block_state in ras["ecc_block_state"].items():
                                if block_state == "ENABLED":
                                    ecc_enabled.append(block_name)
                                elif block_state == "DISABLED":
                                    ecc_disabled.append(block_name)
                            if ecc_enabled:
                                record["ECC Enabled Blocks"] = ", ".join(ecc_enabled)
                            if ecc_disabled:
                                record["ECC Disabled Blocks"] = ", ".join(ecc_disabled)

                    # === Additional GPU Features ===
                    if "process_isolation" in gpu_data:
                        process_iso = gpu_data["process_isolation"]
                        if process_iso and process_iso != "N/A":
                            record["Process Isolation"] = process_iso

                    if "soc_pstate" in gpu_data:
                        soc_pstate = gpu_data["soc_pstate"]
                        if soc_pstate and soc_pstate != "N/A":
                            record["SoC P-State"] = soc_pstate

                    if "xgmi_plpd" in gpu_data:
                        xgmi_plpd = gpu_data["xgmi_plpd"]
                        if xgmi_plpd and xgmi_plpd != "N/A":
                            record["XGMI PLPD"] = xgmi_plpd

                    entries.append(record)
            elif isinstance(data, dict):
                parse_error_details = f"amd-smi JSON missing 'gpu_data' key. Available keys: {list(data.keys())}"
            else:
                parse_error_details = f"amd-smi returned unexpected JSON type: {type(data).__name__}"
        except json.JSONDecodeError as e:
            parse_error_details = f"JSON parsing failed: {e.msg} at line {e.lineno}, column {e.colno}"
        except Exception as e:
            parse_error_details = f"Unexpected error parsing amd-smi output: {str(e)}"

    # === XGMI/Topology Information ===
    xgmi_info = OrderedDict()
    xgmi_info["Section"] = "XGMI Topology and Interconnect"
    amd_smi_topology = run_command(["amd-smi", "topology", "--json"])
    if amd_smi_topology:
        try:
            topo_data = json.loads(amd_smi_topology)
            if isinstance(topo_data, dict):
                for gpu_id, topo_info in topo_data.items():
                    if isinstance(topo_info, dict):
                        # XGMI links information
                        if "xgmi" in topo_info and isinstance(topo_info["xgmi"], dict):
                            xgmi = topo_info["xgmi"]
                            gpu_label = f"{gpu_id} XGMI"
                            xgmi_details = []
                            if "num_hops" in xgmi:
                                xgmi_details.append(f"Hops: {xgmi['num_hops']}")
                            if "link_type" in xgmi:
                                xgmi_details.append(f"Type: {xgmi['link_type']}")
                            if "link_count" in xgmi:
                                xgmi_details.append(f"Links: {xgmi['link_count']}")
                            if "bandwidth" in xgmi:
                                xgmi_details.append(f"Bandwidth: {xgmi['bandwidth']}")
                            if xgmi_details:
                                xgmi_info[gpu_label] = ", ".join(xgmi_details)

                        # Access table (which GPUs can access each other)
                        if "access_table" in topo_info and isinstance(topo_info["access_table"], dict):
                            for peer_gpu, access_type in topo_info["access_table"].items():
                                xgmi_info[f"{gpu_id} → {peer_gpu}"] = str(access_type)

                        # Weight table (topology distance/cost)
                        if "weight" in topo_info and isinstance(topo_info["weight"], dict):
                            for peer_gpu, weight in topo_info["weight"].items():
                                xgmi_info[f"{gpu_id} ↔ {peer_gpu} Weight"] = str(weight)
        except json.JSONDecodeError:
            pass

    if len(xgmi_info) > 1:
        entries.append(xgmi_info)

    # === Firmware Information ===
    fw_info = OrderedDict()
    fw_info["Section"] = "Firmware and Microcode Versions"
    amd_smi_firmware = run_command(["amd-smi", "firmware", "--json"])
    if amd_smi_firmware:
        try:
            fw_data = json.loads(amd_smi_firmware)
            if isinstance(fw_data, dict):
                for gpu_id, fw_versions in fw_data.items():
                    if isinstance(fw_versions, dict):
                        for fw_component, version in fw_versions.items():
                            if version and str(version).strip():
                                fw_info[f"{gpu_id} {fw_component}"] = str(version)
        except json.JSONDecodeError:
            pass

    if len(fw_info) > 1:
        entries.append(fw_info)

    # Add error info if no amd-smi data was collected
    if not entries:
        error_info = OrderedDict()
        error_info["Section"] = "GPU Information - Error"
        if not amd_smi_static:
            error_info["Message"] = "amd-smi static returned no data. Please ensure ROCm and amd-smi are installed."
        elif parse_error_details:
            error_info["Message"] = f"Failed to parse amd-smi output: {parse_error_details}"
            error_info["Troubleshooting"] = "Try running 'amd-smi static --json' manually to see the raw output"
        else:
            error_info["Message"] = "amd-smi commands ran but no GPU data was found"
            error_info["Possible Causes"] = "No AMD GPUs detected, or GPU data was filtered out"
        entries.append(error_info)

    return entries

def mac_gpu_info() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    profiler = run_command(["system_profiler", "SPDisplaysDataType", "-json"])
    if profiler:
        try:
            data = json.loads(profiler)
            for adapter in data.get("SPDisplaysDataType", []):
                record = OrderedDict()
                for source_key, target_key in (
                    ("sppci_model", "Model"),
                    ("spdisplays_vendor", "Vendor"),
                    ("spdisplays_vram", "VRAM"),
                    ("spdisplays_metal", "Metal"),
                    ("spdisplays_bus", "Bus"),
                ):
                    value = adapter.get(source_key)
                    if value:
                        record[target_key] = str(value)
                if record:
                    entries.append(record)
        except json.JSONDecodeError:
            pass
    if not entries:
        profiler_text = run_command(["system_profiler", "SPDisplaysDataType"])
        if profiler_text:
            record = OrderedDict()
            for line in profiler_text.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()
                if key.lower() in {"chipset model", "vendor", "bus", "vram", "metal"}:
                    record[key] = value
            if record:
                entries.append(record)
    return entries

def generic_gpu_info() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    env_record = OrderedDict()
    for env_key in ("CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        value = os.environ.get(env_key)
        if value:
            env_record[env_key] = value
    if env_record:
        entries.append(env_record)
    return entries

def gather_cpu_details() -> Dict[str, Dict[str, str]]:
    system = platform.system().lower()
    details = OrderedDict()
    if system == "windows":
        details["windows"] = windows_cpu_info()
    elif system == "linux":
        details["linux"] = linux_cpu_info()
    elif system == "darwin":
        details["macos"] = mac_cpu_info()
    return details

def _gpu_data_rows(payload: object) -> List[Dict]:
    """Normalise amd-smi's per-GPU JSON into a list of dicts.

    Most subcommands wrap their rows in {"gpu_data": [...]}, but some (bad-pages)
    return the bare list, and a single-GPU query can return a bare dict. Callers
    should not have to care which shape they got.
    """
    if isinstance(payload, dict):
        rows = payload.get("gpu_data", payload)
    else:
        rows = payload
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _gpu_index(row: Dict, fallback: int) -> int:
    """The GPU index a row describes, falling back to its position."""
    index = _int(row.get("gpu"))
    return fallback if index is None else index


def _amd_smi_value(entry: object) -> Optional[float]:
    """Unwrap amd-smi's {"value": N, "unit": "W"} wrapper to the number.

    Fields switch between the wrapped form and a bare scalar depending on the
    subcommand, and unsupported ones are the string "N/A", so all three go through
    _num and come out as either a number or None.
    """
    if isinstance(entry, dict):
        return _num(entry.get("value"))
    return _num(entry)


def _amd_smi_unit(entry: object, default: str = "") -> str:
    """The unit string from amd-smi's {"value", "unit"} wrapper."""
    if isinstance(entry, dict):
        unit = entry.get("unit")
        if isinstance(unit, str) and unit:
            return unit
    return default


# ECC blocks amd-smi enumerates on Instinct parts. Blocks that report "N/A" are
# unsupported on the ASIC, which is different from "supported and reporting zero" --
# conflating the two would invent clean bills of health the hardware never gave.
_ECC_COUNT_KINDS = [
    ("correctable_count", "Correctable"),
    ("uncorrectable_count", "Uncorrectable"),
    ("deferred_count", "Deferred"),
]


def _bad_page_count(entry: object) -> Optional[int]:
    """Number of bad pages in an amd-smi bad-pages field.

    The clean case is not an empty list: amd-smi returns the literal string
    "No bad pages found.". A list is the populated case, one entry per page.
    """
    if isinstance(entry, list):
        return len(entry)
    if isinstance(entry, str):
        return 0 if "no bad pages" in entry.lower() else None
    return _int(entry)


def gather_ras_health() -> Dict[str, List[Dict[str, str]]]:
    """ECC counters, retired pages and XGMI link errors, per GPU plus a node summary.

    This is the data an acceptance audit turns down a node over: uncorrectable ECC
    errors and retired memory pages are the documented RMA triggers, and until now
    rapido could not see either. Correctable counts are reported but not flagged --
    on HBM they are expected and corrected in hardware.
    """
    if platform.system().lower() != "linux":
        return OrderedDict()

    ecc = _gpu_data_rows(_amd_smi_json(["metric", "--ecc"]))
    blocks = _gpu_data_rows(_amd_smi_json(["metric", "--ecc-blocks"]))
    pages = _gpu_data_rows(_amd_smi_json(["bad-pages"]))
    xgmi_err = _gpu_data_rows(_amd_smi_json(["metric", "--xgmi-err"]))
    if not (ecc or blocks or pages or xgmi_err):
        return OrderedDict()

    def by_gpu(rows: List[Dict]) -> Dict[int, Dict]:
        return {_gpu_index(row, position): row for position, row in enumerate(rows)}

    ecc_by_gpu = by_gpu(ecc)
    blocks_by_gpu = by_gpu(blocks)
    pages_by_gpu = by_gpu(pages)
    xgmi_by_gpu = by_gpu(xgmi_err)

    gpu_ids = sorted(set(ecc_by_gpu) | set(blocks_by_gpu) | set(pages_by_gpu) | set(xgmi_by_gpu))
    entries: List[Dict[str, str]] = []

    node_uncorrectable = 0
    node_correctable = 0
    node_deferred = 0
    node_retired = 0
    node_pending = 0
    unhealthy_gpus: List[str] = []

    for gpu_id in gpu_ids:
        card: Dict[str, str] = OrderedDict()
        card["Section"] = f"GPU {gpu_id} - RAS Health"
        raw: Dict[str, object] = {"gpu_id": gpu_id}

        totals = ecc_by_gpu.get(gpu_id, {}).get("ecc")
        totals = totals if isinstance(totals, dict) else {}
        correctable = _int(totals.get("total_correctable_count"))
        uncorrectable = _int(totals.get("total_uncorrectable_count"))
        deferred = _int(totals.get("total_deferred_count"))
        if correctable is not None:
            card["Total Correctable Errors"] = str(correctable)
            raw["ecc_correctable"] = correctable
            node_correctable += correctable
        if uncorrectable is not None:
            card["Total Uncorrectable Errors"] = str(uncorrectable)
            raw["ecc_uncorrectable"] = uncorrectable
            node_uncorrectable += uncorrectable
        if deferred is not None:
            card["Total Deferred Errors"] = str(deferred)
            raw["ecc_deferred"] = deferred
            node_deferred += deferred

        block_data = blocks_by_gpu.get(gpu_id, {}).get("ecc_blocks")
        if isinstance(block_data, dict):
            unsupported: List[str] = []
            for block_name, counts in block_data.items():
                if not isinstance(counts, dict):
                    continue
                parts = []
                for key, label in _ECC_COUNT_KINDS:
                    count = _int(counts.get(key))
                    if count is not None:
                        parts.append(f"{label} {count}")
                        raw[f"ecc_{str(block_name).lower()}_{key}"] = count
                if parts:
                    card[f"ECC Block {block_name}"] = ", ".join(parts)
                else:
                    # Every count was "N/A": the ASIC does not instrument this block.
                    unsupported.append(str(block_name))
            if unsupported:
                card["ECC Blocks Not Instrumented"] = ", ".join(sorted(unsupported))

        page_row = pages_by_gpu.get(gpu_id, {})
        retired = _bad_page_count(page_row.get("retired"))
        pending = _bad_page_count(page_row.get("pending"))
        unreserved = _bad_page_count(page_row.get("un_res"))
        if retired is not None:
            card["Retired Memory Pages"] = str(retired)
            raw["bad_pages_retired"] = retired
            node_retired += retired
        if pending is not None:
            card["Pending Memory Pages"] = str(pending)
            raw["bad_pages_pending"] = pending
            node_pending += pending
        if unreserved is not None:
            card["Unreservable Memory Pages"] = str(unreserved)
            raw["bad_pages_unreservable"] = unreserved

        xgmi_value = xgmi_by_gpu.get(gpu_id, {}).get("xgmi_err")
        xgmi_count = _int(xgmi_value)
        if xgmi_count is not None:
            card["XGMI Link Errors"] = str(xgmi_count)
            raw["xgmi_errors"] = xgmi_count
        elif xgmi_value is not None:
            card["XGMI Link Errors"] = "Not reported by driver"

        flags = []
        if uncorrectable:
            flags.append(f"{uncorrectable} uncorrectable ECC error(s)")
        if retired:
            flags.append(f"{retired} retired page(s)")
        if pending:
            flags.append(f"{pending} pending page(s)")
        if xgmi_count:
            flags.append(f"{xgmi_count} XGMI error(s)")
        if flags:
            card["Health"] = "ATTENTION: " + "; ".join(flags)
            unhealthy_gpus.append(f"GPU {gpu_id} ({'; '.join(flags)})")
            raw["healthy"] = False
        else:
            card["Health"] = "OK"
            raw["healthy"] = True

        if len(card) > 1:
            card["_raw"] = raw
            entries.append(card)

    if entries:
        summary: Dict[str, str] = OrderedDict()
        summary["Section"] = "RAS Health Summary"
        summary["GPUs Checked"] = str(len(entries))
        summary["Node Correctable Errors"] = str(node_correctable)
        summary["Node Uncorrectable Errors"] = str(node_uncorrectable)
        summary["Node Deferred Errors"] = str(node_deferred)
        summary["Node Retired Pages"] = str(node_retired)
        summary["Node Pending Pages"] = str(node_pending)
        if unhealthy_gpus:
            summary["Status"] = ("ATTENTION: uncorrectable errors or bad pages present; "
                                 "these are RMA indicators")
            summary["Affected GPUs"] = "; ".join(unhealthy_gpus)
        else:
            summary["Status"] = "OK - no uncorrectable errors and no bad pages"
        summary["_raw"] = {
            "gpus_checked": len(entries),
            "node_ecc_correctable": node_correctable,
            "node_ecc_uncorrectable": node_uncorrectable,
            "node_ecc_deferred": node_deferred,
            "node_bad_pages_retired": node_retired,
            "node_bad_pages_pending": node_pending,
            "healthy": not unhealthy_gpus,
        }
        entries.insert(0, summary)

    return OrderedDict([("linux", entries)]) if entries else OrderedDict()


def _num_prefix(value: object) -> Optional[float]:
    """Leading number of a string like "16 lanes" or "32 GT/s", else None.

    linux_gpu_info() stores the static PCIe maximums already formatted with their
    units, and those strings are the comparison baseline for the live link state.
    """
    number = _num(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    digits = ""
    for char in value.strip():
        if char.isdigit() or (char == "." and "." not in digits):
            digits += char
        else:
            break
    return _num(digits) if digits else None


def _gpu_id_from_section(section: object) -> Optional[int]:
    """GPU index out of a card's Section label.

    Covers both shapes rapido produces: "GPU 3 - Telemetry" and linux_gpu_info()'s
    "AMD Instinct MI300X (GPU 3)".
    """
    if not isinstance(section, str):
        return None
    words = section.replace("(", " ").replace(")", " ").split()
    for position, word in enumerate(words):
        if word.upper() == "GPU" and position + 1 < len(words):
            index = _int(words[position + 1].rstrip(":-"))
            if index is not None:
                return index
    return None


def _pcie_speed_gts(value: object) -> Optional[float]:
    """PCIe speed in GT/s from amd-smi's wrapped value or from a string like "32 GT/s"."""
    number = _amd_smi_value(value)
    if number is not None:
        return number
    return _num_prefix(value)


def _max_pcie_by_gpu(gpu_details: Optional[Dict[str, List[Dict[str, str]]]]) -> Dict[int, Dict[str, float]]:
    """Per-GPU maximum PCIe width/speed already captured by linux_gpu_info().

    Reusing the static maximums avoids a second amd-smi call and, more importantly,
    gives the current-vs-maximum comparison that turns a bare "x16" into a verdict.
    """
    maxima: Dict[int, Dict[str, float]] = {}
    if not isinstance(gpu_details, dict):
        return maxima
    for entries in gpu_details.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            index = _gpu_id_from_section(entry.get("Section"))
            if index is None:
                continue
            record: Dict[str, float] = {}
            # linux_gpu_info() formats these as "16 lanes" and "32 GT/s"; both
            # helpers pull the leading number back out.
            width = _num_prefix(entry.get("Max PCIe Link Width"))
            speed = _num_prefix(entry.get("Max PCIe Speed"))
            if width is not None:
                record["width"] = width
            if speed is not None:
                record["speed"] = speed
            if record:
                maxima.setdefault(index, {}).update(record)
    return maxima


def gather_gpu_telemetry(gpu_details: Optional[Dict[str, List[Dict[str, str]]]] = None
                         ) -> Dict[str, List[Dict[str, str]]]:
    """Point-in-time power, thermal, clock, PCIe link and throttle state per GPU.

    The point is not the readings themselves but two derived verdicts the readings
    make possible: whether the PCIe link trained to its full width and speed, and
    whether the GPU is currently in violation of a power or thermal limit. A link
    that trained to x8 in an x16 slot is a seated-card defect that no static
    inventory of "Max PCIe Width: 16" can reveal.

    One targeted amd-smi invocation per metric group, with selector flags. amd-smi is
    itself a Python program with a several-hundred-millisecond startup, and an
    unselected `metric` call dumps every group including a full ECC enumeration, so
    polling here would cost far more than the snapshot is worth.
    """
    if platform.system().lower() != "linux":
        return OrderedDict()

    readings = _gpu_data_rows(_amd_smi_json(
        ["metric", "--power", "--temperature", "--clock", "--usage"]))
    pcie = _gpu_data_rows(_amd_smi_json(["metric", "--pcie"]))
    violation = _gpu_data_rows(_amd_smi_json(["metric", "--violation"]))
    if not (readings or pcie or violation):
        return OrderedDict()

    def by_gpu(rows: List[Dict]) -> Dict[int, Dict]:
        return {_gpu_index(row, position): row for position, row in enumerate(rows)}

    readings_by_gpu = by_gpu(readings)
    pcie_by_gpu = by_gpu(pcie)
    violation_by_gpu = by_gpu(violation)
    maxima = _max_pcie_by_gpu(gpu_details)

    entries: List[Dict[str, str]] = []
    degraded_links: List[str] = []
    active_violations: List[str] = []

    for gpu_id in sorted(set(readings_by_gpu) | set(pcie_by_gpu) | set(violation_by_gpu)):
        card: Dict[str, str] = OrderedDict()
        card["Section"] = f"GPU {gpu_id} - Telemetry"
        raw: Dict[str, object] = {"gpu_id": gpu_id}

        row = readings_by_gpu.get(gpu_id, {})

        power = row.get("power")
        if isinstance(power, dict):
            socket_power = _amd_smi_value(power.get("socket_power"))
            if socket_power is not None:
                unit = _amd_smi_unit(power.get("socket_power"), "W")
                card["Socket Power"] = f"{socket_power:g} {unit}"
                raw["socket_power_w"] = socket_power
            management = power.get("power_management")
            if isinstance(management, str) and management:
                card["Power Management"] = management

        temperature = row.get("temperature")
        if isinstance(temperature, dict):
            for key, label in (("hotspot", "Hotspot Temperature"),
                               ("edge", "Edge Temperature"),
                               ("mem", "Memory Temperature")):
                value = _amd_smi_value(temperature.get(key))
                if value is not None:
                    unit = _amd_smi_unit(temperature.get(key), "C")
                    card[label] = f"{value:g} {unit}"
                    raw[f"temp_{key}_c"] = value

        # MI300 exposes one clock domain per XCD (gfx_0..gfx_7). Reporting eight
        # near-identical rows buries the reading, so summarise the range instead.
        clock = row.get("clock")
        if isinstance(clock, dict):
            gfx_clocks = []
            max_clocks = []
            locked = set()
            for domain, info in clock.items():
                if not (isinstance(domain, str) and domain.startswith("gfx") and isinstance(info, dict)):
                    continue
                current = _amd_smi_value(info.get("clk"))
                ceiling = _amd_smi_value(info.get("max_clk"))
                if current is not None:
                    gfx_clocks.append(current)
                if ceiling is not None:
                    max_clocks.append(ceiling)
                state = info.get("clk_locked")
                if isinstance(state, str) and state:
                    locked.add(state)
            if gfx_clocks:
                low, high = min(gfx_clocks), max(gfx_clocks)
                card["GFX Clock"] = (f"{low:g} MHz" if low == high
                                     else f"{low:g}-{high:g} MHz across {len(gfx_clocks)} domains")
                raw["gfx_clock_min_mhz"] = low
                raw["gfx_clock_max_mhz"] = high
            if max_clocks:
                card["GFX Clock Limit"] = f"{max(max_clocks):g} MHz"
                raw["gfx_clock_limit_mhz"] = max(max_clocks)
            if locked:
                card["Clock Locked"] = ", ".join(sorted(locked))

        usage = row.get("usage")
        if isinstance(usage, dict):
            for key, label in (("gfx_activity", "GFX Utilization"),
                               ("umc_activity", "Memory Utilization")):
                value = _amd_smi_value(usage.get(key))
                if value is not None:
                    card[label] = f"{value:g} %"
                    raw[key] = value

        link = pcie_by_gpu.get(gpu_id, {}).get("pcie")
        if isinstance(link, dict):
            width = _int(link.get("width"))
            speed = _pcie_speed_gts(link.get("speed"))
            ceiling = maxima.get(gpu_id, {})
            if width is not None:
                card["PCIe Width (Current)"] = f"x{width}"
                raw["pcie_width"] = width
            if speed is not None:
                card["PCIe Speed (Current)"] = f"{speed:g} GT/s"
                raw["pcie_speed_gts"] = speed

            # Caveat worth knowing before trusting a DEGRADED verdict: some
            # platforms downtrain an idle link to save power, so a shortfall seen
            # on an otherwise idle GPU is worth re-reading under load before it is
            # treated as a seating defect. MI300X did not downtrain when measured.
            shortfalls = []
            max_width = ceiling.get("width")
            max_speed = ceiling.get("speed")
            if width is not None and max_width and width < max_width:
                shortfalls.append(f"width x{width} of x{max_width:g}")
            if speed is not None and max_speed and speed < max_speed:
                shortfalls.append(f"speed {speed:g} of {max_speed:g} GT/s")
            if shortfalls:
                card["PCIe Link"] = "DEGRADED: " + ", ".join(shortfalls)
                degraded_links.append(f"GPU {gpu_id}: {', '.join(shortfalls)}")
                raw["pcie_degraded"] = True
            elif width is not None or speed is not None:
                card["PCIe Link"] = ("OK - trained to maximum" if (max_width or max_speed)
                                     else "OK - no maximum reported for comparison")
                raw["pcie_degraded"] = False

            # Replays and NAKs accumulate only on signal-integrity problems, so
            # unlike the throttle accumulators any nonzero value here is notable.
            for key, label in (("replay_count", "PCIe Replay Count"),
                               ("nak_sent_count", "PCIe NAKs Sent"),
                               ("nak_received_count", "PCIe NAKs Received"),
                               ("l0_to_recovery_count", "PCIe L0-to-Recovery Count")):
                count = _int(link.get(key))
                if count is not None:
                    card[label] = str(count)
                    raw[key] = count

        throttle = violation_by_gpu.get(gpu_id, {}).get("throttle")
        if isinstance(throttle, dict):
            # The *_accumulated counters are lifetime residency totals and are
            # routinely nonzero on a perfectly healthy GPU (a board that has ever
            # touched its power limit has a nonzero ppt_accumulated forever), so
            # they are reported as context. The *_violation_status fields are the
            # actual present-tense signal and are the only thing flagged.
            active = []
            for key, value in throttle.items():
                if not (isinstance(key, str) and key.endswith("_violation_status")):
                    continue
                if isinstance(value, str) and value.strip().upper() not in ("NOT ACTIVE", "N/A", ""):
                    active.append(key[:-len("_violation_status")].replace("_", " "))
            for key, label in (("ppt_accumulated", "Power Limit Residency"),
                               ("socket_thermal_accumulated", "Socket Thermal Residency"),
                               ("hbm_thermal_accumulated", "HBM Thermal Residency"),
                               ("vr_thermal_accumulated", "VR Thermal Residency"),
                               ("prochot_accumulated", "PROCHOT Residency")):
                count = _int(throttle.get(key))
                if count is not None:
                    card[label] = str(count)
                    raw[key] = count
            if active:
                card["Throttling"] = "ACTIVE: " + ", ".join(sorted(active))
                active_violations.append(f"GPU {gpu_id}: {', '.join(sorted(active))}")
                raw["throttling_active"] = True
            else:
                card["Throttling"] = "None active"
                raw["throttling_active"] = False

        if len(card) > 1:
            card["_raw"] = raw
            entries.append(card)

    if entries:
        summary: Dict[str, str] = OrderedDict()
        summary["Section"] = "GPU Telemetry Summary"
        summary["GPUs Sampled"] = str(len(entries))
        summary["Degraded PCIe Links"] = ("; ".join(degraded_links) if degraded_links
                                          else "None - all links trained to maximum")
        summary["Active Throttling"] = ("; ".join(active_violations) if active_violations
                                        else "None")
        summary["Note"] = ("Point-in-time snapshot taken during collection, not a "
                           "sustained-load measurement")
        summary["_raw"] = {
            "gpus_sampled": len(entries),
            "degraded_pcie_links": len(degraded_links),
            "active_throttle_violations": len(active_violations),
        }
        entries.insert(0, summary)

    return OrderedDict([("linux", entries)]) if entries else OrderedDict()


def _gpu_partition_entries() -> List[Dict[str, str]]:
    """Accelerator (SPX/DPX/CPX) and memory (NPS) partition mode per GPU.

    Partition mode is recorded as context, not as a correction to anything: rocminfo
    enumerates one HSA agent per partition, so in CPX each agent already reports its
    own CU count and the peak-performance math needs no rescaling. What this does
    explain is why a CPX node appears to have more GPUs with fewer CUs each -- which
    otherwise reads as a hardware discrepancy on a comparison report.
    """
    if platform.system().lower() != "linux":
        return []

    # `static --partition` first: `partition --json` only exists on newer amd-smi builds.
    rows = _gpu_data_rows(_amd_smi_json(["static", "--partition"], record=False))
    if not any(isinstance(row.get("partition"), dict) for row in rows):
        rows = _gpu_data_rows(_amd_smi_json(["partition"], record=False))
    if not rows:
        return []

    entries: List[Dict[str, str]] = []
    modes = set()
    memory_modes = set()

    for position, row in enumerate(rows):
        info = row.get("partition")
        if not isinstance(info, dict):
            continue
        gpu_id = _gpu_index(row, position)
        card: Dict[str, str] = OrderedDict()
        card["Section"] = f"GPU {gpu_id} - Partition Mode"
        raw: Dict[str, object] = {"gpu_id": gpu_id}

        accelerator = info.get("accelerator_partition") or info.get("compute_partition")
        memory = info.get("memory_partition")
        if isinstance(accelerator, str) and accelerator:
            card["Accelerator Partition"] = accelerator
            raw["accelerator_partition"] = accelerator
            modes.add(accelerator)
        if isinstance(memory, str) and memory:
            card["Memory Partition"] = memory
            raw["memory_partition"] = memory
            memory_modes.add(memory)
        partition_id = _int(info.get("partition_id"))
        if partition_id is not None:
            card["Partition ID"] = str(partition_id)
            raw["partition_id"] = partition_id
        alloc = info.get("compute_partition_mem_alloc_mode")
        if isinstance(alloc, str) and alloc:
            card["Memory Allocation Mode"] = alloc

        if len(card) > 1:
            card["_raw"] = raw
            entries.append(card)

    if entries:
        summary: Dict[str, str] = OrderedDict()
        summary["Section"] = "GPU Partitioning Summary"
        summary["Partitioned Devices"] = str(len(entries))
        if modes:
            summary["Accelerator Modes"] = ", ".join(sorted(modes))
        if memory_modes:
            summary["Memory Modes"] = ", ".join(sorted(memory_modes))
        # A node running two different modes at once is legal but almost never
        # intentional, and it invalidates any GPU-to-GPU comparison on the node.
        if len(modes) > 1 or len(memory_modes) > 1:
            summary["Status"] = ("ATTENTION: partition modes are not uniform across "
                                 "the node; per-GPU results are not comparable")
        else:
            summary["Status"] = "Uniform across all devices"
        summary["Note"] = ("Each partition is enumerated as its own device, so reported "
                           "CU counts and peak figures are already per-partition")
        summary["_raw"] = {
            "partitioned_devices": len(entries),
            "accelerator_modes": sorted(modes),
            "memory_modes": sorted(memory_modes),
            "uniform": len(modes) <= 1 and len(memory_modes) <= 1,
        }
        entries.insert(0, summary)

    return entries


def _platform_firmware_entry() -> Optional[Dict[str, str]]:
    """BIOS and chassis identity from /sys/class/dmi/id.

    dmidecode would give more (per-DIMM population, for one) but needs root, and
    prompting for a password would hang an unattended collection. Everything here is
    world-readable, so the common case -- "which BIOS is this node on?" -- is answered
    without privilege. The fields root-only DMI would have added are listed as absent
    rather than silently omitted.
    """
    fields = [
        ("bios_vendor", "BIOS Vendor"),
        ("bios_version", "BIOS Version"),
        ("bios_date", "BIOS Date"),
        ("sys_vendor", "System Vendor"),
        ("product_name", "System Model"),
        ("product_serial", "System Serial"),
        ("board_vendor", "Baseboard Vendor"),
        ("board_name", "Baseboard Model"),
        ("chassis_vendor", "Chassis Vendor"),
    ]
    record: Dict[str, str] = OrderedDict()
    record["Section"] = "Platform Firmware"
    for name, label in fields:
        value = _read_sysfs("/sys/class/dmi/id/" + name)
        # Unpopulated DMI strings are these placeholders far more often than not.
        if value and value.lower() not in ("to be filled by o.e.m.", "not specified",
                                           "default string", "none", "unknown"):
            record[label] = value
    if len(record) == 1:
        return None
    record["DMI Detail Level"] = ("Unprivileged sysfs only; per-DIMM population and "
                                  "slot inventory require root and were not collected")
    return record


def _platform_memory_entry() -> Optional[Dict[str, str]]:
    """Memory controllers and their EDAC error counts.

    EDAC is the host-side counterpart of the GPU ECC counters: correctable and
    uncorrectable DIMM errors, readable without privilege.
    """
    base = "/sys/devices/system/edac/mc"
    try:
        controllers = sorted(name for name in os.listdir(base) if name.startswith("mc"))
    except OSError:
        return None
    if not controllers:
        return None

    record: Dict[str, str] = OrderedDict()
    record["Section"] = "Platform Memory (EDAC)"
    record["Memory Controllers"] = str(len(controllers))
    raw: Dict[str, object] = {"controllers": len(controllers)}

    total_ce = 0
    total_ue = 0
    seen = False
    for controller in controllers:
        correctable = _int(_read_sysfs(os.path.join(base, controller, "ce_count")))
        uncorrectable = _int(_read_sysfs(os.path.join(base, controller, "ue_count")))
        if correctable is None and uncorrectable is None:
            continue
        seen = True
        total_ce += correctable or 0
        total_ue += uncorrectable or 0
        record[f"{controller} Errors"] = (f"Correctable {correctable if correctable is not None else '?'}, "
                                          f"Uncorrectable {uncorrectable if uncorrectable is not None else '?'}")
    if seen:
        record["Total Correctable Errors"] = str(total_ce)
        record["Total Uncorrectable Errors"] = str(total_ue)
        raw["edac_correctable"] = total_ce
        raw["edac_uncorrectable"] = total_ue
        record["Status"] = ("ATTENTION: host memory uncorrectable errors present"
                            if total_ue else "OK - no host memory uncorrectable errors")
        raw["healthy"] = not total_ue

    size = _read_sysfs(os.path.join(base, controllers[0], "size_mb"))
    if size:
        record["Controller 0 Size"] = f"{size} MB"

    record["_raw"] = raw
    return record


def _platform_numa_entry() -> Optional[Dict[str, str]]:
    """NUMA node inventory and the inter-node distance matrix.

    Distances matter on these hosts because a GPU's memory-staging buffer allocated
    on the far node pays the penalty on every host-to-device transfer.
    """
    base = "/sys/devices/system/node"
    try:
        nodes = sorted((name for name in os.listdir(base)
                        if name.startswith("node") and name[4:].isdigit()),
                       key=lambda name: int(name[4:]))
    except OSError:
        return None
    if not nodes:
        return None

    record: Dict[str, str] = OrderedDict()
    record["Section"] = "Platform NUMA Topology"
    record["NUMA Nodes"] = str(len(nodes))
    raw: Dict[str, object] = {"numa_nodes": len(nodes)}

    distances: List[List[int]] = []
    for node in nodes:
        cpulist = _read_sysfs(os.path.join(base, node, "cpulist"))
        if cpulist:
            record[f"{node} CPUs"] = cpulist
        meminfo = _read_sysfs(os.path.join(base, node, "meminfo"))
        if meminfo:
            for line in meminfo.splitlines():
                if "MemTotal" in line:
                    record[f"{node} Memory"] = line.split(":", 1)[1].strip()
                    break
        distance = _read_sysfs(os.path.join(base, node, "distance"))
        if distance:
            row = [value for value in (_int(field) for field in distance.split())
                   if value is not None]
            if row:
                distances.append(row)
                record[f"{node} Distances"] = " ".join(str(value) for value in row)
    if distances:
        raw["numa_distances"] = distances
        remote = [value for index, row in enumerate(distances)
                  for position, value in enumerate(row) if index != position]
        if remote:
            raw["numa_max_distance"] = max(remote)

    record["_raw"] = raw
    return record


def _platform_tuning_entry() -> Optional[Dict[str, str]]:
    """Kernel and CPU settings that shape GPU host-side performance.

    Reported, not graded. The right IOMMU mode depends on whether the node does
    SR-IOV passthrough, and the right THP setting depends on the workload, so
    encoding a pass/fail here would assert an opinion the tool has no basis for.
    A human -- or a later assertion policy -- decides; this makes the state visible.
    """
    record: Dict[str, str] = OrderedDict()
    record["Section"] = "Platform Tuning"

    cmdline = _read_sysfs("/proc/cmdline")
    if cmdline:
        iommu = [token for token in cmdline.split()
                 if "iommu" in token.lower()]
        record["IOMMU Boot Parameters"] = " ".join(iommu) if iommu else "None specified"
        mitigations = [token for token in cmdline.split()
                       if token.startswith("mitigations=")]
        if mitigations:
            record["CPU Mitigations"] = " ".join(mitigations)
        if len(cmdline) <= 500:
            record["Kernel Command Line"] = cmdline

    thp = _read_sysfs("/sys/kernel/mm/transparent_hugepage/enabled")
    if thp:
        # sysfs marks the active choice with brackets: "always [madvise] never".
        active = [token.strip("[]") for token in thp.split() if token.startswith("[")]
        record["Transparent Hugepages"] = active[0] if active else thp
    defrag = _read_sysfs("/sys/kernel/mm/transparent_hugepage/defrag")
    if defrag:
        active = [token.strip("[]") for token in defrag.split() if token.startswith("[")]
        if active:
            record["Transparent Hugepage Defrag"] = active[0]

    governor = _read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor:
        record["CPU Frequency Governor"] = governor
    driver = _read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver")
    if driver:
        record["CPU Frequency Driver"] = driver
    boost = _read_sysfs("/sys/devices/system/cpu/cpufreq/boost")
    if boost is not None:
        record["CPU Boost"] = {"1": "Enabled", "0": "Disabled"}.get(boost, boost)

    idle_driver = _read_sysfs("/sys/devices/system/cpu/cpuidle/current_driver")
    if idle_driver:
        record["CPU Idle Driver"] = idle_driver

    smt = _read_sysfs("/sys/devices/system/cpu/smt/control")
    if smt:
        record["SMT"] = smt

    numa_balancing = _read_sysfs("/proc/sys/kernel/numa_balancing")
    if numa_balancing is not None:
        record["Automatic NUMA Balancing"] = {"1": "Enabled",
                                              "0": "Disabled"}.get(numa_balancing, numa_balancing)

    if len(record) == 1:
        return None
    record["Note"] = ("Settings are reported as found; the appropriate values depend on "
                      "the deployment (for example SR-IOV hosts need full IOMMU translation)")
    return record


def _platform_affinity_entry() -> Optional[Dict[str, str]]:
    """Which NUMA node each GPU and NIC is attached to.

    A GPU on node 0 fed by a NIC on node 1 pays a cross-socket hop on every inbound
    byte. The mapping is cheap to read and invisible in every other section.
    """
    record: Dict[str, str] = OrderedDict()
    record["Section"] = "Platform Device Affinity"
    raw: Dict[str, object] = {}

    gpu_nodes: Dict[str, int] = OrderedDict()
    try:
        cards = sorted((name for name in os.listdir("/sys/class/drm")
                        if name.startswith("card") and name[4:].isdigit()),
                       key=lambda name: int(name[4:]))
    except OSError:
        cards = []
    for card in cards:
        node = _int(_read_sysfs(f"/sys/class/drm/{card}/device/numa_node"))
        vendor = _read_sysfs(f"/sys/class/drm/{card}/device/vendor")
        # 0x1002 is AMD; other vendors' display adapters are not what this is about.
        if node is None or node < 0 or vendor != "0x1002":
            continue
        gpu_nodes[card] = node
        record[f"GPU {card}"] = f"NUMA node {node}"
    if gpu_nodes:
        raw["gpu_numa_nodes"] = dict(gpu_nodes)

    nic_nodes: Dict[str, int] = OrderedDict()
    try:
        interfaces = sorted(os.listdir("/sys/class/net"))
    except OSError:
        interfaces = []
    for interface in interfaces:
        if interface == "lo":
            continue
        node = _int(_read_sysfs(f"/sys/class/net/{interface}/device/numa_node"))
        if node is None or node < 0:
            continue
        nic_nodes[interface] = node
        record[f"NIC {interface}"] = f"NUMA node {node}"
    if nic_nodes:
        raw["nic_numa_nodes"] = dict(nic_nodes)

    if not gpu_nodes and not nic_nodes:
        return None

    if gpu_nodes and nic_nodes:
        gpu_set = set(gpu_nodes.values())
        nic_set = set(nic_nodes.values())
        if gpu_set & nic_set:
            record["GPU/NIC Locality"] = (
                f"Shared NUMA node(s): {', '.join(str(node) for node in sorted(gpu_set & nic_set))}")
        else:
            record["GPU/NIC Locality"] = (
                f"No shared NUMA node: GPUs on {sorted(gpu_set)}, NICs on {sorted(nic_set)}; "
                f"host-staged transfers cross sockets")
        raw["gpu_nic_share_numa"] = bool(gpu_set & nic_set)

    record["_raw"] = raw
    return record


def gather_platform_details() -> Dict[str, List[Dict[str, str]]]:
    """Host firmware, memory, NUMA, tuning and device-affinity state.

    Fills rapido's largest blind spot: everything it reported was about the GPUs and
    nothing about the machine they sit in, so a node that underperformed because of a
    stale BIOS, a wrong NUMA placement or a power-saving governor looked identical to
    a healthy one. Unprivileged sources only -- no sudo, nothing that can block an
    unattended run on a password prompt.
    """
    if platform.system().lower() != "linux":
        return OrderedDict()

    entries = [entry for entry in (
        _platform_firmware_entry(),
        _platform_memory_entry(),
        _platform_numa_entry(),
        _platform_tuning_entry(),
        _platform_affinity_entry(),
    ) if entry]

    return OrderedDict([("linux", entries)]) if entries else OrderedDict()


def gather_gpu_details() -> Dict[str, List[Dict[str, str]]]:
    system = platform.system().lower()
    details: Dict[str, List[Dict[str, str]]] = OrderedDict()
    if system == "windows":
        adapters = windows_gpu_info()
        if adapters:
            details["windows"] = adapters
    elif system == "linux":
        adapters = linux_gpu_info()
        if adapters:
            details["linux"] = adapters
    elif system == "darwin":
        adapters = mac_gpu_info()
        if adapters:
            details["macos"] = adapters
    partitions = _gpu_partition_entries()
    if partitions:
        details.setdefault("linux", []).extend(partitions)
    generic = generic_gpu_info()
    if generic:
        details["generic"] = generic
    return details

def windows_network_info() -> List[Dict[str, str]]:
    """Collect detailed network interface information for Windows."""
    entries: List[Dict[str, str]] = []

    # Use PowerShell to get comprehensive adapter information
    ps_command = """
    Get-NetAdapter | ForEach-Object {
        $adapter = $_
        $ipConfig = Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -ErrorAction SilentlyContinue
        $stats = Get-NetAdapterStatistics -Name $adapter.Name -ErrorAction SilentlyContinue

        [PSCustomObject]@{
            Name = $adapter.Name
            Status = $adapter.Status
            MacAddress = $adapter.MacAddress
            LinkSpeed = $adapter.LinkSpeed
            MediaType = $adapter.MediaType
            InterfaceDescription = $adapter.InterfaceDescription
            DriverVersion = $adapter.DriverVersion
            DriverDate = $adapter.DriverDate
            DriverProvider = $adapter.DriverProvider
            IPv4Address = ($ipConfig | Where-Object {$_.AddressFamily -eq 'IPv4'} | Select-Object -First 1).IPAddress
            IPv6Address = ($ipConfig | Where-Object {$_.AddressFamily -eq 'IPv6'} | Select-Object -First 1).IPAddress
            ReceivedBytes = $stats.ReceivedBytes
            SentBytes = $stats.SentBytes
            ReceivedPackets = $stats.ReceivedUnicastPackets
            SentPackets = $stats.SentUnicastPackets
        }
    } | ConvertTo-Json
    """

    ps = run_command([
        "powershell",
        "-NoProfile",
        "-Command",
        ps_command
    ])

    if ps:
        try:
            data = json.loads(ps)
            if isinstance(data, dict):
                data = [data]

            for adapter in data:
                record = OrderedDict()

                if "Name" in adapter:
                    record["Interface"] = adapter["Name"]

                if "Status" in adapter:
                    record["Status"] = adapter["Status"]

                if "MacAddress" in adapter and adapter["MacAddress"]:
                    record["MAC Address"] = adapter["MacAddress"]

                if "LinkSpeed" in adapter and adapter["LinkSpeed"]:
                    record["Link Speed"] = adapter["LinkSpeed"]

                if "MediaType" in adapter and adapter["MediaType"]:
                    record["Media Type"] = adapter["MediaType"]

                if "InterfaceDescription" in adapter and adapter["InterfaceDescription"]:
                    record["Description"] = adapter["InterfaceDescription"]

                if "IPv4Address" in adapter and adapter["IPv4Address"]:
                    record["IPv4 Address"] = adapter["IPv4Address"]

                if "IPv6Address" in adapter and adapter["IPv6Address"]:
                    record["IPv6 Address"] = adapter["IPv6Address"]

                if "DriverVersion" in adapter and adapter["DriverVersion"]:
                    record["Driver Version"] = adapter["DriverVersion"]

                if "DriverProvider" in adapter and adapter["DriverProvider"]:
                    record["Driver Provider"] = adapter["DriverProvider"]

                # Format statistics
                def format_bytes(b):
                    if not b:
                        return None
                    b = int(b)
                    if b >= 1024**4:
                        return f"{b / (1024**4):.2f} TB"
                    elif b >= 1024**3:
                        return f"{b / (1024**3):.2f} GB"
                    elif b >= 1024**2:
                        return f"{b / (1024**2):.2f} MB"
                    elif b >= 1024:
                        return f"{b / 1024:.2f} KB"
                    else:
                        return f"{b} B"

                if "ReceivedBytes" in adapter and adapter["ReceivedBytes"]:
                    formatted = format_bytes(adapter["ReceivedBytes"])
                    if formatted:
                        record["RX Bytes"] = formatted

                if "SentBytes" in adapter and adapter["SentBytes"]:
                    formatted = format_bytes(adapter["SentBytes"])
                    if formatted:
                        record["TX Bytes"] = formatted

                if "ReceivedPackets" in adapter and adapter["ReceivedPackets"]:
                    try:
                        record["RX Packets"] = f"{int(adapter['ReceivedPackets']):,}"
                    except (ValueError, TypeError):
                        pass

                if "SentPackets" in adapter and adapter["SentPackets"]:
                    try:
                        record["TX Packets"] = f"{int(adapter['SentPackets']):,}"
                    except (ValueError, TypeError):
                        pass

                entries.append(record)

        except json.JSONDecodeError:
            pass

    return entries

def linux_network_info() -> List[Dict[str, str]]:
    """Collect detailed network interface information."""
    entries: List[Dict[str, str]] = []

    # Try 'ip -json' commands for comprehensive information
    ip_link = run_command(["ip", "-json", "link"])
    ip_addr = run_command(["ip", "-json", "addr"])

    link_data = {}
    addr_data = {}

    if ip_link:
        try:
            link_data = {iface["ifname"]: iface for iface in json.loads(ip_link)}
        except json.JSONDecodeError:
            pass

    if ip_addr:
        try:
            addr_data = {iface["ifname"]: iface for iface in json.loads(ip_addr)}
        except json.JSONDecodeError:
            pass

    # Merge link and addr data
    all_interfaces = set(list(link_data.keys()) + list(addr_data.keys()))

    for ifname in sorted(all_interfaces):
        record = OrderedDict()
        record["Interface"] = ifname

        # Get link information
        if ifname in link_data:
            link = link_data[ifname]

            # Operational state
            if "operstate" in link:
                record["State"] = link["operstate"]

            # MAC address
            if "address" in link:
                record["MAC Address"] = link["address"]

            # Link type
            if "link_type" in link:
                record["Type"] = link["link_type"]

            # MTU
            if "mtu" in link:
                record["MTU"] = str(link["mtu"])

            # Speed (if available)
            if "speed" in link and link["speed"] > 0:
                speed = link["speed"]
                if speed >= 1000:
                    record["Speed"] = f"{speed / 1000:.0f} Gbps"
                else:
                    record["Speed"] = f"{speed} Mbps"

        # Get address information
        if ifname in addr_data:
            addr = addr_data[ifname]

            # IPv4 and IPv6 addresses
            ipv4_addrs = []
            ipv6_addrs = []

            if "addr_info" in addr:
                for addr_info in addr["addr_info"]:
                    if "family" in addr_info and "local" in addr_info:
                        if addr_info["family"] == "inet":
                            prefix = addr_info.get("prefixlen", "")
                            ipv4_addrs.append(f"{addr_info['local']}/{prefix}" if prefix else addr_info['local'])
                        elif addr_info["family"] == "inet6":
                            prefix = addr_info.get("prefixlen", "")
                            ipv6_addrs.append(f"{addr_info['local']}/{prefix}" if prefix else addr_info['local'])

            if ipv4_addrs:
                record["IPv4 Address"] = ", ".join(ipv4_addrs)
            if ipv6_addrs:
                record["IPv6 Address"] = ", ".join(ipv6_addrs[:2])  # Limit to first 2 IPv6 addresses

        # Get additional details from ethtool (if available)
        ethtool_output = run_command(["ethtool", ifname])
        if ethtool_output:
            for line in ethtool_output.splitlines():
                line = line.strip()
                if "Speed:" in line and "Speed" not in record:
                    speed_match = line.split("Speed:")
                    if len(speed_match) > 1:
                        record["Speed"] = speed_match[1].strip()
                elif "Duplex:" in line:
                    duplex_match = line.split("Duplex:")
                    if len(duplex_match) > 1:
                        record["Duplex"] = duplex_match[1].strip()
                elif "Link detected:" in line:
                    link_match = line.split("Link detected:")
                    if len(link_match) > 1:
                        record["Link Detected"] = link_match[1].strip()

        # Get driver information from ethtool -i
        ethtool_driver = run_command(["ethtool", "-i", ifname])
        if ethtool_driver:
            for line in ethtool_driver.splitlines():
                line = line.strip()
                if line.startswith("driver:"):
                    record["Driver"] = line.split(":", 1)[1].strip()
                elif line.startswith("version:"):
                    record["Driver Version"] = line.split(":", 1)[1].strip()
                elif line.startswith("firmware-version:"):
                    fw_version = line.split(":", 1)[1].strip()
                    if fw_version and fw_version != "N/A":
                        record["Firmware"] = fw_version
                elif line.startswith("bus-info:"):
                    record["PCI Address"] = line.split(":", 1)[1].strip()

        # Get statistics
        ethtool_stats = run_command(["ethtool", "-S", ifname])
        if ethtool_stats:
            rx_bytes = 0
            tx_bytes = 0
            rx_packets = 0
            tx_packets = 0
            rx_errors = 0
            tx_errors = 0

            for line in ethtool_stats.splitlines():
                line = line.strip().lower()
                if "rx_bytes:" in line or "rx bytes:" in line:
                    try:
                        rx_bytes = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "tx_bytes:" in line or "tx bytes:" in line:
                    try:
                        tx_bytes = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "rx_packets:" in line or "rx packets:" in line:
                    try:
                        rx_packets = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "tx_packets:" in line or "tx packets:" in line:
                    try:
                        tx_packets = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "rx_errors:" in line or "rx errors:" in line:
                    try:
                        rx_errors = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                elif "tx_errors:" in line or "tx errors:" in line:
                    try:
                        tx_errors = int(line.split(":")[-1].strip())
                    except ValueError:
                        pass

            # Format bytes in human-readable format
            def format_bytes(b):
                if b >= 1024**4:
                    return f"{b / (1024**4):.2f} TB"
                elif b >= 1024**3:
                    return f"{b / (1024**3):.2f} GB"
                elif b >= 1024**2:
                    return f"{b / (1024**2):.2f} MB"
                elif b >= 1024:
                    return f"{b / 1024:.2f} KB"
                else:
                    return f"{b} B"

            if rx_bytes > 0:
                record["RX Bytes"] = format_bytes(rx_bytes)
            if tx_bytes > 0:
                record["TX Bytes"] = format_bytes(tx_bytes)
            if rx_packets > 0:
                record["RX Packets"] = f"{rx_packets:,}"
            if tx_packets > 0:
                record["TX Packets"] = f"{tx_packets:,}"
            if rx_errors > 0:
                record["RX Errors"] = f"{rx_errors:,}"
            if tx_errors > 0:
                record["TX Errors"] = f"{tx_errors:,}"

        if record:
            entries.append(record)

    # Fallback to basic 'ip link' without JSON if no interfaces found
    if not entries:
        ip_link_text = run_command(["ip", "link"])
        if ip_link_text:
            for line in ip_link_text.splitlines():
                line = line.strip()
                if line and line[0].isdigit() and ":" in line:
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        record = OrderedDict()
                        record["Interface"] = parts[1].strip().split("@")[0]
                        if "state" in line.upper():
                            state_part = line.split("state")
                            if len(state_part) > 1:
                                record["Status"] = state_part[1].strip().split()[0]
                        entries.append(record)

    return entries

def mac_network_info() -> List[Dict[str, str]]:
    """Collect detailed network interface information for macOS."""
    entries: List[Dict[str, str]] = []

    # Use ifconfig to get detailed interface information
    ifconfig_output = run_command(["ifconfig", "-a"])
    if ifconfig_output:
        record = None

        for line in ifconfig_output.splitlines():
            # New interface starts (line doesn't start with whitespace)
            if line and not line[0].isspace():
                # Save previous interface
                if record:
                    entries.append(record)

                # Start new interface
                parts = line.split(":", 1)
                interface_name = parts[0].strip()
                record = OrderedDict()
                record["Interface"] = interface_name

                # Parse flags from first line
                if "flags=" in line:
                    if "UP" in line and "RUNNING" in line:
                        record["Status"] = "UP"
                    elif "UP" in line:
                        record["Status"] = "UP (not running)"
                    else:
                        record["Status"] = "DOWN"

                # Parse MTU
                if "mtu" in line.lower():
                    mtu_match = line.lower().split("mtu")
                    if len(mtu_match) > 1:
                        mtu_parts = mtu_match[1].strip().split()
                        if mtu_parts:
                            record["MTU"] = mtu_parts[0]

            elif record and line.strip():
                line = line.strip()

                # MAC address (ether)
                if line.startswith("ether "):
                    mac = line.split()[1] if len(line.split()) > 1 else ""
                    if mac:
                        record["MAC Address"] = mac

                # IPv4 address
                elif line.startswith("inet "):
                    parts = line.split()
                    if len(parts) > 1:
                        ip_addr = parts[1]
                        netmask = ""
                        if "netmask" in line and len(parts) > 3:
                            netmask = parts[3]
                        if netmask and netmask.startswith("0x"):
                            # Convert hex netmask to CIDR
                            try:
                                mask_int = int(netmask, 16)
                                cidr = bin(mask_int).count('1')
                                record["IPv4 Address"] = f"{ip_addr}/{cidr}"
                            except ValueError:
                                record["IPv4 Address"] = ip_addr
                        else:
                            record["IPv4 Address"] = ip_addr

                # IPv6 address
                elif line.startswith("inet6 "):
                    parts = line.split()
                    if len(parts) > 1:
                        ipv6_addr = parts[1]
                        if "IPv6 Address" not in record:
                            record["IPv6 Address"] = ipv6_addr

                # Media type and status
                elif line.startswith("media:"):
                    media_info = line.replace("media:", "").strip()
                    record["Media"] = media_info

                elif line.startswith("status:"):
                    status_info = line.replace("status:", "").strip()
                    record["Link Status"] = status_info

        # Append last interface
        if record:
            entries.append(record)

    # Get additional network statistics using netstat
    netstat_output = run_command(["netstat", "-ibn"])
    if netstat_output and entries:
        lines = netstat_output.splitlines()
        if len(lines) > 1:
            # Parse header to find column positions
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 7:
                    iface_name = parts[0]
                    # Find matching interface in entries
                    for entry in entries:
                        if entry.get("Interface") == iface_name:
                            # Format bytes
                            def format_bytes(b):
                                try:
                                    b = int(b)
                                    if b >= 1024**4:
                                        return f"{b / (1024**4):.2f} TB"
                                    elif b >= 1024**3:
                                        return f"{b / (1024**3):.2f} GB"
                                    elif b >= 1024**2:
                                        return f"{b / (1024**2):.2f} MB"
                                    elif b >= 1024:
                                        return f"{b / 1024:.2f} KB"
                                    else:
                                        return f"{b} B"
                                except (ValueError, TypeError):
                                    return None

                            try:
                                # columns: Name Mtu Network Address Ipkts Ierrs Ibytes Opkts Oerrs Obytes
                                if len(parts) >= 10:
                                    ipkts = parts[4]
                                    ibytes = parts[6]
                                    opkts = parts[7]
                                    obytes = parts[9]

                                    rx_bytes_formatted = format_bytes(ibytes)
                                    if rx_bytes_formatted:
                                        entry["RX Bytes"] = rx_bytes_formatted

                                    tx_bytes_formatted = format_bytes(obytes)
                                    if tx_bytes_formatted:
                                        entry["TX Bytes"] = tx_bytes_formatted

                                    entry["RX Packets"] = f"{int(ipkts):,}"
                                    entry["TX Packets"] = f"{int(opkts):,}"
                            except (ValueError, IndexError):
                                pass
                            break

    return entries

def gather_network_details() -> Dict[str, List[Dict[str, str]]]:
    """Gather network interface names only - simplified."""
    system = platform.system().lower()
    details: Dict[str, List[Dict[str, str]]] = OrderedDict()

    if system == "windows":
        interfaces = windows_network_info()
        if interfaces:
            details["windows"] = interfaces
    elif system == "linux":
        interfaces = linux_network_info()
        if interfaces:
            details["linux"] = interfaces
    elif system == "darwin":
        interfaces = mac_network_info()
        if interfaces:
            details["macos"] = interfaces

    return details

def gather_bmc_info() -> Dict[str, List[Dict[str, str]]]:
    """Gather BMC (Baseboard Management Controller) information using IPMI tools."""
    details: Dict[str, List[Dict[str, str]]] = OrderedDict()
    system = platform.system().lower()

    # BMC information is primarily available on Linux servers
    if system != "linux":
        return details

    bmc_list = []

    # === BMC Device Information ===
    # Check if ipmitool is available
    ipmitool_version = run_command(["ipmitool", "-V"])
    if not ipmitool_version:
        return details

    # BMC Info section
    bmc_info = OrderedDict()
    bmc_info["Section"] = "BMC Device Information"

    # Get BMC device information
    bmc_info_output = run_command(["ipmitool", "bmc", "info"])
    if bmc_info_output:
        for line in bmc_info_output.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if value and value != "0":
                        bmc_info[key] = value

    if len(bmc_info) > 1:  # More than just the Section
        bmc_list.append(bmc_info)

    # === BMC Network Configuration ===
    lan_info = OrderedDict()
    lan_info["Section"] = "BMC Network Configuration"

    lan_output = run_command(["ipmitool", "lan", "print"])
    if lan_output:
        for line in lan_output.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if value:
                        lan_info[key] = value

    if len(lan_info) > 1:
        bmc_list.append(lan_info)

    # === Sensor Data Records (SDR) ===
    sdr_output = run_command(["ipmitool", "sdr", "list"])
    if sdr_output:
        # Group sensors by type
        temperature_sensors = OrderedDict()
        temperature_sensors["Section"] = "Temperature Sensors"

        voltage_sensors = OrderedDict()
        voltage_sensors["Section"] = "Voltage Sensors"

        fan_sensors = OrderedDict()
        fan_sensors["Section"] = "Fan Sensors"

        power_sensors = OrderedDict()
        power_sensors["Section"] = "Power Sensors"

        other_sensors = OrderedDict()
        other_sensors["Section"] = "Other Sensors"

        for line in sdr_output.splitlines():
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    sensor_name = parts[0]
                    sensor_value = parts[1]
                    sensor_status = parts[2] if len(parts) > 2 else ""

                    # Categorize by sensor name or value
                    sensor_lower = sensor_name.lower()
                    value_lower = sensor_value.lower()

                    if "temp" in sensor_lower or "degrees" in value_lower or "°" in sensor_value:
                        temperature_sensors[sensor_name] = f"{sensor_value} ({sensor_status})"
                    elif "volt" in sensor_lower or "v" in value_lower:
                        voltage_sensors[sensor_name] = f"{sensor_value} ({sensor_status})"
                    elif "fan" in sensor_lower or "rpm" in value_lower:
                        fan_sensors[sensor_name] = f"{sensor_value} ({sensor_status})"
                    elif "power" in sensor_lower or "watt" in value_lower or "w" in value_lower:
                        power_sensors[sensor_name] = f"{sensor_value} ({sensor_status})"
                    else:
                        other_sensors[sensor_name] = f"{sensor_value} ({sensor_status})"

        # Add non-empty sensor groups
        if len(temperature_sensors) > 1:
            bmc_list.append(temperature_sensors)
        if len(voltage_sensors) > 1:
            bmc_list.append(voltage_sensors)
        if len(fan_sensors) > 1:
            bmc_list.append(fan_sensors)
        if len(power_sensors) > 1:
            bmc_list.append(power_sensors)
        if len(other_sensors) > 1:
            bmc_list.append(other_sensors)

    # === FRU (Field Replaceable Unit) Information ===
    fru_output = run_command(["ipmitool", "fru", "print"])
    if fru_output:
        current_fru = None
        fru_record = None

        for line in fru_output.splitlines():
            line_stripped = line.strip()

            # New FRU section (starts with "FRU Device Description")
            if line_stripped.startswith("FRU Device Description"):
                # Save previous FRU
                if fru_record and len(fru_record) > 1:
                    bmc_list.append(fru_record)

                # Start new FRU
                parts = line_stripped.split(":", 1)
                if len(parts) == 2:
                    current_fru = parts[1].strip()
                    fru_record = OrderedDict()
                    fru_record["Section"] = f"FRU: {current_fru}"

            # FRU field
            elif ":" in line_stripped and fru_record is not None:
                parts = line_stripped.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if value and value.lower() != "unspecified":
                        fru_record[key] = value

        # Save last FRU
        if fru_record and len(fru_record) > 1:
            bmc_list.append(fru_record)

    # === System Event Log (SEL) Info ===
    sel_info = OrderedDict()
    sel_info["Section"] = "System Event Log (SEL) Information"

    sel_info_output = run_command(["ipmitool", "sel", "info"])
    if sel_info_output:
        for line in sel_info_output.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if value:
                        sel_info[key] = value

    if len(sel_info) > 1:
        bmc_list.append(sel_info)

    # === Recent SEL Entries (last 10) ===
    sel_list_output = run_command(["ipmitool", "sel", "list", "last", "10"])
    if sel_list_output:
        sel_entries = OrderedDict()
        sel_entries["Section"] = "Recent System Events (Last 10)"

        entry_count = 1
        for line in sel_list_output.splitlines():
            if line.strip():
                # Format: ID | Date | Time | Sensor | Event | Status
                sel_entries[f"Event {entry_count}"] = line.strip()
                entry_count += 1

        if len(sel_entries) > 1:
            bmc_list.append(sel_entries)

    # === Power Status ===
    power_status = run_command(["ipmitool", "chassis", "power", "status"])
    if power_status:
        chassis_info = OrderedDict()
        chassis_info["Section"] = "Chassis Power Information"
        chassis_info["Power Status"] = power_status.strip()

        # Get chassis status
        chassis_status_output = run_command(["ipmitool", "chassis", "status"])
        if chassis_status_output:
            for line in chassis_status_output.splitlines():
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        if value:
                            chassis_info[key] = value

        if len(chassis_info) > 1:
            bmc_list.append(chassis_info)

    if bmc_list:
        details["bmc"] = bmc_list

    return details

# Peak theoretical throughput expressed as FLOPS (or OPS) per compute unit per clock.
# This is the architecture-intrinsic form: TFLOPS = CUs * flops_per_cu_per_clock * clock_MHz / 1e6.
#
# Verified against AMD published peak figures, e.g.:
#   MI300X  (gfx942, 304 CU @ 2100 MHz): FP64 vec 128 -> 81.7 TFLOPS, FP32 vec 256 -> 163.4,
#                                        FP16 mat 2048 -> 1307.4, FP8 mat 4096 -> 2614.9
#   MI355X  (gfx950, 256 CU @ 2400 MHz): FP64 vec 128 -> 78.6 TFLOPS, FP16 mat 4096 -> 2516.6
#   MI250X  (gfx90a, 220 CU @ 1700 MHz): FP64 vec 128 -> 47.9 TFLOPS, FP16 mat 1024 -> 383.0
#
# Notes:
#   - "vector" and "matrix" rates are tracked separately; they differ on CDNA (e.g. CDNA3 FP64
#     matrix is 2x its vector rate, while CDNA4 halves FP64 matrix relative to CDNA3).
#   - TF32 is a real CDNA3 matrix format, but CDNA4 removed the hardware path (emulated via BF16),
#     so it is only listed for gfx94x. It was never present on CDNA1/CDNA2 or RDNA.
#   - "sparse" lists the precisions that support 2:4 structured sparsity at 2x the dense matrix
#     rate. CDNA2 has no structured sparsity support; CDNA4 dropped it for FP32/FP64.
GPU_ARCH_PEAK_TABLE = {
    # CDNA4 - MI350X / MI355X
    "gfx950": {
        "name": "CDNA4",
        "vector": {"FP64": 128, "FP32": 256},
        "matrix": {"FP64": 128, "FP32": 256, "BF16": 4096, "FP16": 4096,
                   "FP8": 8192, "INT8": 8192, "FP6": 16384, "FP4": 16384},
        "sparse": ["BF16", "FP16", "FP8", "INT8", "FP6", "FP4"],
    },
    # CDNA3 - MI300A / MI300X / MI325X
    "gfx942": {
        "name": "CDNA3",
        "vector": {"FP64": 128, "FP32": 256},
        "matrix": {"FP64": 256, "FP32": 256, "TF32": 1024, "BF16": 2048,
                   "FP16": 2048, "FP8": 4096, "INT8": 4096},
        "sparse": ["TF32", "BF16", "FP16", "FP8", "INT8"],
    },
    # CDNA2 - MI210 / MI250 / MI250X. No structured sparsity.
    "gfx90a": {
        "name": "CDNA2",
        "vector": {"FP64": 128, "FP32": 128},
        "matrix": {"FP64": 256, "FP32": 256, "BF16": 1024, "FP16": 1024, "INT8": 1024},
        "sparse": [],
    },
    # CDNA1 - MI100
    "gfx908": {
        "name": "CDNA1",
        "vector": {"FP64": 64, "FP32": 128},
        # CDNA1 runs BF16 MFMA at half the FP16 rate (K=4 vs K=8);
        # BF16 only reaches FP16 parity from CDNA2 onwards.
        "matrix": {"FP32": 256, "BF16": 512, "FP16": 1024, "INT8": 1024},
        "sparse": [],
    },
    # GCN5 (Vega) - MI50 / MI60
    "gfx906": {
        "name": "GCN5",
        "vector": {"FP64": 64, "FP32": 128},
        "matrix": {},
        "sparse": [],
    },
    "gfx900": {
        "name": "GCN5",
        "vector": {"FP64": 8, "FP32": 128},
        "matrix": {},
        "sparse": [],
    },
}

# gfx940/gfx941 are pre-production CDNA3 variants sharing the gfx942 rates.
GPU_ARCH_PEAK_TABLE["gfx940"] = GPU_ARCH_PEAK_TABLE["gfx942"]
GPU_ARCH_PEAK_TABLE["gfx941"] = GPU_ARCH_PEAK_TABLE["gfx942"]


def _lookup_gpu_arch(gfx_version: Optional[str]) -> Optional[Dict]:
    """Resolve a gfx version string (e.g. 'gfx942:sramecc+:xnack-') to its peak-rate table entry."""
    if not gfx_version:
        return None
    gfx = gfx_version.lower()
    # Match the longest key first so gfx9xx variants cannot shadow a more specific entry.
    for key in sorted(GPU_ARCH_PEAK_TABLE, key=len, reverse=True):
        if key in gfx:
            return GPU_ARCH_PEAK_TABLE[key]
    return None


def compute_peak_performance(gfx_version: Optional[str], compute_units: int,
                             clock_mhz: float) -> Dict[str, Dict[str, float]]:
    """Compute peak theoretical throughput in TFLOPS/TOPS for a GPU.

    Returns a dict with 'vector', 'matrix' and 'sparse' sub-dicts mapping precision -> TFLOPS,
    plus an 'arch' name. Returns an empty 'arch' when the architecture is unknown, so callers
    can say so explicitly rather than silently reporting a fabricated number.
    """
    arch = _lookup_gpu_arch(gfx_version)
    if not arch or not compute_units or not clock_mhz:
        return {"arch": None, "vector": {}, "matrix": {}, "sparse": {}}

    def scale(rates: Dict[str, int]) -> Dict[str, float]:
        return {
            precision: (compute_units * per_clock * clock_mhz) / 1e6
            for precision, per_clock in rates.items()
        }

    matrix = scale(arch["matrix"])
    sparse = {p: matrix[p] * 2 for p in arch["sparse"] if p in matrix}

    return {
        "arch": arch["name"],
        "vector": scale(arch["vector"]),
        "matrix": matrix,
        "sparse": sparse,
    }


def _collect_gpu_link_types() -> Dict[int, Dict[int, str]]:
    """Map src GPU -> dst GPU -> link type ("XGMI", "PCIE", ...).

    Source is `rocm-smi --showtopotype`, which prints a GPU x GPU table of link types.
    Note that `amd-smi topology --linktype` is *not* a valid invocation on shipping
    amd-smi builds (it raises AmdSmiInvalidParameterException), so rocm-smi is the
    one reliable source here. Returns {} when the tool or table is unavailable; the
    matrix card then simply carries no link types and the heatmap skips the overlay.
    """
    output = run_command(["rocm-smi", "--showtopotype"], record=False)
    if not output:
        return {}

    links: Dict[int, Dict[int, str]] = {}
    header: List[int] = []
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if not header:
            # Header row: the column labels, all of the form GPU<n>.
            if all(f.upper().startswith("GPU") and f[3:].isdigit() for f in fields):
                header = [int(f[3:]) for f in fields]
            continue
        row_label = fields[0].upper()
        if not (row_label.startswith("GPU") and row_label[3:].isdigit()):
            continue
        src = int(row_label[3:])
        row: Dict[int, str] = {}
        for dst, value in zip(header, fields[1:]):
            value = value.upper()
            # The self-link is printed as "0"; it is not a link type.
            if dst != src and value not in ("0", "N/A", "NA"):
                row[dst] = value
        if row:
            links[src] = row
    return links


def _build_p2p_matrix_card(pairs: List[Dict],
                           link_types: Optional[Dict[int, Dict[int, str]]] = None) -> Optional[Dict]:
    """Fold the per-pair P2P results into a single N x N bandwidth matrix card.

    The per-pair cards stay as they are; this adds one card the report can draw a
    heatmap from without having to re-parse 56 separate sections.
    """
    ids = sorted({p[k] for p in pairs for k in ("src_gpu", "dst_gpu")
                  if isinstance(p[k], int)})
    if not ids:
        return None

    index = {gpu: i for i, gpu in enumerate(ids)}
    n = len(ids)
    bandwidth = [[None] * n for _ in range(n)]
    enabled = [[None] * n for _ in range(n)]
    for p in pairs:
        src, dst = p["src_gpu"], p["dst_gpu"]
        if src in index and dst in index:
            bandwidth[index[src]][index[dst]] = p["bandwidth_gbps"]
            enabled[index[src]][index[dst]] = p["p2p_enabled"]

    # Summarise the matrix itself, not the raw pair list: a pair whose GPU ids did
    # not make it into the index is absent from the heatmap, so counting it here
    # would make the stats disagree with the picture they sit above.
    measured = [v for row in bandwidth for v in row if v]

    # Link type per ordered pair, same shape as the bandwidth matrix so the report can
    # overlay "is this hop XGMI or PCIe?" directly onto each heatmap cell. A PCIe hop
    # in a node advertised as fully XGMI-connected is exactly the defect worth seeing.
    link_matrix: Optional[List[List[Optional[str]]]] = None
    if link_types:
        link_matrix = [[None] * n for _ in range(n)]
        for src, row in link_types.items():
            if src not in index:
                continue
            for dst, kind in row.items():
                if dst in index:
                    link_matrix[index[src]][index[dst]] = kind

    card = OrderedDict()
    card["Section"] = "GPU P2P Bandwidth Matrix"
    card["GPU Count"] = str(n)
    card["Measured Links"] = str(len(measured))
    if measured:
        card["Min Bandwidth"] = f"{min(measured):.2f} GB/s"
        card["Max Bandwidth"] = f"{max(measured):.2f} GB/s"
        card["Mean Bandwidth"] = f"{sum(measured) / len(measured):.2f} GB/s"
    if link_matrix:
        kinds = sorted({k for row in link_matrix for k in row if k})
        if kinds:
            card["Link Types"] = ", ".join(kinds)
    card["_raw"] = {
        "gpu_ids": ids,
        "bandwidth_gbps": bandwidth,
        "p2p_enabled": enabled,
        "link_types": link_matrix,
        "min_gbps": min(measured) if measured else None,
        "max_gbps": max(measured) if measured else None,
        "mean_gbps": (sum(measured) / len(measured)) if measured else None,
    }
    return card


def _build_peak_performance_cards(gpu_key: str, gfx_version: Optional[str], compute_units: int,
                                  clock_mhz: float, max_memory: Optional[float]) -> List[Dict]:
    """Build the display cards (and raw numerics) for a GPU's theoretical peak performance."""
    peaks = compute_peak_performance(gfx_version, compute_units, clock_mhz)
    cards: List[Dict] = []

    # Units differ: integer formats are counted in TOPS, floating point in TFLOPS.
    def fmt(precision: str, value: float) -> str:
        return f"{value:.1f} {'TOPS' if precision.startswith('INT') else 'TFLOPS'}"

    dense_info = OrderedDict()
    dense_info["Section"] = f"{gpu_key} - Dense Peak Performance"
    dense_info["Compute Units"] = str(compute_units)
    dense_info["Max Clock Frequency"] = f"{clock_mhz:.0f} MHz"

    if peaks["arch"]:
        dense_info["Architecture"] = peaks["arch"]
    else:
        # Unknown architecture: report the fact rather than guessing at multipliers.
        dense_info["Architecture"] = f"Unknown ({gfx_version or 'gfx version not detected'})"
        dense_info["Note"] = "Peak performance not calculated - architecture not in rate table"
        if max_memory:
            dense_info["Max Memory"] = f"{max_memory:.2f} GB"
        return [dense_info]

    for precision, value in peaks["vector"].items():
        dense_info[f"{precision} (vector)"] = fmt(precision, value)
    for precision, value in peaks["matrix"].items():
        dense_info[f"{precision} (matrix)"] = fmt(precision, value)

    if max_memory:
        dense_info["Max Memory"] = f"{max_memory:.2f} GB"

    # Machine-readable numerics alongside the formatted strings, for charting and thresholds.
    dense_info["_raw"] = {
        "gfx_version": gfx_version,
        "architecture": peaks["arch"],
        "compute_units": compute_units,
        "clock_mhz": clock_mhz,
        "memory_gb": round(max_memory, 2) if max_memory else None,
        "peak_vector_tflops": {p: round(v, 1) for p, v in peaks["vector"].items()},
        "peak_matrix_tflops": {p: round(v, 1) for p, v in peaks["matrix"].items()},
    }
    cards.append(dense_info)

    if peaks["sparse"]:
        sparse_info = OrderedDict()
        sparse_info["Section"] = f"{gpu_key} - Sparse Peak Performance"
        sparse_info["Compute Units"] = str(compute_units)
        sparse_info["Max Clock Frequency"] = f"{clock_mhz:.0f} MHz"
        sparse_info["Architecture"] = peaks["arch"]
        sparse_info["Note"] = "2:4 structured sparsity (2x dense matrix rate)"
        for precision, value in peaks["sparse"].items():
            sparse_info[f"{precision} (matrix, sparse)"] = fmt(precision, value)
        sparse_info["_raw"] = {
            "gfx_version": gfx_version,
            "architecture": peaks["arch"],
            "peak_sparse_tflops": {p: round(v, 1) for p, v in peaks["sparse"].items()},
        }
        cards.append(sparse_info)

    return cards


def _dense_matrix_peaks(cards: List[Dict[str, str]]) -> Dict[str, float]:
    """Theoretical dense *matrix* peak TFLOPS per precision, from the peak cards.

    Matrix rather than vector rates, because rocBLAS dispatches to the matrix cores;
    comparing a library GEMM against the vector peak would report efficiencies well
    above 100% and mean nothing. The peak cards are per GPU model, and the efficiency
    comparison below assumes the node is homogeneous -- which every other per-GPU
    comparison in this tool already assumes.
    """
    peaks: Dict[str, float] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        raw = card.get("_raw")
        if not isinstance(raw, dict):
            continue
        matrix = raw.get("peak_matrix_tflops")
        if not isinstance(matrix, dict):
            continue
        for precision, value in matrix.items():
            number = _num(value)
            if number:
                peaks[str(precision).upper()] = number
    return peaks


def _build_library_gemm_cards(data: Dict, peak_cards: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Cards for the rocBLAS GEMM results, including efficiency against theoretical peak.

    Efficiency is the point of this benchmark. An absolute TFLOPS figure means nothing
    without the peak it is being measured against, and until Tier 1 fixed the peak model
    rapido could not compute that ratio honestly for any precision.
    """
    model_peaks = _dense_matrix_peaks(peak_cards)
    cards: List[Dict[str, str]] = []

    for result in data.get("results", []):
        if not isinstance(result, dict):
            continue
        gpu_id = result.get("gpu_id", "?")
        card: Dict[str, str] = OrderedDict()
        card["Section"] = f"GPU {gpu_id} Library GEMM (rocBLAS)"
        card["GPU Architecture"] = str(result.get("gpu_name", "Unknown"))
        card["Library"] = str(result.get("library", "rocBLAS"))
        raw: Dict[str, object] = {"gpu_id": gpu_id}

        # Keep only the best result per precision: smaller matrices are included to
        # cover partitioned GPUs with less memory, not to be reported separately.
        best: Dict[str, Dict] = {}
        for measurement in result.get("measurements", []):
            if not isinstance(measurement, dict):
                continue
            precision = str(measurement.get("precision", ""))
            tflops = _num(measurement.get("tflops"))
            if not precision or tflops is None:
                continue
            if precision not in best or tflops > (_num(best[precision].get("tflops")) or 0.0):
                best[precision] = measurement

        for precision in sorted(best):
            measurement = best[precision]
            tflops = _num(measurement.get("tflops")) or 0.0
            size = _int(measurement.get("size"))
            label = f"{tflops:.2f} TFLOPS"
            if size:
                label += f" at {size}x{size}"
            peak = model_peaks.get(precision)
            if peak:
                efficiency = tflops / peak * 100.0
                label += f" ({efficiency:.1f}% of {peak:.1f} TFLOPS peak)"
                raw[f"gemm_{precision.lower()}_efficiency_pct"] = round(efficiency, 2)
            card[f"Achieved GEMM {precision}"] = label
            raw[f"gemm_{precision.lower()}_tflops"] = tflops

        if len(card) > 3:
            card["_raw"] = raw
            cards.append(card)

    return cards


def _build_rccl_cards(data: Dict) -> List[Dict[str, str]]:
    """Cards for the RCCL collective results.

    Bus bandwidth is reported alongside the algorithm bandwidth because only the
    former is comparable to the fabric's link rate; the algorithm figure is what the
    application sees and the two differ by a factor that depends on the rank count.
    """
    ranks = _int(data.get("ranks"))
    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        return []

    cards: List[Dict[str, str]] = []
    by_collective: Dict[str, List[Dict]] = OrderedDict()
    for entry in results:
        if isinstance(entry, dict) and entry.get("collective"):
            by_collective.setdefault(str(entry["collective"]), []).append(entry)

    summary: Dict[str, str] = OrderedDict()
    summary["Section"] = "RCCL Collective Bandwidth"
    if ranks:
        summary["Ranks"] = str(ranks)
    version = _int(data.get("rccl_version"))
    if version:
        # RCCL reports its version packed as major*10000 + minor*100 + patch.
        summary["RCCL Version"] = (f"{version // 10000}.{(version // 100) % 100}."
                                   f"{version % 100}")
    raw: Dict[str, object] = {"ranks": ranks}

    for collective, entries in by_collective.items():
        peak = max(entries, key=lambda e: _num(e.get("bus_gbps")) or 0.0)
        bus = _num(peak.get("bus_gbps"))
        algorithm = _num(peak.get("algorithm_gbps"))
        size_mb = (_num(peak.get("bytes")) or 0.0) / (1024 * 1024)
        if bus is None:
            continue
        summary[collective] = (f"{bus:.2f} GB/s bus bandwidth "
                               f"({algorithm:.2f} GB/s algorithm) at {size_mb:.0f} MB"
                               if algorithm is not None else f"{bus:.2f} GB/s bus bandwidth")
        raw[f"rccl_{collective.lower()}_bus_gbps"] = bus
        if algorithm is not None:
            raw[f"rccl_{collective.lower()}_algorithm_gbps"] = algorithm

    if len(summary) <= 1:
        return []
    summary["Note"] = ("Measured across all visible GPUs in one process via "
                       "ncclCommInitAll; bus bandwidth applies the standard "
                       "rank-count correction and is what compares to the link rate")
    summary["_raw"] = raw
    cards.append(summary)

    detail: Dict[str, str] = OrderedDict()
    detail["Section"] = "RCCL Collective Bandwidth by Message Size"
    for entry in results:
        if not isinstance(entry, dict):
            continue
        bus = _num(entry.get("bus_gbps"))
        size_mb = (_num(entry.get("bytes")) or 0.0) / (1024 * 1024)
        if bus is None:
            continue
        detail[f"{entry.get('collective')} {size_mb:.0f} MB"] = f"{bus:.2f} GB/s bus bandwidth"
    if len(detail) > 1:
        cards.append(detail)

    return cards


def _rccl_available() -> bool:
    """Whether RCCL is present to link against.

    Detection is by library and header on disk, not by running anything: RCCL is a
    shared library with no executable, so the usual check_tool_availability() probe
    would report it missing on every node that has it.
    """
    rocm_path = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for directory in (os.path.join(rocm_path, "lib"), os.path.join(rocm_path, "lib64"),
                      "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
        if os.path.exists(os.path.join(directory, "librccl.so")):
            return True
    return False


def gather_gpu_microbenchmarks(include_p2p: bool = False, verbose: bool = True,
                               full: bool = False) -> Dict[str, List[Dict[str, str]]]:
    """Gather GPU microbenchmark information including peak performance and optionally GPU-to-GPU communication.

    Pass full=True to additionally run the heavyweight library benchmarks (rocBLAS
    GEMM and RCCL collectives). They are gated because together they add several
    minutes and saturate every GPU on the node, which is not acceptable by default
    on a machine that may be running someone else's job.
    """
    details: Dict[str, List[Dict[str, str]]] = OrderedDict()
    system = platform.system().lower()

    # Only gather microbenchmark info on Linux
    if system != "linux":
        return details

    microbenchmark_list = []

    # Get GPU specifications for roofline calculations from rocminfo
    rocminfo_output = run_command(["rocminfo"])
    if rocminfo_output:
        current_gpu = None
        gfx_version = None
        compute_units = None
        max_clock_freq = None
        max_memory = None
        # rocminfo lists the host CPU as an HSA agent too, and its Marketing Name
        # ("AMD EPYC ...") matches the vendor test below. Only "Device Type: GPU"
        # agents get peak-performance cards; one card per socket would otherwise appear.
        device_type = None

        for line in rocminfo_output.splitlines():
            line = line.strip()

            if line.startswith("*******"):
                # Process previous GPU
                if current_gpu and compute_units and device_type == "GPU":
                    gpu_key = f"{current_gpu}"
                    if gfx_version:
                        gpu_key += f" ({gfx_version})"

                    # Use detected or estimated clock frequency
                    base_clock = max_clock_freq if max_clock_freq else 1500  # Default 1.5 GHz if not detected

                    microbenchmark_list.extend(
                        _build_peak_performance_cards(gpu_key, gfx_version, compute_units,
                                                      base_clock, max_memory)
                    )
                # Reset for next GPU
                current_gpu = None
                gfx_version = None
                compute_units = None
                max_clock_freq = None
                max_memory = None
                device_type = None

            elif "Device Type:" in line and ":" in line:
                device_type = line.split(":", 1)[1].strip().upper()

            elif "Marketing Name" in line and ":" in line:
                gpu_name = line.split(":", 1)[1].strip()
                if gpu_name and ("AMD" in gpu_name or "Radeon" in gpu_name or "Instinct" in gpu_name):
                    current_gpu = gpu_name

            elif current_gpu and "Compute Unit" in line and ":" in line:
                try:
                    compute_units = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif current_gpu and "Max Clock Freq. (MHz)" in line and ":" in line:
                try:
                    max_clock_freq = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

            elif current_gpu and "Max Memory (MB)" in line and ":" in line:
                try:
                    max_memory_mb = float(line.split(":", 1)[1].strip())
                    max_memory = max_memory_mb / 1024  # Convert to GB
                except ValueError:
                    pass

            elif "Name:" in line and "gfx" in line.lower():
                # rocminfo prints the agent's "Name: gfx942" BEFORE its "Marketing Name",
                # so this must not be gated on current_gpu or the gfx version is never
                # captured and every GPU lands in the unknown-architecture branch.
                parts = line.split()
                for part in parts:
                    if part.lower().startswith("gfx"):
                        gfx_version = part
                        break

        # Don't forget the last GPU
        if current_gpu and compute_units and device_type == "GPU":
            gpu_key = f"{current_gpu}"
            if gfx_version:
                gpu_key += f" ({gfx_version})"

            base_clock = max_clock_freq if max_clock_freq else 1500

            microbenchmark_list.extend(
                _build_peak_performance_cards(gpu_key, gfx_version, compute_units,
                                              base_clock, max_memory)
            )

    # Add kernel benchmarks (GEMM, memory bandwidth, vector ops, convolution)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_cpp_file = os.path.join(script_dir, "gpu_kernel_benchmarks.cpp")
    kernel_exe_file = os.path.join(script_dir, "gpu_kernel_benchmarks")
    
    if os.path.exists(kernel_cpp_file):
        if verbose:
            print(f"Compiling GPU kernel benchmarks: {kernel_cpp_file}")
        compile_result = run_command(["hipcc", "-O3", "-o", kernel_exe_file, kernel_cpp_file], timeout=COMPILE_TIMEOUT)
        
        if compile_result is not None or os.path.exists(kernel_exe_file):
            if verbose:
                print(f"Running GPU kernel benchmarks: {kernel_exe_file}")
            kernel_output = run_command([kernel_exe_file], timeout=BENCHMARK_TIMEOUT, env=_hip_runtime_env())
            
            if kernel_output:
                try:
                    json_start = kernel_output.find("{")
                    if json_start != -1:
                        json_data = kernel_output[json_start:]
                        data = json.loads(json_data)
                        
                        if "error" in data:
                            error_info = OrderedDict()
                            error_info["Section"] = "Kernel Benchmarks - Error"
                            error_info["Status"] = data["error"]
                            microbenchmark_list.append(error_info)
                        elif "results" in data:
                            for result in data["results"]:
                                gpu_id = result.get("gpu_id", "?")
                                gpu_name = result.get("gpu_name", "Unknown")
                                
                                benchmark_info = OrderedDict()
                                benchmark_info["Section"] = f"GPU {gpu_id} Kernel Benchmarks"
                                benchmark_info["GPU Architecture"] = gpu_name
                                
                                if "memory_bandwidth_test" in result:
                                    bw_test = result["memory_bandwidth_test"]
                                    benchmark_info["Memory Bandwidth"] = f"{bw_test.get('bandwidth_gbps', 0):.2f} GB/s"
                                    benchmark_info["Memory Test Size"] = f"{bw_test.get('test_size_mb', 0)} MB"
                                
                                if "gemm_fp32_test" in result:
                                    gemm32 = result["gemm_fp32_test"]
                                    benchmark_info["GEMM FP32"] = f"{gemm32.get('gflops', 0):.2f} GFLOPS"
                                    benchmark_info["GEMM FP32 Matrix"] = f"{gemm32.get('matrix_size', 0)}x{gemm32.get('matrix_size', 0)}"
                                
                                if "gemm_fp64_test" in result:
                                    gemm64 = result["gemm_fp64_test"]
                                    benchmark_info["GEMM FP64"] = f"{gemm64.get('gflops', 0):.2f} GFLOPS"
                                    benchmark_info["GEMM FP64 Matrix"] = f"{gemm64.get('matrix_size', 0)}x{gemm64.get('matrix_size', 0)}"
                                
                                if "vector_add_test" in result:
                                    vec_add = result["vector_add_test"]
                                    benchmark_info["Vector Add"] = f"{vec_add.get('gflops', 0):.2f} GFLOPS"
                                
                                if "fma_throughput_test" in result:
                                    fma = result["fma_throughput_test"]
                                    benchmark_info["FMA Throughput"] = f"{fma.get('tflops', 0):.2f} TFLOPS"
                                
                                if "convolution_test" in result:
                                    conv = result["convolution_test"]
                                    benchmark_info["1D Convolution"] = f"{conv.get('gflops', 0):.2f} GFLOPS"
                                    benchmark_info["Conv Kernel Size"] = f"{conv.get('kernel_size', 0)}"

                                # Machine-readable numerics for charting and outlier detection.
                                # The display values above are formatted strings; these are not.
                                benchmark_info["_raw"] = {
                                    "gpu_id": gpu_id,
                                    "gpu_name": gpu_name,
                                    "memory_bandwidth_gbps": result.get("memory_bandwidth_test", {}).get("bandwidth_gbps"),
                                    "gemm_fp32_gflops": result.get("gemm_fp32_test", {}).get("gflops"),
                                    "gemm_fp64_gflops": result.get("gemm_fp64_test", {}).get("gflops"),
                                    "vector_add_gflops": result.get("vector_add_test", {}).get("gflops"),
                                    "fma_tflops": result.get("fma_throughput_test", {}).get("tflops"),
                                    "convolution_gflops": result.get("convolution_test", {}).get("gflops"),
                                }

                                microbenchmark_list.append(benchmark_info)
                except json.JSONDecodeError as e:
                    error_info = OrderedDict()
                    error_info["Section"] = "Kernel Benchmarks - Error"
                    error_info["Message"] = f"JSON parsing failed: {str(e)}"
                    microbenchmark_list.append(error_info)

    # Add GPU-to-GPU communication benchmarks if requested
    if include_p2p:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_file = os.path.join(script_dir, "gpu_p2p_bandwidth.cpp")
        exe_file = os.path.join(script_dir, "gpu_p2p_bandwidth")

        # Check if source file exists
        if not os.path.exists(cpp_file):
            error_info = OrderedDict()
            error_info["Section"] = "GPU P2P Communication - Error"
            error_info["Message"] = f"Source file not found: {cpp_file}"
            microbenchmark_list.append(error_info)
        else:
            # Try to compile the benchmark
            if verbose:
                print(f"Compiling GPU P2P benchmark: {cpp_file}")
            compile_result = run_command(["hipcc", "-o", exe_file, cpp_file], timeout=COMPILE_TIMEOUT)

            if compile_result is None and not os.path.exists(exe_file):
                error_info = OrderedDict()
                error_info["Section"] = "GPU P2P Communication - Error"
                error_info["Message"] = "Failed to compile (hipcc not found or compilation error)"
                microbenchmark_list.append(error_info)
            else:
                # Run the compiled benchmark
                if verbose:
                    print(f"Running GPU P2P benchmark: {exe_file}")
                p2p_output = run_command([exe_file], timeout=BENCHMARK_TIMEOUT, env=_hip_runtime_env())

                if not p2p_output:
                    error_info = OrderedDict()
                    error_info["Section"] = "GPU P2P Communication - Error"
                    error_info["Message"] = "Benchmark execution failed or produced no output"
                    microbenchmark_list.append(error_info)
                else:
                    try:
                        # Extract JSON from output
                        json_start = p2p_output.find("{")
                        if json_start != -1:
                            json_data = p2p_output[json_start:]
                            data = json.loads(json_data)

                            if "error" in data:
                                error_info = OrderedDict()
                                error_info["Section"] = "GPU P2P Communication"
                                error_info["Status"] = data["error"]
                                microbenchmark_list.append(error_info)
                            elif "results" in data:
                                # Collect pairs as we go so we can also emit one N x N
                                # matrix card, which is what the report heatmap consumes.
                                p2p_pairs = []
                                for result in data["results"]:
                                    p2p_info = OrderedDict()
                                    src_gpu = result.get("src_gpu", "?")
                                    dst_gpu = result.get("dst_gpu", "?")
                                    src_name = result.get("src_name", "Unknown")
                                    dst_name = result.get("dst_name", "Unknown")

                                    p2p_info["Section"] = f"GPU {src_gpu} → GPU {dst_gpu} Communication"
                                    p2p_info["Source GPU"] = f"GPU {src_gpu} ({src_name})"
                                    p2p_info["Destination GPU"] = f"GPU {dst_gpu} ({dst_name})"
                                    p2p_info["P2P Enabled"] = str(result.get("p2p_enabled", False))
                                    bandwidth = result.get("bandwidth_gbps", 0.0)
                                    if bandwidth > 0:
                                        p2p_info["Bandwidth"] = f"{bandwidth:.2f} GB/s"
                                    else:
                                        p2p_info["Bandwidth"] = "Not Available"

                                    p2p_info["_raw"] = {
                                        "src_gpu": src_gpu,
                                        "dst_gpu": dst_gpu,
                                        "p2p_enabled": bool(result.get("p2p_enabled", False)),
                                        "bandwidth_gbps": bandwidth if bandwidth > 0 else None,
                                    }
                                    p2p_pairs.append(p2p_info["_raw"])

                                    microbenchmark_list.append(p2p_info)

                                matrix_card = _build_p2p_matrix_card(
                                    p2p_pairs, _collect_gpu_link_types())
                                if matrix_card:
                                    microbenchmark_list.append(matrix_card)
                            else:
                                error_info = OrderedDict()
                                error_info["Section"] = "GPU P2P Communication - Error"
                                error_info["Message"] = "Invalid JSON format - no results found"
                                microbenchmark_list.append(error_info)
                    except json.JSONDecodeError as e:
                        error_info = OrderedDict()
                        error_info["Section"] = "GPU P2P Communication - Error"
                        error_info["Message"] = f"JSON parsing failed: {str(e)}"
                        error_info["Raw Output"] = p2p_output[:500]  # First 500 chars
                        microbenchmark_list.append(error_info)

        # Add GPU-CPU (Host) bandwidth benchmarks
        host_cpp_file = os.path.join(script_dir, "gpu_host_bandwidth.cpp")
        host_exe_file = os.path.join(script_dir, "gpu_host_bandwidth")

        if os.path.exists(host_cpp_file):
            if verbose:
                print(f"Compiling GPU-CPU bandwidth benchmark: {host_cpp_file}")
            compile_result = run_command(["hipcc", "-o", host_exe_file, host_cpp_file], timeout=COMPILE_TIMEOUT)

            if compile_result is not None or os.path.exists(host_exe_file):
                if verbose:
                    print(f"Running GPU-CPU bandwidth benchmark: {host_exe_file}")
                host_output = run_command([host_exe_file], timeout=BENCHMARK_TIMEOUT, env=_hip_runtime_env())

                if host_output:
                    try:
                        json_start = host_output.find("{")
                        if json_start != -1:
                            json_data = host_output[json_start:]
                            data = json.loads(json_data)

                            if "error" in data:
                                error_info = OrderedDict()
                                error_info["Section"] = "GPU-CPU Transfer Bandwidth - Error"
                                error_info["Status"] = data["error"]
                                microbenchmark_list.append(error_info)
                            elif "results" in data:
                                for result in data["results"]:
                                    gpu_id = result.get("gpu", "?")
                                    gpu_name = result.get("gpu_name", "Unknown")

                                    host_bw_info = OrderedDict()
                                    host_bw_info["Section"] = f"GPU {gpu_id} Host Transfer Bandwidth"
                                    host_bw_info["GPU Architecture"] = gpu_name
                                    host_bw_info["Host→Device (Pageable)"] = f"{result.get('h2d_pageable_gbps', 0):.2f} GB/s"
                                    host_bw_info["Device→Host (Pageable)"] = f"{result.get('d2h_pageable_gbps', 0):.2f} GB/s"
                                    host_bw_info["Host→Device (Pinned)"] = f"{result.get('h2d_pinned_gbps', 0):.2f} GB/s"
                                    host_bw_info["Device→Host (Pinned)"] = f"{result.get('d2h_pinned_gbps', 0):.2f} GB/s"

                                    host_bw_info["_raw"] = {
                                        "gpu_id": gpu_id,
                                        "gpu_name": gpu_name,
                                        "h2d_pageable_gbps": result.get("h2d_pageable_gbps"),
                                        "d2h_pageable_gbps": result.get("d2h_pageable_gbps"),
                                        "h2d_pinned_gbps": result.get("h2d_pinned_gbps"),
                                        "d2h_pinned_gbps": result.get("d2h_pinned_gbps"),
                                    }

                                    microbenchmark_list.append(host_bw_info)
                    except json.JSONDecodeError as e:
                        error_info = OrderedDict()
                        error_info["Section"] = "GPU-CPU Transfer Bandwidth - Error"
                        error_info["Message"] = f"JSON parsing failed: {str(e)}"
                        microbenchmark_list.append(error_info)

        # Add GPU topology analysis (XGMI/Infinity Fabric)
        topology_cpp_file = os.path.join(script_dir, "gpu_topology.cpp")
        topology_exe_file = os.path.join(script_dir, "gpu_topology")

        if os.path.exists(topology_cpp_file):
            if verbose:
                print(f"Compiling GPU topology analysis: {topology_cpp_file}")
            compile_result = run_command(["hipcc", "-o", topology_exe_file, topology_cpp_file], timeout=COMPILE_TIMEOUT)

            if compile_result is not None or os.path.exists(topology_exe_file):
                if verbose:
                    print(f"Running GPU topology analysis: {topology_exe_file}")
                topology_output = run_command([topology_exe_file], timeout=BENCHMARK_TIMEOUT, env=_hip_runtime_env())

                if topology_output:
                    try:
                        json_start = topology_output.find("{")
                        if json_start != -1:
                            json_data = topology_output[json_start:]
                            data = json.loads(json_data)

                            if "error" in data:
                                error_info = OrderedDict()
                                error_info["Section"] = "GPU Topology (XGMI/Infinity Fabric) - Error"
                                error_info["Status"] = data["error"]
                                microbenchmark_list.append(error_info)
                            elif "bandwidth_matrix" in data:
                                # Add topology summary
                                topo_summary = OrderedDict()
                                topo_summary["Section"] = "GPU Topology Summary"
                                topo_summary["GPU Count"] = str(data.get("gpu_count", 0))
                                
                                # Count link types
                                xgmi_links = 0
                                pcie_links = 0
                                no_p2p_links = 0
                                
                                for row in data["bandwidth_matrix"]:
                                    for link in row:
                                        link_type = link.get("link_type", "")
                                        if "XGMI" in link_type and link_type != "Self":
                                            xgmi_links += 1
                                        elif link_type == "PCIe":
                                            pcie_links += 1
                                        elif link_type == "No P2P":
                                            no_p2p_links += 1
                                
                                topo_summary["XGMI Links"] = str(xgmi_links)
                                topo_summary["PCIe Links"] = str(pcie_links)
                                if no_p2p_links > 0:
                                    topo_summary["No P2P Links"] = str(no_p2p_links)

                                # Full link matrix in numeric form, for the report heatmap
                                # and for link-type uniformity checks.
                                topo_summary["_raw"] = {
                                    "gpu_count": data.get("gpu_count", 0),
                                    "xgmi_links": xgmi_links,
                                    "pcie_links": pcie_links,
                                    "no_p2p_links": no_p2p_links,
                                    "link_matrix": [
                                        [
                                            {
                                                "dst": link.get("dst"),
                                                "link_type": link.get("link_type"),
                                                "hops": link.get("hops"),
                                                "bandwidth_gbps": link.get("bandwidth_gbps"),
                                            }
                                            for link in row
                                        ]
                                        for row in data["bandwidth_matrix"]
                                    ],
                                }

                                microbenchmark_list.append(topo_summary)
                                
                                # Add detailed bandwidth matrix for each GPU
                                for i, row in enumerate(data["bandwidth_matrix"]):
                                    gpu_info = OrderedDict()
                                    gpu_name = "Unknown"
                                    
                                    # Get GPU name from gpus list
                                    if "gpus" in data and i < len(data["gpus"]):
                                        gpu_name = data["gpus"][i].get("name", "Unknown")
                                    
                                    gpu_info["Section"] = f"GPU {i} Topology Links"
                                    gpu_info["GPU Architecture"] = gpu_name
                                    
                                    for link in row:
                                        dst = link.get("dst", "?")
                                        if dst != i:  # Skip self-links
                                            link_type = link.get("link_type", "Unknown")
                                            bandwidth = link.get("bandwidth_gbps", 0.0)
                                            hops = link.get("hops", -1)
                                            
                                            link_desc = f"{link_type}"
                                            if hops > 0:
                                                link_desc += f" ({hops} hops)"
                                            if bandwidth > 0:
                                                link_desc += f" - {bandwidth:.2f} GB/s"
                                            
                                            gpu_info[f"→ GPU {dst}"] = link_desc
                                    
                                    if len(gpu_info) > 2:  # Only add if there are actual links
                                        microbenchmark_list.append(gpu_info)
                                        
                    except json.JSONDecodeError as e:
                        error_info = OrderedDict()
                        error_info["Section"] = "GPU Topology (XGMI/Infinity Fabric) - Error"
                        error_info["Message"] = f"JSON parsing failed: {str(e)}"
                        microbenchmark_list.append(error_info)

        # Add Storage I/O Profiling
        storage_py_file = os.path.join(script_dir, "storage_benchmark.py")
        
        if os.path.exists(storage_py_file):
            if verbose:
                print(f"Running storage I/O profiling: {storage_py_file}")
            storage_output = run_command(["python3", storage_py_file], timeout=BENCHMARK_TIMEOUT)
            
            if storage_output:
                try:
                    # Find JSON in output
                    json_start = storage_output.find("{")
                    if json_start != -1:
                        json_data = storage_output[json_start:]
                        data = json.loads(json_data)
                        
                        # Storage Devices Summary
                        if data.get("storage_devices"):
                            storage_summary = OrderedDict()
                            storage_summary["Section"] = "Storage Devices Detected"
                            
                            ssd_count = sum(1 for d in data["storage_devices"] if not d.get("rotational"))
                            hdd_count = sum(1 for d in data["storage_devices"] if d.get("rotational"))
                            
                            storage_summary["Total Devices"] = str(len(data["storage_devices"]))
                            if ssd_count > 0:
                                storage_summary["SSD/NVMe Devices"] = str(ssd_count)
                            if hdd_count > 0:
                                storage_summary["HDD Devices"] = str(hdd_count)
                            
                            microbenchmark_list.append(storage_summary)
                            
                            # Individual device details
                            for device in data["storage_devices"]:
                                dev_info = OrderedDict()
                                dev_info["Section"] = f"Storage: {device.get('name', 'Unknown')}"
                                dev_info["Model"] = device.get("model", "Unknown")
                                dev_info["Size"] = device.get("size", "Unknown")
                                dev_info["Type"] = device.get("type", "Unknown")
                                dev_info["Transport"] = device.get("transport", "Unknown")
                                microbenchmark_list.append(dev_info)
                        
                        # NVMe Devices
                        if data.get("nvme_devices"):
                            for nvme in data["nvme_devices"]:
                                nvme_info = OrderedDict()
                                nvme_info["Section"] = f"NVMe: {nvme.get('Device', 'Unknown')}"
                                nvme_info["Model"] = nvme.get("Model", "Unknown")
                                nvme_info["Size"] = nvme.get("Size", "Unknown")
                                nvme_info["Serial Number"] = nvme.get("Serial", "Unknown")
                                nvme_info["Firmware"] = nvme.get("Firmware", "Unknown")
                                nvme_info["Namespace"] = nvme.get("Namespace", "Unknown")
                                microbenchmark_list.append(nvme_info)
                        
                        # RAID Configuration
                        if data.get("raid_configs"):
                            raid_summary = OrderedDict()
                            raid_summary["Section"] = "RAID Configuration Detected"
                            raid_summary["Arrays Found"] = str(len(data["raid_configs"]))
                            microbenchmark_list.append(raid_summary)
                            
                            for raid in data["raid_configs"]:
                                raid_info = OrderedDict()
                                if "Array Device" in raid:
                                    raid_info["Section"] = f"RAID: {raid.get('Array Device', 'Unknown')}"
                                elif "LVM Volume" in raid:
                                    raid_info["Section"] = f"LVM RAID: {raid.get('LVM Volume', 'Unknown')}"
                                
                                for key, value in raid.items():
                                    if key not in ["Section", "Array Device", "LVM Volume"]:
                                        raid_info[key] = value
                                
                                microbenchmark_list.append(raid_info)
                        
                        # GPU Direct Storage (GDS) Capability
                        if data.get("gds_capability"):
                            gds_info = OrderedDict()
                            gds_info["Section"] = "GPU Direct Storage (GDS) Capability"
                            
                            for key, value in data["gds_capability"].items():
                                gds_info[key] = value
                            
                            microbenchmark_list.append(gds_info)
                        
                        # Disk Benchmark Results (if any)
                        if data.get("benchmark_results"):
                            for bench in data["benchmark_results"]:
                                bench_info = OrderedDict()
                                bench_info["Section"] = f"Storage Benchmark: {bench.get('device', 'Unknown')}"
                                
                                for key, value in bench.items():
                                    if key != "device":
                                        bench_info[key] = value
                                
                                microbenchmark_list.append(bench_info)
                                
                except json.JSONDecodeError as e:
                    error_info = OrderedDict()
                    error_info["Section"] = "Storage I/O Profiling - Error"
                    error_info["Message"] = f"JSON parsing failed: {str(e)}"
                    microbenchmark_list.append(error_info)

        # Add Network Performance Testing
        network_py_file = os.path.join(script_dir, "network_benchmark.py")
        
        if os.path.exists(network_py_file):
            if verbose:
                print(f"Running network performance testing: {network_py_file}")
            network_output = run_command(["python3", network_py_file], timeout=BENCHMARK_TIMEOUT)
            
            if network_output:
                try:
                    # Find JSON in output
                    json_start = network_output.find("{")
                    if json_start != -1:
                        json_data = network_output[json_start:]
                        data = json.loads(json_data)
                        
                        # RDMA Devices
                        if data.get("rdma_devices"):
                            rdma_summary = OrderedDict()
                            rdma_summary["Section"] = "RDMA/InfiniBand Devices Detected"
                            rdma_summary["Devices Found"] = str(len(data["rdma_devices"]))
                            microbenchmark_list.append(rdma_summary)
                            
                            for rdma in data["rdma_devices"]:
                                rdma_info = OrderedDict()
                                rdma_info["Section"] = f"RDMA: {rdma.get('Device', 'Unknown')}"
                                
                                for key, value in rdma.items():
                                    if key != "Device":
                                        rdma_info[key] = value
                                
                                microbenchmark_list.append(rdma_info)
                        
                        # RoCE Capability
                        if data.get("roce_capability"):
                            roce_info = OrderedDict()
                            roce_info["Section"] = "RoCE (RDMA over Converged Ethernet) Capability"
                            
                            for key, value in data["roce_capability"].items():
                                roce_info[key] = value
                            
                            microbenchmark_list.append(roce_info)
                        
                        # Network Topology
                        if data.get("network_topology"):
                            topo_info = OrderedDict()
                            topo_info["Section"] = "Network Topology Information"
                            
                            for key, value in data["network_topology"].items():
                                topo_info[key] = value
                            
                            microbenchmark_list.append(topo_info)
                        
                        # Bandwidth Tools
                        if data.get("bandwidth_tools"):
                            bw_info = OrderedDict()
                            bw_info["Section"] = "Network Bandwidth Testing Tools"
                            
                            for key, value in data["bandwidth_tools"].items():
                                bw_info[key] = value
                            
                            microbenchmark_list.append(bw_info)
                        
                        # MPI Benchmarks
                        if data.get("mpi_benchmarks"):
                            mpi_info = OrderedDict()
                            mpi_info["Section"] = "MPI Benchmark Tools"
                            
                            for key, value in data["mpi_benchmarks"].items():
                                mpi_info[key] = value
                            
                            microbenchmark_list.append(mpi_info)
                                
                except json.JSONDecodeError as e:
                    error_info = OrderedDict()
                    error_info["Section"] = "Network Performance Testing - Error"
                    error_info["Message"] = f"JSON parsing failed: {str(e)}"
                    microbenchmark_list.append(error_info)

    # Heavyweight library benchmarks, only under --full. Both saturate every GPU on
    # the node, so they must never run by default on a machine that might be busy.
    if full:
        # Snapshot the peak cards before appending: the GEMM efficiency ratio reads
        # them, and they are all that is in the list at this point that carries peaks.
        peak_cards = list(microbenchmark_list)

        gemm_cpp_file = os.path.join(script_dir, "gpu_gemm_library.cpp")
        gemm_exe_file = os.path.join(script_dir, "gpu_gemm_library")
        if os.path.exists(gemm_cpp_file):
            if verbose:
                print(f"Compiling library GEMM benchmark: {gemm_cpp_file}")
            compile_result = run_command(
                ["hipcc", "-O3", "-o", gemm_exe_file, gemm_cpp_file, "-lrocblas"],
                timeout=COMPILE_TIMEOUT)

            if compile_result is None and not os.path.exists(gemm_exe_file):
                error_info = OrderedDict()
                error_info["Section"] = "Library GEMM - Error"
                error_info["Message"] = ("Failed to compile (rocBLAS development files "
                                         "may not be installed)")
                microbenchmark_list.append(error_info)
            else:
                if verbose:
                    print(f"Running library GEMM benchmark: {gemm_exe_file}")
                gemm_output = run_command([gemm_exe_file], timeout=BENCHMARK_TIMEOUT,
                                          env=_hip_runtime_env())
                if not gemm_output:
                    error_info = OrderedDict()
                    error_info["Section"] = "Library GEMM - Error"
                    error_info["Message"] = "Benchmark execution failed or produced no output"
                    microbenchmark_list.append(error_info)
                else:
                    try:
                        json_start = gemm_output.find("{")
                        data = json.loads(gemm_output[json_start:]) if json_start != -1 else {}
                        if "error" in data:
                            error_info = OrderedDict()
                            error_info["Section"] = "Library GEMM - Error"
                            error_info["Status"] = data["error"]
                            microbenchmark_list.append(error_info)
                        else:
                            microbenchmark_list.extend(
                                _build_library_gemm_cards(data, peak_cards))
                    except json.JSONDecodeError as e:
                        error_info = OrderedDict()
                        error_info["Section"] = "Library GEMM - Error"
                        error_info["Message"] = f"JSON parsing failed: {str(e)}"
                        microbenchmark_list.append(error_info)

        rccl_cpp_file = os.path.join(script_dir, "gpu_rccl_collectives.cpp")
        rccl_exe_file = os.path.join(script_dir, "gpu_rccl_collectives")
        if os.path.exists(rccl_cpp_file):
            if not _rccl_available():
                error_info = OrderedDict()
                error_info["Section"] = "RCCL Collectives - Skipped"
                error_info["Message"] = ("librccl.so not found; install RCCL to measure "
                                         "collective bandwidth")
                microbenchmark_list.append(error_info)
            else:
                if verbose:
                    print(f"Compiling RCCL collectives benchmark: {rccl_cpp_file}")
                compile_result = run_command(
                    ["hipcc", "-O3", "-o", rccl_exe_file, rccl_cpp_file, "-lrccl"],
                    timeout=COMPILE_TIMEOUT)

                if compile_result is None and not os.path.exists(rccl_exe_file):
                    error_info = OrderedDict()
                    error_info["Section"] = "RCCL Collectives - Error"
                    error_info["Message"] = "Failed to compile (RCCL headers may not be installed)"
                    microbenchmark_list.append(error_info)
                else:
                    if verbose:
                        print(f"Running RCCL collectives benchmark: {rccl_exe_file}")
                    rccl_output = run_command([rccl_exe_file], timeout=BENCHMARK_TIMEOUT,
                                              env=_hip_runtime_env())
                    if not rccl_output:
                        error_info = OrderedDict()
                        error_info["Section"] = "RCCL Collectives - Error"
                        error_info["Message"] = "Benchmark execution failed or produced no output"
                        microbenchmark_list.append(error_info)
                    else:
                        try:
                            json_start = rccl_output.find("{")
                            data = json.loads(rccl_output[json_start:]) if json_start != -1 else {}
                            if "skipped" in data:
                                # A single-GPU host is a valid configuration, not a fault.
                                info = OrderedDict()
                                info["Section"] = "RCCL Collectives - Skipped"
                                info["Message"] = data["skipped"]
                                microbenchmark_list.append(info)
                            elif "error" in data:
                                error_info = OrderedDict()
                                error_info["Section"] = "RCCL Collectives - Error"
                                error_info["Status"] = data["error"]
                                microbenchmark_list.append(error_info)
                            else:
                                microbenchmark_list.extend(_build_rccl_cards(data))
                        except json.JSONDecodeError as e:
                            error_info = OrderedDict()
                            error_info["Section"] = "RCCL Collectives - Error"
                            error_info["Message"] = f"JSON parsing failed: {str(e)}"
                            microbenchmark_list.append(error_info)

    if microbenchmark_list:
        details["linux"] = microbenchmark_list

    return details

def gather_rocm_details() -> Dict[str, List[Dict[str, str]]]:
    """Gather ROCm-specific information."""
    details: Dict[str, List[Dict[str, str]]] = OrderedDict()
    system = platform.system().lower()

    # Only gather ROCm info on Linux (ROCm is primarily Linux-based)
    if system != "linux":
        return details

    rocm_info_list = []

    # ROCm Version Information
    version_info = OrderedDict()
    version_info["Section"] = "Version Information"

    # Check for ROCm installation path
    rocm_path = os.environ.get("ROCM_PATH") or "/opt/rocm"
    if os.path.exists(rocm_path):
        version_info["ROCm Installation Path"] = rocm_path

        # Check .info file for version
        info_file = os.path.join(rocm_path, ".info", "version")
        if os.path.exists(info_file):
            try:
                with open(info_file, "r", encoding="utf-8") as f:
                    rocm_ver = f.read().strip()
                    version_info["ROCm Version"] = rocm_ver
            except Exception:
                pass

    # HIP Version
    hipcc_version = run_command(["hipcc", "--version"])
    if hipcc_version:
        for line in hipcc_version.splitlines():
            if "HIP version" in line or "hipcc" in line.lower():
                version_info["HIP Compiler Version"] = line.strip()
                break

    # ROCm SMI Version
    rocm_smi_version = run_command(["rocm-smi", "--version"])
    if rocm_smi_version:
        version_info["rocm-smi Version"] = rocm_smi_version.strip()

    # AMD SMI Version
    amd_smi_version = run_command(["amd-smi", "version"])
    if amd_smi_version:
        version_info["amd-smi Version"] = amd_smi_version.strip()

    if len(version_info) > 1:
        rocm_info_list.append(version_info)

    # Environment Variables
    env_info = OrderedDict()
    env_info["Section"] = "Environment Variables"
    rocm_env_vars = [
        "ROCM_PATH", "ROCM_HOME", "HIP_PATH", "HIP_PLATFORM",
        "HSA_PATH", "ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL", "HSA_OVERRIDE_GFX_VERSION",
        "ROCM_VERSION", "HIP_COMPILER"
    ]

    for var in rocm_env_vars:
        value = os.environ.get(var)
        if value:
            env_info[var] = value

    if len(env_info) > 1:
        rocm_info_list.append(env_info)

    # ROCm Libraries and Packages
    packages_info = OrderedDict()
    packages_info["Section"] = "Installed ROCm Packages"

    # Check dpkg for ROCm packages (Debian/Ubuntu)
    dpkg_list = run_command(["dpkg", "-l"])
    if dpkg_list:
        rocm_packages = []
        for line in dpkg_list.splitlines():
            if "rocm" in line.lower() or "hip" in line.lower() or "hsa" in line.lower():
                parts = line.split()
                if len(parts) >= 3 and parts[0] == "ii":
                    pkg_name = parts[1]
                    pkg_version = parts[2]
                    rocm_packages.append(f"{pkg_name} ({pkg_version})")

        if rocm_packages:
            packages_info["Debian Packages"] = ", ".join(rocm_packages[:20])  # Limit to first 20

    # Check rpm for ROCm packages (RHEL/CentOS)
    if "Debian Packages" not in packages_info:
        rpm_list = run_command(["rpm", "-qa"])
        if rpm_list:
            rocm_packages = []
            for line in rpm_list.splitlines():
                if "rocm" in line.lower() or "hip" in line.lower() or "hsa" in line.lower():
                    rocm_packages.append(line.strip())

            if rocm_packages:
                packages_info["RPM Packages"] = ", ".join(rocm_packages[:20])  # Limit to first 20

    if len(packages_info) > 1:
        rocm_info_list.append(packages_info)

    # HSA Runtime Information
    hsa_info = OrderedDict()
    hsa_info["Section"] = "HSA Runtime Information"

    # Run rocminfo for HSA details
    rocminfo_output = run_command(["rocminfo"])
    if rocminfo_output:
        in_system_section = False
        for line in rocminfo_output.splitlines():
            line = line.strip()

            if line.startswith("=====") or line.startswith("*****"):
                in_system_section = "System" in line or "Runtime" in line
                continue

            if in_system_section and ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                if key in ["Runtime Version", "System Version", "HSA Runtime Version",
                          "Runtime Name", "Version", "Timestamp"]:
                    hsa_info[key] = value

    if len(hsa_info) > 1:
        rocm_info_list.append(hsa_info)

    # ROCm Compute Capabilities
    compute_info = OrderedDict()
    compute_info["Section"] = "ROCm Compute Details"

    # Check for ROCm-capable devices
    clinfo = run_command(["clinfo"])
    if clinfo:
        opencl_devices = []
        for line in clinfo.splitlines():
            line = line.strip()
            if "Device Name" in line and ":" in line:
                device_name = line.split(":", 1)[1].strip()
                if "AMD" in device_name or "Radeon" in device_name:
                    opencl_devices.append(device_name)

        if opencl_devices:
            compute_info["OpenCL Devices"] = ", ".join(opencl_devices)

    # HIP Runtime Info
    hipconfig = run_command(["hipconfig", "--full"])
    if hipconfig:
        for line in hipconfig.splitlines():
            if ":" in line:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key in ["HIP_PLATFORM", "HIP_COMPILER", "HIP_RUNTIME", "HIP_PATH"]:
                        compute_info[key] = value

    if len(compute_info) > 1:
        rocm_info_list.append(compute_info)

    # Kernel Module Information
    kernel_info = OrderedDict()
    kernel_info["Section"] = "Kernel Module Information"

    # Check loaded AMD GPU kernel modules
    lsmod = run_command(["lsmod"])
    if lsmod:
        amd_modules = []
        for line in lsmod.splitlines():
            if any(mod in line.lower() for mod in ["amdgpu", "amd_iommu", "kfd", "amdkfd"]):
                parts = line.split()
                if parts:
                    module_name = parts[0]
                    module_size = parts[1] if len(parts) > 1 else "N/A"
                    amd_modules.append(f"{module_name} ({module_size})")

        if amd_modules:
            kernel_info["Loaded AMD Modules"] = ", ".join(amd_modules)

    # Check modinfo for amdgpu
    modinfo_amdgpu = run_command(["modinfo", "amdgpu"])
    if modinfo_amdgpu:
        for line in modinfo_amdgpu.splitlines():
            if line.startswith("version:"):
                kernel_info["amdgpu Driver Version"] = line.split(":", 1)[1].strip()
            elif line.startswith("firmware:"):
                fw = line.split(":", 1)[1].strip()
                if "Firmware" not in kernel_info:
                    kernel_info["Firmware"] = fw
                break

    if len(kernel_info) > 1:
        rocm_info_list.append(kernel_info)

    rocm_smi_showall = OrderedDict()
    rocm_smi_showall["Section"] = "rocm-smi --showall"
    
    showall_output = run_command(["rocm-smi", "--showall"])
    if showall_output:
        lines = showall_output.strip().splitlines()
        current_gpu = None
        gpu_data = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            
            if line.startswith("GPU["):
                if current_gpu and gpu_data:
                    rocm_smi_showall[current_gpu] = str(gpu_data)
                    gpu_data = {}
                current_gpu = line.split("]")[0] + "]"
            elif ":" in line and current_gpu:
                key, value = line.split(":", 1)
                gpu_data[key.strip()] = value.strip()
        
        if current_gpu and gpu_data:
            rocm_smi_showall[current_gpu] = str(gpu_data)
    
    if len(rocm_smi_showall) > 1:
        rocm_info_list.append(rocm_smi_showall)

    rocm_smi_showinfo = OrderedDict()
    rocm_smi_showinfo["Section"] = "rocm-smi --showinfo"
    
    showinfo_output = run_command(["rocm-smi", "--showinfo"])
    if showinfo_output:
        lines = showinfo_output.strip().splitlines()
        current_gpu = None
        gpu_data = {}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("="):
                continue
            
            if line.startswith("GPU["):
                if current_gpu and gpu_data:
                    rocm_smi_showinfo[current_gpu] = str(gpu_data)
                    gpu_data = {}
                current_gpu = line.split("]")[0] + "]"
            elif ":" in line and current_gpu:
                key, value = line.split(":", 1)
                gpu_data[key.strip()] = value.strip()
        
        if current_gpu and gpu_data:
            rocm_smi_showinfo[current_gpu] = str(gpu_data)
    
    if len(rocm_smi_showinfo) > 1:
        rocm_info_list.append(rocm_smi_showinfo)

    smc_version_info = OrderedDict()
    smc_version_info["Section"] = "SMC Version Information"
    
    smc_output = run_command(["rocm-smi", "--showfwinfo"])
    if smc_output:
        lines = smc_output.strip().splitlines()
        for line in lines:
            line = line.strip()
            if "SMC" in line.upper() and ":" in line:
                key, value = line.split(":", 1)
                smc_version_info[key.strip()] = value.strip()
            elif line.startswith("GPU[") and "SMC" in line.upper():
                parts = line.split()
                for i, part in enumerate(parts):
                    if "SMC" in part.upper() and i + 1 < len(parts):
                        gpu_id = line.split("]")[0] + "]"
                        smc_version_info[f"{gpu_id} SMC Version"] = parts[i + 1]
    
    if len(smc_version_info) > 1:
        rocm_info_list.append(smc_version_info)

    if rocm_info_list:
        details["linux"] = rocm_info_list

    return details

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AMD Rapido - Collect server hardware information (CPU, GPU, Network, ROCm)"
    )
    parser.add_argument(
        "-c",
        "--cpu",
        action="store_true",
        help="Collect CPU information only",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        action="store_true",
        help="Collect GPU information only",
    )
    parser.add_argument(
        "-n",
        "--network",
        action="store_true",
        help="Collect network information only",
    )
    parser.add_argument(
        "-b",
        "--bmc",
        action="store_true",
        help="Collect BMC information only",
    )
    parser.add_argument(
        "-r",
        "--rocm",
        action="store_true",
        help="Collect ROCm information only",
    )
    parser.add_argument(
        "-m",
        "--microbenchmarks",
        action="store_true",
        help="Collect GPU microbenchmarks only (automatically includes ROCm info)",
    )
    parser.add_argument(
        "-t",
        "--platform",
        action="store_true",
        help="Collect host platform information only (BIOS/DMI, EDAC, NUMA, kernel tuning, GPU/NIC affinity)",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Collect all basic information: CPU, GPU, Network, BMC, ROCm, Platform (does NOT include microbenchmarks - use -m)",
    )
    parser.add_argument(
        "-p",
        "--p2p",
        action="store_true",
        help="Include GPU-to-GPU peer-to-peer communication bandwidth tests (requires -m flag)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the heavy library benchmarks as well: rocBLAS achieved GEMM and RCCL collectives "
             "(requires -m flag; adds several minutes to the run)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show tool availability check and progress messages)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Write the JSON to this exact path (overrides the default serverinfo_<hostname>.json)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory to write the JSON into (created if it does not exist)",
    )
    parser.add_argument(
        "--tag",
        metavar="TAG",
        help="Suffix for the output filename, e.g. --tag before -> serverinfo_<hostname>_before.json. "
             "Useful for collecting the same host twice without overwriting.",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    # Determine which sections to collect
    # If no specific flags are provided, or -a is used, collect all basic sections (NOT microbenchmarks)
    collect_all = args.all or not (args.cpu or args.gpu or args.network or args.bmc or args.rocm
                                   or args.platform or args.microbenchmarks)

    collect_cpu = collect_all or args.cpu
    collect_gpu = collect_all or args.gpu
    collect_network = collect_all or args.network
    collect_bmc = collect_all or args.bmc
    collect_rocm = collect_all or args.rocm or args.microbenchmarks  # ROCm is collected with microbenchmarks
    collect_platform = collect_all or args.platform
    # Microbenchmarks ONLY run when explicitly requested with -m flag
    collect_microbenchmarks = args.microbenchmarks

    # Check and report tool availability at the beginning
    check_tool_availability(verbose=args.verbose)

    # Warn if -p is used without -m
    if args.p2p and not collect_microbenchmarks:
        if args.verbose:
            print("Warning: -p/--p2p flag requires -m/--microbenchmarks flag to be effective")
            print("GPU P2P communication tests will be skipped. Use: python rapido-collect.py -m -p")

    # Same reasoning as -p: --full only selects extra benchmarks, it does not turn
    # benchmarking on, so on its own it does nothing and the user should hear about it.
    if args.full and not collect_microbenchmarks:
        if args.verbose:
            print("Warning: --full flag requires -m/--microbenchmarks flag to be effective")
            print("Library GEMM and RCCL collective tests will be skipped. Use: python rapido-collect.py -m --full")


    if args.verbose:
        print("Starting data collection...")
        print()

    # Collect data based on flags with error handling
    cpu_details = {}
    gpu_details = {}
    network_details = {}
    bmc_details = {}
    rocm_details = {}
    ras_details = {}
    telemetry_details = {}
    platform_details = {}
    microbenchmark_details = {}
    collection_errors = []
    
    try:
        if collect_cpu:
            try:
                cpu_details = gather_cpu_details()
            except Exception as e:
                collection_errors.append(f"CPU collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: CPU collection failed: {str(e)}")
        
        if collect_gpu:
            try:
                gpu_details = gather_gpu_details()
            except Exception as e:
                collection_errors.append(f"GPU collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: GPU collection failed: {str(e)}")

            # RAS and telemetry are separate try blocks, not part of the one above:
            # they are independent amd-smi subcommands, and losing the whole GPU
            # inventory because one counter query misbehaved would be a bad trade.
            try:
                ras_details = gather_ras_health()
            except Exception as e:
                collection_errors.append(f"RAS health collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: RAS health collection failed: {str(e)}")

            try:
                # gpu_details supplies the max PCIe width/speed that the current
                # trained link is compared against, so this must run after it.
                telemetry_details = gather_gpu_telemetry(gpu_details)
            except Exception as e:
                collection_errors.append(f"GPU telemetry collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: GPU telemetry collection failed: {str(e)}")

        if collect_platform:
            try:
                platform_details = gather_platform_details()
            except Exception as e:
                collection_errors.append(f"Platform collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: Platform collection failed: {str(e)}")


        if collect_network:
            try:
                network_details = gather_network_details()
            except Exception as e:
                collection_errors.append(f"Network collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: Network collection failed: {str(e)}")
        
        if collect_bmc:
            try:
                bmc_details = gather_bmc_info()
            except Exception as e:
                collection_errors.append(f"BMC collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: BMC collection failed: {str(e)}")
        
        if collect_rocm:
            try:
                rocm_details = gather_rocm_details()
            except Exception as e:
                collection_errors.append(f"ROCm collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: ROCm collection failed: {str(e)}")
        
        if collect_microbenchmarks:
            try:
                microbenchmark_details = gather_gpu_microbenchmarks(
                    include_p2p=args.p2p, verbose=args.verbose, full=args.full)
            except Exception as e:
                collection_errors.append(f"Microbenchmark collection failed: {str(e)}")
                if args.verbose:
                    print(f"Warning: Microbenchmark collection failed: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user (Ctrl+C)")
        print("Saving partial data collected so far...")
        collection_errors.append("Collection interrupted by user")

    # Build payload with metadata
    payload = OrderedDict()
    
    # Add collection metadata
    import datetime
    import sys
    payload["_metadata"] = {
        "collection_date": datetime.datetime.now().isoformat(),
        "collection_status": "partial" if collection_errors else "complete",
        "errors": collection_errors if collection_errors else [],
        "command_line": ' '.join(sys.argv),
        # A misconfigured resolver can make gethostname() raise; losing the whole
        # collected payload over the machine's name would be absurd.
        "hostname": _safe_hostname(),
        # Commands that timed out or errored, so a blank section can be explained
        # rather than being mistaken for "this machine has none of that hardware".
        "command_failures": COMMAND_FAILURES,
    }
    
    payload["cpu"] = cpu_details
    if gpu_details:
        payload["gpu"] = gpu_details
    else:
        payload["gpu"] = []
    if network_details:
        payload["network"] = network_details
    else:
        payload["network"] = []
    if bmc_details:
        payload["bmc"] = bmc_details
    else:
        payload["bmc"] = []
    if rocm_details:
        payload["rocm"] = rocm_details
    else:
        payload["rocm"] = []
    # Unlike the sections above, these are omitted entirely when empty rather than
    # written as []: an absent key makes the report drop the tab, whereas an empty
    # list on a host without amd-smi would render an empty tab that looks broken.
    if ras_details:
        payload["ras"] = ras_details
    if telemetry_details:
        payload["telemetry"] = telemetry_details
    if platform_details:
        payload["platform"] = platform_details
    if microbenchmark_details:
        payload["microbenchmarks"] = microbenchmark_details

    # Determine the output path: an explicit --output wins, otherwise build
    # serverinfo_<hostname>[_<tag>].json inside --output-dir (default: current directory).
    if args.output:
        # --output names the file outright, so the pieces the name would have been
        # built from no longer apply. Say so rather than ignoring them silently.
        ignored = [flag for flag, value in (("--tag", args.tag), ("--output-dir", args.output_dir)) if value]
        if ignored:
            print(f"Note: {' and '.join(ignored)} ignored because --output specifies the full path")
        filename = args.output
    else:
        hostname = _safe_hostname()
        # Sanitize hostname for filename (replace invalid characters)
        safe_hostname = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in hostname)
        basename = f"serverinfo_{safe_hostname}" if safe_hostname else "serverinfo"

        if args.tag:
            safe_tag = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in args.tag)
            basename = f"{basename}_{safe_tag}"

        filename = os.path.join(args.output_dir, f"{basename}.json") if args.output_dir else f"{basename}.json"

    # Create the target directory if the user pointed somewhere that does not exist yet
    out_dir = os.path.dirname(os.path.abspath(filename))
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            print(f"\nERROR: Could not create output directory {out_dir}: {e}")
            raise SystemExit(1)

    # Write JSON with error handling to ensure file is always complete
    try:
        with open(filename, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2)
        
        # Validate the written JSON
        try:
            with open(filename, "r", encoding="utf-8") as json_file:
                json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"\nWARNING: Generated JSON file may be corrupted!")
            print(f"Validation error: {e.msg} at line {e.lineno}")
            print(f"The file was written but may not be readable.")
            return
        
        print(f"Server information saved to: {filename}")
        
        if collection_errors:
            print(f"\nNote: Collection completed with {len(collection_errors)} error(s):")
            for error in collection_errors:
                print(f"  - {error}")
            print(f"\nPartial data has been saved. Re-run collection to get complete data.")

        # Timeouts always deserve a mention: they usually mean wedged hardware, and the
        # resulting empty section looks identical to "this machine has no such device".
        timed_out = [c for c, f in COMMAND_FAILURES.items() if f["reason"] == "timeout"]
        if timed_out:
            print(f"\nWarning: {len(timed_out)} command(s) timed out and were skipped:")
            for cmd in timed_out:
                print(f"  - {cmd} ({COMMAND_FAILURES[cmd]['detail']})")

        if args.verbose:
            errored = [c for c, f in COMMAND_FAILURES.items() if f["reason"] == "error"]
            if errored:
                print(f"\n{len(errored)} command(s) returned an error:")
                for cmd in errored:
                    print(f"  - {cmd}: {COMMAND_FAILURES[cmd]['detail']}")
    
    except Exception as e:
        print(f"\nERROR: Failed to write JSON file: {filename}")
        print(f"Reason: {str(e)}")
        print(f"Data collection completed but could not be saved.")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

