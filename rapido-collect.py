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

def run_command(cmd: List[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # Python 3.6 compatible; use text=True on 3.7+
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
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
            # Check if tool exists
            result = run_command([tool_cmd, "--version"]) if system != "windows" else run_command([tool_cmd, "/?"])
            
            # Some tools don't support --version, try different approaches
            if result is None:
                if tool_cmd in ["ip", "ethtool", "ipmitool"]:
                    result = run_command([tool_cmd])
                elif tool_cmd == "system_profiler":
                    result = run_command(["which", tool_cmd])
                elif tool_cmd in ["dpkg", "rpm"]:
                    result = run_command(["which", tool_cmd])
            
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

def gather_gpu_microbenchmarks(include_p2p: bool = False, verbose: bool = True) -> Dict[str, List[Dict[str, str]]]:
    """Gather GPU microbenchmark information including peak performance and optionally GPU-to-GPU communication."""
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

        for line in rocminfo_output.splitlines():
            line = line.strip()

            if line.startswith("*******"):
                # Process previous GPU
                if current_gpu and compute_units:
                    gpu_key = f"{current_gpu}"
                    if gfx_version:
                        gpu_key += f" ({gfx_version})"

                    # Calculate maximum theoretical performance
                    # Standard assumption: 128 FP32 ops per CU per clock (64 SPs * 2 ops/clock FMA)
                    ops_per_cu_per_clock = 128

                    # Use detected or estimated clock frequency
                    base_clock = max_clock_freq if max_clock_freq else 1500  # Default 1.5 GHz if not detected

                    # Calculate FP32 peak performance (dense)
                    fp32_dense_tflops = (compute_units * ops_per_cu_per_clock * base_clock) / 1e6
                    
                    # Sparse performance is typically 2x dense for certain architectures
                    fp32_sparse_tflops = fp32_dense_tflops * 2 if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]) else fp32_dense_tflops

                    # Create separate cards for Dense and Sparse calculations
                    
                    # DENSE CALCULATIONS
                    dense_info = OrderedDict()
                    dense_info["Section"] = f"{gpu_key} - Dense Peak Performance"
                    dense_info["Compute Units"] = str(compute_units)
                    dense_info["Max Clock Frequency"] = f"{base_clock:.0f} MHz"
                    
                    # FP64 (double precision)
                    if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                        fp64_dense = fp32_dense_tflops
                        dense_info["FP64 (double precision)"] = f"{fp64_dense:.1f} TFLOPS"
                    elif gfx_version and any(arch in gfx_version.lower() for arch in ["gfx900", "gfx906", "gfx908"]):
                        fp64_dense = fp32_dense_tflops / 2
                        dense_info["FP64 (double precision)"] = f"{fp64_dense:.1f} TFLOPS"
                    else:
                        fp64_dense = fp32_dense_tflops / 16
                        dense_info["FP64 (double precision)"] = f"{fp64_dense:.1f} TFLOPS"
                    
                    # FP32 (single precision)
                    dense_info["FP32 (single precision)"] = f"{fp32_dense_tflops:.1f} TFLOPS (matrix and vector)"
                    
                    # TF32 (TensorFloat-32) - 4x FP32 for CDNA2+
                    tf32_dense = fp32_dense_tflops * 4
                    if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                        dense_info["TF32 (TensorFloat-32)"] = f"{tf32_dense:.1f} TFLOPS"
                    
                    # FP16 / BF16
                    fp16_dense = fp32_dense_tflops * 2
                    if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101"]):
                        dense_info["FP16 / BF16"] = f"{fp16_dense:.1f} TFLOPS"
                    else:
                        dense_info["FP16"] = f"{fp16_dense:.1f} TFLOPS"
                    
                    # FP8
                    fp8_dense = fp32_dense_tflops * 8
                    if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx940", "gfx941", "gfx942"]):
                        dense_info["FP8"] = f"{fp8_dense:.1f} TFLOPS"
                    
                    # INT8
                    int8_dense = fp32_dense_tflops * 4
                    if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                        dense_info["INT8"] = f"{int8_dense:.1f} TOPS"
                    
                    if max_memory:
                        dense_info["Max Memory"] = f"{max_memory:.2f} GB"
                    
                    microbenchmark_list.append(dense_info)
                    
                    # SPARSE CALCULATIONS (only for supported architectures)
                    if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                        sparse_info = OrderedDict()
                        sparse_info["Section"] = f"{gpu_key} - Sparse Peak Performance"
                        sparse_info["Compute Units"] = str(compute_units)
                        sparse_info["Max Clock Frequency"] = f"{base_clock:.0f} MHz"
                        sparse_info["Note"] = "2:1 sparse matrix operations (50% sparsity)"
                        
                        # FP64 sparse
                        fp64_sparse = fp64_dense * 2
                        sparse_info["FP64 (double precision)"] = f"{fp64_sparse:.1f} TFLOPS"
                        
                        # FP32 sparse
                        sparse_info["FP32 (single precision)"] = f"{fp32_sparse_tflops:.1f} TFLOPS (matrix and vector)"
                        
                        # TF32 sparse
                        tf32_sparse = tf32_dense * 2
                        sparse_info["TF32 (TensorFloat-32)"] = f"{tf32_sparse:.1f} TFLOPS"
                        
                        # FP16 / BF16 sparse
                        fp16_sparse = fp16_dense * 2
                        sparse_info["FP16 / BF16"] = f"{fp16_sparse:.1f} TFLOPS"
                        
                        # FP8 sparse
                        if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx940", "gfx941", "gfx942"]):
                            fp8_sparse = fp8_dense * 2
                            # FP8 range for sparse
                            fp8_min = fp8_dense
                            fp8_max = fp8_sparse
                            sparse_info["FP8"] = f"{fp8_min:.1f}–{fp8_max:.1f} TFLOPS"
                        
                        # INT8 sparse
                        int8_sparse = int8_dense * 2
                        sparse_info["INT8"] = f"{int8_sparse:.1f} TOPS"
                        
                        microbenchmark_list.append(sparse_info)

                # Reset for next GPU
                current_gpu = None
                gfx_version = None
                compute_units = None
                max_clock_freq = None
                max_memory = None

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

            elif current_gpu and "Name:" in line and "gfx" in line.lower():
                # Try to extract GFX version
                parts = line.split()
                for part in parts:
                    if part.lower().startswith("gfx"):
                        gfx_version = part
                        break

        # Don't forget the last GPU
        if current_gpu and compute_units:
            gpu_key = f"{current_gpu}"
            if gfx_version:
                gpu_key += f" ({gfx_version})"

            ops_per_cu_per_clock = 128
            base_clock = max_clock_freq if max_clock_freq else 1500

            # Calculate FP32 peak performance (dense)
            fp32_dense_tflops = (compute_units * ops_per_cu_per_clock * base_clock) / 1e6
            
            # Sparse performance is typically 2x dense for certain architectures
            fp32_sparse_tflops = fp32_dense_tflops * 2 if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]) else fp32_dense_tflops

            # DENSE CALCULATIONS
            dense_info = OrderedDict()
            dense_info["Section"] = f"{gpu_key} - Dense Peak Performance"
            dense_info["Compute Units"] = str(compute_units)
            dense_info["Max Clock Frequency"] = f"{base_clock:.0f} MHz"
            
            # FP64 (double precision)
            if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                fp64_dense = fp32_dense_tflops
                dense_info["FP64 (double precision)"] = f"{fp64_dense:.1f} TFLOPS"
            elif gfx_version and any(arch in gfx_version.lower() for arch in ["gfx900", "gfx906", "gfx908"]):
                fp64_dense = fp32_dense_tflops / 2
                dense_info["FP64 (double precision)"] = f"{fp64_dense:.1f} TFLOPS"
            else:
                fp64_dense = fp32_dense_tflops / 16
                dense_info["FP64 (double precision)"] = f"{fp64_dense:.1f} TFLOPS"
            
            # FP32 (single precision)
            dense_info["FP32 (single precision)"] = f"{fp32_dense_tflops:.1f} TFLOPS (matrix and vector)"
            
            # TF32 (TensorFloat-32) - 4x FP32 for CDNA2+
            tf32_dense = fp32_dense_tflops * 4
            if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                dense_info["TF32 (TensorFloat-32)"] = f"{tf32_dense:.1f} TFLOPS"
            
            # FP16 / BF16
            fp16_dense = fp32_dense_tflops * 2
            if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101"]):
                dense_info["FP16 / BF16"] = f"{fp16_dense:.1f} TFLOPS"
            else:
                dense_info["FP16"] = f"{fp16_dense:.1f} TFLOPS"
            
            # FP8
            fp8_dense = fp32_dense_tflops * 8
            if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx940", "gfx941", "gfx942"]):
                dense_info["FP8"] = f"{fp8_dense:.1f} TFLOPS"
            
            # INT8
            int8_dense = fp32_dense_tflops * 4
            if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                dense_info["INT8"] = f"{int8_dense:.1f} TOPS"
            
            if max_memory:
                dense_info["Max Memory"] = f"{max_memory:.2f} GB"
            
            microbenchmark_list.append(dense_info)
            
            # SPARSE CALCULATIONS (only for supported architectures)
            if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                sparse_info = OrderedDict()
                sparse_info["Section"] = f"{gpu_key} - Sparse Peak Performance"
                sparse_info["Compute Units"] = str(compute_units)
                sparse_info["Max Clock Frequency"] = f"{base_clock:.0f} MHz"
                sparse_info["Note"] = "2:1 sparse matrix operations (50% sparsity)"
                
                # FP64 sparse
                fp64_sparse = fp64_dense * 2
                sparse_info["FP64 (double precision)"] = f"{fp64_sparse:.1f} TFLOPS"
                
                # FP32 sparse
                sparse_info["FP32 (single precision)"] = f"{fp32_sparse_tflops:.1f} TFLOPS (matrix and vector)"
                
                # TF32 sparse
                tf32_sparse = tf32_dense * 2
                sparse_info["TF32 (TensorFloat-32)"] = f"{tf32_sparse:.1f} TFLOPS"
                
                # FP16 / BF16 sparse
                fp16_sparse = fp16_dense * 2
                sparse_info["FP16 / BF16"] = f"{fp16_sparse:.1f} TFLOPS"
                
                # FP8 sparse
                if gfx_version and any(arch in gfx_version.lower() for arch in ["gfx940", "gfx941", "gfx942"]):
                    fp8_sparse = fp8_dense * 2
                    # FP8 range for sparse
                    fp8_min = fp8_dense
                    fp8_max = fp8_sparse
                    sparse_info["FP8"] = f"{fp8_min:.1f}–{fp8_max:.1f} TFLOPS"
                
                # INT8 sparse
                int8_sparse = int8_dense * 2
                sparse_info["INT8"] = f"{int8_sparse:.1f} TOPS"
                
                microbenchmark_list.append(sparse_info)

    # Add kernel benchmarks (GEMM, memory bandwidth, vector ops, convolution)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_cpp_file = os.path.join(script_dir, "gpu_kernel_benchmarks.cpp")
    kernel_exe_file = os.path.join(script_dir, "gpu_kernel_benchmarks")
    
    if os.path.exists(kernel_cpp_file):
        if verbose:
            print(f"Compiling GPU kernel benchmarks: {kernel_cpp_file}")
        compile_result = run_command(["hipcc", "-O3", "-o", kernel_exe_file, kernel_cpp_file])
        
        if compile_result is not None or os.path.exists(kernel_exe_file):
            if verbose:
                print(f"Running GPU kernel benchmarks: {kernel_exe_file}")
            kernel_output = run_command([kernel_exe_file])
            
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
            compile_result = run_command(["hipcc", "-o", exe_file, cpp_file])

            if compile_result is None and not os.path.exists(exe_file):
                error_info = OrderedDict()
                error_info["Section"] = "GPU P2P Communication - Error"
                error_info["Message"] = "Failed to compile (hipcc not found or compilation error)"
                microbenchmark_list.append(error_info)
            else:
                # Run the compiled benchmark
                if verbose:
                    print(f"Running GPU P2P benchmark: {exe_file}")
                p2p_output = run_command([exe_file])

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
                                # Add each P2P result as a card
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

                                    microbenchmark_list.append(p2p_info)
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
            compile_result = run_command(["hipcc", "-o", host_exe_file, host_cpp_file])

            if compile_result is not None or os.path.exists(host_exe_file):
                if verbose:
                    print(f"Running GPU-CPU bandwidth benchmark: {host_exe_file}")
                host_output = run_command([host_exe_file])

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
            compile_result = run_command(["hipcc", "-o", topology_exe_file, topology_cpp_file])

            if compile_result is not None or os.path.exists(topology_exe_file):
                if verbose:
                    print(f"Running GPU topology analysis: {topology_exe_file}")
                topology_output = run_command([topology_exe_file])

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
            storage_output = run_command(["python3", storage_py_file])
            
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
            network_output = run_command(["python3", network_py_file])
            
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
        "-a",
        "--all",
        action="store_true",
        help="Collect all basic information: CPU, GPU, Network, BMC, ROCm (does NOT include microbenchmarks - use -m)",
    )
    parser.add_argument(
        "-p",
        "--p2p",
        action="store_true",
        help="Include GPU-to-GPU peer-to-peer communication bandwidth tests (requires -m flag)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output (show tool availability check and progress messages)",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()

    # Determine which sections to collect
    # If no specific flags are provided, or -a is used, collect all basic sections (NOT microbenchmarks)
    collect_all = args.all or not (args.cpu or args.gpu or args.network or args.bmc or args.rocm or args.microbenchmarks)
    
    collect_cpu = collect_all or args.cpu
    collect_gpu = collect_all or args.gpu
    collect_network = collect_all or args.network
    collect_bmc = collect_all or args.bmc
    collect_rocm = collect_all or args.rocm or args.microbenchmarks  # ROCm is collected with microbenchmarks
    # Microbenchmarks ONLY run when explicitly requested with -m flag
    collect_microbenchmarks = args.microbenchmarks

    # Check and report tool availability at the beginning
    check_tool_availability(verbose=args.verbose)

    # Warn if -p is used without -m
    if args.p2p and not collect_microbenchmarks:
        if args.verbose:
            print("Warning: -p/--p2p flag requires -m/--microbenchmarks flag to be effective")
            print("GPU P2P communication tests will be skipped. Use: python rapido-collect.py -m -p")
    
    if args.verbose:
        print("Starting data collection...")
        print()

    # Collect data based on flags with error handling
    cpu_details = {}
    gpu_details = {}
    network_details = {}
    bmc_details = {}
    rocm_details = {}
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
                microbenchmark_details = gather_gpu_microbenchmarks(include_p2p=args.p2p, verbose=args.verbose)
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
        "command_line": ' '.join(sys.argv)
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
    if microbenchmark_details:
        payload["microbenchmarks"] = microbenchmark_details

    # Generate filename with system name
    try:
        hostname = socket.gethostname()
        # Sanitize hostname for filename (replace invalid characters)
        safe_hostname = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in hostname)
        filename = f"serverinfo_{safe_hostname}.json"
    except Exception:
        filename = "serverinfo.json"

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
    
    except Exception as e:
        print(f"\nERROR: Failed to write JSON file: {filename}")
        print(f"Reason: {str(e)}")
        print(f"Data collection completed but could not be saved.")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

