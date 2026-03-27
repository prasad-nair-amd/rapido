#!/usr/bin/env python3
"""
Storage I/O Profiling Benchmark
Measures disk read/write speeds, NVMe performance, RAID detection, and GDS capability
"""

import os
import sys
import json
import time
import subprocess
import tempfile
from pathlib import Path
from collections import OrderedDict


def run_command(cmd):
    """Execute command and return output."""
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=30
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def detect_storage_devices():
    """Detect available storage devices and their types."""
    devices = []
    
    # Use lsblk to get device information
    lsblk_output = run_command(["lsblk", "-J", "-o", "NAME,TYPE,SIZE,ROTA,MODEL,TRAN"])
    if lsblk_output:
        try:
            data = json.loads(lsblk_output)
            for device in data.get("blockdevices", []):
                if device.get("type") == "disk":
                    dev_info = {
                        "name": device.get("name", ""),
                        "size": device.get("size", ""),
                        "model": device.get("model", "Unknown"),
                        "transport": device.get("tran", "Unknown"),
                        "rotational": device.get("rota", False),
                        "type": "HDD" if device.get("rota") else "SSD/NVMe"
                    }
                    devices.append(dev_info)
        except json.JSONDecodeError:
            pass
    
    return devices


def detect_nvme_devices():
    """Detect NVMe-specific devices and get detailed info."""
    nvme_devices = []
    
    # Check for nvme command
    nvme_list = run_command(["nvme", "list", "-o", "json"])
    if nvme_list:
        try:
            data = json.loads(nvme_list)
            for device in data.get("Devices", []):
                nvme_info = OrderedDict()
                nvme_info["Device"] = device.get("DevicePath", "")
                nvme_info["Model"] = device.get("ModelNumber", "").strip()
                nvme_info["Serial"] = device.get("SerialNumber", "").strip()
                nvme_info["Firmware"] = device.get("Firmware", "").strip()
                
                # Get size in human-readable format
                size_bytes = device.get("PhysicalSize", 0)
                size_gb = size_bytes / (1024**3) if size_bytes else 0
                nvme_info["Size"] = f"{size_gb:.2f} GB"
                
                # Get namespace info
                nvme_info["Namespace"] = str(device.get("NameSpace", ""))
                
                nvme_devices.append(nvme_info)
        except json.JSONDecodeError:
            pass
    
    return nvme_devices


def detect_raid_config():
    """Detect RAID configuration using mdadm."""
    raid_configs = []
    
    # Check for mdadm
    mdadm_output = run_command(["mdadm", "--detail", "--scan"])
    if mdadm_output:
        for line in mdadm_output.splitlines():
            if line.startswith("ARRAY"):
                parts = line.split()
                if len(parts) >= 2:
                    array_device = parts[1]
                    
                    # Get detailed info for this array
                    detail_output = run_command(["mdadm", "--detail", array_device])
                    if detail_output:
                        raid_info = OrderedDict()
                        raid_info["Array Device"] = array_device
                        
                        for detail_line in detail_output.splitlines():
                            detail_line = detail_line.strip()
                            if ":" in detail_line:
                                key, value = detail_line.split(":", 1)
                                key = key.strip()
                                value = value.strip()
                                
                                if key == "Raid Level":
                                    raid_info["RAID Level"] = value
                                elif key == "Array Size":
                                    raid_info["Array Size"] = value
                                elif key == "Raid Devices":
                                    raid_info["RAID Devices"] = value
                                elif key == "Total Devices":
                                    raid_info["Total Devices"] = value
                                elif key == "State":
                                    raid_info["State"] = value
                        
                        raid_configs.append(raid_info)
    
    # Also check for LVM RAID
    lvs_output = run_command(["lvs", "--reportformat", "json"])
    if lvs_output:
        try:
            data = json.loads(lvs_output)
            for report in data.get("report", []):
                for lv in report.get("lv", []):
                    if "raid" in lv.get("segtype", "").lower():
                        raid_info = OrderedDict()
                        raid_info["LVM Volume"] = f"{lv.get('vg_name', '')}/{lv.get('lv_name', '')}"
                        raid_info["Type"] = lv.get("segtype", "")
                        raid_info["Size"] = lv.get("lv_size", "")
                        raid_configs.append(raid_info)
        except json.JSONDecodeError:
            pass
    
    return raid_configs


def check_gds_capability():
    """Check for GPU Direct Storage (GDS) capability."""
    gds_info = OrderedDict()
    
    # Check for GDS kernel module
    lsmod_output = run_command(["lsmod"])
    gds_module_loaded = False
    if lsmod_output:
        gds_module_loaded = "nvidia_fs" in lsmod_output or "gdrdrv" in lsmod_output
    
    gds_info["GDS Kernel Module"] = "Loaded" if gds_module_loaded else "Not Loaded"
    
    # Check for cuFile library (NVIDIA GDS)
    cufile_path = "/usr/local/cuda/lib64/libcufile.so"
    gds_info["cuFile Library"] = "Found" if os.path.exists(cufile_path) else "Not Found"
    
    # Check for AMD equivalent (if exists)
    # Note: AMD's solution may use different paths/names
    
    # Check /etc/cufile.json configuration
    cufile_config = "/etc/cufile.json"
    if os.path.exists(cufile_config):
        gds_info["GDS Config"] = "Found at /etc/cufile.json"
    else:
        gds_info["GDS Config"] = "Not Found"
    
    # Overall capability
    gds_capable = gds_module_loaded or os.path.exists(cufile_path)
    gds_info["GDS Capable"] = "Yes" if gds_capable else "No"
    
    return gds_info


def benchmark_disk_speed(device_path, test_size_mb=1024):
    """Benchmark disk read/write speeds using dd."""
    results = OrderedDict()
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory(dir=f"/tmp") as temp_dir:
        test_file = os.path.join(temp_dir, "test_file")
        
        # Write test (sequential write)
        write_cmd = [
            "dd",
            f"if=/dev/zero",
            f"of={test_file}",
            f"bs=1M",
            f"count={test_size_mb}",
            "conv=fdatasync",
            "oflag=direct"
        ]
        
        start_time = time.time()
        result = run_command(write_cmd)
        write_time = time.time() - start_time
        
        if write_time > 0:
            write_speed = test_size_mb / write_time
            results["Sequential Write"] = f"{write_speed:.2f} MB/s"
        
        # Read test (sequential read)
        # Clear page cache first
        run_command(["sync"])
        run_command(["sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        
        read_cmd = [
            "dd",
            f"if={test_file}",
            "of=/dev/null",
            "bs=1M",
            "iflag=direct"
        ]
        
        start_time = time.time()
        result = run_command(read_cmd)
        read_time = time.time() - start_time
        
        if read_time > 0:
            read_speed = test_size_mb / read_time
            results["Sequential Read"] = f"{read_speed:.2f} MB/s"
    
    return results


def main():
    """Main function to run storage benchmarks."""
    output = {
        "storage_devices": [],
        "nvme_devices": [],
        "raid_configs": [],
        "gds_capability": {},
        "benchmark_results": []
    }
    
    # Detect storage devices
    print("Detecting storage devices...", file=sys.stderr)
    output["storage_devices"] = detect_storage_devices()
    
    # Detect NVMe devices
    print("Detecting NVMe devices...", file=sys.stderr)
    output["nvme_devices"] = detect_nvme_devices()
    
    # Detect RAID configuration
    print("Detecting RAID configuration...", file=sys.stderr)
    output["raid_configs"] = detect_raid_config()
    
    # Check GDS capability
    print("Checking GPU Direct Storage capability...", file=sys.stderr)
    output["gds_capability"] = check_gds_capability()
    
    # Run basic disk benchmarks (optional, can be slow)
    # Uncomment to enable disk speed tests
    # print("Running disk speed benchmarks...", file=sys.stderr)
    # for device in output["storage_devices"]:
    #     if not device["rotational"]:  # Only test SSDs
    #         device_path = f"/dev/{device['name']}"
    #         bench_results = benchmark_disk_speed(device_path)
    #         if bench_results:
    #             bench_info = {"device": device["name"]}
    #             bench_info.update(bench_results)
    #             output["benchmark_results"].append(bench_info)
    
    # Output JSON
    print("STORAGE_BENCHMARK_RESULT: " + json.dumps(output, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
