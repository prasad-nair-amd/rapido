#!/usr/bin/env python3
import json
import os
import platform
import socket
import subprocess
from collections import OrderedDict
from typing import Dict, List, Optional

def run_command(cmd: list[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

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
    nvidia = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,uuid",
        "--format=csv,noheader",
    ])
    if nvidia:
        for line in nvidia.splitlines():
            parts = [segment.strip() for segment in line.split(",")]
            if len(parts) >= 3:
                record = OrderedDict()
                record["Name"] = parts[0]
                record["Total Memory"] = parts[1]
                record["Driver Version"] = parts[2]
                if len(parts) > 3:
                    record["UUID"] = parts[3]
                entries.append(record)

    # AMD GPU information - try amd-smi first (newer tool), then rocm-smi
    amd_smi_static = run_command(["amd-smi", "static", "--json"])
    if amd_smi_static:
        try:
            data = json.loads(amd_smi_static)
            if isinstance(data, dict):
                for _, gpu_data in data.items():
                    if isinstance(gpu_data, dict):
                        record = OrderedDict()
                        record["Vendor"] = "AMD"

                        # Basic identification
                        if "asic" in gpu_data and isinstance(gpu_data["asic"], dict):
                            asic = gpu_data["asic"]
                            if "market_name" in asic:
                                record["Name"] = asic["market_name"]
                            if "vendor_name" in asic:
                                record["Vendor Name"] = asic["vendor_name"]
                            if "asic_serial" in asic:
                                record["Serial Number"] = asic["asic_serial"]

                        # VBIOS information
                        if "vbios" in gpu_data and isinstance(gpu_data["vbios"], dict):
                            vbios = gpu_data["vbios"]
                            if "part_number" in vbios:
                                record["VBIOS Part Number"] = vbios["part_number"]
                            if "version" in vbios:
                                record["VBIOS Version"] = vbios["version"]
                            if "build_date" in vbios:
                                record["VBIOS Build Date"] = vbios["build_date"]

                        # Bus information
                        if "bus" in gpu_data and isinstance(gpu_data["bus"], dict):
                            bus = gpu_data["bus"]
                            if "bdf" in bus:
                                record["BDF"] = bus["bdf"]
                            if "max_pcie_width" in bus:
                                record["Max PCIe Width"] = f"{bus['max_pcie_width']} lanes"
                            if "max_pcie_speed" in bus:
                                record["Max PCIe Speed"] = bus["max_pcie_speed"]
                            if "pcie_interface_version" in bus:
                                record["PCIe Version"] = bus["pcie_interface_version"]

                        # VRAM information
                        if "vram" in gpu_data and isinstance(gpu_data["vram"], dict):
                            vram = gpu_data["vram"]
                            if "vram_size" in vram:
                                try:
                                    vram_mb = int(vram["vram_size"])
                                    record["VRAM Size"] = f"{vram_mb / 1024:.2f} GB"
                                except (ValueError, TypeError):
                                    record["VRAM Size"] = str(vram["vram_size"])
                            if "vram_type" in vram:
                                record["VRAM Type"] = vram["vram_type"]
                            if "vram_bit_width" in vram:
                                record["VRAM Bit Width"] = f"{vram['vram_bit_width']}-bit"

                        # Compute capabilities
                        if "asic" in gpu_data and isinstance(gpu_data["asic"], dict):
                            asic = gpu_data["asic"]
                            if "num_compute_units" in asic:
                                record["Compute Units"] = str(asic["num_compute_units"])
                            if "target_graphics_version" in asic:
                                record["GFX Version"] = asic["target_graphics_version"]

                        # Cache information
                        if "cache_info" in gpu_data and isinstance(gpu_data["cache_info"], dict):
                            for cache_level, cache_data in gpu_data["cache_info"].items():
                                if isinstance(cache_data, dict) and "cache_size" in cache_data:
                                    record[f"{cache_level.upper()} Cache"] = cache_data["cache_size"]

                        if record.get("Name"):
                            entries.append(record)
        except json.JSONDecodeError:
            pass

    # Fallback to rocm-smi if amd-smi is not available
    if not amd_smi_static:
        rocm_smi_all = run_command(["rocm-smi", "--showid", "--showproductname", "--showvbios", "--showbus",
                                     "--showmeminfo", "vram", "--showmemvendor", "--showdriverversion", "--json"])
        if rocm_smi_all:
            try:
                data = json.loads(rocm_smi_all)
                for _, gpu_info in data.items():
                    if isinstance(gpu_info, dict):
                        record = OrderedDict()
                        record["Vendor"] = "AMD"

                        # Product name
                        if "Card series" in gpu_info:
                            record["Name"] = gpu_info["Card series"]
                        elif "Card model" in gpu_info:
                            record["Name"] = gpu_info["Card model"]

                        # VBIOS
                        if "VBIOS version" in gpu_info:
                            record["VBIOS Version"] = gpu_info["VBIOS version"]

                        # GPU ID and device information
                        if "GPU ID" in gpu_info:
                            record["GPU ID"] = gpu_info["GPU ID"]
                        if "Device ID" in gpu_info:
                            record["Device ID"] = gpu_info["Device ID"]
                        if "PCI Bus" in gpu_info:
                            record["PCI Bus"] = gpu_info["PCI Bus"]

                        # VRAM information
                        if "VRAM Total Memory (B)" in gpu_info:
                            vram_bytes = gpu_info["VRAM Total Memory (B)"]
                            try:
                                vram_gb = int(vram_bytes) / (1024**3)
                                record["VRAM Size"] = f"{vram_gb:.2f} GB"
                            except (ValueError, TypeError):
                                record["VRAM Size"] = str(vram_bytes)
                        if "VRAM Total Used Memory (B)" in gpu_info:
                            used_bytes = gpu_info["VRAM Total Used Memory (B)"]
                            try:
                                used_gb = int(used_bytes) / (1024**3)
                                record["VRAM Used"] = f"{used_gb:.2f} GB"
                            except (ValueError, TypeError):
                                record["VRAM Used"] = str(used_bytes)
                        if "Memory vendor" in gpu_info:
                            record["Memory Vendor"] = gpu_info["Memory vendor"]

                        # Driver version
                        if "Driver version" in gpu_info:
                            record["ROCm Driver Version"] = gpu_info["Driver version"]

                        if record.get("Name"):
                            entries.append(record)
            except json.JSONDecodeError:
                pass

    # Additional static information from rocminfo
    rocminfo = run_command(["rocminfo"])
    if rocminfo:
        current_agent = OrderedDict()
        in_gpu_agent = False

        for line in rocminfo.splitlines():
            line = line.strip()

            # Detect agent boundaries
            if line.startswith("*******"):
                if current_agent and current_agent.get("Name") and in_gpu_agent:
                    # Check if we already have this GPU from amd-smi/rocm-smi
                    gpu_exists = False
                    for existing_gpu in entries:
                        if existing_gpu.get("Name") == current_agent.get("Name"):
                            # Merge additional info
                            for key, value in current_agent.items():
                                if key not in existing_gpu:
                                    existing_gpu[key] = value
                            gpu_exists = True
                            break
                    if not gpu_exists:
                        current_agent["Vendor"] = "AMD"
                        entries.append(current_agent)
                current_agent = OrderedDict()
                in_gpu_agent = False
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                # Identify GPU agents
                if key == "Marketing Name" and value:
                    current_agent["Name"] = value
                    in_gpu_agent = True
                elif key == "Vendor Name" and ("AMD" in value or "Advanced Micro Devices" in value):
                    in_gpu_agent = True

                # Only collect info for GPU agents
                if in_gpu_agent:
                    if key == "Chip ID" and value:
                        current_agent["Chip ID"] = value
                    elif key == "SIMD count" and value:
                        current_agent["SIMD Count"] = value
                    elif key == "Shader Engines" and value:
                        current_agent["Shader Engines"] = value
                    elif key == "Shader Arrs per Eng" and value:
                        current_agent["Shader Arrays per Engine"] = value
                    elif key == "Compute Unit" and value:
                        if "Compute Units" not in current_agent:
                            current_agent["Compute Units"] = value
                    elif key == "SIMDs per CU" and value:
                        current_agent["SIMDs per CU"] = value
                    elif key == "Wavefront Size" and value:
                        current_agent["Wavefront Size"] = value
                    elif key == "Max Memory (MB)" and value:
                        try:
                            mem_mb = float(value)
                            if "VRAM Size" not in current_agent:
                                current_agent["VRAM Size"] = f"{mem_mb / 1024:.2f} GB"
                        except ValueError:
                            pass
                    elif key == "Max Clock Freq. (MHz)" and value:
                        current_agent["Max Clock Frequency"] = f"{value} MHz"
                    elif key == "Device ID" and value:
                        if "Device ID" not in current_agent:
                            current_agent["Device ID"] = value

        # Don't forget the last agent
        if current_agent and current_agent.get("Name") and in_gpu_agent:
            gpu_exists = False
            for existing_gpu in entries:
                if existing_gpu.get("Name") == current_agent.get("Name"):
                    for key, value in current_agent.items():
                        if key not in existing_gpu:
                            existing_gpu[key] = value
                    gpu_exists = True
                    break
            if not gpu_exists:
                current_agent["Vendor"] = "AMD"
                entries.append(current_agent)

    lspci = run_command(["lspci", "-nn"])
    if lspci:
        for line in lspci.splitlines():
            line = line.strip()
            lowered = line.lower()
            if "vga compatible controller" in lowered or "3d controller" in lowered or "display controller" in lowered:
                record = OrderedDict()
                parts = line.split(" ", 1)
                record["Slot"] = parts[0]
                if len(parts) > 1:
                    record["Description"] = parts[1].strip()
                entries.append(record)
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
    entries: List[Dict[str, str]] = []
    ipconfig = run_command(["ipconfig", "/all"])
    if ipconfig:
        current = OrderedDict()
        for line in ipconfig.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                if current and current.get("Adapter"):
                    entries.append(current)
                    current = OrderedDict()
                continue

            # Check for adapter header (starts at beginning of line, ends with colon)
            if line and not line[0].isspace() and line.rstrip().endswith(":"):
                if current and current.get("Adapter"):
                    entries.append(current)
                current = OrderedDict()
                current["Adapter"] = line.rstrip(":").strip()
            elif ":" in line_stripped:
                key, value = line_stripped.split(":", 1)
                key = key.strip().rstrip(".")
                value = value.strip()
                if value:
                    current[key] = value

        if current and current.get("Adapter"):
            entries.append(current)

    # Try PowerShell as alternative
    if not entries:
        ps = run_command([
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-NetAdapter | Select-Object Name,Status,MacAddress,LinkSpeed,InterfaceDescription) | ConvertTo-Json -Compress"
        ])
        if ps:
            try:
                data = json.loads(ps)
                if isinstance(data, dict):
                    data = [data]
                for adapter in data:
                    record = OrderedDict()
                    for key, value in adapter.items():
                        if value not in (None, ""):
                            record[key] = str(value)
                    if record:
                        entries.append(record)
            except json.JSONDecodeError:
                pass

    return entries

def linux_network_info() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []

    # Try 'ip addr' first (modern Linux)
    ip_addr = run_command(["ip", "-json", "addr"])
    if ip_addr:
        try:
            data = json.loads(ip_addr)
            for interface in data:
                record = OrderedDict()
                if "ifname" in interface:
                    record["Interface"] = interface["ifname"]
                if "address" in interface:
                    record["MAC Address"] = interface["address"]
                if "operstate" in interface:
                    record["State"] = interface["operstate"]
                if "mtu" in interface:
                    record["MTU"] = str(interface["mtu"])

                # Extract IP addresses
                ipv4_addrs = []
                ipv6_addrs = []
                if "addr_info" in interface:
                    for addr in interface["addr_info"]:
                        if addr.get("family") == "inet":
                            ipv4_addrs.append(f"{addr.get('local')}/{addr.get('prefixlen', '')}")
                        elif addr.get("family") == "inet6":
                            ipv6_addrs.append(f"{addr.get('local')}/{addr.get('prefixlen', '')}")

                if ipv4_addrs:
                    record["IPv4 Addresses"] = ", ".join(ipv4_addrs)
                if ipv6_addrs:
                    record["IPv6 Addresses"] = ", ".join(ipv6_addrs)

                if record:
                    entries.append(record)
        except json.JSONDecodeError:
            pass

    # Fallback to 'ip addr' without JSON
    if not entries:
        ip_addr_text = run_command(["ip", "addr"])
        if ip_addr_text:
            current = OrderedDict()
            for line in ip_addr_text.splitlines():
                line = line.strip()

                # Interface line starts with number
                if line and line[0].isdigit() and ":" in line:
                    if current and current.get("Interface"):
                        entries.append(current)
                    parts = line.split(":", 2)
                    if len(parts) >= 2:
                        current = OrderedDict()
                        current["Interface"] = parts[1].strip().split("@")[0]
                        if "state" in line.upper():
                            state_part = line.split("state")
                            if len(state_part) > 1:
                                current["State"] = state_part[1].strip().split()[0]
                elif line.startswith("link/ether"):
                    parts = line.split()
                    if len(parts) >= 2:
                        current["MAC Address"] = parts[1]
                elif line.startswith("inet "):
                    parts = line.split()
                    if len(parts) >= 2:
                        if "IPv4 Addresses" in current:
                            current["IPv4 Addresses"] += f", {parts[1]}"
                        else:
                            current["IPv4 Addresses"] = parts[1]
                elif line.startswith("inet6"):
                    parts = line.split()
                    if len(parts) >= 2:
                        if "IPv6 Addresses" in current:
                            current["IPv6 Addresses"] += f", {parts[1]}"
                        else:
                            current["IPv6 Addresses"] = parts[1]

            if current and current.get("Interface"):
                entries.append(current)

    # Additional info from ethtool (if available)
    for entry in entries:
        interface_name = entry.get("Interface")
        if interface_name:
            ethtool = run_command(["ethtool", interface_name])
            if ethtool:
                for line in ethtool.splitlines():
                    if "Speed:" in line:
                        entry["Link Speed"] = line.split("Speed:", 1)[1].strip()
                    elif "Link detected:" in line:
                        entry["Link Detected"] = line.split("Link detected:", 1)[1].strip()

    return entries

def mac_network_info() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []

    # Use networksetup to list interfaces
    networksetup = run_command(["networksetup", "-listallhardwareports"])
    ifconfig_output = run_command(["ifconfig"])

    interfaces = {}
    if networksetup:
        current_port = None
        for line in networksetup.splitlines():
            line = line.strip()
            if line.startswith("Hardware Port:"):
                current_port = line.split(":", 1)[1].strip()
                interfaces[current_port] = OrderedDict()
                interfaces[current_port]["Hardware Port"] = current_port
            elif line.startswith("Device:") and current_port:
                device = line.split(":", 1)[1].strip()
                interfaces[current_port]["Device"] = device
            elif line.startswith("Ethernet Address:") and current_port:
                mac = line.split(":", 1)[1].strip()
                interfaces[current_port]["MAC Address"] = mac

    # Parse ifconfig for additional details
    if ifconfig_output:
        current_interface = None
        for line in ifconfig_output.splitlines():
            if line and not line[0].isspace():
                parts = line.split(":")
                if parts:
                    current_interface = parts[0].strip()
                    # Find matching entry or create new one
                    found = False
                    for port_data in interfaces.values():
                        if port_data.get("Device") == current_interface:
                            found = True
                            break
                    if not found:
                        interfaces[current_interface] = OrderedDict()
                        interfaces[current_interface]["Interface"] = current_interface
            elif current_interface and line.strip():
                line = line.strip()
                if line.startswith("inet "):
                    parts = line.split()
                    if len(parts) >= 2:
                        for port_data in interfaces.values():
                            if port_data.get("Device") == current_interface or port_data.get("Interface") == current_interface:
                                port_data["IPv4 Address"] = parts[1]
                elif line.startswith("inet6"):
                    parts = line.split()
                    if len(parts) >= 2:
                        for port_data in interfaces.values():
                            if port_data.get("Device") == current_interface or port_data.get("Interface") == current_interface:
                                if "IPv6 Address" in port_data:
                                    port_data["IPv6 Address"] += f", {parts[1]}"
                                else:
                                    port_data["IPv6 Address"] = parts[1]
                elif line.startswith("status:"):
                    status = line.split(":", 1)[1].strip()
                    for port_data in interfaces.values():
                        if port_data.get("Device") == current_interface or port_data.get("Interface") == current_interface:
                            port_data["Status"] = status

    entries = list(interfaces.values())
    return entries

def gather_network_details() -> Dict[str, List[Dict[str, str]]]:
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

    # Add hostname
    try:
        hostname = socket.gethostname()
        fqdn = socket.getfqdn()
        system_info = OrderedDict()
        system_info["hostname"] = hostname
        if fqdn != hostname:
            system_info["fqdn"] = fqdn
        details["system"] = [system_info]
    except Exception:
        pass

    return details

def gather_gpu_microbenchmarks() -> Dict[str, List[Dict[str, str]]]:
    """Gather GPU microbenchmark information including supported data formats."""
    details: Dict[str, List[Dict[str, str]]] = OrderedDict()
    system = platform.system().lower()

    # Only gather microbenchmark info on Linux
    if system != "linux":
        return details

    microbenchmark_list = []

    # Supported Data Formats
    formats_info = OrderedDict()
    formats_info["Section"] = "Supported Data Formats"

    # Check for AMD GPU architecture support from rocminfo
    rocminfo_output = run_command(["rocminfo"])
    if rocminfo_output:
        current_gpu = None
        gpu_formats = OrderedDict()

        for line in rocminfo_output.splitlines():
            line = line.strip()

            if line.startswith("*******"):
                # Save previous GPU if exists
                if current_gpu and gpu_formats:
                    formats_info[current_gpu] = ", ".join(gpu_formats.values())
                current_gpu = None
                gpu_formats = OrderedDict()

            elif "Marketing Name" in line and ":" in line:
                gpu_name = line.split(":", 1)[1].strip()
                if gpu_name and ("AMD" in gpu_name or "Radeon" in gpu_name or "Instinct" in gpu_name):
                    current_gpu = gpu_name

            elif current_gpu and "target_graphics_version" in line.lower():
                # Extract GFX version to determine format support
                if ":" in line:
                    gfx_version = line.split(":", 1)[1].strip()
                    gpu_formats["GFX Version"] = gfx_version

        # Save last GPU
        if current_gpu and gpu_formats:
            formats_info[current_gpu] = f"GFX {gpu_formats.get('GFX Version', 'Unknown')}"

    # Get GPU-specific format support from amd-smi or rocm-smi
    amd_smi_static = run_command(["amd-smi", "static", "--json"])
    if amd_smi_static:
        try:
            data = json.loads(amd_smi_static)
            if isinstance(data, dict):
                for _, gpu_data in data.items():
                    if isinstance(gpu_data, dict):
                        gpu_name = None
                        gfx_version = None

                        if "asic" in gpu_data and isinstance(gpu_data["asic"], dict):
                            asic = gpu_data["asic"]
                            if "market_name" in asic:
                                gpu_name = asic["market_name"]
                            if "target_graphics_version" in asic:
                                gfx_version = asic["target_graphics_version"]

                        if gpu_name and gfx_version:
                            # Determine format support based on GFX version
                            supported_formats = []

                            # All modern AMD GPUs support FP32
                            supported_formats.append("FP32")

                            # FP64 support (CDNA, some RDNA)
                            if any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942"]):
                                supported_formats.append("FP64 (Full Rate)")
                            elif any(arch in gfx_version.lower() for arch in ["gfx900", "gfx906", "gfx908"]):
                                supported_formats.append("FP64 (1/2 Rate)")
                            else:
                                supported_formats.append("FP64 (1/16 Rate)")

                            # INT8 support (most modern GPUs)
                            supported_formats.append("INT8")

                            # FP16 support (most modern GPUs)
                            supported_formats.append("FP16")

                            # BF16 support (newer architectures)
                            if any(arch in gfx_version.lower() for arch in ["gfx90a", "gfx940", "gfx941", "gfx942", "gfx1100", "gfx1101"]):
                                supported_formats.append("BF16")

                            # FP8 support (CDNA3 and newer)
                            if any(arch in gfx_version.lower() for arch in ["gfx940", "gfx941", "gfx942"]):
                                supported_formats.append("FP8")

                            # FP4 support (limited to specific architectures)
                            if any(arch in gfx_version.lower() for arch in ["gfx942"]):
                                supported_formats.append("FP4")

                            formats_info[f"{gpu_name} ({gfx_version})"] = ", ".join(supported_formats)
        except json.JSONDecodeError:
            pass

    # If no specific GPU data found, provide general format information
    if len(formats_info) == 1:
        formats_info["FP32"] = "32-bit floating-point (standard precision)"
        formats_info["FP64"] = "64-bit floating-point (double precision)"
        formats_info["FP16"] = "16-bit floating-point (half precision)"
        formats_info["BF16"] = "16-bit brain floating-point"
        formats_info["INT8"] = "8-bit integer (quantized operations)"
        formats_info["FP8"] = "8-bit floating-point (AI/ML workloads)"
        formats_info["FP4"] = "4-bit floating-point (extreme quantization)"

    if len(formats_info) > 1:
        microbenchmark_list.append(formats_info)

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

    if rocm_info_list:
        details["linux"] = rocm_info_list

    return details

def main() -> None:
    cpu_details = gather_cpu_details()
    gpu_details = gather_gpu_details()
    network_details = gather_network_details()
    rocm_details = gather_rocm_details()
    microbenchmark_details = gather_gpu_microbenchmarks()

    payload = OrderedDict()
    payload["cpu"] = cpu_details
    if gpu_details:
        payload["gpu"] = gpu_details
    else:
        payload["gpu"] = []
    if network_details:
        payload["network"] = network_details
    else:
        payload["network"] = []
    if rocm_details:
        payload["rocm"] = rocm_details
    else:
        payload["rocm"] = []
    if microbenchmark_details:
        payload["microbenchmarks"] = microbenchmark_details
    else:
        payload["microbenchmarks"] = []

    # Generate filename with system name
    try:
        hostname = socket.gethostname()
        # Sanitize hostname for filename (replace invalid characters)
        safe_hostname = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in hostname)
        filename = f"serverinfo_{safe_hostname}.json"
    except Exception:
        filename = "serverinfo.json"

    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

    print(f"Server information saved to: {filename}")

if __name__ == "__main__":
    main()

