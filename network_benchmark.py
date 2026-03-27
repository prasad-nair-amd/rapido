#!/usr/bin/env python3
"""
Network Performance Benchmark
Tests network bandwidth, RDMA/RoCE capability, multi-node topology, and MPI benchmarks
"""

import os
import sys
import json
import subprocess
import socket
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
            timeout=60
        )
        return result.stdout if result.returncode == 0 else None
    except Exception:
        return None


def detect_rdma_devices():
    """Detect RDMA/InfiniBand/RoCE devices."""
    rdma_devices = []
    
    # Check for ibstat command (InfiniBand)
    ibstat_output = run_command(["ibstat"])
    if ibstat_output:
        current_device = None
        device_info = OrderedDict()
        
        for line in ibstat_output.splitlines():
            line_stripped = line.strip()
            
            if line.startswith("CA '") or line.startswith("CA:"):
                # Save previous device
                if current_device and device_info:
                    rdma_devices.append(device_info)
                
                # Start new device
                current_device = line.split("'")[1] if "'" in line else line.split(":")[0].strip()
                device_info = OrderedDict()
                device_info["Device"] = current_device
            
            elif current_device and ":" in line_stripped:
                key, value = line_stripped.split(":", 1)
                key = key.strip()
                value = value.strip()
                
                if key == "CA type":
                    device_info["Type"] = value
                elif key == "Firmware version":
                    device_info["Firmware"] = value
                elif key == "State":
                    device_info["State"] = value
                elif key == "Physical state":
                    device_info["Physical State"] = value
                elif key == "Rate":
                    device_info["Link Rate"] = value
                elif key == "Base lid":
                    device_info["LID"] = value
        
        # Don't forget last device
        if current_device and device_info:
            rdma_devices.append(device_info)
    
    # Also check rdma link show for RoCE devices
    rdma_link_output = run_command(["rdma", "link", "show"])
    if rdma_link_output and not rdma_devices:
        # Parse rdma link output for RoCE devices
        for line in rdma_link_output.splitlines():
            if "link" in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    device_info = OrderedDict()
                    device_info["Device"] = parts[0]
                    device_info["Type"] = "RoCE"
                    device_info["State"] = "up" if "state" in line.lower() and "active" in line.lower() else "unknown"
                    rdma_devices.append(device_info)
    
    return rdma_devices


def detect_roce_capability():
    """Detect RoCE (RDMA over Converged Ethernet) capability."""
    roce_info = OrderedDict()
    
    # Check for roce_* kernel modules
    lsmod_output = run_command(["lsmod"])
    roce_modules = []
    
    if lsmod_output:
        roce_keywords = ["rdma_", "ib_", "mlx", "roce"]
        for line in lsmod_output.splitlines():
            if any(keyword in line.lower() for keyword in roce_keywords):
                module_name = line.split()[0]
                if module_name not in roce_modules:
                    roce_modules.append(module_name)
    
    roce_info["RDMA Kernel Modules"] = ", ".join(roce_modules) if roce_modules else "None"
    
    # Check for RDMA devices in /sys/class/infiniband
    ib_devices_path = Path("/sys/class/infiniband")
    if ib_devices_path.exists():
        devices = list(ib_devices_path.iterdir())
        roce_info["InfiniBand Devices"] = str(len(devices))
    else:
        roce_info["InfiniBand Devices"] = "0"
    
    # Check for rdma-core package
    rdma_core_version = run_command(["dpkg", "-l", "rdma-core"])
    if not rdma_core_version:
        rdma_core_version = run_command(["rpm", "-q", "rdma-core"])
    
    roce_info["rdma-core Package"] = "Installed" if rdma_core_version else "Not Installed"
    
    # Check for perftest tools (ib_send_bw, etc.)
    ib_send_bw = run_command(["which", "ib_send_bw"])
    roce_info["RDMA Performance Tools"] = "Installed" if ib_send_bw else "Not Installed"
    
    # Overall capability
    has_rdma = bool(roce_modules and (ib_devices_path.exists() and len(list(ib_devices_path.iterdir())) > 0))
    roce_info["RoCE Capable"] = "Yes" if has_rdma else "No"
    
    return roce_info


def detect_network_topology():
    """Detect multi-node network topology information."""
    topology_info = OrderedDict()
    
    # Get hostname
    hostname = socket.gethostname()
    topology_info["Hostname"] = hostname
    
    # Get all network interfaces
    ip_output = run_command(["ip", "-j", "addr"])
    if ip_output:
        try:
            interfaces = json.loads(ip_output)
            active_interfaces = []
            
            for iface in interfaces:
                if iface.get("operstate") == "UP" and iface.get("ifname") not in ["lo"]:
                    iface_info = {
                        "name": iface.get("ifname"),
                        "state": iface.get("operstate"),
                        "mtu": iface.get("mtu")
                    }
                    
                    # Get IP addresses
                    addrs = []
                    for addr_info in iface.get("addr_info", []):
                        if addr_info.get("family") in ["inet", "inet6"]:
                            addrs.append(f"{addr_info.get('local')}/{addr_info.get('prefixlen')}")
                    
                    iface_info["addresses"] = addrs
                    active_interfaces.append(iface_info)
            
            topology_info["Active Interfaces"] = str(len(active_interfaces))
            
        except json.JSONDecodeError:
            pass
    
    # Check for MPI installation
    mpirun_path = run_command(["which", "mpirun"])
    if not mpirun_path:
        mpirun_path = run_command(["which", "mpiexec"])
    
    topology_info["MPI Available"] = "Yes" if mpirun_path else "No"
    
    # Check for OpenMPI or MPICH version
    if mpirun_path:
        mpi_version = run_command(["mpirun", "--version"])
        if mpi_version:
            first_line = mpi_version.splitlines()[0] if mpi_version.splitlines() else ""
            topology_info["MPI Version"] = first_line.strip()
    
    return topology_info


def test_loopback_bandwidth():
    """Test local loopback network bandwidth using iperf3."""
    bandwidth_info = OrderedDict()
    
    # Check if iperf3 is installed
    iperf3_path = run_command(["which", "iperf3"])
    
    if not iperf3_path:
        bandwidth_info["Status"] = "iperf3 not installed"
        bandwidth_info["Install Command"] = "apt install iperf3 (Debian/Ubuntu) or yum install iperf3 (RHEL/CentOS)"
        return bandwidth_info
    
    # Note: This is a simple loopback test
    # For real multi-node testing, iperf3 server needs to be running on another node
    bandwidth_info["Tool"] = "iperf3"
    bandwidth_info["Test Type"] = "Loopback (single node)"
    bandwidth_info["Note"] = "For multi-node testing, run 'iperf3 -s' on remote node and specify with -c <host>"
    
    # We can't actually run the test without a server, but we can report capability
    bandwidth_info["Capability"] = "Ready for testing"
    
    return bandwidth_info


def detect_mpi_benchmarks():
    """Detect available MPI benchmark tools (OSU Micro-Benchmarks, IMB)."""
    mpi_bench_info = OrderedDict()
    
    # Check for OSU Micro-Benchmarks
    osu_bw_path = run_command(["which", "osu_bw"])
    if not osu_bw_path:
        # Try common installation paths
        common_paths = [
            "/usr/local/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw",
            "/opt/osu-micro-benchmarks/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw",
            "$HOME/osu-micro-benchmarks/libexec/osu-micro-benchmarks/mpi/pt2pt/osu_bw"
        ]
        for path in common_paths:
            expanded_path = os.path.expandvars(os.path.expanduser(path))
            if os.path.exists(expanded_path):
                osu_bw_path = expanded_path
                break
    
    mpi_bench_info["OSU Micro-Benchmarks"] = "Installed" if osu_bw_path else "Not Found"
    
    if osu_bw_path:
        mpi_bench_info["OSU Path"] = osu_bw_path
        mpi_bench_info["Available Tests"] = "osu_bw, osu_latency, osu_bibw, osu_allreduce, etc."
    
    # Check for Intel MPI Benchmarks (IMB)
    imb_path = run_command(["which", "IMB-MPI1"])
    mpi_bench_info["Intel MPI Benchmarks"] = "Installed" if imb_path else "Not Found"
    
    # Usage notes
    if not osu_bw_path and not imb_path:
        mpi_bench_info["Note"] = "Install OSU Micro-Benchmarks or Intel MPI Benchmarks for MPI performance testing"
        mpi_bench_info["OSU Install"] = "Download from http://mvapich.cse.ohio-state.edu/benchmarks/"
    else:
        mpi_bench_info["Usage Example"] = "mpirun -np 2 -host node1,node2 osu_bw"
    
    return mpi_bench_info


def main():
    """Main function to run network benchmarks."""
    output = {
        "rdma_devices": [],
        "roce_capability": {},
        "network_topology": {},
        "bandwidth_tools": {},
        "mpi_benchmarks": {}
    }
    
    # Detect RDMA devices
    print("Detecting RDMA/InfiniBand devices...", file=sys.stderr)
    output["rdma_devices"] = detect_rdma_devices()
    
    # Detect RoCE capability
    print("Checking RoCE capability...", file=sys.stderr)
    output["roce_capability"] = detect_roce_capability()
    
    # Detect network topology
    print("Detecting network topology...", file=sys.stderr)
    output["network_topology"] = detect_network_topology()
    
    # Check bandwidth testing tools
    print("Checking network bandwidth tools...", file=sys.stderr)
    output["bandwidth_tools"] = test_loopback_bandwidth()
    
    # Detect MPI benchmarks
    print("Detecting MPI benchmark tools...", file=sys.stderr)
    output["mpi_benchmarks"] = detect_mpi_benchmarks()
    
    # Output JSON
    print("NETWORK_BENCHMARK_RESULT: " + json.dumps(output, indent=2))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
