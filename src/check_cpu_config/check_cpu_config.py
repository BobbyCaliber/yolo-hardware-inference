import csv
import os
import re
import subprocess

import cpuinfo
import psutil


def _bytes(value):
    """Return a numeric byte count from cpuinfo, or None if unavailable."""
    return value if isinstance(value, (int, float)) else None


def _detect_dram_bandwidth_gbps() -> float:
    """Best-effort host DRAM bandwidth in GB/s.

    Tries `dmidecode -t memory` (needs root, usually unavailable in a
    rootless container — silently fails). Falls back to 51.2 GB/s, the
    DDR4-3200 dual-channel value that fits most modern desktops/laptops.
    The regressor learns per-platform corrections from the bench data, so a
    slightly off default isn't fatal — we just want a sane order of magnitude.
    """
    try:
        out = subprocess.check_output(
            ["dmidecode", "-t", "memory"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired,
            PermissionError):
        return 51.2
    speeds = [int(m.group(1)) for m in re.finditer(r"Configured Memory Speed:\s+(\d+)\s*MT/s", out)]
    bus_widths = [int(m.group(1)) for m in re.finditer(r"Data Width:\s+(\d+)\s*bits", out)]
    if not speeds or not bus_widths:
        return 51.2
    # bandwidth = effective_MT/s × bus_bits/8 × num_channels (one device per channel approx)
    speed_avg = sum(speeds) / len(speeds)
    width_avg = sum(bus_widths) / len(bus_widths)
    n_channels = len(speeds)
    return speed_avg * width_avg / 8 * n_channels / 1000  # → GB/s


cpu_info = cpuinfo.get_cpu_info()
freq = psutil.cpu_freq()
cores = psutil.cpu_count(logical=False) or 1
threads = psutil.cpu_count(logical=True) or cores

l2_bytes = _bytes(cpu_info.get('l2_cache_size'))
l3_bytes = _bytes(cpu_info.get('l3_cache_size'))

config = {
    'Processor': cpu_info.get('brand_raw', 'Unknown'),
    'Frequency (MHz)': freq.current if freq else 'Unknown',
    'Max Frequency (MHz)': freq.max if freq else 'Unknown',
    'Cores': cores,
    'Threads': threads,
    'l1_cache_size (KB)': cpu_info.get('l1_data_cache_size', 'Unknown'),
    'l1_instruction_cache_size (KB)': cpu_info.get('l1_instruction_cache_size', 'Unknown'),
    'l2_cache_size_per_core (KB)': l2_bytes / cores / 1e3 if l2_bytes else 'Unknown',
    'l2_cache_size (KB)': l2_bytes / 1e3 if l2_bytes else 'Unknown',
    'l3_cache_size (MB)': l3_bytes / 1e6 if l3_bytes else 'Unknown',
    'SMP Supported': threads > cores,
    'cpu_peak_bw_gbps': _detect_dram_bandwidth_gbps(),
    # Coarse but stable proxy for the per-op overhead a CPU thread incurs;
    # learnable correction lives in the regressor.
    'launch_overhead_us_cpu': 2.0,
}

output_dir = os.environ.get('DATA_DIR', '/app/data')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'cpu_config.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(config.keys())
    writer.writerow(config.values())
