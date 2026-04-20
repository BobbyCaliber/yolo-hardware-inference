import csv
import os

import cpuinfo
import psutil


def _bytes(value):
    """Return a numeric byte count from cpuinfo, or None if unavailable."""
    return value if isinstance(value, (int, float)) else None


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
}

output_dir = os.environ.get('DATA_DIR', '/app/data')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'cpu_config.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(config.keys())
    writer.writerow(config.values())
