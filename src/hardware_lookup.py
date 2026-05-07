"""Resolve hardware descriptors for CPUs/GPUs the regressor has never seen.

The regressor learned `f(arch_features, hardware_features) → time`, so as
long as we can fill in `hardware_features` for an arbitrary CPU/GPU we can
predict for it — no benchmarks required. Hardware features come from:

  - data_base/data_base.csv          — 5 already-benchmarked platforms (preferred)
  - specs/amd-cpus.csv, intel-cpus.csv, benchmark-cpus.csv  — thousands of CPUs
  - specs/gpu_1986-2026.csv          — thousands of GPUs

Lookup is exact-name only: the Streamlit dropdown supplies the canonical
name (it lists `list_known_cpus()` / `list_known_gpus()`), so there's no
need for typo-correcting fuzzy match. The CSV-cell parsers strip units
(`3.4GHz` → 3400 MHz) but never guess at the user's input.

If a platform is in the baseline, the row from there is used directly
(best — it carries measured cache sizes etc.). Otherwise we synthesise the
descriptor from spec CSVs with defensible defaults for fields the spec
doesn't expose (e.g. cpu memory bandwidth → DDR4-3200 dual-channel ≈ 51.2).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"
DATA_BASE = PROJECT_ROOT / "data_base" / "data_base.csv"

DEFAULT_CPU_BW_GBPS = 51.2          # DDR4-3200 dual-channel
DEFAULT_LAUNCH_US_CPU = 2.0
DEFAULT_LAUNCH_US_GPU = 12.0


# ---- unit parsers (cell content, not user input) --------------------------

def _parse_mhz(s: Any) -> float:
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    n, u = float(m.group(1)), m.group(2).lower()
    return n * 1000 if u == "ghz" else n


def _parse_kb(s: Any) -> float:
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB)", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    n, u = float(m.group(1)), m.group(2).upper()
    return n * {"KB": 1, "MB": 1024, "GB": 1024 * 1024}[u]


def _parse_gb(s: Any) -> float:
    kb = _parse_kb(s)
    return kb / (1024 * 1024) if not pd.isna(kb) else float("nan")


def _parse_flops(s: Any, unit: str = "TFLOPS") -> float:
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TFLOPS|GFLOPS)", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    n, src = float(m.group(1)), m.group(2).upper()
    if unit.upper() == src:
        return n
    return n / 1000 if unit.upper() == "TFLOPS" else n * 1000


def _parse_bandwidth_gbps(s: Any) -> float:
    """'360 GB/s' / '1.5 TB/s' → float in GB/s."""
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GB|TB)/s", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    n, u = float(m.group(1)), m.group(2).upper()
    return n * 1000 if u == "TB" else n


# ---- CSV access -----------------------------------------------------------

@lru_cache(maxsize=1)
def _amd_cpus() -> pd.DataFrame:
    df = pd.read_csv(SPECS_DIR / "amd-cpus.csv", low_memory=False)
    df.columns = [c.strip().replace('"', "") for c in df.columns]
    df["Model"] = df["Model"].str.replace(r"[™®]", "", regex=True).str.strip()
    return df


@lru_cache(maxsize=1)
def _intel_cpus() -> pd.DataFrame:
    return pd.read_csv(SPECS_DIR / "intel-cpus.csv", low_memory=False)


@lru_cache(maxsize=1)
def _bench_cpus() -> pd.DataFrame:
    return pd.read_csv(SPECS_DIR / "benchmark-cpus.csv", low_memory=False)


@lru_cache(maxsize=1)
def _gpus() -> pd.DataFrame:
    df = pd.read_csv(SPECS_DIR / "gpu_1986-2026.csv", low_memory=False)
    df["_full_name"] = (df["Brand"].fillna("").str.strip() + " "
                        + df["Name"].fillna("").str.strip()).str.strip()
    return df


@lru_cache(maxsize=1)
def _baseline_hosts() -> pd.DataFrame:
    df = pd.read_csv(DATA_BASE, low_memory=False)
    return df.drop_duplicates(subset=["cpu_name", "gpu_name"]).reset_index(drop=True)


# ---- catalogue listings (for Streamlit dropdowns) -------------------------

def list_known_cpus() -> list[str]:
    names: set[str] = set()
    names.update(_baseline_hosts()["cpu_name"].dropna().astype(str))
    names.update(_amd_cpus()["Model"].dropna().astype(str).map(lambda s: s.strip()))
    names.update(_intel_cpus()["CpuName"].dropna().astype(str))
    names.update(_bench_cpus()["CpuName"].dropna().astype(str))
    return sorted(names)


def list_known_gpus() -> list[str]:
    names: set[str] = set()
    names.update(_baseline_hosts()["gpu_name"].dropna().astype(str))
    names.update(_gpus()["_full_name"].dropna().astype(str))
    return sorted(n for n in names if n)


# ---- exact-name lookups ---------------------------------------------------

def _baseline_lookup(cpu_name: str | None, gpu_name: str | None) -> pd.Series | None:
    """Return one full row from data_base if BOTH cpu and gpu match."""
    if not cpu_name or not gpu_name:
        return None
    hosts = _baseline_hosts()
    m = (hosts["cpu_name"] == cpu_name) & (hosts["gpu_name"] == gpu_name)
    return hosts[m].iloc[0] if m.any() else None


def lookup_cpu(name: str) -> dict[str, Any]:
    """CPU descriptor by exact catalog name. KeyError if not found."""
    # AMD
    amd = _amd_cpus()
    hit = amd[amd["Model"] == name]
    if len(hit):
        row = hit.iloc[0]
        cores = int(float(row["# of CPU Cores"]))
        threads = int(float(row["# of Threads"]))
        l2 = _parse_kb(row["L2 Cache"])
        l3 = _parse_kb(row["L3 Cache"])
        return {
            "cpu_name": name, "Processor": name,
            "Cores": cores, "Threads": threads,
            "Frequency (MHz)": _parse_mhz(row["Base Clock"]),
            "Max Frequency (MHz)": _parse_mhz(row["Max. Boost Clock ¹ ²"]),
            "l1_cache_size (KB)": _parse_kb(row["L1 Cache"]),
            "l1_instruction_cache_size (KB)": float("nan"),
            "l2_cache_size (KB)": l2,
            "l2_cache_size_per_core (KB)": l2 / cores if pd.notna(l2) else float("nan"),
            "l3_cache_size (MB)": l3 / 1024 if pd.notna(l3) else float("nan"),
            "SMP Supported": threads > cores,
            "cpu_peak_bw_gbps": DEFAULT_CPU_BW_GBPS,
            "launch_overhead_us_cpu": DEFAULT_LAUNCH_US_CPU,
        }
    # Intel
    intel = _intel_cpus()
    hit = intel[intel["CpuName"] == name]
    if len(hit):
        row = hit.iloc[0]
        cores = int(float(row.get("CoreCount", 0) or 0)) or 1
        threads = int(float(row.get("ThreadCount", 0) or 0)) or cores
        l2 = _parse_kb(row.get("TotalL2Cache"))
        l3 = _parse_kb(row.get("Cache"))
        return {
            "cpu_name": name, "Processor": name,
            "Cores": cores, "Threads": threads,
            "Frequency (MHz)": _parse_mhz(row.get("ClockSpeed")),
            "Max Frequency (MHz)": _parse_mhz(row.get("ClockSpeedMax")),
            "l1_cache_size (KB)": float("nan"),
            "l1_instruction_cache_size (KB)": float("nan"),
            "l2_cache_size (KB)": l2,
            "l2_cache_size_per_core (KB)": l2 / cores if pd.notna(l2) else float("nan"),
            "l3_cache_size (MB)": l3 / 1024 if pd.notna(l3) else float("nan"),
            "SMP Supported": threads > cores,
            "cpu_peak_bw_gbps": DEFAULT_CPU_BW_GBPS,
            "launch_overhead_us_cpu": DEFAULT_LAUNCH_US_CPU,
        }
    # ARM/misc
    bench = _bench_cpus()
    hit = bench[bench["CpuName"] == name]
    if len(hit):
        row = hit.iloc[0]
        cores = int(float(row["Cores"]))
        threads = int(float(row["Threads"]))
        return {
            "cpu_name": name, "Processor": name,
            "Cores": cores, "Threads": threads,
            "Frequency (MHz)": _parse_mhz(row["ClockSpeed"]),
            "Max Frequency (MHz)": _parse_mhz(row["TurboSpeed"]),
            "l1_cache_size (KB)": float("nan"),
            "l1_instruction_cache_size (KB)": float("nan"),
            "l2_cache_size (KB)": float("nan"),
            "l2_cache_size_per_core (KB)": float("nan"),
            "l3_cache_size (MB)": float("nan"),
            "SMP Supported": threads > cores,
            "cpu_peak_bw_gbps": DEFAULT_CPU_BW_GBPS,
            "launch_overhead_us_cpu": DEFAULT_LAUNCH_US_CPU,
        }
    raise KeyError(f"CPU {name!r} not in any spec CSV. Use list_known_cpus().")


def lookup_gpu(name: str) -> dict[str, Any]:
    """GPU descriptor by exact catalog name. KeyError if not found."""
    g = _gpus()
    hit = g[g["_full_name"] == name]
    if not len(hit):
        raise KeyError(f"GPU {name!r} not in specs/gpu_1986-2026.csv. Use list_known_gpus().")
    row = hit.iloc[0]
    num_sms = float(row.get("Render Config__SM Count") or 0)
    shading = float(row.get("Render Config__Shading Units") or 0)
    tensor = row.get("Render Config__Tensor Cores")
    cuda_cc = row.get("Graphics Features__CUDA")
    return {
        "gpu_name": name, "GPU Name": name,
        "gpu_memory": _parse_gb(row.get("Top__MEMORY SIZE")),
        "Clock Rate (GHz) boost": _parse_mhz(row.get("Clock Speeds__Boost Clock")) / 1000,
        "Number of SMs": num_sms or float("nan"),
        "Total CUDA Cores": shading or float("nan"),
        "Number of Tensor Cores": float(tensor) if pd.notna(tensor) else 0.0,
        "Compute Capability (cuda version)": float(cuda_cc) if pd.notna(cuda_cc) else float("nan"),
        "FP32 FLOPS (TFLOPS)": _parse_flops(row.get("Theoretical Performance__FP32 (float)"), "TFLOPS"),
        "FP16 FLOPS (TFLOPS)": _parse_flops(row.get("Theoretical Performance__FP16 (half)"), "TFLOPS"),
        "FP64 FLOPS (GFLOPS)": _parse_flops(row.get("Theoretical Performance__FP64 (double)"), "GFLOPS"),
        "Cores per SM": (shading / num_sms) if (num_sms and shading) else float("nan"),
        "gpu_peak_bw_gbps": _parse_bandwidth_gbps(row.get("Memory__Bandwidth")),
        "launch_overhead_us_gpu": DEFAULT_LAUNCH_US_GPU,
    }


def resolve_host(cpu_name: str, gpu_name: str, *, ram_gb: int) -> dict[str, Any]:
    """Build a complete host descriptor dict. Prefers baseline rows, falls back to specs."""
    baseline = _baseline_lookup(cpu_name, gpu_name)
    if baseline is not None:
        return baseline.to_dict()

    cpu = lookup_cpu(cpu_name)
    gpu = lookup_gpu(gpu_name)
    return {**cpu, **gpu, "ram_memory": ram_gb}
