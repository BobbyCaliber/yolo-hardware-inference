"""Predict YOLO inference time from hardware + model specs.

User supplies 7 values (CPU name, GPU name, used_gpu flag, RAM GB, YOLO
model, img size, batch size). Everything else — CPU frequency / cores /
cache, GPU SMs / CUDA cores / FLOPS, YOLO param count — is resolved from
the CSVs in specs/ and data_base/data_base.csv.

Preprocessing matches src/train_model.py so the saved CatBoost weights
apply directly without retraining.

Usage:
    python src/predict.py --cpu "AMD Ryzen 7 5700X" --gpu "RTX 3060" \
                          --used-gpu --ram 32 \
                          --model yolov8m.pt --img-size 640 --batch 5

    python src/predict.py --list-models            # show supported YOLO weights
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
SPECS_DIR = PROJECT_ROOT / "specs"
DATA_BASE = PROJECT_ROOT / "data_base" / "data_base.csv"
# Baseline weights shipped with the repo (trained on data_base.csv) live in
# data_base/reg_weights_base/. After `make train`, fresher weights land in
# data_new/reg_weights_new/ — prefer those when present.
BASE_WEIGHTS = PROJECT_ROOT / "data_base" / "reg_weights_base" / "catboost_model.cbm"
NEW_WEIGHTS = PROJECT_ROOT / "data_new" / "reg_weights_new" / "catboost_model.cbm"


def _default_weights() -> Path:
    """Prefer freshly-trained weights in data_new; fall back to repo baseline."""
    return NEW_WEIGHTS if NEW_WEIGHTS.is_file() else BASE_WEIGHTS


# ---------------------------------------------------------------- parsers

def _parse_mhz(s: Any) -> float:
    """'3.4GHz' / '1777 MHz' / 'Up to 4.6GHz' → float MHz."""
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz)", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    num, unit = float(m.group(1)), m.group(2).lower()
    return num * 1000 if unit == "ghz" else num


def _parse_kb(s: Any) -> float:
    """'512KB' / '4MB' / '32MB' / '30 MB Intel® Smart Cache' → float KB."""
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB)", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    num, unit = float(m.group(1)), m.group(2).upper()
    scale = {"KB": 1, "MB": 1024, "GB": 1024 * 1024}[unit]
    return num * scale


def _parse_gb(s: Any) -> float:
    """'12 GB' / '40 GB' → float GB."""
    kb = _parse_kb(s)
    return kb / (1024 * 1024) if not pd.isna(kb) else float("nan")


def _parse_flops(s: Any, unit: str = "TFLOPS") -> float:
    """'12.74 TFLOPS (1:1)' / '199.0 GFLOPS (1:64)' → float in requested unit."""
    if pd.isna(s):
        return float("nan")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(TFLOPS|GFLOPS)", str(s), re.IGNORECASE)
    if not m:
        return float("nan")
    num, src_unit = float(m.group(1)), m.group(2).upper()
    if unit.upper() == src_unit:
        return num
    if unit.upper() == "TFLOPS":
        return num / 1000
    return num * 1000


# GPU model variants that should NOT be auto-selected when the user didn't
# explicitly ask for them — "RTX 3060" must not match "RTX 3060 Ti", etc.
_VARIANT_SUFFIXES = {
    "ti", "super", "xt", "xtx", "max-q", "mobile", "laptop", "oem", "koduri",
}


def _fuzzy_match(series: pd.Series, query: str) -> int | None:
    """Lookup with word-boundary matching + variant-suffix filter.

    - Uses \\b word boundaries so 'A100' does not match 'A100X'.
    - Rejects candidates where the next token is a variant suffix (Ti, Super,
      XT, etc.) unless the user explicitly included it in the query.
    - Ties broken by shortest candidate name.
    """
    q = query.lower().strip()
    q_tokens = set(re.split(r"\s+", q))
    lower = series.fillna("").str.lower()

    def _best(mask: pd.Series) -> int:
        candidates = series.index[mask]
        return int(min(candidates, key=lambda i: len(str(series.loc[i]))))

    # 1) exact case-insensitive
    hit = lower == q
    if hit.any():
        return _best(hit)

    # 2) word-boundary regex + variant-suffix filter
    pattern = r"\b" + re.escape(q) + r"\b"
    def accept(s: str) -> bool:
        m = re.search(pattern, s)
        if not m:
            return False
        after = s[m.end():].lstrip()
        if after:
            next_word = re.split(r"\s+", after, maxsplit=1)[0].strip(",;()[]")
            if next_word in _VARIANT_SUFFIXES and next_word not in q_tokens:
                return False
        return True

    hit = lower.apply(accept)
    if hit.any():
        return _best(hit)

    # 3) last-ditch: every query word appears somewhere
    words = [w for w in re.split(r"\s+", q) if w]
    hit = lower.apply(lambda s: all(w in s for w in words))
    if hit.any():
        return _best(hit)
    return None


# ------------------------------------------------------------ resolvers

def resolve_cpu(cpu_name: str) -> dict[str, Any]:
    n = cpu_name.lower()
    spec: dict[str, Any] = {"cpu_name": cpu_name, "Processor": cpu_name}

    if any(k in n for k in ("amd", "ryzen", "epyc", "threadripper")):
        df = pd.read_csv(SPECS_DIR / "amd-cpus.csv", low_memory=False)
        df.columns = [c.strip().replace('"', "") for c in df.columns]
        # user's "AMD Ryzen 7 5700X" vs CSV's "AMD Ryzen™ 7 5700X" — strip punctuation + AMD prefix
        query = re.sub(r"[™®]", "", cpu_name)
        idx = _fuzzy_match(df["Model"].str.replace(r"[™®]", "", regex=True), query)
        if idx is not None:
            row = df.loc[idx]
            cores = int(float(row["# of CPU Cores"]))
            threads = int(float(row["# of Threads"]))
            l2_total_kb = _parse_kb(row["L2 Cache"])
            l3_kb = _parse_kb(row["L3 Cache"])
            spec.update({
                "Cores": cores,
                "Threads": threads,
                "Frequency (MHz)": _parse_mhz(row["Base Clock"]),
                "Max Frequency (MHz)": _parse_mhz(row["Max. Boost Clock ¹ ²"]),
                "l1_cache_size (KB)": _parse_kb(row["L1 Cache"]),
                "l1_instruction_cache_size (KB)": float("nan"),
                "l2_cache_size (KB)": l2_total_kb,
                "l2_cache_size_per_core (KB)": l2_total_kb / cores if not pd.isna(l2_total_kb) else float("nan"),
                "l3_cache_size (MB)": l3_kb / 1024 if not pd.isna(l3_kb) else float("nan"),
                "SMP Supported": threads > cores,
                "_source": "amd-cpus.csv",
                "_matched": row["Model"],
            })
            return spec

    if any(k in n for k in ("intel", "core", "xeon", "pentium", "celeron")):
        df = pd.read_csv(SPECS_DIR / "intel-cpus.csv", low_memory=False)
        idx = _fuzzy_match(df["CpuName"].astype(str), cpu_name)
        if idx is not None:
            row = df.loc[idx]
            cores = int(float(row.get("CoreCount", 0)) or 0) or None
            threads = int(float(row.get("ThreadCount", 0)) or 0) or cores
            l2_kb = _parse_kb(row.get("TotalL2Cache"))
            l3_kb = _parse_kb(row.get("Cache"))
            spec.update({
                "Cores": cores,
                "Threads": threads,
                "Frequency (MHz)": _parse_mhz(row.get("ClockSpeed")),
                "Max Frequency (MHz)": _parse_mhz(row.get("ClockSpeedMax")),
                "l1_cache_size (KB)": float("nan"),
                "l1_instruction_cache_size (KB)": float("nan"),
                "l2_cache_size (KB)": l2_kb,
                "l2_cache_size_per_core (KB)": l2_kb / cores if (l2_kb and cores) else float("nan"),
                "l3_cache_size (MB)": l3_kb / 1024 if not pd.isna(l3_kb) else float("nan"),
                "SMP Supported": (threads or 0) > (cores or 0),
                "_source": "intel-cpus.csv",
                "_matched": row["CpuName"],
            })
            return spec

    # fallback — ARM / misc
    df = pd.read_csv(SPECS_DIR / "benchmark-cpus.csv", low_memory=False)
    idx = _fuzzy_match(df["CpuName"].astype(str), cpu_name)
    if idx is not None:
        row = df.loc[idx]
        cores = int(float(row["Cores"]))
        threads = int(float(row["Threads"]))
        spec.update({
            "Cores": cores,
            "Threads": threads,
            "Frequency (MHz)": _parse_mhz(row["ClockSpeed"]),
            "Max Frequency (MHz)": _parse_mhz(row["TurboSpeed"]),
            "l1_cache_size (KB)": float("nan"),
            "l1_instruction_cache_size (KB)": float("nan"),
            "l2_cache_size (KB)": float("nan"),
            "l2_cache_size_per_core (KB)": float("nan"),
            "l3_cache_size (MB)": float("nan"),
            "SMP Supported": threads > cores,
            "_source": "benchmark-cpus.csv",
            "_matched": row["CpuName"],
        })
        return spec

    raise SystemExit(f"CPU '{cpu_name}' not found in any of specs/*.csv")


def resolve_gpu(gpu_name: str) -> dict[str, Any]:
    df = pd.read_csv(SPECS_DIR / "gpu_1986-2026.csv", low_memory=False)
    df["_full_name"] = df["Brand"].fillna("") + " " + df["Name"].fillna("")
    # normalise query: "NVIDIA GeForce RTX 3060" / "RTX 3060" → match either
    query = gpu_name
    idx = _fuzzy_match(df["_full_name"], query)
    if idx is None:
        # relax: drop "NVIDIA"/"GeForce" prefixes
        bare = re.sub(r"(NVIDIA|GeForce|AMD|Radeon|Intel|Arc)\s*", "", gpu_name, flags=re.I).strip()
        idx = _fuzzy_match(df["_full_name"], bare)
    if idx is None:
        raise SystemExit(f"GPU '{gpu_name}' not found in specs/gpu_1986-2026.csv")

    row = df.loc[idx]
    name_full = f"{row['Brand']} {row['Name']}".strip()
    num_sms = float(row.get("Render Config__SM Count") or 0)
    shading = float(row.get("Render Config__Shading Units") or 0)
    tensor = row.get("Render Config__Tensor Cores")
    cuda_cc = row.get("Graphics Features__CUDA")

    spec = {
        "gpu_name": name_full,
        "GPU Name": name_full,
        "gpu_memory": _parse_gb(row.get("Top__MEMORY SIZE")),
        "Clock Rate (GHz) boost": _parse_mhz(row.get("Clock Speeds__Boost Clock")) / 1000,
        "Number of SMs": num_sms if num_sms else float("nan"),
        "Total CUDA Cores": shading if shading else float("nan"),
        "Number of Tensor Cores": float(tensor) if pd.notna(tensor) else 0.0,
        "Compute Capability (cuda version)": float(cuda_cc) if pd.notna(cuda_cc) else float("nan"),
        "FP32 FLOPS (TFLOPS)": _parse_flops(row.get("Theoretical Performance__FP32 (float)"), "TFLOPS"),
        "FP16 FLOPS (TFLOPS)": _parse_flops(row.get("Theoretical Performance__FP16 (half)"), "TFLOPS"),
        "FP64 FLOPS (GFLOPS)": _parse_flops(row.get("Theoretical Performance__FP64 (double)"), "GFLOPS"),
        "Cores per SM": (shading / num_sms) if (num_sms and shading) else float("nan"),
        "_matched": name_full,
    }
    return spec


def load_model_param_lookup() -> dict[str, int]:
    df = pd.read_csv(DATA_BASE, usecols=["model_name", "num_all_params"])
    return df.groupby("model_name")["num_all_params"].first().to_dict()


# ----------------------------------------------------------- preprocess

def build_row(cpu: dict, gpu: dict, *, used_gpu: int, ram: int,
              model_name: str, img_size: int, batch: int,
              model_params: dict[str, int]) -> pd.DataFrame:
    if model_name not in model_params:
        known = sorted(model_params.keys())
        raise SystemExit(f"unknown YOLO model '{model_name}'. Known: {known}")

    row = {
        "cpu_name": cpu["cpu_name"],
        "gpu_name": gpu["gpu_name"],
        "gpu_memory": gpu["gpu_memory"],
        "used_gpu": used_gpu,
        "ram_memory": ram,
        "model_name": model_name,
        "num_all_params": model_params[model_name],
        "model_img_size": img_size,
        "image_batch_size": batch,
    }
    for k, v in cpu.items():
        if not k.startswith("_") and k != "cpu_name":
            row[k] = v
    for k, v in gpu.items():
        if not k.startswith("_") and k != "gpu_name":
            row[k] = v

    df = pd.DataFrame([row])
    # derived features — exactly same formulas as train_model.preprocess()
    df["pixels"] = df["model_img_size"] ** 2 * df["image_batch_size"]
    df["work_estimate"] = df["num_all_params"] * df["pixels"]
    df["cpu_throughput"] = df["Cores"] * df["Frequency (MHz)"] * 2e-3
    df["compute_throughput"] = np.where(
        df["used_gpu"] == 1,
        df["FP32 FLOPS (TFLOPS)"] * 1000,
        df["cpu_throughput"],
    )
    df["model_family"] = df["model_name"].str.extract(r"^(yolo\w*?\d+)")[0]
    return df


# ----------------------------------------------------------- prediction

def predict(row_df: pd.DataFrame, weights_path: Path, meta_path: Path) -> float:
    if not weights_path.is_file():
        raise SystemExit(f"model weights missing: {weights_path} (run `make train`)")
    if not meta_path.is_file():
        raise SystemExit(f"metadata missing: {meta_path} (run `make train`)")

    model = CatBoostRegressor()
    model.load_model(str(weights_path))
    with meta_path.open() as f:
        meta = json.load(f)

    missing = [c for c in meta["features"] if c not in row_df.columns]
    if missing:
        raise SystemExit(f"missing features required by model: {missing}")

    X = row_df[meta["features"]].copy()
    for c in meta["categorical_features"]:
        X[c] = X[c].astype("category")

    log_pred = model.predict(X)
    return float(np.exp(log_pred[0]))


# ------------------------------------------------------------------ CLI

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu", help='CPU name (e.g. "AMD Ryzen 7 5700X")')
    p.add_argument("--gpu", help='GPU name (e.g. "RTX 3060")')
    p.add_argument("--used-gpu", action="store_true",
                   help="use GPU for inference (omit for CPU-only)")
    p.add_argument("--ram", type=int, help="host RAM in GB")
    p.add_argument("--model", help="YOLO weights (e.g. yolov8m.pt)")
    p.add_argument("--img-size", type=int, help="input image side in px (e.g. 640)")
    p.add_argument("--batch", type=int, help="batch size (e.g. 5)")
    p.add_argument("--weights", type=Path, default=None,
                   help="path to catboost_model.cbm (default: data_new/reg_weights_new/ "
                        "if present, else data_base/reg_weights_base/)")
    p.add_argument("--meta", type=Path, default=None,
                   help="path to metadata.json (default: alongside --weights)")
    p.add_argument("--list-models", action="store_true",
                   help="list YOLO models known to the training set and exit")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_models:
        print("YOLO models in training data:")
        for m in sorted(load_model_param_lookup().keys()):
            print(f"  {m}")
        return

    required = ["cpu", "gpu", "ram", "model", "img_size", "batch"]
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        sys.exit(f"missing required args: {', '.join('--' + r.replace('_', '-') for r in missing)}")

    weights_path = args.weights or _default_weights()
    meta_path = args.meta or (weights_path.parent / "metadata.json")
    source = "data_new (fresh)" if weights_path == NEW_WEIGHTS else "data_base (baseline)"
    print(f"[INFO] weights: {weights_path}  [{source}]")

    cpu_spec = resolve_cpu(args.cpu)
    gpu_spec = resolve_gpu(args.gpu)
    model_params = load_model_param_lookup()

    print(f"[INFO] CPU matched: {cpu_spec['_matched']}  (source: {cpu_spec['_source']})")
    print(f"       {cpu_spec['Cores']}c/{cpu_spec['Threads']}t  "
          f"{cpu_spec['Frequency (MHz)']}→{cpu_spec['Max Frequency (MHz)']} MHz  "
          f"L2={cpu_spec['l2_cache_size (KB)']}KB  L3={cpu_spec['l3_cache_size (MB)']}MB")
    print(f"[INFO] GPU matched: {gpu_spec['_matched']}")
    print(f"       {gpu_spec['Number of SMs']:.0f} SMs × {gpu_spec['Cores per SM']:.0f} = "
          f"{gpu_spec['Total CUDA Cores']:.0f} cores, "
          f"FP32 {gpu_spec['FP32 FLOPS (TFLOPS)']:.2f} TFLOPS, CC {gpu_spec['Compute Capability (cuda version)']}")

    row = build_row(
        cpu_spec, gpu_spec,
        used_gpu=1 if args.used_gpu else 0,
        ram=args.ram,
        model_name=args.model,
        img_size=args.img_size,
        batch=args.batch,
        model_params=model_params,
    )

    sec = predict(row, weights_path, meta_path)
    print(f"\npredicted inference time: {sec:.4f} sec  ({sec * 1000:.1f} ms)  "
          f"for {args.model} @ {args.img_size}px × batch {args.batch} on "
          f"{'GPU' if args.used_gpu else 'CPU'}")


if __name__ == "__main__":
    main()
