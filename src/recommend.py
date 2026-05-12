"""Inverse problem: cheapest pre-built PC for a given (model, latency, budget).

Given a CV model + target inference time + max RUB budget, walks the DNS-shop
PC catalogue (`specs/dns_pcs.json`, produced by `scripts/scrape_dns_pcs.py`)
and returns the cheapest builds whose predicted inference time is within the
target.

Pipeline per build:
    DNS slug-name → fuzzy-match against specs/ catalogues → hardware_lookup
    → catboost predict → keep if t ≤ target ∧ price ≤ budget → sort by price.

Usage:
    python src/recommend.py --model yolov8m.pt --img-size 640 --batch 1 \
        --max-latency 0.05 --max-budget 200000
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from rapidfuzz import fuzz, process

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
import enrich_helpers  # noqa: E402
import hardware_lookup as hw  # noqa: E402
from train_model import _parse_compute_capability  # noqa: E402

ARCH_CSV = PROJECT_ROOT / "data_base" / "model_arch_features.csv"
NEW_WEIGHTS = PROJECT_ROOT / "data_new" / "reg_weights_new"
BASE_WEIGHTS = PROJECT_ROOT / "data_base" / "reg_weights_base"
DEFAULT_DNS_JSON = PROJECT_ROOT / "specs" / "dns_pcs.json"

CACHE_COLS = [
    "l1_cache_size (KB)", "l1_instruction_cache_size (KB)",
    "l2_cache_size_per_core (KB)", "l2_cache_size (KB)", "l3_cache_size (MB)",
]

LOG = logging.getLogger("recommend")


# ---------------------------------------------------------------- name match

# DNS prepends retail brand to GPUs ("KFA2 GeForce RTX 3050 X Black …") and
# appends OEM/BOX/WOF to CPUs ("AMD Ryzen 5 5500 OEM"). Both need to be peeled
# back to a chip-level identifier before fuzzy-matching against specs/.

_CPU_TRIM_RE = re.compile(r"\b(oem|box|wof|tray|mpk)\b\.?", re.IGNORECASE)
_GPU_CHIP_RE = re.compile(
    r"\b(GeForce|Radeon|Arc)\s+"
    r"(RTX|GTX|RX|A)\s*"
    r"(\d{3,4})"
    r"(?:\s*(Ti\s*Super|Ti|XT|XTX|Super))?",
    re.IGNORECASE,
)
_INTEL_DASH_RE = re.compile(r"\b(i[3579])\s+(\d{4,5}[a-z]{0,2})\b", re.IGNORECASE)


def _normalize_cpu(dns_name: str) -> str:
    """`'INTEL Core I5 12400f OEM'` → `'intel core i5-12400f'`."""
    s = _CPU_TRIM_RE.sub("", dns_name).strip()
    s = _INTEL_DASH_RE.sub(lambda m: f"{m.group(1)}-{m.group(2)}", s)
    s = re.sub(r"\s+", " ", s).lower()
    return s


def _extract_gpu_chip(dns_name: str) -> str | None:
    """`'KFA2 GEFORCE RTX 3050 X Black 35nrldhp9odk'` → `'GeForce RTX 3050'`."""
    m = _GPU_CHIP_RE.search(dns_name)
    if not m:
        return None
    family = m.group(1).capitalize()         # GeForce / Radeon / Arc
    cls = m.group(2).upper()                 # RTX / GTX / RX / A
    num = m.group(3)
    suffix = m.group(4) or ""
    suffix = re.sub(r"\s+", " ", suffix).title() if suffix else ""
    pieces = [family, cls, num] + ([suffix] if suffix else [])
    return " ".join(pieces)


@lru_cache(maxsize=1)
def _known_cpus_normalised() -> list[tuple[str, str]]:
    """List of (normalised_key, original_catalog_name) tuples."""
    return [(_normalize_cpu(c), c) for c in hw.list_known_cpus()]


@lru_cache(maxsize=1)
def _known_gpus_list() -> list[str]:
    return hw.list_known_gpus()


# SKU tokens: 3+ digits with optional alphanumeric suffix (handles `5500`,
# `7800X3D`, `12400F`, `9800X3D`).
_MODEL_NUM_RE = re.compile(r"\b(\d{3,5}(?:[a-z][a-z0-9]*)?)\b", re.IGNORECASE)


def _model_numbers(s: str) -> list[str]:
    """Extract numeric SKU tokens. rapidfuzz scores numeric strings by edit
    distance ('3050' ≈ '3060'), so we use the SKU as a hard filter before any
    fuzzy step.
    """
    return [m.group(1).lower() for m in _MODEL_NUM_RE.finditer(s)]


def _suffix_signature(s: str) -> tuple[bool, bool, bool, bool]:
    """(has_ti, has_xt, has_super, has_x3d) — must match across query and candidate."""
    return (
        bool(re.search(r"\bti\b", s, re.IGNORECASE)),
        bool(re.search(r"\bxt[x]?\b", s, re.IGNORECASE)),
        bool(re.search(r"\bsuper\b", s, re.IGNORECASE)),
        bool(re.search(r"x3d\b", s, re.IGNORECASE)),
    )


@lru_cache(maxsize=4096)
def match_cpu(dns_name: str, threshold: int = 70) -> str | None:
    key = _normalize_cpu(dns_name)
    nums = _model_numbers(key)
    if not nums:
        return None
    q_sig = _suffix_signature(key)
    pool = _known_cpus_normalised()
    # Require an exact SKU token match + suffix parity (X3D / F / etc. land in
    # the SKU itself so they're already filtered there).
    filtered = [(k, orig) for k, orig in pool
                if any(re.search(rf"\b{re.escape(n)}\b", k) for n in nums)
                and _suffix_signature(k) == q_sig]
    if not filtered:
        return None
    res = process.extractOne(key, [k for k, _ in filtered],
                             scorer=fuzz.token_set_ratio)
    if res is None or res[1] < threshold:
        return None
    return filtered[res[2]][1]


@lru_cache(maxsize=4096)
def match_gpu(dns_name: str) -> str | None:
    chip = _extract_gpu_chip(dns_name)
    if not chip:
        return None
    nums = _model_numbers(chip)
    if not nums:
        return None
    q_sig = _suffix_signature(chip)
    pool = _known_gpus_list()
    cands = [g for g in pool
             if any(re.search(rf"\b{re.escape(n)}\b", g, re.IGNORECASE) for n in nums)
             and _suffix_signature(g) == q_sig
             and "Mobile" not in g and "Max-Q" not in g]
    if not cands:
        return None
    # SKU + suffix parity already guarantees same chip — pick the most generic
    # entry (shortest / no memory qualifier).
    cands.sort(key=len)
    return cands[0]


# ---------------------------------------------------------------- predictor

def _default_weights_dir() -> Path:
    return NEW_WEIGHTS if (NEW_WEIGHTS / "catboost_model.cbm").is_file() else BASE_WEIGHTS


def _load_predictor(weights_dir: Path) -> tuple[CatBoostRegressor, dict]:
    weights = weights_dir / "catboost_model.cbm"
    meta_path = weights_dir / "metadata.json"
    if not weights.is_file():
        sys.exit(f"weights missing: {weights} (run `make train`)")
    if not meta_path.is_file():
        sys.exit(f"metadata missing: {meta_path}")
    model = CatBoostRegressor()
    model.load_model(str(weights))
    meta = json.loads(meta_path.read_text())
    return model, meta


def _build_row(cpu_name: str, gpu_name: str, ram_gb: int,
               model_name: str, img_size: int, batch: int,
               arch_row: pd.Series) -> dict:
    host = hw.resolve_host(cpu_name, gpu_name, ram_gb=ram_gb)
    row = {**host,
           "model_name": model_name,
           "model_family": arch_row["model_family"],
           "num_all_params": int(arch_row["params"]),
           "model_img_size": img_size,
           "image_batch_size": batch,
           "used_gpu": 1}
    row.setdefault("ram_memory", ram_gb)
    return row


def _enrich_frame(df: pd.DataFrame) -> pd.DataFrame:
    for c in CACHE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Compute Capability (cuda version)"] = (
        df["Compute Capability (cuda version)"].map(_parse_compute_capability).astype(float)
    )
    df["pixels"] = df["model_img_size"] ** 2 * df["image_batch_size"]
    df["work_estimate"] = df["num_all_params"] * df["pixels"]
    df["cpu_throughput"] = df["Cores"] * df["Frequency (MHz)"] * 2e-3
    df["compute_throughput"] = np.where(
        df["used_gpu"] == 1, df["FP32 FLOPS (TFLOPS)"] * 1000, df["cpu_throughput"],
    )
    return enrich_helpers.enrich(df, arch_csv=ARCH_CSV)


# ---------------------------------------------------------------- main

def recommend(model_name: str, img_size: int, batch: int,
              max_latency_s: float, max_budget_rub: float,
              dns_json: Path = DEFAULT_DNS_JSON,
              weights_dir: Path | None = None,
              top_k: int = 10) -> pd.DataFrame:
    if not dns_json.is_file():
        sys.exit(f"DNS catalogue missing: {dns_json} (run scripts/scrape_dns_pcs.py)")
    payload = json.loads(dns_json.read_text(encoding="utf-8"))
    builds = payload["builds"]
    LOG.info("loaded %d builds from %s (scraped %s)",
             len(builds), dns_json.name, payload.get("scraped_at"))

    arch = pd.read_csv(ARCH_CSV)
    if model_name not in set(arch["model_name"]):
        sys.exit(f"unknown model {model_name!r}. See `predict.py --list-models`.")
    arch_row = arch[arch["model_name"] == model_name].iloc[0]

    weights_dir = weights_dir or _default_weights_dir()
    model, meta = _load_predictor(weights_dir)
    LOG.info("predictor: %s", weights_dir.relative_to(PROJECT_ROOT))

    rows = []
    n_over_budget = n_no_gpu = n_no_cpu_match = n_no_gpu_match = n_no_host = 0
    for b in builds:
        if b["price_rub"] is None or b["price_rub"] > max_budget_rub:
            n_over_budget += 1
            continue
        if not b.get("gpu_name"):
            n_no_gpu += 1
            continue
        cpu_canon = match_cpu(b["cpu_name"])
        if cpu_canon is None:
            n_no_cpu_match += 1
            continue
        gpu_canon = match_gpu(b["gpu_name"])
        if gpu_canon is None:
            n_no_gpu_match += 1
            continue
        ram = b.get("ram_gb") or 16
        try:
            row = _build_row(cpu_canon, gpu_canon, ram, model_name,
                             img_size, batch, arch_row)
        except (KeyError, ValueError, TypeError):
            # Some spec rows have NaN cores/cache (mobile/ARM stubs) — skip.
            n_no_host += 1
            continue
        row["_build_url"] = b["url"]
        row["_price_rub"] = b["price_rub"]
        row["_dns_cpu"] = b["cpu_name"]
        row["_dns_gpu"] = b["gpu_name"]
        row["_cpu_canon"] = cpu_canon
        row["_gpu_canon"] = gpu_canon
        rows.append(row)

    LOG.info("filtered: %d candidates  (skipped: %d over-budget, %d no-GPU, "
             "%d unmatched-CPU, %d unmatched-GPU, %d no-host)",
             len(rows), n_over_budget, n_no_gpu, n_no_cpu_match,
             n_no_gpu_match, n_no_host)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    meta_cols = [c for c in df.columns if c.startswith("_")]
    feat_df = df.drop(columns=meta_cols)
    feat_df = _enrich_frame(feat_df)

    X = feat_df[meta["features"]].copy()
    for c in meta["categorical_features"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    trained_with_uncertainty = (
        meta.get("model_params", {}).get("loss_function") == "RMSEWithUncertainty"
    )
    if trained_with_uncertainty:
        # cols: (mean, knowledge_variance — epistemic, data_variance — aleatoric)
        out = model.virtual_ensembles_predict(
            X, prediction_type="TotalUncertainty", virtual_ensembles_count=10,
        )
        mu = out[:, 0]
        sigma = np.sqrt(out[:, 1] + out[:, 2])
        df["predicted_t_s"] = np.exp(mu)
        df["t_lo"] = np.exp(mu - sigma)              # ~16th percentile
        df["t_hi"] = np.exp(mu + sigma)              # ~84th percentile
    else:
        preds = model.predict(X)
        if preds.ndim == 2:
            preds = preds[:, 0]
        df["predicted_t_s"] = np.exp(preds)
        df["t_lo"] = df["predicted_t_s"]
        df["t_hi"] = df["predicted_t_s"]

    # Conservative filter: a build only counts if its UPPER ~84%-bound fits the
    # latency target. This biases against high-uncertainty unseen hardware
    # whose point estimate might happen to fall under the target by luck.
    keep = df[df["t_hi"] <= max_latency_s].copy()
    LOG.info("after latency filter (t_hi ≤%.4fs): %d / %d builds pass",
             max_latency_s, len(keep), len(df))
    if keep.empty:
        return keep
    keep = keep.sort_values("_price_rub").head(top_k)

    result = keep[[
        "_price_rub", "predicted_t_s", "t_lo", "t_hi",
        "_cpu_canon", "_gpu_canon", "ram_memory",
        "_dns_cpu", "_dns_gpu", "_build_url",
    ]].rename(columns={
        "_price_rub": "price_rub",
        "_cpu_canon": "cpu_matched",
        "_gpu_canon": "gpu_matched",
        "ram_memory": "ram_gb",
        "_dns_cpu": "dns_cpu_name",
        "_dns_gpu": "dns_gpu_name",
        "_build_url": "url",
    }).reset_index(drop=True)
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="e.g. yolov8m.pt")
    p.add_argument("--img-size", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--max-latency", type=float, required=True,
                   help="target inference time in seconds")
    p.add_argument("--max-budget", type=float, required=True,
                   help="budget cap in ₽")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dns-json", type=Path, default=DEFAULT_DNS_JSON)
    p.add_argument("--weights-dir", type=Path, default=None,
                   help=f"override CatBoost weights (default: {NEW_WEIGHTS.relative_to(PROJECT_ROOT)} if present, else baseline)")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level,
                        format="[%(levelname)s] %(message)s")
    result = recommend(
        model_name=args.model,
        img_size=args.img_size,
        batch=args.batch,
        max_latency_s=args.max_latency,
        max_budget_rub=args.max_budget,
        dns_json=args.dns_json,
        weights_dir=args.weights_dir,
        top_k=args.top_k,
    )
    if result.empty:
        print("no builds satisfy the constraints.")
        return
    pd.options.display.max_colwidth = 60
    pd.options.display.width = 200
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
