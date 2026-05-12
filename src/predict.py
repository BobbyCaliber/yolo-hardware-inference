"""Predict inference time from hardware + model + sweep dimensions.

Hardware descriptor is resolved from `hardware_lookup`:
  1. Try `data_base/data_base.csv` (5 already-benchmarked platforms — best,
     since it carries the exact measured cache sizes).
  2. Fall back to `specs/{amd-cpus,intel-cpus,benchmark-cpus,gpu_1986-2026}.csv`
     — thousands of CPUs / GPUs catalogued, exact-name match.

Model is looked up in `data_base/model_arch_features.csv`.

Predictions work for any CPU+GPU pair the user can spell from the spec
catalogues — see `python src/predict.py --list-platforms` (5 baseline pairs)
and the much larger `--list-known-cpus` / `--list-known-gpus` for everything
the spec CSVs know about. Streamlit uses the `list_*` helpers as dropdown
sources.

If you want better-than-baseline accuracy on YOUR machine, run
`make collect` to add real benchmarks for it and `make train` to retrain.

Usage:
    python src/predict.py --cpu "AMD Ryzen 7 5700X 8-Core Processor" \
                          --gpu "NVIDIA GeForce RTX 3060" \
                          --used-gpu --ram 32 \
                          --model yolov8m.pt --img-size 640 --batch 5
    python src/predict.py --list-platforms        # baseline (5 measured)
    python src/predict.py --list-known-gpus       # full GPU catalogue
    python src/predict.py --list-known-cpus       # full CPU catalogue
    python src/predict.py --list-models           # registered models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
import enrich_helpers  # noqa: E402
import hardware_lookup as hw  # noqa: E402

DATA_BASE = PROJECT_ROOT / "data_base" / "data_base.csv"
ARCH_CSV = PROJECT_ROOT / "data_base" / "model_arch_features.csv"
BASE_WEIGHTS = PROJECT_ROOT / "data_base" / "reg_weights_base" / "catboost_model.cbm"
NEW_WEIGHTS = PROJECT_ROOT / "data_new" / "reg_weights_new" / "catboost_model.cbm"

CACHE_COLS = (
    "l1_cache_size (KB)", "l1_instruction_cache_size (KB)",
    "l2_cache_size_per_core (KB)", "l2_cache_size (KB)", "l3_cache_size (MB)",
)


def _default_weights() -> Path:
    return NEW_WEIGHTS if NEW_WEIGHTS.is_file() else BASE_WEIGHTS


def list_platforms() -> pd.DataFrame:
    df = pd.read_csv(DATA_BASE, usecols=["cpu_name", "gpu_name"], low_memory=False)
    return df.drop_duplicates().reset_index(drop=True)


def list_models() -> list[str]:
    return sorted(pd.read_csv(ARCH_CSV)["model_name"].unique())


def build_row(*, cpu_name: str, gpu_name: str, used_gpu: int, ram_gb: int,
              model_name: str, img_size: int, batch: int) -> pd.DataFrame:
    if model_name not in list_models():
        sys.exit(f"unknown model {model_name!r}. Use --list-models.")

    try:
        host = hw.resolve_host(cpu_name, gpu_name, ram_gb=ram_gb)
    except KeyError as e:
        sys.exit(str(e))

    arch = pd.read_csv(ARCH_CSV)
    arch_row = arch[arch["model_name"] == model_name].iloc[0]

    row = {**host,
           "model_name": model_name,
           "model_family": arch_row["model_family"],
           "num_all_params": int(arch_row["params"]),
           "model_img_size": img_size,
           "image_batch_size": batch,
           "used_gpu": used_gpu}
    row.setdefault("ram_memory", ram_gb)
    df = pd.DataFrame([row])

    for c in CACHE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    from train_model import _parse_compute_capability
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


def predict(row_df: pd.DataFrame, weights_path: Path, meta_path: Path,
            with_uncertainty: bool = False) -> float | tuple[float, float, float]:
    """Returns predicted seconds. With `with_uncertainty=True`, returns
    `(mean, t_lo, t_hi)` — the central estimate plus a ~84% band derived from
    CatBoost virtual ensembles (μ ± σ_total in log-space, exp'd to seconds)."""
    if not weights_path.is_file():
        sys.exit(f"model weights missing: {weights_path} (run `make train`)")
    if not meta_path.is_file():
        sys.exit(f"metadata missing: {meta_path} (run `make train`)")

    model = CatBoostRegressor()
    model.load_model(str(weights_path))
    with meta_path.open() as f:
        meta = json.load(f)

    missing = [c for c in meta["features"] if c not in row_df.columns]
    if missing:
        sys.exit(f"missing features required by model: {missing}")

    X = row_df[meta["features"]].copy()
    for c in meta["categorical_features"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    trained_with_uncertainty = (
        meta.get("model_params", {}).get("loss_function") == "RMSEWithUncertainty"
    )
    if not trained_with_uncertainty:
        # Legacy weights — fall back to a point estimate even if caller asked
        # for uncertainty (we don't have the variance head to inspect).
        pred = model.predict(X)
        if pred.ndim == 2:
            pred = pred[:, 0]
        sec = float(np.exp(pred[0]))
        return (sec, sec, sec) if with_uncertainty else sec

    if not with_uncertainty:
        pred = model.predict(X)
        if pred.ndim == 2:
            pred = pred[:, 0]
        return float(np.exp(pred[0]))

    # TotalUncertainty: columns are (mean, knowledge_variance, data_variance).
    out = model.virtual_ensembles_predict(
        X, prediction_type="TotalUncertainty", virtual_ensembles_count=10,
    )
    mu = out[:, 0]
    sigma_total = np.sqrt(out[:, 1] + out[:, 2])
    # μ ± 1σ in log-space → ~84% band when back-exponentiated.
    t_pred = np.exp(mu)
    t_lo = np.exp(mu - sigma_total)
    t_hi = np.exp(mu + sigma_total)
    return float(t_pred[0]), float(t_lo[0]), float(t_hi[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu")
    p.add_argument("--gpu")
    p.add_argument("--used-gpu", action="store_true")
    p.add_argument("--ram", type=int, default=32, help="host RAM in GB (default 32)")
    p.add_argument("--model")
    p.add_argument("--img-size", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--meta", type=Path, default=None)
    p.add_argument("--list-platforms", action="store_true",
                   help="show the 5 platforms with measured baseline data")
    p.add_argument("--list-known-cpus", action="store_true",
                   help="show every CPU name in specs/*.csv (~thousands)")
    p.add_argument("--list-known-gpus", action="store_true",
                   help="show every GPU name in specs/gpu_1986-2026.csv (~thousands)")
    p.add_argument("--list-models", action="store_true")
    args = p.parse_args()

    if args.list_platforms:
        print(list_platforms().to_string(index=False)); return
    if args.list_known_cpus:
        for n in hw.list_known_cpus(): print(n)
        return
    if args.list_known_gpus:
        for n in hw.list_known_gpus(): print(n)
        return
    if args.list_models:
        for m in list_models(): print(m)
        return

    required = ["cpu", "gpu", "model", "img_size", "batch"]
    missing = [r for r in required if getattr(args, r) is None]
    if missing:
        sys.exit(f"missing required args: {', '.join('--' + r.replace('_', '-') for r in missing)}")

    weights_path = args.weights or _default_weights()
    meta_path = args.meta or (weights_path.parent / "metadata.json")
    print(f"[INFO] weights: {weights_path}")

    row = build_row(cpu_name=args.cpu, gpu_name=args.gpu,
                    used_gpu=1 if args.used_gpu else 0, ram_gb=args.ram,
                    model_name=args.model, img_size=args.img_size, batch=args.batch)

    sec, t_lo, t_hi = predict(row, weights_path, meta_path, with_uncertainty=True)
    band = f"  [{t_lo*1000:.1f} – {t_hi*1000:.1f} ms ~84% band]" if t_lo != t_hi else ""
    print(f"\npredicted inference time: {sec:.4f} sec  ({sec * 1000:.1f} ms){band}  "
          f"for {args.model} @ {args.img_size}px × batch {args.batch} on "
          f"{'GPU' if args.used_gpu else 'CPU'}")


if __name__ == "__main__":
    main()
