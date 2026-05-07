"""Stage-2 ablation: do architectural features replace `model_name`?

Trains four feature variants (A/B/C/D) under two GroupKFold schemes and
writes a markdown report.

Variants
  A baseline       — current features, includes `model_name` & `model_family`
  B +arch          — A plus arch features
  C −id +arch      — arch features only, NO `model_name` / `model_family`
  D C + roofline   — variant C plus `log_t_theoretical`

Validation schemes
  H. by hardware       — GroupKFold(groups = cpu_name + " | " + gpu_name)
  F. by model_family   — GroupKFold(groups = model_family)   ← the new test

Output: research/ablation_arch_features.md

Pass-gate (per the plan):
  variant D under scheme F should lose ≤ ~5 pp R² vs variant A under scheme H.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --data data_base/data_base_enriched.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupKFold

LOG = logging.getLogger("run_ablation")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data_base" / "data_base_enriched.csv"
DEFAULT_OUT = PROJECT_ROOT / "research" / "ablation_arch_features.md"

DROP_COLS = [
    "model_param_dict",
    "Processor",
    "GPU Name",
    "FP64 TENSOR FLOPS (TFLOPS)",
    "BF16 FLOPS (TFLOPS)",
    "TF32 FLOPS (TFLOPS)",
]
CACHE_COLS = [
    "l1_cache_size (KB)",
    "l1_instruction_cache_size (KB)",
    "l2_cache_size_per_core (KB)",
    "l2_cache_size (KB)",
    "l3_cache_size (MB)",
]
TARGET_RAW = "predicting_times"
TARGET_LOG = "log_time"
ARCH_NUMERIC_PREFIXES = ("op_",)
ARCH_NUMERIC = [
    "flops_ref", "macs_ref", "params_ref",
    "activation_bytes_ref", "peak_activation_ref",
    "num_ops", "depth", "arithmetic_intensity",
    "img_size_ref",
]
ROOFLINE_FEATS = ["t_theoretical", "log_t_theoretical",
                  "flops_actual", "activation_bytes_actual"]
BANDWIDTH_FEATS = ["cpu_peak_bw_gbps", "gpu_peak_bw_gbps",
                   "launch_overhead_us_cpu", "launch_overhead_us_gpu"]
MODEL_PARAMS = dict(iterations=600, depth=8, learning_rate=0.05, random_seed=42, verbose=0)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns]).copy()
    for c in CACHE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["Compute Capability (cuda version)"].dtype == object:
        df["Compute Capability (cuda version)"] = (
            df["Compute Capability (cuda version)"]
            .str.replace(",", ".").str[1:-1].str.replace(" ", "").astype(float)
        )
    df["used_gpu"] = df["used_gpu"].astype(int)
    df["pixels"] = df["model_img_size"] ** 2 * df["image_batch_size"]
    df["work_estimate"] = df["num_all_params"] * df["pixels"]
    df["cpu_throughput"] = df["Cores"] * df["Frequency (MHz)"] * 2e-3
    df["compute_throughput"] = np.where(
        df["used_gpu"] == 1,
        df["FP32 FLOPS (TFLOPS)"] * 1000,
        df["cpu_throughput"],
    )
    df[TARGET_LOG] = np.log(df[TARGET_RAW])
    df["model_family"] = df["model_name"].str.extract(r"^(yolo\w*?\d+)")[0]
    return df


@dataclass
class Variant:
    code: str
    label: str
    use_id: bool          # include model_name / model_family
    use_arch: bool        # include arch numeric + op histogram
    use_roofline: bool    # include log_t_theoretical


VARIANTS = [
    Variant("A", "baseline (with model_name)",          use_id=True,  use_arch=False, use_roofline=False),
    Variant("B", "+arch, model_name kept",              use_id=True,  use_arch=True,  use_roofline=False),
    Variant("C", "−id +arch (no model_name/family)",    use_id=False, use_arch=True,  use_roofline=False),
    Variant("D", "C + roofline (log_t_theoretical)",    use_id=False, use_arch=True,  use_roofline=True),
]


def feature_set(df: pd.DataFrame, v: Variant) -> tuple[list[str], list[str]]:
    drop = {TARGET_RAW, TARGET_LOG}

    arch_op = [c for c in df.columns if c.startswith(ARCH_NUMERIC_PREFIXES)]
    arch_scalar = [c for c in ARCH_NUMERIC if c in df.columns]
    roofline_cols = [c for c in ROOFLINE_FEATS if c in df.columns]
    bw_cols = [c for c in BANDWIDTH_FEATS if c in df.columns]

    if not v.use_arch:
        drop.update(arch_op + arch_scalar + bw_cols)
    if not v.use_roofline:
        drop.update(roofline_cols)
    else:
        # always keep only log version, drop the scalar t_theoretical (avoid leak through max)
        for col in ("t_theoretical", "flops_actual", "activation_bytes_actual"):
            drop.add(col)

    if not v.use_id:
        drop.update({"model_name", "model_family"})

    feats = [c for c in df.columns if c not in drop]
    cat = [c for c in ("cpu_name", "gpu_name", "model_name", "model_family") if c in feats]
    return feats, cat


def _frame(X: pd.DataFrame, cat: list[str]) -> pd.DataFrame:
    out = X.copy()
    for c in cat:
        out[c] = out[c].astype("category")
    # CatBoost needs numerics filled; nan in cat is OK
    for c in out.columns:
        if c not in cat:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def cv_one(df: pd.DataFrame, v: Variant, scheme: str) -> dict:
    feats, cat = feature_set(df, v)
    X = df[feats]
    y = df[TARGET_LOG]
    y_real = df[TARGET_RAW].to_numpy()

    if scheme == "hardware":
        groups = (df["cpu_name"].astype(str) + " | " + df["gpu_name"].astype(str)).to_numpy()
    elif scheme == "family":
        groups = df["model_family"].astype(str).to_numpy()
    else:
        raise ValueError(scheme)

    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if scheme == "family":
        n_splits = n_groups  # leave-one-family-out (6 splits)

    Xt = _frame(X, cat)
    r2s, mapes, maes = [], [], []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(gkf.split(Xt, y, groups=groups), start=1):
        m = CatBoostRegressor(cat_features=cat, **MODEL_PARAMS)
        m.fit(Xt.iloc[tr], y.iloc[tr])
        pred_log = m.predict(Xt.iloc[va])
        # clip to a sane log range to avoid wild exp() on degenerate folds
        lo, hi = y.iloc[tr].min() - 2, y.iloc[tr].max() + 2
        pred_log = np.clip(pred_log, lo, hi)
        pred_sec = np.exp(pred_log)
        r2s.append(r2_score(y.iloc[va], pred_log))
        mapes.append(mean_absolute_percentage_error(y_real[va], pred_sec))
        maes.append(mean_absolute_error(y_real[va], pred_sec))
        LOG.info("    %s/%s fold %d: R²(log)=%.3f MAPE=%.1f%% MAE=%.3fs",
                 v.code, scheme, fold, r2s[-1], 100 * mapes[-1], maes[-1])

    return {
        "r2_log_mean": float(np.mean(r2s)),
        "r2_log_std": float(np.std(r2s)),
        "mape_mean": float(np.mean(mapes)),
        "mae_sec_mean": float(np.mean(maes)),
        "n_features": len(feats),
        "n_categorical": len(cat),
        "n_splits": n_splits,
    }


def render_md(results: list[dict], out: Path) -> None:
    lines = []
    lines.append("# Stage-2 ablation: do arch features replace `model_name`?")
    lines.append("")
    lines.append("Each cell shows mean across folds. **R²** is on log-time; "
                 "**MAPE** is in seconds-space.")
    lines.append("")
    lines.append("## CV by hardware (GroupKFold on cpu+gpu)")
    lines.append("")
    lines.append("| variant | description | n_feat | R² log | MAPE | MAE sec |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in results:
        if r["scheme"] != "hardware":
            continue
        lines.append(f"| **{r['code']}** | {r['label']} | {r['n_features']} | "
                     f"{r['r2_log_mean']:.3f} | {r['mape_mean']*100:.1f}% | {r['mae_sec_mean']:.3f} |")
    lines.append("")
    lines.append("## CV by model_family (leave-one-family-out)")
    lines.append("")
    lines.append("| variant | description | n_feat | R² log | MAPE | MAE sec |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for r in results:
        if r["scheme"] != "family":
            continue
        lines.append(f"| **{r['code']}** | {r['label']} | {r['n_features']} | "
                     f"{r['r2_log_mean']:.3f} | {r['mape_mean']*100:.1f}% | {r['mae_sec_mean']:.3f} |")
    lines.append("")

    a_h = next(r for r in results if r["code"] == "A" and r["scheme"] == "hardware")
    d_f = next(r for r in results if r["code"] == "D" and r["scheme"] == "family")
    delta = a_h["r2_log_mean"] - d_f["r2_log_mean"]

    lines.append("## Pass-gate")
    lines.append("")
    lines.append(f"- A under **hardware** CV: R² = {a_h['r2_log_mean']:.3f}")
    lines.append(f"- D under **family**   CV: R² = {d_f['r2_log_mean']:.3f}")
    lines.append(f"- Δ = {delta*100:.1f} pp")
    lines.append("")
    if delta <= 0.05:
        lines.append("**PASS** (Δ ≤ 5 pp): arch features successfully replace model identity. "
                     "Proceed to Stage 3 (ModelRunner abstraction).")
    elif delta <= 0.10:
        lines.append("**MARGINAL** (5 pp < Δ ≤ 10 pp): arch features capture most of the signal "
                     "but some structure is missing. Consider adding skip-connection counts, "
                     "receptive-field, or attention-FLOPs share before Stage 3.")
    else:
        lines.append(f"**FAIL** (Δ = {delta*100:.1f} pp > 5 pp): arch features do not yet replace "
                     "model identity. Stay on Stage 1: enrich the feature set "
                     "(skip connections, attention-FLOPs share, depthwise-FLOPs share, "
                     "ONNX-based op accounting) before re-running ablation.")
    lines.append("")
    lines.append("## Raw results")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    LOG.info("wrote %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    raw = pd.read_csv(args.data, low_memory=False)
    df = preprocess(raw)
    LOG.info("dataset: %d rows, %d cols", *df.shape)

    results = []
    for v in VARIANTS:
        for scheme in ("hardware", "family"):
            LOG.info("running variant %s (%s) under scheme %s …", v.code, v.label, scheme)
            r = cv_one(df, v, scheme)
            r.update({"code": v.code, "label": v.label, "scheme": scheme})
            results.append(r)
            LOG.info("  → R²=%.3f MAPE=%.1f%% MAE=%.3fs (feat=%d cat=%d splits=%d)",
                     r["r2_log_mean"], r["mape_mean"]*100, r["mae_sec_mean"],
                     r["n_features"], r["n_categorical"], r["n_splits"])

    render_md(results, args.out)


if __name__ == "__main__":
    main()
