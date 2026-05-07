"""Shared enrichment logic: arch features + roofline `t_theoretical`.

Used by both `scripts/enrich_dataset.py` (whole baseline) and any future
per-platform pipeline. Bandwidth/launch-overhead columns are expected to
already live in the input dataframe — they are written by the GPU/CPU
benchmark containers, so there's no separate lookup CSV.

Math:
  flops_actual = flops_ref × (img_size/img_ref)^2 × batch
  bytes_actual = activation_bytes_ref × (img_size/img_ref)^2 × batch
                 + params × bytes_per_elem
  t_theoretical = max(flops_actual / peak_flops_device,
                      bytes_actual / peak_bw_device)
                  + launch_overhead × num_ops
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BYTES_PER_ELEM = 4  # FP32 inference

REQUIRED_BW_COLS = (
    "cpu_peak_bw_gbps", "gpu_peak_bw_gbps",
    "launch_overhead_us_cpu", "launch_overhead_us_gpu",
)


def _cpu_peak_flops_tflops(df: pd.DataFrame) -> pd.Series:
    """Coarse CPU peak FLOPS in TFLOPS (AVX2 FP32: 8-wide × 2-FMA)."""
    freq_ghz = df["Frequency (MHz)"] / 1000.0
    return df["Cores"] * freq_ghz * 8 * 2 / 1000.0


def enrich(base: pd.DataFrame, *, arch_csv: Path) -> pd.DataFrame:
    """Join `base` with arch features and add roofline columns. Returns a copy.

    `base` must contain bandwidth/launch-overhead columns (written by the
    benchmark containers — see check_gpu_config.py / check_cpu_config.py).
    """
    arch = pd.read_csv(arch_csv).rename(columns={
        "flops": "flops_ref",
        "macs": "macs_ref",
        "params": "params_ref",
        "activation_bytes": "activation_bytes_ref",
        "peak_activation": "peak_activation_ref",
    })
    arch_for_join = arch.drop(columns=["model_family"]) if "model_family" in arch.columns else arch
    df = base.merge(arch_for_join, on="model_name", how="left")

    if "model_family" not in df.columns:
        df["model_family"] = df["model_name"].map(arch.set_index("model_name")["model_family"])
    else:
        df["model_family"] = df["model_family"].fillna(
            df["model_name"].map(arch.set_index("model_name")["model_family"])
        )

    missing_bw = [c for c in REQUIRED_BW_COLS if c not in df.columns]
    if missing_bw:
        raise ValueError(
            f"input is missing bandwidth columns {missing_bw}. "
            "Re-run `make build` (the GPU/CPU containers now write bandwidth) "
            "or backfill them manually."
        )

    img_ref = df["img_size_ref"]
    scale = (df["model_img_size"] / img_ref) ** 2 * df["image_batch_size"]
    df["flops_actual"] = df["flops_ref"] * scale
    df["activation_bytes_actual"] = (
        df["activation_bytes_ref"] * scale + df["params_ref"] * BYTES_PER_ELEM
    )

    # Structural shares: distribute total ref FLOPs / op-count across operator
    # bins so the regressor sees architecture composition (transformer-heavy,
    # depthwise-heavy, conv1x1-heavy …) independently of total work.
    flops_total = df["flops_ref"].replace(0, np.nan)
    ops_total = df["num_ops"].replace(0, np.nan)
    for bin_ in ("attention", "depthwise_conv", "conv1x1", "conv3x3", "linear"):
        col = f"op_{bin_}_flops"
        if col in df.columns:
            df[f"{bin_}_flops_ratio"] = (df[col] / flops_total).fillna(0.0)
    for bin_ in ("attention", "norm_ln", "norm_bn", "depthwise_conv"):
        col = f"op_{bin_}_n"
        if col in df.columns:
            df[f"{bin_}_op_ratio"] = (df[col] / ops_total).fillna(0.0)

    used_gpu = df["used_gpu"].astype(bool)
    cpu_peak_tflops = _cpu_peak_flops_tflops(df)
    peak_flops = np.where(used_gpu, df["FP32 FLOPS (TFLOPS)"], cpu_peak_tflops) * 1e12
    peak_bw = np.where(used_gpu, df["gpu_peak_bw_gbps"], df["cpu_peak_bw_gbps"]) * 1e9
    launch_s = np.where(used_gpu, df["launch_overhead_us_gpu"], df["launch_overhead_us_cpu"]) * 1e-6

    t_compute = df["flops_actual"] / peak_flops
    t_memory = df["activation_bytes_actual"] / peak_bw
    df["t_theoretical"] = np.maximum(t_compute, t_memory) + launch_s * df["num_ops"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_t_theoretical"] = np.log(df["t_theoretical"])
    return df
