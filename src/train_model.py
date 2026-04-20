"""Train a CatBoost regressor on the combined benchmark dataset.

Pulls rows from:
    - data_base/data_base.csv (reference, read-only — never rewritten)
    - data_new/*.csv          (merged per-run datasets from merge_results.py)

Applies the preprocessing pipeline established in
`research/predict_yolo_analysis.ipynb` (drop uninformative columns, coerce
Unknown caches to NaN, derive `pixels`/`work_estimate`/`compute_throughput`,
log-transform the target, extract `model_family`), reports a GroupKFold CV
score, trains CatBoost on the full data, and saves the model + metadata to
`data_new/reg_weights_new/`.

CatBoost is chosen because the notebook analysis showed it tied for the best
R² and works natively with categorical features — the model ships as one
`.cbm` file and can be loaded with `CatBoostRegressor().load_model(path)`.
Predictions are in log-space; apply `np.exp()` to recover seconds.

Usage:
    python src/train_model.py
    python src/train_model.py --skip-cv                 # just fit + save
    python src/train_model.py --out-dir /other/weights  # override output
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

LOG = logging.getLogger("train_model")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DEFAULT_REF = PROJECT_ROOT / "data_base" / "data_base.csv"
DEFAULT_NEW_DIR = PROJECT_ROOT / "data_new"
DEFAULT_OUT = DEFAULT_NEW_DIR / "reg_weights_new"

CATEGORICAL = ["cpu_name", "gpu_name", "model_name", "model_family"]
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
MODEL_PARAMS = dict(iterations=600, depth=8, learning_rate=0.05, random_seed=42, verbose=0)


@dataclass
class TrainReport:
    rows_total: int
    cv: dict | None


def load_all_csvs(ref_path: Path, new_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    if ref_path.is_file():
        frames.append(pd.read_csv(ref_path))
        sources.append(f"{ref_path.name} ({len(frames[-1])} rows)")
    else:
        LOG.warning("reference dataset missing: %s", ref_path)

    if new_dir.is_dir():
        for csv in sorted(new_dir.glob("*.csv")):
            frames.append(pd.read_csv(csv))
            sources.append(f"{csv.name} ({len(frames[-1])} rows)")

    if not frames:
        raise SystemExit(f"no CSVs found under {ref_path.parent} or {new_dir}")

    LOG.info("loaded sources: %s", "; ".join(sources))
    return pd.concat(frames, ignore_index=True)


def preprocess(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=[c for c in DROP_COLS if c in raw.columns]).copy()
    for c in CACHE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df['Compute Capability (cuda version)'] = df['Compute Capability (cuda version)'].str.replace(',', '.').str[1:-1].str.replace(' ', '').astype(float)
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


def _tree_frame(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out


def cv_score(df: pd.DataFrame) -> dict | None:
    features = [c for c in df.columns if c not in (TARGET_RAW, TARGET_LOG)]
    X = df[features]
    y = df[TARGET_LOG]
    groups = (df["cpu_name"].astype(str) + " | " + df["gpu_name"].astype(str)).values

    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    if n_splits < 2:
        LOG.warning("only %d unique (cpu,gpu) groups — skipping CV", n_groups)
        return None

    X_tree = _tree_frame(X)
    r2_log, mae_sec = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, va) in enumerate(gkf.split(X_tree, y, groups=groups), start=1):
        m = CatBoostRegressor(cat_features=CATEGORICAL, **MODEL_PARAMS)
        m.fit(X_tree.iloc[tr], y.iloc[tr])
        pred = m.predict(X_tree.iloc[va])
        r2_log.append(r2_score(y.iloc[va], pred))
        lo, hi = y.iloc[tr].min() - 2, y.iloc[tr].max() + 2
        pred_c = np.clip(pred, lo, hi)
        mae_sec.append(mean_absolute_error(np.exp(y.iloc[va]), np.exp(pred_c)))
        LOG.info("  fold %d: R²=%.3f | MAE=%.3fs", fold, r2_log[-1], mae_sec[-1])

    return {
        "folds": n_splits,
        "n_groups": n_groups,
        "cv_r2_log": float(np.mean(r2_log)),
        "cv_mae_sec": float(np.mean(mae_sec)),
    }


def fit_final(df: pd.DataFrame) -> tuple[CatBoostRegressor, list[str]]:
    features = [c for c in df.columns if c not in (TARGET_RAW, TARGET_LOG)]
    X_tree = _tree_frame(df[features])
    y = df[TARGET_LOG]
    model = CatBoostRegressor(cat_features=CATEGORICAL, **MODEL_PARAMS)
    model.fit(X_tree, y)
    return model, features


def save(model: CatBoostRegressor, features: list[str], report: TrainReport,
         out_dir: Path, ref_csv: Path, new_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "catboost_model.cbm"
    model.save_model(str(model_path))

    meta = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {"reference": str(ref_csv), "new_dir": str(new_dir)},
        "rows_trained": report.rows_total,
        "features": features,
        "categorical_features": CATEGORICAL,
        "target_log": TARGET_LOG,
        "target_inverse_transform": "np.exp(prediction)",
        "model_params": MODEL_PARAMS,
        "cv": report.cv,
    }
    meta_path = out_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    LOG.info("saved model:    %s (%.1f KB)", model_path, model_path.stat().st_size / 1024)
    LOG.info("saved metadata: %s", meta_path)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref-csv", type=Path, default=DEFAULT_REF,
                   help="reference dataset (default: data_base/data_base.csv)")
    p.add_argument("--new-dir", type=Path, default=DEFAULT_NEW_DIR,
                   help="directory with merged per-run CSVs (default: data_new/)")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                   help="destination for model + metadata (default: data_new/reg_weights_new/)")
    p.add_argument("--skip-cv", action="store_true", help="skip cross-validation report")
    p.add_argument("--log-level", default="INFO", help="logging level (DEBUG, INFO, WARNING)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    raw = load_all_csvs(args.ref_csv, args.new_dir)
    LOG.info("combined raw dataset: %d rows", len(raw))

    df = preprocess(raw)

    cv = None
    if not args.skip_cv:
        LOG.info("running GroupKFold CV (groups = cpu_name × gpu_name) …")
        cv = cv_score(df)
        if cv:
            LOG.info("CV summary: R²=%.3f | MAE=%.3fs across %d folds",
                     cv["cv_r2_log"], cv["cv_mae_sec"], cv["folds"])

    LOG.info("fitting CatBoost on all %d rows …", len(df))
    model, features = fit_final(df)

    report = TrainReport(rows_total=len(df), cv=cv)
    save(model, features, report, args.out_dir, args.ref_csv, args.new_dir)


if __name__ == "__main__":
    main()
