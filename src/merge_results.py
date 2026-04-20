"""Merge per-run benchmark CSVs into a single row-per-measurement dataset.

`make run` produces three files in ./tmp/data/:
    - cpu_config.csv    (1 row, host CPU descriptor)
    - gpu_config.csv    (1 row, host GPU descriptor)
    - yolo_predict.csv  (many rows, one per inference run)

This script cross-joins them so every inference row is tagged with the host
hardware descriptors, producing the same 35-column schema as
`data_base/data_base.csv`. Each invocation produces one timestamped file in
./data_new/ — one run, one file, no accumulation.

Usage:
    python src/merge_results.py            # uses ./tmp/data → ./data_new
    python src/merge_results.py --data-dir path --out-dir other_path
    python src/merge_results.py --tag a100 # adds _a100 suffix to the filename
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
REFERENCE_SCHEMA = PROJECT_ROOT / "data_base" / "data_base.csv"


def _read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"missing input: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise SystemExit(f"empty input: {path}")
    return df


def _reference_columns() -> list[str] | None:
    if not REFERENCE_SCHEMA.is_file():
        return None
    return list(pd.read_csv(REFERENCE_SCHEMA, nrows=0).columns)


def merge(data_dir: Path) -> pd.DataFrame:
    yolo = _read_required(data_dir / "yolo_predict.csv")
    cpu = _read_required(data_dir / "cpu_config.csv")
    gpu = _read_required(data_dir / "gpu_config.csv")

    if len(cpu) != 1:
        print(f"warn: cpu_config.csv has {len(cpu)} rows, expected 1; using first", file=sys.stderr)
        cpu = cpu.head(1)
    if len(gpu) != 1:
        print(f"warn: gpu_config.csv has {len(gpu)} rows, expected 1; using first", file=sys.stderr)
        gpu = gpu.head(1)

    merged = yolo.join(cpu, how="cross").join(gpu, how="cross")

    ref_cols = _reference_columns()
    if ref_cols is not None:
        missing = [c for c in ref_cols if c not in merged.columns]
        extra = [c for c in merged.columns if c not in ref_cols]
        if missing:
            print(f"warn: columns missing vs reference: {missing}", file=sys.stderr)
        if extra:
            print(f"warn: extra columns not in reference: {extra}", file=sys.stderr)
        # reorder to reference layout, keeping any extras at the tail
        ordered = [c for c in ref_cols if c in merged.columns] + extra
        merged = merged[ordered]

    return merged


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "tmp" / "data",
                   help="directory containing cpu_config.csv, gpu_config.csv, yolo_predict.csv")
    p.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data_new",
                   help="destination directory for merged CSV(s)")
    p.add_argument("--tag", default=None,
                   help="optional tag included in the output filename (e.g. 'a100')")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge(args.data_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.tag}" if args.tag else ""
    out_path = args.out_dir / f"merged_{stamp}{suffix}.csv"
    merged.to_csv(out_path, index=False)
    print(f"wrote {len(merged)} rows → {out_path}")


if __name__ == "__main__":
    main()
