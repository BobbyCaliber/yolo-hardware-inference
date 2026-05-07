"""Enrich data_base.csv with architecture features + roofline `t_theoretical`.

Pipeline:
  data_base/data_base.csv  (already has bandwidth cols from GPU/CPU containers)
    + data_base/model_arch_features.csv          (all profiled models, by model_name)
    -> data_base/data_base_enriched.csv

The actual math lives in `src/enrich_helpers.enrich()`.

Usage:
    python scripts/enrich_dataset.py
    python scripts/enrich_dataset.py --in path/to.csv --out path/to_out.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("enrich_dataset")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import enrich_helpers  # noqa: E402

DEFAULT_IN = PROJECT_ROOT / "data_base" / "data_base.csv"
DEFAULT_ARCH = PROJECT_ROOT / "data_base" / "model_arch_features.csv"
DEFAULT_OUT = PROJECT_ROOT / "data_base" / "data_base_enriched.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--arch", type=Path, default=DEFAULT_ARCH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")

    base = pd.read_csv(args.inp, low_memory=False)
    LOG.info("loaded base=%d rows", len(base))

    df = enrich_helpers.enrich(base, arch_csv=args.arch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    LOG.info("wrote %d rows × %d cols → %s", len(df), df.shape[1], args.out)

    if "predicting_times" in df.columns and "t_theoretical" in df.columns:
        ratio = df["predicting_times"] / df["t_theoretical"]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        if len(ratio):
            LOG.info("measured / t_theoretical:  median=%.2f  p10=%.2f  p90=%.2f",
                     ratio.median(), ratio.quantile(0.1), ratio.quantile(0.9))


if __name__ == "__main__":
    main()
