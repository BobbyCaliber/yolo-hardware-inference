"""Inverse problem, throughput edition: cheapest hardware for a whole workload.

`recommend.py` sizes a single model against a latency target. This sizes a set
of models running **concurrently** (several processes on the same GPU) against a
shared **deadline** and **budget**, and tells you the cheapest pre-built PC — or,
with `--gpu-only`, the cheapest GPU — that finishes everything in time, spinning
up multiple identical GPUs when one can't keep up.

The throughput maths live in `workload.py` (additive GPU-seconds; see its
docstring). This file is the plumbing: parse the workload, walk the DNS-shop
catalogue, reuse `recommend.py`'s fuzzy CPU/GPU matchers and the CatBoost
predictor, sweep batch sizes per stream, and rank feasible configs by price.

Unknown models (no benchmark of their own) are first-class:
  * `"proxy": "yolov8l.pt"`      — predict with a similar registered model, or
  * `"fixed_latency_ms": 120`    — plug in a latency you measured yourself.

Usage:
    python src/recommend_workload.py --workload examples/workload_3min.json \
        --deadline 180 --max-budget 250000
    python src/recommend_workload.py --gpu-only --max-budget 250000 \
        --stream "model=yolov8n.pt,img=480,frames=3000" \
        --stream "proxy=yolov8l.pt,name=yolo26l,img=640,frames=48000" \
        --stream "name=RoMa,img=560,frames=500,fixed_latency_ms=120"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

import hardware_lookup as hw  # noqa: E402
import workload as wl  # noqa: E402
# Reuse the catalogue matchers, row builder, enricher and predictor loader —
# do not re-implement them here.
import recommend as rec  # noqa: E402

ARCH_CSV = PROJECT_ROOT / "data_base" / "model_arch_features.csv"
DEFAULT_DNS_JSON = rec.DEFAULT_DNS_JSON

LOG = logging.getLogger("recommend_workload")


# ------------------------------------------------------------------ prediction

def _predict_upper(model, meta: dict, feat_df: pd.DataFrame) -> np.ndarray:
    """Conservative per-batch latency (seconds): the ~84 % upper band when the
    model carries uncertainty, else the point estimate."""
    X = feat_df[meta["features"]].copy()
    for c in meta["categorical_features"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    if meta.get("model_params", {}).get("loss_function") == "RMSEWithUncertainty":
        out = model.virtual_ensembles_predict(
            X, prediction_type="TotalUncertainty", virtual_ensembles_count=10,
        )
        mu, sigma = out[:, 0], np.sqrt(out[:, 1] + out[:, 2])
        return np.exp(mu + sigma)
    preds = model.predict(X)
    if preds.ndim == 2:
        preds = preds[:, 0]
    return np.exp(preds)


# ------------------------------------------------------------------ platforms

def _matched_platforms(builds: list[dict], gpu_only: bool) -> dict[tuple, dict]:
    """Collapse DNS builds to unique hardware (throughput depends only on the
    platform). Returns key → {cpu, gpu, price, ram, url, dns_cpu, dns_gpu}.

    key is `gpu_canon` in `--gpu-only` mode (cheapest carrier build wins), else
    `(cpu_canon, gpu_canon)`. Cheapest build per key is kept for costing.
    """
    out: dict[tuple, dict] = {}
    n_skip = 0
    for b in builds:
        if not b.get("gpu_name") or b.get("price_rub") is None:
            n_skip += 1
            continue
        cpu_canon = rec.match_cpu(b["cpu_name"])
        gpu_canon = rec.match_gpu(b["gpu_name"])
        if cpu_canon is None or gpu_canon is None:
            n_skip += 1
            continue
        key = (gpu_canon,) if gpu_only else (cpu_canon, gpu_canon)
        cur = out.get(key)
        if cur is None or b["price_rub"] < cur["price"]:
            out[key] = {
                "cpu": cpu_canon, "gpu": gpu_canon, "price": b["price_rub"],
                "ram": b.get("ram_gb") or wl.DEFAULT_RAM_GB, "url": b["url"],
                "dns_cpu": b["cpu_name"], "dns_gpu": b["gpu_name"],
            }
    LOG.info("matched %d unique %s (skipped %d builds)",
             len(out), "GPUs" if gpu_only else "platforms", n_skip)
    return out


def _arch_rows(streams: list[wl.Stream]) -> dict[str, pd.Series]:
    """Look up arch features for every registered model/proxy used. Exits on an
    unknown model name."""
    arch = pd.read_csv(ARCH_CSV)
    known = set(arch["model_name"])
    rows: dict[str, pd.Series] = {}
    for s in streams:
        m = s.predict_model
        if m is None:
            continue
        if m not in known:
            sys.exit(f"stream {s.name!r}: model {m!r} is not registered. "
                     f"Use a registered model/proxy (see predict.py --list-models) "
                     f"or supply fixed_latency_ms.")
        rows[m] = arch[arch["model_name"] == m].iloc[0]
    return rows


# ------------------------------------------------------------------ core

def recommend_workload(streams: list[wl.Stream], deadline_s: float,
                       max_budget_rub: float, *,
                       util: float = wl.DEFAULT_UTIL,
                       max_gpus: int = wl.DEFAULT_MAX_GPUS,
                       ram_gb: int = wl.DEFAULT_RAM_GB,
                       gpu_only: bool = False,
                       dns_json: Path = DEFAULT_DNS_JSON,
                       weights_dir: Path | None = None,
                       top_k: int = 10) -> pd.DataFrame:
    if not dns_json.is_file():
        sys.exit(f"DNS catalogue missing: {dns_json} (run scripts/scrape_dns_pcs.py)")
    builds = json.loads(dns_json.read_text(encoding="utf-8"))["builds"]
    platforms = _matched_platforms(builds, gpu_only)
    if not platforms:
        return pd.DataFrame()

    arch_rows = _arch_rows(streams)
    pred_streams = [s for s in streams if s.needs_prediction]
    weights_dir = weights_dir or rec._default_weights_dir()
    model, meta = rec._load_predictor(weights_dir)
    LOG.info("predictor: %s", weights_dir.relative_to(PROJECT_ROOT))

    # --- one big feature frame over (platform × prediction-stream × batch) ----
    feat_rows: list[dict] = []
    index: list[tuple] = []          # (platform_key, stream_name, batch)
    gpu_mem: dict[tuple, float] = {}  # platform_key → VRAM GB
    for key, p in platforms.items():
        try:
            host = hw.resolve_host(p["cpu"], p["gpu"], ram_gb=ram_gb)
        except (KeyError, ValueError, TypeError):
            continue
        gpu_mem[key] = float(host.get("gpu_memory") or float("nan"))
        for s in pred_streams:
            arch_row = arch_rows[s.predict_model]
            for batch in s.candidate_batches():
                feat_rows.append(rec._build_row(p["cpu"], p["gpu"], ram_gb,
                                                s.predict_model, s.img_size,
                                                batch, arch_row))
                index.append((key, s.name, batch))

    latency_lookup: dict[tuple, float] = {}
    if feat_rows:
        feat_df = rec._enrich_frame(pd.DataFrame(feat_rows))
        t_hi = _predict_upper(model, meta, feat_df)
        for idx, t in zip(index, t_hi):
            latency_lookup[idx] = float(t)

    # --- evaluate each platform ----------------------------------------------
    candidates: list[dict] = []
    for key, p in platforms.items():
        if key not in gpu_mem:
            continue
        mem = gpu_mem[key]
        per_frame: dict[str, float | None] = {}
        chosen_batch: dict[str, int | None] = {}
        for s in streams:
            custom = s.custom_per_frame_latency()
            if custom is not None:
                per_frame[s.name], chosen_batch[s.name] = custom, None
                continue
            arch_row = arch_rows[s.predict_model]
            feasible = {
                b for b in s.candidate_batches()
                if wl.fits_vram(float(arch_row["params"]),
                                float(arch_row["peak_activation"]),
                                float(arch_row["img_size_ref"]),
                                s.img_size, b, mem)
            }
            lat_by_batch = {b: latency_lookup.get((key, s.name, b))
                            for b in s.candidate_batches()}
            best = wl.choose_best_batch(lat_by_batch, feasible)
            if best is None:
                per_frame[s.name], chosen_batch[s.name] = None, None
            else:
                chosen_batch[s.name], per_frame[s.name] = best

        result = wl.aggregate(streams, per_frame, chosen_batch,
                              deadline_s=deadline_s, util=util, max_gpus=max_gpus)
        if not result.feasible:
            continue  # needs more than --max-gpus identical cards
        total_price = result.gpus_needed * p["price"]
        candidates.append({
            "total_price_rub": total_price,
            "within_budget": total_price <= max_budget_rub,
            "gpus_needed": result.gpus_needed,
            "unit_price_rub": p["price"],
            "gpu": p["gpu"],
            "cpu": p["cpu"],
            "ram_gb": p["ram"],
            "gpu_seconds_needed": round(result.required_gpu_seconds, 1),
            "deadline_s": deadline_s,
            "headroom_pct": round(result.headroom_frac * 100, 1),
            "streams_summary": _summary(result),
            "url": p["url"],
            "_result": result,
        })

    if not candidates:
        return pd.DataFrame()
    # Full sorted feasible set (within `--max-gpus`); budget split is the CLI's
    # job so it can fall back to "nearest over budget" when nothing fits.
    return pd.DataFrame(candidates).sort_values(
        ["total_price_rub", "gpus_needed"]).reset_index(drop=True)


def _summary(result: wl.WorkloadResult) -> str:
    parts = []
    for s in result.streams:
        if not s.feasible:
            parts.append(f"{s.name}:UNSERVED")
            continue
        b = "custom" if s.is_custom else f"b{s.batch}"
        parts.append(f"{s.name}@{s.img_size} {b} {s.fps_per_gpu:.0f}fps")
    return " | ".join(parts)


# ------------------------------------------------------------------ CLI

def _load_streams(args: argparse.Namespace) -> tuple[float, list[wl.Stream]]:
    if args.workload:
        deadline, streams = wl.load_workload(args.workload)
    elif args.stream:
        streams = [wl.parse_stream_arg(s) for s in args.stream]
        deadline = wl.DEFAULT_DEADLINE_S
    else:
        sys.exit("provide --workload FILE or one or more --stream tokens")
    if args.deadline is not None:
        deadline = args.deadline
    return deadline, streams


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("workload (use --workload OR --stream)")
    src.add_argument("--workload", type=Path, help="workload JSON file")
    src.add_argument("--stream", action="append",
                     help="inline stream, e.g. 'model=yolov8n.pt,img=480,frames=3000' "
                          "(repeatable). Custom: 'name=RoMa,frames=500,fixed_latency_ms=120'")
    p.add_argument("--deadline", type=float, default=None,
                   help=f"seconds to finish the whole workload (default {wl.DEFAULT_DEADLINE_S:g}, "
                        "or deadline_s from the workload file)")
    p.add_argument("--max-budget", type=float, required=True, help="budget cap in ₽")
    p.add_argument("--max-gpus", type=int, default=wl.DEFAULT_MAX_GPUS,
                   help=f"most identical GPUs to shard across (default {wl.DEFAULT_MAX_GPUS})")
    p.add_argument("--util", type=float, default=wl.DEFAULT_UTIL,
                   help=f"GPU duty-cycle efficiency (default {wl.DEFAULT_UTIL})")
    p.add_argument("--ram", type=int, default=wl.DEFAULT_RAM_GB,
                   help=f"host RAM GB for prediction (default {wl.DEFAULT_RAM_GB})")
    p.add_argument("--gpu-only", action="store_true",
                   help="rank GPUs (cheapest carrier build as price proxy), not full builds")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--dns-json", type=Path, default=DEFAULT_DNS_JSON)
    p.add_argument("--weights-dir", type=Path, default=None)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=args.log_level, format="[%(levelname)s] %(message)s")
    deadline, streams = _load_streams(args)
    LOG.info("workload: %d streams, deadline %.0fs, budget ₽%.0f, max_gpus %d, %s",
             len(streams), deadline, args.max_budget, args.max_gpus,
             "GPU-only" if args.gpu_only else "full PC builds")
    total_frames = sum(s.frames for s in streams)
    LOG.info("total frames across streams: %d", total_frames)

    df = recommend_workload(
        streams, deadline_s=deadline, max_budget_rub=args.max_budget,
        util=args.util, max_gpus=args.max_gpus, ram_gb=args.ram,
        gpu_only=args.gpu_only, dns_json=args.dns_json,
        weights_dir=args.weights_dir, top_k=args.top_k,
    )
    if df.empty:
        print("\nNo configuration can finish this workload within the deadline "
              f"using ≤{args.max_gpus} identical GPUs. Raise --max-gpus, relax "
              "--deadline, lower resolutions/frame counts, or proxy the heavy "
              "stream with a lighter model.")
        return

    noun = "GPUs" if args.gpu_only else "builds"
    in_budget = df[df["within_budget"]].head(args.top_k)
    if not in_budget.empty:
        _print_table(in_budget, noun, deadline, within_budget=True)
        _print_breakdown(in_budget.iloc[0], deadline)
        return

    # Nothing fits the budget — show the cheapest *achievable* configs instead,
    # so the user sees what the workload actually costs and why.
    nearest = df.head(args.top_k)
    cheapest = nearest.iloc[0]
    print(f"\nNothing fits ₽{args.max_budget:,.0f}. The workload needs at least "
          f"₽{cheapest['total_price_rub']:,.0f} "
          f"({cheapest['gpus_needed']}× {cheapest['gpu']}). Nearest achievable "
          f"{noun} (over budget):\n")
    _print_table(nearest, noun, deadline, within_budget=False)
    _print_breakdown(cheapest, deadline)
    print("\nTo fit the budget: relax --deadline, cut frame counts/resolution, "
          "proxy the bottleneck stream with a lighter model, or raise --max-gpus "
          "if a single cheaper GPU type can be scaled out further.")


def _print_table(df: pd.DataFrame, noun: str, deadline: float, within_budget: bool) -> None:
    show = df.drop(columns=["_result", "within_budget"])
    pd.options.display.max_colwidth = 80
    pd.options.display.width = 240
    head = "Cheapest" if within_budget else "Closest"
    print(f"\n{head} {noun} that finish the workload within {deadline:.0f}s "
          f"(conservative ~84% latency band):\n")
    print(show.to_string(index=False))


def _print_breakdown(row: pd.Series, deadline: float) -> None:
    best = row["_result"]
    print(f"\nPer-stream breakdown for the top pick "
          f"({best.gpus_needed}× {row['gpu']}):")
    for s in best.streams:
        if not s.feasible:
            print(f"  - {s.name}: UNSERVED (no batch fit VRAM / unknown model)")
            continue
        kind = "custom latency" if s.is_custom else f"batch {s.batch}"
        print(f"  - {s.name:<10} {s.img_size}px  {kind:<14}  "
              f"{s.per_frame_latency_s*1000:7.2f} ms/frame  "
              f"{s.fps_per_gpu:7.1f} fps/GPU  needs {s.required_fps:7.1f} fps  "
              f"({s.gpu_seconds:6.1f} GPU-s for {s.frames} frames)")
    print(f"\n  required {best.required_gpu_seconds:.1f} GPU-s vs "
          f"{best.gpus_needed}×{deadline:.0f}s = {best.gpus_needed*deadline:.0f}s capacity "
          f"→ {best.headroom_frac*100:.1f}% headroom  (bottleneck: {best.bottleneck_stream})")


if __name__ == "__main__":
    main()
