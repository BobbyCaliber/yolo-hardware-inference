"""Generic model-inference benchmark — sweeps any registered model family.

Replaces check_yolo_predict.py for non-YOLO families (the YOLO container
is kept for backward compatibility with the existing data_base.csv).

Configured entirely through environment variables so the run_benchmark.py
orchestrator doesn't need per-runner code:

    RUNNER_FAMILY  : family code from runners.list_families() (e.g. "rtdetr").
                     If set, MODELS defaults to runners.list_models(family).
    MODELS         : comma-separated model names. Overrides RUNNER_FAMILY.
    IMG_SIZES      : comma-separated input sides (default: spec defaults).
    BATCHES        : comma-separated batch sizes (default: spec defaults).
    DEVICES        : comma-separated devices (default: cpu,cuda when available).
    DATA_DIR       : output directory (default /app/data).
    IMAGES_DIR     : path to images for runners that take file paths (YOLO).
    OUT_NAME       : output csv filename (default: model_predict.csv).

The CSV schema matches yolo_predict.csv plus a `model_family` column so the
merged dataset has a uniform structure across families.
"""

from __future__ import annotations

import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

# runners package lives at /app/runners inside the container
sys.path.insert(0, "/app")
import runners  # noqa: E402

import cpuinfo  # noqa: E402
import GPUtil  # noqa: E402


def _env_list(name: str, default: list | None = None, cast=str) -> list | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [cast(x.strip()) for x in raw.split(",") if x.strip()]


def _resolve_models() -> list[str]:
    runners.load_all()
    explicit = _env_list("MODELS")
    if explicit:
        return explicit
    family = os.environ.get("RUNNER_FAMILY")
    if family:
        return runners.list_models(family)
    raise SystemExit("set RUNNER_FAMILY=<family> or MODELS=name1,name2,...")


def _devices_default() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


def _gather_inputs(runner: runners.ModelRunner, img_size: int, batch: int,
                   images_dir: Path) -> object:
    """Build inputs in the format the runner expects."""
    cls = runner.__class__.__name__
    if cls in ("YOLORunner", "RTDETRRunner"):
        # ultralytics-based: file paths
        files = sorted(p for p in images_dir.iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(files) < batch:
            raise SystemExit(f"need {batch} images in {images_dir}, found {len(files)}")
        return [str(p) for p in random.sample(files, batch)]
    # tensor-based runners
    return torch.zeros(batch, 3, img_size, img_size)


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    images_dir = Path(os.environ.get("IMAGES_DIR", "/app/data/images"))
    data_dir = Path(os.environ.get("DATA_DIR", "/app/data"))
    weights_dir = Path(os.environ.get("WEIGHTS_DIR", "/app/weights"))
    out_name = os.environ.get("OUT_NAME", "model_predict.csv")
    data_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    devices = _env_list("DEVICES") or _devices_default()
    img_sizes_override = _env_list("IMG_SIZES", cast=int)
    batches_override = _env_list("BATCHES", cast=int)

    targets = _resolve_models()

    # host descriptors — same shape as the YOLO container so merge_results.py
    # handles the new CSV identically.
    ram = int(np.round(psutil.virtual_memory().total / (1024.0 ** 3)))
    cpu_name = cpuinfo.get_cpu_info().get("brand_raw", "Unknown")
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu_name = gpus[0].name
        gpu_mem = gpus[0].memoryTotal / 1024
    else:
        gpu_name = "none"
        gpu_mem = 0

    out_path = data_dir / out_name
    fieldnames = [
        "cpu_name", "gpu_name", "gpu_memory", "used_gpu", "ram_memory",
        "model_name", "model_family",
        "model_param_dict",  # kept blank — superseded by arch features
        "num_all_params",
        "model_img_size", "image_batch_size", "predicting_times",
    ]

    t_start = time.monotonic()
    n_rows = 0
    with open(out_path, "w", newline="", buffering=1) as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)

        for name in targets:
            try:
                spec = runners.get(name)
            except KeyError as e:
                print(f"[WARN] {e}", file=sys.stderr)
                continue
            spec.load_kwargs.setdefault("weights_dir", str(weights_dir))
            img_sizes = img_sizes_override or list(spec.default_img_sizes)
            batches = batches_override or list(spec.default_batches)

            for device in devices:
                used_gpu = 1 if device == "cuda" else 0
                runner = runners.make_runner(name)
                try:
                    runner.load(device=device)
                except Exception as exc:
                    print(f"[ERROR] load failed for {name} on {device}: {exc}", file=sys.stderr)
                    continue
                num_params = _count_params(runner.get_module())

                for batch in batches:
                    for img_size in img_sizes:
                        try:
                            inputs = _gather_inputs(runner, img_size, batch, images_dir)
                        except SystemExit as exc:
                            print(f"[ERROR] {exc}", file=sys.stderr)
                            continue
                        try:
                            runner.prepare(img_size, batch)
                        except Exception as exc:
                            print(f"[ERROR] prepare failed {name}/{device}/{img_size}/{batch}: {exc}",
                                  file=sys.stderr)
                            continue
                        t0 = time.time()
                        try:
                            runner.infer(inputs, img_size=img_size)
                        except TypeError:
                            runner.infer(inputs)
                        except Exception as exc:
                            print(f"[ERROR] infer failed {name}/{device}/{img_size}/{batch}: {exc}",
                                  file=sys.stderr)
                            continue
                        dt = time.time() - t0
                        writer.writerow([
                            cpu_name, gpu_name, gpu_mem, bool(used_gpu), ram,
                            name, spec.family, "", num_params,
                            img_size, batch, dt,
                        ])
                        n_rows += 1
                        if device == "cuda":
                            torch.cuda.empty_cache()
                del runner
    elapsed = time.monotonic() - t_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"wrote {out_path}: {n_rows} rows in {h:02d}:{m:02d}:{s:02d} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
