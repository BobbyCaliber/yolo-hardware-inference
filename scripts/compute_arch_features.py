"""Extract architecture-only features for any model registered in `runners`.

For each model we run a single forward pass at a reference input shape
(default 640x640, batch=1) with hooks attached to every leaf module, and
accumulate:

  - scalar features:  flops, macs, params, activation_bytes,
                      peak_activation, num_ops, depth, arithmetic_intensity
  - operator histogram (count, flops, activation_bytes per type):
        conv3x3, conv1x1, depthwise_conv, linear, attention,
        pool, upsample, norm, activation_silu, activation_relu, activation_gelu

Output: data_base/model_arch_features.csv (one row per model_name) plus a
column `model_family` so downstream merge can group across families.

These features replace the per-model identity (`model_name`, `model_family`)
in the regressor — they describe the architecture in a way that generalises
to non-YOLO families. FLOPs/activations at other (img_size, batch) values
are recovered downstream as ref_value * (img_size/640)^2 * batch.

Usage:
    python scripts/compute_arch_features.py                       # all registered
    python scripts/compute_arch_features.py --family yolov8       # one family
    python scripts/compute_arch_features.py --models yolov8n.pt rtdetr-l.pt
    python scripts/compute_arch_features.py --append              # merge into existing CSV
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from thop import profile

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import runners  # noqa: E402

LOG = logging.getLogger("compute_arch_features")

DEFAULT_OUT = PROJECT_ROOT / "data_base" / "model_arch_features.csv"
WEIGHTS_CACHE = PROJECT_ROOT / "tmp" / "weights"

# operator histogram bins
OP_BINS = [
    "conv3x3", "conv1x1", "depthwise_conv",
    "linear", "attention",
    "pool", "upsample",
    "norm_bn", "norm_ln",
    "activation_silu", "activation_relu", "activation_gelu",
]

# bytes per element (FP32 inference for everything in the existing dataset)
BYTES_PER_ELEM = 4


_ATTN_PATH_TOKENS = ("attn", "attention", "mhsa", "self_attn", "cross_attn")


def _is_in_attention(qual_name: str) -> bool:
    # timm / HF transformers don't use nn.MultiheadAttention — their custom
    # Attention modules contain plain nn.Linear children whose qualified path
    # passes through ".attn." / ".attention." / similar. Reclassify those as
    # attention so the histogram reflects transformer FLOPs correctly.
    return any(tok in qual_name.split(".") for tok in _ATTN_PATH_TOKENS)


def _classify_module(m: nn.Module, qual_name: str = "") -> str | None:
    """Return histogram bin name for a leaf module, or None if it's not counted."""
    if isinstance(m, nn.MultiheadAttention):
        return "attention"
    if isinstance(m, nn.Linear):
        return "attention" if _is_in_attention(qual_name) else "linear"
    if isinstance(m, nn.Conv2d):
        if m.groups == m.in_channels and m.in_channels > 1:
            return "depthwise_conv"
        k = m.kernel_size
        if k == (1, 1):
            return "conv1x1"
        if k == (3, 3):
            return "conv3x3"
        return "conv3x3"  # bin larger kernels with conv3x3 as a coarse proxy
    if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
        return "pool"
    if isinstance(m, nn.Upsample):
        return "upsample"
    if isinstance(m, (nn.LayerNorm,)):
        return "norm_ln"
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm2d)):
        return "norm_bn"
    if isinstance(m, nn.SiLU):
        return "activation_silu"
    if isinstance(m, nn.ReLU):
        return "activation_relu"
    if isinstance(m, nn.GELU):
        return "activation_gelu"
    return None


def _module_flops(m: nn.Module, output_shape: tuple[int, ...]) -> float:
    """Coarse per-module FLOP estimate for histogram aggregation.

    Honest enough for the bins we care about (Conv, Linear). Activations,
    norms and pools are essentially free vs. conv FLOPs so we return 0 — the
    operator-count is what matters for those bins.
    """
    if isinstance(m, nn.Conv2d):
        # 2 * Cin * Kh * Kw / groups * H_out * W_out * Cout
        if len(output_shape) != 4:
            return 0.0
        b, cout, h_out, w_out = output_shape
        kh, kw = m.kernel_size
        return 2.0 * b * cout * (m.in_channels // m.groups) * kh * kw * h_out * w_out
    if isinstance(m, nn.Linear):
        # 2 * in_features * out_features * batch
        if not output_shape:
            return 0.0
        prefix = 1
        for d in output_shape[:-1]:
            prefix *= d
        return 2.0 * prefix * m.in_features * m.out_features
    return 0.0


def _output_size_bytes(out: object) -> int:
    if isinstance(out, torch.Tensor):
        return out.numel() * BYTES_PER_ELEM
    if isinstance(out, (list, tuple)):
        return sum(_output_size_bytes(o) for o in out)
    if isinstance(out, dict):
        return sum(_output_size_bytes(o) for o in out.values())
    return 0


def profile_model(model: nn.Module, img_size: int) -> dict:
    """Run one forward pass with hooks; return arch-feature dict."""
    model = model.eval()
    dummy = torch.zeros(1, 3, img_size, img_size)

    histogram = {b: {"n": 0, "flops": 0.0, "act_bytes": 0} for b in OP_BINS}
    totals = {"num_ops": 0, "activation_bytes": 0, "peak_activation": 0}
    handles = []

    def make_hook(name: str, m: nn.Module):
        bin_name = _classify_module(m, name)

        def hook(_module, _inp, out):
            out_bytes = _output_size_bytes(out)
            totals["num_ops"] += 1
            totals["activation_bytes"] += out_bytes
            if out_bytes > totals["peak_activation"]:
                totals["peak_activation"] = out_bytes
            if bin_name is None:
                return
            out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else ()
            flops = _module_flops(m, out_shape)
            histogram[bin_name]["n"] += 1
            histogram[bin_name]["flops"] += flops
            histogram[bin_name]["act_bytes"] += out_bytes

        return hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:  # leaf only
            handles.append(m.register_forward_hook(make_hook(name, m)))

    try:
        with torch.no_grad():
            model(dummy)
    finally:
        for h in handles:
            h.remove()

    # thop totals (independent, more rigorous accounting of FLOPs/MACs/params)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    flops_total = 2.0 * macs

    activation_bytes = totals["activation_bytes"]
    arith_intensity = flops_total / activation_bytes if activation_bytes else 0.0

    out = {
        "img_size_ref": img_size,
        "flops": flops_total,
        "macs": macs,
        "params": params,
        "activation_bytes": activation_bytes,
        "peak_activation": totals["peak_activation"],
        "num_ops": totals["num_ops"],
        "depth": totals["num_ops"],  # leaf-module count is a depth proxy
        "arithmetic_intensity": arith_intensity,
    }
    for b in OP_BINS:
        out[f"op_{b}_n"] = histogram[b]["n"]
        out[f"op_{b}_flops"] = histogram[b]["flops"]
        out[f"op_{b}_act"] = histogram[b]["act_bytes"]
    return out


def load_module(name: str) -> tuple[nn.Module, str]:
    """Load model via the runner registry. Returns (module, family)."""
    spec = runners.get(name)
    # Pass weights_dir hint so ultralytics-based runners cache properly.
    spec.load_kwargs.setdefault("weights_dir", str(WEIGHTS_CACHE))
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    runner = runners.make_runner(name)
    runner.load(device="cpu")
    return runner.get_module(), spec.family


def _resolve_models(args: argparse.Namespace) -> list[str]:
    if args.models:
        return list(args.models)
    if args.family:
        return runners.list_models(args.family)
    return runners.list_models()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--img-size", type=int, default=None,
                   help="override per-model arch_ref_img_size (otherwise uses spec.arch_ref_img_size)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="destination CSV")
    p.add_argument("--models", nargs="*", default=None, help="subset of model names")
    p.add_argument("--family", default=None, help="profile every model in one family")
    p.add_argument("--append", action="store_true",
                   help="merge into existing --out file (replaces rows for the same model_name)")
    p.add_argument("--list", action="store_true", help="list registered models and exit")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="[%(levelname)s] %(message)s")
    runners.load_all()

    if args.list:
        for fam in runners.list_families():
            print(f"{fam}:")
            for n in runners.list_models(fam):
                print(f"  {n}")
        return

    targets = _resolve_models(args)
    if not targets:
        # Silently no-op so `make collect` can iterate FAMILIES even when
        # some families' deps (timm / transformers) aren't installed on the
        # host. The user runs `make arch-features-docker` to fill those in.
        scope = f"family={args.family}" if args.family else "all"
        LOG.warning("no registered models match %s — skipping (install timm/"
                    "transformers locally or run `make arch-features-docker`)", scope)
        return
    LOG.info("profiling %d models", len(targets))

    rows = []
    for name in targets:
        # Per-model reference size — each spec carries its own canonical input
        # for FLOPs/activation profiling (224 for classification, 640 for
        # detection/seg, 512 for segformer). --img-size on the CLI overrides
        # this for ad-hoc runs.
        try:
            spec = runners.get(name)
            img_size = args.img_size if args.img_size is not None else spec.arch_ref_img_size
        except KeyError:
            img_size = args.img_size or 640
        LOG.info("profiling %s @ %dpx …", name, img_size)
        try:
            mod, family = load_module(name)
            feats = profile_model(mod, img_size)
        except Exception as exc:
            LOG.error("  FAILED %s: %s", name, exc)
            continue
        feats["model_name"] = name
        feats["model_family"] = family
        rows.append(feats)
        LOG.info("  flops=%.2fG  params=%.2fM  acts=%.1fMB  num_ops=%d",
                 feats["flops"] / 1e9, feats["params"] / 1e6,
                 feats["activation_bytes"] / 1e6, feats["num_ops"])

    if not rows:
        LOG.warning("no models profiled successfully — nothing to write")
        return

    df = pd.DataFrame(rows)
    cols = ["model_name", "model_family"] + [c for c in df.columns if c not in ("model_name", "model_family")]
    df = df[cols]

    if args.append and args.out.is_file():
        existing = pd.read_csv(args.out)
        existing = existing[~existing["model_name"].isin(df["model_name"])]
        df = pd.concat([existing, df], ignore_index=True)
        df = df.sort_values(["model_family", "model_name"]).reset_index(drop=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    LOG.info("wrote %d rows → %s", len(df), args.out)


if __name__ == "__main__":
    main()
