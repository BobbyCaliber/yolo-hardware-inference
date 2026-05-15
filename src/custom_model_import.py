"""Import user-supplied .pt / .onnx models into the predict catalogue.

Streamlit's "Models" tab lets a user upload a model file. We profile it once
(arch-features only — no inference benchmark) and append a row to
data_base/model_arch_features.csv. After that the model_name shows up in the
Predict / Recommend dropdowns alongside the native catalogue, and CatBoost
extrapolates inference time from the architectural fingerprint.

Accepted inputs:
  - .pt   — full pickled nn.Module saved via `torch.save(model, ...)`.
            State_dicts and TorchScript ScriptModules are rejected: state_dicts
            have no class to instantiate against on the server, and ScriptModule
            children fail `isinstance(m, nn.Conv2d)` checks so the operator
            histogram comes out empty.
  - .onnx — converted to nn.Module via onnx2torch.convert, then profiled with
            the same hooks as everything else.

Uploaded files persist under user_models/weights/<name><ext> so they survive
Streamlit restarts.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# profile_model lives in scripts/, src/ must be importable for its `runners` import
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compute_arch_features import profile_model  # noqa: E402

LOG = logging.getLogger("custom_model_import")

ARCH_CSV = PROJECT_ROOT / "data_base" / "model_arch_features.csv"
WEIGHTS_DIR = PROJECT_ROOT / "user_models" / "weights"

SUPPORTED_EXT = (".pt", ".onnx")
CUSTOM_FAMILY = "custom"


class CustomModelError(ValueError):
    """Raised when a user-supplied file can't be turned into a profileable nn.Module."""


def _load_pt(path: Path) -> nn.Module:
    # Reject TorchScript up front — ScriptModule children fail isinstance()
    # checks and the op histogram comes out empty (verified at design time).
    try:
        torch.jit.load(str(path), map_location="cpu")
    except RuntimeError:
        pass  # not TorchScript, good — fall through to torch.load
    else:
        raise CustomModelError(
            "this .pt file is a TorchScript module. Re-save it as a regular pickle\n"
            "with  `torch.save(model, 'name.pt')`  (full module, not just state_dict),\n"
            "or export to ONNX with  `torch.onnx.export(...)`."
        )

    try:
        obj = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        raise CustomModelError(
            f"torch.load failed: {e}\n"
            "If your file is a state_dict, you need the original model class — "
            "save the full module instead with `torch.save(model, 'name.pt')`, "
            "or export to ONNX."
        ) from e

    if not isinstance(obj, nn.Module):
        kind = type(obj).__name__
        raise CustomModelError(
            f"file contains a {kind}, not an nn.Module. "
            "Most likely it's a state_dict — re-save with "
            "`torch.save(model, 'name.pt')` (the model itself, not its state_dict), "
            "or export to ONNX."
        )
    return obj.eval()


def _fold_identity_nodes(onnx_model) -> None:
    """In-place: remove Identity nodes by rewriting consumers to read the source.

    PyTorch's ONNX exporter routes weights/biases through Identity nodes when
    the same constant feeds multiple consumers (or when a Conv has bias=False
    and the exporter still wires a zero-bias initializer through Identity).
    onnx2torch's Conv converter reads bias directly from the initializer table
    by name, so a bias arriving via Identity-of-initializer KeyErrors. Folding
    every Identity removes that indirection and is semantically a no-op.
    """
    remap: dict[str, str] = {}
    keep = []
    for node in onnx_model.graph.node:
        if (node.op_type == "Identity"
                and len(node.input) == 1 and len(node.output) == 1):
            src = node.input[0]
            while src in remap:  # chain through previously folded Identities
                src = remap[src]
            remap[node.output[0]] = src
        else:
            keep.append(node)
    if not remap:
        return
    for node in keep:
        for i, name in enumerate(node.input):
            if name in remap:
                node.input[i] = remap[name]
    for out in onnx_model.graph.output:
        if out.name in remap:
            out.name = remap[out.name]
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(keep)


def _normalize_optional_inputs(onnx_model) -> None:
    """In-place: replace any node input that points at an unresolvable name with "".

    PyTorch sometimes emits Conv (and other ops with optional inputs) with a
    placeholder name in the bias slot instead of "". Canonicalizing dangling
    inputs to "" (the ONNX-blessed way to signal an absent optional input)
    makes onnx2torch's Conv converter take the no-bias path.
    """
    known: set[str] = set()
    for init in onnx_model.graph.initializer:
        known.add(init.name)
    for inp in onnx_model.graph.input:
        known.add(inp.name)
    for node in onnx_model.graph.node:
        for out in node.output:
            if out:
                known.add(out)
    for node in onnx_model.graph.node:
        for i, name in enumerate(node.input):
            if name and name not in known:
                node.input[i] = ""


def _load_onnx(path: Path) -> nn.Module:
    try:
        import onnx
        from onnx2torch import convert
    except ImportError as e:
        raise CustomModelError(
            "ONNX upload requires `onnx` and `onnx2torch`. Install with:\n"
            "    pip install onnx onnx2torch"
        ) from e

    try:
        onnx_model = onnx.load(str(path))
    except Exception as e:
        raise CustomModelError(f"onnx.load failed: {e}") from e

    _fold_identity_nodes(onnx_model)
    _normalize_optional_inputs(onnx_model)

    try:
        torch_model = convert(onnx_model)
    except Exception as e:
        raise CustomModelError(
            f"onnx2torch couldn't convert {path.name}: {e}\n"
            "The file probably uses an op onnx2torch doesn't support. "
            "Try re-exporting from PyTorch (`torch.onnx.export(..., opset_version=17)`) "
            "or upload a full-pickle .pt instead."
        ) from e
    return torch_model.eval()


def load_custom_module(path: Path) -> nn.Module:
    """Load a .pt or .onnx file as a profileable nn.Module."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return _load_pt(path)
    if suffix == ".onnx":
        return _load_onnx(path)
    raise CustomModelError(f"unsupported extension {suffix!r}. Use .pt or .onnx.")


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise CustomModelError("model name is required")
    bad = set(name) & set(",/\\\n\r\t\"'")
    if bad:
        raise CustomModelError(f"name contains illegal characters: {sorted(bad)}")
    return name


def register_custom_model(
    *,
    src_path: Path,
    name: str,
    family: str = CUSTOM_FAMILY,
    img_size: int = 640,
    arch_csv: Path = ARCH_CSV,
    weights_dir: Path = WEIGHTS_DIR,
) -> dict:
    """Profile a user-supplied model and append a row to `arch_csv`.

    Copies `src_path` into `weights_dir/<name><ext>` so the upload persists.
    Any existing row for `name` is replaced. Returns the feature dict written.
    """
    name = _validate_name(name)
    family = (family or CUSTOM_FAMILY).strip() or CUSTOM_FAMILY

    src_path = Path(src_path)
    if not src_path.is_file():
        raise FileNotFoundError(src_path)
    ext = src_path.suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise CustomModelError(f"unsupported extension {ext!r}. Use .pt or .onnx.")

    weights_dir.mkdir(parents=True, exist_ok=True)
    dest_path = weights_dir / f"{name}{ext}"
    if src_path.resolve() != dest_path.resolve():
        shutil.copy2(src_path, dest_path)

    LOG.info("profiling %s as %s @ %dpx", dest_path.name, name, img_size)
    module = load_custom_module(dest_path)
    feats = profile_model(module, img_size)
    feats["model_name"] = name
    feats["model_family"] = family

    new_row = pd.DataFrame([feats])

    if arch_csv.is_file():
        existing = pd.read_csv(arch_csv)
        existing = existing[existing["model_name"] != name]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row

    cols = ["model_name", "model_family"] + [
        c for c in combined.columns if c not in ("model_name", "model_family")
    ]
    combined = (
        combined[cols].sort_values(["model_family", "model_name"]).reset_index(drop=True)
    )
    arch_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(arch_csv, index=False)

    LOG.info(
        "registered %s — flops=%.2fG params=%.2fM ops=%d → %s",
        name,
        feats["flops"] / 1e9,
        feats["params"] / 1e6,
        feats["num_ops"],
        arch_csv,
    )
    return feats


def list_custom_models(arch_csv: Path = ARCH_CSV) -> pd.DataFrame:
    """Subset of `arch_csv` where model_family == 'custom'."""
    cols = ["model_name", "model_family", "params", "flops", "num_ops", "img_size_ref"]
    if not arch_csv.is_file():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(arch_csv)
    if "model_family" not in df.columns:
        return pd.DataFrame(columns=cols)
    out = df[df["model_family"] == CUSTOM_FAMILY]
    present = [c for c in cols if c in out.columns]
    return out[present].reset_index(drop=True)


def remove_custom_model(
    name: str,
    *,
    arch_csv: Path = ARCH_CSV,
    weights_dir: Path = WEIGHTS_DIR,
    delete_file: bool = True,
) -> bool:
    """Drop a custom-family row by name. Refuses to touch native rows."""
    if not arch_csv.is_file():
        return False
    df = pd.read_csv(arch_csv)
    if "model_family" not in df.columns:
        return False
    mask = (df["model_name"] == name) & (df["model_family"] == CUSTOM_FAMILY)
    if not mask.any():
        return False
    df = df[~mask].reset_index(drop=True)
    df.to_csv(arch_csv, index=False)
    if delete_file:
        for ext in SUPPORTED_EXT:
            p = weights_dir / f"{name}{ext}"
            if p.is_file():
                p.unlink()
    return True
