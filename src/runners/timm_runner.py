"""timm runner — classification backbones (ViT, DeiT, Swin, EfficientNet, ResNet, ConvNeXt).

Only registers if the `timm` package is importable. Inference takes a
preprocessed tensor; the benchmark script generates a synthetic one.

For ViT/DeiT we enable `dynamic_img_size=True` so the model interpolates
positional embeddings to the input shape — without this, `_224`/`_384`
named variants reject any other size. Swin uses `dynamic_img_pad=True`
which pads inputs to a multiple of the window size; both flags require
timm >= 1.0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModelRunner, ModelSpec
from .registry import register

TIMM_CATALOG: dict[str, list[str]] = {
    "vit": [
        "vit_small_patch16_224", "vit_base_patch16_224", "vit_large_patch16_224",
        "vit_small_patch16_384", "vit_base_patch16_384", "vit_large_patch16_384",
    ],
    "deit": [
        "deit_small_patch16_224", "deit_base_patch16_224",
        "deit_base_patch16_384",
        "deit3_large_patch16_224",
    ],
    "efficientnet":["efficientnet_b0", "efficientnet_b1", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5"],
    "resnet":      ["resnet18", "resnet50", "resnet101", "resnet152", "resnext50_32x4d"],
    "convnext":    ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"],
    "swin":        ["swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224"],
}

# All timm classification models share the same bench sweep — 6 sizes × 5
# batches × 2 devices = 60 measurements per model, matching YOLO coverage.
TIMM_BENCH_SIZES = (160, 224, 288, 384, 448, 512)
TIMM_BENCH_BATCHES = (1, 2, 4, 8, 16)
TIMM_ARCH_REF = 224  # canonical reference for FLOPs/params profiling


class TimmRunner(ModelRunner):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model: nn.Module | None = None
        self._current_img_size: int | None = None

    def _build_model(self, img_size: int) -> None:
        import timm
        kwargs = {}
        if self.spec.family in ("vit", "deit"):
            kwargs["dynamic_img_size"] = True
        elif self.spec.family == "swin":
            # Swin v1: relative_position_bias_table is sized at construction
            # from img_size — must rebuild for each new input side.
            kwargs["img_size"] = img_size
        self._model = timm.create_model(self.spec.name, pretrained=False, **kwargs).eval()
        if self.device == "cuda":
            self._model = self._model.cuda()
        self._current_img_size = img_size

    def load(self, device: str = "cpu") -> None:
        self.device = device
        self._build_model(self.spec.arch_ref_img_size)
        self._loaded = True

    def prepare(self, img_size: int, batch: int) -> None:
        # Only swin needs a rebuild; vit/deit interpolate pos_embed, CNNs are
        # fully convolutional.
        if self.spec.family == "swin" and self._current_img_size != img_size:
            self._build_model(img_size)

    def get_module(self) -> nn.Module:
        assert self._model is not None, "call load() first"
        return self._model

    def infer(self, inputs, *, img_size: int = 224) -> None:
        assert self._model is not None, "call load() first"
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("TimmRunner expects a torch.Tensor input")
        if self.device == "cuda":
            inputs = inputs.cuda()
        with torch.no_grad():
            self._model(inputs)
        if self.device == "cuda":
            torch.cuda.empty_cache()


try:
    import timm  # noqa: F401  - probe availability before registering
    for family, names in TIMM_CATALOG.items():
        for n in names:
            register(ModelSpec(
                name=n,
                family=family,
                runner_cls=TimmRunner,
                default_img_sizes=TIMM_BENCH_SIZES,
                default_batches=TIMM_BENCH_BATCHES,
                arch_ref_img_size=TIMM_ARCH_REF,
            ))
except ImportError:
    pass
