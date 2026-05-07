"""Hugging Face transformers runner — DETR / Segformer.

For RT-DETR via HF (`PekingU/rtdetr_r50vd`) we recommend using RTDETRRunner
(ultralytics) — it's lighter and reuses the existing infra. HFRunner is for
models that *only* live on the Hub.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ModelRunner, ModelSpec
from .registry import register

HF_CATALOG: dict[str, list[str]] = {
    "detr":      ["facebook/detr-resnet-50", "facebook/detr-resnet-101"],
    "segformer": ["nvidia/segformer-b0-finetuned-ade-512-512",
                  "nvidia/segformer-b1-finetuned-ade-512-512",
                  "nvidia/segformer-b2-finetuned-ade-512-512",
                  "nvidia/segformer-b3-finetuned-ade-512-512",
                  "nvidia/segformer-b4-finetuned-ade-512-512"],
}


class HFRunner(ModelRunner):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model: nn.Module | None = None

    def load(self, device: str = "cpu") -> None:
        from transformers import AutoModel
        self._model = AutoModel.from_pretrained(self.spec.name).eval()
        if device == "cuda":
            self._model = self._model.cuda()
        self.device = device
        self._loaded = True

    def get_module(self) -> nn.Module:
        assert self._model is not None, "call load() first"
        return self._model

    def infer(self, inputs, *, img_size: int = 800) -> None:
        assert self._model is not None, "call load() first"
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("HFRunner expects a torch.Tensor (pixel_values)")
        if self.device == "cuda":
            inputs = inputs.cuda()
        with torch.no_grad():
            self._model(pixel_values=inputs)
        if self.device == "cuda":
            torch.cuda.empty_cache()


_HF_ARCH_REF = {"detr": 800, "segformer": 512}

try:
    import transformers  # noqa: F401
    for family, names in HF_CATALOG.items():
        for n in names:
            register(ModelSpec(
                name=n,
                family=family,
                runner_cls=HFRunner,
                default_img_sizes=(320, 480, 640, 800, 960, 1120),
                default_batches=(1, 2, 4, 6, 8),
                arch_ref_img_size=_HF_ARCH_REF[family],
            ))
except ImportError:
    pass
