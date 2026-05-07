"""RT-DETR runner — uses ultralytics' RTDETR class (same backend as YOLO).

Catalog covers small / large variants from ultralytics' shipped weights.
The plan asks for 3 sizes (small / base / large) — ultralytics ships
`rtdetr-l.pt` and `rtdetr-x.pt`; we add the HF-mirrored `rtdetr-resnet50`
variant later through HFRunner if needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from .base import ModelRunner, ModelSpec
from .registry import register

RTDETR_CATALOG = ["rtdetr-l.pt", "rtdetr-x.pt"]


class RTDETRRunner(ModelRunner):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._wrapper = None

    def load(self, device: str = "cpu") -> None:
        from ultralytics import RTDETR
        weights_dir = Path(self.spec.load_kwargs.get("weights_dir", "/app/weights"))
        weights_dir.mkdir(parents=True, exist_ok=True)
        cwd = Path.cwd()
        try:
            os.chdir(weights_dir)
            self._wrapper = RTDETR(self.spec.name)
        finally:
            os.chdir(cwd)
        self.device = device
        self._loaded = True

    def get_module(self) -> nn.Module:
        assert self._wrapper is not None, "call load() first"
        return self._wrapper.model

    def warmup(self, input_shape=(1, 3, 640, 640), n: int = 1) -> None:
        pass

    def infer(self, inputs, *, img_size: int = 640) -> None:
        assert self._wrapper is not None, "call load() first"
        device = "cuda" if self.device == "cuda" else "cpu"
        self._wrapper.predict(source=inputs, imgsz=img_size, device=device, save=False, verbose=False)
        if device == "cuda":
            torch.cuda.empty_cache()


for n in RTDETR_CATALOG:
    register(ModelSpec(
        name=n,
        family="rtdetr",
        runner_cls=RTDETRRunner,
        default_img_sizes=(320, 480, 640, 800, 960, 1120),
        default_batches=(1, 2, 4, 6, 8),
        arch_ref_img_size=640,
    ))
