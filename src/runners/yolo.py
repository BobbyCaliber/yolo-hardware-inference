"""YOLO runner — wraps ultralytics.YOLO for the existing 24-model catalog."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

from .base import ModelRunner, ModelSpec
from .registry import register

YOLO_CATALOG: dict[str, list[str]] = {
    "yolov5": ["yolov5nu.pt", "yolov5su.pt", "yolov5mu.pt", "yolov5lu.pt"],
    "yolov6": ["yolov6n.yaml", "yolov6s.yaml", "yolov6m.yaml", "yolov6l.yaml"],
    "yolov8": ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
    "yolov9": ["yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt"],
    "yolov10": ["yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt"],
    "yolo11": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt"],
}


class YOLORunner(ModelRunner):
    """Loads via `ultralytics.YOLO(name)`. Inference takes file paths or tensors."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._wrapper = None  # ultralytics.YOLO instance

    def load(self, device: str = "cpu") -> None:
        from ultralytics import YOLO
        weights_dir = Path(self.spec.load_kwargs.get("weights_dir", "/app/weights"))
        weights_dir.mkdir(parents=True, exist_ok=True)
        # ultralytics caches weights to CWD on first download; isolate to weights_dir
        cwd = Path.cwd()
        try:
            os.chdir(weights_dir)
            self._wrapper = YOLO(self.spec.name)
        finally:
            os.chdir(cwd)
        self.device = device
        self._loaded = True

    def get_module(self) -> nn.Module:
        assert self._wrapper is not None, "call load() first"
        return self._wrapper.model

    def warmup(self, input_shape: tuple[int, ...] = (1, 3, 640, 640), n: int = 1) -> None:
        # ultralytics' .predict() expects file paths or PIL/numpy images; the
        # benchmark loop already handles warmup by running through real inputs,
        # so this is a no-op for YOLO unless a tensor is explicitly passed.
        pass

    def infer(self, inputs, *, img_size: int = 640) -> None:
        assert self._wrapper is not None, "call load() first"
        device = "cuda" if self.device == "cuda" else "cpu"
        self._wrapper.predict(source=inputs, imgsz=img_size, device=device, save=False, verbose=False)
        if device == "cuda":
            torch.cuda.empty_cache()


# ---- registry -----
for family, names in YOLO_CATALOG.items():
    for n in names:
        register(ModelSpec(
            name=n,
            family=family,
            runner_cls=YOLORunner,
            default_img_sizes=(160, 320, 640, 800, 960, 1120),
            default_batches=(1, 2, 5, 8, 10),
            arch_ref_img_size=640,
        ))
