"""ModelRunner — common interface for benchmarking any CV model.

Each runner wraps one model family (YOLO, RT-DETR, timm, torchvision, HF) and
exposes a uniform load/warmup/infer API so the benchmark loop and arch-feature
profiler don't need to know which framework the weights came from.

Concrete runners live in `src/runners/<family>.py` and register themselves
via `runners.registry.register()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class ModelSpec:
    """One catalog entry — what to load + how to label it in the dataset."""
    name: str                       # canonical model name used as model_name in the CSV
    family: str                     # used as model_family
    runner_cls: type["ModelRunner"]
    load_kwargs: dict = field(default_factory=dict)
    # default benchmark sweep — runners can be parametrized differently if needed
    default_img_sizes: tuple[int, ...] = (640,)
    default_batches: tuple[int, ...] = (1, 8)
    # single canonical size used by arch-feature profiling (FLOPs/activations
    # are then extrapolated to other input sizes via (img/img_ref)^2 in enrich)
    arch_ref_img_size: int = 640


class ModelRunner(ABC):
    """Abstract runner. One instance per model.

    Lifecycle:
        runner = SomeRunner(spec)
        runner.load()
        runner.warmup(input_shape=(1, 3, 640, 640))
        for ... :
            t = runner.infer(input_tensor_or_paths)
        module = runner.get_module()   # for arch profiling
    """

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.device: str = "cpu"
        self._loaded = False

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load weights and move to `device` ('cpu' or 'cuda'). Idempotent."""

    @abstractmethod
    def get_module(self) -> nn.Module:
        """Return the underlying nn.Module for arch profiling (FLOPs, hooks)."""

    @abstractmethod
    def infer(self, inputs) -> None:
        """One inference call. `inputs` may be a tensor, list of paths, or PIL images."""

    def prepare(self, img_size: int, batch: int) -> None:
        """Hook called before each (img_size, batch) inference, OUTSIDE the timer.

        Default no-op. Override in runners that need per-input-size re-init —
        e.g. timm Swin v1 hard-binds `relative_position_bias_table` to img_size
        at construction, so the model is rebuilt when the size changes.
        """
        pass

    def warmup(self, input_shape: tuple[int, ...] = (1, 3, 640, 640), n: int = 2) -> None:
        """Default warmup: build a dummy tensor and run `n` forward passes.

        Runners that take file paths (e.g. ultralytics) override this to use
        the real benchmark inputs instead.
        """
        dummy = torch.zeros(*input_shape)
        if self.device == "cuda":
            dummy = dummy.cuda()
        for _ in range(n):
            self.infer(dummy)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def family(self) -> str:
        return self.spec.family
