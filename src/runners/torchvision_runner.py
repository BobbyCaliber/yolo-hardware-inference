"""torchvision runner — detection / segmentation backbones.

Covers the stage-5 catalog: Faster R-CNN, Mask R-CNN, DeepLabV3, plus a
generic U-Net (assembled from torchvision blocks since torchvision doesn't
ship a canonical U-Net — placeholder kept until segmentation_models_pytorch
is added if needed).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from .base import ModelRunner, ModelSpec
from .registry import register


def _fasterrcnn_resnet50():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)


def _fasterrcnn_resnet50_v2():
    from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
    return fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)


def _fasterrcnn_mobilenet_v3_large_fpn():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    return fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None)


def _fasterrcnn_mobilenet_v3_large_320_fpn():
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    return fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)


def _maskrcnn_resnet50():
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    return maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)


def _maskrcnn_resnet50_v2():
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    return maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)


def _keypointrcnn_resnet50():
    from torchvision.models.detection import keypointrcnn_resnet50_fpn
    return keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)


def _deeplabv3_resnet50():
    from torchvision.models.segmentation import deeplabv3_resnet50
    return deeplabv3_resnet50(weights=None, weights_backbone=None)


def _deeplabv3_resnet101():
    from torchvision.models.segmentation import deeplabv3_resnet101
    return deeplabv3_resnet101(weights=None, weights_backbone=None)


def _deeplabv3_mobilenet_v3_large():
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    return deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None)


def _fcn_resnet50():
    from torchvision.models.segmentation import fcn_resnet50
    return fcn_resnet50(weights=None, weights_backbone=None)


def _fcn_resnet101():
    from torchvision.models.segmentation import fcn_resnet101
    return fcn_resnet101(weights=None, weights_backbone=None)


def _lraspp_mobilenet_v3_large():
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large
    return lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)


TV_CATALOG: dict[str, dict[str, Callable[[], nn.Module]]] = {
    "fasterrcnn": {
        "fasterrcnn_resnet50_fpn": _fasterrcnn_resnet50,
        "fasterrcnn_resnet50_fpn_v2": _fasterrcnn_resnet50_v2,
        "fasterrcnn_mobilenet_v3_large_fpn": _fasterrcnn_mobilenet_v3_large_fpn,
        "fasterrcnn_mobilenet_v3_large_320_fpn": _fasterrcnn_mobilenet_v3_large_320_fpn,
    },
    "maskrcnn": {
        "maskrcnn_resnet50_fpn": _maskrcnn_resnet50,
        "maskrcnn_resnet50_fpn_v2": _maskrcnn_resnet50_v2,
    },
    "keypointrcnn": {
        "keypointrcnn_resnet50_fpn": _keypointrcnn_resnet50,
    },
    "deeplabv3": {
        "deeplabv3_resnet50": _deeplabv3_resnet50,
        "deeplabv3_resnet101": _deeplabv3_resnet101,
        "deeplabv3_mobilenet_v3_large": _deeplabv3_mobilenet_v3_large,
    },
    "fcn": {
        "fcn_resnet50": _fcn_resnet50,
        "fcn_resnet101": _fcn_resnet101,
    },
    "lraspp": {
        "lraspp_mobilenet_v3_large": _lraspp_mobilenet_v3_large,
    },
}

# Detection models from torchvision return loss dicts in train mode and
# detection lists in eval mode; both expect a *list* of tensors as input.
DETECTION_FAMILIES = {"fasterrcnn", "maskrcnn", "keypointrcnn"}


class TorchvisionRunner(ModelRunner):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model: nn.Module | None = None
        self._builder: Callable[[], nn.Module] = spec.load_kwargs["builder"]
        self._is_detection: bool = spec.family in DETECTION_FAMILIES

    def load(self, device: str = "cpu") -> None:
        self._model = self._builder().eval()
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
            raise TypeError("TorchvisionRunner expects a torch.Tensor input")
        if self.device == "cuda":
            inputs = inputs.cuda()
        with torch.no_grad():
            if self._is_detection:
                # detection models want list[Tensor], one per image
                self._model([img for img in inputs])
            else:
                self._model(inputs)
        if self.device == "cuda":
            torch.cuda.empty_cache()


try:
    import torchvision  # noqa: F401
    for family, models in TV_CATALOG.items():
        for n, builder in models.items():
            register(ModelSpec(
                name=n,
                family=family,
                runner_cls=TorchvisionRunner,
                load_kwargs={"builder": builder},
                default_img_sizes=(320, 480, 640, 800, 960, 1120),
                default_batches=(1, 2, 4, 6, 8),
                arch_ref_img_size=640,
            ))
except ImportError:
    pass
