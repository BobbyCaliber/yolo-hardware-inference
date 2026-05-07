"""Central catalog of model_name → ModelSpec.

Runners self-register by calling `register()` on import. The benchmark loop,
the arch-feature profiler, and the prediction CLI all look models up here
instead of hard-coding a list.
"""

from __future__ import annotations

from typing import Iterable

from .base import ModelSpec, ModelRunner

_REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    if spec.name in _REGISTRY:
        # Re-registration with the same spec is fine (re-imports during tests).
        if _REGISTRY[spec.name] is spec:
            return
        raise ValueError(f"model already registered: {spec.name}")
    _REGISTRY[spec.name] = spec


def get(name: str) -> ModelSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def all_specs() -> list[ModelSpec]:
    return list(_REGISTRY.values())


def list_models(family: str | None = None) -> list[str]:
    if family is None:
        return sorted(_REGISTRY)
    return sorted(n for n, s in _REGISTRY.items() if s.family == family)


def list_families() -> list[str]:
    return sorted({s.family for s in _REGISTRY.values()})


def make_runner(name: str) -> ModelRunner:
    spec = get(name)
    return spec.runner_cls(spec)


def load_all() -> None:
    """Import every runner module so all registrations land.

    Imports are guarded — if a backend (timm, transformers, torchvision) isn't
    installed, the corresponding runner module is skipped silently. The model
    just won't appear in the registry on that machine.
    """
    import importlib
    for mod in ("yolo", "rt_detr", "timm_runner", "torchvision_runner", "hf_runner"):
        try:
            importlib.import_module(f"runners.{mod}")
        except Exception as e:  # noqa: BLE001 — backend missing or broken, intentional fallthrough
            import logging
            logging.getLogger("runners.registry").debug("skip runners.%s: %s", mod, e)
