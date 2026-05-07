"""runners package — pluggable model loaders for the benchmark + profiler."""

from .base import ModelRunner, ModelSpec
from .registry import (
    all_specs, get, list_families, list_models, load_all, make_runner, register,
)

__all__ = [
    "ModelRunner", "ModelSpec",
    "register", "get", "all_specs", "list_models", "list_families",
    "make_runner", "load_all",
]
