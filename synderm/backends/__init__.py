"""Generator backends.

>>> from synderm.backends import get_backend
>>> gen = get_backend("sdxl")                      # or "qwen-image-2.1", "flux-3-klein"
>>> images = gen.text_to_image(gen.caption("psoriasis"), num_images_per_prompt=4, seed=0)

Backends are registered by name in :data:`BACKENDS`. Each exposes ``info`` (a
:class:`~synderm.backends.base.BackendInfo`) describing what it supports and how it is trained.
"""
from __future__ import annotations

from typing import Type

from .base import BackendInfo, GeneratorBackend
from .sdxl import SDXLBackend
from .qwen_image import QwenImageBackend
from .flux3_klein import Flux3KleinBackend
from . import prompts, masks

BACKENDS: dict[str, Type[GeneratorBackend]] = {
    SDXLBackend.info.name: SDXLBackend,
    QwenImageBackend.info.name: QwenImageBackend,
    Flux3KleinBackend.info.name: Flux3KleinBackend,
}
ALIASES = {"qwen": "qwen-image-2.1", "qwen-image": "qwen-image-2.1", "flux": "flux-3-klein", "flux3": "flux-3-klein"}


def list_backends() -> list[BackendInfo]:
    return [cls.info for cls in BACKENDS.values()]


def get_backend(name: str, **kwargs) -> GeneratorBackend:
    key = ALIASES.get(name.lower(), name.lower())
    if key not in BACKENDS:
        raise KeyError(f"Unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[key](**kwargs)


__all__ = ["BACKENDS", "BackendInfo", "GeneratorBackend", "get_backend", "list_backends",
           "SDXLBackend", "QwenImageBackend", "Flux3KleinBackend", "prompts", "masks"]
