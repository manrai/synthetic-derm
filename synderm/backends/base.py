"""Common interface for image generator backends."""
from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image

from . import prompts as P


@dataclass(frozen=True)
class BackendInfo:
    """Static facts about a backend; see :mod:`synderm.backends` for the registry."""

    name: str
    model_id: str
    train_model_id: str
    supports_edit: bool
    supports_inpaint: bool
    default_resolution: int
    default_steps: int
    default_guidance: Optional[float]
    train_script_t2i: Optional[str]
    train_script_i2i: Optional[str]
    train_resolution_multiple: int
    license: str
    notes: str = ""
    caption_template: str = P.PAPER_TEMPLATE
    edit_template: str = P.EDIT_TEMPLATE


class GeneratorBackend(ABC):
    """A diffusion model wrapped behind three operations.

    - :meth:`text_to_image` generates from a prompt alone.
    - :meth:`edit` generates a new image conditioned on a reference image and a prompt
      (the 2026 replacement for the paper's in-then-outpaint method).
    - :meth:`inpaint` regenerates the white region of a mask (the paper's original method);
      only backends with ``info.supports_inpaint`` implement it, the others raise.

    Backends load lazily on first use; call :meth:`unload` to free the GPU. A LoRA adapter
    trained with :func:`synderm.fine_tune.train_lora` is applied when ``lora_path`` is given.
    """

    info: BackendInfo

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        lora_path: Optional[str | Path] = None,
        lora_scale: float = 1.0,
        offload: bool = False,
    ):
        self.model_id = model_id or self.info.model_id
        self.device = device
        self.dtype = dtype
        self.lora_path = Path(lora_path) if lora_path else None
        self.lora_scale = lora_scale
        self.offload = offload
        self._loaded = False
        if self.lora_path is not None and not self.lora_path.exists():
            raise FileNotFoundError(f"LoRA weights not found: {self.lora_path}")

    # ----- lifecycle -------------------------------------------------------------------
    def load(self) -> "GeneratorBackend":
        if not self._loaded:
            self._load()
            self._loaded = True
        return self

    def unload(self) -> None:
        self._unload()
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self.load()

    def __exit__(self, *exc):
        self.unload()

    @abstractmethod
    def _load(self) -> None: ...

    @abstractmethod
    def _unload(self) -> None: ...

    # ----- prompts ---------------------------------------------------------------------
    def caption(self, label: str, template: Optional[str] = None, fitzpatrick: Optional[int] = None) -> str:
        """Text-to-image prompt for a label; ``fitzpatrick`` (1 to 6) states the skin tone."""
        return P.caption(label, template or self.info.caption_template, fitzpatrick)

    def edit_caption(self, label: str, template: Optional[str] = None, fitzpatrick: Optional[int] = None) -> str:
        return P.caption(label, template or self.info.edit_template, fitzpatrick)

    # ----- generation ------------------------------------------------------------------
    def text_to_image(
        self,
        prompt: str | Sequence[str],
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> list[Image.Image]:
        """Generate ``num_images_per_prompt`` images for each prompt; returns them prompt-major."""
        self.load()
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        out: list[Image.Image] = []
        for i, p in enumerate(prompts):
            gen = self._generator(None if seed is None else seed + i)
            out.extend(self._text_to_image(p, num_images_per_prompt, gen, height, width,
                                           num_inference_steps, guidance_scale, negative_prompt))
        return out

    def edit(
        self,
        prompt: str | Sequence[str],
        image: Image.Image | Sequence[Image.Image],
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> list[Image.Image]:
        """Generate images conditioned on each reference image; returns them image-major."""
        if not self.info.supports_edit:
            raise NotImplementedError(f"{self.info.name} does not support reference-image editing")
        self.load()
        images = [image] if isinstance(image, Image.Image) else list(image)
        prompts = [prompt] * len(images) if isinstance(prompt, str) else list(prompt)
        if len(prompts) != len(images):
            raise ValueError("edit() needs one prompt per image, or a single prompt")
        out: list[Image.Image] = []
        for i, (p, im) in enumerate(zip(prompts, images)):
            gen = self._generator(None if seed is None else seed + i)
            out.extend(self._edit(p, im.convert("RGB"), num_images_per_prompt, gen, height, width,
                                  num_inference_steps, guidance_scale, negative_prompt))
        return out

    def inpaint(
        self,
        prompt: str | Sequence[str],
        image: Image.Image | Sequence[Image.Image],
        mask: Image.Image | Sequence[Image.Image],
        strength: float = 0.97,
        num_images_per_prompt: int = 1,
        seed: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
    ) -> list[Image.Image]:
        """Regenerate the white region of ``mask`` in ``image``; returns images image-major."""
        if not self.info.supports_inpaint:
            raise NotImplementedError(
                f"{self.info.name} has no hard-mask inpainting pipeline in diffusers; use edit() or the sdxl backend"
            )
        self.load()
        images = [image] if isinstance(image, Image.Image) else list(image)
        masks = [mask] * len(images) if isinstance(mask, Image.Image) else list(mask)
        prompts = [prompt] * len(images) if isinstance(prompt, str) else list(prompt)
        if not (len(prompts) == len(images) == len(masks)):
            raise ValueError("inpaint() needs matching prompts, images and masks (or single prompt/mask)")
        out: list[Image.Image] = []
        for i, (p, im, m) in enumerate(zip(prompts, images, masks)):
            gen = self._generator(None if seed is None else seed + i)
            out.extend(self._inpaint(p, im.convert("RGB"), m.convert("L"), strength, num_images_per_prompt, gen,
                                     num_inference_steps, guidance_scale, negative_prompt))
        return out

    # ----- hooks for subclasses ---------------------------------------------------------
    @abstractmethod
    def _text_to_image(self, prompt, n, generator, height, width, steps, guidance, negative_prompt) -> list[Image.Image]: ...

    def _edit(self, prompt, image, n, generator, height, width, steps, guidance, negative_prompt) -> list[Image.Image]:
        raise NotImplementedError

    def _inpaint(self, prompt, image, mask, strength, n, generator, steps, guidance, negative_prompt) -> list[Image.Image]:
        raise NotImplementedError

    # ----- helpers ---------------------------------------------------------------------
    def _generator(self, seed: Optional[int]) -> Optional[torch.Generator]:
        if seed is None:
            return None
        return torch.Generator(device="cpu").manual_seed(int(seed))

    def _size(self, height, width, multiple: int) -> tuple[int, int]:
        h = height or self.info.default_resolution
        w = width or self.info.default_resolution
        return (h // multiple) * multiple, (w // multiple) * multiple

    def describe(self) -> dict:
        return {
            "backend": self.info.name,
            "model_id": self.model_id,
            "lora": str(self.lora_path) if self.lora_path else None,
            "lora_scale": self.lora_scale if self.lora_path else None,
            "dtype": str(self.dtype).replace("torch.", ""),
        }
