"""Qwen-Image-2.1: unified text-to-image and reference-image editing (Alibaba, Sept 2026)."""
from __future__ import annotations

import torch
from PIL import Image

from .base import BackendInfo, GeneratorBackend

INFO = BackendInfo(
    name="qwen-image-2.1",
    model_id="Qwen/Qwen-Image-2.1",
    train_model_id="Qwen/Qwen-Image-2.1",
    supports_edit=True,
    supports_inpaint=False,
    default_resolution=1024,
    default_steps=40,
    default_guidance=None,  # sampled without classifier-free guidance by default
    train_script_t2i="train_dreambooth_lora_qwenimage21.py",
    train_script_i2i="train_dreambooth_lora_qwenimage21_img2img.py",
    train_resolution_multiple=32,
    license="Qwen Research License (research only); Apache-2.0 fallbacks: Qwen-Image-2512, Qwen-Image-Edit-2511",
    notes="7B DiT plus Qwen3-VL text encoder; the encoder is large, so cpu offload is on by default on 32 GB cards.",
)


class QwenImageBackend(GeneratorBackend):
    info = INFO

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("offload", True)
        super().__init__(*args, **kwargs)
        self.pipe = None

    def _load(self) -> None:
        from diffusers import QwenImage21Pipeline
        self.pipe = QwenImage21Pipeline.from_pretrained(self.model_id, dtype=self.dtype)
        if self.lora_path:
            self.pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)
            self.pipe.set_adapters(self.pipe.get_active_adapters(), adapter_weights=[self.lora_scale])
        if self.offload:
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def _unload(self) -> None:
        self.pipe = None

    def _call(self, prompt, image, n, generator, height, width, steps, guidance, negative_prompt):
        kwargs = dict(
            prompt=prompt, num_images_per_prompt=n, generator=generator,
            num_inference_steps=steps or self.info.default_steps,
        )
        if image is not None:
            kwargs["image"] = image
        if height or width:
            h, w = self._size(height or width, width or height, 32)
            kwargs.update(height=h, width=w)
        else:
            kwargs["output_resolution"] = self.info.default_resolution
        if guidance and guidance > 1.0:
            kwargs.update(true_cfg_scale=guidance, negative_prompt=negative_prompt or "")
        return self.pipe(**kwargs).images

    def _text_to_image(self, prompt, n, generator, height, width, steps, guidance, negative_prompt):
        return self._call(prompt, None, n, generator, height, width, steps, guidance, negative_prompt)

    def _edit(self, prompt, image, n, generator, height, width, steps, guidance, negative_prompt):
        return self._call(prompt, image, n, generator, height, width, steps, guidance, negative_prompt)
