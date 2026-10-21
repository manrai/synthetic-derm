"""Stable Diffusion XL 1.0: the classical baseline, with true hard-mask inpainting."""
from __future__ import annotations

import torch
from PIL import Image

from .base import BackendInfo, GeneratorBackend

INFO = BackendInfo(
    name="sdxl",
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    train_model_id="stabilityai/stable-diffusion-xl-base-1.0",
    supports_edit=False,
    supports_inpaint=True,
    default_resolution=1024,
    default_steps=30,
    default_guidance=5.0,
    train_script_t2i="train_dreambooth_lora_sdxl.py",
    train_script_i2i=None,
    train_resolution_multiple=8,
    license="CreativeML OpenRAIL++-M",
    notes="Inpainting uses diffusers/stable-diffusion-xl-1.0-inpainting-0.1; a LoRA trained on the base "
          "model is applied to the inpainting UNet as well.",
)
INPAINT_MODEL_ID = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"


class SDXLBackend(GeneratorBackend):
    info = INFO

    def __init__(self, *args, inpaint_model_id: str = INPAINT_MODEL_ID, **kwargs):
        # SDXL ships fp16 weights; bf16 works but fp16 is the released variant.
        kwargs.setdefault("dtype", torch.float16)
        super().__init__(*args, **kwargs)
        self.inpaint_model_id = inpaint_model_id
        self.pipe = None
        self.inpaint_pipe = None

    def _load(self) -> None:
        from diffusers import StableDiffusionXLPipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id, dtype=self.dtype, variant="fp16", use_safetensors=True
        )
        if self.lora_path:
            self.pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)
            self.pipe.set_adapters(self.pipe.get_active_adapters(), adapter_weights=[self.lora_scale])
        self._place(self.pipe)

    def _load_inpaint(self) -> None:
        if self.inpaint_pipe is not None:
            return
        from diffusers import StableDiffusionXLInpaintPipeline
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            self.inpaint_model_id, dtype=self.dtype, variant="fp16", use_safetensors=True
        )
        if self.lora_path:
            self.inpaint_pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)
            self.inpaint_pipe.set_adapters(self.inpaint_pipe.get_active_adapters(), adapter_weights=[self.lora_scale])
        self._place(self.inpaint_pipe)

    def _place(self, pipe) -> None:
        if self.offload:
            pipe.enable_model_cpu_offload(device=self.device)
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)

    def _unload(self) -> None:
        self.pipe = None
        self.inpaint_pipe = None

    def _text_to_image(self, prompt, n, generator, height, width, steps, guidance, negative_prompt):
        h, w = self._size(height, width, 8)
        return self.pipe(
            prompt=prompt, negative_prompt=negative_prompt, num_images_per_prompt=n, generator=generator,
            height=h, width=w, num_inference_steps=steps or self.info.default_steps,
            guidance_scale=self.info.default_guidance if guidance is None else guidance,
        ).images

    def _inpaint(self, prompt, image, mask, strength, n, generator, steps, guidance, negative_prompt):
        self._load_inpaint()
        # SDXL works at ~1024 px; scale the reference so its longer side matches, keeping aspect ratio.
        scale = self.info.default_resolution / max(image.size)
        h, w = self._size(round(image.height * scale), round(image.width * scale), 8)
        image = image.resize((w, h), Image.BICUBIC)
        mask = mask.resize((w, h), Image.NEAREST)
        return self.inpaint_pipe(
            prompt=prompt, negative_prompt=negative_prompt, image=image, mask_image=mask, strength=strength,
            num_images_per_prompt=n, generator=generator, height=h, width=w,
            num_inference_steps=steps or self.info.default_steps,
            guidance_scale=self.info.default_guidance if guidance is None else guidance,
        ).images
