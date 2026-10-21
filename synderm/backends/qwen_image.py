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
    default_resolution=768,  # 1024 px needs text_encoder_bits=4 on a 32 GB card
    default_steps=40,
    default_guidance=None,  # sampled without classifier-free guidance by default
    train_script_t2i="train_dreambooth_lora_qwenimage21.py",
    train_script_i2i="train_dreambooth_lora_qwenimage21_img2img.py",
    train_resolution_multiple=32,
    license="Qwen Research License (research only); Apache-2.0 fallbacks: Qwen-Image-2512, Qwen-Image-Edit-2511",
    notes="7B DiT plus a 17 GB Qwen3-VL text encoder. On a 32 GB card the encoder is loaded in 8-bit "
          "(bitsandbytes) so everything stays on the GPU; cpu offload is about 7x slower per step.",
)


class QwenImageBackend(GeneratorBackend):
    info = INFO

    def __init__(self, *args, text_encoder_bits: int | None = 8, **kwargs):
        """``text_encoder_bits``: 8 (default) or 4 quantizes the Qwen3-VL text encoder with
        bitsandbytes; ``None`` keeps bf16, which needs ``offload=True`` on a 32 GB card."""
        super().__init__(*args, **kwargs)
        self.text_encoder_bits = text_encoder_bits
        if text_encoder_bits is None and not kwargs.get("offload", False):
            self.offload = True
        self.pipe = None

    def _load(self) -> None:
        from diffusers import QwenImage21Pipeline
        kwargs = {}
        if self.text_encoder_bits in (4, 8):
            from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
            qconf = (BitsAndBytesConfig(load_in_8bit=True) if self.text_encoder_bits == 8 else
                     BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=self.dtype))
            kwargs["text_encoder"] = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id, subfolder="text_encoder", dtype=self.dtype, quantization_config=qconf, device_map=self.device)
        elif self.text_encoder_bits is not None:
            raise ValueError("text_encoder_bits must be 8, 4 or None")
        self.pipe = QwenImage21Pipeline.from_pretrained(self.model_id, dtype=self.dtype, **kwargs)
        if self.lora_path:
            self.pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)
            self.pipe.set_adapters(self.pipe.get_active_adapters(), adapter_weights=[self.lora_scale])
        if self.offload:
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            # a bitsandbytes-quantized text encoder is already on the device and must not be moved
            for name in ("transformer", "vae"):
                getattr(self.pipe, name).to(self.device)
        if hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()  # decode one image at a time: the VAE decode is the memory spike
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
        # The reference image adds vision tokens and latents; one edit at 768 px already peaks near 31 GB
        # on a 32 GB card, so edits are generated one at a time with consecutive seeds.
        out = []
        base_seed = generator.initial_seed() if generator is not None else None
        for k in range(n):
            g = None if base_seed is None else torch.Generator(device="cpu").manual_seed(base_seed + k)
            out.extend(self._call(prompt, image, 1, g, height, width, steps, guidance, negative_prompt))
        return out
