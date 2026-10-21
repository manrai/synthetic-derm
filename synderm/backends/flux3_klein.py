"""FLUX 3 Image Klein (Black Forest Labs, Oct 2026): text-to-image and multi-reference editing.

Two checkpoints: ``flux-3-image-klein`` is step-distilled (16 steps, no guidance) and is used for
zero-shot generation; ``flux-3-image-klein-base`` (50 steps, guidance 3.5) is the one LoRA
adapters are trained on, so it is selected automatically when ``lora_path`` is given.

The model expects full scene captions. Its 9B prompt upsampler is a separate model that does
not fit next to the transformer on a 32 GB card, so :meth:`upsample_prompts` runs it as a
separate stage (load, caption, unload) with an on-disk cache.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image

from . import prompts as P
from .base import BackendInfo, GeneratorBackend

INFO = BackendInfo(
    name="flux-3-klein",
    model_id="black-forest-labs/flux-3-image-klein",
    train_model_id="black-forest-labs/flux-3-image-klein-base",
    supports_edit=True,
    supports_inpaint=False,
    default_resolution=1024,
    default_steps=16,
    default_guidance=3.5,  # base model only; the distilled checkpoint ignores guidance
    train_script_t2i="train_dreambooth_lora_flux3_klein.py",
    train_script_i2i="train_dreambooth_lora_flux3_klein_img2img.py",
    train_resolution_multiple=16,
    license="FLUX Kommunity License (non-commercial; small-revenue exception); Apache-2.0 fallback: FLUX.2-klein-4B",
    notes="Prompts should be full scene captions; use upsample_prompts() or CLINICAL_TEMPLATE.",
    caption_template=P.CLINICAL_TEMPLATE,
    edit_template=P.FLUX_EDIT_TEMPLATE,
)
BASE_MODEL_ID = "black-forest-labs/flux-3-image-klein-base"
BASE_STEPS = 50


class Flux3KleinBackend(GeneratorBackend):
    info = INFO

    def __init__(self, *args, upsampler_cache: Optional[str | os.PathLike] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.lora_path is not None and "model_id" not in kwargs and (not args):
            # LoRA adapters are trained on the base checkpoint.
            self.model_id = BASE_MODEL_ID
        self.is_distilled = "klein-base" not in self.model_id
        self.pipe = None
        self.upsampler_cache = Path(upsampler_cache) if upsampler_cache else (
            Path(os.environ.get("SYNDERM_CACHE_DIR", "/workspace/synderm/cache")) / "flux3-upsampled-prompts.json"
        )

    # ----- loading ---------------------------------------------------------------------
    def _blocks_without_upsampler(self):
        from diffusers import ModularPipeline
        blocks = ModularPipeline.from_pretrained(self.model_id).blocks
        pup = blocks.sub_blocks.pop("prompt_upsampler", None)
        return blocks, pup

    def _load(self) -> None:
        blocks, _ = self._blocks_without_upsampler()
        self.pipe = blocks.init_pipeline(self.model_id)
        self.pipe.load_components(dtype={"vae": torch.float32, "default": self.dtype})
        if self.lora_path:
            self.pipe.load_lora_weights(str(self.lora_path.parent), weight_name=self.lora_path.name)
            try:
                self.pipe.set_adapters(self.pipe.get_active_adapters(), adapter_weights=[self.lora_scale])
            except Exception:
                pass  # single adapter at scale 1.0 is the default
        if self.offload:
            self.pipe.enable_model_cpu_offload(device=self.device)
        else:
            self.pipe.to(self.device)

    def _unload(self) -> None:
        self.pipe = None

    # ----- prompt upsampling ------------------------------------------------------------
    def upsample_prompts(self, prompts: Sequence[str], images: Optional[Sequence[Image.Image]] = None) -> list[str]:
        """Expand short prompts into full scene captions with the official 9B upsampler.

        Loads the upsampler alone (the main pipeline is unloaded first if needed), captions,
        then frees it. Results are cached on disk keyed by prompt (and reference image hash).
        """
        cache: dict[str, str] = {}
        if self.upsampler_cache.exists():
            cache = json.loads(self.upsampler_cache.read_text())
        keys = []
        for i, p in enumerate(prompts):
            key = p
            if images is not None:
                key = p + "|" + hashlib.md5(images[i].tobytes()).hexdigest()
            keys.append(key)
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            was_loaded = self._loaded
            if was_loaded:
                self.unload()
            blocks, pup_block = self._blocks_without_upsampler()
            if pup_block is None:
                raise RuntimeError("This checkpoint's pipeline has no prompt_upsampler block")
            pup = pup_block.init_pipeline(self.model_id)
            pup.load_components(dtype=self.dtype)
            pup.to(self.device)
            for i in missing:
                kwargs = {"prompt": prompts[i]}
                if images is not None:
                    kwargs["image"] = images[i]
                cache[keys[i]] = pup(**kwargs, output="prompt")
            del pup
            torch.cuda.empty_cache()
            self.upsampler_cache.parent.mkdir(parents=True, exist_ok=True)
            self.upsampler_cache.write_text(json.dumps(cache, indent=1))
            if was_loaded:
                self.load()
        return [cache[k] for k in keys]

    # ----- generation ------------------------------------------------------------------
    def _call(self, prompt, image, n, generator, height, width, steps, guidance, negative_prompt):
        kwargs = dict(prompt=prompt, num_images_per_prompt=n, generator=generator, output="images")
        kwargs["num_inference_steps"] = steps or (BASE_STEPS if not self.is_distilled else self.info.default_steps)
        if image is not None:
            kwargs["image"] = image
        if height or width:
            h, w = self._size(height or width, width or height, 16)
            kwargs.update(height=h, width=w)
        elif image is None:
            kwargs.update(height=self.info.default_resolution, width=self.info.default_resolution)
        if not self.is_distilled:
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt
            if guidance is not None and hasattr(self.pipe, "guider"):
                self.pipe.guider = self.pipe.guider.new(guidance_scale=guidance)
        return list(self.pipe(**kwargs))

    def _text_to_image(self, prompt, n, generator, height, width, steps, guidance, negative_prompt):
        return self._call(prompt, None, n, generator, height, width, steps, guidance, negative_prompt)

    def _edit(self, prompt, image, n, generator, height, width, steps, guidance, negative_prompt):
        return self._call(prompt, image, n, generator, height, width, steps, guidance, negative_prompt)
