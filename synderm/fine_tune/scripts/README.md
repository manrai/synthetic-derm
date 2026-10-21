# Vendored diffusers training scripts

These files are copied unmodified from `huggingface/diffusers`, `examples/dreambooth`, at
commit `0c7d56d6f4ea` (2026-10-21):

- `train_dreambooth_lora_sdxl.py`
- `train_dreambooth_lora_qwenimage21.py`, `train_dreambooth_lora_qwenimage21_img2img.py` (`README_qwenimage21.md`)
- `train_dreambooth_lora_flux3_klein.py`, `train_dreambooth_lora_flux3_klein_img2img.py` (`README_flux3_klein.md`)

`synderm.fine_tune.train_lora` launches them with `accelerate`. They are vendored so that a
given synderm version trains with a known script revision; to update, copy the files from the
upstream repository and record the commit here. License: Apache-2.0 (diffusers).

## Local patches

- `train_dreambooth_lora_flux3_klein.py` and `train_dreambooth_lora_flux3_klein_img2img.py`: the
  tokenizer (`Qwen3VLProcessor`) is loaded from `subfolder="text_encoder"` instead of
  `subfolder="tokenizer"`, because the `black-forest-labs/flux-3-image-klein-base` repository keeps
  the processor files next to the text encoder (its `modular_model_index.json` maps `tokenizer` to
  that folder). Upstream fails with "Unrecognized processing class" against that layout.
