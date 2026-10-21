"""LoRA fine-tuning of generator backends via the upstream diffusers DreamBooth scripts.

The scripts under ``synderm/fine_tune/scripts`` are vendored unmodified from
``huggingface/diffusers`` (``examples/dreambooth``); :func:`train_lora` exports a dataset in the
format they read and launches them with ``accelerate``.
"""
from .lora import train_lora, TrainingError, SCRIPTS_DIR

__all__ = ["train_lora", "TrainingError", "SCRIPTS_DIR"]
