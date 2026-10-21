from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset

from synderm.backends import get_backend, GeneratorBackend
from synderm.backends import prompts as P
from synderm.data.export import export_training_parquet

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
LORA_WEIGHTS_NAME = "pytorch_lora_weights.safetensors"


class TrainingError(RuntimeError):
    """The training subprocess failed; the message carries the tail of its log."""


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return None


def train_lora(
    backend: str | GeneratorBackend,
    dataset: Dataset,
    output_dir: str | os.PathLike,
    mode: str = "text-to-image",
    caption_template: Optional[str] = None,
    caption_fn: Optional[Callable[[dict], str]] = None,
    resolution: int = 768,
    max_train_steps: int = 800,
    rank: int = 16,
    lora_alpha: Optional[int] = None,
    learning_rate: float = 1e-4,
    train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    seed: int = 0,
    validation_prompt: Optional[str] = None,
    validation_image: Optional[str | os.PathLike] = None,
    validation_epochs: int = 50,
    pairs_per_image: int = 1,
    offload: bool = True,
    cache_latents: bool = True,
    gradient_checkpointing: bool = True,
    use_8bit_adam: bool = True,
    mixed_precision: str = "bf16",
    checkpointing_steps: int = 250,
    extra_args: Optional[list[str]] = None,
    log_path: Optional[str | os.PathLike] = None,
    dry_run: bool = False,
) -> Path:
    """Train a LoRA adapter and return the path to ``pytorch_lora_weights.safetensors``.

    ``mode`` is ``"text-to-image"`` or ``"image-to-image"`` (reference-conditioned editing;
    Qwen-Image-2.1 and FLUX 3 Klein only). Captions default to the backend's template with the
    item's label; pass ``caption_template`` (``"{condition}"`` placeholder) or ``caption_fn`` to
    change that. One adapter covers every label in ``dataset``.

    The training data is exported to ``<output_dir>/train-data`` and the command, log and
    config are written next to the weights. Raises :class:`TrainingError` on failure.
    """
    gen = get_backend(backend) if isinstance(backend, str) else backend
    info = gen.info
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "text-to-image":
        script = info.train_script_t2i
    elif mode == "image-to-image":
        script = info.train_script_i2i
    else:
        raise ValueError("mode must be 'text-to-image' or 'image-to-image'")
    if script is None:
        raise NotImplementedError(f"{info.name} has no {mode} training script")
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        raise FileNotFoundError(script_path)
    if resolution % info.train_resolution_multiple:
        raise ValueError(f"{info.name} needs resolution divisible by {info.train_resolution_multiple}")
    if mode == "image-to-image" and train_batch_size != 1:
        raise ValueError("image-to-image training needs train_batch_size=1 (one layout per batch)")

    template = caption_template or (info.edit_template if mode == "image-to-image" else info.caption_template)
    cap = caption_fn or (lambda item: P.caption(item["label"], template))
    data_dir = export_training_parquet(
        dataset, output_dir / "train-data", caption_fn=cap, mode=mode,
        pairs_per_image=pairs_per_image, seed=seed, resize_to=max(resolution, 1024),
    )

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch", "--mixed_precision", mixed_precision,
        str(script_path),
        "--pretrained_model_name_or_path", info.train_model_id,
        "--dataset_name", str(data_dir),
        "--image_column", "image",
        "--caption_column", "caption",
        "--instance_prompt", P.caption("skin disease", template),
        "--output_dir", str(output_dir),
        "--resolution", str(resolution),
        "--train_batch_size", str(train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--learning_rate", str(learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(max_train_steps),
        "--rank", str(rank),
        "--seed", str(seed),
        "--mixed_precision", mixed_precision,
        "--checkpointing_steps", str(checkpointing_steps),
        "--checkpoints_total_limit", "2",
        "--report_to", "tensorboard",
    ]
    is_sdxl = info.name == "sdxl"
    if not is_sdxl:
        cmd += ["--lora_alpha", str(lora_alpha or rank)]
        if cache_latents:
            cmd.append("--cache_latents")
        if offload:
            cmd.append("--offload")
        if validation_prompt is None:
            cmd.append("--skip_final_inference")
    if mode == "image-to-image":
        cmd += ["--cond_image_column", "cond_image"]
    if gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if use_8bit_adam:
        cmd.append("--use_8bit_adam")
    if validation_prompt:
        cmd += ["--validation_prompt", validation_prompt, "--validation_epochs", str(validation_epochs)]
        if mode == "image-to-image":
            if validation_image is None:
                raise ValueError("image-to-image validation needs validation_image")
            cmd += ["--validation_image", str(validation_image)]
    if extra_args:
        cmd += list(extra_args)

    config = {
        "backend": info.name, "train_model_id": info.train_model_id, "mode": mode, "script": script,
        "caption_template": template, "resolution": resolution, "max_train_steps": max_train_steps,
        "rank": rank, "lora_alpha": lora_alpha or rank, "learning_rate": learning_rate,
        "train_batch_size": train_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
        "seed": seed, "num_train_images": len(dataset), "pairs_per_image": pairs_per_image,
        "command": " ".join(shlex.quote(c) for c in cmd), "git_commit": _git_commit(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    if dry_run:
        return output_dir / LORA_WEIGHTS_NAME

    log_path = Path(log_path) if log_path else output_dir / "train.log"
    env = dict(os.environ)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(output_dir))
    if proc.returncode != 0:
        tail = "".join(open(log_path).readlines()[-40:])
        raise TrainingError(f"{script} exited with code {proc.returncode}. Log: {log_path}\n--- tail ---\n{tail}")
    weights = output_dir / LORA_WEIGHTS_NAME
    if not weights.exists():
        raise TrainingError(f"Training finished but {weights} was not written. Log: {log_path}")
    config["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2))
    return weights
