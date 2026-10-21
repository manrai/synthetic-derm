from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

from torch.utils.data import Dataset

from synderm.backends import GeneratorBackend, get_backend
from synderm.backends import masks as M
from synderm.backends import prompts as P
from synderm.data.datasets import SyntheticDataset, label_of, normalize_label

#: Generation methods. ``inpaint`` and ``inpaint-outpaint`` are the paper's hard-mask methods
#: (SDXL only); ``edit`` is reference-image editing (Qwen, FLUX); ``text-to-image`` is all three.
METHODS = ("text-to-image", "edit", "inpaint", "inpaint-outpaint")


def generate_synthetic_dataset(
    backend: str | GeneratorBackend,
    output_dir: str | os.PathLike,
    method: str = "text-to-image",
    dataset: Optional[Dataset] = None,
    labels: Optional[Sequence[str]] = None,
    num_per_image: int = 10,
    num_per_label: Optional[int] = None,
    prompt_template: Optional[str] = None,
    prompts_by_label: Optional[dict[str, str]] = None,
    seed: int = 42,
    batch_size: int = 4,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    strength_inpaint: float = 0.97,
    strength_outpaint: float = 0.95,
    center_fraction: float = 0.5,
    overwrite: bool = False,
    limit: Optional[int] = None,
    progress: bool = True,
) -> SyntheticDataset:
    """Generate synthetic images and write ``<output_dir>/<label>/<id>.png`` plus ``<id>.json``.

    With ``dataset`` (items with ``id``, ``image``, ``label``) every real image gets
    ``num_per_image`` generations and the JSON records ``ref_id`` so a split can keep
    synthetic images of validation reals out of training (the paper's protocol). For
    ``text-to-image`` the real image only supplies the label unless ``prompts_by_label`` maps
    labels to prompts. Without ``dataset``, pass ``labels`` and ``num_per_label``.

    Existing files are skipped unless ``overwrite``; the call is resumable. Returns a
    :class:`~synderm.data.SyntheticDataset` over ``output_dir``.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    gen = get_backend(backend) if isinstance(backend, str) else backend
    info = gen.info
    if method == "edit" and not info.supports_edit:
        raise NotImplementedError(f"{info.name} does not support edit; use text-to-image")
    if method.startswith("inpaint") and not info.supports_inpaint:
        raise NotImplementedError(f"{info.name} does not support hard-mask inpainting; use edit or the sdxl backend")
    if method != "text-to-image" and dataset is None:
        raise ValueError(f"method {method!r} conditions on real images: pass dataset")
    if dataset is None and not (labels and num_per_label):
        raise ValueError("text-to-image without a dataset needs labels and num_per_label")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    template = prompt_template or (info.edit_template if method == "edit" else info.caption_template)

    def prompt_for(label: str) -> str:
        if prompts_by_label and label in prompts_by_label:
            return prompts_by_label[label]
        return P.caption(label, template)

    # Work list: (label, index into dataset or None)
    jobs: list[tuple[str, Optional[int]]] = []
    if dataset is not None:
        keep = {normalize_label(l) for l in labels} if labels else None
        for i in range(len(dataset)):
            lab = label_of(dataset, i)
            if keep is not None and lab not in keep:
                continue
            jobs.append((lab, i))
    else:
        for lab in labels:
            jobs.append((normalize_label(lab), None))
    if limit:
        jobs = jobs[:limit]

    base_meta = {**gen.describe(), "method": method, "prompt_template": template,
                 "strength_inpaint": strength_inpaint if method.startswith("inpaint") else None,
                 "strength_outpaint": strength_outpaint if method == "inpaint-outpaint" else None,
                 "center_fraction": center_fraction if method.startswith("inpaint") else None,
                 "num_inference_steps": num_inference_steps, "guidance_scale": guidance_scale}

    iterator = jobs
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(jobs, desc=f"{info.name} {method}", unit="ref")

    n_written = 0
    for lab, idx in iterator:
        item = dataset[idx] if idx is not None else None
        ref_id = item["id"] if item is not None else None
        count = num_per_image if item is not None else num_per_label
        label_dir = output_dir / lab
        label_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{ref_id}-{method}" if ref_id else f"{lab}-{method}-seed{seed}"
        todo = [k for k in range(count) if overwrite or not (label_dir / f"{stem}-{k:03d}.png").exists()]
        if not todo:
            continue
        prompt = prompt_for(lab)
        # Deterministic per-reference seed so reruns reproduce and references differ.
        ref_seed = seed + (int(hashlib.md5(ref_id.encode()).hexdigest()[:8], 16) % 1_000_000 if ref_id else 0)
        images = []
        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            s = ref_seed + chunk[0]
            if method == "text-to-image":
                out = gen.text_to_image(prompt, num_images_per_prompt=len(chunk), seed=s, height=height, width=width,
                                        num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            elif method == "edit":
                out = gen.edit(prompt, item["image"], num_images_per_prompt=len(chunk), seed=s, height=height,
                               width=width, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            else:
                size = item["image"].size
                out = gen.inpaint(prompt, item["image"], M.center_mask(size, center_fraction), strength=strength_inpaint,
                                  num_images_per_prompt=len(chunk), seed=s, num_inference_steps=num_inference_steps,
                                  guidance_scale=guidance_scale)
                if method == "inpaint-outpaint":
                    out = gen.inpaint([prompt] * len(out), out, M.border_mask(out[0].size, center_fraction),
                                      strength=strength_outpaint, num_images_per_prompt=1, seed=s + 10_000,
                                      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
            images.extend(zip(chunk, out))
        for k, im in images:
            sid = f"{stem}-{k:03d}"
            im.save(label_dir / f"{sid}.png")
            meta = {"id": sid, "label": lab, "ref_id": ref_id, "k": k, "prompt": prompt, "seed": ref_seed + k,
                    "height": im.height, "width": im.width, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **base_meta}
            if item is not None and "fitzpatrick_scale" in item:
                meta["ref_fitzpatrick_scale"] = item["fitzpatrick_scale"]
            (label_dir / f"{sid}.json").write_text(json.dumps(meta, indent=1))
            n_written += 1
    return SyntheticDataset(output_dir)
