"""Export a synderm dataset to the on-disk format the diffusers DreamBooth LoRA scripts read."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset

from .datasets import label_of, label_words


def export_training_parquet(
    dataset: Dataset,
    output_dir: str | os.PathLike,
    caption_fn: Callable[[dict], str],
    mode: str = "text-to-image",
    cond_caption_fn: Optional[Callable[[dict, dict], str]] = None,
    pairs_per_image: int = 1,
    seed: int = 0,
    resize_to: Optional[int] = None,
) -> Path:
    """Write ``<output_dir>/data/train-00000-of-00001.parquet`` with decoded image columns.

    ``mode="text-to-image"`` writes columns ``image``, ``caption``, ``id``, ``label``.
    ``mode="image-to-image"`` pairs every image with ``pairs_per_image`` other images of the
    same label as conditioning images (``cond_image``), so an adapter learns "same condition,
    different patient". The caption for a pair comes from ``cond_caption_fn(target_item,
    cond_item)``; it defaults to ``caption_fn(target_item)``.
    """
    from datasets import Dataset as HFDataset, Image as HFImage

    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    img_dir = output_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    def _save(item: dict) -> str:
        p = img_dir / f"{item['id']}.png"
        if not p.exists():
            im = item["image"]
            if resize_to:
                im = im.copy()
                im.thumbnail((resize_to, resize_to))
            im.save(p)
        return str(p)

    rows: dict[str, list] = {"image": [], "caption": [], "id": [], "label": []}
    if mode == "image-to-image":
        rows["cond_image"] = []
        rows["cond_id"] = []
        by_label: dict[str, list[int]] = {}
        for i in range(len(dataset)):
            by_label.setdefault(label_of(dataset, i), []).append(i)
        rng = random.Random(seed)
        for i in range(len(dataset)):
            item = dataset[i]
            same = [j for j in by_label[item["label"]] if j != i]
            if not same:
                raise ValueError(f"Label {item['label']!r} has a single image; image-to-image pairs need at least two")
            for j in rng.sample(same, min(pairs_per_image, len(same))):
                cond = dataset[j]
                rows["image"].append(_save(item))
                rows["cond_image"].append(_save(cond))
                rows["caption"].append((cond_caption_fn or (lambda t, c: caption_fn(t)))(item, cond))
                rows["id"].append(item["id"])
                rows["cond_id"].append(cond["id"])
                rows["label"].append(item["label"])
    elif mode == "text-to-image":
        for i in range(len(dataset)):
            item = dataset[i]
            rows["image"].append(_save(item))
            rows["caption"].append(caption_fn(item))
            rows["id"].append(item["id"])
            rows["label"].append(item["label"])
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    ds = HFDataset.from_dict(rows).cast_column("image", HFImage())
    if "cond_image" in rows:
        ds = ds.cast_column("cond_image", HFImage())
    out = data_dir / "train-00000-of-00001.parquet"
    ds.to_parquet(str(out))
    return output_dir
