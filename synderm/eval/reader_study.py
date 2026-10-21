"""Export a stratified image sample for a physician reader study in the paper's format."""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd
from torch.utils.data import Dataset

from synderm.data.datasets import label_of, normalize_label


def export_reader_study(
    sources: dict[str, Dataset],
    output_dir: str | os.PathLike,
    labels: Sequence[str],
    per_label_per_source: int = 7,
    seed: int = 0,
) -> pd.DataFrame:
    """Copy a stratified random sample into ``output_dir/images`` under anonymized names.

    ``sources`` maps a source name (e.g. ``"real"``, ``"sdxl-inpaint-outpaint"``,
    ``"qwen-text-to-image"``) to a dataset. The paper used 7 images per label per source for
    three sources and nine labels (189 images). Writes ``survey.csv`` (anonymized id, label,
    the two questions as empty columns) and ``answer_key.csv`` (source and original id).
    """
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for source, ds in sources.items():
        by_label: dict[str, list[int]] = {}
        for i in range(len(ds)):
            by_label.setdefault(label_of(ds, i), []).append(i)
        for label in labels:
            label = normalize_label(label)
            pool = by_label.get(label, [])
            if len(pool) < per_label_per_source:
                raise ValueError(f"Source {source!r} has {len(pool)} images for {label!r}; need {per_label_per_source}")
            for i in rng.sample(pool, per_label_per_source):
                item = ds[i]
                rows.append({"source": source, "label": label, "original_id": item["id"], "image": item["image"]})
    rng.shuffle(rows)
    key, survey = [], []
    for n, r in enumerate(rows):
        anon = f"img_{n:04d}"
        r["image"].convert("RGB").save(img_dir / f"{anon}.png")
        key.append({"anon_id": anon, "source": r["source"], "label": r["label"], "original_id": r["original_id"]})
        survey.append({"anon_id": anon, "label": r["label"].replace("-", " "),
                       "q1_machine_generated (Yes/No/Unclear)": "", "q2_consistent_with_label (Yes/No/Unclear)": ""})
    pd.DataFrame(key).to_csv(output_dir / "answer_key.csv", index=False)
    df = pd.DataFrame(survey)
    df.to_csv(output_dir / "survey.csv", index=False)
    return df
