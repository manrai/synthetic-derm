"""Real/synthetic train-validation splitting that follows the paper's protocol."""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

from torch.utils.data import ConcatDataset, Dataset, Subset

from synderm.data.datasets import label_of


def split_by_label(dataset: Dataset, per_label: int, random_state: Optional[int] = None,
                   labels: Optional[list[str]] = None) -> tuple[Subset, Subset]:
    """Take ``per_label`` items of each label into the first subset, the rest into the second."""
    rng = random.Random(random_state)
    by = defaultdict(list)
    for i in range(len(dataset)):
        by[label_of(dataset, i)].append(i)
    first, rest = [], []
    for lab in sorted(by):
        if labels and lab not in labels:
            continue
        idx = by[lab]
        rng.shuffle(idx)
        first.extend(idx[:per_label])
        rest.extend(idx[per_label:])
    return Subset(dataset, sorted(first)), Subset(dataset, sorted(rest))


def synthetic_train_val_split(
    real_data: Dataset,
    synthetic_data: Optional[Dataset] = None,
    per_class_test_size: int = 40,
    n_real_per_class: Optional[int] = None,
    n_synthetic_per_real: Optional[int] = None,
    random_state: Optional[int] = None,
    mapping_real_to_synthetic: str = "ref_id",
    synthetic_policy: str = "ref-in-train",
    exclude_ref_ids: Optional[set] = None,
) -> tuple[Dataset, Dataset]:
    """Split real images into train and validation per class, then attach synthetic images.

    Synthetic images are included only when their ``ref_id`` (the real image they were
    generated from) is in the real training subset, so nothing derived from a validation
    image leaks into training. ``n_real_per_class`` caps real training images per class (the
    paper's 1, 8, 16, ... 228 sweep); ``n_synthetic_per_real`` caps synthetic images per real
    training image (the dose-response sweep). Synthetic images without a ``ref_id`` (label-only
    text-to-image) are included for every label present in training, capped by
    ``n_synthetic_per_real`` times the number of real training images of that label.

    ``synthetic_policy`` selects how synthetic images with a ``ref_id`` are admitted:
    ``"ref-in-train"`` (the paper's protocol) keeps only those generated from a real training
    image; ``"ref-not-in-val"`` keeps every synthetic image of a training label except those
    generated from a validation image or from ``exclude_ref_ids`` (use this with the public
    2024 shards, whose source images are mostly outside your sample). Per-label caps apply in
    both cases.
    """
    if synthetic_policy not in ("ref-in-train", "ref-not-in-val"):
        raise ValueError("synthetic_policy must be 'ref-in-train' or 'ref-not-in-val'")
    rng = random.Random(random_state)
    by = defaultdict(list)
    for i in range(len(real_data)):
        by[label_of(real_data, i)].append(i)
    val_idx, train_idx = [], []
    for lab in sorted(by):
        idx = list(by[lab])
        rng.shuffle(idx)
        n_val = min(per_class_test_size, len(idx))
        val_idx.extend(idx[:n_val])
        remaining = idx[n_val:]
        if n_real_per_class is not None:
            remaining = remaining[:n_real_per_class]
        train_idx.extend(remaining)
    train_real = Subset(real_data, sorted(train_idx))
    val = Subset(real_data, sorted(val_idx))
    if synthetic_data is None:
        return train_real, val

    train_ids_by_label: dict[str, set] = defaultdict(set)
    for i in train_idx:
        item = real_data[i] if not hasattr(real_data, "df") else None
        rid = real_data.df.iloc[i]["md5hash"] if item is None else item["id"]
        train_ids_by_label[label_of(real_data, i)].add(rid)
    train_ids = set().union(*train_ids_by_label.values()) if train_ids_by_label else set()
    val_ids = set()
    for i in val_idx:
        val_ids.add(real_data.df.iloc[i]["md5hash"] if hasattr(real_data, "df") else real_data[i]["id"])
    banned = val_ids | (exclude_ref_ids or set())

    per_ref: dict = defaultdict(int)
    per_label_unref: dict = defaultdict(int)
    syn_idx = []
    metas = synthetic_data.items if hasattr(synthetic_data, "items") else None
    for j in range(len(synthetic_data)):
        meta = metas[j][1] if metas is not None else {k: v for k, v in synthetic_data[j].items() if k != "image"}
        lab = label_of(synthetic_data, j)
        ref = meta.get(mapping_real_to_synthetic)
        if ref and synthetic_policy == "ref-in-train":
            if ref not in train_ids:
                continue
            if n_synthetic_per_real is not None and per_ref[ref] >= n_synthetic_per_real:
                continue
            per_ref[ref] += 1
        elif ref and synthetic_policy == "ref-not-in-val":
            if ref in banned or lab not in train_ids_by_label:
                continue
            cap = None if n_synthetic_per_real is None else n_synthetic_per_real * len(train_ids_by_label[lab])
            if cap is not None and per_label_unref[lab] >= cap:
                continue
            per_label_unref[lab] += 1
        else:
            if lab not in train_ids_by_label:
                continue
            cap = None if n_synthetic_per_real is None else n_synthetic_per_real * len(train_ids_by_label[lab])
            if cap is not None and per_label_unref[lab] >= cap:
                continue
            per_label_unref[lab] += 1
        syn_idx.append(j)
    return ConcatDataset([train_real, Subset(synthetic_data, syn_idx)]), val
