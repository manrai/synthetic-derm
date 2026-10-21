"""Torch datasets that follow the synderm item convention (``id``, ``image``, ``label``)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def normalize_label(label: str) -> str:
    """Return the canonical label form used throughout synderm: lowercase, hyphen-separated."""
    return "-".join(str(label).strip().lower().replace("_", " ").replace("-", " ").split())


def label_words(label: str) -> str:
    """``basal-cell-carcinoma`` -> ``basal cell carcinoma`` for use in prompts."""
    return normalize_label(label).replace("-", " ")


class ImageFolderDataset(Dataset):
    """A directory of ``<root>/<label>/<image>`` files.

    ``id`` is the file stem. Labels are normalized with :func:`normalize_label`.
    """

    def __init__(self, root: str | os.PathLike, labels: Optional[Sequence[str]] = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"ImageFolderDataset root does not exist: {self.root}")
        keep = {normalize_label(l) for l in labels} if labels else None
        self.items: list[tuple[Path, str]] = []
        for label_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label = normalize_label(label_dir.name)
            if keep is not None and label not in keep:
                continue
            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() in IMAGE_EXTENSIONS:
                    self.items.append((f, label))
        if not self.items:
            raise ValueError(f"No images found under {self.root}")

    @property
    def labels(self) -> list[str]:
        return sorted({l for _, l in self.items})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        path, label = self.items[idx]
        return {"id": path.stem, "image": Image.open(path).convert("RGB"), "label": label, "path": str(path)}


class FitzpatrickDataset(Dataset):
    """Fitzpatrick 17k images named ``<md5hash>.jpg`` plus one of the metadata CSVs in ``fitz_metadata/``.

    Items carry ``fitzpatrick_scale`` (1-6, or -1 when unknown) and ``url`` in addition to the
    standard keys. Rows whose image file is missing raise at construction time unless
    ``require_all=False``.
    """

    def __init__(
        self,
        images_dir: str | os.PathLike,
        metadata_csv: str | os.PathLike,
        labels: Optional[Sequence[str]] = None,
        require_all: bool = True,
    ):
        self.images_dir = Path(images_dir)
        df = pd.read_csv(metadata_csv)
        df["label"] = df["label"].map(normalize_label)
        if labels:
            keep = {normalize_label(l) for l in labels}
            df = df[df["label"].isin(keep)]
        df = df.reset_index(drop=True)
        paths = [self.images_dir / f"{h}.jpg" for h in df["md5hash"]]
        missing = [p for p in paths if not p.exists()]
        if missing and require_all:
            raise FileNotFoundError(
                f"{len(missing)} of {len(paths)} Fitzpatrick 17k images are missing under {self.images_dir}. "
                f"First missing: {missing[0]}"
            )
        if missing:
            exists = [p.exists() for p in paths]
            df = df[exists].reset_index(drop=True)
        self.df = df

    @property
    def labels(self) -> list[str]:
        return sorted(self.df["label"].unique())

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        path = self.images_dir / f"{row['md5hash']}.jpg"
        return {
            "id": row["md5hash"],
            "image": Image.open(path).convert("RGB"),
            "label": row["label"],
            "fitzpatrick_scale": int(row.get("fitzpatrick_scale", -1)),
            "url": row.get("url", ""),
            "path": str(path),
        }


class SyntheticDataset(Dataset):
    """Images written by :func:`synderm.generation.generate_synthetic_dataset`.

    Layout: ``<root>/<label>/<id>.png`` with a sibling ``<id>.json`` holding the generation
    metadata (``generator``, ``method``, ``ref_id``, ``prompt``, ``seed`` ...). Items expose the
    JSON fields; ``ref_id`` maps a synthetic image back to the real image it was conditioned on,
    which :func:`synderm.utils.synthetic_train_val_split` uses to keep validation images out of
    the training set.
    """

    def __init__(self, root: str | os.PathLike, labels: Optional[Sequence[str]] = None,
                 methods: Optional[Sequence[str]] = None, generators: Optional[Sequence[str]] = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"SyntheticDataset root does not exist: {self.root}")
        keep_labels = {normalize_label(l) for l in labels} if labels else None
        self.items: list[tuple[Path, dict]] = []
        for png in sorted(self.root.rglob("*.png")):
            meta_path = png.with_suffix(".json")
            if not meta_path.exists():
                raise FileNotFoundError(f"Synthetic image {png} has no metadata JSON next to it")
            meta = json.loads(meta_path.read_text())
            if keep_labels is not None and normalize_label(meta["label"]) not in keep_labels:
                continue
            if methods and meta.get("method") not in methods:
                continue
            if generators and meta.get("generator") not in generators:
                continue
            self.items.append((png, meta))
        if not self.items:
            raise ValueError(f"No synthetic images matched under {self.root}")

    @property
    def labels(self) -> list[str]:
        return sorted({normalize_label(m["label"]) for _, m in self.items})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        path, meta = self.items[idx]
        item = dict(meta)
        item["label"] = normalize_label(meta["label"])
        item["id"] = meta.get("id", path.stem)
        item["image"] = Image.open(path).convert("RGB")
        item["path"] = str(path)
        item["synthetic"] = True
        return item


class LabelFilter(Dataset):
    """View of a dataset restricted to one or more labels (indexes computed once)."""

    def __init__(self, dataset: Dataset, labels: Iterable[str], label_fn: Callable[[dict], str] | None = None):
        keep = {normalize_label(l) for l in labels}
        self.dataset = dataset
        if hasattr(dataset, "items"):
            self.indices = [i for i, (_, lab) in enumerate(dataset.items)
                            if normalize_label(lab if isinstance(lab, str) else lab["label"]) in keep]
        elif hasattr(dataset, "df"):
            self.indices = [i for i, lab in enumerate(dataset.df["label"]) if normalize_label(lab) in keep]
        else:
            fn = label_fn or (lambda item: item["label"])
            self.indices = [i for i in range(len(dataset)) if normalize_label(fn(dataset[i])) in keep]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]


def label_of(dataset: Dataset, idx: int) -> str:
    """Label of item ``idx`` without decoding the image when the dataset allows it."""
    from torch.utils.data import Subset, ConcatDataset
    if isinstance(dataset, Subset):
        return label_of(dataset.dataset, dataset.indices[idx])
    if isinstance(dataset, ConcatDataset):
        import bisect
        d = bisect.bisect_right(dataset.cumulative_sizes, idx)
        inner = idx if d == 0 else idx - dataset.cumulative_sizes[d - 1]
        return label_of(dataset.datasets[d], inner)
    if hasattr(dataset, "items"):
        lab = dataset.items[idx][1]
        return normalize_label(lab if isinstance(lab, str) else lab["label"])
    if hasattr(dataset, "df"):
        return normalize_label(dataset.df.iloc[idx]["label"])
    if isinstance(dataset, LabelFilter):
        return label_of(dataset.dataset, dataset.indices[idx])
    return normalize_label(dataset[idx]["label"])
