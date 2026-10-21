"""Fetch the Atlas Dermatologico portion of Fitzpatrick 17k directly from the source.

Fitzpatrick 17k ships as a CSV of URLs. As of October 2026 the dermaamin.com links are dead but
the atlasdermatologico.com.br links still resolve, and the site's robots.txt permits fetching
image URLs. :func:`load_sample` builds a fixed, reproducible six-label sample (64 train + 32
validation per label) from that portion and verifies every file against the md5 hash in the
CSV. There is deliberately no fallback: if the source is unreachable or a file does not verify,
an :class:`AtlasDownloadError` names the failed URLs so that the dataset becoming unavailable is
noticed rather than silently replaced.

Images are CC BY-NC-SA 3.0 (Atlas Dermatologico, via Groh et al., Fitzpatrick 17k).
"""
from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import requests

from .datasets import FitzpatrickDataset, normalize_label

FITZ_METADATA_DIR = Path(__file__).resolve().parents[2] / "fitz_metadata"
FULL_CSV = FITZ_METADATA_DIR / "fitzpatrick17k.csv"
HELD_OUT_CSV = FITZ_METADATA_DIR / "fitzpatrick17k_10label_clean_held_out_set.csv"
CLEAN_TRAIN_CSV = FITZ_METADATA_DIR / "fitzpatrick17k_10label_clean_training.csv"

ATLAS_HOST = "atlasdermatologico.com.br"
USER_AGENT = "synderm/0.2 (research; https://github.com/manrai/synthetic-derm)"

#: The six Fitzpatrick 17k labels with at least 96 Atlas Dermatologico images outside the held-out set.
SAMPLE_LABELS = (
    "basal-cell-carcinoma",
    "lichen-planus",
    "squamous-cell-carcinoma",
    "lupus-erythematosus",
    "psoriasis",
    "folliculitis",
)
SAMPLE_TRAIN_PER_LABEL = 64
SAMPLE_VAL_PER_LABEL = 32
SAMPLE_SEED = 20261021


class AtlasDownloadError(RuntimeError):
    """Raised when any Atlas Dermatologico image cannot be fetched or does not verify."""


def default_data_root() -> Path:
    return Path(os.environ.get("SYNDERM_DATA_DIR", "/workspace/data"))


def atlas_rows(exclude_held_out: bool = True) -> pd.DataFrame:
    """All Fitzpatrick 17k rows hosted on Atlas Dermatologico, labels normalized."""
    df = pd.read_csv(FULL_CSV)
    df["label"] = df["label"].map(normalize_label)
    df = df[df["url"].str.contains(ATLAS_HOST, na=False)]
    if exclude_held_out:
        held = set(pd.read_csv(HELD_OUT_CSV)["md5hash"])
        df = df[~df["md5hash"].isin(held)]
    return df.reset_index(drop=True)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_atlas_images(
    rows: pd.DataFrame,
    images_dir: str | os.PathLike,
    delay_s: float = 0.5,
    timeout_s: float = 30.0,
    retries: int = 3,
    verify_md5: bool = True,
    progress: bool = True,
) -> Path:
    """Download every row's image to ``<images_dir>/<md5hash>.jpg`` and verify it.

    Already-present files that verify are skipped. Raises :class:`AtlasDownloadError` listing
    every URL that failed after ``retries`` attempts or whose md5 did not match the CSV.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    failures: list[tuple[str, str]] = []
    todo = [(r.md5hash, r.url) for r in rows.itertuples()]
    iterator = todo
    if progress:
        from tqdm.auto import tqdm
        iterator = tqdm(todo, desc=f"Atlas Dermatologico -> {images_dir}", unit="img")
    for md5hash, url in iterator:
        dest = images_dir / f"{md5hash}.jpg"
        if dest.exists() and (not verify_md5 or _md5(dest) == md5hash):
            continue
        last_err = "unknown"
        ok = False
        for attempt in range(retries):
            try:
                r = session.get(url, timeout=timeout_s)
                if r.status_code != 200 or not r.headers.get("content-type", "").startswith("image"):
                    last_err = f"HTTP {r.status_code} content-type={r.headers.get('content-type')}"
                else:
                    dest.write_bytes(r.content)
                    if verify_md5 and _md5(dest) != md5hash:
                        last_err = f"md5 mismatch (got {_md5(dest)})"
                        dest.unlink(missing_ok=True)
                    else:
                        ok = True
                        break
            except requests.RequestException as e:
                last_err = f"{type(e).__name__}: {e}"
            time.sleep(delay_s * (attempt + 1))
        if not ok:
            failures.append((url, last_err))
        time.sleep(delay_s)
    if failures:
        listing = "\n".join(f"  {u}  ->  {err}" for u, err in failures[:25])
        more = f"\n  ... and {len(failures) - 25} more" if len(failures) > 25 else ""
        raise AtlasDownloadError(
            f"{len(failures)} of {len(todo)} Atlas Dermatologico images could not be fetched or verified. "
            f"No fallback is attempted on purpose; check whether {ATLAS_HOST} is still serving images.\n{listing}{more}"
        )
    return images_dir


def sample_rows(
    labels: Sequence[str] = SAMPLE_LABELS,
    train_per_label: int = SAMPLE_TRAIN_PER_LABEL,
    val_per_label: int = SAMPLE_VAL_PER_LABEL,
    seed: int = SAMPLE_SEED,
) -> pd.DataFrame:
    """Deterministic per-label sample of Atlas rows with a ``split`` column (train/val)."""
    df = atlas_rows(exclude_held_out=True)
    clean_train = set(pd.read_csv(CLEAN_TRAIN_CSV)["md5hash"])
    rng = random.Random(seed)
    parts = []
    for label in labels:
        label = normalize_label(label)
        rows = df[df["label"] == label].copy()
        need = train_per_label + val_per_label
        if len(rows) < need:
            raise ValueError(f"Label {label!r} has only {len(rows)} Atlas images outside the held-out set; need {need}")
        # Prefer rows that passed the paper's SelfClean filtering, then fill from the rest.
        rows["clean"] = rows["md5hash"].isin(clean_train)
        clean = rows[rows["clean"]].sort_values("md5hash")
        rest = rows[~rows["clean"]].sort_values("md5hash")
        order = list(clean["md5hash"]) 
        rng.shuffle(order)
        extra = list(rest["md5hash"])
        rng.shuffle(extra)
        chosen = (order + extra)[:need]
        picked = rows.set_index("md5hash").loc[chosen].reset_index()
        picked["split"] = ["train"] * train_per_label + ["val"] * val_per_label
        parts.append(picked)
    return pd.concat(parts, ignore_index=True)


def load_sample(
    data_root: str | os.PathLike | None = None,
    split: str = "train",
    labels: Sequence[str] = SAMPLE_LABELS,
    train_per_label: int = SAMPLE_TRAIN_PER_LABEL,
    val_per_label: int = SAMPLE_VAL_PER_LABEL,
    seed: int = SAMPLE_SEED,
    download: bool = True,
) -> FitzpatrickDataset:
    """The vignette sample: real Fitzpatrick 17k images from Atlas Dermatologico.

    Images are cached under ``<data_root>/fitzpatrick17k-atlas/images`` (default data root
    ``/workspace/data``, override with ``SYNDERM_DATA_DIR``) and the sample's metadata CSV is
    written next to them. ``split`` is ``"train"`` (64 per label) or ``"val"`` (32 per label).
    """
    root = Path(data_root) if data_root else default_data_root()
    base = root / "fitzpatrick17k-atlas"
    images_dir = base / "images"
    rows = sample_rows(labels, train_per_label, val_per_label, seed)
    csv_path = base / f"sample-{len(labels)}labels-{train_per_label}-{val_per_label}-seed{seed}.csv"
    base.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False)
    if download:
        fetch_atlas_images(rows, images_dir)
    if split not in ("train", "val"):
        raise ValueError("split must be 'train' or 'val'")
    split_csv = csv_path.with_name(csv_path.stem + f"-{split}.csv")
    rows[rows["split"] == split].to_csv(split_csv, index=False)
    return FitzpatrickDataset(images_dir, split_csv, require_all=True)


def fetch_all_atlas(data_root: str | os.PathLike | None = None, exclude_held_out: bool = False) -> Path:
    """Fetch every Atlas Dermatologico image in Fitzpatrick 17k (about 3,900 files, ~230 MB)."""
    root = Path(data_root) if data_root else default_data_root()
    return fetch_atlas_images(atlas_rows(exclude_held_out=exclude_held_out), root / "fitzpatrick17k-atlas" / "images")
