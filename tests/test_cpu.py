"""CPU-only tests of the data, split, export and metrics code.

Tests that need the Atlas Dermatologico sample are skipped unless it is already cached
(``load_sample(download=False)``), so the suite never fetches data on its own.
"""
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import Subset

from synderm.backends import masks, prompts
from synderm.classify import ClassificationDataset, accuracy_ci, build_transforms
from synderm.data import ImageFolderDataset, export_training_parquet, normalize_label
from synderm.data.datasets import label_of
from synderm.utils import synthetic_train_val_split


def test_normalize_label():
    assert normalize_label("Basal Cell Carcinoma") == "basal-cell-carcinoma"
    assert normalize_label("lichen_planus") == "lichen-planus"
    assert prompts.caption("basal-cell-carcinoma") == "An image of basal cell carcinoma, a skin disease"


def test_masks_are_complements():
    c, b = masks.center_mask((100, 80), 0.5), masks.border_mask((100, 80), 0.5)
    assert c.getpixel((50, 40)) == 255 and b.getpixel((50, 40)) == 0
    assert c.getpixel((1, 1)) == 0 and b.getpixel((1, 1)) == 255


def test_accuracy_ci():
    acc, lo, hi = accuracy_ci(80, 100)
    assert acc == 0.8 and lo < 0.8 < hi


@pytest.fixture
def folder_dataset(tmp_path):
    for label in ["a-label", "b-label"]:
        d = tmp_path / label
        d.mkdir()
        for i in range(4):
            Image.new("RGB", (32, 32), (i * 50, 0, 0)).save(d / f"{label}-{i}.png")
    return ImageFolderDataset(tmp_path)


def test_image_folder_and_label_of(folder_dataset):
    assert len(folder_dataset) == 8 and folder_dataset.labels == ["a-label", "b-label"]
    item = folder_dataset[0]
    assert set(item) >= {"id", "image", "label"}
    assert label_of(Subset(folder_dataset, [5]), 0) == "b-label"


def test_export_parquet_t2i_and_i2i(folder_dataset, tmp_path):
    from datasets import load_dataset
    out = export_training_parquet(folder_dataset, tmp_path / "t2i", lambda it: prompts.caption(it["label"]))
    ds = load_dataset(str(out))["train"]
    assert len(ds) == 8 and ds.column_names == ["image", "caption", "id", "label"]
    out2 = export_training_parquet(folder_dataset, tmp_path / "i2i", lambda it: prompts.caption(it["label"]),
                                   mode="image-to-image", pairs_per_image=2)
    ds2 = load_dataset(str(out2))["train"]
    assert len(ds2) == 16 and "cond_image" in ds2.column_names
    assert ds2[0]["id"] != ds2[0]["cond_id"]


def test_split_policies(folder_dataset, tmp_path):
    from synderm.data import SyntheticDataset
    import json
    # synthetic images referencing the real ids
    root = tmp_path / "syn"
    for i in range(8):
        real = folder_dataset[i]
        d = root / real["label"]
        d.mkdir(parents=True, exist_ok=True)
        for k in range(3):
            sid = f"{real['id']}-text-to-image-{k:03d}"
            real["image"].save(d / f"{sid}.png")
            (d / f"{sid}.json").write_text(json.dumps({"id": sid, "label": real["label"], "ref_id": real["id"],
                                                       "generator": "test", "method": "text-to-image"}))
    syn = SyntheticDataset(root)
    assert len(syn) == 24
    train, val = synthetic_train_val_split(folder_dataset, syn, per_class_test_size=1, n_synthetic_per_real=2, random_state=0)
    # 3 real per class in train, 2 synthetic per real -> 6 real + 12 synthetic
    assert len(val) == 2 and len(train) == 6 + 12
    train2, _ = synthetic_train_val_split(folder_dataset, syn, per_class_test_size=1, random_state=0,
                                          synthetic_policy="ref-not-in-val")
    # exclusion policy keeps everything not derived from the 2 validation images: 6 real + 18 synthetic
    assert len(train2) == 6 + 18
    x, y = ClassificationDataset(train, folder_dataset.labels, build_transforms(augment=True))[0]
    assert tuple(x.shape) == (3, 224, 224) and y in (0, 1)


def test_atlas_sample_if_cached():
    from synderm.data import load_sample
    try:
        ds = load_sample(split="val", download=False)
    except FileNotFoundError:
        pytest.skip("Atlas sample not cached")
    assert len(ds) == 192 and len(ds.labels) == 6
    item = ds[0]
    assert item["image"].mode == "RGB" and 1 <= item["fitzpatrick_scale"] <= 6 or item["fitzpatrick_scale"] == -1


def test_distances():
    import numpy as np
    from synderm.eval import frechet_distance, kernel_distance
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(300, 16)), rng.normal(size=(300, 16))
    c = rng.normal(loc=2.0, size=(300, 16))
    assert frechet_distance(a, b) < frechet_distance(a, c)
    assert kernel_distance(a, b, subsets=5, subset_size=100)[0] < kernel_distance(a, c, subsets=5, subset_size=100)[0]
