"""Datasets and data access for synderm.

Every dataset in synderm yields dictionaries with at least three keys:

- ``id``: a string that uniquely identifies the image (for Fitzpatrick 17k, the md5 hash)
- ``image``: a ``PIL.Image.Image`` in RGB mode
- ``label``: the class label as a lowercase, hyphen-separated string, e.g. ``basal-cell-carcinoma``

Synthetic images carry extra keys (``generator``, ``method``, ``ref_id``, ``prompt``) that mirror
the per-image JSON stored in the Hugging Face WebDataset shards.
"""
from .datasets import ImageFolderDataset, FitzpatrickDataset, SyntheticDataset, LabelFilter, normalize_label
from .atlas import load_sample, fetch_atlas_images, SAMPLE_LABELS, AtlasDownloadError
from .export import export_training_parquet
from .hub import load_2024_shards, list_2024_shards, REPO_1M

__all__ = [
    "ImageFolderDataset", "FitzpatrickDataset", "SyntheticDataset", "LabelFilter", "normalize_label",
    "load_sample", "fetch_atlas_images", "SAMPLE_LABELS", "AtlasDownloadError",
    "export_training_parquet", "load_2024_shards", "list_2024_shards", "REPO_1M",
]
