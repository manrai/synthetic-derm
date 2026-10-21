from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms as T

from synderm.data.datasets import normalize_label

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 224, augment: bool = False):
    """Approximation of fastai's ``aug_transforms`` (flip, crop/zoom, rotation, lighting)."""
    if augment:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.02, 0.02)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ClassificationDataset(Dataset):
    """Wrap any synderm dataset (``id``, ``image``, ``label`` items) as ``(tensor, class_index)``.

    ``classes`` fixes the label order so train, validation and test share one index space.
    ``group_key`` (e.g. ``"fitzpatrick_scale"``) is returned as a third element when set.
    """

    def __init__(self, dataset: Dataset, classes: Sequence[str], transform=None, group_key: Optional[str] = None):
        self.dataset = dataset
        self.classes = [normalize_label(c) for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform or build_transforms()
        self.group_key = group_key

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        label = normalize_label(item["label"])
        if label not in self.class_to_idx:
            raise KeyError(f"Label {label!r} is not in classes {self.classes}")
        x = self.transform(item["image"].convert("RGB"))
        y = self.class_to_idx[label]
        if self.group_key:
            return x, y, item.get(self.group_key, -1)
        return x, y


def mix_datasets(*datasets: Dataset) -> Dataset:
    """Concatenate real and synthetic datasets for training (a plain ConcatDataset)."""
    return ConcatDataset(list(datasets))
