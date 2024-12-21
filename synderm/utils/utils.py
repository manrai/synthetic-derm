from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import os
import random
from torch.utils.data import Subset, ConcatDataset
from collections import defaultdict

def save_images(images, directory):
    pass

def load_images(directory):
    pass


def count_png_files(directory_path):
    """
    Count the total number of PNG files in a directory and its subdirectories
    
    Args:
        directory_path (str): Path to the directory to search
        
    Returns:
        int: Total number of PNG files found
    """
    total_png = 0
    for root, dirs, files in os.walk(directory_path, followlinks=True):
        png_files = [f for f in files if f.endswith('.png')]
        total_png += len(png_files)
    return total_png

def save(image, path):
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(exist_ok=True, parents=True)
    image.save(path)

def synthetic_train_val_split(
        real_data: Dataset, 
        synthetic_data: Dataset | None = None,
        test_size: float | int | None = None,
        per_class_test_size: int | None = 40,
        n_real_per_class: int | None = None,
        n_synthetic_per_class: int | None = None,
        random_state: int | None = None,
        class_column: str = "label",
        mapping_real_to_synthetic: str = "id"
    ):
    """
    Splits the combined dataset of real and synthetic images into training and validation sets.
    The training set consists of synthetic images and a limited number of real images per class.
    In the training set, synthetic entries are included only if they have a corresponding entry in the training set.
    The validation set consists only of real images.

    # TODO: adjust this logic to instead exclude synthetic images that have a corresponding version in the validation set

    # TODO: add n_synthetic_per_class logic

    Parameters:
    -----------
    real_data : torch.utils.data.Dataset
        Dataset containing the real images data. Each item should be a dict with at least 'id' and 'label'.
    synthetic_data : torch.utils.data.Dataset or None, default=None
        Dataset containing the synthetic images data. If provided, it should have a field that maps to real data 'id'.
    test_size : float, int, or None, default=None
        If float, represents the proportion of the real dataset to include in the validation split.
        If int, represents the total number of validation samples.
        If None, `per_class_test_size` must be specified.
    per_class_test_size : int or None, default=40
        Number of samples per class to include in the validation set.
        If None, `test_size` must be specified.
    n_real_per_class : int or None, default=None
        Maximum number of real images per class to include in the training data.
        If None, all remaining real images (excluding validation set) are included.
    n_synthetic_per_class : int or None, default=None
        (Not implemented) Raises NotImplementedError if provided.
    random_state : int or None, default=None
        Random seed for reproducibility.
    class_column : str, default="label"
        Name of the field in the dataset that contains the class labels.
    mapping_real_to_synthetic : str, default="id"
        Name of the field in the synthetic dataset that maps to the real data's 'id'.

    Returns:
    --------
    train_data : torch.utils.data.Dataset
        Dataset containing the training data (real and synthetic).
    val_data : torch.utils.data.Dataset
        Dataset containing the validation data.
    """
    if (test_size is None and per_class_test_size is None) or (test_size is not None and per_class_test_size is not None):
        raise ValueError("Specify exactly one of 'test_size' or 'per_class_test_size'.")

    if n_synthetic_per_class is not None:
        raise NotImplementedError("'n_synthetic_per_class' parameter is not implemented.")

    # Set random seed for reproducibility
    rng = random.Random(random_state)

    # Extract real data entries
    real_entries = [{'index': idx, 'id': real_data[idx]['id'], 'label': real_data[idx][class_column]} for idx in range(len(real_data))]

    # Group real data indices by class
    class_to_indices = defaultdict(list)
    for entry in real_entries:
        class_to_indices[entry['label']].append(entry['index'])

    val_indices = []
    train_real_indices = []

    if per_class_test_size is not None:
        # Sample specified number per class for validation
        for cls, indices in class_to_indices.items():
            n_samples = min(len(indices), per_class_test_size)
            sampled_indices = rng.sample(indices, n_samples)
            val_indices.extend(sampled_indices)
            remaining = list(set(indices) - set(sampled_indices))
            train_real_indices.extend(remaining)
    else:
        # Compute per-class validation sizes based on test_size
        if isinstance(test_size, float):
            if not 0.0 < test_size < 1.0:
                raise ValueError("When 'test_size' is a float, it must be between 0.0 and 1.0.")
            for cls, indices in class_to_indices.items():
                n_samples = max(1, int(round(len(indices) * test_size)))
                sampled_indices = rng.sample(indices, n_samples)
                val_indices.extend(sampled_indices)
                remaining = list(set(indices) - set(sampled_indices))
                train_real_indices.extend(remaining)
        elif isinstance(test_size, int):
            total_samples = test_size
            num_classes = len(class_to_indices)
            samples_per_class = total_samples // num_classes
            remainder = total_samples % num_classes
            for idx, (cls, indices) in enumerate(class_to_indices.items()):
                n_samples = samples_per_class + (1 if idx < remainder else 0)
                n_samples = min(len(indices), n_samples)
                sampled_indices = rng.sample(indices, n_samples)
                val_indices.extend(sampled_indices)
                remaining = list(set(indices) - set(sampled_indices))
                train_real_indices.extend(remaining)
        else:
            raise ValueError("'test_size' must be a float or int.")

    # If n_real_per_class is specified, limit the number of real training samples per class
    if n_real_per_class is not None:
        limited_train_real_indices = []
        # Regroup training indices by class
        train_class_to_indices = defaultdict(list)
        for idx in train_real_indices:
            cls = real_data[idx][class_column]
            train_class_to_indices[cls].append(idx)
        for cls, indices in train_class_to_indices.items():
            n_samples = min(len(indices), n_real_per_class)
            sampled_indices = rng.sample(indices, n_samples)
            limited_train_real_indices.extend(sampled_indices)
        train_real_indices = limited_train_real_indices

    # Process synthetic data
    synthetic_train_indices = []
    if synthetic_data is not None:
        # Extract real training ids for mapping
        train_real_ids = set([real_data[idx]['id'] for idx in train_real_indices])
        
        # Iterate over synthetic_data to find matching entries
        for idx in range(len(synthetic_data)):
            synthetic_entry = synthetic_data[idx]
            mapped_id = synthetic_entry.get(mapping_real_to_synthetic)
            if mapped_id in train_real_ids:
                synthetic_train_indices.append(idx)
        
        train_synthetic_subset = Subset(synthetic_data, synthetic_train_indices)
    else:
        train_synthetic_subset = None

    # Create Subsets for training and validation
    train_real_subset = Subset(real_data, train_real_indices)
    val_subset = Subset(real_data, val_indices)

    if synthetic_data is not None:
        train_data = ConcatDataset([train_real_subset, train_synthetic_subset])
    else:
        train_data = train_real_subset

    val_data = val_subset

    return train_data, val_data


