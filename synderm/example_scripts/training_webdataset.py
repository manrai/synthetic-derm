from synderm.generation.generate import generate_synthetic_dataset
from synderm.utils.utils import synthetic_train_val_split
from huggingface_hub import get_token
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from huggingface_hub import HfApi
import matplotlib.pyplot as plt
import webdataset as wds
from pathlib import Path
from PIL import Image
import pandas as pd
import os
import json
import io
import re

class SampleDataset(Dataset):
    def __init__(self, dataset_dir, split="train"):
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = []
        self.labels = []
        self.split = split

        # Walk through class folders
        data_dir = self.dataset_dir / self.split
        for class_name in os.listdir(data_dir):
            class_dir = data_dir / class_name
            if not class_dir.is_dir():
                continue
                
            # Get all png images in this class folder
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith('.png'):
                    self.image_paths.append(class_dir / img_name)
                    self.labels.append(class_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load and convert image to RGB
        image = Image.open(image_path).convert('RGB')
        image_name = image_path.stem

        return {"id": image_name, "image": image, "label": label}


train_data = SampleDataset("sample_derm_dataset")

# Change this to a desired directory for caching shards. Shards that are downloaded will populate here
cache_dir = "/n/scratch/users/t/thb286/wds_cache"

# Select finetune-text-to-image shards
url = "https://huggingface.co/datasets/tbuckley/synthetic-derm-1M-train/resolve/main/data/shard-finetune-text-to-image-{00000..00001}.tar"

labels = [
    "basal-cell-carcinoma",
    "allergic-contact-dermatitis",
    "lupus-erythematosus"
]

def select(sample):
    _, metadata = sample
    if metadata["label"] in labels:
        return sample
    else:
        return None

# Create a WebDataset
dataset = (
    wds.WebDataset(url, shardshuffle=True)
    .shuffle(40000)
    .decode("pil")
    .to_tuple("png", "json")
    .map(select)
)

for i, item in enumerate(dataset):
    print(item)
    if i > 10:
        break



train, val = synthetic_train_val_split(
    real_data=train_data,
    synthetic_data=dataset,
    per_class_test_size=5,
    random_state=42,
    mapping_real_to_synthetic="id"
)

