from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import os

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


class FitzDataset(Dataset):
    def __init__(self, images_path, metadata_path="fitzpatrick17k_10label_clean_training.csv"):
        self.class_names = [
            "allergic-contact-dermatitis",
            "basal-cell-carcinoma", 
            "folliculitis",
            "lichen-planus",
            "lupus-erythematosus",
            "neutrophilic-dermatoses",
            "photodermatoses",
            "psoriasis",
            "sarcoidosis",
            "squamous-cell-carcinoma",
        ]

        self.data_images_root = Path(images_path)
        csv_file = Path(metadata_path)
        print(f'Using Fitz metadata file: {csv_file}')

        self.data_df = pd.read_csv(csv_file)
        self.data_df["label"] = self.data_df["label"].apply(lambda x: x.replace(" ", "-"))

        # Filter to only include entries with labels in class_names
        self.data_df = self.data_df[self.data_df['label'].isin(self.class_names)]
        self.num_instance_images = len(self.data_df)

    def __len__(self):
        return self.data_df.shape[0]

    def __getitem__(self, index):
        example = {}

        entry = self.data_df.iloc[index]
        example['id'] = entry['md5hash']
        example['label'] = entry['label']

        filename = f"{entry['md5hash']}.jpg"

        instance_image_path = self.data_images_root / filename
        example["image"] = Image.open(instance_image_path).convert('RGB')

        return example

