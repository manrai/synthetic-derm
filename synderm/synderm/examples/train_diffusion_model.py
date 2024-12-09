from functools import lru_cache
from pathlib import Path
from typing import Literal
import gc
import hashlib
import itertools
import logging
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import model_info
from packaging import version
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import builtins
import rich
from lovely_tensors import monkey_patch

from synderm.fine_tune.text_to_image_diffusion import fine_tune_text_to_image

builtins.print = rich.print
monkey_patch()


class CustomDataset(Dataset):
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

class SimpleFitzDataset(Dataset):
    def __init__(self, dataset_dir, fitz_path='fitzpatrick17k_10label_clean_training.csv'):
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

        self.dataset_dir = Path(dataset_dir)

        self.data_images_root = self.dataset_dir / 'finalfitz17k'
        csv_file = self.dataset_dir / fitz_path
        print(f'Using csv file: {csv_file}')

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

# TODO: change functionality so that a new model is trained automatically for each label
# TODO: Create a new wandb log for each model

#simple_dataset = CustomDataset(dataset_dir="sample_dataset", split="train")
simple_dataset = SimpleFitzDataset(dataset_dir="/n/data1/hms/dbmi/manrai/derm/Fitzpatrick17k")

fine_tune_text_to_image(
    train_dataset=simple_dataset,
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base",
    instance_prompt = "An image of {}, a skin disease",
    validation_prompt = "An image of allergic contact dermatitis, a skin disease",
    output_dir = "dreambooth-outputs/allergic-contact-dermatitis",
    #label_filter = "allergic-contact-dermatitis",
    resolution = 512,
    train_batch_size = 4,
    gradient_accumulation_steps = 1,
    learning_rate = 5e-6,
    lr_scheduler = "constant",
    lr_warmup_steps = 0,
    num_train_epochs = 4,
    report_to = "wandb"
)
