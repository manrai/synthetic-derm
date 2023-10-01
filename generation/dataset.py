import argparse
import gc
import hashlib
import itertools
import logging
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, model_info, upload_folder
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
builtins.print = rich.print
monkey_patch()


CLASS_NAMES = [
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


def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs


class FitzpatrickDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        *,
        dataset_type: Literal["fitzpatrick", "ddi"],
        disease_class: str,
        instance_data_root: str,
        instance_prompt: str,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        instance_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        add_fitzpatrick_scale_to_prompt: bool = False,
        split: str = 'train',
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = instance_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance {self.instance_data_root} images root doesn't exists.")

        # Get images
        self.dataset_type = dataset_type
        if self.dataset_type == 'fitzpatrick':
            self.data_disease_class = disease_class.replace('-', ' ')
            self.data_images_root = self.instance_data_root / 'finalfitz17k'
            if disease_class in CLASS_NAMES:
                csv_file = self.instance_data_root / 'fitzpatrick17k_10label_clean_training.csv'
            else:
                csv_file = self.instance_data_root / 'fitzpatrick17k.csv'
            print(f'Using csv file: {csv_file}')
            self.data_df = pd.read_csv(csv_file)
            if self.data_disease_class != 'all':
                self.data_df = self.data_df[self.data_df['label'] == self.data_disease_class]
            self.num_instance_images = len(self.data_df)
            self.add_fitzpatrick_scale_to_prompt = add_fitzpatrick_scale_to_prompt
        elif self.dataset_type == 'ddi':
            self.data_disease_class = disease_class.replace('-', ' ')
            self.data_images_root = self.instance_data_root / 'ddi_images'
            self.data_df = pd.read_csv(self.instance_data_root / 'ddi_training.csv')
            if self.data_disease_class != 'all':
                self.data_df = self.data_df[self.data_df['disease'] == self.data_disease_class.replace(' ', '-')]
            self.num_instance_images = len(self.data_df)
            self.add_fitzpatrick_scale_to_prompt = False
        else:
            raise ValueError(self.dataset_type)

        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num is not None:
                self.num_class_images = min(len(self.class_images_path), class_num)
            else:
                self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

        # Image transforms
        self.split = split
        if self.split == 'train':
            image_transforms = [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.9, 1.1), ratio=(0.9, 1.1), 
                    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.RandomHorizontalFlip(),
            ]
        else: image_transforms = [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
        ]
        normalize_and_to_tensor = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        self.image_transforms = transforms.Compose(image_transforms + normalize_and_to_tensor)

    def __len__(self):
        return self._length

    @lru_cache()
    def tokenize_prompt_with_caching(self, instance_prompt: str):
        return tokenize_prompt(self.tokenizer, instance_prompt, tokenizer_max_length=self.tokenizer_max_length)

    def __getitem__(self, index):
        example = {}

        entry = self.data_df.iloc[index % self.num_instance_images]
        filename = entry.DDI_file if hasattr(entry, "DDI_file") else f'{entry.md5hash}.jpg'
        example['image_name'] = filename.split('.')[0]
        instance_image_path = self.data_images_root / filename
        instance_class_name = entry.disease.replace('-', ' ') if hasattr(entry, 'disease') else entry.label
        if not instance_image_path.is_file():
            raise ValueError(str(instance_image_path))
        instance_image = exif_transpose(Image.open(instance_image_path))
        instance_prompt = self.instance_prompt.format(instance_class_name)
        if self.add_fitzpatrick_scale_to_prompt:
            fitzpatrick_scale = str(entry.fitzpatrick_scale) if entry.fitzpatrick_scale > 0 else 'unknown'
            instance_prompt += f', Fitzpatrick skin type {fitzpatrick_scale}'

        # instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        # instance_image = exif_transpose(instance_image)

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)

        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = self.tokenize_prompt_with_caching(instance_prompt)
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
            example["prompt"] = instance_prompt

        if self.class_data_root:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            example["class_images"] = self.image_transforms(class_image)

            if self.instance_prompt_encoder_hidden_states is not None:
                example["class_prompt_ids"] = self.instance_prompt_encoder_hidden_states
            else:
                class_text_inputs = tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                example["class_prompt_ids"] = class_text_inputs.input_ids
                example["class_attention_mask"] = class_text_inputs.attention_mask

        return example


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base',
        subfolder="tokenizer",
        use_fast=False,
    )
    # instance_data_root = str(Path(os.getenv("DERM_ROOT", "/n/data1/hms/dbmi/manrai/derm")) / 'Fitzpatrick17k')
    # dataset = FitzpatrickDataset(
    #     dataset_type="fitzpatrick",
    #     disease_class="all",
    #     instance_data_root=instance_data_root,
    #     instance_prompt="An image of {}, a skin disease",
    #     tokenizer=tokenizer,
    # )
    instance_data_root = str(Path(os.getenv("DERM_ROOT", "/n/data1/hms/dbmi/manrai/derm")) / 'Stanford_DDI')
    dataset = FitzpatrickDataset(
        dataset_type="ddi",
        disease_class="mycosis-fungoides",
        instance_data_root=instance_data_root,
        instance_prompt="An image of {}, a skin disease",
        tokenizer=tokenizer,
    )
    for i in range(10):
        item = dataset[i]
        print(item)
    
