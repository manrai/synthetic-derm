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

builtins.print = rich.print
monkey_patch()

from utils.helpers import tokenize_prompt



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


# TODO: create a very simple pytorch dataset with a train/test split
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



class DiffusionTrainWrapper(Dataset):
    def __init__(
        self,
        train_dataset: Dataset,
        tokenizer,
        instance_prompt: str,
        label_filter=None,
        size=512,
        center_crop=False,
        encoder_hidden_states=None,
        instance_prompt_encoder_hidden_states=None,
        tokenizer_max_length=None,
        add_fitzpatrick_scale_to_prompt: bool = False
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.instance_prompt_encoder_hidden_states = instance_prompt_encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length
        self.label_filter = label_filter

        #self.data_disease_class = disease_class.replace('-', ' ')

        # Filter the dataset to only include samples with the desired label
        if self.label_filter != None:
            self.filtered_indices = [
                i for i, label in enumerate(train_dataset.labels)
                if label == self.label_filter
            ]
        else:
            self.filtered_indices = list(range(len(train_dataset)))

        self.num_instance_images = len(self.filtered_indices)
        self.add_fitzpatrick_scale_to_prompt = add_fitzpatrick_scale_to_prompt

        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images

        # Image transforms
        if self.split == 'train':
            image_transforms = [
                transforms.RandomResizedCrop(
                    size=size, scale=(0.9, 1.1), ratio=(0.9, 1.1),
                    interpolation=transforms.InterpolationMode.BILINEAR, antialias=True
                ),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            image_transforms = [
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

        # Access the data from the underlying train_dataset
        train_sample = self.train_dataset[index]
        image = train_sample["image"]
        label = train_sample["label"]
        image_name = train_sample["id"]

        # Prepare the instance prompt
        instance_class_name = label.replace('-', ' ')
        instance_prompt = self.instance_prompt.format(instance_class_name)
        if self.add_fitzpatrick_scale_to_prompt and hasattr(train_sample, "fitzpatrick_scale"):
            fitzpatrick_scale = str(train_sample.fitzpatrick_scale) if train_sample.fitzpatrick_scale > 0 else 'unknown'
            instance_prompt += f', Fitzpatrick skin type {fitzpatrick_scale}'

        # Apply image transformations
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["instance_images"] = self.image_transforms(image)

        # Tokenize the prompt with caching
        if self.encoder_hidden_states is not None:
            example["instance_prompt_ids"] = self.encoder_hidden_states
        else:
            text_inputs = self.tokenize_prompt_with_caching(instance_prompt)
            example["instance_prompt_ids"] = text_inputs.input_ids
            example["instance_attention_mask"] = text_inputs.attention_mask
            example["prompt"] = instance_prompt

        return example



# TODO: After refactoring, test that the script works equivalently to the original (walk through with the debugger)

simple_dataset = CustomDataset(dataset_dir="sample_dataset", split="val")
training_dataset = DiffusionTrainWrapper(

)





# # Adjust arguments here
# pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
# instance_data_dir = "/n/data1/hms/dbmi/manrai/derm/Fitzpatrick17k"
# dataset_type = "fitzpatrick"
# instance_prompt = "An image of {}, a skin disease"
# validation_prompt = "An image of allergic contact dermatitis, a skin disease"
# output_dir = "dreambooth-outputs/allergic-contact-dermatitis"
# disease_class = "allergic-contact-dermatitis"
# resolution = 512
# train_batch_size = 4
# gradient_accumulation_steps = 1
# learning_rate = 5e-6
# lr_scheduler = "constant"
# lr_warmup_steps = 0
# num_train_epochs = 4
# report_to = "wandb"

# # Model arguments
# revision = None
# tokenizer_name = None

# # Data arguments
# class_data_dir = None
# class_prompt = None
# with_prior_preservation = False
# prior_loss_weight = 1.0
# num_class_images = 100

# # Output arguments
# seed = None
# center_crop = False

# # Training arguments
# train_text_encoder = False
# sample_batch_size = 4
# max_train_steps = None
# checkpointing_steps = 1_000_000
# checkpoints_total_limit = None
# resume_from_checkpoint = None
# gradient_checkpointing = False

# # Optimizer arguments
# scale_lr = False
# lr_num_cycles = 1
# lr_power = 1.0
# use_8bit_adam = False
# dataloader_num_workers = 0
# adam_beta1 = 0.9
# adam_beta2 = 0.999
# adam_weight_decay = 1e-2
# adam_epsilon = 1e-08
# max_grad_norm = 1.0

# # Hugging Face Hub arguments
# push_to_hub = False
# hub_token = None
# hub_model_id = None

# # Logging arguments
# log_dir = "logs"
# allow_tf32 = False

# # Validation arguments
# num_validation_images = 8
# validation_steps = 100

# # Precision arguments
# mixed_precision = None
# prior_generation_precision = None
# local_rank = -1
# set_grads_to_none = False

# # Additional training arguments
# offset_noise = False
# pre_compute_text_embeddings = False
# tokenizer_max_length = None
# text_encoder_use_attention_mask = False
# skip_save_text_encoder = False
# validation_images = None
# class_labels_conditioning = None

# # Dataset arguments
# add_fitzpatrick_scale_to_prompt = False

