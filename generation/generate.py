import inspect
import math
import os
import pdb
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Union, Literal

import pandas as pd
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler
from IPython.display import display
from PIL import Image
from torch import Tensor, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.utils import make_grid
from tqdm.rich import tqdm, trange
from transformers import AutoTokenizer
from tap import Tap
from IPython import get_ipython
from dataset import FitzpatrickDataset

is_jupyter = False

# Device and autograd
ctx = torch.inference_mode()
ctx.__enter__()
device = 'cuda'
dtype = torch.float16

class Args(Tap):

    output_root: Optional[str] = None
    output_dir: Optional[str] = None

    # Data options
    dataset_type: Literal['fitzpatrick', 'ddi'] = 'fitzpatrick'
    disease_class: str = 'psoriasis'
    instance_prompt: str = 'An image of {}, a skin disease'
    instance_data_dir: str = '/home/lukemelas/data/Fitzpatrick17k'
    add_fitzpatrick_scale_to_prompt: bool = False
    num_generations_per_image: int = 10
    batch_size: int = 16

    # Model options
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-inpainting"
    #model_type: Literal['text-to-image', 'inpaint'] = 'text-to-image'
    model_type: Literal['text-to-image', 'inpaint'] = 'inpaint'
    resolution: int = 512
    
    # Generation options
    guidance_scale: float = 3.0
    num_inference_steps: int = 50
    start_index: int = 0
    seed: int = 42
    
    # Mask options
    strength_inpaint: float = 0.970
    strength_outpaint: float = 0.950
    mask_fraction: float = 0.25

    def process_args(self):
        self.disease_class = self.disease_class.lower()
        if self.output_root is None:
            self.output_root = 'generations'
        if self.output_dir is None:
            self.output_dir = Path(self.output_root) / self.model_type / self.disease_class # / timestamp
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)


# Parse args
args = Args().parse_args([] if is_jupyter else None)
output_dir_path = Path(args.output_dir)
args.save(output_dir_path / 'args.json')

def create_square_mask():
    H_lat, W_lat = args.resolution // 8, args.resolution // 8
    mask = torch.zeros((args.batch_size, 1, H_lat, W_lat), device=device)
    start, end = int(H_lat * args.mask_fraction), int(H_lat * (1 - args.mask_fraction))
    mask[:, :, start:end, start:end] = torch.ones_like(mask[:, :, start:end, start:end])
    mask = F.interpolate(mask, size=(args.resolution, args.resolution), mode='nearest')
    mask = mask.to(device)
    mask_inpaint = mask
    mask_outpaint = 1 - mask
    s = 16  # small number for border region
    mask_outpaint[:, :, :s] = mask_outpaint[:, :, -s:] = torch.zeros_like(mask_outpaint[:, :, -s:])
    mask_outpaint[:, :, :, :s] = mask_outpaint[:, :, :, -s:] = torch.zeros_like(mask_outpaint[:, :, :, -s:])
    return mask_inpaint.to(device), mask_outpaint.to(device)


def collate_fn(examples):
    batch = {
        "image_name": [x['image_name'] for x in examples],  # md5hash
        "prompt": [x['prompt'] for x in examples],
        "pixel_values": torch.stack([example["instance_images"] for example in examples]),
    }
    return batch

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer", use_fast=False,
)

# Create dataset
dataset = FitzpatrickDataset(
    dataset_type=args.dataset_type,
    disease_class=args.disease_class,
    add_fitzpatrick_scale_to_prompt=args.add_fitzpatrick_scale_to_prompt,
    instance_data_root=args.instance_data_dir,
    instance_prompt=args.instance_prompt,
    class_data_root=None,
    tokenizer=tokenizer,
    size=args.resolution,
    center_crop=True,
    encoder_hidden_states=None,
    instance_prompt_encoder_hidden_states=None,
    split='val',
)

# Create dataloader
dataloader = DataLoader(
    dataset, collate_fn=collate_fn,
    batch_size=args.batch_size, num_workers=min(args.batch_size, 12), 
)

# Sizes
print(f'{len(dataset) = }')

# Model
print('Loading model')
if args.model_type == 'inpaint':
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
elif args.model_type == 'text-to-image':
    pipeline = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
elif args.model_type == 'unclip':
    # pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=dtype)
    raise ValueError(args.model_type)
else:
    raise ValueError(args.model_type)
pipeline.set_progress_bar_config(disable=True)
pipeline.to(device)
print(f'Loaded pipeline with {sum(p.numel() for p in pipeline.unet.parameters()):_} unet parameters')

# Replace scheduler
pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
print('Replaced scheduler with:')
print(pipeline.scheduler)

# Generate masks
if args.model_type == 'inpaint':
    mask_inpaint, mask_outpaint = create_square_mask()

def get_output_paths(batch: dict, stage: str, idx: int) -> list[Path]:
    return [
        output_dir_path / stage / f'{idx:02d}' / f'{image_name}.png'
        for image_name in batch['image_name']
    ]


def save(image: Image.Image, path: Union[str, Path]):
    path = Path(path) if isinstance(path, str) else path
    path.parent.mkdir(exist_ok=True, parents=True)
    image.save(path)


# Randomness
generator = torch.Generator(device=device)
generator.manual_seed(args.seed + args.start_index)

# Loop over repeats
for idx in range(args.start_index, args.start_index + args.num_generations_per_image):
    for batch_idx, batch in enumerate(tqdm(dataloader)):

        # Shared arguments
        gen_kwargs = dict(
            prompt=batch["prompt"], guidance_scale=args.guidance_scale, generator=generator,
            num_inference_steps=args.num_inference_steps, height=args.resolution, width=args.resolution, 
        )

        # Text-to-image
        if args.model_type == 'text-to-image':
            output_paths = get_output_paths(batch, 'text-to-image', idx)
            if all(output_path.is_file() for output_path in output_paths):
                continue  # images have alread been generated, can skip this batch

            # Generate images
            images: list[Image.Image] = pipeline(**gen_kwargs).images
            assert len(images) == len(output_paths)
            for image, path in zip(images, output_paths, strict=True):
                save(image, path)

            # Image grid
            if batch_idx < 10:
                grid_path = output_dir_path / "grid" / f'{idx:02d}-batch-{batch_idx:02d}.png'
                save(TVF.to_pil_image(make_grid(
                    [TVF.to_tensor(image) for image in images] + 
                    [image * 0.5 + 0.5 for image in batch["pixel_values"]],
                    nrow=args.batch_size, padding=4, pad_value=1.0,
                )), grid_path)
                if batch_idx % 1000 == 0:
                    print(f'[Repeat {idx}, batch {batch_idx}] Saved image grid to {grid_path}')
            
        # Inpaint
        elif args.model_type == 'inpaint':
            output_paths = get_output_paths(batch, 'inpaint-outpaint', idx)
            if all(output_path.is_file() for output_path in output_paths):
                continue  # images have alread been generated, can skip this batch
            
            # To device
            batch["pixel_values"] = batch["pixel_values"].to(device)

            # Inpaint
            inpainted_images: list[Image.Image] = pipeline(
                image=batch["pixel_values"], mask_image=mask_inpaint[:batch["pixel_values"].shape[0]], strength=args.strength_inpaint, **gen_kwargs
            ).images
            output_paths = get_output_paths(batch, 'inpaint', idx)
            assert len(inpainted_images) == len(output_paths)
            for image, path in zip(inpainted_images, output_paths, strict=True):
                save(image, path)
            inpainted_images = torch.stack([TVF.to_tensor(image) for image in inpainted_images]).to(device)

            # Then outpaint
            outpainted_images: list[Image.Image] = pipeline(
                image=((inpainted_images - 0.5) / 0.5), mask_image=mask_outpaint[:batch["pixel_values"].shape[0]], strength=args.strength_outpaint, **gen_kwargs
            ).images
            output_paths = get_output_paths(batch, 'inpaint-outpaint', idx)
            assert len(outpainted_images) == len(output_paths)
            for image, path in zip(outpainted_images, output_paths, strict=True):
                save(image, path)
            outpainted_images = torch.stack([TVF.to_tensor(image) for image in outpainted_images]).to(device)

            # Image grid
            if batch_idx < 10:
                grid_path = output_dir_path / "grid" / f'{idx:02d}-batch-{batch_idx:02d}.png'
                save(TVF.to_pil_image(make_grid(
                    list(mask_inpaint.expand(-1, 3, -1, -1)) + 
                    list(inpainted_images) + 
                    list(mask_outpaint.expand(-1, 3, -1, -1)) + 
                    list(outpainted_images) + 
                    list(batch["pixel_values"] * 0.5 + 0.5),
                    nrow=args.batch_size, padding=4, pad_value=1.0,
                )), grid_path)
                if batch_idx % 1000 == 0:
                    print(f'[Repeat {idx}, batch {batch_idx}] Saved image grid to {grid_path}')
        
        else:
            raise ValueError(args.model_type)
