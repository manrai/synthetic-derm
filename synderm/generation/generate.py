from synderm.generation.util import GenerationWrapper

from typing import Optional, Union, Literal
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.utils import make_grid
from tqdm.rich import tqdm
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
import os


def generate_synthetic_dataset(
    dataset: Dataset,
    generation_type: Literal["inpaint", "text-to-image"] = "inpaint",
    model_path: str = "runwayml/stable-diffusion-inpainting",
    label_filter: Optional[str] = None,
    output_dir_path: Optional[Union[str, Path]] = None,
    batch_size: int = 16,
    start_index: int = 0,
    num_generations_per_image: int = 1,
    seed: int = 42,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 50,
    strength_inpaint: float = 0.970,
    strength_outpaint: float = 0.950,
    mask_fraction: float = 0.25,
    instance_prompt: str = "An image of {}, a skin disease",
    device: Literal["cuda", "cpu"] = "cuda"
):
    """
    Generate synthetic images from a given dataset using a Stable Diffusion model pipeline, 
    either by inpainting or pure text-to-image generation. The synthetic images can be used 
    to augment training data for image classification or other downstream tasks.

    This function processes a dataset of images, applies a prompt format (optionally conditioned 
    on a specific label), and uses a Stable Diffusion pipeline to produce new images. Depending 
    on `generation_type`, it either:
    - Inpaints masked regions of the original images to create variations.
    - Generates entirely new images from textual prompts.

    The generated images are saved to the specified output directory. Optional image grids are 
    saved for inspection of intermediate results.

    Args:
        dataset (Dataset):
            A Torch Dataset containing images and associated metadata. Each element should 
            at least contain:
            - "image": The original PIL image
            - "label": The class label (str) of the image.
            - "id": A unique identifier (e.g., filename) for the image. This is used to map the source (real) image to generated synthetic images.
        
        generation_type (Literal["inpaint", "text-to-image"], optional):
            The type of generation to perform:
            - "inpaint": Mask and inpaint parts of existing images, then optionally outpaint beyond the mask.
            - "text-to-image": Generate images purely from prompts.
            Default is "inpaint".
        
        model_path (str, optional):
            Path or identifier of the pretrained Stable Diffusion model. This may be a Hugging Face Hub model name or 
            a local directory. Defaults to "runwayml/stable-diffusion-inpainting".
        
        label_filter (str, optional):
            If provided, only images from this label will be processed. If None, an error is raised 
            because handling multiple labels is not implemented. Default is None.
        
        output_dir_path (Union[str, Path], optional):
            Directory where the generated images and grids will be saved. Will be created if it doesn't exist.
            If None, no images are saved (not recommended).
        
        batch_size (int, optional):
            Number of images processed per batch. Default is 16.
        
        start_index (int, optional):
            Starting index for naming and seeding repeat generations. Useful for continuing runs 
            or generating multiple variations. Default is 0.
        
        num_generations_per_image (int, optional):
            Number of times to repeat the generation process for each image in the dataset. 
            E.g., if set to 3, each image in the dataset will produce 3 sets of synthetic images.
            Default is 1.
        
        seed (int, optional):
            Random seed for deterministic generation. Default is 42.
        
        guidance_scale (float, optional):
            Guidance scale controls how strongly the model adheres to the prompt. Higher values 
            typically yield images more closely aligned with the prompt. Default is 3.0.
        
        num_inference_steps (int, optional):
            Number of diffusion steps during generation. More steps can yield higher quality images 
            but will be slower. Default is 50.
        
        strength_inpaint (float, optional):
            Control how strongly the model modifies the masked region during inpainting. 
            1.0 would fully rely on the diffusion process; lower values rely more on the initial image.
            Default is 0.970.
        
        strength_outpaint (float, optional):
            Similar to `strength_inpaint`, but for the outpainting phase. Default is 0.950.
        
        mask_fraction (float, optional):
            Fraction of the image (square region) to mask for inpainting/outpainting. 0.25 means 
            mask a 25% square in the center. Default is 0.25.
        
        instance_prompt (str, optional):
            A template string for instance-based prompts. The label will be inserted into this template 
            if the prompt is a format string. Default is "An image of {}, a skin disease".
        
        device (Literal["cuda", "cpu"], optional):
            Device to run inference on. Currently only "cuda" is supported. Default is "cuda".
    
    Returns:
        None. The function saves generated images and image grids to the specified output directory.
    
    Raises:
        NotImplementedError:
            If `label_filter` is None (multi-label generation not implemented) or `device` is not "cuda".

    Example:
        Suppose you have a dataset of dermatology images and want to generate 
        new psoriasis images by inpainting followed by outpainting.
        
        >>> generate_synthetic_dataset(
        ...     dataset=my_dataset,
        ...     generation_type="inpaint",
        ...     label_filter="psoriasis",
        ...     output_dir_path="synthetic_data/",
        ...     batch_size=8,
        ...     start_index=0,
        ...     num_generations_per_image=2,
        ...     seed=123,
        ...     guidance_scale=5.0,
        ...     strength_inpaint=0.9,
        ...     mask_fraction=0.3
        ... )
        
        This will create 2 sets of inpainted images for each image in `my_dataset`, saving them to "synthetic_data/".
    """


    if label_filter is None:
        raise NotImplementedError("Generation for multiple labels not yet implemented.")

    if device != "cuda":
        raise NotImplementedError("Cuda device required to generate the synthetic dataset")

    # Ensure ourput_dir_path is a Path object
    output_dir_path = Path(output_dir_path)  

    # Device and autograd
    ctx = torch.inference_mode()
    ctx.__enter__()
    device = 'cuda'
    dtype = torch.float16

    print('Loading model')
    if generation_type == 'inpaint':
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None, 
            feature_extractor=None, 
            requires_safety_checker=False)
    elif generation_type == 'text-to-image':
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            safety_checker=None, 
            feature_extractor=None, 
            requires_safety_checker=False)

    resolution = pipeline.unet.config.sample_size * 8

    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)

    print(f'Loaded pipeline with {sum(p.numel() for p in pipeline.unet.parameters()):_} unet parameters')

    wrapped_dataset = GenerationWrapper(dataset=dataset, instance_prompt=instance_prompt, label_filter=label_filter)
    dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + start_index)

    def create_square_mask():
        H_lat, W_lat = resolution // 8, resolution // 8
        mask = torch.zeros((batch_size, 1, H_lat, W_lat), device=device)
        start, end = int(H_lat * mask_fraction), int(H_lat * (1 - mask_fraction))
        mask[:, :, start:end, start:end] = torch.ones_like(mask[:, :, start:end, start:end])
        mask = F.interpolate(mask, size=(resolution, resolution), mode='nearest')
        mask = mask.to(device)
        mask_inpaint = mask
        mask_outpaint = 1 - mask
        s = 16  # small number for border region
        mask_outpaint[:, :, :s] = mask_outpaint[:, :, -s:] = torch.zeros_like(mask_outpaint[:, :, -s:])
        mask_outpaint[:, :, :, :s] = mask_outpaint[:, :, :, -s:] = torch.zeros_like(mask_outpaint[:, :, :, -s:])
        return mask_inpaint.to(device), mask_outpaint.to(device)

    # Generate masks
    if generation_type == 'inpaint':
        mask_inpaint, mask_outpaint = create_square_mask()

    def save(image, path):
        path = Path(path) if isinstance(path, str) else path
        path.parent.mkdir(exist_ok=True, parents=True)
        image.save(path)

    # In the generation loop:
    for idx in tqdm(range(start_index, start_index + num_generations_per_image)):
        for batch_idx, batch in enumerate(dataloader):
            pixel_values = batch["image"]
            batch["pixel_values"] = pixel_values

            gen_kwargs = dict(
                prompt=batch["prompt"],
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                height=resolution,
                width=resolution,
            )

            # Text-to-image
            if generation_type == 'text-to-image':
                output_paths = [
                    output_dir_path / "text-to-image" / f"{idx:02d}" / label / f"{name}.png"
                    for label, name in zip(batch["label"], batch["id"])
                ]

                if all(output_path.is_file() for output_path in output_paths):
                    continue  # Images have already been generated, skip this batch

                images = pipeline(**gen_kwargs).images
                assert len(images) == len(output_paths)
                for image, path in zip(images, output_paths):
                    save(image, path)

                if batch_idx < 10:
                    grid_images = [transforms.ToTensor()(img) for img in images]
                    original_images = [img * 0.5 + 0.5 for img in batch["pixel_values"]]
                    grid = make_grid(grid_images + original_images, nrow=batch_size, padding=4, pad_value=1.0)
                    grid_path = output_dir_path / "grid" / f'{idx:02d}-batch-{batch_idx:02d}.png'
                    save(transforms.ToPILImage()(grid), grid_path)

                    if batch_idx % 1000 == 0:
                        print(f'[Repeat {idx}, batch {batch_idx}] Saved image grid to {grid_path}')
                
            # Inpaint
            elif generation_type == 'inpaint':
                batch["pixel_values"] = batch["pixel_values"].to(device)

                # Inpaint
                inpainted_images: list[Image.Image] = pipeline(
                    image=batch["pixel_values"], 
                    mask_image=mask_inpaint[:batch["pixel_values"].shape[0]], 
                    strength=strength_inpaint, 
                    **gen_kwargs
                ).images

                output_paths = [
                    output_dir_path / "inpaint" / f"{idx:02d}" / label / f"{name}.png"
                    for label, name in zip(batch["label"], batch["id"])
                ]

                assert len(inpainted_images) == len(output_paths)
                for image, path in zip(inpainted_images, output_paths, strict=True):
                    save(image, path)
                inpainted_images = torch.stack([TVF.to_tensor(image) for image in inpainted_images]).to(device)

                # Then outpaint
                outpainted_images: list[Image.Image] = pipeline(
                    image=((inpainted_images - 0.5) / 0.5), 
                    mask_image=mask_outpaint[:batch["pixel_values"].shape[0]], 
                    strength=strength_outpaint, 
                    **gen_kwargs
                ).images

                output_paths = [
                    output_dir_path / "inpaint-outpaint" / f"{idx:02d}" / label / f"{name}.png"
                    for label, name in zip(batch["label"], batch["id"])
                ]

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
                        nrow=batch_size, padding=4, pad_value=1.0,
                    )), grid_path)
                    if batch_idx % 1000 == 0:
                        print(f'[Repeat {idx}, batch {batch_idx}] Saved image grid to {grid_path}')