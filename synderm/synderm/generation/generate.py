from synderm.generation.util import GenerationWrapper

from pathlib import Path
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.utils import make_grid
from tqdm.rich import tqdm
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path


def generate_synthetic_dataset(
    dataset,
    generation_type = "inpaint",
    model_path = "runwayml/stable-diffusion-inpainting",
    label_filter = None,
    output_dir_path = None,
    batch_size = 16,
    start_index = 0,
    num_generations_per_image = 1,
    seed = 42,
    guidance_scale = 3.0,
    num_inference_steps = 50,
    strength_inpaint = 0.970,
    strength_outpaint = 0.950,
    mask_fraction = 0.25,
    instance_prompt = "An image of {}, a skin disease",
    device = "cuda"
    ):

    if label_filter is None:
        raise NotImplementedError("Generation for multiple labels not yet implemented.")

    if device != "cuda":
        raise NotImplementedError("Cuda device required to generate the synthetic dataset")

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
    else:
        raise ValueError(generation_type)

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
    for idx in range(start_index, start_index + num_generations_per_image):
        for batch_idx, batch in enumerate(tqdm(dataloader)):
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
            
            else:
                raise ValueError(generation_type)
