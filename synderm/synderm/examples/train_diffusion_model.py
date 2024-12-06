import argparse
import hashlib
import itertools
import math
import os
import random
from pathlib import Path
import bitsandbytes

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version

from dataset import FitzpatrickDataset



# Assuming these helper functions exist in your code:
# from your_package.helpers import log_validation, random_mask, prepare_mask_and_masked_image, get_scheduler

logger = get_logger(__name__)


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


# generate random masks
def random_mask(im_shape, ratio=1, mask_full_image=False):
    mask = Image.new("L", im_shape, 0)
    draw = ImageDraw.Draw(mask)
    size = (random.randint(0, int(im_shape[0] * ratio)), random.randint(0, int(im_shape[1] * ratio)))
    # use this to always mask the whole image
    if mask_full_image:
        size = (int(im_shape[0] * ratio), int(im_shape[1] * ratio))
    limits = (im_shape[0] - size[0] // 2, im_shape[1] - size[1] // 2)
    center = (random.randint(size[0] // 2, limits[0]), random.randint(size[1] // 2, limits[1]))
    draw_type = random.randint(0, 1)
    if draw_type == 0 or mask_full_image:
        draw.rectangle(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )
    else:
        draw.ellipse(
            (center[0] - size[0] // 2, center[1] - size[1] // 2, center[0] + size[0] // 2, center[1] + size[1] // 2),
            fill=255,
        )

    return mask


def log_validation(
    text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch, prompt_embeds, negative_prompt_embeds, train_dataloader
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
        torch_dtype=weight_dtype,
        **pipeline_args,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # if args.pre_compute_text_embeddings:
    #     pipeline_args = {
    #         "prompt_embeds": prompt_embeds,
    #         "negative_prompt_embeds": negative_prompt_embeds,
    #     }
    # else:
    #     pipeline_args = {"prompt": args.validation_prompt}

    # Get imaes 
    # fake_images = torch.rand((3, args.resolution, args.resolution))
    # transform_to_pil = transforms.ToPILImage()
    # fake_pil_images = transform_to_pil(fake_images)
    # fake_mask = random_mask((args.resolution, args.resolution), ratio=1, mask_full_image=True)
    # images = pipeline(prompt=example["prompt"], mask_image=fake_mask, image=fake_pil_images).images

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    gt_images = []
    input_images = []
    for batch in train_dataloader:

        # Sample
        image = pipeline(prompt=batch["prompt"][:1], image=batch["pixel_values"][:1], mask_image=batch["masks"][:1], 
                         num_inference_steps=50, generator=generator).images[0]
        images.append(image)
        gt_images.append(to_pil_image(batch["pixel_values"][0] * 0.5 + 0.5))
        input_images.append(to_pil_image(batch["masked_images"][0] * 0.5 + 0.5))

        if len(images) >= args.num_validation_images:
            break

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            import wandb
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i} GEN: {args.validation_prompt}") for i, image in enumerate(images)
                    ] + [
                        wandb.Image(image, caption=f"{i} INPUT: {args.validation_prompt}") for i, image in enumerate(input_images)
                    ] + [
                        wandb.Image(image, caption=f"{i} GT: {args.validation_prompt}") for i, image in enumerate(gt_images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()

    return images



def train_synthetic_generator(
    train_dataset: Dataset,
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-inpainting",
    output_dir: str = "synthetic_model",
    logging_dir: str = "logs",
    checkpoints_total_limit: int = None,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "fp16",
    report_to: str = "tensorboard",
    allow_tf32: bool = True,
    train_text_encoder: bool = False,
    seed: int = 42,
    with_prior_preservation: bool = False,
    class_data_dir: str = None,
    num_class_images: int = 100,
    sample_batch_size: int = 4,
    resolution: int = 512,
    center_crop: bool = False,
    tokenizer_name: str = None,
    hub_model_id: str = None,
    hub_token: str = None,
    push_to_hub: bool = False,
    learning_rate: float = 5e-6,
    train_batch_size: int = 1,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-8,
    use_8bit_adam: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    scale_lr: bool = False,
    max_train_steps: int = None,
    num_train_epochs: int = 1,
    gradient_checkpointing: bool = False,
    max_grad_norm: float = 1.0,
    validation_prompt: str = None,
    validation_steps: int = 500,
    checkpointing_steps: int = 500,
    prior_loss_weight: float = 1.0,
    resume_from_checkpoint: str = None
):
    """
    Train a synthetic image generator model (such as Stable Diffusion) on a given dataset.

    Args:
        train_dataset (Dataset): A PyTorch dataset that yields dictionaries with the keys required by the training loop.
                                 For instance, it should produce:
                                 {
                                     "instance_prompt_ids": ...,
                                     "instance_images": ...,
                                     # Optionally, for prior preservation:
                                     "class_prompt_ids": ...,
                                     "class_images": ...
                                 }
        output_dir (str): Where to store the final model and checkpoints.
        ... (many other arguments as described above)
    """

    logging_dir = Path(output_dir, logging_dir)
    project_config = ProjectConfiguration(total_limit=checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        logging_dir=logging_dir,
        project_config=project_config,
    )

    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if train_text_encoder and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1."
        )

    if seed is not None:
        set_seed(seed)

    # If prior preservation is requested, generate class images if needed
    if with_prior_preservation:
        class_images_dir = Path(class_data_dir)
        class_images_dir.mkdir(parents=True, exist_ok=True)
        cur_class_images = len(list(class_images_dir.iterdir()))
        if cur_class_images < num_class_images:
            # Generate or handle class images as per your requirements
            # Here we just log the information
            logger.info(f"Need to generate {num_class_images - cur_class_images} class images. "
                        f"Please implement class image generation logic if required.")
            # The original code did class image sampling and generation here.

    # Handle output directory
    if accelerator.is_main_process and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # Handle huggingface hub push
    if accelerator.is_main_process and push_to_hub:
        from huggingface_hub import create_repo, upload_folder
        repo_id = create_repo(
            repo_id=hub_model_id or Path(output_dir).name, exist_ok=True, token=hub_token
        ).repo_id

    # Load the tokenizer
    if tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

    # Load models
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    if not train_text_encoder:
        text_encoder.requires_grad_(False)

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes

    # Use 8-bit Adam if requested
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, install bitsandbytes: `pip install bitsandbytes`.")
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = (
        itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else unet.parameters()
    )
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")

    # Collate function
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Prior preservation: combine instance and class examples
        if with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]
            pior_pil = [to_pil_image(example["class_images"] * 0.5 + 0.5) for example in examples]

        masks = []
        masked_images = []
        for example in examples:
            pil_image = to_pil_image(example["instance_images"] * 0.5 + 0.5)
            # generate a random mask
            mask = random_mask(pil_image.size, 1, False)
            # prepare mask and masked image
            mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)

            masks.append(mask)
            masked_images.append(masked_image)

        if with_prior_preservation:
            for pil_image in pior_pil:
                # generate a random mask
                mask = random_mask(pil_image.size, 1, False)
                # prepare mask and masked image
                mask, masked_image = prepare_mask_and_masked_image(pil_image, mask)
                masks.append(mask)
                masked_images.append(masked_image)

        pixel_values = torch.stack(pixel_values).float()
        input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids.squeeze(1)
        masks = torch.stack(masks).squeeze(1)
        masked_images = torch.stack(masked_images).squeeze(1)

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "masks": masks,
            "masked_images": masked_images
        }
        return batch

    train_dataloader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Determine training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Scheduler
    from transformers import get_scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    accelerator.register_for_checkpointing(lr_scheduler)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("synthetic_training", config={})

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting new.")
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach resume_step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                masked_latents = vae.encode(
                    batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                ).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

                masks = batch["masks"]
                mask = torch.nn.functional.interpolate(masks, size=(resolution // 8, resolution // 8))

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if with_prior_preservation:
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none").mean([1, 2, 3]).mean()
                    prior_loss = F.mse_loss(noise_pred_prior.float(), target_prior.float(), reduction="mean")
                    loss = loss + prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % checkpointing_steps == 0:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Optional validation
                    if validation_prompt is not None and global_step % validation_steps == 0:
                        # Implement log_validation if needed
                        pass

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Save final pipeline
    if accelerator.is_main_process:
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
        )
        pipeline.save_pretrained(output_dir)

        if push_to_hub:
            from huggingface_hub import upload_folder
            upload_folder(
                repo_id=repo_id,
                folder_path=output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()

