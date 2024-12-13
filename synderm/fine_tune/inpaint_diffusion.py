import builtins
import gc
import itertools
import logging
import hashlib
import math
import os
from pathlib import Path
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from torch.utils.data import Dataset
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from torchvision.transforms.functional import to_pil_image
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, ImageDraw
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from synderm.fine_tune.util import (
    PromptDataset,
    DiffusionTrainWrapper,
    collate_fn,
    tokenize_prompt,
    encode_prompt,
    import_model_class_from_model_name_or_path,
)

from transformers import CLIPTextModel, CLIPTokenizer
# Instance prompt can be a formatted string, in which case the label will be inserted 
def fine_tune_inpaint(
    train_dataset: Dataset,
    pretrained_model_name_or_path: str = "runwayml/stable-diffusion-inpainting",  
    instance_prompt: str = "An image of {}",
    validation_prompt_format: str = "An image of {}", 
    output_dir: str = None,
    label_filter: str | None = None,
    resolution: int = 512,
    train_batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-6,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    num_train_epochs: int = 4,
    report_to: str = "wandb",
    verbose: bool = True,
    num_validation_images: int = 8,
    validation_steps: int = 100
):
    """
    Fine-tune an inpainting diffusion model (e.g., Stable Diffusion). This command takes in 
    a dataset containing images and labels, and trains a diffusion model to inpaint randomly 
    generated masks. The produced model can then be used downstream to generate synthetic data 
    to augment an image classifer. This function sets up training loops, 
    handles prompt-tokenization, mask generation, and logs intermediate results.

    Args:
        train_dataset (Dataset): 
            A Torch `Dataset` object containing an image and a label.
            Each example in the Dataset object should be a dictionary that includes:
            - "image": This should be a PIL image.
            - "label": The label/class name associated with the image.
        
        pretrained_model_name_or_path (str): 
            The path to a pretrained diffusion model or a model name on the HuggingFace Hub.
            Default: "runwayml/stable-diffusion-inpainting",  
        
        instance_prompt (str): 
            A prompt template string used for generating instance embeddings. The label will be inserted 
            into this template if it's a formatted string. Default: "An image of {}"
        
        validation_prompt_format (str): 
            A prompt template string for generating validation images. The label will be inserted into 
            this format if needed. Default: "An image of {}"
        
        output_dir (str): 
            Directory where model checkpoints and final pipeline should be saved.
        
        label_filter (str | None): 
            If provided, only data from this particular label/class will be used for training. If None, 
            all available labels in the dataset are used. Default: None
        
        resolution (int): 
            The size of the images for training and inference. Images will be resized to `resolution x resolution`.
            Default: 512
        
        train_batch_size (int): 
            Batch size used during training. Default: 4
        
        gradient_accumulation_steps (int): 
            Number of gradient accumulation steps before a parameter update is performed. 
            Useful for simulating a larger effective batch size. Default: 1
        
        learning_rate (float): 
            Initial learning rate for the optimizer. Default: 5e-6
        
        lr_scheduler (str): 
            Learning rate scheduler type. Default: "constant"
        
        lr_warmup_steps (int): 
            Number of warmup steps for the learning rate scheduler. Default: 0
        
        num_train_epochs (int): 
            Total number of training epochs. Default: 4
        
        report_to (str): 
            Logging integration to use (e.g., "wandb", "tensorboard"). Default: "wandb"
        
        verbose (bool): 
            If True, print additional debug information during training. 
            Note that warnings are surpressed if verbose is set to false. Default: True 
        
        num_validation_images (int): 
            Number of images to generate for validation at each validation step. Default: 8
        
        validation_steps (int): 
            Interval (in steps) at which validation images are generated. Default: 100

    Returns:
        None. The function saves the trained model (and intermediate checkpoints) to `output_dir`.
    """


    # Model training settings
    tokenizer_name = None
    class_prompt = None
    prior_loss_weight = 1.0
    num_class_images = 100
    center_crop = False
    train_text_encoder = False
    sample_batch_size = 4
    max_train_steps = None
    gradient_checkpointing = False
    scale_lr = False
    
    # Optimizer settings
    use_8bit_adam = False
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    max_grad_norm = 1.0
    
    # Logging arguments
    log_dir = "logs"
    allow_tf32 = False

    # Other
    mixed_precision = "no"
    checkpointing_steps = 1_000_000
    checkpoints_total_limit = None
    resume_from_checkpoint = None

    seed = None

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
        text_encoder, 
        tokenizer, 
        unet, 
        vae, 
        accelerator, 
        weight_dtype, 
        epoch, 
        prompt_embeds, 
        negative_prompt_embeds, 
        train_dataloader
    ):
        logger.info(
            f"Running validation... \n Generating {num_validation_images} images with prompt:"
            f" {validation_prompt}."
        )

        pipeline_args = {}

        if vae is not None:
            pipeline_args["vae"] = vae

        if text_encoder is not None:
            text_encoder = accelerator.unwrap_model(text_encoder)

        # create pipeline (note: unet and vae are loaded again in float32)
        pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path,
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

        # run inference
        generator = None if seed is None else torch.Generator(device=accelerator.device).manual_seed(seed)
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

            if len(images) >= num_validation_images:
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
                            wandb.Image(image, caption=f"{i} GEN: {validation_prompt}") for i, image in enumerate(images)
                        ] + [
                            wandb.Image(image, caption=f"{i} INPUT: {validation_prompt}") for i, image in enumerate(input_images)
                        ] + [
                            wandb.Image(image, caption=f"{i} GT: {validation_prompt}") for i, image in enumerate(gt_images)
                        ]
                    }
                )

        del pipeline
        torch.cuda.empty_cache()

        return images

    # Handle label_filter for stratified training
    if label_filter is None:
        unique_labels = list({item.get("label") for item in train_dataset if item.get("label") is not None})
    else:
        unique_labels = [label_filter]  

    for label in unique_labels:
        if is_wandb_available():
            import wandb

        # Will error if the minimal version of diffusers is not installed. Remove at your own risks.
        check_min_version("0.17.0.dev0")

        logger = get_logger(__name__)

        current_output_dir = os.path.join(output_dir, label)
        current_log_dir = os.path.join(current_output_dir, log_dir)

        label_formatted = label.replace("-", " ")
        validation_prompt = validation_prompt_format.format(label_formatted)

        project_config = ProjectConfiguration(
            logging_dir=current_log_dir,
            total_limit=checkpoints_total_limit
            )

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            project_config=project_config,
        )

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
        # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
        # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
        if train_text_encoder and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        if seed is not None:
            set_seed(seed)

       # Handle the repository creation
        if accelerator.is_main_process:
            if current_output_dir is not None:
                os.makedirs(current_output_dir, exist_ok=True)

        # Load the tokenizer
        if tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)
        elif pretrained_model_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")

        # Load models and create wrapper for stable diffusion
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
            learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

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

        # Dataset and DataLoaders creation:
        train_dataset_label = DiffusionTrainWrapper(
            train_dataset = train_dataset,
            tokenizer=tokenizer,
            instance_prompt=instance_prompt,
            label_filter=label,
            size=resolution,
            center_crop=center_crop
        )

        logger.info(f"Starting training for label: {label}")
        logger.info(f"The length of the training dataset for label '{label}' is: {len(train_dataset_label)}")

        def collate_fn(examples):
            prompt = [example["prompt"] for example in examples]
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

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

            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids.squeeze(1)
            masks = torch.stack(masks).squeeze(1)
            masked_images = torch.stack(masked_images).squeeze(1)
            batch = {"input_ids": input_ids, "pixel_values": pixel_values, "masks": masks, "masked_images": masked_images, "prompt": prompt}
            return batch

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_label, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if max_train_steps is None:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

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

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        vae.to(accelerator.device, dtype=weight_dtype)
        if not train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps = num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        config = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "instance_prompt": instance_prompt,
            "validation_prompt": validation_prompt,
            "output_dir": output_dir,
            "label_filter": label_filter,
            "resolution": resolution,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "num_train_epochs": num_train_epochs,
            "report_to": report_to,
            "verbose": verbose,
            "num_validation_images": num_validation_images,
            "validation_steps": validation_steps,
            "tokenizer_name": tokenizer_name,
            "class_prompt": class_prompt,
            "prior_loss_weight": prior_loss_weight,
            "num_class_images": num_class_images,
            "center_crop": center_crop,
            "train_text_encoder": train_text_encoder,
            "sample_batch_size": sample_batch_size,
            "max_train_steps": max_train_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "scale_lr": scale_lr,
            "use_8bit_adam": use_8bit_adam,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_weight_decay": adam_weight_decay,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "log_dir": log_dir,
            "allow_tf32": allow_tf32,
            "mixed_precision": mixed_precision,
            "checkpointing_steps": checkpointing_steps,
            "checkpoints_total_limit": checkpoints_total_limit,
            "resume_from_checkpoint": resume_from_checkpoint,
            "seed": seed
        }

        if accelerator.is_main_process:
            if label is not None:
                run_name = f"derm_{label}"
            else:
                run_name = "derm"

            accelerator.init_trackers(run_name, config=config)

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        global_step = 0
        first_epoch = 0

        if resume_from_checkpoint:
            if resume_from_checkpoint != "latest":
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0:
                    pass
                    # print(f'Printing batch at step {step}')
                    # print(batch)
                
                # Skip steps until we reach the resumed step
                if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with accelerator.accumulate(unet):
                    # Convert images to latent space

                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Convert masked images to latent space
                    masked_latents = vae.encode(
                        batch["masked_images"].reshape(batch["pixel_values"].shape).to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    masked_latents = masked_latents * vae.config.scaling_factor

                    masks = batch["masks"]
                    
                    mask = torch.nn.functional.interpolate(masks, size=(resolution // 8, resolution // 8))

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # concatenate the noised latents with the mask and the masked latents
                    latent_model_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        if global_step % checkpointing_steps == 0:
                            save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        images = []
                        if validation_prompt is not None and global_step % validation_steps == 0:
                            images = log_validation(
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                unet=unet,
                                vae=vae,
                                accelerator=accelerator,
                                weight_dtype=weight_dtype,
                                epoch=epoch,
                                prompt_embeds=None,
                                negative_prompt_embeds=None,
                                train_dataloader=train_dataloader
                            )

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
            )
            pipeline.save_pretrained(current_output_dir)

        accelerator.end_training()

