import builtins
import gc
import itertools
import logging
import hashlib
import math
import os
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from torch.utils.data import Dataset
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub import model_info
from PIL import Image
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

# Instance prompt can be a formatted string, in which case the label will be inserted 
def fine_tune_text_to_image(
    train_dataset: Dataset,
    pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-2-1-base",
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
    Fine-tune a diffusion model (e.g., Stable Diffusion) on a given training dataset. This function sets up training loops, 
    handles prompt-tokenization and embedding computation, and logs intermediate results.

    Args:
        train_dataset (Dataset): 
            A Torch `Dataset` object containing an image and a label.
            Each example in the Dataset object should be a dictionary that includes:
            - "image": This should be a PIL image.
            - "label": The label/class name associated with the image.
        
        pretrained_model_name_or_path (str): 
            The path to a pretrained diffusion model or a model name on the HuggingFace Hub.
            Default: "stabilityai/stable-diffusion-2-1-base"
        
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

    # Model arguments
    revision = None
    tokenizer_name = None

    # Data arguments
    class_data_dir = None
    class_prompt = None
    with_prior_preservation = False
    prior_loss_weight = 1.0
    num_class_images = 100

    # Output arguments
    seed = None
    center_crop = False

    # Training arguments
    train_text_encoder = False
    sample_batch_size = 4
    max_train_steps = None
    checkpointing_steps = 1_000_000
    checkpoints_total_limit = None
    resume_from_checkpoint = None
    gradient_checkpointing = False

    # Optimizer arguments
    scale_lr = False
    lr_num_cycles = 1
    lr_power = 1.0
    use_8bit_adam = False
    dataloader_num_workers = 0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    max_grad_norm = 1.0

    # Logging arguments
    log_dir = "logs"
    allow_tf32 = False

    # Precision arguments
    mixed_precision = None
    prior_generation_precision = None
    set_grads_to_none = False

    # Additional training arguments
    offset_noise = False
    pre_compute_text_embeddings = False
    tokenizer_max_length = None
    text_encoder_use_attention_mask = False
    skip_save_text_encoder = False
    validation_images = None
    class_labels_conditioning = None

    # Dataset arguments
    add_fitzpatrick_scale_to_prompt = False

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
        validation_prompt
    ):
        logger.info(
            f"running validation... \n generating {num_validation_images} images with prompt:"
            f" {validation_prompt}."
        )

        pipeline_args = {}

        if vae is not None:
            pipeline_args["vae"] = vae

        if text_encoder is not None:
            text_encoder = accelerator.unwrap_model(text_encoder)

        # Create pipeline (note: unet and vae are loaded again in float32)
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=accelerator.unwrap_model(unet),
            revision=revision,
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

        if pre_compute_text_embeddings:
            pipeline_args = {
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
            }
        else:
            pipeline_args = {"prompt": validation_prompt}

        # Run inference
        generator = None if seed is None else torch.Generator(device=accelerator.device).manual_seed(seed)
        images = []
        if validation_images is None:
            for _ in range(num_validation_images):
                with torch.autocast("cuda"):
                    image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
                images.append(image)
        else:
            for img_path in validation_images:
                image = Image.open(img_path)
                image = pipeline(**pipeline_args, image=image, generator=generator).images[0]
                images.append(image)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(image, caption=f"{i}: {validation_prompt}") for i, image in enumerate(images)
                        ]
                    }
                )

        del pipeline
        torch.cuda.empty_cache()

        return images

    def model_has_vae():
        config_file_name = os.path.join("vae", AutoencoderKL.config_name)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file_name = os.path.join(pretrained_model_name_or_path, config_file_name)
            return os.path.isfile(config_file_name)
        else:
            files_in_repo = model_info(pretrained_model_name_or_path, revision=revision).siblings
            return any(file.rfilename == config_file_name for file in files_in_repo)

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

        accelerator_project_config = ProjectConfiguration(
            total_limit=checkpoints_total_limit,
            logging_dir=current_log_dir
            )

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            project_config=accelerator_project_config,
        )


        if report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        if verbose:
            logger.info(accelerator.state, main_process_only=False)

        if not verbose:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()
        else:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()

        # If passed along, set the training seed now.
        if seed is not None:
            set_seed(seed)

        # Generate class images if prior preservation is enabled.
        if with_prior_preservation:
            class_images_dir = Path(class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)

                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"]).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
        if accelerator.is_main_process:
            if current_output_dir is not None:
                os.makedirs(current_output_dir, exist_ok=True)

        # Load the tokenizer
        if tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, use_fast=False)
        elif pretrained_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=revision,
                use_fast=False,
            )

        # import correct text encoder class
        text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = text_encoder_cls.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
        )

        if model_has_vae():
            vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path, subfolder="vae", revision=revision
            )
        else:
            vae = None

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet", revision=revision
        )

        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir_hook):
            for model in models:
                sub_dir = "unet" if isinstance(model, type(accelerator.unwrap_model(unet))) else "text_encoder"
                model.save_pretrained(os.path.join(output_dir_hook, sub_dir))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir_hook):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                if isinstance(model, type(accelerator.unwrap_model(text_encoder))):
                    # load transformers style into model
                    load_model = text_encoder_cls.from_pretrained(input_dir_hook, subfolder="text_encoder")
                    model.config = load_model.config
                else:
                    # load diffusers style into model
                    load_model = UNet2DConditionModel.from_pretrained(input_dir_hook, subfolder="unet")
                    model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        if vae is not None:
            vae.requires_grad_(False)

        if not train_text_encoder:
            text_encoder.requires_grad_(False)

        if gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        # Check that all trainable models are in full precision
        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if accelerator.unwrap_model(unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
            )

        if train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
            raise ValueError(
                f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
                f" {low_precision_error_string}"
            )

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        lr = learning_rate
        if scale_lr:
            lr = (
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

        # Optimizer creation
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else unet.parameters()
        )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

        if pre_compute_text_embeddings:
            def compute_text_embeddings(prompt):
                with torch.no_grad():
                    text_inputs = tokenize_prompt(tokenizer, prompt, tokenizer_max_length=tokenizer_max_length)
                    prompt_embeds = encode_prompt(
                        text_encoder,
                        text_inputs.input_ids,
                        text_inputs.attention_mask,
                        text_encoder_use_attention_mask=text_encoder_use_attention_mask,
                    )

                return prompt_embeds

            pre_computed_encoder_hidden_states = compute_text_embeddings(instance_prompt)
            validation_prompt_negative_prompt_embeds = compute_text_embeddings("")

            if validation_prompt is not None:
                validation_prompt_encoder_hidden_states = compute_text_embeddings(validation_prompt)
            else:
                validation_prompt_encoder_hidden_states = None

            if instance_prompt is not None:
                pre_computed_instance_prompt_encoder_hidden_states = compute_text_embeddings(instance_prompt)
            else:
                pre_computed_instance_prompt_encoder_hidden_states = None

            text_encoder = None
            tokenizer = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            pre_computed_encoder_hidden_states = None
            validation_prompt_encoder_hidden_states = None
            validation_prompt_negative_prompt_embeds = None
            pre_computed_instance_prompt_encoder_hidden_states = None

        # Dataset and DataLoaders creation:
        train_dataset_label = DiffusionTrainWrapper(
            train_dataset = train_dataset,
            tokenizer=tokenizer,
            instance_prompt=instance_prompt,
            label_filter=label,
            size=resolution,
            center_crop=center_crop,
            encoder_hidden_states=pre_computed_encoder_hidden_states,
            instance_prompt_encoder_hidden_states=pre_computed_instance_prompt_encoder_hidden_states,
            tokenizer_max_length=tokenizer_max_length,
            add_fitzpatrick_scale_to_prompt=add_fitzpatrick_scale_to_prompt
        )

        logger.info(f"Starting training for label: {label}")
        logger.info(f"The length of the training dataset for label '{label}' is: {len(train_dataset_label)}")

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_label,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),
            num_workers=dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

        if max_train_steps is None:
            max_train_steps_current = num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True
        else:
            max_train_steps_current = max_train_steps
    
        scheduler_obj = get_scheduler(
            lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
            num_training_steps=max_train_steps_current * gradient_accumulation_steps,
            num_cycles=lr_num_cycles,
            power=lr_power,
        )

        # Prepare everything with our `accelerator`.
        if train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, scheduler_obj = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, scheduler_obj
            )
        else:
            unet, optimizer, train_dataloader, scheduler_obj = accelerator.prepare(
                unet, optimizer, train_dataloader, scheduler_obj
            )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        if vae is not None:
            vae.to(accelerator.device, dtype=weight_dtype)

        if not train_text_encoder and text_encoder is not None:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        if overrode_max_train_steps:
            max_train_steps_current = num_train_epochs * num_update_steps_per_epoch

        # Afterwards we recalculate our number of training epochs
        num_train_epochs = math.ceil(max_train_steps_current / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        config = {
            "pretrained_model_name_or_path": pretrained_model_name_or_path,
            "instance_prompt": instance_prompt,
            "validation_prompt": validation_prompt,
            "output_dir": current_output_dir,
            "label_filter": label_filter,
            "current_label": label,
            "resolution": resolution,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "num_train_epochs": num_train_epochs,
            "report_to": report_to,
            "with_prior_preservation": with_prior_preservation,
            "prior_loss_weight": prior_loss_weight,
            "num_class_images": num_class_images,
            "seed": seed,
            "center_crop": center_crop,
            "train_text_encoder": train_text_encoder,
            "sample_batch_size": sample_batch_size,
            "max_train_steps": max_train_steps_current,
            "checkpointing_steps": checkpointing_steps,
            "checkpoints_total_limit": checkpoints_total_limit,
            "resume_from_checkpoint": resume_from_checkpoint,
            "gradient_checkpointing": gradient_checkpointing,
            "scale_lr": scale_lr,
            "lr_num_cycles": lr_num_cycles,
            "lr_power": lr_power,
            "use_8bit_adam": use_8bit_adam,
            "dataloader_num_workers": dataloader_num_workers,
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_weight_decay": adam_weight_decay,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "log_dir": log_dir,
            "allow_tf32": allow_tf32,
            "num_validation_images": num_validation_images,
            "validation_steps": validation_steps,
            "mixed_precision": mixed_precision,
            "prior_generation_precision": prior_generation_precision,
            "set_grads_to_none": set_grads_to_none,
            "offset_noise": offset_noise,
            "pre_compute_text_embeddings": pre_compute_text_embeddings,
            "tokenizer_max_length": tokenizer_max_length,
            "text_encoder_use_attention_mask": text_encoder_use_attention_mask,
            "skip_save_text_encoder": skip_save_text_encoder,
            "validation_images": validation_images,
            "class_labels_conditioning": class_labels_conditioning,
            "add_fitzpatrick_scale_to_prompt": add_fitzpatrick_scale_to_prompt,
        }

        if accelerator.is_main_process:
            if label is not None:
                run_name = f"derm_{label}"
            else:
                run_name = "derm"

            accelerator.init_trackers(run_name, config=config)

        # **Run validation before training starts -- see how the model performs at baseline
        if accelerator.is_main_process:
            images = log_validation(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                vae=vae,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                epoch=0,
                prompt_embeds=validation_prompt_encoder_hidden_states,
                negative_prompt_embeds=validation_prompt_negative_prompt_embeds,
                validation_prompt=validation_prompt
            )
            logger.info(f"Validation at epoch 0 completed for label '{label}'. Generated {len(images)} images.")

        # Train!
        total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Label = {label}")
        logger.info(f"  Num examples = {len(train_dataset_label)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps_current}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if resume_from_checkpoint:
            if resume_from_checkpoint != "latest":
                path = os.path.basename(resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(current_output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run for label '{label}'."
                )
                resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path} for label '{label}'.")
                accelerator.load_state(os.path.join(current_output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(global_step, max_train_steps_current), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")

        for epoch in range(first_epoch, num_train_epochs):
            unet.train()
            if train_text_encoder:
                text_encoder.train()
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
                    pixel_values = batch["pixel_values"].to(dtype=weight_dtype)

                    if vae is not None:
                        # Convert images to latent space
                        model_input = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        model_input = model_input * vae.config.scaling_factor
                    else:
                        model_input = pixel_values

                    # Sample noise that we'll add to the input
                    if offset_noise:
                        noise = torch.randn_like(model_input) + 0.1 * torch.randn(
                            model_input.shape[0], model_input.shape[1], 1, 1, device=model_input.device
                        )
                    else:
                        noise = torch.randn_like(model_input)
                    bsz, channels, height, width = model_input.shape
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device
                    )
                    timesteps = timesteps.long()

                    # Add noise to the model input according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

                    # Get the text embedding for conditioning
                    if pre_compute_text_embeddings:
                        encoder_hidden_states = batch["input_ids"]
                    else:
                        encoder_hidden_states = encode_prompt(
                            text_encoder,
                            batch["input_ids"],
                            batch["attention_mask"],
                            text_encoder_use_attention_mask=text_encoder_use_attention_mask,
                        )

                    if unet.config.in_channels > channels:
                        needed_additional_channels = unet.config.in_channels - channels
                        additional_latents = randn_tensor(
                            (bsz, needed_additional_channels, height, width),
                            device=noisy_model_input.device,
                            dtype=noisy_model_input.dtype,
                        )
                        noisy_model_input = torch.cat([additional_latents, noisy_model_input], dim=1)

                    if class_labels_conditioning == "timesteps":
                        class_labels = timesteps
                    else:
                        class_labels = None

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_model_input, timesteps, encoder_hidden_states, class_labels=class_labels
                    ).sample

                    if model_pred.shape[1] == 6:
                        model_pred, _ = torch.chunk(model_pred, 2, dim=1)

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(model_input, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    if with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                        target, target_prior = torch.chunk(target, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                        # Compute prior loss
                        prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                    optimizer.step()
                    scheduler_obj.step()
                    optimizer.zero_grad(set_to_none=set_grads_to_none)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        images = []
                        if global_step % checkpointing_steps == 0:
                            save_path = os.path.join(current_output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                        if validation_prompt is not None and global_step % validation_steps == 0:
                            images = log_validation(
                                text_encoder,
                                tokenizer,
                                unet,
                                vae,
                                accelerator,
                                weight_dtype,
                                epoch,
                                validation_prompt_encoder_hidden_states,
                                validation_prompt_negative_prompt_embeds,
                                validation_prompt=validation_prompt
                            )

                logs = {"loss": loss.detach().item(), "lr": scheduler_obj.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= max_train_steps_current:
                    break

            if global_step >= max_train_steps_current:
                break

        # Create the pipeline using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            pipeline_args = {}

            if text_encoder is not None:
                pipeline_args["text_encoder"] = accelerator.unwrap_model(text_encoder)

            if skip_save_text_encoder:
                pipeline_args["text_encoder"] = None

            pipeline = DiffusionPipeline.from_pretrained(
                pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                revision=revision,
                **pipeline_args,
            )

            # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
            scheduler_args = {}

            if "variance_type" in pipeline.scheduler.config:
                variance_type = pipeline.scheduler.config.variance_type

                if variance_type in ["learned", "learned_range"]:
                    variance_type = "fixed_small"

                scheduler_args["variance_type"] = variance_type

            pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
            pipeline.save_pretrained(current_output_dir)
            logger.info(f"Saved model for label '{label}' to {current_output_dir}")

        accelerator.end_training()
