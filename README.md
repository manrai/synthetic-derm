# Synderm
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://synthetic-derm.readthedocs.io/en/latest/index.html)
[![Complete Dataset](https://img.shields.io/badge/complete_dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/tbuckley/synthetic-derm-1M)
[![Train Dataset](https://img.shields.io/badge/train_dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/tbuckley/synthetic-derm-1M-train)
[![PyPI](https://img.shields.io/pypi/v/synderm.svg)](https://pypi.org/project/synderm/)

Synderm is a package designed to enhance image classification tasks using synthetic data generation. It provides tools to generate high-quality synthetic images using diffusion models, fine-tune these models on your specific datasets, and seamlessly integrate synthetic data into your training pipelines to improve classifier performance.

## Table of Contents

- [Features](#features)
- [Dataset Details](#dataset-details)
- [Models](#models)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Creating the Dataset](#1-creating-the-dataset)
  - [2. Training the Synthetic Image Generator](#2-training-the-synthetic-image-generator)
  - [3. Generate Synthetic Images](#3-generate-synthetic-images)
  - [4. Augmenting the Classifier with Synthetic Images](#4-augmenting-the-classifier-with-synthetic-images)
- [Example Scripts](#example-scripts)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Synthetic Data Generation:** Utilize diffusion models to create high-quality synthetic images tailored to your dataset.
- **Fine-Tuning:** Adapt pre-trained diffusion models to your specific classes using minimal real data.
- **Dataset Augmentation:** Combine real and synthetic data effortlessly to enhance your training datasets.
- **Seamless Integration:** Compatible with popular deep learning frameworks like PyTorch and FastAI.
- **Flexible Configuration:** Easily customize prompts, training parameters, and data splits to fit your project's needs.

### Dataset Details

We have developed a HuggingFace dataset with over 1 millions (more than 600GB). To support efficient use and reuse of such a large dataset, we use the [WebDataset](https://github.com/webdataset/webdataset) format. Using this format, data is split into into TAR shards that contain at most 5,000 images (up to ~2GB). This allows for fine-grained data subsetting with minimal memory and time overhead.

We have developed two versions of the dataset to support different applications. These are:
1) [synthetic-derm-1M](https://huggingface.co/datasets/tbuckley/synthetic-derm-1M): This dataset is intended for fine-grained retrieval of particular labels and generation methods. Each shard is named using the format: `shard-{disease-label}-{synthetic-generation-method}-{submethod}-{index}.tar`. An example shard name is `shard-vitiligo-finetune-text-to-image-text-to-image-00000.tar`. 

1) [synthetic-derm-1M-train](https://huggingface.co/datasets/tbuckley/synthetic-derm-1M-train): This dataset is intended to be used directly for training models. We group images by generation method, perform a shuffle, and then shard the images. Each shard is named using the format: `shard-{synthetic-generation-method}-{index}.tar`. For model training, the dataset can still be subset to specific labels.

See [WebDataset FAQ](https://github.com/webdataset/webdataset/blob/main/FAQ.md) for many more examples of how to use these two datasets. We also provide a vignette demonstrating how to use these dataset.


## Models

Synderm directly supports the following models for image generation:

- **Text-to-image:** [`stabilityai/stable-diffusion-2-1-base`](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- **Inpainting/outpainting:** [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting)

Other diffusion models can be used but are currently untested.

All functions assume that the training and validation datasets return entries with an `image`, `label`, and `id` field. If your dataset does not conform to this structure, please adjust it accordingly (see examples below).

## Installation
```bash
# Install the requirements
pip install -r requirements.txt

# To install from the Python Package Index:
pip install synderm

# Build from source
pip install -e .
```

*Ensure you have [PyTorch](https://pytorch.org/) and [FastAI](https://fast.ai/) installed.*

## Quick Start

### 1. Creating the Dataset

Synderm requires datasets to return entries with `image`, `label`, and `id` fields. Here's an example of how to create a custom dataset:

```python
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
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
```

### 2. Training the Synthetic Image Generator

Fine-tune a diffusion model using your dataset to generate synthetic images:

```python
from synderm.synderm.fine_tune.text_to_image_diffusion import fine_tune_text_to_image

output_dir = os.path.join(EXPERIMENT_DIR, "dreambooth-outputs")

fine_tune_text_to_image(
    train_dataset=train_dataset,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
    instance_prompt="An image of an English Springer",
    validation_prompt_format="An image of an English Springer",
    output_dir=output_dir,
    label_filter="English_springer",
    resolution=512,
    train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    lr_scheduler="constant",
    # Additional parameters...
)
```

### 3. Generate Synthetic Images

Use the fine-tuned diffusion model to generate a set of synthetic images

```python
model_path = os.path.join(EXPERIMENT_DIR, "dreambooth-outputs", "English_springer")
image_output_path = os.path.join(EXPERIMENT_DIR, "generations")

generate_synthetic_dataset(
    dataset= train_dataset,
    model_path = model_path,
    output_dir_path = image_output_path,
    generation_type = "text-to-image", 
    label_filter = "English_springer",
    instance_prompt = "An image of an English Springer",
    batch_size = 16,
    start_index = 0,
    num_generations_per_image = 10,
    guidance_scale = 3.0,
    num_inference_steps = 50,
    strength_inpaint = 0.970,
    strength_outpaint = 0.950,
    mask_fraction = 0.25
)
```


### 4. Augmenting the Classifier with Synthetic Images

Combine real and synthetic data to train and evaluate the classifier:

```python
from synderm.utils.utils import synthetic_train_val_split

synthetic_dataset = SyntheticDataset(os.path.join(image_output_path, "text-to-image"))

train, val = synthetic_train_val_split(
    real_data=train_dataset,
    synthetic_data=synthetic_dataset,
    per_class_test_size=5,
    random_state=42,
    mapping_real_to_synthetic="id"
)
```

## Example Scripts

We include several example scripts at `synderm/example_scripts`:
- `train_diffusion_model_text_to_image.py`: Script for fine-tuning the Stable Diffusion model conditioned on a text prompt.
- `train_diffusion_model_inpaint.py`: Script for fine-tuning the Stable Diffusion model conditioned on a text prompt, and random masks of an image.
- `generate_synthetic_images.py`: Script for generating synthetic images using fine-tuned models
- `sample_datasets.py`: A few example Torch datasets that are compatible with this package. Includes a `FitzDataset` sample that can be used once the original images are downloaded (see [Data](#data))

## Data

The original Fitzpatrick17k dataset can be installed from [this GitHub link.](https://github.com/mattgroh/fitzpatrick17k) The images need to be downloaded from original source. We include clean training and held-out splits in the `fitz_metadata` folder.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).