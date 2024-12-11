# Synderm

Synderm is a package designed to enhance image classification tasks using synthetic data generation. It provides tools to generate high-quality synthetic images using diffusion models, fine-tune these models on your specific datasets, and seamlessly integrate synthetic data into your training pipelines to improve classifier performance.

## Table of Contents

- [Features](#features)
- [Models](#models)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1. Creating the Dataset](#1-creating-the-dataset)
  - [2. Training the Synthetic Image Generator](#2-training-the-synthetic-image-generator)
  - [3. Generate Synthetic Images](#3-generate-synthetic-images)
  - [4. Augmenting the Classifier with Synthetic Images](#4-augmenting-the-classifier-with-synthetic-images)
- [Examples](#usage-example)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Synthetic Data Generation:** Utilize diffusion models to create high-quality synthetic images tailored to your dataset.
- **Fine-Tuning:** Adapt pre-trained diffusion models to your specific classes using minimal real data.
- **Dataset Augmentation:** Combine real and synthetic data effortlessly to enhance your training datasets.
- **Seamless Integration:** Compatible with popular deep learning frameworks like PyTorch and FastAI.
- **Flexible Configuration:** Easily customize prompts, training parameters, and data splits to fit your project's needs.

## Models

Synderm directly supports the following models for image generation:

- **Inpainting:** [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- **Outpainting:** [`stabilityai/stable-diffusion-2-1-base`](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)

Other diffusion models can be used but are currently untested.

All functions assume that the training and validation datasets return entries with an `image`, `label`, and `id` field. If your dataset does not conform to this structure, please adjust it accordingly (see examples below).

## Installation
```bash
# To install from the Python Package Index:
pip install synderm

# Build from source
pip install -e .
```

*Ensure you have [PyTorch](https://pytorch.org/) and [FastAI](https://fast.ai/) installed.*

## Quick Start

### 1. Creating the Dataset

Synderm requires datasets to return entries with `image`, `label`, and `id` fields. Here's an example of how to create a custom dataset:

```python:synderm/examples/train_with_synthetic_images.ipynb
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

```python:synderm/examples/train_with_synthetic_images.ipynb
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

```python:synderm/synderm/examples/train_with_synthetic_images.ipynb
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

```python:synderm/synderm/examples/train_with_synthetic_images.ipynb
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

## Examples

Please see the notebook at `examples/train_with_synthetic_images.ipynb` shows a complete examples.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the [MIT License](LICENSE).