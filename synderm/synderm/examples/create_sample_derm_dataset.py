from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

"""
This script creates a small sample dataset from a larger synthetic dermatology dataset hosted on HuggingFace (10k samples).
It downloads images from the 'tbuckley/synthetic-derm-10k' dataset and creates a balanced sample by:
- Taking 32 images from each disease class
- Saving them to a local directory structure organized by disease label

The sample dataset is useful for:
- Quick model prototyping and testing
- Validating training pipelines
- Debugging data loading code
- Creating minimal examples
"""

cache_directory = "/n/scratch/users/t/thb286/hf_cache"
dataset = load_dataset("tbuckley/synthetic-derm-10k", cache_dir=cache_directory, streaming=True)

labels = ["folliculitis", "neutrophilic-dermatoses", "sarcoidosis",
          "allergic-contact-dermatitis", "lichen-planus", "photodermatoses",
          "squamous-cell-carcinoma", "basal-cell-carcinoma", "lupus-erythematosus", "psoriasis"]

output_dir = Path("sample_dataset")

sampled_datasets = []
counts = {label: 0 for label in labels}
for entry in tqdm(dataset["train"]):
    label = entry["json"]["label"]
    if all([i >= 32 for i in counts.values()]):
        break

    if label in labels:
        print(counts)
        if counts[label] >= 32:
            continue

        counts[label] += 1

        image = entry["png"]
        example_num = counts[label]
        label_dir = output_dir / label
        image_path = label_dir / f"{example_num:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
    label_dir = output_dir / label
