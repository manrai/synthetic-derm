from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

"""
This script creates a sample dataset with separate training and validation sets from a larger synthetic dermatology dataset hosted on HuggingFace (10k samples).
It downloads images from the 'tbuckley/synthetic-derm-10k' dataset and creates a balanced sample by:
- Taking 64 images from each disease class for training
- Taking 32 images from each disease class for validation
- Saving them to local directories organized by disease label within 'train' and 'val' folders

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
train_dir = output_dir / "train"
val_dir = output_dir / "val"

# Initialize counts for each label in train and val
counts = {label: {"train": 0, "val": 0} for label in labels}

for entry in tqdm(dataset["train"]):
    label = entry["json"]["label"]
    if label not in labels:
        continue

    # Check if both train and val counts have reached their limits
    if counts[label]["train"] >= 110 and counts[label]["val"] >= 32:
        continue

    image = entry["png"]

    if counts[label]["train"] < 110:
        counts[label]["train"] += 1
        example_num = counts[label]["train"]
        label_dir = train_dir / label
        image_path = label_dir / f"{example_num:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)
    elif counts[label]["val"] < 32:
        counts[label]["val"] += 1
        example_num = counts[label]["val"]
        label_dir = val_dir / label
        image_path = label_dir / f"{example_num:04d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

    # Optional: Print counts for debugging
    print(f"{label} - Train: {counts[label]['train']}, Val: {counts[label]['val']}")

    # Check if all labels have reached their limits
    if all(counts[l]["train"] >= 110 and counts[l]["val"] >= 32 for l in labels):
        break
