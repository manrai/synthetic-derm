from pathlib import Path
from tqdm import tqdm
import random
import shutil
import json
import os

#output_directory = Path("/n/data1/hms/dbmi/manrai/derm/synderm2024/complete_10k")
output_directory = Path("/n/data1/hms/dbmi/manrai/derm/synderm2024/complete")

#limit = 10000
limit = None

root = Path("/n/data1/hms/dbmi/manrai/derm")

methods = {
    "finetune_inpaint": {
        "path": "generations/inpaint",
        "submethods": ["inpaint-outpaint", "inpaint"] 
    },
    "finetune_text_to_image": {
        "path": "generations/text-to-image",
        "submethods": ["text-to-image"]
    },
    "pretrained_inpaint": {
        "path": "generations-pretrained/inpaint",
        "submethods": ["inpaint-outpaint", "inpaint"] 
    },
    "pretrained_text_to_image": {
        "path": "generations-pretrained/text-to-image",
        "submethods": ["text-to-image"]
    }
}

labels = ["all", "folliculitis", "neutrophilic-dermatoses", "sarcoidosis",
          "allergic-contact-dermatitis", "lichen-planus", "photodermatoses",
          "squamous-cell-carcinoma", "basal-cell-carcinoma", "lupus-erythematosus", "psoriasis"]

# First collect all image files to process
all_image_files = []
for method_key in methods.keys():
    method_path = methods[method_key]["path"]
    submethods = methods[method_key]["submethods"]
    for label in labels:
        for submethod in submethods:
            submethod_path = root / method_path / label / submethod

            if not os.path.exists(submethod_path):
                continue

            # Iterate over generation numbers
            for gen_dir in submethod_path.iterdir():
                if not gen_dir.is_dir():
                    continue
                generation_num = gen_dir.name  # e.g., "01", "02"

                for image_file in gen_dir.glob('*.*'):
                    if not image_file.is_file():
                        continue
                    all_image_files.append((method_key, label, submethod, generation_num, image_file))

if limit is not None:
    random.seed(42)
    if len(all_image_files) > limit:
        all_image_files = random.sample(all_image_files, limit)

for method_key, label, submethod, generation_num, image_file in tqdm(all_image_files, desc="Processing images"):
    file_name = image_file.name
    relative_path = image_file.relative_to(root)

    md5hash = file_name.split(".")[0]
    unique_name = f"{method_key}_{label}_{submethod}_{generation_num}_{md5hash}"

    image_extension = image_file.suffix
    image_path = f"{unique_name}{image_extension}"

    # Skip if image already exists in output directory
    image_output_path = output_directory / image_path
    if os.path.exists(image_output_path):
        continue

    metadata = {
        "name": image_path,
        "md5hash": md5hash,
        "label": label,
        "method": method_key,
        "submethod": submethod,
        "generation_num": generation_num
    }

    json_filename = f"{unique_name}.json"
    json_path = output_directory / json_filename
    with open(json_path, 'w') as json_file:
        json.dump(metadata, json_file)
    
    shutil.copy2(image_file, image_output_path)
