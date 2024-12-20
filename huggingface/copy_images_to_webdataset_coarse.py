from pathlib import Path
from tqdm import tqdm
import shutil
import json
import os

output_directory = Path("/n/data1/hms/dbmi/manrai/derm/synderm2024/complete_train")

root = Path("/n/data1/hms/dbmi/manrai/derm")

methods = {
    "finetune-text-to-image": {
        "paths_tags": [
            {"path": "nature-revisions/fitz_ddi_crossover_full_fitz/generations/text-to-image/", "tag": "fitz-ddi-crossover-full-fitz"},
            {"path": "nature-revisions/new_fitz_diseases/generations/text-to-image/", "tag": "new-fitz-diseases"},
            {"path": "generations/text-to-image", "tag": "generations1"},
            {"path": "generations-more/text-to-image", "tag": "generations2"},
            {"path": "generations-lots-more/text-to-image", "tag": "generations3"}
        ],
        "submethods": ["text-to-image"]
    },
    "pretrained-text-to-image": {
        "paths_tags": [
            {"path": "generations-pretrained/text-to-image", "tag": "generations1"}
        ],
        "submethods": ["text-to-image"]
    },
    "finetune-inpaint": {
        "paths_tags": [
            {"path": "generations/inpaint", "tag": "generations1"}
        ],
        "submethods": ["inpaint-outpaint", "inpaint"]
    },
    "pretrained-inpaint": {
        "paths_tags": [
            {"path": "generations-pretrained/inpaint", "tag": "generations1"}
        ],
        "submethods": ["inpaint-outpaint", "inpaint"]
    }
}

# First collect all image files to process
all_image_files = []
for method_key in tqdm(methods.keys(), desc="Processing methods"):
    method_info = methods[method_key]
    paths_tags = method_info["paths_tags"]
    submethods = method_info["submethods"]

    for path_tag in tqdm(paths_tags, desc=f"Processing paths for {method_key}", leave=False):

        method_path = path_tag["path"]
        tag = path_tag["tag"]

        root_method_path = root / method_path

        for label_dir in tqdm(root_method_path.iterdir(), desc=f"Processing labels for {method_key} - {tag}", leave=False):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            for submethod in tqdm(submethods, desc=f"Processing submethods for {label}", leave=False):
                submethod_path = root_method_path / label / submethod

                if not submethod_path.exists():
                    print(f"Warning: Expected path does not exist: {submethod_path}")
                    continue

                sub_output_directory = output_directory / method_key.replace("-", "_")

                # iterate over all generation numbers
                for gen_dir in submethod_path.iterdir(): 
                    if not gen_dir.is_dir():
                        continue
                    generation_num = gen_dir.name  # e.g., "01", "02"

                    for image_file in tqdm(gen_dir.glob('*.*'), desc=f"Processing images in {gen_dir.name}", leave=False):
                        if not image_file.is_file():
                            continue

                        file_name = image_file.name
                        relative_path = image_file.relative_to(root)

                        md5hash = file_name.split(".")[0]

                        unique_name = f"{label}_{method_key}_{submethod}_{tag}_{generation_num}_{md5hash}"
                        #unique_name = f"{file_prefix}-{tag}-{generation_num}-{md5hash}"

                        image_extension = image_file.suffix
                        image_path = f"{unique_name}{image_extension}"

                        # Skip if image already exists in output directory
                        image_output_path = sub_output_directory / image_path
                        os.makedirs(sub_output_directory, exist_ok=True)
                        if image_output_path.exists():
                            continue

                        metadata = {
                            "name": image_path,
                            "md5hash": md5hash,
                            "label": label,
                            "method": method_key,
                            "submethod": submethod,
                            "generation_num": generation_num,
                            "tag": tag
                        }

                        json_filename = f"{unique_name}.json"
                        json_path = sub_output_directory / json_filename
                        with open(json_path, 'w') as json_file:
                            json.dump(metadata, json_file)
                        
                        shutil.copy2(image_file, image_output_path)

