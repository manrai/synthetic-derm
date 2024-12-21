from tqdm.auto import tqdm
import webdataset as wds
from pathlib import Path
import random
import json
import os
import argparse


# TODO: 
# 1) load all links (count/categorize them at the same time)
# 2) shuffle all links
# 3) iterate through shuffled links and shard


def parse_args():
    parser = argparse.ArgumentParser(description='Compile images and metadata into WebDataset shards')
    parser.add_argument('--input_directory', type=str, required=True,
                      help='Directory containing source images and JSON files')
    parser.add_argument('--output_directory', type=str, required=True, 
                      help='Directory where WebDataset shards will be written')
    parser.add_argument('--output_metadata_directory', type=str, required=True, 
                      help='Directory to store crawl data / counts')
    parser.add_argument('--shard_size', type=int, default=5000,
                      help='Number of samples per shard (default: 5000)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)

    print("Processing and creating shards...")

    counts = {
        'labels': {},
        'methods': {},
        'submethods': {},
        'tags': {}
    }

    for subdirectory in os.scandir(input_directory):
        if not subdirectory.is_dir():
            continue

        shard_prefix = subdirectory.name.replace("_", "-")

        base_names = []
        with tqdm(os.scandir(subdirectory.path), desc=f"Querying file names in {subdirectory.name}") as it:
            for entry in it:
                if entry.name.endswith('.png'):
                    base_name = os.path.splitext(entry.name)[0]

                    metadata = base_name.split("_")
                    label = metadata[0]
                    if label == "all":
                        continue

                    method = metadata[1]
                    submethod = metadata[2]
                    tag = metadata[3]

                    # Update global counts
                    counts['labels'][label] = counts['labels'].get(label, 0) + 1
                    counts['methods'][method] = counts['methods'].get(method, 0) + 1
                    counts['submethods'][submethod] = counts['submethods'].get(submethod, 0) + 1
                    counts['tags'][tag] = counts['tags'].get(tag, 0) + 1

                    base_names.append(base_name)

        # Save current counts after processing each subdirectory
        with open(args.output_metadata_directory, 'w') as f:
            json.dump(counts, f, indent=2)
            
        # Perform shuffle
        random.shuffle(base_names)

        shard_number = 0
        shard_samples = []
        for base_name in tqdm(base_names, desc="Creating shards from shuffled list"):
            img_path = Path(subdirectory.path) / f"{base_name}.png"
            json_path = Path(subdirectory.path) / f"{base_name}.json"

            if os.path.exists(json_path):
                with open(img_path, 'rb') as img_file:
                    img_bytes = img_file.read()

                with open(json_path, 'rb') as json_file:
                    json_bytes = json_file.read()

                sample_id = base_name

                sample = {
                    "__key__": sample_id,
                    "png": img_bytes,
                    "json": json_bytes
                }

                shard_samples.append(sample)

                if len(shard_samples) >= args.shard_size:
                    shard_filename = os.path.join(output_directory, f"shard-{shard_prefix}-{shard_number:05d}.tar")
                    with wds.TarWriter(shard_filename) as tar:
                        for sample in shard_samples:
                            tar.write(sample)

                    print(f"Created {shard_filename} with {len(shard_samples)} samples.")
                    shard_samples = []  # Reset for the next shard
                    shard_number += 1

        if shard_samples:
            shard_filename = os.path.join(output_directory, f"shard-{shard_prefix}-{shard_number:05d}.tar")
            with wds.TarWriter(shard_filename) as tar:
                for sample in shard_samples:
                    tar.write(sample)

            print(f"Created {shard_filename} with {len(shard_samples)} samples.")

if __name__ == "__main__":
    main()
