import webdataset as wds
from pathlib import Path
import json
import os
import argparse
from tqdm.auto import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Compile images and metadata into WebDataset shards')
    parser.add_argument('--input_directory', type=str, required=True,
                      help='Directory containing source images and JSON files')
    parser.add_argument('--output_directory', type=str, required=True, 
                      help='Directory where WebDataset shards will be written')
    parser.add_argument('--shard_size', type=int, default=5000,
                      help='Number of samples per shard (default: 5000)')
    return parser.parse_args()

# exclude "all" images 
# change shard names

def main():
    args = parse_args()
    
    input_directory = Path(args.input_directory)
    output_directory = Path(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)

    shard_number = 0
    shard_samples = []

    print("Processing and creating shards...")
    with tqdm(os.scandir(input_directory), desc="Processing files") as it:
        for entry in it:
            if entry.name.endswith('.png'):
                base_name = os.path.splitext(entry.name)[0]
                img_path = input_directory / f"{base_name}.png"
                json_path = input_directory / f"{base_name}.json"

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
                        shard_filename = os.path.join(output_directory, f"shard-{shard_number:05d}.tar")
                        with wds.TarWriter(shard_filename) as tar:
                            for sample in shard_samples:
                                tar.write(sample)

                        print(f"Created {shard_filename} with {len(shard_samples)} samples.")
                        shard_samples = []  # Reset for the next shard
                        shard_number += 1

    if shard_samples:
        shard_filename = os.path.join(output_directory, f"shard-{shard_number:05d}.tar")
        with wds.TarWriter(shard_filename) as tar:
            for sample in shard_samples:
                tar.write(sample)

        print(f"Created {shard_filename} with {len(shard_samples)} samples.")

if __name__ == "__main__":
    main()
