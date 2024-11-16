import webdataset as wds
from pathlib import Path
import json
import os

shard_size = 5000

input_directory = Path("/n/data1/hms/dbmi/manrai/derm/synderm2024/complete_10k")
output_directory = Path("/n/data1/hms/dbmi/manrai/derm/synderm2024/complete_10k_webdataset_shards")

os.makedirs(output_directory, exist_ok=True)

# Collect all samples with their image and JSON paths
samples = []

for file_name in os.listdir(input_directory):
    if file_name.endswith('.png'):
        base_name = os.path.splitext(file_name)[0]
        img_path = input_directory / f"{base_name}.png"
        json_path = input_directory / f"{base_name}.json"

        if os.path.exists(json_path):
            samples.append((img_path, json_path))

# Create shards
shard_number = 0
for i in range(0, len(samples), shard_size):
    shard_samples = samples[i:i + shard_size]
    shard_filename = os.path.join(output_directory, f"shard-{shard_number:05d}.tar")

    with wds.TarWriter(shard_filename) as tar:
        for img_path, json_path in shard_samples:
            with open(img_path, 'rb') as img_file:
                img_bytes = img_file.read()

            with open(json_path, 'rb') as json_file:
                json_bytes = json_file.read()

            # Use a unique identifier for the sample
            sample_id = os.path.splitext(os.path.basename(img_path))[0]

            # Prepare the sample dictionary
            sample = {
                "__key__": sample_id,
                "png": img_bytes,
                "json": json_bytes
            }

            tar.write(sample)

    print(f"Created {shard_filename} with {len(shard_samples)} samples.")
    shard_number += 1
