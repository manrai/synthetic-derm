import re
import json
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from huggingface_hub import HfApi
import webdataset as wds
from PIL import Image

class InterleaveWebdataset(IterableDataset):
    def __init__(self, webdataset, buffer_size=32, shuffle=True):
        # Num workers required for shuffle across shards
        self.loader = DataLoader(webdataset, batch_size=None, num_workers=8) 
        self.iterator = iter(self.loader)

        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.buffer = []

    def __iter__(self):
        while True:
            while len(self.buffer) < self.buffer_size:
                item = next(self.iterator)
                self.buffer.append(item)
            
            # Shuffle buffer before yielding items
            if self.shuffle:
                random.shuffle(self.buffer)
            
            # Yield items from shuffled buffer
            for _ in range(self.buffer_size):
                yield self.buffer.pop(0)

api = HfApi()

labels = [
    "basal-cell-carcinoma",
    "allergic-contact-dermatitis"
]

repo_id = "tbuckley/synthetic-derm-1M"
files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

pattern = re.compile(
    r"data/shard-(?P<label>{})-finetune-text-to-image-text-to-image-(?P<index>\d{{5}})\.tar".format("|".join(labels))
)

available_shards = {label: [] for label in labels}
for file in files:
    match = pattern.match(file)
    if match:
        label = match.group("label")
        index = int(match.group("index"))
        available_shards[label].append(index)

def generate_shard_urls(label, indices):
    return [
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/data/shard-{label}-finetune-text-to-image-text-to-image-{i:05d}.tar"
        for i in indices
    ]

all_shard_urls = []
for label, indices in available_shards.items():
    all_shard_urls.extend(generate_shard_urls(label, indices))

print(f"Total shards: {len(all_shard_urls)}")

transform = transforms.ToTensor()

def custom_collate(batch):
    """
    Custom collate function to handle batches of samples.

    Args:
        batch (list): A list of tuples containing (PIL Image, JSON metadata)

    Returns:
        tuple: Batched tensors and metadata
    """
    try:
        images, metadata = zip(*batch)
        images = [transform(img) for img in images]  # Convert PIL Images to tensors
        images = torch.stack(images, dim=0)          # Stack tensors into a batch
        return images, metadata
    except Exception as e:
        print(f"Error in custom_collate: {e}")
        raise

# Create a WebDataset
dataset = (
    wds.WebDataset(all_shard_urls, shardshuffle=True)
    .shuffle(40000)
    .decode("pil")
    .to_tuple("png", "json")
)

shuffled_buffer_dataset = InterleaveWebdataset(
    webdataset=dataset,
    buffer_size=1000
)

# Configure the DataLoader
loader = DataLoader(
    shuffled_buffer_dataset,
    batch_size=32,
    collate_fn=custom_collate
)

for i, batch in enumerate(loader):
    images, metadata = batch

    print(f"Batch {i+1}:")
    for item in metadata:
        print(item["name"])
