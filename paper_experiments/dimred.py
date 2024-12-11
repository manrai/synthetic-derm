# %%

import math
import os
import random
import sys
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
from IPython.display import display
from IPython import get_ipython
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TVF
from torchvision.utils import make_grid
from tqdm import tqdm

# import lukeutils

# %%

# Set autograd and device
# torch.set_grad_enabled(False)
inference_context = torch.inference_mode()
inference_context.__enter__()
device = 'cuda'

# %%

# Read
root = Path('/n/data1/hms/dbmi/manrai/derm')
csv_file = 'Fitzpatrick17k/fitzpatrick17k_10label_clean_training.csv' # Fitzpatrick17k/fitzpatrick17k.csv
df = pd.read_csv(root / csv_file)
# df = df[df['label'] == disease_class]  # filter by disease class
output_dir = root / 'embeddings'

# Get paths for real images
image_files = {}
for i, (file_hash, label) in enumerate(zip(tqdm(df['md5hash']), df['label'])):
    real_image_file = root / 'Fitzpatrick17k/finalfitz17k' / f'{file_hash}.jpg'
    synthetic_image_file_dio = root / f'generations/inpaint/all/inpaint-outpaint' / '00' / f'{file_hash}.png'
    synthetic_image_file_dti = root / f'generations/text-to-image/all/text-to-image' / '00' / f'{file_hash}.png'
    synthetic_image_file_pio = root / f'generations/inpaint/{label.replace(" ", "-")}/inpaint-outpaint' / '00' / f'{file_hash}.png'
    synthetic_image_file_pti = root / f'generations/text-to-image/{label.replace(" ", "-")}/text-to-image' / '00' / f'{file_hash}.png'
    assert real_image_file.is_file(), real_image_file
    assert synthetic_image_file_dio.is_file(), synthetic_image_file_dio
    assert synthetic_image_file_dti.is_file(), synthetic_image_file_dti
    assert synthetic_image_file_pio.is_file(), synthetic_image_file_pio
    assert synthetic_image_file_pti.is_file(), synthetic_image_file_pti
    image_files[file_hash] = (real_image_file, synthetic_image_file_dio, synthetic_image_file_dti, synthetic_image_file_pio, synthetic_image_file_pti)

# Disease labels
disease_labels_categorical = df['label'].astype('category')
disease_labels_numerical = disease_labels_categorical.cat.codes
disease_label_names = disease_labels_categorical.dtype.categories

# %%

from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection

# Create model
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval().to(device)
print(sum(p.numel() for p in model.parameters()))

def transform(x):
    return processor(images=x, return_tensors="pt").pixel_values.squeeze(0)

# %%

class ImageDataset:
    def __init__(self, files_list: list[Path], transform) -> None:
        self.files_list = files_list
        self.transform = transform

    def __getitem__(self, i: int):
        path = str(self.files_list[i])
        image = Image.open(path).convert('RGB')
        image = self.transform(image) if self.transform is not None else image
        return path, image
    
    def __len__(self):
        return len(self.files_list)


# Create
datasets = [
    ImageDataset([v[i] for v in image_files.values()], transform=transform) 
    for i in range(5)  # 1 real dataloader and 4 synthetic dataloaders
]
dataloaders: Sequence[Sequence[tuple[str, torch.Tensor]]] = [
    DataLoader(datasets[i], batch_size=32, num_workers=8, shuffle=False)
    for i in range(5)  # 1 real dataloader and 4 synthetic dataloaders
]
print([(len(datasets[i]), len(dataloaders[i])) for i in range(5)])

# %%

# def compute_embedding(images):
#     return model(images.to(device)).image_embeds.detach().cpu().numpy()

# # Compute embeddings
# all_hashes = [[], [], [], [], []]
# all_embeds = [[], [], [], [], []]
# for dataloader_idx, dataloader in enumerate(dataloaders):
#     for (paths, images) in tqdm(dataloader):
#         embeds = compute_embedding(images)
#         hashes = [Path(f).stem for f in paths]
#         all_hashes[dataloader_idx].extend(hashes)
#         all_embeds[dataloader_idx].extend(embeds)
#     all_embeds[dataloader_idx] = np.stack(all_embeds[dataloader_idx], axis=0)
# all_embeds = np.concatenate(all_embeds, axis=0)
# all_embeds_labels = (np.arange(len(all_embeds)) < len(all_embeds) // 5).astype(int)
# for hashes in all_hashes:
#     assert hashes == all_hashes[0]
# all_hashes = [h for hashes in all_hashes for h in hashes]  # same length as all_embeds
# print(all_embeds.shape)
# print(all_embeds_labels.shape)

# # # Save
# # torch.save((all_embeds, all_embeds_labels, all_hashes), 'tmp/dimred-tmp.pth')

# Load
(all_embeds, all_embeds_labels, all_hashes) = torch.load('tmp/dimred-tmp.pth')

# Normalize
all_embeds = all_embeds / np.linalg.norm(all_embeds, axis=1, keepdims=True)

# %%

# Plot settings
# Note: For more styles see https://www.dunderdata.com/blog/view-all-available-matplotlib-styles
# Note: If you are doing segmentation, you might use: from skimage.color import label2rgb
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_theme()
sns.set_theme(
    font=["serif"], 
    font_scale=1.5, 
    rc={'figure.figsize': (16, 9),}
)

# %%

from sklearn.decomposition import PCA

# Dimensionality reduction  
pca = PCA(n_components=2)
X_pca = pca.fit_transform(all_embeds)

# %%

# Show
N = len(disease_labels_categorical)
for i, name in enumerate(['Finetune Inpaint-Outpaint', 'Finetune Text-to-Image', 'Pretrained Inpaint-Outpaint', 'Pretrained Text-to-Image']):
    tmp_X_pca = np.concatenate((X_pca[:N], X_pca[(i+1)*N:(i+2)*N]))
    tmp_X_labels = all_embeds_labels[:2*N]

    # Show 
    plot = plt.scatter(tmp_X_pca[:,0], tmp_X_pca[:,1], c=tmp_X_labels, s=10, cmap=sns.color_palette("crest", as_cmap=True))
    plt.legend(handles=plot.legend_elements()[0], labels=['Real', name])
    plt.title('PCA of Embeddings from Real and Synthetic Images (Colored by Real/Synthetic)')
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/pca-real-{name.lower().replace(" ", "-")}.png', bbox_inches='tight', )
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/pca-real-{name.lower().replace(" ", "-")}.pdf', bbox_inches='tight', )
    plt.show()

    # Show by disease
    plot = plt.scatter(tmp_X_pca[:,0], tmp_X_pca[:,1], c=pd.concat((disease_labels_numerical, disease_labels_numerical)), s=10, cmap=sns.color_palette("viridis", as_cmap=True))
    plt.legend(handles=plot.legend_elements()[0], labels=disease_label_names.tolist())
    plt.title('PCA of Embeddings from Real and Synthetic Images (Colored by Disease)')
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/pca-real-{name.lower().replace(" ", "-")}-colored-by-disease.png', bbox_inches='tight', )
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/pca-real-{name.lower().replace(" ", "-")}-colored-by-disease.pdf', bbox_inches='tight', )
    plt.show()

# %%

# Note: UMAP is SLOW

import umap

# Dimensionality reduction  
reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(all_embeds)

# Show
N = len(disease_labels_categorical)
for i, name in enumerate(['Finetune Inpaint-Outpaint', 'Finetune Text-to-Image', 'Pretrained Inpaint-Outpaint', 'Pretrained Text-to-Image']):
    tmp_X_umap = np.concatenate((X_umap[:N], X_umap[(i+1)*N:(i+2)*N]))
    tmp_X_labels = all_embeds_labels[:2*N]

    # Show 
    plot = plt.scatter(tmp_X_umap[:,0], tmp_X_umap[:,1], c=tmp_X_labels, s=10, cmap=sns.color_palette("crest", as_cmap=True))
    plt.legend(handles=plot.legend_elements()[0], labels=['Real', name])
    plt.title('UMAP of Embeddings from Real and Synthetic Images (Colored by Real/Synthetic)')
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/umap-real-{name.lower().replace(" ", "-")}.png', bbox_inches='tight', )
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/umap-real-{name.lower().replace(" ", "-")}.pdf', bbox_inches='tight', )
    plt.show()

    # Show by disease
    plot = plt.scatter(tmp_X_umap[:,0], tmp_X_umap[:,1], c=pd.concat((disease_labels_numerical, disease_labels_numerical)), s=10, cmap=sns.color_palette("viridis", as_cmap=True))
    plt.legend(handles=plot.legend_elements()[0], labels=disease_label_names.tolist())
    plt.title('UMAP of Embeddings from Real and Synthetic Images (Colored by Disease)')
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/umap-real-{name.lower().replace(" ", "-")}-colored-by-disease.png', bbox_inches='tight', )
    plt.savefig(f'/n/data1/hms/dbmi/manrai/derm/embeddings/umap-real-{name.lower().replace(" ", "-")}-colored-by-disease.pdf', bbox_inches='tight', )
    plt.show()


# %%

# Create CSV for James
csv_hash = df['md5hash'].tolist() * 5
csv_name = [*(['Real'] * N), *(['Finetune Inpaint-Outpaint'] * N), *(['Finetune Text-to-Image'] * N), *(['Pretrained Inpaint-Outpaint'] * N), *(['Pretrained Text-to-Image'] * N)]
csv_X_pca_1 = X_pca[:, 0]
csv_X_pca_2 = X_pca[:, 1]
csv_X_umap_1 = X_umap[:, 0]
csv_X_umap_2 = X_umap[:, 1]
csv_labels = df['label'].tolist() * 5
df_csv = pd.DataFrame([csv_hash, csv_name, csv_X_pca_1, csv_X_pca_2, csv_X_umap_1, csv_X_umap_2, csv_labels]).T
df_csv = df_csv.rename(columns={0: 'md5hash', 1: 'image_type', 2: 'pca_1', 3: 'pca_2', 4: 'umap_1', 5: 'umap_2', 6: 'disease_label'})
df_csv.to_csv('/n/data1/hms/dbmi/manrai/derm/embeddings/csv-for-james.tsv', sep='\t')
np.save('/n/data1/hms/dbmi/manrai/derm/embeddings/full-embeddings', all_embeds)

# %%

# %%

# Random permutation
N = X_pca.shape[0] // 5
randperm = np.random.permutation(N)
plt.hist(np.sum((X_pca[:N] - X_pca[N:2*N]) ** 2, axis=1), bins=np.arange(0, 0.30, 0.003))
plt.title('Histogram of distances in 2-dimensional PCA space of real and synthetic image pairs')
plt.xlim(0, 0.3)
plt.show()
plt.hist(np.sum((X_pca[:N] - X_pca[N:2*N][randperm]) ** 2, axis=1), bins=np.arange(0, 0.30, 0.003))
plt.title('Histogram of distances in 2-dimensional PCA space of real image to a random synthetic image')
plt.xlim(0, 0.3)
plt.show()


# %%

# Random permutation, cosine distance in embedding space
_all_embeds_normalized = all_embeds / np.linalg.norm(all_embeds, axis=1, keepdims=True)
randperm = np.random.permutation(N)
plt.hist(1 - (_all_embeds_normalized[:N] * _all_embeds_normalized[N:2*N]).sum(axis=-1), bins=np.arange(0, 0.4, 0.004))
plt.title('Histogram of cosine distances in 512-dimensional embedding space of real and synthetic image pairs')
plt.xlim(0, 0.4)
plt.show()
plt.hist(1 - (_all_embeds_normalized[:N] * _all_embeds_normalized[N:2*N][randperm]).sum(axis=-1), bins=np.arange(0, 0.4, 0.004))
plt.title('Histogram of cosine distances in 512-dimensional embedding space of real image to a random synthetic image')
plt.xlim(0, 0.4)
plt.show()

# %%
