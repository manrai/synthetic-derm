# Synderm

[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://synthetic-derm.readthedocs.io/en/latest/index.html)
[![Complete Dataset](https://img.shields.io/badge/complete_dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/tbuckley/synthetic-derm-1M)
[![Train Dataset](https://img.shields.io/badge/train_dataset-HuggingFace-yellow.svg)](https://huggingface.co/datasets/tbuckley/synthetic-derm-1M-train)
[![PyPI](https://img.shields.io/pypi/v/synderm.svg)](https://pypi.org/project/synderm/)

Synderm generates synthetic dermatology images with diffusion models and mixes them into
classifier training. It is the package behind *Augmenting medical image classifiers with
synthetic data from latent diffusion models* (Sagers, Diao, Melas-Kyriazi et al.) and the
one-million-image synthetic dataset on Hugging Face.

Version 0.2 (October 2026) replaces the Stable Diffusion 2.1 / 1.5 code of the paper with three
current generator backends behind one interface, trains LoRA adapters through the upstream
diffusers scripts instead of custom training loops, and trains classifiers with timm instead of
fastai. The code that produced the paper is kept unchanged in `paper_experiments/`.

## Supported generators

| backend | model | text-to-image | reference-image edit | hard-mask inpaint | LoRA training | license |
|---|---|---|---|---|---|---|
| `qwen-image-2.1` | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) (7B) | yes | yes | no | t2i, i2i | Qwen Research (research only) |
| `flux-3-klein` | [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) (5.6B) | yes | yes | no | t2i, i2i | FLUX Kommunity (non-commercial) |
| `sdxl` | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) + SDXL inpainting | yes | no | yes | t2i | CreativeML OpenRAIL++-M |

The FLUX repositories are gated: accept their licenses on Hugging Face and log in with
`hf auth login` before use. See `MODERNIZATION_PLAN.md` for how these were chosen and for the
comparison of every open model considered.

## Installation

```bash
# PyTorch with the CUDA build for your driver first: https://pytorch.org/get-started/locally/
pip install -r requirements.txt     # includes diffusers from GitHub (needed for the 2026 models)
pip install -e .
```

Model weights total roughly 150 GB. Point the Hugging Face cache at a large disk before the
first download:

```bash
export HF_HOME=/workspace/.cache/huggingface
export SYNDERM_DATA_DIR=/workspace/data        # real images and unpacked shards
export SYNDERM_WORK_DIR=/workspace/synderm     # adapters, generations, vignette outputs
```

## Quick start

### 1. Real images

Every synderm dataset yields dictionaries with `id`, `image` (PIL, RGB) and `label`
(lowercase, hyphenated). `load_sample` fetches a fixed six-label sample of Fitzpatrick 17k from
Atlas Dermatologico (64 train + 32 validation images per label), verifies each file against
its md5 hash, and raises if the source is unavailable; there is no silent fallback.

```python
from synderm.data import load_sample, ImageFolderDataset, FitzpatrickDataset

train = load_sample(split="train")              # cached under $SYNDERM_DATA_DIR
mine = ImageFolderDataset("my_images")           # <root>/<label>/<image>
full = FitzpatrickDataset("/data/fitz17k", "fitz_metadata/fitzpatrick17k_10label_clean_training.csv")
```

### 2. Generate

```python
from synderm.backends import get_backend

gen = get_backend("qwen-image-2.1")
images = gen.text_to_image(gen.caption("psoriasis"), num_images_per_prompt=4, seed=0)
edits = gen.edit(gen.edit_caption("psoriasis"), train[0]["image"], num_images_per_prompt=2)

sdxl = get_backend("sdxl")
from synderm.backends import masks
inpainted = sdxl.inpaint(sdxl.caption("psoriasis"), train[0]["image"], masks.center_mask(train[0]["image"].size))
```

### 3. Fine-tune a LoRA

One adapter covers every label in the dataset; the disease name goes in the caption.

```python
from synderm.fine_tune import train_lora

weights = train_lora("qwen-image-2.1", train, "/workspace/synderm/lora/qwen-t2i",
                     mode="text-to-image", resolution=768, max_train_steps=800, rank=16)
gen = get_backend("qwen-image-2.1", lora_path=weights)
```

`mode="image-to-image"` trains an editing adapter on same-label image pairs (Qwen and FLUX).

### 4. Generate a dataset

```python
from synderm.generation import generate_synthetic_dataset

synthetic = generate_synthetic_dataset(gen, "/workspace/synderm/generations/qwen-t2i",
                                       method="text-to-image", dataset=train, num_per_image=10)
```

Writes `<out>/<label>/<id>.png` with a JSON per image (`ref_id`, `prompt`, `seed`, `generator`,
`method`), the layout `huggingface/compile_webdataset_v2.py` turns into shards.

### 5. Train a classifier

```python
from synderm.utils import synthetic_train_val_split
from synderm.classify import ClassificationDataset, build_transforms, train_classifier

train_mix, val = synthetic_train_val_split(train, synthetic, per_class_test_size=8,
                                           n_real_per_class=32, n_synthetic_per_real=10, random_state=0)
result = train_classifier(ClassificationDataset(train_mix, train.labels, build_transforms(augment=True)),
                          ClassificationDataset(val, train.labels), train.labels,
                          test_dataset=ClassificationDataset(load_sample(split="val"), train.labels))
print(result.test_metrics["accuracy"], result.test_metrics["ci_low"], result.test_metrics["ci_high"])
```

## Vignettes

Notebooks under `synderm/docs/examples/` run end to end on one 32 GB GPU against real images.
Each has a `parameters` cell (`SMALL=True` keeps runs short) and runs with papermill:

```bash
papermill synderm/docs/examples/01_zero_shot_generation.ipynb out.ipynb -p SMALL True
```

| notebook | shows |
|---|---|
| `00_public_dataset` | browsing and streaming the 1M-image Hugging Face dataset |
| `01_zero_shot_generation` | the three backends side by side without fine-tuning |
| `02_finetune_lora` | training a LoRA and comparing with the 2024 SD 2.1 generations |
| `03_image_conditioned_augmentation` | SDXL inpaint / in-then-outpaint and Qwen editing, with the paper's embedding-distance analysis |
| `04_augment_with_public_dataset` | classifier with and without synthetic images, Wald CIs |
| `05_low_data_benchmark` | the paper's Figure 3 sweep at small scale |
| `06_synthetic_image_quality` | Fréchet/kernel distances, label consistency, reader-study export |
| `07_performance_by_skin_tone` | accuracy stratified by Fitzpatrick skin type |

## The public dataset

`tbuckley/synthetic-derm-1M` (per-label shards) and `tbuckley/synthetic-derm-1M-train`
(shuffled shards) hold the 2024 release in WebDataset format. `synderm.data.load_2024_shards`
unpacks a bounded subset into the synderm layout; vignette 00 shows direct WebDataset use.

## Reproducing the 2023 paper

`paper_experiments/` contains the original DreamBooth scripts, generation code, fastai
experiments and conda environment, unmodified. Note that the Stability AI SD 2.x weights are no
longer downloadable from their official repositories (see the note in
`paper_experiments/README.md`); the released images on Hugging Face remain the primary artifact.

## Data

Fitzpatrick 17k: https://github.com/mattgroh/fitzpatrick17k (CC BY-NC-SA 3.0; many source URLs
are now dead, the Atlas Dermatologico subset still resolves). Stanford DDI:
https://ddi-dataset.github.io/ (registration required). Clean training and held-out splits are
in `fitz_metadata/`.

## License

MIT for the code. Generated images inherit the terms of the model that produced them.
