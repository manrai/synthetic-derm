# Synderm modernization plan (October 2026)

Status: draft, written 2026-10-21.

The plan has three stages that build on each other:

- **Stage A, now, this machine.** Modernize the package and vignettes with small image sets on
  one RTX 5090. No new money.
- **Stage B, open models at scale.** Regenerate the full synthetic dataset with the three
  open-weight models on a multi-GPU box, rerun the paper's experiments, push new shards.
- **Stage C, proprietary models with funding.** Generate a smaller companion dataset from the
  top closed models via API and run the same experiments on it.

Sections 1 and 2 are background: where the package stands and which models to support. The
appendix at the end compares every open model considered.

**Branching.** All of this work happens on a git branch, not on `main`. `main` stays as the
code that produced the 2023 paper and the 2024 dataset so the published results remain
reproducible from the repo. Suggested layout:

- `modernize` is the long-lived integration branch for the whole effort.
- Each Stage A item gets its own short-lived branch off `modernize` (for example
  `modernize/backend-interface`, `modernize/timm-classifier`, `modernize/vignettes`) and
  merges back by pull request.
- Tag `main` as `v0.1.2-paper` before starting, so the old state has a stable name.
- The 2023 paper's code is never deleted. `paper_experiments/` stays where it is, untouched
  (see A5). The follow-up paper will report the original model's results alongside the new
  ones, so the original pipeline must remain runnable. The package's own SD-era loops in
  `synderm/fine_tune/` are late-2024 rewrites that never produced a published image; they are
  deleted, and the `v0.1.2-paper` tag keeps them in history.
- `modernize` merges to `main` only when Stage A is complete and the vignettes run end to end.
- No PyPI release until the maintainer has reviewed and tested the branch. The merge to `main`
  and the `synderm` 0.2 upload are separate, maintainer-triggered steps; nothing in Stage A
  publishes automatically.
- Stage B and Stage C experiment code lives under `paper_experiments/2026/` on the same branch
  lineage; generated images and weights never enter git and stay on `/workspace`.

## 1. Where things stand

### The paper

*Augmenting medical image classifiers with synthetic data from latent diffusion models*
(Sagers, Diao, Melas-Kyriazi et al.) fine-tuned Stable Diffusion 2.1-base (text-to-image) and
SD 1.5 inpainting (inpaint, in-then-outpaint) with DreamBooth, one model per disease class, and
generated 458,920 synthetic images from Fitzpatrick 17k (nine most common conditions, "F17k-9")
and Stanford DDI. Classifiers were EfficientNetV2-M trained in fastai.

Findings the new experiments should be able to reproduce:

- 10 synthetic per real improved nine-way accuracy by up to 13 points at 16 to 64 real images
  per class, with and without image transforms. Gains vanished at 228 per class.
- Dose-response saturated above roughly 50 synthetic images per real image.
- Physician-rated photorealism correlated with classifier gain (coefficient 0.26, p = 0.003).
- Synthetic data improved benign vs malignant accuracy on DDI across all Fitzpatrick skin type
  groups.
- Readers flagged 72 to 75 percent of synthetic images as synthetic; only 38 percent of
  synthetic images were judged consistent with their label vs 71 percent of real images.

### The package

| Component | File(s) | State |
|---|---|---|
| Text-to-image DreamBooth loop | `synderm/fine_tune/text_to_image_diffusion.py` (946 lines) | Copied from an old diffusers example; SD 2.1 only |
| Inpaint DreamBooth loop | `synderm/fine_tune/inpaint_diffusion.py` (706 lines) | Same; SD 1.5 inpainting only |
| Generation | `synderm/generation/generate.py` | Hardwired to two SD pipelines |
| Classifier experiments | `paper_experiments/Experiments/*.py` | fastai, EfficientNetV2-M, wandb, hardcoded HMS cluster paths |
| HF dataset build | `huggingface/*.py` | WebDataset shards; shard names carry no generator-model field |
| Vignettes | `synderm/docs/examples/*.ipynb` | See section 3, Stage A |
| Pins | `requirements.txt` | diffusers 0.31, torch 2.5.1 |

Broken as of 2026-10-21:

- `stabilityai/stable-diffusion-2-1-base` (and the other Stability SD 2.x repos) return HTTP 401
  to anonymous users; the official weights are no longer downloadable, only community mirrors.
- `runwayml/stable-diffusion-inpainting` redirects to a community mirror.
- diffusers 0.31 predates every model worth using now.
- `imagenette2_subset/` (658 MB of ImageNet dog and horn photos) is checked into the repo.

Existing HF datasets (last updated December 2024):

- `tbuckley/synthetic-derm-1M` (264 files, per-label shards named
  `shard-{label}-{method}-{submethod}-{index}.tar`)
- `tbuckley/synthetic-derm-1M-train` (175 files, shuffled; groups finetune-inpaint,
  finetune-text-to-image, pretrained-inpaint, pretrained-text-to-image)
- `tbuckley/synthetic-derm-10k` (small sample used by `create_sample_derm_dataset.py`)

Real data: the clean Fitzpatrick 17k training split in `fitz_metadata/` has 3,530 images across
10 labels (the nine paper classes plus folliculitis); the held-out set has 400 (40 per label).
Images are not in the repo and must be fetched from source URLs. Two unofficial mirrors exist on
HF (`ZYXue/Fitzpatrick_17k`, `spycoder/fitzpatrick`); license and completeness need checking
before vignettes depend on them. DDI requires registration and prohibits commercial use.

### Hardware on this machine

- One NVIDIA RTX 5090, 32 GB VRAM. torch 2.8.0+cu128 installed; diffusers not installed.
- Root disk 50 GB (46 GB free). `/workspace` is network storage with effectively unbounded
  capacity. All weights, HF cache, and generations go there.

## 2. Open-weight models to support

Selection principle: two current state-of-the-art models from labs with a track record, plus
one widely used classical baseline that keeps continuity with the 2023 paper. All three are
well known, fit a 32 GB GPU, and have upstream diffusers training scripts. The full candidate
comparison, including models that were considered and dropped, is in the appendix.

| | Qwen-Image-2.1 | FLUX 3 Image Klein | SDXL 1.0 |
|---|---|---|---|
| Role | SOTA, primary | SOTA, second lab | Classical baseline, continuity with 2023 paper |
| Creator, release | Alibaba, 2026-09-14 | Black Forest Labs, 2026-10-13 | Stability AI, 2023-07-25 |
| Size | 7B DiT + Qwen3-VL encoder | 5.6B + Qwen3-VL-4B encoder | 2.6B UNet + two CLIP encoders |
| Arena Elo, open board | 1042, #1 open | not yet listed; FLUX line led the open board 2024 to 2025 | far below (2023-era model) |
| Reputation (HF, 30 days) | 183k downloads, 3.4k likes; Qwen-Image line 290k | 93k downloads in 8 days; FLUX.1-dev 650k, 15.7k likes | 4.3M downloads, 8.4k likes; largest LoRA and tooling ecosystem |
| Text-to-image | Yes | Yes | Yes |
| Image-conditioned editing | Yes, up to 10 reference images, mask-guided local edits | Yes, up to 10 refs, bounding-box edits | No native instruction editing |
| Hard-mask inpainting | Not in diffusers | Not in diffusers | Yes, `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`; runs the paper's inpaint and in-then-outpaint unchanged |
| diffusers training | `train_dreambooth_lora_qwenimage21[_img2img].py` | `train_dreambooth_lora_flux3_klein[_img2img].py` | `train_dreambooth_lora_sdxl.py`; inpainting fine-tune via community scripts |
| diffusers inference | `QwenImage21Pipeline` | `ModularPipeline` (dev install); expects separate 9B prompt upsampler | `StableDiffusionXLPipeline`, `StableDiffusionXLInpaintPipeline` |
| VRAM on 32 GB | bf16 plus cpu offload fits | tight: 31 GB bf16 at 1024px; use FP8 (26 GB) or 768px | comfortable; fastest of the three |
| License | Qwen Research (research only) | FLUX Kommunity (non-commercial; small-revenue exception) | CreativeML OpenRAIL++-M; permissive with use restrictions |
| Gated | No | Yes, auto-approval | No |
| Commercial fallback | Qwen-Image-2512 + Qwen-Image-Edit-2511 (Apache 2.0) | FLUX.2-klein-4B (Apache 2.0) | none needed |

Why SDXL rather than SD 2.1: SD 2.1-base is now gated, SDXL is the most-downloaded open image
model by a wide margin, it has a maintained inpainting checkpoint, and it is the closest living
relative of the paper's method (UNet latent diffusion, DreamBooth, hard-mask inpainting). It
gives the follow-up paper a "2023-era" control arm against the two 2026 models.

Generation methods per backend:

- **Text-to-image:** all three, with a LoRA trained on F17k captions.
- **Image-conditioned edit** (the 2026 analogue of in-then-outpaint): Qwen and FLUX.
- **Hard-mask inpaint and in-then-outpaint** (the paper's original methods): SDXL inpainting.
  Optionally a masked latent-blend callback for Qwen and FLUX if a like-for-like comparison is
  wanted.

## 3. Stage A: now, on this machine, small sets

Goal: a working `synderm` 0.2 with the new backends, a timm classifier path, and vignettes that
run end to end on real dermatology images on one 5090 in under an hour each. No new money.

### A0. Data prerequisite: Fitzpatrick 17k images

Steps 1 to 7 below need no real data; the synthetic 10k sample suffices for development and
tests. The smoke experiment (step 8) and vignettes 2 to 7 need real Fitzpatrick 17k images.

Checked 2026-10-21: the images cannot be rebuilt from the URLs in the repo. All sampled
dermaamin.com links (77 percent of the clean training split) return 404; atlasdermatologico
links still resolve. The official repo offers a form to request a download link from the
authors. Neither HF mirror is usable: `ZYXue/Fitzpatrick_17k` is a 5k-row VQA remix with no
md5 hashes, `spycoder/fitzpatrick` has no provenance. Images are CC BY-NC-SA 3.0.

**Vignettes: use the Atlas Dermatologico subset, fetched at runtime.** Checked 2026-10-21: all
60 sampled `atlasdermatologico.com.br/img?imageId=` URLs return JPEGs (median 60 KB), and the
site's robots.txt blocks only admin pages. The `load_sample()` helper downloads directly from
the source at about one image per second with caching under `/workspace`, so no images are
redistributed. Six labels have at least 96 Atlas images outside the held-out set: basal cell
carcinoma (257), lichen planus (230), squamous cell carcinoma (212), lupus erythematosus (108),
psoriasis (104), folliculitis (102). The vignette sample is therefore six labels at 64 train
and 32 validation each (576 images, about 35 MB, about 12 minutes to fetch). Counts use the
full `fitzpatrick17k.csv` rather than the SelfClean-filtered split, which has too few Atlas
rows; the 400 held-out hashes are always excluded. Fitzpatrick types I to VI are all present,
skewed to III and IV. The full Atlas portion (3,905 images, ~230 MB, ~1.5 hours) is also
worth caching once for vignette development.

**Smoke experiment and Stage B: need the full dataset.** Order of preference:

1. Copy the lab's existing md5-named image folder (the one `paper_experiments/` reads from on
   the HMS cluster) to `/workspace/data/fitzpatrick17k/`. Matches the clean split CSVs directly.
2. Request the download link through the authors' form on the official GitHub repo.

DDI is not needed for Stage A; request access during Stage A for Stage B.

### A1. Package

0. **Branch first.** Tag `main` as `v0.1.2-paper`, create `modernize`, and commit this plan
   there. Every item below is a branch off `modernize` merged by pull request.
1. **Delete the two SD-era training loops** in `synderm/fine_tune/` (`text_to_image_diffusion.py`,
   `inpaint_diffusion.py`, `util.py`, about 1,850 lines) and the SD-specific generation code.
   They are package rewrites from late 2024, not the scripts that produced the paper's images;
   those live in `paper_experiments/` and stay (A5). The new `synderm/fine_tune/` becomes thin
   wrappers that call the upstream diffusers DreamBooth LoRA scripts via `accelerate launch`
   with a config dict. One LoRA per generator with the disease name in the caption rather than
   one DreamBooth model per class.
2. **Add a generator backend interface** with `text_to_image(prompt)`,
   `edit(prompt, reference_image)`, and `inpaint(prompt, image, mask)`. Implement SDXL first
   (fastest to validate, all three methods), then Qwen-Image-2.1, then FLUX 3 Klein.
3. **Replace fastai** with timm backbones (EfficientNetV2-M for continuity plus one modern
   backbone such as ConvNeXt-V2 or a DINOv2 ViT) and a short plain PyTorch loop in a new
   `synderm/classify/` module. Keep the paper's protocol: n real per label in
   {1, 8, 16, 32, 64, 128, 228}, 10 synthetic per real, 5 seeds, the 400-image held-out set.
4. **Point everything at /workspace**: `HF_HOME=/workspace/hf`, outputs under
   `/workspace/synderm/`.
5. **Bump dependencies**: current diffusers (development install from GitHub is needed for
   FLUX 3 Klein and Qwen-Image-2.1), transformers, accelerate, peft, timm, bitsandbytes,
   webdataset. Drop fastai and lovely_tensors.
6. **Extend shard naming** in `huggingface/compile_webdataset*.py` to
   `shard-{label}-{generator}-{method}-{index}.tar` and add `generator` and `prompt` fields to
   each sample's JSON.
7. **Remove `imagenette2_subset/`** from the repo and from git history if size matters for
   clones.
8. **Smoke experiment**, the gate for Stage B: two classes, 32 real images each, LoRA on
   Qwen-Image-2.1 and SDXL, 10 generations per real, classifier with and without them,
   compared against the equivalent 2024 SD shards from `tbuckley/synthetic-derm-1M`. Record
   images per second per GPU at 512, 768 and 1024 px for each backend; those numbers size
   Stage B.

### A2. Vignettes

Every vignette switches from Imagenette and fastai to a small real dermatology sample and the
new package APIs. The sample is a fixed subset of Fitzpatrick 17k (six labels, 64 train and 32
validation per label; see A0) loaded through a `synderm.data.load_sample()` helper that fetches
the images from Atlas Dermatologico at runtime and caches them under `/workspace`. There is no
fallback: if the source is unreachable or an image fails to download, the helper raises an
error naming the failed URLs, so that the dataset becoming unavailable is noticed rather than
silently papered over. The helper verifies each cached file against the md5 hash in the CSV.
Each notebook also runs as a script with a `--small` flag so docs builds and CI can execute it.

Current notebooks and what happens to them:

| Notebook | Today | Plan |
|---|---|---|
| `synderm_dataset.ipynb` | Browses the 1M HF dataset; hardcoded HMS cache path | Keep. Update for new shard names and the `generator` field; cache under `/workspace`. |
| `generate_and_train_with_synderm.ipynb` | Imagenette dogs, SD 2.1 DreamBooth, fastai | Replace with vignettes 2 and 4 below. |
| `augment_fitz_classifier.ipynb` | Synthetic 10k sample as "real", HF shards, torchvision EfficientNet | Becomes vignette 4 with real images and timm. |
| `train_fitz_classifier2.ipynb` | Not in the docs toctree; HMS cluster paths; fastai | Becomes vignette 5. |

New vignette set, in reading order:

1. **Zero-shot generation across backends.** Same five prompts through Qwen-Image-2.1, FLUX 3
   Klein and SDXL without fine-tuning; side-by-side grid with a real example per label. Shows
   the backend interface and model loading on a 32 GB card.
2. **Fine-tune a LoRA on your own dermatology data.** 64 images per label, a few hundred
   steps, then generation with the adapter. Replaces the old DreamBooth vignette. Includes the
   old-vs-new comparison: the same labels pulled from the 2024 SD shards next to the new output.
3. **Image-conditioned augmentation.** Real image in, variants out, two ways: SDXL hard-mask
   inpaint and in-then-outpaint (the paper's method) and Qwen or FLUX instruction edits ("same
   condition, different patient, different site"). Reproduces the paper's Figure 2 analysis:
   CLIP or DINOv2 embedding distances for matched vs unmatched pairs, plus a near-duplicate
   check against the training images.
4. **Augment a classifier with the public synthetic dataset.** Stream shards from HF, mix with
   the real sample via WebDataset `RandomMix`, train a timm model, evaluate on real held-out
   images, report accuracy with a Wald confidence interval. Rewrite of the existing augment
   vignette.
5. **Low-data benchmark.** The paper's Figure 3 at small scale: sweep n real per label in
   {8, 16, 32, 64} with and without 10 synthetic per real, 3 seeds, plotted with confidence
   intervals. This is the template Stage B scales up.
6. **Evaluating synthetic image quality.** Metrics that do not need physicians: embedding
   distance to real distribution, FID or KID on a held-out real set, label consistency from a
   classifier trained on real images only, and an export of a stratified image sample to a CSV
   for a reader study in the paper's format (real, edit, text-to-image, 7 per label).
7. **Performance by skin tone.** Stratified evaluation by `fitzpatrick_scale` on the sample,
   with and without synthetic augmentation, mirroring the paper's DDI analysis without needing
   DDI access.

Docs: update the Sphinx toctree in `synderm/docs/index.rst`, add a "Models" page generated from
the backend registry, and keep the readthedocs build on Python 3.13.

### A3. Storage on /workspace

Plan for about 200 GB, 300 GB with headroom. Nothing below may land on the 50 GB root disk;
set `HF_HOME=/workspace/hf` before the first download.

| Item | Estimate | Notes |
|---|---|---|
| Qwen-Image-2.1 weights | 30 to 35 GB | Qwen3-VL text encoder is the bulk |
| FLUX 3 Image Klein weights | 40 to 60 GB | `klein` plus `klein-base`; optional 9B prompt upsampler adds ~18 GB |
| SDXL base plus inpainting checkpoint | 15 to 20 GB | fp16 variants only; no refiner |
| timm, CLIP, DINOv2 backbones | 2 GB | |
| LoRA training outputs | 15 GB | keep only the last two intermediate checkpoints |
| Smoke experiment generations | 3 GB | |
| Vignette generations | 20 to 40 GB | 1 MB per image at 768 px, ~2 MB at 1024 px |
| Real sample data | 1 to 5 GB | |
| Cached 2024 HF shards | 20 to 40 GB | stream rather than cache; the full train set is ~270 GB |
| Classifier checkpoints and logs | 5 to 10 GB | |
| Total | ~150 to 230 GB | |

Fallback models (Qwen-Image-2512 plus Edit-2511, FLUX.2-klein-4B, FLUX.1 Fill) add roughly
100 GB; fetch one only if its primary fails the smoke test.

### A4. Stage A deliverables

- `synderm` 0.2 ready on the `modernize` branch with the three backends, timm classifier
  path, no fastai. Installable with `pip install -e .` for review. PyPI upload only after the
  maintainer has reviewed and tested it.
- `paper_experiments/` untouched, with a short 2026 note on weight availability appended to its
  README, so the 2023 pipeline remains reproducible (A5).
- Seven vignettes that run on one 5090 against real dermatology images.
- Smoke experiment result and measured generation throughput per backend.

### A5. Preserving the 2023 pipeline

The follow-up paper will include results from the original Stable Diffusion models, so the
original pipeline stays in the repo and runnable. `paper_experiments/` is the code that
produced the paper: `train_dreambooth.py`, `train_dreambooth_inpaint.py`, `generate.py`,
`dataset.py`, the fastai classifier scripts in `Experiments/`, and the full conda
`environment.yml`. It is not modified except for a short note appended to its README (point 2
below). New experiment code goes in `paper_experiments/2026/`.

The package's `synderm/fine_tune/` loops are not part of this: they are late-2024 rewrites
(under 10 percent line similarity to the paper scripts) that never generated a published
image, so they are deleted in A1 step 1 and survive only in git history under the tag.

What makes it reproducible in practice:

1. **Pinned environment.** `paper_experiments/environment.yml` is the full conda record of the
   environment that ran the paper's code and is kept as is.
2. **Model weights have moved.** Checked 2026-10-21: `stabilityai/stable-diffusion-2-1-base`,
   `stable-diffusion-2-1`, and `stable-diffusion-2-inpainting` all return HTTP 401 to anonymous
   users, so the official text-to-image weights are no longer downloadable. The SD 1.5
   inpainting weights live at `stable-diffusion-v1-5/stable-diffusion-inpainting` (the old
   `runwayml/` id redirects there). A dated note appended to `paper_experiments/README.md`
   records these substitutions and lists the community mirrors of SD 2.1-base
   (`sd2-community/stable-diffusion-2-1-base` is the most downloaded); a mirror should be
   checksum-verified against the original file hashes before use. If no verified mirror is
   available, the paper's text-to-image generation step can only be reproduced from the
   released images, not re-run.
3. **The released images are the primary artifact.** All 458,920 generated images are on the
   HF datasets, with per-image JSON metadata, so every classifier result in the paper
   reproduces from the dataset plus `paper_experiments/Experiments/` without regenerating
   anything.
4. **Git tag.** `v0.1.2-paper` on `main` fixes the exact code state.

Optional check during Stage A: run `paper_experiments/train_dreambooth.py` and `generate.py` in
the conda environment for one class with the mirror weights and the paper's seed, and compare
the output against the same class in the 2024 shards. A match confirms the mirror weights are
the originals.

## 4. Stage B: open models at scale (more GPUs)

Goal: regenerate the full synthetic dataset with the three open-weight models, rerun the paper's
experiments on it, and push new shard groups to both HF datasets.

A larger GPU (H100, H200) does not unlock a better open model: the top of the open board is
7B to 9B models, and the 32B to 83B open models all rank below them (see appendix). Larger GPUs
buy throughput, 2048 px output, full fine-tuning instead of LoRA, and running FLUX 3 Klein with
its prompt upsampler co-resident.

### B1. Image budget per generator

| Set | Covers | Images |
|---|---|---|
| Core text-to-image | 10 per real, all 3,530 F17k training reals | 35,300 |
| Core image-conditioned (edit, or inpaint for SDXL) | 10 per real | 35,300 |
| Dose-response | text-to-image to 150 per real at 64 per class, 9 classes | 86,400 |
| Extended labels | next 46 F17k conditions, 10 per real (paper supplement) | ~60,000 |
| DDI | 560 training reals, 10 per real, two methods | 11,200 |
| Total per generator | | ~228,000 |
| Total, three generators | | ~685,000 |

That is comparable to the original 458,920, with three generators instead of one.

### B2. Compute

Throughput must be measured in the Stage A smoke test; the planning figure below assumes 8
seconds per 1024 px image on one 5090 for a 7B flow model at 40 steps, roughly 3 seconds at
512 px, and about 2 seconds for SDXL at 1024 px. The paper used 512 px; 768 px is a reasonable
modern compromise.

| | 1 GPU | 4 GPUs | 8 GPUs |
|---|---|---|---|
| Core (70,600 at 1024 px), Qwen or FLUX | 157 h | 39 h | 20 h |
| Everything (228,000 at 1024 px), Qwen or FLUX | 507 h | 127 h | 63 h |
| Everything at 768 px, Qwen or FLUX | ~250 h | ~63 h | ~32 h |
| Everything, SDXL at 1024 px | ~130 h | ~32 h | ~16 h |
| Everything, all three generators, 768 px | ~630 h | ~160 h | ~80 h |

LoRA training is small by comparison: one adapter per generator per dataset, a few GPU-hours
each. Classifier sweeps (7 n-values x 4 conditions x 5 seeds x 3 methods x 3 generators) are
about 1,300 short runs and fit on one GPU in a few days.

Storage: at 768 px PNG roughly 1 MB per image, so about 700 GB for everything. Fits on
`/workspace`; the existing HF dataset is 600 GB for comparison.

### B3. Experiments

Rerun the paper's five experiment families with the new generators, using the Stage A
classifier code and the same held-out sets:

1. n real per label sweep with and without synthetic data and image transforms.
2. Dose-response and saturation.
3. Class imbalance with synthetic upsampling vs duplication.
4. External validation on DDI (after confirming the DDI license allows generation from its
   images with each model's license).
5. Benign vs malignant by skin tone on DDI.

Two new comparisons the 2023 paper could not run:

6. **Generator comparison** at fixed classifier protocol: the paper's SD 2.1 and SD 1.5
   shards (re-evaluated with the new classifier code, and cross-checked against the paper's
   fastai numbers via `paper_experiments/`) vs SDXL vs each 2026 model. SDXL vs Qwen and FLUX
   isolates how much three years of generator progress is worth for augmentation.
7. **Zero-shot vs LoRA**: whether modern models need fine-tuning at all for useful augmentation.

### B4. Stage B deliverables

- New shard groups on `synthetic-derm-1M` and `synthetic-derm-1M-train` named by generator.
- Results tables and figures in the paper's format, suitable for a follow-up preprint.
- A reader-study image export (vignette 6) ready for physicians.

## 5. Stage C: proprietary models with funding

Goal: a smaller companion dataset generated by the top closed models through their APIs, large
enough to run the core experiments and a reader study, released alongside the open-model shards.

### C1. Image budget per proprietary model

| Tier | Covers | Images |
|---|---|---|
| A. Core | Text-to-image, 10 per real, all F17k training reals | 35,000 |
| B. Core + edits | A plus image-conditioned edits, 10 per real | 71,000 |

Dose-response and extended labels stay on open models (Stage B).

### C2. Price per model

Per 1,000 images at 1024x1024 from the Artificial Analysis leaderboard (2026-10-21), checked
against OpenAI and Google pricing pages. OpenAI and Google halve image output cost in batch mode.

| Model (arena Elo) | $ per 1k | Tier A (35k) | Tier B (71k) |
|---|---|---|---|
| GPT Image 2.5 Sunburst, max (1222) | 211, batch ~105 | $7,400 / batch $3,700 | $15,000 / batch $7,500 |
| GPT Image 2, medium quality (1186 at high) | ~53 | ~$1,900 | ~$3,800 |
| Nano Banana 2.1, Google (1176) | 33.6, batch 16.8 | $1,180 / batch $590 | $2,390 / batch $1,190 |
| MAI-Image-2.6, Microsoft (1166) | 38.9 (unofficial) | $1,360 | $2,760 |
| Grok Imagine Image 2.0 (1166) | 60 | $2,100 | $4,260 |
| Muse Image, Meta (1127) | 10 (preview) | $350 | $710 |
| FLUX 3 Image, BFL API (1117) | 48 | $1,680 | $3,410 |
| Seedream 5.0 Pro (1100) | 90 | $3,150 | $6,390 |
| Qwen-Image-3.0-Pro (1095) | 40 | $1,400 | $2,840 |

### C3. Recommended portfolio and totals

Three models spanning quality and price: GPT Image 2.5 Sunburst (frontier), Nano Banana 2.1
(best value in the top five), Muse Image (cheapest credible entry).

| | List price | With batch (OpenAI + Google) |
|---|---|---|
| Tier A, all three | ~$8,900 | ~$4,600 |
| Tier B, all three | ~$18,000 | ~$9,100 |
| Tier A with GPT Image 2 medium instead of 2.5 | ~$3,400 | ~$2,500 |

Add a reader study budget (physician time for roughly 200 images per generator) and 10 to 20
percent for retries.

### C4. Risks specific to this stage

- **Safety refusals.** Clinical skin images, especially lesions on torso or genital regions,
  will trip content filters on some providers; some classes may be partially blocked.
- **Edits cost more than text-to-image** at token-priced providers; expect 10 to 30 percent
  on top of tier B.
- **Data-use terms.** Fitzpatrick 17k images are public web images. DDI prohibits commercial
  use; do not send DDI images to commercial APIs without checking its license.
- **Unofficial prices.** MAI-Image-2.6 pricing is Artificial Analysis's measurement; Muse
  Image's $0.01 is from secondary coverage of a preview announcement.
- **Redistribution.** Check each provider's terms on releasing generated images as a dataset
  before generation starts.

### C5. Stage C deliverables

- A third shard group per proprietary model on the HF datasets, with provider and model
  version in the metadata.
- Core experiment results and reader study comparing open and closed generators.

## Appendix: open-weight model comparison

All candidates considered, as of 2026-10-21. Elo is from the Artificial Analysis text-to-image
arena (open-weights view). Reputation combines Hugging Face downloads over the last 30 days,
likes, the publishing organization's track record, and age. Size is the safetensors total for
the repo, which for some models includes the text encoder.

| Model | Org | Released | Size | Elo | HF downloads 30d | HF likes | Reputation | License | Gated | Editing | Hard-mask inpaint | diffusers train script | Fits 32 GB | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen-Image-2.1 | Alibaba | 2026-09 | 7B | 1042 | 183k | 3.4k | High: top lab, #1 open, Qwen-Image line 290k/mo | Qwen Research | No | Yes | No | Yes, t2i + img2img | Yes | **In** |
| FLUX 3 Image Klein | Black Forest Labs | 2026-10 | 5.5B | n/a | 93k (8 days) | 595 | High: lab behind FLUX.1-dev (650k/mo, 15.7k likes) | FLUX Kommunity (NC) | Yes | Yes | No | Yes, t2i + img2img | Tight (FP8) | **In** |
| SDXL 1.0 | Stability AI | 2023-07 | 2.6B | low | 4.3M | 8.4k | Very high: most-used open image model, huge ecosystem | OpenRAIL++ | No | No | Yes (SDXL inpainting ckpt) | Yes | Yes | **In** (baseline) |
| Fibo 2.0 dev | Bria | 2026-10 | 8B | 1039 | 396 | 96 | Low: 3 days old, small org, little adoption | CC BY-NC 4.0 | Yes (form) | Yes, incl. masks | Yes | No (vendor trainer) | Yes | Out: too new and unproven for the paper |
| Ideogram 4 | Ideogram | 2026-05 | 9.3B | 1026 | 60k | 851 | Medium: known product, first open release, fp8/nf4 only | Ideogram NC | Yes | No | No | Yes, t2i | Yes (fp8) | Out: no editing |
| FLUX.2 dev | Black Forest Labs | 2025-11 | 32B | 1000 | 390k | 2.4k | High | FLUX NC | Yes | Yes | No | Yes | No (4-bit only) | Out: too large for scale on a 5090; option on H100 |
| FLUX.2 klein 4B | Black Forest Labs | 2026-01 | 3.9B | ~929 (9B) | 443k | 1.1k | High | Apache 2.0 | No | Yes | No | Yes | Yes | Fallback for FLUX 3 Klein (commercial) |
| Qwen-Image-2512 + Edit-2511 | Alibaba | 2025-12 | 20B | lower | 51k + 187k | 1.0k + 1.5k | High | Apache 2.0 | No | Yes | No | Yes | Yes (offload) | Fallback for Qwen-2.1 (commercial) |
| Z-Image Turbo | Alibaba Tongyi | 2025-11 | 6B | 936 | 658k | 5.5k | High adoption, distilled turbo | Apache 2.0 | No | No | No | Yes | Yes | Out: ranks below Qwen-2.1 from same company |
| HiDream-O1-Image-Dev | HiDream.ai | 2026-05 | 8.8B | 975 | 631 | 128 | Low: little adoption; HiDream-I1 had 1k likes | MIT | No | Yes | No | Partial (older script) | Yes | Out: low rank and adoption |
| Ming-Image-0.1-Design | InclusionAI | 2026-09 | 6.2B | 999 | 1.2k | 432 | Low | MIT | No | No | No | No | Yes | Out: UI design specialist |
| Cosmos3-Super-Text2Image | NVIDIA | 2026-05 | 65B | 993 | 8k | 194 | Medium: NVIDIA, but world-model line, little adoption | OpenMDW 1.1 | No | No | No | No | No | Out: large, ranks below 7B leaders |
| HunyuanImage 3.0 Instruct | Tencent | 2025-09 | 83B MoE | 987 | 42k | 1.1k | Medium-high | Tencent community | No | Yes | No | No | No | Out: 83B, ranks below 7B leaders |
| SD 3.5 Large | Stability AI | 2024-10 | 8.1B | lower | 64k | 4.1k | High | Stability community | Yes | No | No | Yes | Yes | Out: superseded; SDXL is the better-known baseline |
| FLUX.1 Fill dev | Black Forest Labs | 2024-11 | 12B | n/a | 96k | 1.3k | High | FLUX.1 NC | Yes | No | Yes | No | Yes | Reserve: true-mask model if SDXL inpainting is not enough |

## Sources

- Artificial Analysis text-to-image leaderboard: https://artificialanalysis.ai/image/leaderboard/text-to-image
- Artificial Analysis open-weights filter: https://artificialanalysis.ai/image/leaderboard/text-to-image?open-weights=true
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- OpenAI API pricing: https://developers.openai.com/api/docs/pricing
- GPT Image 2 per-image costs by quality: https://costgoat.com/pricing/openai-images
- Meta Muse Image API pricing (secondary): https://runtimewire.com/article/meta-ships-muse-image-api-at-one-cent-per-image
- MAI-Image-2.6 pricing comparison (secondary): https://book.st-hakky.com/en/data-science/mai-image-2-6-pricing-comparison
- Qwen-Image-2.1 model card: https://huggingface.co/Qwen/Qwen-Image-2.1
- FLUX 3 Image Klein model card: https://huggingface.co/black-forest-labs/flux-3-image-klein
- SDXL base model card: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- SDXL inpainting checkpoint: https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1
- Fibo 2.0 dev model card: https://huggingface.co/briaai/Fibo-2.0-dev
- Ideogram 4 fp8 model card: https://huggingface.co/ideogram-ai/ideogram-4-fp8
- diffusers DreamBooth examples: https://github.com/huggingface/diffusers/tree/main/examples/dreambooth
- diffusers Qwen-Image 2.1 pipeline docs: https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage21
- Black Forest Labs release notes: https://docs.bfl.ml/release-notes
- Hugging Face model API (downloads, likes, licenses, gating), queried 2026-10-21
