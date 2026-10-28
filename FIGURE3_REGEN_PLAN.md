# Plan: regenerate Figure 3 with Qwen-Image-2.1 and Nano Banana 2.1

Written 2026-10-28. Goal: add modern-generator arms to the paper's Figure 3 (nine-way skin
classification accuracy vs number of real training images, with and without 10 synthetic per
real, with and without image transforms) without rerunning the original arms.

## What Figure 3 is, from the code

Source: `paper_experiments/Experiments/n_per_label.py`, `paper_experiments/train_dreambooth.py`,
`paper_experiments/generate.py`, `paper_experiments/dataset.py`.

**Data split (all from `fitz_metadata/`).**

| Set | Definition | Size |
|---|---|---|
| Training pool | `fitzpatrick17k_10label_clean_training.csv`, nine most common labels (folliculitis dropped) | 3,339 |
| Validation | 40 per label sampled from the pool with `random_state` (111108 in the script) | 360 |
| Classifier train | n per label sampled from the pool minus validation, same `random_state`; n in {1, 8, 16, 32, 64, 128, 228} | 9n |
| Held-out test | `fitzpatrick17k_10label_clean_held_out_set.csv`, nine labels; zero overlap with the pool | 360 |

228 is the cap because the smallest class (photodermatoses, 272) has 232 images after
validation is removed. Each of the five runs used a different `random_state`, so each run
drew a different n-per-label subset and validation set. Only 111108 is recorded in the repo;
the other four seeds live in the wandb project `n_real_per_disease_x`.

**Which images trained the generators.** `train_dreambooth.py` builds `FitzpatrickDataset`
on the full training CSV filtered to one class, so each per-class DreamBooth model saw every
training-pool image of its class (272 to 584), including the images later used as validation.
The held-out test images were never seen. `generate.py` then produced 10 generations for every
training-pool image of the class, stored as `{method}/{00..09}/{md5hash}.png`.

**How synthetic images attach to the sweep.** For a run with n real per label, the classifier
gets those 9n reals plus the 10 synthetic images filed under each selected real's md5hash
(`train_data[train_data['md5hash'].isin(md5hashes)]`). For inpaint and in-then-outpaint the
synthetic image was conditioned on that real. For text-to-image the md5hash is only a file
name; the image was generated from the class prompt alone.

**Mix.** None. Each panel is one method at 10 synthetic per real. Fine-tuned models were used
(folder `all_generations/finetune-*`).

**Classifier.** fastai `vision_learner` with `efficientnet_v2_m`, 224 px, up to 50 epochs,
early stopping on validation loss with patience 3, batch size 8/32/64 by training-set size,
`aug_transforms()` or none. Accuracy on the 360 held-out images, Wald 95% CI, five runs.

## Decisions

- Keep the original classifier code (`n_per_label.py`, fastai, EfficientNetV2-M) so new arms
  drop into the existing figure and the original arms are not rerun.
- One LoRA per class (9 text-to-image, 9 edit), matching the paper's per-class DreamBooth.
  The multi-class LoRA from Oct 21 produced near-identical images across classes.
- Qwen-Image-2.1 has no hard-mask inpainting, so inpaint and in-then-outpaint collapse into
  one Qwen "edit" panel. Qwen arms: text-to-image, edit.
- Proprietary arm: chosen by the Step 0.5 screen. Default candidate is Nano Banana 2.1
  (`gemini-nano-banana-2.1`), zero-shot text-to-image and zero-shot edit, batch API ($0.017
  per image). No fine-tuning is possible with any proprietary model.
- Leakage: the paper's generators were trained on the whole training pool, so low-n arms
  carry information from hundreds of reals. The main new arms keep this design for
  comparability. A clean control (generators trained only on the n sampled reals) was
  considered and deferred.
- Generators train on the same pool the paper used: all training-pool images of the class.
- Synthetic images are filed under the same `{method}/{00..09}/{md5hash}.png` layout so the
  unmodified sweep script reads them with a different `generation_folder`.

## Inputs needed

1. Fitzpatrick 17k images: done 2026-10-28, 16,536 files at
   `/root/data/fitzpatrick17k/finalfitz17k/{md5hash}.jpg` (all 3,530 training and 400 held-out
   images present). The root disk is ephemeral, so Step 0 also copies the folder to
   `/workspace/data/fitzpatrick17k/` with plain `cp` (rsync `-a` fails there: the FUSE mount
   rejects chmod).
2. OpenAI and Google AI Studio API keys in the shell.
3. Seeds: the original five `random_state` values are not recoverable (decided 2026-10-28), so
   Step 3 uses five new seeds and reruns the real-only arms with them as well, so every curve
   on the new panels shares the same real subsets.

## Steps

**Step 0. Environment (~20 min).** Check out `modernize`, run `bootstrap_pod.sh` (restores
diffusers, pandas, huggingface_hub on the reset root disk; weights are cached on `/workspace`).
Install `openai` and `google-genai`. Separately build the paper's fastai environment from
`paper_experiments/environment.yml` for the classifier step.

**Step 0.5. Zero-shot screen, notebook (half a day, ~$20 plus ~1 GPU-hour).** Before any
fine-tuning. For each of the nine classes, generate 4 text-to-image samples and 2
reference-image edits from each OpenAI and Google image model and from Qwen-Image-2.1 with no
LoRA, using the paper's prompt "An image of {class}, a skin disease" plus one richer clinical
prompt. Show a grid per class: real examples on the left, one row per model. Record refusals
(`moderation_blocked` on OpenAI, blocked parts on Google), latency and cost per image. The
Qwen rows establish the untuned baseline the LoRA must beat in Step 2a; FLUX 3 Klein and SDXL
are already cached and can be added as rows at the same cost if wanted.

Per-image list prices, 1:1 aspect, checked 2026-10-28 on the OpenAI and Google pricing pages.
OpenAI bills per output token, so the 2.5 models (same token rate as GPT Image 2) are shown at
GPT Image 2's figures; the notebook reports the real cost from each response's token usage.
Google batch mode halves the Google prices; OpenAI batch halves the token rate.

| Model | Low / 0.5K | Medium / 1K | High / 2K | 4K |
|---|---|---|---|---|
| `gpt-image-2.5-sunburst` | ~$0.006 | ~$0.053 | ~$0.211 | n/a |
| `gpt-image-2.5-flare` | ~$0.006 | ~$0.053 | ~$0.211 | n/a |
| `gpt-image-2` | $0.006 | $0.053 | $0.211 | n/a |
| `gpt-image-1.5` | $0.009 | $0.034 | $0.133 | n/a |
| `gemini-nano-banana-2.1` | n/a | $0.034 | $0.050 | $0.113 |
| `gemini-3.1-flash-image` (Nano Banana 2) | $0.045 | $0.067 | $0.101 | $0.151 |
| `gemini-3-pro-image` (Nano Banana Pro) | n/a | $0.134 | $0.134 | $0.240 |
| `Qwen-Image-2.1` zero-shot, local | GPU only, ~8.5 s per 768 px image | | | |

For OpenAI the columns are quality levels at 1024x1024 (low, medium, high); for Google they are
output resolutions. The screen uses medium quality and 1K.

Roughly 8 models x 9 classes x 6 images = 432 images. Outputs: which models are usable for
dermatology at all, which refuse which classes, which proprietary model goes forward, and how
far untuned Qwen is from the real images. Real images for the grid come from
`/root/data/fitzpatrick17k/finalfitz17k/`.

**Step 1. Proof of concept: GPT Image 2.5 Sunburst edits, one seed (about a day, ~$35).**
The Figure 3 pipeline end to end at the smallest useful scale. Pass this before spending GPU
time on Qwen LoRAs. Chosen over Nano Banana 2.1 after the Step 0.5 screen: better label
fidelity (pearly, ulcerated basal cell carcinomas rather than generic pink bumps) and, at
medium quality, about half the price (~$0.019 per edit from returned token counts: ~440
output image tokens at $30/M plus ~650 input image tokens at $8/M; 21 s latency). The
endpoint reports tokens, not dollars, and no reasoning-token line; confirm against the OpenAI
usage dashboard after the first few hundred images.

- One generator, one method (reference-image edit), one seed, medium quality (1024 px).
- Edit prompts: the screen's repeat test (section 6) showed that 10 edits of one reference
  with one prompt look nearly alike, because the APIs expose no seed. Ten edits with ten
  varied prompts (section 6b: body site, framing, lighting; same list for every class) gave
  clearly more diverse images, so the pilot uses the varied prompts (`--prompts varied`,
  slot k gets variation k). The single-prompt mode stays in the script for a later comparison.
  The single-prompt runs started earlier (GPT $10.78, Nano Banana $19.92, both partial) are
  kept on disk but not used.
- Draw 16 real images per class with the paper's sampling (the 1 and 8 subsets nest inside the
  16), 144 reals in all, and generate 10 edits of each: 1,440 images, roughly 1 hour at 8
  parallel calls.
- Refusals: the screen saw 4 of 36 edit references rejected (11%), clustered on sensitive
  body sites. Handle in two stages. (1) Retry each refusal once with a new request; this
  catches rejections of the output rather than the input. (2) If a reference is refused
  twice, mark it unusable and top up its 10 slots with extra edits of the other accepted
  references in the same class, so every class still reaches 16 x 10 = 160 synthetic images.
  The real subset stays as drawn, so there is no selection toward "safe" images; only the
  real-to-synthetic mapping changes, which is fine for the edit method. Expected extra calls
  about 300, total about 1,750 images, ~$35.
- Classifier: n in {1, 8, 16} real per class, with and without the 10 synthetic per real, with
  and without image transforms, the original fastai code and the 360-image held-out test. 12
  short runs, under an hour on one GPU.
- Output: three points of Figure 3 for one proprietary model, no error bars, plus the refusal
  rate per class and which references were rejected.

Text-to-image, more seeds and the second generator are added later only if this shows an
effect worth chasing. Text-to-image needs prompt-level diversity first (body site, skin type,
framing), since the screen showed near-identical samples from a fixed prompt.

**Step 2a. Single-class pilot notebook (half a day).** Superseded 2026-10-28 by
`paper_experiments/2026/QWEN_LORA.md` (3 GPUs, descriptive captions, steps 2a to 4 for Qwen). One class (psoriasis). Train a Qwen
text-to-image LoRA at the Oct 21 settings and at 500, 1000 and 2000 steps; generate a grid
with fixed seeds; compare against real psoriasis images and the 2024 SD generations. Same for
one edit-mode LoRA on real references. Confirm images are class-specific and not memorized
copies. This step fixes steps, rank and learning rate.

**Step 2b. Confirm across classes (half a day).** Apply the chosen settings to two more
classes with visually distinct presentations (basal cell carcinoma, lupus erythematosus).
Check a label-consistency score from a classifier trained on real images only.

**Step 3. Full generation.**

| Arm | Adapters | Images | Cost |
|---|---|---|---|
| Qwen text-to-image | 9 LoRAs, ~2 GPU-h | 33,390 (10 per pool image) | ~85 GPU-h at 768 px |
| Qwen edit | 9 LoRAs, ~2 GPU-h | 33,390 | ~85 GPU-h |
| Nano Banana 2.1 text-to-image + edit | none | 66,780 | ~$1,150 batch |

At 512 px Qwen time roughly halves. Only the images attached to sampled reals are used, but
generating for the whole pool keeps the five-seed sampling free.

**Step 4. Classifier sweep (2 to 3 days, one GPU).** Run `n_per_label.py` unchanged except
`image_dir`, `generation_folder`, `generation_type_list` and `random_state`, for the four new
generator-method arms plus the real-only baselines, five new seeds. That is
7 n-values x 2 transform settings x (4 arms + 1 real-only) x 5 seeds = 350 runs. Add to the
existing figure as new panels.

Total: about one week of wall-clock after the images land, with one GPU.
