# Qwen-Image-2.1 per-class LoRAs for Figure 3

Plan written 2026-10-28; the run log below records every step actually executed. This replaces
Steps 2a, 2b, 3 (Qwen rows) and 4 of `FIGURE3_REGEN_PLAN.md`; the data split, classifier and
leakage decisions there still apply.

**Ground rules.** If training, captioning, generation or a classifier run fails, stop and report
the error with its log. No workarounds that change the experiment (other model, other data,
skipped classes) and no numbers that were not produced by a run. **Paid API budget for this
agent session: $50 total, hard stop. Ask before spending more.** Every paid call is recorded in
the spend ledger at the bottom. The default path spends nothing; only Stage C
captioning (~$5-10) uses the API.

## Hardware and environment

3x NVIDIA RTX PRO 6000 Blackwell (96 GB each), torch 2.8.0+cu128, branch `modernize`, set up with
`bootstrap_pod.sh` plus `pip install fastai`. One job per GPU, pinned with `CUDA_VISIBLE_DEVICES`.
With 96 GB the 32 GB workarounds (8-bit text encoder, CPU offload, one edit per call) are
switched off unless a run shows they are needed.

Data: training pool `fitz_metadata/fitzpatrick17k_10label_clean_training.csv` (nine labels,
3,339 images), held-out test `fitzpatrick17k_10label_clean_held_out_set.csv` (360), images at
`/root/data/fitzpatrick17k/finalfitz17k/{md5hash}.jpg` (copy of `/workspace/data/...`).

## Training captions and the 10 images per real: staged

Two separate choices: how the LoRA is captioned in training, and where the variety among the 10
synthetic images per real comes from. Start with the simplest version and add complexity only if
the pilot shows it is needed (decided 2026-10-28).

**Default (Stage A): static captions, seed-only variety.**
- Text-to-image captions: the backend's class caption, e.g. `An image of psoriasis, a skin disease`.
- Edit captions: the API pilot's `EDIT_FRAME` with no variation sentence ("A clinical photograph of
  the same skin disease, {condition}, keeping ... the same skin tone as the reference, on a
  different patient. No text, watermarks or annotations.").
- Generation: one prompt, slot k uses seed `base + k`.

**Edit training pairs.** There are no real before/after pairs. Each pair is two different real
images of the same label (reference = patient A, target = patient B), so the LoRA learns "same
disease, different patient", which is what generation asks for. Because the edit prompt asks to
keep the reference's skin tone, pairs are drawn within the same Fitzpatrick type
(`fitzpatrick_scale`) where possible, else an adjacent type; counts recorded. This needs a pairing
function passed to the export in place of the current random same-label pairing.

**Stage B (only if Stage A lacks variety): add the 10 prompts at generation time.** Same LoRAs, no
retraining, no API cost. Slot k uses `EDIT_VARIATIONS[k]` (from `pilot_generate.py`) and seed
`base + k`. Text-to-image gets the same variation sentences. Used if it adds variety without
losing the disease look.

**Stage C (only if Stage A lacks variety AND the LoRAs ignore the Stage B prompts): descriptive
captions from a VLM, then retrain.** Captions describe what is in each training image, in the
vocabulary of the 10 prompts, so the LoRA learns to respond to them.
- Skin type from the CSV (`fitzpatrick_scale`; -1 = unknown, left out).
- Body site, extent, framing, lighting, apparent age, background from `gemini-3.8-flash`, JSON
  schema with fixed choices (body site: forearm, upper arm, hand/fingers, upper back, lower back,
  chest/abdomen, face, scalp/forehead, neck/ear, thigh, lower leg/shin, foot/ankle, genital/groin,
  mouth/lips, other, unclear; extent: single lesion, several lesions, widespread; framing:
  close-up, mid-range, distant; lighting: exam-room, flash, daylight, outdoor, unclear; apparent
  age: child, young adult, older adult, unclear; background: none, exam table/sheet, clothing,
  other). The VLM is not asked to diagnose; the disease name always comes from the CSV.
- Templates: text-to-image `A clinical photograph of {condition}, a skin disease. {extent} on the
  {site}, {framing}, {lighting}. Fitzpatrick skin type {n}.`; edit uses `EDIT_FRAME` with the
  variation sentence built from the **target's** descriptors.
- Check: spot-check 50 captions by eye; if body site is wrong in more than about 1 in 5, stop.
- Cost: $0.75/M input, $3.75/M output tokens (list, through 2026-12-31); ~$5-10 for the 3,339
  training-pool images. 20-image test first; the script tracks spend from `usage_metadata` and
  stops at a $15 cap. Held-out images are never sent.

**Comparability note.** If Qwen ends at Stage A while the API pilot used 10 prompts, Qwen-vs-API
differences mix the model with the source of variety. Recorded with the results.

**Measures (all stages).** Within-reference diversity (mean pairwise DINOv2 distance among the 10
outputs), label consistency (a real-only classifier's top-1 agreement with the label),
skin-tone preservation for edits (mean ITA difference to the reference), nearest-real DINOv2
distance against the class's training images (memorization), and the image grids.
"Lacks variety" is judged against the GPT Image varied-prompt pilot outputs for the same 16
psoriasis reals: the threshold is set from the grids and the DINOv2 numbers and written down here
before the decision.

## Steps

**Step 1. Single-class pilot, psoriasis (~half a day, GPU only, $0).**
- 1A: Stage A LoRAs for edit and text-to-image at 500, 1000 and 2000 steps (rank 16, lr 1e-4,
  512 px; 6 runs, 3 at a time). For each, generate 10 seed-only images for the 16 psoriasis reals
  of the API pilot, shown next to the reals, untuned Qwen (edit and text-to-image), and the 2024 SD
  generations. Edit LoRAs must beat untuned Qwen edit; if none does, the edit arm uses untuned
  Qwen and this is recorded.
- 1B: if variety is lacking, the Stage B prompts on the same LoRAs.
- 1C: if still lacking and the prompts are ignored, Stage C (captioning, ~$5-10) and retrain.
- Decision recorded here: steps, rank, lr, stage.

**Step 2. Confirm on two more classes (basal cell carcinoma, lupus erythematosus).** Same
settings, same measures. Adjust once if needed and record why.

**Step 3. Train all 9 LoRAs per method** (9 edit, 9 text-to-image), three at a time. Outputs
`/workspace/synderm/qwen_lora/{method}/{label}/`.

**Step 4. Generate.** 10 images for every training-pool image, both methods (~33k each), 512 px,
in the paper's layout `/workspace/synderm/qwen_lora/generations/{method}/{00..09}/{md5hash}.png`
with a JSON sidecar (prompt, seed, LoRA). Edit first, then text-to-image. Time measured in
Step 1 and recorded before starting.

**Step 5. Figure 3 sweep.** The paper's fastai protocol (EfficientNetV2-M, 224 px, early stopping,
held-out 360 test). n in {1, 8, 16, 32, 64, 128, 228} x {real only, + Qwen edit, + Qwen text-to-image}
x {no transforms, aug_transforms} x 5 new seeds = 210 runs, across 3 GPUs. Results to one CSV;
analysis notebook makes the plot and a table, both recorded here.

## Run log

**2026-10-28, session 1 (new pod, 3x RTX PRO 6000). Nothing trained, nothing generated, $0 spent.**

- Environment: `bootstrap_pod.sh` on branch `modernize` (exit 0, 8 CPU tests pass), `pip install fastai scikit-image`.
  torch 2.8.0+cu128 runs on all three GPUs.
- Scripts written, compiled, not yet run to completion: `qwen_lora_train.py`, `qwen_lora_generate.py`,
  `qwen_lora_eval.py`. Psoriasis has 584 training-pool images; the training-pool CSV has 132 images with
  unknown Fitzpatrick type (-1).
- Smoke tests (20-step t2i and edit training on GPUs 0 and 1, untuned edit generation on GPU 2) never
  reached a training step. All three spent 15-20 min loading the 32 GB Qwen-Image-2.1 weights from
  `/workspace`; reads ran at about 8 MB/s. Stopped by hand.
- Tried copying the model into RAM (`/dev/shm`, 263 GB) to load from there: 0.5 GB of 32 GB in 53 min,
  then no progress. The image copy to `/root/data` stalled at the same time (9,186 of 16,536 files;
  2,007 of 3,530 training-pool images).
- Diagnosis: the `/workspace` geesefs mount lists directories but no longer returns file contents; a read
  of one 100 KB image did not return in 5 min and the reading processes sit in uninterruptible I/O (state D).
  Not fixable inside the pod. Stopped here for a pod restart / volume remount.
- After restart: download Qwen-Image-2.1 at revision `d26bb612` from Hugging Face into `/dev/shm` (same
  weights, no dependence on volume read speed), finish the image copy to local disk before any GPU job,
  and write only outputs (weights, logs, images) to `/workspace`. Training data already goes to
  `/dev/shm/qwen_lora_data` (`--data-root`).

**2026-10-28, session 2 (pod rebuilt; volume remounted). Setup.**

- Pod: the container exposes **2** of the advertised 3 RTX PRO 6000 GPUs (`nvidia-smi -L` lists GPU 0 and 1;
  `/proc/driver/nvidia/gpus` shows four PCI devices on the host, `NVIDIA_VISIBLE_DEVICES=void`). Not fixable from
  inside the container. Plan runs two jobs at a time instead of three. 64 CPUs, 1.1 TB RAM, 263 GB `/dev/shm`.
- Volume check: `timeout 30 head -c 100 /workspace/backups/README.txt` returned at once (reads work again).
- Repo: `/root/synthetic-derm` on `modernize` at b48f9d2. Claude memory restored from `/workspace/backups/claude-memory/`.
- `.env` recreated with an empty `GEMINI_API_KEY=` (key to be supplied by the user; the default path needs none).
- `HF_HOME=/workspace/.cache/huggingface bash bootstrap_pod.sh`: exit 0, 8 CPU tests pass. `pip install fastai scikit-image`.
  Versions: torch 2.8.0+cu128, diffusers 0.42.0.dev0 (git 0c7d56d6), peft 0.21.2, accelerate 1.15.0, datasets 5.1.0,
  timm 1.0.30, fastai 2.8.12, scikit-image 0.26.0, huggingface_hub 2.2.0. Log: `/workspace/logs/pod-2026-10-28b/bootstrap.log`.
- Models in RAM: `HF_HOME=/dev/shm/hf hf download Qwen/Qwen-Image-2.1 --revision d26bb612` (32 GB, under 1 min from the
  Hub), plus timm `vit_small_patch14_dinov2.lvd142m` and `tf_efficientnetv2_m.in1k`. All GPU jobs run with
  `HF_HOME=/dev/shm/hf`; nothing is read from `/workspace` at run time except the pilot reals CSV.
- Images: `cp -r` from `/workspace` ran at ~7 files/s, so switched to a 24-way parallel copy (`xargs -P 24 cp`,
  script in the session scratchpad): all 16,536 files in 68 s. Verified: every file's md5 equals its name, no empty
  files, 1.46 GB; all 3,530 training-pool and 400 held-out md5hashes present in `/root/data/fitzpatrick17k/finalfitz17k`.
- Vendored trainers: confirmed against GitHub that `train_dreambooth_lora_qwenimage21{,_img2img}.py` equal
  diffusers `examples/dreambooth` at commit 0c7d56d6 apart from the documented 11-line `--offload` patch
  (`synderm/fine_tune/scripts/README.md`).
- Smoke tests started: 20-step psoriasis t2i (GPU 0) and edit (GPU 1) trainings, `--checkpoint-every 10`,
  run name `smoke-20steps-b`. Edit pairs for psoriasis: 1,108 same Fitzpatrick type, 0 adjacent, 0 any,
  60 with unknown target type (584 images x 2 pairs).
- Smoke tests, all passed (psoriasis, `HF_HOME=/dev/shm/hf`):
  - `qwen_lora_train.py --method t2i --steps 20 --checkpoint-every 10`: 1.0 min end to end; latent caching of 584
    images in 15 s; training 1.1 steps/s (batch 1 x grad-accum 4, 512 px, gradient checkpointing); 34 GB GPU memory;
    checkpoint-10, checkpoint-20 and final weights written to `/workspace`.
  - `... --method edit --steps 20 --checkpoint-every 10`: 2.5 min; caching 1,168 pairs in 74 s; 1.8 s/step.
  - `qwen_lora_generate.py` with each smoke LoRA, 2 reals x 10 slots, 40 sampling steps, 512 px: 2.7 s/image (t2i)
    and 2.8 s/image (edit), 51 GB GPU memory, loads from RAM in under a minute. Images and JSON sidecars look right
    (t2i: hand lesions from the prompt; edit: the reference's composition with altered lesions).
  - `qwen_lora_eval.py` on the two smoke folders (GPU shared with a training job): ran end to end, wrote
    `metrics.csv` and grids. The DINOv2 ViT-S/14 linear probe on the 3,339-image pool has 5-fold CV accuracy 0.51
    over nine labels (chance 0.11), so `label_match` is a relative measure between folders, not an absolute one.
    Smoke numbers (20-step LoRAs, 2 reals, not a result): t2i label_match 0.45, within_dist 0.53 (real sets of 10:
    0.59); edit label_match 1.0 but within_dist 0.10 and ref_dist 0.10, i.e. near copies of the reference, as the
    side-by-side showed.
- Projected timings: 2000-step LoRA ~32 min (t2i) / ~62 min (edit) per class; full generation 33,390 images per
  method at 2.7 s = ~25 GPU-h per method, ~13 h wall-clock per method on two GPUs.
- Pilot trainings launched 21:59 UTC: `qwen_lora_train.py --label psoriasis --method {t2i,edit} --steps 2000
  --checkpoint-every 500` (rank 16, lr 1e-4, 512 px, seed 0), t2i on GPU 0, edit on GPU 1; run name
  `r16-lr0.0001-512px-2000steps`. Generation and scoring for Step 1A are scripted in `qwen_lora_pilot.sh` (`generate <method> <gpu> <label> <run>`
  per method as soon as its training finishes, then `eval <gpu> <label>`). A 30-minute check-in loop was set up at
  the user's request (22:01 UTC).
- 22:34 UTC: t2i psoriasis LoRA finished, 2000 steps in 32.9 min (1.03 steps/s), final loss noisy around 0.3-1.0
  as expected for flow matching; checkpoints 500/1000/1500/2000 and final weights in
  `/workspace/synderm/qwen_lora/t2i/psoriasis/r16-lr0.0001-512px-2000steps/`. Edit LoRA at step 1103/2000 (1.75 s/step).
- 22:37 UTC: t2i pilot generation started on GPU 0 (`qwen_lora_pilot.sh generate t2i 0 psoriasis r16-lr0.0001-512px-2000steps`:
  untuned, ckpt500, ckpt1000, ckpt1500, ckpt2000; 16 reals x 10). Started by hand 3 min late: the automatic chain had
  used a `pgrep -f` pattern that matched its own shell and so never fired (fixed with a `[q]wen...` pattern for the
  edit chain).
- 23:01 UTC: edit psoriasis LoRA finished, 2000 steps in 60.6 min (1.76 s/step, 1,168 pairs cached in 74 s);
  checkpoints 500/1000/1500/2000 and final weights in `/workspace/synderm/qwen_lora/edit/psoriasis/r16-lr0.0001-512px-2000steps/`.
  Edit pilot generation started on GPU 1 (same five folders as t2i). t2i generation at folder 4 of 5, 2.95 s/image.
- Protocol for the cut-off: the GPT Image varied-prompt pilot folder is scored on its own first (eval on GPU 0, shared
  with the t2i generation), and the "lacks variety" threshold is written below from its numbers and grid before any
  Qwen LoRA folder is scored.
- **"Lacks variety" cut-off (written 23:03 UTC, before any Qwen LoRA folder was scored).** Reference numbers from
  `eval-psoriasis/metrics.csv`: GPT Image varied-prompt pilot (146 images, 15 reals, top-ups dropped): label_match 0.89,
  within_dist 0.43, ref_dist 0.52, nn_real_dist 0.17, near_copies 0, ita_shift 15.7 deg; random sets of 10 real
  psoriasis images: within_dist 0.59; nearest-neighbour distance among reals: 0.17. In the GPT grid the 10 slots of a
  real differ in body site, framing and skin region, i.e. the variety is prompt-driven. A Qwen Stage A folder
  (seed-only variety) is judged to **lack variety** if either (a) `within_dist` < 0.21, i.e. under half the GPT value,
  or (b) its grid shows the 10 slots of a real as the same composition with only texture changes (judged by eye on
  the 6-real grid). For edit folders a third condition applies: (c) `ref_dist` < 0.20 means the outputs are near
  copies of the reference and add nothing for a classifier. Separately, a folder keeps the **disease look** if
  `label_match` >= 0.74 (within 0.15 of GPT) and `near_copies` <= 0.05; the LoRA checkpoint is chosen among those
  that keep the disease look, preferring the highest `within_dist`, ties to the earlier checkpoint. Edit LoRAs must
  beat untuned Qwen edit on label_match or within_dist without failing (c), else the edit arm uses untuned Qwen.
- 23:17 UTC: t2i pilot generation finished: 5 folders x 160 images (untuned, ckpt500, ckpt1000, ckpt1500, ckpt2000),
  2.8-2.95 s/image at 512 px, 40 sampling steps, 10 prompts batched per call; ~8.5 min per folder including the
  one-minute pipeline load. Scoring the five t2i folders on GPU 0 while the edit generation continues on GPU 1.

## Spend ledger

| When | What | Model | Images | Cost (from usage) | Session total |
|---|---|---|---|---|---|
