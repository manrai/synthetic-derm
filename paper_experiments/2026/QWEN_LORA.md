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
- **t2i pilot scores (psoriasis, 16 reals x 10, `eval-psoriasis/metrics.csv`), 23:19 UTC:**

  | folder | label_match | top wrong label | within_dist | nn_real_dist | near_copies |
  |---|---|---|---|---|---|
  | GPT Image varied (reference) | 0.89 | sarcoidosis 6 | 0.43 | 0.17 | 0 |
  | t2i untuned | 0.49 | neutrophilic dermatoses 24 | 0.37 | 0.20 | 0 |
  | t2i ckpt500 | 0.68 | squamous cell carcinoma 21 | 0.55 | 0.29 | 0 |
  | t2i ckpt1000 | 0.63 | allergic contact dermatitis 25 | 0.51 | 0.26 | 0 |
  | t2i ckpt1500 | 0.78 | photodermatoses 7 | 0.53 | 0.26 | 0 |
  | t2i ckpt2000 | 0.66 | allergic contact dermatitis 16 | 0.54 | 0.25 | 0 |

  Variety: every LoRA folder has within_dist 0.51-0.55, above the GPT reference (0.43) and close to random sets of
  10 reals (0.59), so by measure (a) seed-only variety is **not** lacking for t2i; untuned Qwen is lower (0.37) and
  its grid is mostly hands on white backgrounds. By eye (grid 6 reals x 10), the LoRA folders vary body site,
  framing, background and skin tone across the 10 seeds, condition (b) also passes. No near copies (nn_real_dist
  0.25-0.29 vs 0.17 among reals).
  Disease look: only ckpt1500 reaches the label_match threshold (0.78 >= 0.74); ckpt500/1000/2000 sit at 0.63-0.68
  (standard error about 0.04 at n=160, so 1500 vs the rest is about 2-3 SE, not a large margin). Grids: the LoRA
  has learned the Fitzpatrick 17k photographic style (varied clinics, dark backgrounds, occasional text/watermark
  artifacts, some nude anatomy, as in the training pool); ckpt500 has many blurred or blotchy textures, ckpt1500
  is the cleanest, but in all LoRA folders fewer tiles show the classic scaly plaque than in the GPT grid, where
  nearly every tile does. Untuned Qwen draws clean, generic pink papules/plaques (probe: neutrophilic dermatoses).
  Provisional t2i choice: ckpt1500 (Stage A); final decision after the edit folders are scored.
- **Edit pilot, early scores (untuned and ckpt500 folders, scored 23:19 UTC while ckpt1000-2000 were still generating):**

  | folder | label_match | within_dist | ref_dist | nn_real_dist | near_copies | ita_shift |
  |---|---|---|---|---|---|---|
  | GPT Image varied (reference) | 0.89 | 0.43 | 0.52 | 0.17 | 0 | 15.7 |
  | edit untuned | 0.74 | 0.06 | 0.07 | 0.05 | 0.56 | 5.3 |
  | edit ckpt500 | 0.61 | 0.55 | 0.55 | 0.29 | 0 | 46.8 |

  Untuned Qwen edit with one prompt and 10 seeds returns near copies of the reference (grid: the same photograph
  ten times with small lesion-texture changes): fails cut-off (a) within_dist 0.06 < 0.21 and (c) ref_dist 0.07 <
  0.20, and 56% of its images are within 0.05 of a training-pool image (the reference itself). The 500-step edit
  LoRA has the opposite failure: its outputs are as far from the reference as from unrelated images (ref_dist 0.55
  = nn distance of unrelated reals), skin tone is not kept (ITA shift 47 deg), and the grid shows arbitrary
  Fitzpatrick-style photographs with no relation to the reference. Training on pairs of *different* patients
  taught the LoRA to ignore the condition image, which is the risk the plan noted. Label match 0.61 is below the
  0.74 threshold.
- Consequence, per the plan's escalation rule: Stage A lacks variety for the edit arm (untuned) and the LoRA does
  not preserve the reference, so **Stage B (the 10 varied prompts at generation time, no retraining, $0) is run
  for edit**. 23:20 UTC: `qwen_lora_generate.py --method edit --prompts varied` with untuned Qwen on GPU 0, output
  `pilot/edit-psoriasis-untuned-varied`. The remaining edit LoRA checkpoints (1000/1500/2000) are scored when their
  Stage A folders finish; if one keeps the reference (ref_dist between 0.20 and ~0.45, ITA shift under ~20 deg)
  it also gets a Stage B run.
- **Stage B result for edit (scored 23:30 UTC):**

  | folder | label_match | within_dist | ref_dist | nn_real_dist | near_copies | ita_shift |
  |---|---|---|---|---|---|---|
  | edit untuned, varied prompts (Stage B) | 0.72 | 0.35 | 0.34 | 0.17 | 0.08 | 15.7 |
  | edit ckpt1000, single prompt (Stage A) | 0.60 | 0.57 | 0.60 | 0.32 | 0 | 44.8 |

  Untuned Qwen edit with the 10 varied prompts passes (a) within_dist 0.35 >= 0.21 and (c) ref_dist 0.34 >= 0.20,
  keeps skin tone (ITA shift 15.7 deg, same as GPT), and its label_match 0.72 is within one standard error of the
  0.74 line (GPT 0.89). The grid shows what the numbers average over: for most references about half the slots
  follow the prompt (new body site, same lesion morphology and skin tone, e.g. plaques moved to the back, chest,
  foot or knuckles) and the other half stay close to the reference; for two of the six references nearly all ten
  slots remain near copies (near_copies 0.08 overall). The ckpt1000 edit LoRA behaves like ckpt500: it ignores
  the reference (ref_dist 0.60, ITA shift 45 deg) and has the lowest label_match (0.60).
- 23:31 UTC: Step 2 started for the t2i arm on idle GPU 0 before the edit decision, because the t2i outcome is fixed
  by the cut-offs above (Stage A, ckpt1500): `qwen_lora_train.py --label "basal cell carcinoma" --method t2i
  --steps 2000 --checkpoint-every 500`, same rank/lr/resolution as psoriasis.
- 23:38 UTC: edit ckpt1500 (Stage A) scored: label_match 0.61, within_dist 0.55, ref_dist 0.56, ITA shift 41 deg,
  near_copies 0: same reference-ignoring behaviour as ckpt500 and ckpt1000. ckpt2000 folder still generating.
- 23:42 UTC: edit ckpt2000 (Stage A) scored: label_match 0.56, within_dist 0.57, ref_dist 0.55, ITA shift 43 deg.
  All ten Qwen psoriasis folders are now in `pilot/eval-psoriasis/metrics.csv`.

**Step 1 decision (psoriasis), 23:43 UTC.**

| arm | choice | stage | evidence |
|---|---|---|---|
| text-to-image | psoriasis LoRA, rank 16, lr 1e-4, 512 px, **checkpoint 1500** of 2000, single prompt, seed-only variety | **A** | within_dist 0.53 (cut-off 0.21, GPT 0.43), label_match 0.78 (threshold 0.74), no near copies |
| edit | **untuned Qwen-Image-2.1 edit with the 10 varied prompts**; no edit LoRA | **B** | all four edit LoRA checkpoints ignore the reference (ref_dist 0.55-0.60, ITA shift 41-47 deg, label_match 0.56-0.61), so none beats untuned edit under the plan's rule; untuned single-prompt edit fails (a) and (c) (near copies); untuned varied passes (a) 0.35 and (c) 0.34, keeps skin tone (15.7 deg) |

Caveats recorded with the decision:
- The Stage B edit folder narrowly misses two written lines: label_match 0.72 vs 0.74 (about one standard error) and
  near_copies 0.08 vs 0.05 (for two of six references in the grid most slots stay close to the reference). It is
  used because the plan's fallback for "no LoRA beats untuned" is untuned Qwen, and the varied prompts are the only
  tested way to give untuned edit variety. The t2i LoRA checkpoints other than 1500 also miss the label line
  (0.63-0.68); 1500 vs the rest is 2-3 standard errors.
- Stage C (VLM captions, paid) is not triggered: for t2i Stage A has enough variety; for edit the Stage B prompts
  are followed by untuned Qwen. The edit LoRAs' failure is the pairing design (reference = a different patient), which
  captions would not fix; a within-patient or self-reference (augmented copy) pairing would be a different experiment
  and is not tried here.
- Comparability: the Qwen t2i arm has seed-only variety while the Qwen edit arm and the GPT Image arm have
  prompt-driven variety. Recorded for the results write-up.
- Stage B on a t2i LoRA was not run (not required by the plan: Stage A did not lack variety).

**Step 2 (confirm on basal cell carcinoma and lupus erythematosus), started 23:31/23:43 UTC.**
- t2i LoRAs at the chosen settings: BCC on GPU 0 (started 23:31), lupus on GPU 1 (23:43); 2000 steps, checkpoints
  every 500, the 1500 checkpoint is the candidate.
- Generation for the 16 reals of each class in the API pilot `reals.csv`: t2i untuned and ckpt1500 (single prompt),
  edit untuned varied; chained automatically after each training. Then `qwen_lora_eval.py --label <class>` on
  those folders plus the GPT Image pilot folder for the same class, with the same cut-offs as psoriasis.
- 2026-10-29 00:04 UTC: BCC t2i LoRA trained (406 training-pool images, 2000 steps, 32.7 min). GPU 0 chain
  started: BCC t2i untuned, BCC t2i ckpt1500, edit-varied BCC, edit-varied lupus. Lupus t2i LoRA training on GPU 1.
- Push to GitHub: the first push of the session (80e9c88) succeeded; later pushes fail with "Authentication failed"
  (no credential helper configured on this pod). Commits stay local until credentials are supplied.
- 00:15 UTC: lupus t2i LoRA trained (297 training-pool images, 2000 steps, 31.3 min). 00:30: lupus t2i untuned and
  ckpt1500 folders done on GPU 1; BCC t2i untuned, ckpt1500 and edit-varied done on GPU 0; lupus edit-varied generating.
  Scoring BCC (all three folders) and lupus t2i on GPU 1, each against the GPT Image pilot folder of the same class.
- **Step 2 scores, 00:40 UTC** (16 reals x 10 per folder; label_match from the same 9-label DINOv2 probe; the
  "within 0.15 of GPT" label line and the 0.21 variety line are applied per class):

  | class | folder | label_match | top wrong label | within_dist | ref_dist | near_copies | ita_shift |
  |---|---|---|---|---|---|---|---|
  | BCC | GPT Image varied (reference) | 0.33 | lichen planus 38 | 0.43 | 0.43 | 0 | 11.1 |
  | BCC | t2i untuned | 0.16 | squamous cell carcinoma 85 | 0.30 | 0.49 | 0 | 21.6 |
  | BCC | t2i ckpt1500 | **0.76** | squamous cell carcinoma 26 | 0.42 | 0.45 | 0 | 26.1 |
  | BCC | edit untuned varied | 0.13 | sarcoidosis 38 | 0.41 | 0.41 | 0.01 | 9.5 |
  | lupus | GPT Image varied (reference) | 0.21 | psoriasis 49 | 0.43 | 0.53 | 0 | 18.3 |
  | lupus | t2i untuned | 0.16 | allergic contact dermatitis 46 | 0.50 | 0.62 | 0 | 31.2 |
  | lupus | t2i ckpt1500 | **0.40** | psoriasis 39 | 0.58 | 0.60 | 0 | 48.1 |
  | lupus | edit untuned varied | (generating) | | | | | |

  Reference values: random sets of 10 reals have within_dist 0.45 (BCC) and 0.60 (lupus).
  t2i: the ckpt1500 LoRA is the best folder on label_match in both classes by a wide margin (BCC 0.76 vs GPT 0.33
  and untuned 0.16; lupus 0.40 vs 0.21 and 0.16), variety is at the level of real sets (0.42 vs 0.45; 0.58 vs 0.60),
  no near copies. Grids: BCC LoRA tiles are periocular/nasal/cheek lesions on older skin, pearly or ulcerated
  nodules, with the pool's text fragments; lupus LoRA tiles are malar and discoid facial lesions, scarring, oral
  lesions, mixed skin tones, plus a few off-target tiles (a butterfly, full faces, a nude torso). Untuned Qwen
  draws generic pink papules for both (probe: SCC / contact dermatitis). The t2i decision (Stage A, ckpt1500) holds.
  Edit (BCC): untuned edit with varied prompts keeps skin tone (9.5 deg) and follows the prompts, but the lesion
  becomes a small generic papule or bruise and the probe no longer calls it BCC (0.13, below the per-class line
  GPT-0.15 = 0.18). The GPT edits also score low on BCC (0.33), so this class is hard for both edit models; recorded,
  the edit decision is reviewed once the lupus edit folder is scored.
- 00:36 UTC: lupus edit untuned varied scored: label_match 0.25 (GPT 0.21; line 0.06), within_dist 0.42,
  ref_dist 0.43, near_copies 0.05, ITA shift 17.2 deg: passes every line.

**Step 2 decision, 00:38 UTC.** The Step 1 settings hold; no adjustment made.
- t2i: Stage A with the ckpt1500 per-class LoRA, confirmed on BCC (label_match 0.76 vs GPT 0.33, untuned 0.16) and
  lupus (0.40 vs 0.21, 0.16); variety at the level of random real sets in both; no near copies.
- edit: untuned Qwen with the 10 varied prompts (plan fallback). Per class: lupus passes all lines; psoriasis
  misses label_match by 0.02 and near_copies by 0.03; BCC fails label_match (0.13 vs line 0.18) because the edit
  turns the lesion into a generic papule, a class on which the GPT Image edits are also weak (0.33). Recorded as a
  known weakness of the edit arm; no in-plan adjustment fixes it without an edit LoRA, and the trained edit LoRAs
  ignore the reference (Step 1).
- The user was given the pilot update (23:43 UTC) and answered "continue as you were" (00:10 UTC); the three
  edit-LoRA alternatives listed there were not run.

**Step 3 and 4 launched, 00:38 UTC** (`qwen_lora_full_run.sh`, commands in the script header):
- GPU 0: `train-t2i` for the six remaining classes in sequence (lichen planus, allergic contact dermatitis,
  neutrophilic dermatoses, photodermatoses, sarcoidosis, squamous cell carcinoma; ~33 min each, ~3.3 h), then
  `gen-t2i`: 10 images per pool image for all nine classes with each class's ckpt1500 LoRA, single prompt,
  seed-only (33,390 images, ~2.9 s each, ~27 h).
- GPU 1: `gen-edit`: 10 images per pool image, untuned Qwen, varied prompts (slot k = variation k), reference =
  the real image (33,390 images, ~3.2 s each, ~30 h).
- Only 9 LoRAs are trained, not 18: the edit arm uses no LoRA. Edit LoRAs can be trained later if wanted.
- Pool: `generations/pool9.csv` (3,339 images, nine labels, from the training CSV). Output layout
  `generations/{t2i,edit}/{00..09}/{md5hash}.png` + `.json`. Note: `gen-t2i` runs one class at a time, so the
  `synthetic.csv` in `generations/t2i/` holds only the last class; the sweep reads the folder layout, and a
  combined CSV is built afterwards.
- Expected end: both arms about 2026-10-30 07:00 UTC. Step 5 (classifier sweep) follows.
- 00:41-00:45 UTC: Step 5 script `qwen_lora_sweep.py` written (the paper's `n_per_label.py` protocol, results to
  CSV, missing synthetic files are an error) and smoke-tested on GPU 1 next to the edit generation: real-only arm,
  seed 2026001, n in {1, 8}, both augmentations: 4 runs, 8-21 s each, early stopping after 4-11 epochs, top-1
  0.12-0.23 on the 360 test images (not results; smoke only, `sweep-smoke/`). Synthetic arms are smoke-tested once
  a seed's n=1 subset has its generated images. Sweep seeds will be 2026001-2026005 (new; the paper's other four
  seeds are not recoverable). Generations are copied from `/workspace` to `/dev/shm/qwen_lora_generations` before
  the sweep so training does not read PNGs over the network mount.
- 01:12 UTC: lichen planus t2i LoRA trained (412 images, 32.8 min); allergic contact dermatitis training (2 of 6).
  01:34: edit generation at 100/3,339 reals, 3.23 s/image (on pace for ~30 h).
- 01:44 UTC: allergic contact dermatitis t2i LoRA trained (32.4 min); neutrophilic dermatoses training (3 of 6).
  02:04: edit generation at 155/3,339 reals, 3.30 s/image (rate drifting up from 3.17; ETA ~2026-10-30 07:00 UTC).
- 02:17 UTC: neutrophilic dermatoses t2i LoRA trained (32.8 min); photodermatoses training (4 of 6).
  02:34: edit generation at 200/3,339 reals, 3.37 s/image.
- 02:50 UTC: photodermatoses t2i LoRA trained (32.8 min); sarcoidosis training (5 of 6).
- 03:04-03:17 UTC, **generation output moved to RAM.** The cumulative edit rate had drifted from 3.17 to 3.44 s/image
  while the per-call GPU time stayed at ~2.9 s: the difference is file I/O on the volume. Measured: saving one 512 px
  PNG to `/workspace` 780 ms vs 51 ms to `/dev/shm` (stat/mkdir are fast), i.e. ~7 s per pool image, ~6.8 h per arm.
  Change (operational only, same images): `qwen_lora_full_run.sh` now writes to `/dev/shm/qwen_lora_generations`
  and a `sync` loop mirrors it to `/workspace/synderm/qwen_lora/generations` every 20 min with
  `rsync -rt --no-owner --no-group`. The edit generation was stopped at 257 reals (03:06), the 2,570 existing images
  were mirrored into RAM (rsync from the volume, ~10 min), and `gen-edit 1` restarted; it skips existing files so
  nothing is regenerated. The GPU 0 chain picks up the new output path when it reaches `gen-t2i`. The pool CSV stays
  on the volume. Note for restarts: RAM output is lost with the pod, so the sync loop must be running.
- 03:23 UTC: sarcoidosis t2i LoRA trained (32.6 min); squamous cell carcinoma training (6 of 6, last).
  03:34: edit generation at 290/3,339 reals; cumulative rate since the restart 2.98 s/image (was 3.44 on the volume).
- **03:56 UTC: Step 3 complete, all 9 t2i LoRAs trained** (psoriasis, BCC, lupus, lichen planus, allergic contact
  dermatitis, neutrophilic dermatoses, photodermatoses, sarcoidosis, squamous cell carcinoma; 31-33 min each;
  checkpoints 500/1000/1500/2000 under `/workspace/synderm/qwen_lora/t2i/<slug>/r16-lr0.0001-512px-2000steps/`).
- 03:57: the chained `gen-t2i` did not start: the `train-t2i` runner ended with "syntax error near unexpected token
  `done`" because `qwen_lora_full_run.sh` was edited (sync flags) while that bash process was still reading it, so
  its `&&` chain stopped. No training was affected (all nine weight files exist). `gen-t2i 0` launched by hand at
  03:57:55 UTC, writing to RAM like the edit arm; one minute of GPU 0 time lost. Rule for the rest of the run: do not
  edit a shell script that a running job is executing; copy it first.
- 04:04 UTC: t2i generation at 10/584 psoriasis reals, 2.79 s/image; edit at 350/3,339, 3.00 s/image.
- 04:34 UTC: t2i at 70/584 psoriasis reals (2.81 s/image); edit at 410/3,339 (3.00 s/image); 35 GB in /dev/shm.
- 05:04 UTC: t2i at 140/584 psoriasis reals (2.82 s/image); edit at 470/3,339 (3.01 s/image); last sync 04:37, 5,010 images.
- 05:34 UTC: t2i at 200/584 psoriasis reals (2.82 s/image); edit at 530/3,339 (3.01 s/image); last sync 05:20, 6,790 images.
- 06:04 UTC: t2i at 260/584 psoriasis reals (2.82 s/image); edit at 590/3,339 (3.02 s/image).
- 06:34 UTC: t2i at 330/584 psoriasis reals (2.82 s/image); edit at 640/3,339 (3.02 s/image); last sync 06:15, 9,030 images.
- 07:04 UTC: t2i at 390/584 psoriasis reals (2.82 s/image); edit at 700/3,339 (3.02 s/image); sync pass in progress.
- 07:34 UTC: t2i at 460/584 psoriasis reals (2.82 s/image); edit at 760/3,339 (3.01 s/image).
- 07:40 UTC: sync changed to 20 parallel streams (`qwen_lora_sync.sh`, one rsync per method/slot folder, every
  20 min). The single-stream loop wrote ~1 file per 0.8 s, about the rate both arms produce files (one pass took
  59 min and was still running), so the mirror could not catch up. New script rather than editing
  `qwen_lora_full_run.sh`, which the two generation runners are still executing.
- 07:45 UTC: first parallel sync pass cleared the backlog in 573 s (12,730 images in RAM, mirror current).
- 08:04 UTC: t2i at 520/584 psoriasis reals (2.82 s/image); edit at 820/3,339 (3.02 s/image).
- 08:32 UTC: t2i psoriasis class complete (584 reals, 5,840 images, 2.82 s/image, 4.6 h); lichen planus started
  (class 2 of 9). Edit at 880/3,339 (3.02 s/image). Sync passes now take ~4 min.
- 09:04 UTC: t2i lichen planus at 60/412 reals (2.81 s/image); edit at 940/3,339 (3.02 s/image); mirror current 08:59.
- 09:34 UTC: t2i lichen planus at 130/412 reals (2.81 s/image); edit at 1,000/3,339 (3.02 s/image); mirror current 09:22.
- 10:04 UTC: t2i lichen planus at 190/412 reals (2.81 s/image); edit at 1,060/3,339 (3.02 s/image); mirror current 09:45.
- 10:34 UTC: t2i lichen planus at 260/412 reals (2.81 s/image); edit at 1,120/3,339 (3.02 s/image); mirror current 10:30.
- 11:04 UTC: t2i lichen planus at 320/412 reals (2.81 s/image); edit at 1,180/3,339 (3.02 s/image); mirror current 10:53.
- 11:34 UTC: t2i lichen planus at 380/412 reals (2.81 s/image); edit at 1,240/3,339 (3.02 s/image); mirror current 11:16.
- 12:04 UTC: t2i lichen planus class complete (412 reals, 4,120 images); basal cell carcinoma (class 3 of 9) at
  40/406 reals, 2.79 s/image. Edit at 1,300/3,339 (3.02 s/image); mirror current 12:01, 23,270 images.
- 12:34 UTC: t2i BCC at 100/406 reals (2.79 s/image); edit at 1,360/3,339 (3.02 s/image); mirror current 12:23.
- 13:04 UTC: t2i BCC at 160/406 reals (2.79 s/image); edit at 1,420/3,339 (3.02 s/image); mirror current 12:46.
- 13:34 UTC: t2i BCC at 230/406 reals (2.79 s/image); edit at 1,480/3,339 (3.02 s/image); mirror current 13:31.
- 14:04 UTC: t2i BCC at 290/406 reals (2.79 s/image); edit at 1,540/3,339 (3.02 s/image); mirror current 13:54.
- 14:34 UTC: t2i BCC at 360/406 reals (2.80 s/image); edit at 1,600/3,339 (3.02 s/image); mirror current 14:16.
- 14:55 UTC: t2i BCC class complete (406 reals, 4,060 images); allergic contact dermatitis (class 4 of 9) started,
  10/358 reals at 15:04. Edit at 1,660/3,339 (3.02 s/image); mirror current 15:01, 30,730 images.
- 15:34 UTC: t2i ACD at 80/358 reals (2.83 s/image); edit at 1,720/3,339 (3.02 s/image); mirror current 15:24.
- 16:04 UTC: t2i ACD at 140/358 reals (2.83 s/image); edit at 1,780/3,339 (3.02 s/image); mirror current 15:47.
- 16:05-16:17 UTC, **early single-seed sweep requested by the user.** Preview notebook
  `03_qwen_lora_preview.ipynb` written (pilot tables, grids, Step 4 sample strips; executed copy at
  `/workspace/synderm/qwen_lora/03_qwen_lora_preview.executed.ipynb`). Then the Step 4 queue was reordered, not
  changed: `qwen_lora_priority.sh build-csv 2026001` wrote the 144 reals of seed 2026001's n=16 subset (drawn exactly
  as the sweep draws them; n=1 and n=8 nest inside, checked), of which t2i already had 59 and edit 81. Both full runs
  were stopped at 16:16 (t2i at ACD 150/358, edit at 1,790/3,339; nothing lost, the scripts resume) and relaunched as
  `priority gen-t2i 0 2026001 && full_run gen-t2i 0` and `priority gen-edit 1 2026001 && full_run gen-edit 1`.
  A chain waits for both `priority-2026001.done` markers and then runs `qwen_lora_sweep.py --seeds 2026001 --n 1 8 16`
  (18 runs: 3 arms x 2 augmentations x 3 sizes) on GPU 1 next to the edit generation, writing into the final sweep
  folder `/workspace/synderm/qwen_lora/sweep` (the full sweep later skips these run ids). Expected: images ready
  ~17:15, sweep results ~17:45 UTC. Cost to the full run: two pipeline reloads plus nine extra LoRA loads (~10 min).
- 16:35 UTC: priority pass for seed 2026001: t2i at class 6 of 9 (neutrophilic dermatoses, 16 reals per class,
  ~2.5 min per class plus LoRA load); edit at 110/144 reals (3.0 s/image). Early sweep chain waiting for both
  markers. Mirror current 16:12 (33,630 images). Side work: slide deck `paper_experiments/2026/slides/` (commit 1059799).
- 16:59 UTC: priority pass for seed 2026001 complete on both arms (t2i nine classes, edit 144 reals); the full
  runs resumed (t2i at allergic contact dermatitis, edit from pool position 1,790) and the early sweep started on
  GPU 1 next to the edit generation: `qwen_lora_sweep.py --gpu 1 --seeds 2026001 --n 1 8 16`, 18 runs.
- 17:04 UTC: early sweep running on GPU 1 (results accumulate in `sweep/results.csv`); full runs resumed: t2i at
  ACD 180/358, edit at 1,830/3,339 (3.1 s/image with the sweep sharing GPU 1). Push to GitHub failed again
  (authentication); the 16:48 push succeeded, so the failure is intermittent.
- **17:08 UTC: early single-seed sweep done** (seed 2026001, 18 runs in ~8 min on GPU 1 next to the edit generation;
  `sweep/results.csv`, plot `sweep/figure3_early_seed2026001.png`). Top-1 accuracy on the 360 held-out images
  (Wald 95% CI about +-0.04):

  | n real per class | transforms | real only | + Qwen edit | + Qwen t2i |
  |---|---|---|---|---|
  | 1 | none | 0.122 | 0.153 | 0.167 |
  | 1 | aug_transforms | 0.122 | 0.144 | 0.175 |
  | 8 | none | 0.139 | 0.181 | 0.225 |
  | 8 | aug_transforms | 0.228 | 0.189 | 0.192 |
  | 16 | none | 0.222 | 0.228 | 0.244 |
  | 16 | aug_transforms | 0.219 | 0.228 | 0.267 |

  One seed, so none of the differences is significant on its own. Pattern: the Qwen t2i LoRA arm is above real-only
  in 5 of 6 cells (by 0.02-0.09), the edit arm in 5 of 6 (by 0.01-0.04); the exception for both is n=8 with
  transforms, where the real-only run happened to land high (0.228). For comparison the proprietary pilot (seed
  20261028, different real subsets) had GPT Image edits at 0.25 and Nano Banana at 0.25 at n=16. These runs are
  part of the final sweep (same run ids), not a separate experiment.
- 17:34 UTC: t2i at ACD 240/358 (2.83 s/image); edit at 1,890/3,339 (3.20 s/image since the resume); mirror current 17:21.
- 18:04 UTC: t2i at ACD 310/358 (2.83 s/image); edit at 1,950/3,339 (3.13 s/image); mirror current 17:43.
- 18:35 UTC: t2i ACD class complete; lupus (class 5 of 9) at 20/297 reals (2.82 s/image). Edit at 2,010/3,339
  (3.10 s/image). Mirror current 18:28, 39,100 images. Pushed to GitHub at 18:2x (bf85b10); pushes succeed intermittently.
- 19:04 UTC: t2i lupus at 80/297 reals (2.83 s/image); edit at 2,070/3,339 (3.08 s/image); mirror current 18:51, 40,026 images.
- 19:34 UTC: t2i lupus at 150/297 reals (2.82 s/image); edit at 2,140/3,339 (3.07 s/image); mirror current 19:14, 40,960 images.
- 20:04 UTC: t2i lupus at 220/297 reals (2.83 s/image); edit at 2,200/3,339 (3.07 s/image); mirror current 19:59, 42,800 images.
- 20:34 UTC: t2i lupus at 290/297 reals (2.83 s/image), neutrophilic dermatoses next; edit at 2,260/3,339 (3.06 s/image); mirror current 20:21, 43,720 images.

## Spend ledger

| When | What | Model | Images | Cost (from usage) | Session total |
|---|---|---|---|---|---|
