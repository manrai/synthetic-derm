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

## Spend ledger

| When | What | Model | Images | Cost (from usage) | Session total |
|---|---|---|---|---|---|
