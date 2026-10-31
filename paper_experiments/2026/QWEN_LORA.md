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

**Step 6. Figure 5 (dose-response) with Nano Banana 2.1 edits, 16 real per class first, one seed (planned
2026-10-30, not started; runs only after the Figure 3 Nano Banana arm is finished and the user says start).**
Paper Figure 5 (main.tex, "dose-response and saturation"): 16, 32, 64 real per class x 0, 10, 25, 50, 75,
100, 125, 150 synthetic per real x {no transforms, aug_transforms}; the paper used DreamBooth text-to-image,
this arm uses Nano Banana reference-image edits. Step 6 runs the 16-per-class panel; Step 7 adds 32 and 64.
- Protocol: `paper_experiments/Experiments/dose_response.py`, not `n_per_label.py`. Same split as Step 5
  (nine labels, 40 val per label from the pool with `random_state`, n per label from the rest with the same
  `random_state`, 360 held-out test), but two differences that the new sweep script keeps: a batch size of 128
  when n_train/10 >= 500, and `SaveModelCallback` + load of the best valid_loss model before testing.
- Seed: **119108**, one of the two 2023 Figure 5 seeds (119108, 329101: the first field of the model file names in
  the 2023 `derm/dose-response/` folder, which cover 16/32/64 x doses 10-75 x both transform settings; 4 of the 48
  files are missing). Confirmed 2026-10-31 that the draw reproduces 2023: the 2023 machine's
  `Fitzpatrick17k/fitzpatrick17k_10label_clean_{training,held_out_set}.csv` have the same md5 as `fitz_metadata/`
  (11b4b1c7..., 8ff2299d...), and the 2023 generation folders (`generations-more`, `generations-lots-more`) cover
  exactly the 3,530 training-CSV images with the same labels and no held-out image (`fig5_per_real_counts.csv` from
  `check_original_dose_response_images.py`). The groupby sampling of `dose_response.py` and the per-label loop in
  `fig5_dose_response_sweep.py` give identical draws for both seeds. 16 and 32 nest inside 64, so the Step 6 images are
  reused by Step 7. Step 6 reals: `/workspace/synderm/fig5/reals-119108-n16.csv` (144, all in the Figure 3 pool, so
  slots 0-9 exist).
- 2023 originals, for a same-reals comparison arm (GPU only): every training image has 145-200 text-to-image
  generations (`generations-more` slots 10-99 + `generations-lots-more` slots 0-109, separate images: 0 of 300
  overlapping pairs identical). Lichen planus, psoriasis and squamous cell carcinoma have only 145-149, so a dose of
  150 with the 2023 images is short for those classes; the 2023 model files stop at dose 75.
- Prompts: 150 edit variations in `api_generate_pool.py`. Slots 0-9 keep the current 10 and reuse the Figure 3
  images; slots 10-149 are 14 body sites (`DOSE_SITES`) x 10 photo setups (`DOSE_SETUPS`), slot 10 + j = site j % 14,
  setup (j // 14 + j % 14) % 10, so any 14 consecutive slots cover every site and setup. Slot k uses prompt k; a run
  with k per real uses slots 0..k-1, so the doses nest.
- Generation: `api_generate_pool.py --reals /workspace/synderm/fig5/reals-119108-n16.csv --k 150 --topup-ref same-slot
  --pilot-dir ''`, same output folder (slots `10..149` next to `00..09`). `--topup-ref same-slot`: a refused slot is
  filled from up to 3 other reals of the class whose own edit for that slot was accepted (the Figure 3 rule, "all
  slots accepted", almost never holds at 150 slots).
- Sweep: `fig5_dose_response_sweep.py` (dose_response.py protocol: batch 128 tier, best-epoch weights) on 512 px
  copies, doses 0 10 25 50 75 100 125 150.
- Paid pilot first: 1 real x 140 new slots for 2 classes (280 images, ~$10) to check the prompts give
  varied images, then the run.
- Size and cost: 144 x 140 = 20,160 new images + ~2% top-ups = ~20,600 at the measured $0.0336/image =
  **~$690** including the pilot (Batch API, if it supports this model: ~$350 with up to 24 h turnaround).
  Separate cap to be set by the user; not part of the $1,200 Figure 3 cap. ~10 h at 20 threads (35 images/min
  measured).
- Classifier: 8 doses x 2 transforms = 16 runs at n=16, one GPU. The 0 and 10 doses are rerun with this
  protocol rather than taken from Step 5 (batch size differs). ~2M image-epochs, about 1-2 GPU-hours.
  Storage: the 21,600 synthetic images used are ~29 GB at 1K PNG; copied to local disk (45 GB free) for training.
- Output: `results.csv` and the 16-per-class panel of a Figure 5-style plot with Wald 95% CIs and BH-adjusted
  two-proportion tests as in the paper; one seed, so no between-run spread. Reviewed with the user before Step 7.

**Step 7. Figure 5 full run: add 32 and 64 real per class (planned 2026-10-30, not started; only after the
user reviews Step 6 and says start).** Same protocol, seed, prompts and script as Step 6.
- Generation: `--reals` = `/workspace/synderm/fig5/reals-119108-n64.csv` (576 reals); the 144 from Step 6 are skipped as
  done. 432 x 140 = 60,480 new images + ~2% top-ups = ~61,700 = **~$2,070** (Batch API: ~$1,040). Steps 6 + 7
  together: ~$2,760. ~30 h at 20 threads. Separate cap to be set by the user.
- Classifier: n in {32, 64} x 8 doses x 2 transforms = 32 runs (n=16 is reused from Step 6). ~12M image-epochs,
  about 5-10 GPU-hours.
- Storage: all 86,400 synthetic images used are ~117 GB at 1K PNG; this pod has 24 GB /dev/shm and 45 GB root
  free, so training reads 512 px copies made once (originals untouched) or runs on a pod with more RAM.
- Output: the full three-panel Figure 5-style plot and table, recorded here.

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
- 21:04 UTC: t2i lupus class complete; neutrophilic dermatoses (class 6 of 9) at 50/284 reals (2.83 s/image).
  Edit at 2,320/3,339 (3.06 s/image). Mirror current 20:43, 44,621 images.
- 21:34 UTC: t2i neutrophilic dermatoses at 120/284 reals (2.81 s/image); edit at 2,390/3,339 (3.05 s/image); mirror current 21:28, 46,490 images.
- 22:04 UTC: t2i neutrophilic dermatoses at 190/284 reals (2.81 s/image); edit at 2,450/3,339 (3.05 s/image); mirror current 21:51, 47,420 images.
- 22:34 UTC: t2i neutrophilic dermatoses at 260/284 reals (2.81 s/image), photodermatoses next; edit at 2,510/3,339 (3.05 s/image); mirror current 22:13, 48,340 images.
- 23:04 UTC: t2i neutrophilic dermatoses class complete; photodermatoses (class 7 of 9) at 40/272 reals (2.83 s/image).
  Edit at 2,580/3,339 (3.04 s/image). Mirror current 22:58, 50,190 images.
- 23:34 UTC: t2i photodermatoses at 110/272 reals (2.82 s/image); edit at 2,640/3,339 (3.04 s/image); mirror current 23:21, 51,115 images.
- 2026-10-30 00:04 UTC: t2i photodermatoses at 180/272 reals (2.81 s/image); edit at 2,700/3,339 (3.04 s/image); mirror current 23:43, 52,050 images.
- 00:34 UTC: t2i photodermatoses at 250/272 reals (2.81 s/image), sarcoidosis next; edit at 2,760/3,339 (3.03 s/image); mirror current 00:28, 53,890 images.
- 01:04 UTC: t2i photodermatoses class complete; sarcoidosis (class 8 of 9) at 40/286 reals (2.82 s/image).
  Edit at 2,820/3,339 (3.03 s/image). Mirror current 00:50, 54,810 images.
- 01:34 UTC: t2i sarcoidosis at 110/286 reals (2.80 s/image); edit at 2,880/3,339 (3.03 s/image); mirror current 01:13, 55,740 images.
  Step 5 chains armed: when the edit runner exits and all 33,390 edit images are present, `qwen_lora_sweep.py --gpu 1
  --arms none edit --seeds 2026001..2026005` starts on GPU 1; when the t2i runner exits and all 33,390 t2i images are
  present, `--gpu 0 --arms t2i` starts on GPU 0. Both write to `sweep/results.csv` (run ids already there, e.g. the
  early seed-2026001 n<=16 runs, are skipped). Each chain stops with a report if the image count is short.
- 02:04 UTC: t2i sarcoidosis at 180/286 reals (2.80 s/image); edit at 2,950/3,339 (3.03 s/image); mirror current 01:58, 57,600 images.
- 02:34 UTC: t2i sarcoidosis at 250/286 reals (2.80 s/image), then squamous cell carcinoma (440 reals, ~3.4 h);
  edit at 3,010/3,339 (3.03 s/image, ~2.7 h left at 30 s per pool image). ETA: edit ~05:20, t2i ~06:15 UTC.
  Mirror current 02:20, 58,530 images.
- 03:04 UTC: t2i sarcoidosis class complete; squamous cell carcinoma (class 9 of 9, last) at 30/440 reals
  (2.80 s/image). Edit at 3,070/3,339 (3.03 s/image). Mirror current 02:43, 59,481 images.
- 03:34 UTC: t2i SCC at 90/440 reals (2.81 s/image); edit at 3,130/3,339 (3.03 s/image); mirror current 03:28, 61,330 images.
- 04:04 UTC: t2i SCC at 160/440 reals (2.82 s/image); edit at 3,200/3,339 (3.02 s/image); mirror current 03:50, 62,250 images.
- 04:34 UTC: t2i SCC at 220/440 reals (2.82 s/image); edit at 3,260/3,339 (3.02 s/image, ~40 min left); mirror current 04:13, 63,190 images.
- 05:04 UTC: t2i SCC at 290/440 reals (2.82 s/image); edit at 3,320/3,339 (~10 min left); mirror current 04:59, 65,070 images.
- **05:09 UTC: edit generation complete**: 33,390 images (3,339 pool images x 10, untuned Qwen, varied prompts),
  about 25.5 GPU-hours net; verified count in RAM. The chained sweep did not start on its own: its wait pattern also
  matched the (still alive) shell that launched both generation chains at 16:17, so it was stopped and the sweep was
  started by hand at 05:35 UTC: `qwen_lora_sweep.py --gpu 1 --seeds 2026001..2026005 --arms none edit`
  (140 runs; the 12 early seed-2026001 runs are skipped). The t2i chain is replaced by one that waits for the
  "gen-t2i done" line in the t2i log instead of a process pattern.
- 06:04 UTC: GPU 1 sweep at 32 runs in `results.csv` (seed 2026001 real-only and edit arms nearly complete; runs
  take 40 s to 5 min, n=128 edit with transforms 304 s). t2i generation at SCC 420/440 reals. Mirror current 05:43.
- **06:09 UTC: t2i generation complete, Step 4 done.** 33,390 t2i images (nine classes, ckpt1500 LoRAs, single
  prompt, 10 seeds per pool image) verified in RAM; with the edit arm, 66,780 images, about 52 GPU-hours net over
  ~29.5 h wall-clock on two GPUs (including the 16:17 reorder and the RAM switch). The t2i sweep arm started on
  GPU 0 at 06:09 (`--arms t2i`, five seeds, 70 runs, 6 early runs skipped); the real-only + edit arms continue on
  GPU 1. The sync loop keeps mirroring RAM to `/workspace/synderm/qwen_lora/generations/` every 20 min; the final
  pass is verified before the loop is stopped.
- 06:34 UTC: sweep at 66 of 210 runs (real-only and edit: seed 2 of 5; t2i: seed 1 done, 13 runs). n=228 runs
  take ~5 min (t2i: 22,572 training images, 10 epochs, 303 s). Mirror: 66,780 images in RAM, last pass 40 s (nothing
  left to copy); volume count being verified before the sync loop is stopped. Pushed to GitHub (2fe2cf3 and later).
- 06:45 UTC: mirror verified by stat and size for all 66,780 images (`/workspace/synderm/qwen_lora/generations/`,
  0 missing, 0 size mismatches, 2 s with 32 threads). A plain directory listing of `edit/05` on the volume returned
  only 790 of 3,339 names (stale geesefs listing); stat-based checks are the reliable test on this mount. Sync loop
  stopped. The sweep reads from the RAM copy.
- 07:04 UTC: sweep at 104 of 210 runs (real-only and edit on seed 3 of 5; t2i on seed 2 of 5). No errors.
- 07:34 UTC: sweep at 142 of 210 runs (real-only and edit on seed 4 of 5; t2i on seed 3 of 5). No errors.
- 08:04 UTC: sweep at 180 of 210 runs (real-only and edit on seed 5 of 5; t2i on seed 4 of 5). No errors.
- 08:34 UTC: sweep at 204 of 210 runs (real-only complete; edit 69/70; t2i 65/70). No errors.
- 08:38 UTC: GPU 1 sweep done: real-only and edit arms complete (70 runs each, 3 h 3 min). t2i arm at 66/70 on GPU 0.

**Step 5 complete, 09:00 UTC (2026-10-30).** 210 runs (7 sizes x 3 arms x 2 transform settings x 5 seeds), `sweep/results.csv`,
plot `sweep/figure3_qwen.png`, table `sweep/figure3_qwen_table.{csv,md}` (`qwen_lora_figure3.py`). Real-only and edit arms
ran on GPU 1 05:35-08:38, t2i arm on GPU 0 06:09-09:00. Top-1 accuracy on the 360-image held-out test, mean +- SD over
seeds 2026001-2026005 (n seeds):

| n real per class | transforms | real only | + Qwen edit | + Qwen t2i |
|---|---|---|---|---|
| 1 | None | 0.123 ± 0.020 (5) | 0.130 ± 0.026 (5) | 0.177 ± 0.015 (5) |
| 1 | aug_transforms | 0.136 ± 0.018 (5) | 0.137 ± 0.025 (5) | 0.158 ± 0.010 (5) |
| 8 | None | 0.165 ± 0.028 (5) | 0.198 ± 0.018 (5) | 0.221 ± 0.009 (5) |
| 8 | aug_transforms | 0.183 ± 0.030 (5) | 0.196 ± 0.020 (5) | 0.201 ± 0.011 (5) |
| 16 | None | 0.203 ± 0.015 (5) | 0.236 ± 0.012 (5) | 0.253 ± 0.014 (5) |
| 16 | aug_transforms | 0.215 ± 0.026 (5) | 0.236 ± 0.041 (5) | 0.257 ± 0.008 (5) |
| 32 | None | 0.216 ± 0.016 (5) | 0.293 ± 0.017 (5) | 0.289 ± 0.012 (5) |
| 32 | aug_transforms | 0.213 ± 0.015 (5) | 0.317 ± 0.014 (5) | 0.305 ± 0.014 (5) |
| 64 | None | 0.267 ± 0.011 (5) | 0.387 ± 0.025 (5) | 0.322 ± 0.016 (5) |
| 64 | aug_transforms | 0.301 ± 0.018 (5) | 0.387 ± 0.014 (5) | 0.344 ± 0.013 (5) |
| 128 | None | 0.331 ± 0.013 (5) | 0.444 ± 0.015 (5) | 0.373 ± 0.018 (5) |
| 128 | aug_transforms | 0.359 ± 0.022 (5) | 0.471 ± 0.025 (5) | 0.397 ± 0.041 (5) |
| 228 | None | 0.390 ± 0.031 (5) | 0.521 ± 0.022 (5) | 0.436 ± 0.018 (5) |
| 228 | aug_transforms | 0.444 ± 0.025 (5) | 0.533 ± 0.021 (5) | 0.443 ± 0.027 (5) |

Reading (five seeds; SD across seeds, not a confidence interval):
- Both Qwen arms are above real-only at every n and both transform settings except n=1 edit, where they tie.
- Edit arm (untuned Qwen, varied prompts, images derived from each real): +3 to +4 points at n=8-16, +8 to +10 at
  n=32-64, +11 points at n=128, +9 to +13 at n=228 (0.52-0.53 vs 0.39-0.44). The gain grows with n rather than
  vanishing as in the paper's SD arms. Interpretation to check: about half the edit slots stay close to the reference
  (pilot), so this arm also acts as strong augmentation of the real images; the gain may partly reflect that rather
  than new disease information.
- t2i LoRA arm: +2 to +5 points at n=1-16, +7 to +9 at n=32, +4 to +5 at n=64-128, +5 without transforms and
  ~0 with transforms at n=228. Closer to the paper's pattern (gains at small and mid n, vanishing at 228 with transforms).
- Real-only baseline: 0.12 at n=1 to 0.39/0.44 at n=228 (without/with transforms).
- Comparison with the published SD arms is between figures, not within one run (different seeds, the original seeds
  are not recoverable); the API arms exist only for n<=16, one seed.
Total: $0 of the $50 API cap spent in this session; everything ran locally.

### 2026-10-30, after Step 5: backup check of the generations on /workspace

The user asked whether every synthetic image is on /workspace (they will go to Hugging Face later). Checked the RAM
copy (/dev/shm/qwen_lora_generations) against the volume copy (/workspace/synderm/qwen_lora/generations) with
`find -type f -printf '%P %s'` on both sides, sorted, and `diff`: 133,566 files (66,780 PNG + 66,780 JSON sidecars
+ config files) identical in name and size. Two gaps fixed by `cp`: the top-level `t2i/synthetic.csv` (589,628 B) and
`edit/synthetic.csv` (4,318,898 B) had never been synced (the parallel sync covered only the slot folders), and
`t2i/generate_config.json` on the volume was from an earlier run. One zero-byte rsync temp file
(`t2i/03/.0b4c...json.y4zXyj`) removed from the volume. Volume copy is 28 GB; the pool9.csv and priority-2026001.csv
there are extras written to the volume directly. The LoRA weights (t2i/<class>/.../checkpoint-*, edit/psoriasis),
pilot folders and sweep results were written to /workspace from the start.

Slides: the three-generator comparison slide now uses the five-seed Qwen means for both Qwen arms at n<=16
(`slides/build_figure3_compare.py` -> `fig/figure3_compare.png`); a table slide with the full sweep numbers was
added after the five-seed figure (19 slides).

### 2026-10-30, 20:00 UTC: Figure 3 proprietary arm, Nano Banana 2.1 only (user request)

The user asked for the GPT Image and Nano Banana arms at all n, then restricted it to Nano Banana (only a Gemini key is
available). Caps set by the user and not to be raised by this agent: **$1,200 for Nano Banana** (GPT Image, if run
later: $700). On an API limit or any error that stops the run: stop and ask, no workaround.

Plan: `api_generate_pool.py` (new, derived from `pilot_generate.py`) makes 10 varied-prompt edits for every image of
`generations/pool9.csv` (3,339 reals) in the sweep's layout `api_generations/nano-banana-2.1/<00..09>/<md5>.png`,
importing the 1,440 pilot edits first (no new spend), refusals retried once then topped up with an edit of another
accepted reference of the class. Expected: 31,950 new images at the pilot's $0.0334 = ~$1,070. Then the sweep with
`--arms nano-banana-2.1`, five seeds, 70 runs on one GPU, reusing the real-only rows. A 30-minute check-in loop
(cron a4ed34ef) watches the run.

- 20:04 UTC: dry run started (imports the pilot images; ~0.8 s per file write on the volume).
- 20:17 UTC: dry run done: 1,440 pilot images imported, 33,390 slots, 31,950 to run (~$1,067 at the pilot's rate).
- 20:17-20:20 UTC: paid test, 20 psoriasis reals x 2 slots, $2 cap: 38 ok + 2 refused, both topped up from another
  accepted psoriasis reference; 40 of 40 slots filled, $1.34, 24 s latency, 1,120 output image tokens per image.
- 20:22 UTC: full run launched: `api_generate_pool.py --model nano-banana-2.1 --budget 1198 --threads 20`
  (cap 1,198 so that test + run stay within the user's $1,200), log `api-nano-banana-fig3.log`.
- 20:26-21:52 UTC: steady at 33-37 images/min, 0 errors, 0 rate limits; refusals ~2% (61 of 3,312 by 21:52).
- **22:02 UTC: stopped by hand (SIGTERM) at the user's request** (pod being shut down; the run continues on a new
  pod). Last totals from progress.json: ok=3,697, refused=61, error=0, cost **$124.22** this process (+$1.34 test =
  $125.56 of the $1,200 cap; **$1,074 remains**). Up to 20 in-flight requests were lost at the kill (no file, no JSON;
  they are redone on resume; at most ~$0.70). Images and JSON sidecars are written straight to the volume, nothing
  was in RAM. Resume on any machine (no GPU needed) with the same command and the remaining cap:
  `python paper_experiments/2026/api_generate_pool.py --model nano-banana-2.1 --budget 1074 --threads 20`
  (finished slots are skipped; the top-up pass runs after the main pass). Check-in loop a4ed34ef cancelled.

### 2026-10-30, 22:30 UTC: Figure 3 Nano Banana arm resumed on a new pod (no GPU needed; A40 pod)

- Setup: `pip install pandas pillow google-genai`; reals copied to `/root/data/fitzpatrick17k/finalfitz17k` (16,536 files,
  24-way cp, 1.5 min). Sum of `cost_usd` over all non-imported sidecars: **$125.56** (matches the ledger). Allowance for
  the 20 requests killed at the 22:02 shutdown: $0.70, so $1,073.74 left under the user's $1,200.
- 22:39 dry run: 33,390 slots, 5,240 done, 28,150 to run (~$940).
- Thread test (user asked to push parallelism up to the rate limit). Each step was stopped by SIGTERM; in-flight requests
  lost at each kill are counted as spent (threads x $0.0336) in the budget:
  | threads | images/min | median / p90 latency | 429s | log |
  |---|---|---|---|---|
  | 20 (previous pod) | 33-37 | 24 s | 0 | api-nano-banana-fig3.log |
  | 40 | ~98 | 16.5 / 31.7 s | 0 | pod-2026-10-31/api-nano-banana-fig3-t40.log |
  | 80 | ~142 | 17.7 / 45.8 s | 0 | pod-2026-10-31/api-nano-banana-fig3-t80.log |
  | 120 | ~140 | | 0 | pod-2026-10-31/api-nano-banana-fig3-t120.log |
  Throughput plateaus at ~140/min between 80 and 120 threads with no 429s (a throughput ceiling, not a rate limit).
- 22:57 UTC: running at 120 threads, `--budget 1003` = $1,200 - $187.25 recorded - $4.73 kill allowance - 120 x $0.0336
  overshoot margin (the cap is checked after each request completes). 26,300 requests at start. ETA main pass ~02:10 UTC
  Oct 31, then the top-up pass. Projected total ~$1,090. 30-minute check-in cron 18c38652 (session-only).
- 23:14 UTC check-in: 2,100 of 26,300 (2,081 ok, 19 refused), 0 errors, 0 429s, ~139 images/min; total spend $261.90; ETA main pass ~02:08 UTC.
- 23:44 UTC check-in: 6,610 of 26,300 (6,568 ok, 42 refused), 0 errors, 0 429s, ~149 images/min; total spend $412.67; ETA main pass ~01:57 UTC.
- 2026-10-31 00:14 UTC check-in: 10,841 of 26,300 (10,766 ok, 75 refused), 0 errors, 0 429s, ~141 images/min; total spend $553.72; ETA main pass ~02:04 UTC.
- 00:44 UTC check-in: 14,987 of 26,300 (14,876 ok, 111 refused), 0 errors, 0 429s, ~139 images/min; total spend $691.81; ETA main pass ~02:06 UTC.
- 01:14 UTC check-in: 19,468 of 26,300 (19,267 ok, 201 refused), 0 errors, 0 429s, ~149 images/min; total spend $839.35; ETA main pass ~02:00 UTC.
- 01:44 UTC check-in: 24,090 of 26,300 (23,766 ok, 324 refused), 0 errors, 0 429s, ~154 images/min; total spend $990.52; ETA main pass ~01:59 UTC, then top-up (~450 refused slots).
- 02:14 UTC check-in: **main pass ended 01:58 UTC**: ok=25,909, refused=391, error=0, 0 429s, $870.54 this process; total spend $1,062.52. Top-up pass: 462 refused slots found 02:08; scanning references per class (slow JSON reads on the volume, ~4 min per class) before ~$16 of top-up requests; ETA DONE ~03:00 UTC.
- 02:44 UTC check-in: top-up reference scan at 6 of 9 classes (~5 min each), no new requests yet; total spend $1,062.52; then ~462 top-up requests and the synthetic.csv write (33k sidecar reads); ETA DONE ~03:30 UTC.
- 03:14 UTC check-in: **top-up ended 03:00 UTC**: 461 of 462 refused slots filled, 1 still missing (refused twice as a top-up too); $886.03 this process; total spend **$1,078.01**. No more requests; writing synthetic.csv (33k sidecar reads).
- **03:34 UTC: DONE.** 33,389 of 33,390 slots have images; `synthetic.csv` written (33,389 rows, 470 top-ups). This
  process: ok=26,370, refused=392, error=0, 0 429s, $886.03. Sum of `cost_usd` over all non-imported sidecars:
  **$1,073.28**; with the $4.73 allowance for requests killed at restarts: **$1,078.01 of the $1,200 cap**.
  Missing: lichen planus `f03efb152b525450bdd403f8a51e7405` slot 07 (own edit refused twice, top-up refused too).
  `qwen_lora_sweep.py` stops on a missing synthetic file, so any seed that samples this real for the Nano Banana arm
  will fail until the user decides how to fill it. Check-in cron 18c38652 deleted.

### 2026-10-31: Figure 3 sweep, Hugging Face backup, Figure 5 Step 6 (batch)

- 13:12 UTC: missing lichen planus slot 07 filled (another accepted reference, 1 request, $0.0336). 13:37: 6 PNGs cut
  short by the SIGTERMs of 2026-10-30 (no sidecar, written at the kill times) found by the 512 px copy, moved to
  `nano-banana-2.1-truncated-at-kill/`, regenerated ($0.20); `api_generate_pool.py` now writes PNGs via temp + rename.
- 14:22 UTC: Figure 3 Nano Banana sweep started on the A40 (`qwen_lora_sweep.py --arms nano-banana-2.1`, 5 seeds,
  70 runs, 512 px copies in /root/data/api512), out /workspace/synderm/api_sweep/nano-banana-2.1.
- Cost formula bug found: `api_generate_pool.py` priced output image tokens only; input ($1.50/M standard, ~1,180
  tokens per edit) was never counted, so the Figure 3 ledger was ~5% low: $56.55 for the accepted edits (37.7M input
  tokens), ~$1,135-1,137 in total, still under the $1,200 cap (ledger corrected). Fixed in both scripts.
- Batch mode tested (inline 4 images, 131 s; file-based 2 images, 188 s, $0.10 together): works, $0.0168 per 1K image
  + $0.75/M input. `api_generate_batch.py` written. Step 6 cap set by the user: $380 (expected ~$357 + top-ups).
- 15:04 UTC: Step 6 batch pilot submitted (2 reals x slots 10-149 = 280 requests, worst case $5).
- 15:16 UTC check-in: sweep 11/70 (n128 None seed 2026001 top1 0.431, 891 s; ~2 h per seed, ETA ~00:30 UTC Nov 1).
  Backup had stopped at 14:40: `pilot` tar exited 1 three times with "file changed as we read it" (geesefs timestamp
  quirk); fixed to accept that warning only when the tar's file count and bytes match the source, restarted (PID 84891).
  Batch pilot job running since 15:04.
- 15:39 UTC check-in: sweep 12/70 (n128 aug top1 0.439, 1,459 s; n=128/228 runs slower than estimated, ~2.75 h per seed, ETA ~04:00 UTC Nov 1). Backup: all 9 pilots tars verified (pilot.tar passed with the new check), Qwen edit/00.tar in progress. Batch pilot still RUNNING at Google (35 min), $0 recorded of $380.
- 15:45-15:52 UTC: user asked for a fast non-batch pilot and the full dose-response run (16/32/64) in batch, Figure 5 cap
  $1,500. Non-batch pilot: lupus 460d3e4a + SCC 7075c2cd x slots 10-149, 280 ok, 0 refused, $9.90 (input tokens
  included); contact sheets: body sites follow the prompts, varied framing/age/background, skin tone and class look
  kept. Batch pilot poller stopped (its job stays in batch_state.json and is collected by the full run).
- 16:15 UTC check-in: sweep 13/70 (n228 None seed 2026001 top1 0.508, 1,908 s). Backup 11 units (9 pilots + Qwen
  edit/00, 01). Full-run dry run (576 reals x 140) still checking 80k paths on the volume (started 15:53); batch pilot
  job still RUNNING at Google after 70 min; Figure 5 spend so far $9.90 + $0.11 of $1,500.
- 16:35 UTC: batch pilot job done (~90 min turnaround): 280 ok, 0 refused, $4.95. Path checks in api_generate_batch.py
  parallelized (64 threads; dry run 50 min -> 4 min). 16:48 UTC: full run launched (576 reals x slots 10-149, 80,080
  requests, expected ~$1,416 + top-ups; --budget 1489.99), log fig5-batch-full.log.
- 16:48 UTC check-in: sweep 21/70 (seed 2026001 done; n32 None seed 2026002 top1 0.303; ~2.2 h per seed, ETA ~01:30
  UTC Nov 1). Backup 12 units (Qwen edit/02 done). Full batch run starting (no jobs submitted yet); Figure 5 spend
  $9.90 + $0.11 + $4.95 = $14.96 of $1,500.

## Spend ledger

| When | What | Model | Images | Cost (from usage) | Session total |
|---|---|---|---|---|---|
| 2026-10-30 20:17-20:20 UTC | Figure 3 arm, paid test (20 psoriasis reals x 2 slots + 2 top-ups) | gemini-nano-banana-2.1, 1K | 40 | $1.34 | $1.34 |
| 2026-10-30 20:22-22:02 UTC | Figure 3 arm, full run part 1, cap $1,198, stopped by hand for pod shutdown | gemini-nano-banana-2.1, 1K | 3,697 ok + 61 refused | $124.22 | $125.56 |
| 2026-10-30 22:39-22:57 UTC | Figure 3 arm, part 2: thread tests at 40 and 80 (stopped by hand), killed in-flight allowance $4.03 | gemini-nano-banana-2.1, 1K | ~1,750 ok | $61.69 + $4.03 | $191.98 |
| 2026-10-30 22:57 - 10-31 03:34 UTC | Figure 3 arm, part 3: 120 threads, main pass + top-up, DONE (1 slot missing) | gemini-nano-banana-2.1, 1K | 26,370 ok + 392 refused | $886.03 | $1,078.01 |
| 2026-10-31 13:12-13:37 UTC | Figure 3 arm: missing lichen planus slot 07 + 6 PNGs truncated at the kills, regenerated | gemini-nano-banana-2.1, 1K | 7 ok | $0.24 | $1,078.25 |
| 2026-10-31 (correction) | **Input tokens, never counted by the script**: 37,699,095 input tokens over the 31,950 accepted edits at $1.50/M; plus up to ~$2 for refused and killed requests (their input tokens are not in the sidecars) | gemini-nano-banana-2.1 | | $56.55 (+ <=$2) | **~$1,135-1,137 of $1,200** |
| 2026-10-31 14:40-15:00 UTC | Batch mode tests (4 inline + 2 file-based edits, batch rate) | gemini-nano-banana-2.1, 1K, batch | 6 ok | ~$0.11 | |
