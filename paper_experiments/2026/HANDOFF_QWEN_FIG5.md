# Handoff: Qwen-Image-2.1 edit arm for Figure 5 (dose-response), 4-GPU pod

Written 2026-10-31 by the agent on the A40 pod, which keeps running the Nano Banana (API) work. Read this, then
`QWEN_LORA.md` (Steps 6-7 and the 2026-10-31 run log) and `REPRODUCE.md` sections 4-6, before doing anything.

## Goal

The Qwen edit version of the paper's Figure 5: classifiers trained on 16, 32 and 64 real images per class plus
0, 10, 25, 50, 75, 100, 125 or 150 synthetic images per real, with and without `aug_transforms`, one seed (119108),
tested on the 360 held-out images. The Nano Banana arm of the same experiment is running on the other pod; the two
arms must share the reals, the prompts and the classifier protocol so they can be compared directly.

## What is fixed (do not change without asking the user)

- **Reals:** seed 119108 (a 2023 Figure 5 seed; the split is verified against the 2023 machine). Lists:
  `/workspace/synderm/fig5/reals-119108-n64.csv` (576 reals; the 16 and 32 draws are subsets) and
  `reals-119108-n16.csv` (144). Written by `fig5_dose_response_sweep.py --write-reals`.
- **Generator settings = the Figure 3 Qwen edit arm:** untuned `Qwen/Qwen-Image-2.1` (revision d26bb612, in the volume
  HF cache), reference image = the real, 512 px, 40 steps, seed of slot k = `int(md5[:8], 16) + k`, edit prompt =
  `EDIT_PROMPT` + `" "` + variation k.
- **Prompts:** 150 variations, `EDIT_VARIATIONS` in `api_generate_pool.py` (slots 0-9 = the Figure 3 list, slots
  10-149 = `DOSE_SITES` x `DOSE_SETUPS`). Nano Banana uses exactly this list.
- **Slots 0-9 already exist** for every pool real: `/workspace/synderm/qwen_lora/generations/edit/<00..09>/<md5>.png`
  (+ `.json`). Copy them for the 576 reals; do not regenerate them (the Figure 3 sweep used those files).
- **Classifier:** `fig5_dose_response_sweep.py` (the 2023 `dose_response.py` protocol: batch size 128 tier, best
  valid-loss epoch loaded before testing, `set_seed(119108)`).

## Work

1. **Environment.** Clone `modernize`, recreate `.env` (only needed for HF uploads; no paid API calls on this pod),
   `bash bootstrap_pod.sh`, `pip install fastai==2.8.12 fastcore==2.2.35 statsmodels==0.15.0`. Restore Claude memory
   from `/workspace/backups/claude-memory/`. Run `nvidia-smi` and record the GPU model and memory in the log:
   the time estimates assume RTX PRO 6000 Blackwell (96 GB). If the cards are RTX 6000 Ada (48 GB), Qwen-Image in bf16
   will not fit comfortably: stop and report options (8-bit text encoder as in `synderm/backends/qwen_image.py`,
   CPU offload) with measured speed before generating.
2. **Data to fast storage.** Model weights from the volume HF cache to `/dev/shm/hf` and run with `HF_HOME=/dev/shm/hf`
   (the volume reads at ~8 MB/s and can hang). Reals to `/root/data/fitzpatrick17k/finalfitz17k` with the 24-way
   parallel `cp` from `REPRODUCE.md` section 5.
3. **Script change (small, commit it).** `qwen_lora_generate.py` uses its own 10-item `VARIATIONS` and `k % 10`, so
   slots 10+ would repeat prompts 0-9. Make `--prompts varied` use `EDIT_VARIATIONS` from `api_generate_pool.py`
   for slot k (no modulo; error if k >= 150), keeping slots 0-9 byte-identical to today's prompts (check with an
   assert against the old list). Add a `--slots FIRST LAST` option (or equivalent) so slots 10-149 can be generated
   without touching 0-9.
4. **Smoke test** on one GPU: 2 reals x 3 slots (e.g. 10-12) into a scratch folder; check the prompt text and seed in
   the JSON sidecars, and time per image (Figure 3 measured ~3.1 s per 512 px image on an RTX PRO 6000).
5. **Generation.** Output in RAM, `/dev/shm/fig5-qwen/edit/<kk>/<md5>.png` + `.json`; copy slots 0-9 of the 576 reals
   there first. Split the 576 reals into 4 shards (round-robin, so classes are mixed) and run one process per GPU
   (`CUDA_VISIBLE_DEVICES` / `--gpu`, one job per GPU) for slots 10-149: 80,640 images, ~70 GPU-hours, ~18 h on 4 GPUs.
   If the user wants the 16-per-class panel first, run the 144 reals of `reals-119108-n16.csv` first (~4.5 h).
   Mirror RAM to `/workspace/synderm/fig5-qwen/edit/` every 20 min with `rsync -rt --no-owner --no-group` (the volume
   rejects chmod) and verify the mirror per file (size), not by directory listing (listings can be truncated).
6. **Sweep.** After all slots of a reals set exist (check every file), run
   `fig5_dose_response_sweep.py --seeds 119108 --n 16 32 64 --generations-dir /dev/shm/fig5-qwen --arm edit`
   with a separate `--out` per GPU (e.g. split by `--n`, or by `--doses`), e.g. `/workspace/synderm/fig5-qwen/sweep-n16`.
   Separate output folders avoid concurrent appends to one `results.csv` on the volume. 48 runs, ~6 GPU-hours.

## Shared /workspace: rules

Both pods mount the same volume. This pod writes only to `/workspace/synderm/fig5-qwen/` and
`/workspace/logs/pod-2026-11-01/` (or a new date). Do **not** write to, move or delete anything under
`api_generations/`, `api_sweep/`, `fig5/` (read the reals CSVs only), `qwen_lora/` (read only; a Hugging Face backup on
the other pod is archiving `qwen_lora/generations/` now), `logs/pod-2026-10-31/`, or `backups/`.

## Logging and git

Record every command, setting, timing and decision in a new section of `QWEN_LORA.md` titled
"Figure 5 Qwen edit arm (4-GPU pod)", and the commands in a new `REPRODUCE.md` section. The other pod also appends to
`QWEN_LORA.md`: `git pull --rebase` before every push and keep edits to your own section. Commit to `modernize`
regularly; never merge to `main`, never publish. Stop and report on any failure instead of working around it
(no changes of model, data, classes or settings without the user). Never invent or extrapolate results.
