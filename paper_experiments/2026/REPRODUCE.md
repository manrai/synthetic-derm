# Reproducing the 2026 experiments

Commands as actually run, in order: sections 0 to 3 on the RTX 5090 pod (2026-10-28), section 4 on the RTX PRO 6000 pod (2026-10-28 to 10-30). Plans: `../../FIGURE3_REGEN_PLAN.md`, `QWEN_LORA.md`.
Everything large lives on `/workspace` (network volume); the root disk is ephemeral.

## 0. Environment (2026-10-28)

```bash
cd /root/synthetic-derm && git checkout modernize
export HF_HOME=/workspace/.cache/huggingface SYNDERM_DATA_DIR=/workspace/data SYNDERM_WORK_DIR=/workspace/synderm
bash bootstrap_pod.sh            # torch 2.8 cu128, diffusers (pinned GitHub commit), synderm -e, 8 CPU tests
python repair_hf_cache.py        # rebuild HF cache symlinks the FUSE mount dropped (now inside bootstrap)
pip install openai google-genai  # API SDKs (now inside bootstrap)
pip install fastai               # paper's classifier stack; keeps torch 2.8 (fastai 2.8.12)
```

Real images: Fitzpatrick 17k `finalfitz17k/` (md5-named jpgs) copied from O2 with rsync to
`/root/data/fitzpatrick17k/finalfitz17k/` (fast local reads) and mirrored with plain `cp` to
`/workspace/data/fitzpatrick17k/finalfitz17k/`. Completeness check against the CSVs:

```bash
python3 -c "
import csv,os; d='/root/data/fitzpatrick17k/finalfitz17k'
for f in ['fitz_metadata/fitzpatrick17k_10label_clean_training.csv','fitz_metadata/fitzpatrick17k_10label_clean_held_out_set.csv']:
    rows=list(csv.DictReader(open(f))); print(f, 'missing', sum(not os.path.exists(f'{d}/{r[\"md5hash\"]}.jpg') for r in rows), 'of', len(rows))"
```

API keys live in the gitignored `.env` at the repo root (`OPENAI_API_KEY=...`, `GEMINI_API_KEY=...`);
the notebook and scripts read it themselves.

## 0.5 Zero-shot screen (Step 0.5)

```bash
jupyter lab paper_experiments/2026/00_zero_shot_screen.ipynb   # outputs: /workspace/synderm/zero_shot_screen/
```

Run with `API_MODELS = ["gpt-image-2.5-sunburst", "nano-banana-2.1"]`, `N_EDIT = 4`, Qwen zero-shot
at 768 px. Grids per class in `grids/`, `summary.csv`, `records.csv`.

## 1. Pilot generation (Step 1): reference-image edits, one seed, n up to 16

Seed 20261028. Sampling reproduces `paper_experiments/Experiments/n_per_label.py`
(40 validation per label, then n per label; subsets nest). Dry run writes the split CSVs only.

```bash
python paper_experiments/2026/pilot_generate.py --dry-run
# small live tests (one class, 2 slots per real) before the full runs
python paper_experiments/2026/pilot_generate.py --model gpt-image-2.5-sunburst --labels psoriasis --k 2 --threads 8 --budget 2
python paper_experiments/2026/pilot_generate.py --model nano-banana-2.1      --labels psoriasis --k 2 --threads 8 --budget 3
# full runs, both providers in parallel (2026-10-28, ~15:40)
python paper_experiments/2026/pilot_generate.py --model gpt-image-2.5-sunburst --threads 20 --budget 60 > /workspace/logs/pilot-gpt-image.log 2>&1 &
python paper_experiments/2026/pilot_generate.py --model nano-banana-2.1      --threads 20 --budget 80 > /workspace/logs/pilot-nano-banana.log 2>&1 &
```

Outputs: `/workspace/synderm/pilot/<model>-<quality>-seed20261028-n16/` with `<label>/edit/<kk>/<md5>.png`
plus JSON sidecars, `reals.csv`, `val.csv`, `test.csv`, `synthetic.csv`, `progress.json`, `generate.log`.
Re-running the same command resumes (finished and twice-refused slots are skipped).

Small-test measurements: GPT Image 2.5 Sunburst medium $0.0188 per edit (439 output image tokens,
~670 input image tokens), 21 s; 2 of 16 psoriasis references refused on both attempts, filled by
top-up. Nano Banana 2.1 1K $0.0336 per edit, 19 s, no refusals, no 429s at 8 threads.

`pilot_generate.py` is self-contained (no imports from the screen module); it was rewritten in that
form after the full runs started, with the same on-disk layout, so re-running it resumes them.

Both single-prompt runs were stopped at about 40% (GPT 570 ok / 10 refused / $10.78; Nano Banana
593 ok / 1 refused / $19.92) to check within-reference variability first (screen sections 6 and 6b).
Ten repeats of one prompt looked nearly identical; ten varied prompts did not. The pilot therefore
switched to `--prompts varied` (slot k uses variation k of the ten-prompt list in the script), which
writes to a separate `-varied` folder; the single-prompt folders stay for a later comparison.

```bash
python paper_experiments/2026/pilot_generate.py --prompts varied --dry-run   # same reals.csv as before (checked with cmp)
# varied-prompt runs, both providers in parallel (2026-10-28, 17:03)
python paper_experiments/2026/pilot_generate.py --model gpt-image-2.5-sunburst --prompts varied --threads 20 --budget 60 > /workspace/logs/pilot-gpt-image-varied.log 2>&1 &
python paper_experiments/2026/pilot_generate.py --model nano-banana-2.1      --prompts varied --threads 20 --budget 80 > /workspace/logs/pilot-nano-banana-varied.log 2>&1 &
```

Outputs: `/workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/` and
`/workspace/synderm/pilot/nano-banana-2.1-1K-varied-seed20261028-n16/`; each JSON sidecar records
`prompt_mode`, `variation` and the full prompt.

The synthetic classifier arms (Section 2 below) were queued to start automatically when both generation
runs exit: `/workspace/logs/run_classifier_after_pilot.sh`, log `/workspace/logs/pilot-classifier-varied.log`.

## 2. Pilot classifier (the paper's fastai protocol, results to CSV)

```bash
# real-only arms once (they do not depend on a generator); test run first
python paper_experiments/2026/pilot_classifier.py --pilot-dir /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-seed20261028-n16 --generator none --n 1 --augmentation None --out /workspace/synderm/pilot/results_test
python paper_experiments/2026/pilot_classifier.py --pilot-dir /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-seed20261028-n16 --generator none --n 1 8 16 --augmentation None aug_transforms --out /workspace/synderm/pilot/results
# synthetic arms, one call per generator, after its generation run has written synthetic.csv
# (the real-only rows were computed from the single-prompt folder; the split is identical, so they carry over)
python paper_experiments/2026/pilot_classifier.py --pilot-dir /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16 --generator gpt-image-2.5-sunburst-varied --n 1 8 16 --out /workspace/synderm/pilot/results
python paper_experiments/2026/pilot_classifier.py --pilot-dir /workspace/synderm/pilot/nano-banana-2.1-1K-varied-seed20261028-n16      --generator nano-banana-2.1-varied      --n 1 8 16 --out /workspace/synderm/pilot/results
```

Each run appends one row to `/workspace/synderm/pilot/results/results.csv` (run_id, generator, n, augmentation,
top-1 accuracy with Wald CI, top-3, epochs, timing) and writes `predictions__<run_id>.csv`. Finished run_ids are skipped.

Varied-prompt results (2026-10-28). Generation: GPT Image 2.5 Sunburst 1,440 ok, 97 refused and topped up,
$27.31, done 17:37; Nano Banana 2.1 1,440 ok, 7 refused and topped up, $48.38, done 17:26. All nine classes
reached 160 synthetic images on both providers. Classifier chain finished 18:02 (about 2 min per run).
Top-1 accuracy on the 360-image held-out set, one seed:

| n per class | real only | + GPT | + Nano Banana | real + transforms | + GPT + transforms | + Nano Banana + transforms |
|---|---|---|---|---|---|---|
| 1  | 14.4 | 17.8 | 15.6 | 16.1 | 17.5 | 15.6 |
| 8  | 15.8 | 22.2 | 21.1 | 18.6 | 22.2 | 24.2 |
| 16 | 16.1 | 21.1 | 25.0 | 16.4 | 25.3 | 23.1 |

## 3. Analysis

```bash
jupyter lab paper_experiments/2026/01_pilot_analysis.ipynb   # reads results.csv, writes figure3_pilot.png
jupyter lab paper_experiments/2026/02_pilot_preview.ipynb    # grids of real reference + 10 synthetic slots from the pilot folders
```

## 4. Qwen-Image-2.1 LoRA arms for Figure 3 (2026-10-28 to 2026-10-30, RTX PRO 6000 pod)

Plan, decisions and the timed run log: `QWEN_LORA.md`. Every command below was run from the repo root on
branch `modernize` (commits 80e9c88 to 3621f0e; each LoRA's `train_config.json` records the commit, the full
`accelerate` command and the training time). Two 96 GB GPUs were visible; one job per GPU via `--gpu`
(the scripts set `CUDA_VISIBLE_DEVICES`). Nothing was read from `/workspace` at run time except CSVs:
models were loaded from a copy in RAM and bulk outputs written to RAM, then mirrored.

### 4.0 Environment

```bash
cd /root/synthetic-derm && git checkout modernize
HF_HOME=/workspace/.cache/huggingface bash bootstrap_pod.sh      # torch 2.8.0+cu128, diffusers 0c7d56d6, peft 0.21.2, accelerate 1.15.0
pip install fastai scikit-image                                   # fastai 2.8.12 (classifier), scikit-image 0.26.0 (ITA in the eval script)
# model weights into RAM (263 GB /dev/shm); the volume cache has the same files but reads at ~8 MB/s
export HF_HOME=/dev/shm/hf
hf download Qwen/Qwen-Image-2.1 --revision d26bb612                # 32 GB, under a minute from the Hub
python -c "import timm; [timm.create_model(m, pretrained=True) for m in ('vit_small_patch14_dinov2.lvd142m', 'tf_efficientnetv2_m.in1k')]"
# real images to local disk (24-way parallel cp from /workspace/data/fitzpatrick17k; plain cp runs at ~7 files/s on the volume)
mkdir -p /root/data/fitzpatrick17k/finalfitz17k
ls /workspace/data/fitzpatrick17k/finalfitz17k | xargs -P 24 -I{} cp /workspace/data/fitzpatrick17k/finalfitz17k/{} /root/data/fitzpatrick17k/finalfitz17k/
# then the md5 completeness check from section 0 (all 3,530 training and 400 held-out images present)
```

The vendored trainers `synderm/fine_tune/scripts/train_dreambooth_lora_qwenimage21{,_img2img}.py` are diffusers
`examples/dreambooth` at commit 0c7d56d6 plus the 11-line `--offload` patch described in that folder's README.

### 4.1 Smoke tests (psoriasis, 20 steps; outputs under `/workspace/synderm/qwen_lora/smoke/`, not results)

```bash
python paper_experiments/2026/qwen_lora_train.py --label psoriasis --method t2i  --gpu 0 --steps 20 --checkpoint-every 10 --run-name smoke-20steps-b
python paper_experiments/2026/qwen_lora_train.py --label psoriasis --method edit --gpu 1 --steps 20 --checkpoint-every 10 --run-name smoke-20steps-b
# generate 2 reals x 10 slots with each smoke LoRA, then score both folders
python paper_experiments/2026/qwen_lora_generate.py --method t2i  --gpu 0 --reals /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/reals.csv --label psoriasis --lora /workspace/synderm/qwen_lora/t2i/psoriasis/smoke-20steps-b/pytorch_lora_weights.safetensors  --out /workspace/synderm/qwen_lora/smoke/gen-t2i-smoke20
python paper_experiments/2026/qwen_lora_generate.py --method edit --gpu 1 --reals /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/reals.csv --label psoriasis --lora /workspace/synderm/qwen_lora/edit/psoriasis/smoke-20steps-b/pytorch_lora_weights.safetensors --out /workspace/synderm/qwen_lora/smoke/gen-edit-smoke20
python paper_experiments/2026/qwen_lora_eval.py --gpu 1 --label psoriasis --folders /workspace/synderm/qwen_lora/smoke/gen-t2i-smoke20 /workspace/synderm/qwen_lora/smoke/gen-edit-smoke20 --out /workspace/synderm/qwen_lora/smoke/eval
```

### 4.2 Step 1: psoriasis pilot (LoRA settings and checkpoint choice)

Training settings used for every LoRA in this round, all defaults of `qwen_lora_train.py`: rank 16, alpha 16,
lr 1e-4 constant, 512 px, batch 1 x gradient accumulation 4, 2,000 steps, checkpoints every 500, seed 0, bf16,
cached latents, gradient checkpointing, random flip; run name `r16-lr0.0001-512px-2000steps`. Training data =
every training-pool image of the label (`fitz_metadata/fitzpatrick17k_10label_clean_training.csv`), caption
"An image of {label}, a skin disease" (t2i) or the fixed edit prompt in the script with reference = another
image of the same label and Fitzpatrick type, 2 pairs per image (edit).

```bash
python paper_experiments/2026/qwen_lora_train.py --label psoriasis --method t2i  --gpu 0 --steps 2000 --checkpoint-every 500   # 32.9 min
python paper_experiments/2026/qwen_lora_train.py --label psoriasis --method edit --gpu 1 --steps 2000 --checkpoint-every 500   # 60.6 min
# Stage A pilot generation: untuned Qwen, then checkpoints 500/1000/1500/2000, 16 reals x 10 seeds, one prompt
bash paper_experiments/2026/qwen_lora_pilot.sh generate t2i  0 psoriasis r16-lr0.0001-512px-2000steps
bash paper_experiments/2026/qwen_lora_pilot.sh generate edit 1 psoriasis r16-lr0.0001-512px-2000steps
# the GPT Image pilot folder is scored first and the "lacks variety" cut-off written into QWEN_LORA.md before any Qwen folder
python paper_experiments/2026/qwen_lora_eval.py --gpu 0 --label psoriasis --folders /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16 --out /workspace/synderm/qwen_lora/pilot/eval-psoriasis
bash paper_experiments/2026/qwen_lora_pilot.sh eval 0 psoriasis          # all ten Qwen folders + GPT folder -> pilot/eval-psoriasis/metrics.csv
# Stage B for edit (untuned Qwen, the 10 varied prompts), after Stage A edit failed the cut-off
python paper_experiments/2026/qwen_lora_generate.py --method edit --gpu 0 --reals /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/reals.csv --label psoriasis --prompts varied --out /workspace/synderm/qwen_lora/pilot/edit-psoriasis-untuned-varied
python paper_experiments/2026/qwen_lora_eval.py --gpu 0 --label psoriasis --folders /workspace/synderm/qwen_lora/pilot/edit-psoriasis-untuned-varied --out /workspace/synderm/qwen_lora/pilot/eval-psoriasis
```

Decision (QWEN_LORA.md, 23:43 UTC Oct 28): t2i arm = per-class LoRA checkpoint 1500, one prompt, seed variety;
edit arm = untuned Qwen with the 10 varied prompts (every edit LoRA checkpoint ignores the reference). No edit
LoRA is used; only the psoriasis one was trained.

### 4.3 Step 2: confirm on basal cell carcinoma and lupus erythematosus

```bash
python paper_experiments/2026/qwen_lora_train.py --label "basal cell carcinoma" --method t2i --gpu 0 --steps 2000 --checkpoint-every 500   # 32.7 min
python paper_experiments/2026/qwen_lora_train.py --label "lupus erythematosus"  --method t2i --gpu 1 --steps 2000 --checkpoint-every 500   # 31.3 min
R=/workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/reals.csv; P=/workspace/synderm/qwen_lora/pilot
for L in "basal cell carcinoma" "lupus erythematosus"; do S=${L// /-}
  python paper_experiments/2026/qwen_lora_generate.py --method t2i  --gpu 0 --reals $R --label "$L" --prompts single --out $P/t2i-$S-untuned-single
  python paper_experiments/2026/qwen_lora_generate.py --method t2i  --gpu 0 --reals $R --label "$L" --prompts single --lora /workspace/synderm/qwen_lora/t2i/$S/r16-lr0.0001-512px-2000steps/checkpoint-1500/pytorch_lora_weights.safetensors --out $P/t2i-$S-ckpt1500-single
  python paper_experiments/2026/qwen_lora_generate.py --method edit --gpu 0 --reals $R --label "$L" --prompts varied --out $P/edit-$S-untuned-varied
  python paper_experiments/2026/qwen_lora_eval.py --gpu 1 --label "$L" --folders $(dirname $R) $P/t2i-$S-untuned-single $P/t2i-$S-ckpt1500-single $P/edit-$S-untuned-varied --out $P/eval-$S
done
```

(As run, the two classes were split across the GPUs and chained behind their trainings; the folders and
settings are the same.) Decision held, no change of settings.

### 4.4 Steps 3 and 4: the remaining six LoRAs and the full generation

```bash
# GPU 0: six remaining t2i LoRAs in sequence (~33 min each), then the t2i generation for all nine classes
bash paper_experiments/2026/qwen_lora_full_run.sh train-t2i 0 "lichen planus" "allergic contact dermatitis" "neutrophilic dermatoses" "photodermatoses" "sarcoidosis" "squamous cell carcinoma" \
  && bash paper_experiments/2026/qwen_lora_full_run.sh gen-t2i 0     # 33,390 images, ckpt1500 LoRA per class, one prompt, seeds int(md5[:8],16)+slot
# GPU 1: edit generation, untuned Qwen, 10 varied prompts, reference = the real image
bash paper_experiments/2026/qwen_lora_full_run.sh gen-edit 1          # 33,390 images
# mirror RAM output (/dev/shm/qwen_lora_generations) to the volume while the runs go, 20 rsync streams every 20 min
bash paper_experiments/2026/qwen_lora_sync.sh 20 &
```

All nine t2i LoRAs (training-pool images, minutes on one RTX PRO 6000; `train_config.json` in each folder):

| label | images | min | label | images | min | label | images | min |
|---|---|---|---|---|---|---|---|---|
| psoriasis | 584 | 32.9 | basal cell carcinoma | 406 | 32.7 | lupus erythematosus | 297 | 31.3 |
| lichen planus | 412 | 32.8 | allergic contact dermatitis | 358 | 32.4 | neutrophilic dermatoses | 284 | 32.8 |
| photodermatoses | 272 | 32.8 | sarcoidosis | 286 | 32.6 | squamous cell carcinoma | 440 | 32.7 |

Weights: `/workspace/synderm/qwen_lora/t2i/<label-slug>/r16-lr0.0001-512px-2000steps/checkpoint-{500,1000,1500,2000}/`
plus final `pytorch_lora_weights.safetensors`; the psoriasis edit LoRA under `edit/psoriasis/` (584 images, 1,168 pairs, 60.6 min).
Generation ran at 2.9 s (t2i) and 3.2 s (edit) per image at 512 px, 40 sampling steps, 10 prompts batched per real,
about 27 and 30 h on one GPU each; `qwen_lora_generate.py` skips existing files, so an interrupted run resumes.

Early single-seed sweep (requested Oct 29, same images, only the order changed): the 144 reals of seed 2026001's
n=16 subset were generated first, then the full runs resumed.

```bash
bash paper_experiments/2026/qwen_lora_priority.sh build-csv 2026001
bash paper_experiments/2026/qwen_lora_priority.sh gen-t2i  0 2026001 && bash paper_experiments/2026/qwen_lora_full_run.sh gen-t2i 0
bash paper_experiments/2026/qwen_lora_priority.sh gen-edit 1 2026001 && bash paper_experiments/2026/qwen_lora_full_run.sh gen-edit 1
python paper_experiments/2026/qwen_lora_sweep.py --gpu 1 --seeds 2026001 --n 1 8 16 --out /workspace/synderm/qwen_lora/sweep   # 18 runs, ~8 min
```

### 4.5 Step 5: Figure 3 sweep and figure

```bash
python paper_experiments/2026/qwen_lora_sweep.py --gpu 1 --seeds 2026001 --n 1 8 --arms none --out /workspace/synderm/qwen_lora/sweep-smoke   # smoke, 4 runs
# the sweep itself: 7 sizes x 3 arms x 2 transform settings x 5 seeds = 210 runs; finished run ids are skipped
python paper_experiments/2026/qwen_lora_sweep.py --gpu 1 --seeds 2026001 2026002 2026003 2026004 2026005 --arms none edit --out /workspace/synderm/qwen_lora/sweep   # 140 runs, 3 h 3 min
python paper_experiments/2026/qwen_lora_sweep.py --gpu 0 --seeds 2026001 2026002 2026003 2026004 2026005 --arms t2i       --out /workspace/synderm/qwen_lora/sweep   #  70 runs, 2 h 51 min
python paper_experiments/2026/qwen_lora_figure3.py --sweep /workspace/synderm/qwen_lora/sweep     # figure3_qwen.png, figure3_qwen_table.{csv,md}
python paper_experiments/2026/slides/build_figure3_compare.py                                    # four-arm comparison at n <= 16 for the slides
```

The sweep reads synthetic images from `--generations-dir` (default `/dev/shm/qwen_lora_generations`); on a new pod
copy `/workspace/synderm/qwen_lora/generations/` to RAM or local disk first, or pass that path. Results:
`/workspace/synderm/qwen_lora/sweep/results.csv` (one row per run) and `predictions__<run_id>.csv`.

### 4.6 Outputs and backup

- `/workspace/synderm/qwen_lora/generations/{t2i,edit}/{00..09}/<md5hash>.png` + `.json` (prompt, seed, LoRA, timing),
  `synthetic.csv` per method, `pool9.csv` (the 3,339 pool rows). 66,780 images, 28 GB, verified file by file against the RAM copy.
- `pilot/`: every pilot folder and `eval-<class>/metrics.csv` + grids; `sweep/`: results, figure, table; `smoke/`, `sweep-smoke/`.
- Logs of every job: `/workspace/logs/pod-2026-10-28b/`. Preview notebook: `03_qwen_lora_preview.ipynb`
  (executed copy `/workspace/synderm/qwen_lora/03_qwen_lora_preview.executed.ipynb`).
- $0 of paid API used in this round.
