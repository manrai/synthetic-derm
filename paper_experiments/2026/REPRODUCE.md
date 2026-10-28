# Reproducing the 2026 experiments

Commands as actually run on the RTX 5090 pod, in order. Plan: `../../FIGURE3_REGEN_PLAN.md`.
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

## 3. Analysis

```bash
jupyter lab paper_experiments/2026/01_pilot_analysis.ipynb   # reads results.csv, writes figure3_pilot.png
```
