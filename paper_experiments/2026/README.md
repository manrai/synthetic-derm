# 2026 experiments: regenerating Figure 3 with modern generators

Plan: `../../FIGURE3_REGEN_PLAN.md`. Code for the 2023 paper stays untouched one level up.

| File | Step | What it does |
|---|---|---|
| `zero_shot_screen.py` | 0.5 | Helpers: model table, job list, disk cache, OpenAI / Google / local generation, per-class grids, summary |
| `00_zero_shot_screen.ipynb` | 0.5 | Runs the screen for nine classes across OpenAI, Google and untuned Qwen; writes grids and `summary.csv` to `/workspace/synderm/zero_shot_screen/`; sections 6/6b compare 10 repeated edits against 10 varied-prompt edits of the same reference |
| `pilot_generate.py` | 1 | Self-contained: paper's split for one seed, reference-image edits via OpenAI or Google, `--prompts single|varied`, refusal retry and top-up, writes `synthetic.csv` |
| `pilot_classifier.py` | 1 | The paper's fastai EfficientNetV2-M protocol on the pilot split; appends to `results.csv` |
| `01_pilot_analysis.ipynb` | 1 | Linear pandas: pivot of accuracies, gain vs real-only, Figure 3-style plot |
| `02_pilot_preview.ipynb` | 1 | Hardcoded paths to the pilot folders; shows each real reference next to its ten synthetic slots, per model and class |
| `REPRODUCE.md` | all | Every command run, in order, with measurements |

Run from the repo root or from this folder:

```bash
export OPENAI_API_KEY=...  GEMINI_API_KEY=...
export HF_HOME=/workspace/.cache/huggingface
jupyter lab paper_experiments/2026/00_zero_shot_screen.ipynb
```

Set `LABELS = ["psoriasis"]` in the first code cell for a quick test. Everything is cached per
image, so re-running fills gaps and retries errors without regenerating what exists.
