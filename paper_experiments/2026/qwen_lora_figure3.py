"""Figure 3 panels for the Qwen arms (Step 5 of QWEN_LORA.md): read the sweep results, average over seeds, plot and tabulate.

    python paper_experiments/2026/qwen_lora_figure3.py --sweep /workspace/synderm/qwen_lora/sweep

Reads <sweep>/results.csv (one row per classifier run, written by qwen_lora_sweep.py). For each arm (real only,
+ Qwen edit, + Qwen text-to-image), size n and transform setting, the mean top-1 accuracy over seeds is plotted with
the standard deviation across seeds as the error bar, in the layout of the paper's Figure 3 (one column per synthetic
method, rows = with / without image transforms; the real-only curve is the same in both columns).
Outputs: <sweep>/figure3_qwen.png, <sweep>/figure3_qwen_table.csv (mean, sd, n seeds), <sweep>/figure3_qwen_table.md.
"""
import argparse
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", default="/workspace/synderm/qwen_lora/sweep")
args = ap.parse_args()

r = pd.read_csv(f"{args.sweep}/results.csv")
r["augmentation"] = r["augmentation"].fillna("None")
g = (r.groupby(["generator", "augmentation", "n_training_per_label"])["top1_acc"]
       .agg(mean="mean", sd="std", seeds="count").reset_index())
g.to_csv(f"{args.sweep}/figure3_qwen_table.csv", index=False)

arms = [("qwen-edit", "Qwen-Image-2.1 untuned, varied prompts (edit)"),
        ("qwen-t2i", "Qwen-Image-2.1 + per-class LoRA (text-to-image)")]
settings = [("aug_transforms", "With image transforms", "#5b3f9e"), ("None", "Without image transforms", "#e07b2a")]
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
for j, (arm, title) in enumerate(arms):
    for i, (aug, lab, col) in enumerate(settings):
        ax = axes[i, j]
        for gen, ls, label in [("none", "-", "real only"), (arm, "--", "real + 10 synthetic per real")]:
            d = g[(g.generator == gen) & (g.augmentation == aug)].sort_values("n_training_per_label")
            ax.errorbar(d.n_training_per_label, 100 * d["mean"], yerr=100 * d["sd"].fillna(0), fmt="o", ls=ls,
                        color=col, capsize=3, label=label)
        ax.set_xscale("log"); ax.set_xticks([1, 8, 16, 32, 64, 128, 228]); ax.set_xticklabels(["1", "8", "16", "32", "64", "128", "228"])
        ax.set_ylim(0, 70); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
        if i == 0: ax.set_title(title, fontsize=11)
        if j == 0: ax.set_ylabel(f"{lab}\nMean accuracy (%)")
        if i == 1: ax.set_xlabel("Real training images per disease condition")
seeds = sorted(r.random_state.unique())
fig.suptitle(f"Nine-way skin classification, 360-image held-out test; mean over {len(seeds)} seeds, error bars = SD across seeds", fontsize=11)
fig.tight_layout(); fig.savefig(f"{args.sweep}/figure3_qwen.png", dpi=130)

# markdown table: rows = n, columns = arm x transform setting
lines = ["| n real per class | transforms | real only | + Qwen edit | + Qwen t2i |", "|---|---|---|---|---|"]
for n in sorted(g.n_training_per_label.unique()):
    for aug in ["None", "aug_transforms"]:
        cells = []
        for gen in ["none", "qwen-edit", "qwen-t2i"]:
            d = g[(g.generator == gen) & (g.augmentation == aug) & (g.n_training_per_label == n)]
            cells.append(f"{d['mean'].iloc[0]:.3f} ± {0 if pd.isna(d['sd'].iloc[0]) else d['sd'].iloc[0]:.3f} ({int(d['seeds'].iloc[0])})" if len(d) else "--")
        lines.append(f"| {n} | {aug} | " + " | ".join(cells) + " |")
open(f"{args.sweep}/figure3_qwen_table.md", "w").write("\n".join(lines) + "\n")
print(f"{len(r)} runs, seeds {seeds}"); print("\n".join(lines))
