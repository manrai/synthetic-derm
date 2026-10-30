"""Figure 3 at n <= 16 for all four synthetic arms, side by side (slide "All four arms at n <= 16").

    python paper_experiments/2026/slides/build_figure3_compare.py

Proprietary panels: the one-seed pilot sweep (/workspace/synderm/pilot/results/results.csv), error bars = Wald 95% CI
over the 360 test images. Qwen panels: the five-seed sweep (/workspace/synderm/qwen_lora/sweep/results.csv), error
bars = SD across seeds. Writes fig/figure3_compare.png next to this script.
"""
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

pilot = pd.read_csv("/workspace/synderm/pilot/results/results.csv")
sweep = pd.read_csv("/workspace/synderm/qwen_lora/sweep/results.csv")
for d in (pilot, sweep):
    d["augmentation"] = d["augmentation"].fillna("None")
g = (sweep.groupby(["generator", "augmentation", "n_training_per_label"])["top1_acc"]
          .agg(mean="mean", sd="std").reset_index())

panels = [("GPT Image 2.5 Sunburst\nzero-shot edit, 1 seed", "pilot", "gpt-image-2.5-sunburst-varied"),
          ("Nano Banana 2.1\nzero-shot edit, 1 seed", "pilot", "nano-banana-2.1-varied"),
          ("Qwen-Image-2.1 untuned\nvaried-prompt edit, 5 seeds", "sweep", "qwen-edit"),
          ("Qwen-Image-2.1 + per-class LoRA\ntext-to-image, 5 seeds", "sweep", "qwen-t2i")]
settings = [("aug_transforms", "With image transforms", "#5b3f9e"), ("None", "Without image transforms", "#e07b2a")]
fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), sharey=True)
for j, (title, src, gen) in enumerate(panels):
    for i, (aug, lab, col) in enumerate(settings):
        ax = axes[i, j]
        for gname, ls, label in [("none", "-", "real only"), (gen, "--", "real + 10 synthetic per real")]:
            if src == "pilot":
                d = pilot[(pilot.generator == gname) & (pilot.augmentation == aug) & (pilot.n_training_per_label <= 16)].sort_values("n_training_per_label")
                y, err = 100 * d.top1_acc, [100 * (d.top1_acc - d.top1_ci_lower), 100 * (d.top1_ci_upper - d.top1_acc)]
            else:
                d = g[(g.generator == gname) & (g.augmentation == aug) & (g.n_training_per_label <= 16)].sort_values("n_training_per_label")
                y, err = 100 * d["mean"], 100 * d["sd"]
            ax.errorbar(d.n_training_per_label, y, yerr=err, fmt="o", ls=ls, color=col, capsize=3, label=label)
        ax.set_xscale("log"); ax.set_xticks([1, 8, 16]); ax.set_xticklabels(["1", "8", "16"]); ax.set_ylim(0, 50)
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
        if i == 0: ax.set_title(title, fontsize=11)
        if j == 0: ax.set_ylabel(f"{lab}\nMean accuracy (%)")
        if i == 1: ax.set_xlabel("Real training images per disease condition")
fig.tight_layout()
out = Path(__file__).resolve().parent / "fig" / "figure3_compare.png"
fig.savefig(out, dpi=130); print(f"wrote {out}")
