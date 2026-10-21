from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


def accuracy_ci(correct: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Accuracy with a Wald confidence interval, as in the paper's figures."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = correct / total
    half = z * math.sqrt(p * (1 - p) / total)
    return p, max(0.0, p - half), min(1.0, p + half)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: str = "cuda") -> tuple[np.ndarray, np.ndarray, list]:
    model.eval()
    preds, targets, groups = [], [], []
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            logits = model(x)
        preds.append(logits.argmax(1).cpu().numpy())
        targets.append(np.asarray(y))
        if len(batch) > 2:
            groups.extend(list(batch[2].tolist() if torch.is_tensor(batch[2]) else batch[2]))
    return np.concatenate(preds), np.concatenate(targets), groups


def evaluate(model: torch.nn.Module, loader: DataLoader, classes: Sequence[str], device: str = "cuda") -> dict:
    """Overall and per-class accuracy with Wald CIs, plus the confusion matrix."""
    preds, targets, groups = predict(model, loader, device)
    n = len(targets)
    correct = int((preds == targets).sum())
    acc, lo, hi = accuracy_ci(correct, n)
    per_class = {}
    for i, c in enumerate(classes):
        mask = targets == i
        if mask.any():
            per_class[c] = dict(zip(("accuracy", "ci_low", "ci_high"), accuracy_ci(int((preds[mask] == i).sum()), int(mask.sum()))))
            per_class[c]["n"] = int(mask.sum())
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for p, t in zip(preds, targets):
        cm[t, p] += 1
    out = {"n": n, "correct": correct, "accuracy": acc, "ci_low": lo, "ci_high": hi,
           "per_class": per_class, "confusion_matrix": cm, "preds": preds, "targets": targets}
    if groups:
        out["groups"] = groups
        out["per_group"] = per_group_accuracy(preds, targets, groups)
    return out


def per_group_accuracy(preds: np.ndarray, targets: np.ndarray, groups: Sequence) -> dict:
    """Accuracy with CI per group value (e.g. Fitzpatrick skin type)."""
    by = defaultdict(lambda: [0, 0])
    for p, t, g in zip(preds, targets, groups):
        by[g][0] += int(p == t)
        by[g][1] += 1
    return {g: {"n": tot, **dict(zip(("accuracy", "ci_low", "ci_high"), accuracy_ci(c, tot)))}
            for g, (c, tot) in sorted(by.items(), key=lambda kv: str(kv[0]))}
