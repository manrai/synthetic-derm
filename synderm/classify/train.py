"""A short, plain PyTorch training loop around timm backbones.

Mirrors the paper's protocol: ImageNet-pretrained EfficientNetV2-M, up to ``max_epochs``,
early stopping when validation accuracy has not improved for ``patience`` epochs, the best
validation checkpoint kept for evaluation.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .metrics import evaluate

#: The paper's backbone (ImageNet-1k weights) and a modern alternative.
DEFAULT_BACKBONE = "tf_efficientnetv2_m.in1k"
MODERN_BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"


@dataclass
class TrainResult:
    model: nn.Module
    classes: list[str]
    history: list[dict] = field(default_factory=list)
    best_epoch: int = -1
    best_val_accuracy: float = float("nan")
    val_metrics: Optional[dict] = None
    test_metrics: Optional[dict] = None
    seconds: float = 0.0
    config: dict = field(default_factory=dict)


def create_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def train_classifier(
    train_dataset: Dataset,
    val_dataset: Dataset,
    classes: list[str],
    test_dataset: Optional[Dataset] = None,
    backbone: str = DEFAULT_BACKBONE,
    max_epochs: int = 50,
    patience: int = 3,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-2,
    num_workers: int = 4,
    seed: int = 0,
    device: str = "cuda",
    checkpoint_path: Optional[str | Path] = None,
    verbose: bool = True,
) -> TrainResult:
    """Train, keep the best validation epoch, and (optionally) evaluate on a held-out test set.

    ``train_dataset`` and ``val_dataset`` are :class:`~synderm.classify.ClassificationDataset`
    instances built with the same ``classes``.
    """
    torch.manual_seed(seed)
    model = create_model(backbone, len(classes)).to(device)
    pin = device.startswith("cuda")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin, drop_last=len(train_dataset) > batch_size, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=max_epochs * steps_per_epoch,
                                                    pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_state, best_acc, best_epoch, bad_epochs = None, -1.0, -1, 0
    history = []
    t0 = time.time()
    for epoch in range(max_epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y, *_ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=pin):
                loss = loss_fn(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        val = evaluate(model, val_loader, classes, device)
        rec = {"epoch": epoch, "train_loss": running / max(seen, 1), "val_accuracy": val["accuracy"],
               "lr": scheduler.get_last_lr()[0], "seconds": time.time() - t0}
        history.append(rec)
        if verbose:
            print(f"epoch {epoch:02d}  loss {rec['train_loss']:.3f}  val acc {val['accuracy']:.3f}")
        if val["accuracy"] > best_acc:
            best_acc, best_epoch, bad_epochs = val["accuracy"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stop at epoch {epoch} (best {best_epoch}, val acc {best_acc:.3f})")
                break
    model.load_state_dict(best_state)
    result = TrainResult(model=model, classes=list(classes), history=history, best_epoch=best_epoch,
                         best_val_accuracy=best_acc, seconds=time.time() - t0,
                         config={"backbone": backbone, "max_epochs": max_epochs, "patience": patience,
                                 "batch_size": batch_size, "learning_rate": learning_rate, "seed": seed,
                                 "n_train": len(train_dataset), "n_val": len(val_dataset)})
    result.val_metrics = evaluate(model, val_loader, classes, device)
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin)
        result.test_metrics = evaluate(model, test_loader, classes, device)
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "classes": list(classes), "backbone": backbone}, checkpoint_path)
    return result
