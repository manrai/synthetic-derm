#!/usr/bin/env python3
"""Pilot classifier sweep (plan Step 1): the paper's ``n_per_label.py`` protocol on pilot outputs.

Same model and training as ``paper_experiments/Experiments/n_per_label.py``: fastai
``vision_learner`` with EfficientNetV2-M, 224 px, ``fit(50)`` with early stopping on validation
loss (patience 3), batch size 8/32/64 by training-set size, ``aug_transforms()`` or none,
top-1 accuracy on the 360-image held-out set with a Wald 95% CI. Differences: results go to
CSV instead of wandb, the real subsets come from ``reals.csv`` (written by ``pilot_generate.py``
with the paper's sampling), and torch seeds are fixed per run for reproducibility.

    python pilot_classifier.py --pilot-dir /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-seed20261028-n16 \
        --generator gpt-image-2.5-sunburst --n 1 8 16 --out /workspace/synderm/pilot/results

Real-only arms do not depend on a generator; run them once with ``--generator none``.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import pandas as pd
import torch
from fastai.vision.all import (ImageDataLoaders, Resize, aug_transforms, vision_learner, efficientnet_v2_m,
                               error_rate, accuracy, EarlyStoppingCallback, set_seed)
from statsmodels.stats.proportion import proportion_confint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", required=True, help="output folder of pilot_generate.py (has reals.csv, val.csv, test.csv, synthetic.csv)")
    ap.add_argument("--generator", required=True, help="name recorded in results; 'none' runs the real-only arms")
    ap.add_argument("--n", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--augmentation", nargs="+", default=["None", "aug_transforms"])
    ap.add_argument("--k", type=int, default=10, help="synthetic per real")
    ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
    ap.add_argument("--out", default="/workspace/synderm/pilot/results")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261028)
    a = ap.parse_args()

    pilot = Path(a.pilot_dir); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    reals = pd.read_csv(pilot / "reals.csv"); val = pd.read_csv(pilot / "val.csv"); test = pd.read_csv(pilot / "test.csv")
    include_synthetic = a.generator != "none"
    syn = pd.read_csv(pilot / "synthetic.csv") if include_synthetic else None
    loc = lambda df: df["md5hash"].map(lambda h: f"{a.images_dir}/{h}.jpg")
    val = val.assign(location=loc(val), synthetic=False, is_valid=True)
    test = test.assign(location=loc(test), synthetic=False, is_valid=False)
    augmentation_dict = {"None": [], "aug_transforms": aug_transforms()}
    results_csv = out / "results.csv"

    for n in a.n:
        for augmentation in a.augmentation:
            run_id = f"{a.generator}__n{n}__{augmentation}__seed{a.seed}"
            if results_csv.exists() and run_id in set(pd.read_csv(results_csv)["run_id"]):
                print(f"skip {run_id} (done)"); continue
            # the paper's get_n_per_label_data: n real per label, plus their synthetic images
            sel = reals[reals[f"in_n{n}"]].copy()
            train = sel.assign(location=loc(sel), synthetic=False)[["md5hash", "label", "location", "synthetic"]]
            if include_synthetic:
                s = syn[syn["md5hash"].isin(sel["md5hash"])]
                assert len(s) == a.k * len(sel), f"expected {a.k * len(sel)} synthetic rows for n={n}, got {len(s)}"
                train = pd.concat([train, s.assign(synthetic=True)[["md5hash", "label", "location", "synthetic"]]])
            train = train.assign(is_valid=False)
            df = pd.concat([train, val[train.columns]], ignore_index=True)
            n_train = int((~df.is_valid).sum())
            batch_size = 64 if n_train / 10 >= 100 else 32 if n_train / 10 >= 10 else 8
            set_seed(a.seed, reproducible=True)
            dls = ImageDataLoaders.from_df(df, "/", fn_col="location", label_col="label", valid_col="is_valid",
                                           bs=batch_size, item_tfms=Resize(224), batch_tfms=augmentation_dict[augmentation])
            learn = vision_learner(dls, arch=efficientnet_v2_m, metrics=[error_rate, accuracy])
            t = time.time()
            learn.fit(a.epochs, cbs=[EarlyStoppingCallback(monitor="valid_loss", min_delta=0.0, patience=3)])
            epochs_run = len(learn.recorder.values)
            preds, _ = learn.get_preds(dl=dls.test_dl(test))
            top1 = [learn.dls.vocab[i] for i in preds.argmax(1)]
            top3 = preds.topk(3, dim=1).indices
            correct = (pd.Series(top1) == test["label"].reset_index(drop=True))
            acc = float(correct.mean()); m = len(test)
            lo, hi = proportion_confint(int(correct.sum()), m, alpha=0.05, method="normal")
            top3_acc = float(sum(test["label"].iloc[i] in [learn.dls.vocab[j] for j in top3[i]] for i in range(m)) / m)
            row = dict(run_id=run_id, generator=a.generator, include_synthetic=include_synthetic, n_training_per_label=n,
                       augmentation=augmentation, n_synthetic_per_real=a.k if include_synthetic else 0, seed=a.seed,
                       n_train=n_train, batch_size=batch_size, epochs_run=epochs_run, train_s=round(time.time() - t),
                       top1_acc=acc, top1_ci_lower=lo, top1_ci_upper=hi, top3_acc=top3_acc, n_test=m,
                       architecture="EfficientNet-V2-M", pilot_dir=str(pilot))
            pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not results_csv.exists(), index=False)
            pd.DataFrame({"run_id": run_id, "md5hash": test["md5hash"], "true_label": test["label"], "top1_label": top1,
                          "top1_prob": preds.max(1).values.numpy()}).to_csv(out / f"predictions__{run_id}.csv", index=False)
            print(f"{run_id}: n_train={n_train} bs={batch_size} epochs={epochs_run} top1={acc:.3f} [{lo:.3f}, {hi:.3f}] top3={top3_acc:.3f} ({time.time() - t:.0f}s)", flush=True)
            del learn, dls; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
