"""Figure 3 sweep (Step 5 of QWEN_LORA.md): the paper's n_per_label.py protocol on the Qwen generation folders.

    python paper_experiments/2026/qwen_lora_sweep.py --gpu 0 --seeds 2026001 2026002 --out /workspace/synderm/qwen_lora/sweep
    python paper_experiments/2026/qwen_lora_sweep.py --gpu 1 --seeds 2026003 2026004 2026005 --out /workspace/synderm/qwen_lora/sweep

Same protocol as paper_experiments/Experiments/n_per_label.py (read it first), with results in CSV instead of wandb:
  data      training CSV, nine labels (folliculitis dropped); held-out CSV = test (360); validation = 40 real per label
            sampled from the pool with random_state; classifier train = n real per label sampled from the rest with the
            same random_state, plus, for the synthetic arms, the 10 synthetic images filed under each selected real.
  model     fastai vision_learner, EfficientNetV2-M, 224 px, fit(50) with early stopping on valid_loss (patience 3),
            batch size 64 / 32 / 8 when n_train/10 is >= 100 / >= 10 / else, aug_transforms() or none.
  score     top-1 accuracy on the 360 held-out images with a Wald 95% CI, top-3 accuracy; per-image predictions saved.
  grid      n in {1, 8, 16, 32, 64, 128, 228} x arms {real only, + Qwen edit, + Qwen t2i} x {None, aug_transforms}
            x seeds = 42 runs per seed. Every run id is skipped if already in results.csv (resume).
Synthetic images: --generations-dir/<arm>/<00..09>/<md5hash>.png. Missing files are an error, never skipped.
Real images are read from --images-dir; synthetic ones from --generations-dir (copy both to local disk or RAM first).
"""
import argparse, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, required=True)
ap.add_argument("--seeds", type=int, nargs="+", required=True, help="random_state values (sampling and torch seed)")
ap.add_argument("--n", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128, 228])
ap.add_argument("--arms", nargs="+", default=["none", "edit", "t2i"], help="none = real only; others are generation folders")
ap.add_argument("--augmentation", nargs="+", default=["None", "aug_transforms"])
ap.add_argument("--k", type=int, default=10, help="synthetic images per real")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
ap.add_argument("--generations-dir", default="/dev/shm/qwen_lora_generations")
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import pandas as pd
import torch
from fastai.vision.all import (ImageDataLoaders, Resize, aug_transforms, vision_learner, efficientnet_v2_m,
                               error_rate, accuracy, EarlyStoppingCallback, set_seed)
from statsmodels.stats.proportion import proportion_confint

REPO = Path(__file__).resolve().parents[2]
META = REPO / "fitz_metadata"
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
results_csv = out / "results.csv"

# ---- data split, as in n_per_label.py --------------------------------------------------
metadata = pd.read_csv(META / "fitzpatrick17k_10label_clean_training.csv")
nine = metadata["label"].value_counts().index[:9]
metadata = metadata[metadata["label"].isin(nine)].reset_index(drop=True)
test = pd.read_csv(META / "fitzpatrick17k_10label_clean_held_out_set.csv")
test = test[test["label"].isin(nine)].reset_index(drop=True)
pool = metadata[~metadata["md5hash"].isin(test["md5hash"])].reset_index(drop=True)
real_loc = lambda df: df["md5hash"].map(lambda h: f"{args.images_dir}/{h}.jpg")
test = test.assign(location=real_loc(test), synthetic=False, is_valid=False)
print(f"pool {len(pool)} real images over {pool['label'].nunique()} labels; test {len(test)}", flush=True)

def sample_per_label(df, n, random_state):
    # n_per_label.py: df.groupby('label').apply(lambda x: x.sample(n=n, random_state=random_state)); groups in sorted label order
    return pd.concat([df[df["label"] == lab].sample(n=n, random_state=random_state) for lab in sorted(df["label"].unique())]).reset_index(drop=True)

def synthetic_rows(reals, arm):
    rows = pd.concat([reals.assign(slot=k) for k in range(args.k)], ignore_index=True)
    rows["location"] = rows.apply(lambda r: f"{args.generations_dir}/{arm}/{r.slot:02d}/{r.md5hash}.png", axis=1)
    missing = [p for p in rows["location"] if not os.path.exists(p)]
    if missing:
        sys.exit(f"{len(missing)} synthetic images missing for arm {arm}, e.g. {missing[:3]}")
    return rows.drop(columns="slot").assign(synthetic=True)

augmentation_dict = {"None": [], "aug_transforms": aug_transforms()}
for random_state in args.seeds:
    val = sample_per_label(pool, 40, random_state).assign(synthetic=False, is_valid=True)
    val["location"] = real_loc(val)
    train_pool = pool[~pool["md5hash"].isin(val["md5hash"])].reset_index(drop=True)
    for n in args.n:
        reals = sample_per_label(train_pool, n, random_state)  # same reals for every arm and augmentation
        for arm in args.arms:
            for augmentation in args.augmentation:
                run_id = f"{arm}__n{n}__{augmentation}__seed{random_state}"
                if results_csv.exists() and run_id in set(pd.read_csv(results_csv)["run_id"]):
                    print(f"skip {run_id} (done)", flush=True); continue
                train = reals.assign(location=real_loc(reals), synthetic=False)
                if arm != "none":
                    train = pd.concat([train, synthetic_rows(reals, arm)], ignore_index=True)
                cols = ["md5hash", "label", "location", "synthetic"]
                df = pd.concat([train[cols].assign(is_valid=False), val[cols].assign(is_valid=True)], ignore_index=True)
                n_train = int((~df["is_valid"]).sum())
                batch_size = 64 if n_train / 10 >= 100 else 32 if n_train / 10 >= 10 else 8
                set_seed(random_state, reproducible=True)
                dls = ImageDataLoaders.from_df(df, "/", fn_col="location", label_col="label", valid_col="is_valid",
                                               bs=batch_size, item_tfms=Resize(224), batch_tfms=augmentation_dict[augmentation])
                learn = vision_learner(dls, arch=efficientnet_v2_m, metrics=[error_rate, accuracy])
                t0 = time.time()
                learn.fit(args.epochs, cbs=[EarlyStoppingCallback(monitor="valid_loss", min_delta=0.0, patience=3)])
                epochs_run = len(learn.recorder.values)
                preds, _ = learn.get_preds(dl=dls.test_dl(test))
                vocab = list(learn.dls.vocab)
                top1 = [vocab[i] for i in preds.argmax(1)]
                top3 = [[vocab[j] for j in row] for row in preds.topk(3, dim=1).indices]
                truth = list(test["label"])
                correct = [a == b for a, b in zip(top1, truth)]
                m = len(truth); acc = sum(correct) / m
                lo, hi = proportion_confint(sum(correct), m, alpha=0.05, method="normal")
                top3_acc = sum(t in c for t, c in zip(truth, top3)) / m
                row = dict(run_id=run_id, generator=("none" if arm == "none" else f"qwen-{arm}"), include_synthetic=arm != "none",
                           n_training_per_label=n, augmentation=augmentation, n_synthetic_per_real=(args.k if arm != "none" else 0),
                           random_state=random_state, n_train=n_train, n_val=len(val), batch_size=batch_size, epochs_run=epochs_run,
                           train_s=round(time.time() - t0), top1_acc=acc, top1_ci_lower=lo, top1_ci_upper=hi, top3_acc=top3_acc,
                           n_test=m, architecture="EfficientNet-V2-M", generations_dir=args.generations_dir)
                pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not results_csv.exists(), index=False)
                pd.DataFrame({"run_id": run_id, "md5hash": test["md5hash"], "true_label": truth, "top1_label": top1,
                              "top1_prob": preds.max(1).values.numpy()}).to_csv(out / f"predictions__{run_id}.csv", index=False)
                print(f"{run_id}: n_train={n_train} bs={batch_size} epochs={epochs_run} top1={acc:.3f} [{lo:.3f}, {hi:.3f}] "
                      f"top3={top3_acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
                del learn, dls; torch.cuda.empty_cache()
print("sweep done", flush=True)
