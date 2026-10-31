"""Figure 5 (dose-response) sweep: the paper's dose_response.py protocol on a 2026 generation folder, results to CSV.

    # 1. list the reals that need synthetic images (the largest n of the first seed; smaller n nest inside it)
    python paper_experiments/2026/fig5_dose_response_sweep.py --seeds 119108 --n 16 --write-reals /workspace/synderm/fig5/reals-119108-n16.csv
    # 2. generate 150 slots for them with api_generate_pool.py (--reals that CSV, --k 150), make 512 px copies
    # 3. the sweep (one GPU)
    python paper_experiments/2026/fig5_dose_response_sweep.py --gpu 0 --seeds 119108 --n 16 \\
        --generations-dir /root/data/api512 --arm nano-banana-2.1 --out /workspace/synderm/fig5/sweep

Same protocol as paper_experiments/Experiments/dose_response.py (read it first):
  data      fitz_metadata/fitzpatrick17k_10label_clean_training.csv (md5 11b4b1c7..., identical to the 2023 machine's
            copy), nine most common labels (folliculitis dropped), held-out CSV removed (test = 360, 40 per label).
            Validation = 40 real per label sampled with random_state; classifier reals = n per label from the rest with
            the same random_state, both with the script's own expression groupby(['label']).apply(sample(n, random_state)),
            so seeds 119108 and 329101 (the 2023 Figure 5 runs, from the model file names in derm/dose-response/) draw the
            same reals as in 2023. 16, 32 and 64 nest for a given seed. The same reals for every dose and augmentation.
  synthetic dose k = the k images in slots 00..k-1 filed under each selected real: --generations-dir/<arm>/<kk>/<md5>.png.
            Slots 0-9 are the Figure 3 images, 10-149 the dose-response prompts (api_generate_pool.py). Missing files
            are an error. (2023: slots 10.. of generations-more, a separate set from the Figure 3 slots 0-9.)
            Dose 0 = real only.
  model     fastai vision_learner, EfficientNetV2-M, 224 px, fit(50), EarlyStopping on valid_loss (patience 3) and
            SaveModelCallback on valid_loss: the best epoch's weights are loaded before testing (as in dose_response.py,
            unlike n_per_label.py). Batch size 128 / 64 / 32 / 8 when n_train/10 >= 500 / >= 100 / >= 10 / else (the
            dose_response.py thresholds). aug_transforms() or none. set_seed(random_state) before each run (2026 addition).
  score     top-1 accuracy on the 360 held-out images with a Wald 95% CI, top-3 accuracy; per-image predictions saved.
  grid      seeds x n x doses x {None, aug_transforms}; a run id already in results.csv is skipped (resume).
"""
import argparse, os, shutil, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--seeds", type=int, nargs="+", default=[119108], help="random_state values; 2023 used 119108 and 329101")
ap.add_argument("--n", type=int, nargs="+", default=[16], help="real images per label (paper: 16 32 64)")
ap.add_argument("--doses", type=int, nargs="+", default=[0, 10, 25, 50, 75, 100, 125, 150], help="synthetic per real")
ap.add_argument("--arm", default="nano-banana-2.1", help="generation folder under --generations-dir")
ap.add_argument("--augmentation", nargs="+", default=["None", "aug_transforms"])
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
ap.add_argument("--generations-dir", default="/root/data/api512")
ap.add_argument("--model-dir", default="/root/fig5_models", help="best-epoch weights, local disk; deleted after each run")
ap.add_argument("--out", default="/workspace/synderm/fig5/sweep")
ap.add_argument("--write-reals", default=None, help="write the reals of the first seed at the largest n to this CSV and exit")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
META = REPO / "fitz_metadata"

# ---- data split, as in dose_response.py ------------------------------------------------
metadata = pd.read_csv(META / "fitzpatrick17k_10label_clean_training.csv")
nine = metadata["label"].value_counts().index[:9]
metadata = metadata[metadata["label"].isin(nine)].reset_index(drop=True)
test = pd.read_csv(META / "fitzpatrick17k_10label_clean_held_out_set.csv")
test = test[test["label"].isin(nine)].reset_index(drop=True)
pool = metadata[~metadata["md5hash"].isin(test["md5hash"])].reset_index(drop=True)
real_loc = lambda df: df["md5hash"].map(lambda h: f"{args.images_dir}/{h}.jpg")
print(f"pool {len(pool)} real images over {pool['label'].nunique()} labels; test {len(test)}", flush=True)

def sample_per_label(df, n, random_state):
    # dose_response.py: groupby(['label']).apply(lambda x: x.sample(n=n, random_state=random_state)). Written as a loop
    # over the labels in groupby's (sorted) order because pandas >= 3 drops the label column inside apply; same rows
    # in the same order (checked against the groupby form for seeds 119108 and 329101, 2026-10-31).
    return pd.concat([df[df["label"] == lab].sample(n=n, random_state=random_state, replace=False)
                      for lab in sorted(df["label"].unique())]).reset_index(drop=True)

def split(random_state):
    val = sample_per_label(pool, 40, random_state)
    train_pool = pool[~pool["md5hash"].isin(val["md5hash"])].reset_index(drop=True)
    return val, train_pool

if args.write_reals:
    val, train_pool = split(args.seeds[0])
    reals = sample_per_label(train_pool, max(args.n), args.seeds[0])
    for n in sorted(args.n):  # nesting check: the smaller draws are subsets of the largest
        assert set(sample_per_label(train_pool, n, args.seeds[0])["md5hash"]) <= set(reals["md5hash"]), n
    Path(args.write_reals).parent.mkdir(parents=True, exist_ok=True)
    reals[["md5hash", "label"]].to_csv(args.write_reals, index=False)
    print(f"wrote {len(reals)} reals (seed {args.seeds[0]}, n={max(args.n)} per label) to {args.write_reals}")
    sys.exit(0)

import torch
from fastai.vision.all import (ImageDataLoaders, Resize, aug_transforms, vision_learner, efficientnet_v2_m,
                               error_rate, accuracy, EarlyStoppingCallback, SaveModelCallback, set_seed)
from statsmodels.stats.proportion import proportion_confint

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
results_csv = out / "results.csv"
test = test.assign(location=real_loc(test), synthetic=False, is_valid=False)

def synthetic_rows(reals, k):
    rows = pd.concat([reals.assign(slot=s) for s in range(k)], ignore_index=True)
    rows["location"] = rows.apply(lambda r: f"{args.generations_dir}/{args.arm}/{r.slot:02d}/{r.md5hash}.png", axis=1)
    missing = [p for p in rows["location"] if not os.path.exists(p)]
    if missing:
        sys.exit(f"{len(missing)} synthetic images missing for {args.arm} at dose {k}, e.g. {missing[:3]}")
    return rows.drop(columns="slot").assign(synthetic=True)

augmentation_dict = {"None": [], "aug_transforms": aug_transforms()}
for random_state in args.seeds:
    val, train_pool = split(random_state)
    val = val.assign(location=real_loc(val), synthetic=False, is_valid=True)
    for n in args.n:
        reals = sample_per_label(train_pool, n, random_state)  # same reals for every dose and augmentation
        for k in args.doses:
            for augmentation in args.augmentation:
                arm = args.arm if k else "none"
                run_id = f"{arm}__n{n}__k{k}__{augmentation}__seed{random_state}"
                if results_csv.exists() and run_id in set(pd.read_csv(results_csv)["run_id"]):
                    print(f"skip {run_id} (done)", flush=True); continue
                train = reals.assign(location=real_loc(reals), synthetic=False)
                if k:
                    train = pd.concat([train, synthetic_rows(reals, k)], ignore_index=True)
                cols = ["md5hash", "label", "location", "synthetic"]
                df = pd.concat([train[cols].assign(is_valid=False), val[cols].assign(is_valid=True)], ignore_index=True)
                n_train = int((~df["is_valid"]).sum())
                batch_size = 128 if n_train / 10 >= 500 else 64 if n_train / 10 >= 100 else 32 if n_train / 10 >= 10 else 8
                set_seed(random_state, reproducible=True)
                dls = ImageDataLoaders.from_df(df, "/", fn_col="location", label_col="label", valid_col="is_valid",
                                               bs=batch_size, item_tfms=Resize(224), batch_tfms=augmentation_dict[augmentation])
                learn = vision_learner(dls, arch=efficientnet_v2_m, metrics=[error_rate, accuracy], model_dir=args.model_dir)
                t0 = time.time()
                learn.fit(args.epochs, cbs=[EarlyStoppingCallback(monitor="valid_loss", min_delta=0.0, patience=3),
                                            SaveModelCallback(monitor="valid_loss", fname=run_id)])
                learn.load(run_id)  # best valid_loss epoch, as dose_response.py does
                epochs_run = len(learn.recorder.values)
                preds, _ = learn.get_preds(dl=dls.test_dl(test))
                vocab = list(learn.dls.vocab)
                top1 = [vocab[i] for i in preds.argmax(1)]
                top3 = [[vocab[j] for j in row] for row in preds.topk(3, dim=1).indices]
                truth = list(test["label"])
                correct = [x == y for x, y in zip(top1, truth)]
                m = len(truth); acc = sum(correct) / m
                lo, hi = proportion_confint(sum(correct), m, alpha=0.05, method="normal")
                top3_acc = sum(t in c for t, c in zip(truth, top3)) / m
                row = dict(run_id=run_id, generator=arm, include_synthetic=k > 0, n_training_per_label=n, augmentation=augmentation,
                           n_synthetic_per_real=k, random_state=random_state, n_train=n_train, n_val=len(val),
                           batch_size=batch_size, epochs_run=epochs_run, train_s=round(time.time() - t0), top1_acc=acc,
                           top1_ci_lower=lo, top1_ci_upper=hi, top3_acc=top3_acc, n_test=m, architecture="EfficientNet-V2-M",
                           protocol="dose_response.py", generations_dir=args.generations_dir)
                pd.DataFrame([row]).to_csv(results_csv, mode="a", header=not results_csv.exists(), index=False)
                pd.DataFrame({"run_id": run_id, "md5hash": test["md5hash"], "true_label": truth, "top1_label": top1,
                              "top1_prob": preds.max(1).values.numpy()}).to_csv(out / f"predictions__{run_id}.csv", index=False)
                print(f"{run_id}: n_train={n_train} bs={batch_size} epochs={epochs_run} top1={acc:.3f} [{lo:.3f}, {hi:.3f}] "
                      f"top3={top3_acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
                shutil.rmtree(args.model_dir, ignore_errors=True)
                del learn, dls; torch.cuda.empty_cache()
print("sweep done", flush=True)
