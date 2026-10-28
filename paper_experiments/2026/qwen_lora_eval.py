"""Score synthetic image folders for the Qwen LoRA pilot: label consistency, variety, copying, skin tone.

    python paper_experiments/2026/qwen_lora_eval.py --gpu 2 --label psoriasis \
        --folders /workspace/synderm/qwen_lora/pilot/edit-untuned-single /workspace/synderm/qwen_lora/pilot/edit-ckpt1000-single ... \
        --out /workspace/synderm/qwen_lora/pilot/eval

Plan and run log: paper_experiments/2026/QWEN_LORA.md.

Each folder needs synthetic.csv with md5hash (the real it is filed under), label, slot, location.
The GPT Image varied-prompt pilot folder also works (its synthetic.csv has the same columns plus
topup/source_md5; top-ups and slots >= K are dropped).

FEATURES  DINOv2 ViT-S/14 (timm vit_small_patch14_dinov2.lvd142m, cached), 224 px, L2-normalised CLS.
MEASURES  per folder, over the synthetic images of --label:
  label_match     share of images a real-only 9-label probe calls --label. The probe is logistic
                  regression on DINOv2 features of the training pool (all nine labels); its own
                  accuracy is 5-fold cross-validated on the pool. The held-out test CSV is never read.
  within_dist     mean pairwise cosine distance among the K images of one real, averaged over reals
                  (higher = more variety). Reference: the same statistic over random sets of K real
                  images of the label ("real_within_dist").
  ref_dist        mean cosine distance from each image to the real it is filed under (edit: its reference)
  nn_real_dist    median cosine distance to the nearest training-pool image of the label, and
  near_copies     share of images closer than 0.05 to some training-pool image (memorisation check;
                  for scale, the median nearest-neighbour distance among distinct real images is reported)
  ita_shift       median |ITA(synthetic) - ITA(real it is filed under)| in degrees, ITA from the median
                  L*, b* of the image centre (a rough skin-tone check, lesion pixels included)
OUTPUT  --out/metrics.csv (one row per folder), --out/grid__<folder>.jpg (6 reals x K slots, real on the left)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

TRAIN_CSV = Path(__file__).resolve().parents[2] / "fitz_metadata" / "fitzpatrick17k_10label_clean_training.csv"
NINE = ["psoriasis", "lichen planus", "basal cell carcinoma", "allergic contact dermatitis", "lupus erythematosus",
        "neutrophilic dermatoses", "photodermatoses", "sarcoidosis", "squamous cell carcinoma"]


def ita_of(path):
    from skimage.color import rgb2lab
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im = im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).resize((128, 128))
    lab = rgb2lab(np.asarray(im) / 255.0)
    L, b = np.median(lab[..., 0]), np.median(lab[..., 2])
    return float(np.degrees(np.arctan2(L - 50, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--folders", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{args.gpu}"

    # ---- DINOv2 feature extractor ---------------------------------------------------------
    import timm
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**{**cfg, "input_size": (3, 224, 224), "crop_pct": 1.0})

    @torch.no_grad()
    def embed(paths, bs=128):
        feats = []
        for i in range(0, len(paths), bs):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(device)
            feats.append(torch.nn.functional.normalize(model(x).float(), dim=1).cpu())
        return torch.cat(feats).numpy()

    # ---- real-only probe on the training pool (cached) ------------------------------------
    pool = pd.read_csv(TRAIN_CSV)
    pool = pool[pool["label"].isin(NINE)].reset_index(drop=True)
    cache = out.parent / "pool_dinov2.npy"
    if cache.exists():
        pool_feats = np.load(cache)
    else:
        pool_feats = embed([str(Path(args.images_dir) / f"{m}.jpg") for m in pool["md5hash"]])
        np.save(cache, pool_feats)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    probe = LogisticRegression(max_iter=3000, C=1.0)
    cv_acc = cross_val_score(probe, pool_feats, pool["label"], cv=5).mean()
    probe.fit(pool_feats, pool["label"])
    print(f"probe: 9-label logistic regression on {len(pool)} pool images, 5-fold CV accuracy {cv_acc:.3f}", flush=True)

    cls_mask = (pool["label"] == args.label).values
    cls_feats = pool_feats[cls_mask]
    sims = cls_feats @ cls_feats.T
    np.fill_diagonal(sims, -1)
    real_nn = float(np.median(1 - sims.max(1)))
    rng = np.random.default_rng(0)
    real_within = []
    for _ in range(200):
        idx = rng.choice(len(cls_feats), args.k, replace=False)
        f = cls_feats[idx]
        d = 1 - f @ f.T
        real_within.append(d[np.triu_indices(args.k, 1)].mean())
    real_within = float(np.mean(real_within))
    md5_to_row = {m: i for i, m in enumerate(pool["md5hash"])}

    # ---- score each folder ----------------------------------------------------------------
    rows = []
    for folder in args.folders:
        folder = Path(folder)
        syn = pd.read_csv(folder / "synthetic.csv")
        syn = syn[(syn["label"] == args.label) & (syn["slot"] < args.k)]
        if "topup" in syn.columns:
            syn = syn[~syn["topup"].astype(bool)]
        syn = syn[syn["location"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        if syn.empty:
            sys.exit(f"{folder}: no existing images for {args.label}")
        feats = embed(list(syn["location"]))
        pred = probe.predict(feats)
        within = []
        for m, g in syn.groupby("md5hash"):
            if len(g) > 1:
                f = feats[g.index]
                d = 1 - f @ f.T
                within.append(d[np.triu_indices(len(g), 1)].mean())
        ref_feats = pool_feats[[md5_to_row[m] for m in syn["md5hash"]]]
        ref_dist = 1 - (feats * ref_feats).sum(1)
        nn = 1 - (feats @ cls_feats.T).max(1)
        ita_real = {m: ita_of(Path(args.images_dir) / f"{m}.jpg") for m in syn["md5hash"].unique()}
        ita_shift = [abs(ita_of(p) - ita_real[m]) for p, m in zip(syn["location"], syn["md5hash"])]
        row = {"folder": folder.name, "label": args.label, "n_images": len(syn), "n_reals": syn["md5hash"].nunique(),
               "label_match": float((pred == args.label).mean()),
               "top_wrong_label": pd.Series(pred[pred != args.label]).value_counts().head(1).to_dict(),
               "within_dist": float(np.mean(within)), "real_within_dist": real_within,
               "ref_dist": float(ref_dist.mean()), "nn_real_dist": float(np.median(nn)),
               "real_nn_dist": real_nn, "near_copies": float((nn < 0.05).mean()),
               "ita_shift": float(np.median(ita_shift)), "probe_cv_acc": float(cv_acc)}
        rows.append(row)
        print(json.dumps(row), flush=True)

        # grid: 6 reals x K slots, real on the left
        tile = 160
        reals6 = list(syn["md5hash"].unique()[:6])
        grid = Image.new("RGB", (tile * (args.k + 1), tile * len(reals6)), "white")
        for i, m in enumerate(reals6):
            grid.paste(Image.open(Path(args.images_dir) / f"{m}.jpg").convert("RGB").resize((tile, tile)), (0, i * tile))
            for _, s in syn[syn["md5hash"] == m].iterrows():
                grid.paste(Image.open(s["location"]).convert("RGB").resize((tile, tile)), ((int(s["slot"]) + 1) * tile, i * tile))
        grid.save(out / f"grid__{folder.name}.jpg", quality=88)

    metrics = out / "metrics.csv"
    df = pd.DataFrame(rows)
    if metrics.exists():
        old = pd.read_csv(metrics)
        df = pd.concat([old[~old["folder"].isin(df["folder"])], df])
    df.to_csv(metrics, index=False)
    print(f"wrote {metrics}", flush=True)


if __name__ == "__main__":
    main()
