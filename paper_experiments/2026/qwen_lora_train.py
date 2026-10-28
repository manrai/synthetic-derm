"""Train one Qwen-Image-2.1 LoRA for one Fitzpatrick 17k label (text-to-image or edit).

    python paper_experiments/2026/qwen_lora_train.py --label psoriasis --method t2i  --gpu 0
    python paper_experiments/2026/qwen_lora_train.py --label psoriasis --method edit --gpu 1

Plan and run log: paper_experiments/2026/QWEN_LORA.md.

DATA
  training pool   fitz_metadata/fitzpatrick17k_10label_clean_training.csv, rows with this label.
                  All of them, as in the paper's per-class DreamBooth (the validation rows of the
                  later classifier sweep are inside this pool; the held-out test CSV is never read).
  images          --images-dir/<md5hash>.jpg (default /root/data/fitzpatrick17k/finalfitz17k)

CAPTIONS (Stage A of the plan: static)
  t2i   "An image of {condition}, a skin disease"  (the paper's prompt)
  edit  EDIT_PROMPT below, the API pilot's edit frame without a variation sentence

EDIT PAIRS
  There are no before/after photographs. Each pair is (reference = another image of the same label,
  target = this image). The reference is drawn from images with the same Fitzpatrick type, else an
  adjacent type (+-1), else any type; unknown type (-1) pairs with any. Counts go to pairs_summary.json.

OUTPUT  --out (default /workspace/synderm/qwen_lora/<method>/<label-slug>/<run-name>/)
  (training parquet and resized PNGs: --data-root/<method>/<label-slug>/<run-name>/, RAM by default)
  checkpoint-<N>/   LoRA weights every --checkpoint-every steps (pytorch_lora_weights.safetensors)
  pytorch_lora_weights.safetensors, train_config.json, train.log
"""
import argparse, json, os, random, subprocess, sys, time
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO / "fitz_metadata" / "fitzpatrick17k_10label_clean_training.csv"
SCRIPTS = REPO / "synderm" / "fine_tune" / "scripts"
MODEL_ID = "Qwen/Qwen-Image-2.1"

T2I_PROMPT = "An image of {condition}, a skin disease"
EDIT_PROMPT = ("A clinical photograph of the same skin disease, {condition}, keeping the characteristic appearance "
               "of the lesions and the same skin tone as the reference, on a different patient. "
               "No text, watermarks or annotations.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="label as in the CSV, e.g. psoriasis or 'basal cell carcinoma'")
    ap.add_argument("--method", required=True, choices=["t2i", "edit"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--pairs-per-image", type=int, default=2, help="edit only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data-root", default="/dev/shm/qwen_lora_data")
    args = ap.parse_args()

    slug = args.label.replace(" ", "-")
    run_name = args.run_name or f"r{args.rank}-lr{args.lr:g}-{args.resolution}px-{args.steps}steps"
    out = Path(args.out or f"/workspace/synderm/qwen_lora/{args.method}/{slug}/{run_name}")
    if (out / "pytorch_lora_weights.safetensors").exists():
        sys.exit(f"{out} already has final weights; pick another --run-name or delete it")
    out.mkdir(parents=True, exist_ok=True)
    condition = args.label.replace("-", " ")

    # ---- 1. training images of this label -------------------------------------------------
    pool = pd.read_csv(TRAIN_CSV)
    rows = pool[pool["label"] == args.label].reset_index(drop=True)
    if rows.empty:
        sys.exit(f"no rows with label {args.label!r} in {TRAIN_CSV}; labels: {sorted(pool['label'].unique())}")
    missing = [m for m in rows["md5hash"] if not (Path(args.images_dir) / f"{m}.jpg").exists()]
    if missing:
        sys.exit(f"{len(missing)} images missing from {args.images_dir}, e.g. {missing[:3]}")
    print(f"{args.label}: {len(rows)} training-pool images from {TRAIN_CSV.name}", flush=True)

    # resized training images and the parquet go to RAM (/dev/shm): the /workspace FUSE volume is too slow for them
    data_dir = Path(args.data_root) / args.method / slug / run_name
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for m in rows["md5hash"]:
        p = img_dir / f"{m}.png"
        if not p.exists():
            im = Image.open(Path(args.images_dir) / f"{m}.jpg").convert("RGB")
            im.thumbnail((1024, 1024))
            im.save(p)
        paths[m] = str(p)

    # ---- 2. captions and (for edit) reference pairs ---------------------------------------
    rng = random.Random(args.seed)
    records = []
    if args.method == "t2i":
        caption = T2I_PROMPT.format(condition=condition)
        for m in rows["md5hash"]:
            records.append({"image": paths[m], "caption": caption, "id": m})
    else:
        caption = EDIT_PROMPT.format(condition=condition)
        fitz = dict(zip(rows["md5hash"], rows["fitzpatrick_scale"]))
        counts = {"same_type": 0, "adjacent_type": 0, "any_type": 0, "target_type_unknown": 0}
        for m in rows["md5hash"]:
            others = [o for o in rows["md5hash"] if o != m]
            t = fitz[m]
            if t == -1:
                cands, kind = others, "target_type_unknown"
            else:
                cands, kind = [o for o in others if fitz[o] == t], "same_type"
                if len(cands) < args.pairs_per_image:
                    cands, kind = [o for o in others if fitz[o] != -1 and abs(fitz[o] - t) <= 1], "adjacent_type"
                if len(cands) < args.pairs_per_image:
                    cands, kind = others, "any_type"
            for ref in rng.sample(cands, args.pairs_per_image):
                records.append({"image": paths[m], "cond_image": paths[ref], "caption": caption, "id": m, "cond_id": ref})
                counts[kind] += 1
        (out / "pairs_summary.json").write_text(json.dumps(counts, indent=2))
        print("edit pairs:", counts, flush=True)

    from datasets import Dataset, Image as HFImage
    ds = Dataset.from_list(records).cast_column("image", HFImage())
    if args.method == "edit":
        ds = ds.cast_column("cond_image", HFImage())
    (data_dir / "data").mkdir(exist_ok=True)
    ds.to_parquet(str(data_dir / "data" / "train-00000-of-00001.parquet"))

    # ---- 3. launch the vendored diffusers trainer -----------------------------------------
    script = SCRIPTS / ("train_dreambooth_lora_qwenimage21.py" if args.method == "t2i"
                        else "train_dreambooth_lora_qwenimage21_img2img.py")
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", "1", "--num_machines", "1",
           "--mixed_precision", "bf16", "--dynamo_backend", "no", str(script),
           "--pretrained_model_name_or_path", MODEL_ID,
           "--dataset_name", str(data_dir), "--image_column", "image", "--caption_column", "caption",
           "--instance_prompt", caption,
           "--output_dir", str(out), "--resolution", str(args.resolution),
           "--train_batch_size", str(args.batch_size), "--gradient_accumulation_steps", str(args.grad_accum),
           "--learning_rate", str(args.lr), "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
           "--max_train_steps", str(args.steps), "--rank", str(args.rank), "--lora_alpha", str(args.rank),
           "--seed", str(args.seed), "--mixed_precision", "bf16",
           "--checkpointing_steps", str(args.checkpoint_every), "--checkpoints_total_limit", "100",
           "--cache_latents", "--gradient_checkpointing", "--random_flip",
           "--skip_final_inference", "--report_to", "tensorboard"]
    if args.method == "edit":
        cmd += ["--cond_image_column", "cond_image"]

    config = {**vars(args), "condition": condition, "caption": caption, "num_images": len(rows),
              "num_records": len(records), "data_dir": str(data_dir), "hf_home": os.environ.get("HF_HOME"), "script": script.name, "command": cmd,
              "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
              "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "train_config.json").write_text(json.dumps(config, indent=2))
    print("launching:", " ".join(cmd), flush=True)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu), "TOKENIZERS_PARALLELISM": "false"}
    t0 = time.time()
    with open(out / "train.log", "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(out))
    minutes = (time.time() - t0) / 60
    if proc.returncode != 0:
        tail = "".join(open(out / "train.log").readlines()[-40:])
        sys.exit(f"TRAINING FAILED (exit {proc.returncode}) after {minutes:.1f} min. Log: {out / 'train.log'}\n{tail}")
    if not (out / "pytorch_lora_weights.safetensors").exists():
        sys.exit(f"TRAINING FAILED: trainer exited 0 but wrote no weights in {out}")
    config.update(finished=time.strftime("%Y-%m-%dT%H:%M:%S"), train_minutes=round(minutes, 1))
    (out / "train_config.json").write_text(json.dumps(config, indent=2))
    print(f"done in {minutes:.1f} min: {out / 'pytorch_lora_weights.safetensors'}", flush=True)


if __name__ == "__main__":
    main()
