"""Generate K synthetic images per real image with Qwen-Image-2.1, optionally with a per-class LoRA.

    # pilot: 16 psoriasis reals of the API pilot, edit with a LoRA checkpoint, seed-only variety
    python paper_experiments/2026/qwen_lora_generate.py --method edit --gpu 0 \
        --reals /workspace/synderm/pilot/gpt-image-2.5-sunburst-medium-varied-seed20261028-n16/reals.csv --label psoriasis \
        --lora /workspace/synderm/qwen_lora/edit/psoriasis/<run>/checkpoint-1000/pytorch_lora_weights.safetensors \
        --out /workspace/synderm/qwen_lora/pilot/edit-psoriasis-ckpt1000-single

Plan and run log: paper_experiments/2026/QWEN_LORA.md.

INPUT   --reals CSV with columns md5hash, label (any extra columns ignored); --label keeps one label.
        Real images from --images-dir/<md5hash>.jpg. For t2i the real image is only a file name, as in
        the paper's text-to-image arm; for edit it is the reference image.
PROMPTS --prompts single: every slot uses one prompt (T2I_PROMPT / EDIT_PROMPT); variety comes from seeds.
        --prompts varied: slot k appends VARIATIONS[k % 10] (same list as pilot_generate.py).
SEEDS   slot k of real m uses seed int(m[:8], 16) + k, so a (real, slot) image is reproducible on its own.
OUTPUT  --out/<kk>/<md5hash>.png and .json (prompt, seed, LoRA, timing); synthetic.csv when done.
        Existing images are skipped, so re-running resumes.
"""
import argparse, json, sys, time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-2.1"
T2I_PROMPT = "An image of {condition}, a skin disease"
EDIT_PROMPT = ("A clinical photograph of the same skin disease, {condition}, keeping the characteristic appearance "
               "of the lesions and the same skin tone as the reference, on a different patient. "
               "No text, watermarks or annotations.")
VARIATIONS = [
    "Close-up of a single lesion on the forearm, exam-room lighting.",
    "Several lesions across the upper back, photographed from about one metre.",
    "Lesions on the cheek and jawline of an older adult, soft daylight from a window.",
    "Lesions on the lower leg and shin, straight-on view, camera flash.",
    "Lesions on the back of the hand and fingers, close-up, slightly oblique angle.",
    "Widespread involvement of the chest and abdomen of a young adult, mid-range view.",
    "Lesions on the neck and behind the ear, close-up, overhead clinic lighting.",
    "Lesions on the thigh, photographed on an exam table with a paper sheet in the background.",
    "Lesions on the scalp margin and forehead, close-up, hair partly visible.",
    "Lesions on the foot and ankle, mid-range view, natural outdoor light.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["t2i", "edit"])
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--reals", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--lora", default=None, help="pytorch_lora_weights.safetensors; omit for untuned Qwen")
    ap.add_argument("--prompts", choices=["single", "varied"], default="single")
    ap.add_argument("--k", type=int, default=10, help="synthetic images per real")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--images-dir", default="/root/data/fitzpatrick17k/finalfitz17k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    reals = pd.read_csv(args.reals)
    if args.label:
        reals = reals[reals["label"] == args.label]
    if reals.empty:
        sys.exit(f"no reals left in {args.reals} after --label {args.label!r}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "generate_config.json").write_text(json.dumps({**vars(args), "num_reals": len(reals),
                                                          "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    print(f"{len(reals)} reals, {args.k} per real, method {args.method}, prompts {args.prompts}, lora {args.lora}", flush=True)

    # ---- load the pipeline on one GPU (96 GB: bf16 text encoder, no offload) -------------
    from diffusers import QwenImage21Pipeline
    device = f"cuda:{args.gpu}"
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device)
    if args.lora:
        lora = Path(args.lora)
        if not lora.exists():
            sys.exit(f"LoRA not found: {lora}")
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name)
    pipe.set_progress_bar_config(disable=True)

    # ---- generate: one call per real image, all K slots batched ---------------------------
    rows = []
    t_start = time.time()
    n_done = 0
    for i, r in enumerate(reals.itertuples()):
        condition = r.label.replace("-", " ")
        base = (T2I_PROMPT if args.method == "t2i" else EDIT_PROMPT).format(condition=condition)
        prompts = [base if args.prompts == "single" else base.rstrip(".") + ". " + VARIATIONS[k % 10]
                   for k in range(args.k)]
        seeds = [int(r.md5hash[:8], 16) + k for k in range(args.k)]
        paths = [out / f"{k:02d}" / f"{r.md5hash}.png" for k in range(args.k)]
        todo = [k for k in range(args.k) if not paths[k].exists()]
        if todo:
            gens = [torch.Generator(device="cpu").manual_seed(seeds[k]) for k in todo]
            kwargs = dict(prompt=[prompts[k] for k in todo], generator=gens, num_inference_steps=args.steps,
                          output_resolution=args.resolution)
            if args.method == "edit":
                kwargs["image"] = Image.open(Path(args.images_dir) / f"{r.md5hash}.jpg").convert("RGB")
            else:
                kwargs.update(height=args.resolution, width=args.resolution)
            t0 = time.time()
            images = pipe(**kwargs).images
            sec = (time.time() - t0) / len(todo)
            if len(images) != len(todo):
                sys.exit(f"pipeline returned {len(images)} images for {len(todo)} prompts ({r.md5hash})")
            for k, im in zip(todo, images):
                paths[k].parent.mkdir(exist_ok=True)
                im.save(paths[k])
                paths[k].with_suffix(".json").write_text(json.dumps({
                    "md5hash": r.md5hash, "label": r.label, "slot": k, "seed": seeds[k], "prompt": prompts[k],
                    "method": args.method, "lora": args.lora, "resolution": args.resolution, "steps": args.steps,
                    "seconds_per_image": round(sec, 2)}))
            n_done += len(todo)
        for k in range(args.k):
            rows.append({"md5hash": r.md5hash, "label": r.label, "slot": k, "location": str(paths[k])})
        if (i + 1) % 10 == 0 or i + 1 == len(reals):
            el = time.time() - t_start
            print(f"{i + 1}/{len(reals)} reals, {n_done} new images, {el / max(n_done, 1):.2f} s/image", flush=True)

    pd.DataFrame(rows).to_csv(out / "synthetic.csv", index=False)
    print(f"done: {out / 'synthetic.csv'}", flush=True)


if __name__ == "__main__":
    main()
