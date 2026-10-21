"""Fine-tune a LoRA adapter on the vignette sample.

    python -m synderm.example_scripts.train_lora --backend qwen-image-2.1 --steps 800 \
        --output /workspace/synderm/lora/qwen-t2i
"""
import argparse

from synderm.data import load_sample
from synderm.fine_tune import train_lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sdxl")
    ap.add_argument("--mode", default="text-to-image", choices=["text-to-image", "image-to-image"])
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ds = load_sample(split="train")
    weights = train_lora(args.backend, ds, args.output, mode=args.mode, resolution=args.resolution,
                         max_train_steps=args.steps, rank=args.rank)
    print("LoRA weights:", weights)


if __name__ == "__main__":
    main()
