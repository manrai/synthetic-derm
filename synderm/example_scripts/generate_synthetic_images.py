"""Generate synthetic images for every label of a dataset with one backend.

    python -m synderm.example_scripts.generate_synthetic_images --backend qwen-image-2.1 \
        --method text-to-image --num-per-image 10 --output /workspace/synderm/generations/qwen-t2i
"""
import argparse

from synderm.data import load_sample
from synderm.generation import generate_synthetic_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sdxl")
    ap.add_argument("--method", default="text-to-image")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--num-per-image", type=int, default=10)
    ap.add_argument("--resolution", type=int, default=768)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from synderm.backends import get_backend
    gen = get_backend(args.backend, lora_path=args.lora)
    ds = load_sample(split="train")
    generate_synthetic_dataset(gen, args.output, method=args.method, dataset=ds, num_per_image=args.num_per_image,
                               height=args.resolution, width=args.resolution, limit=args.limit)


if __name__ == "__main__":
    main()
