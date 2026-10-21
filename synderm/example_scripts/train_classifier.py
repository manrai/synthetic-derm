"""Train a classifier on the vignette sample with and without synthetic augmentation.

    python -m synderm.example_scripts.train_classifier --synthetic /workspace/synderm/generations/qwen-t2i
"""
import argparse

from synderm.classify import ClassificationDataset, build_transforms, train_classifier
from synderm.data import SyntheticDataset, load_sample
from synderm.utils import synthetic_train_val_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default=None, help="generation folder; omit for the real-only baseline")
    ap.add_argument("--n-real-per-class", type=int, default=32)
    ap.add_argument("--n-synthetic-per-real", type=int, default=10)
    ap.add_argument("--backbone", default="tf_efficientnetv2_m.in1k")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    real = load_sample(split="train")
    test = load_sample(split="val")
    classes = real.labels
    syn = SyntheticDataset(args.synthetic) if args.synthetic else None
    train, val = synthetic_train_val_split(real, syn, per_class_test_size=8, n_real_per_class=args.n_real_per_class,
                                           n_synthetic_per_real=args.n_synthetic_per_real, random_state=args.seed)
    res = train_classifier(
        ClassificationDataset(train, classes, build_transforms(augment=True)),
        ClassificationDataset(val, classes),
        classes, test_dataset=ClassificationDataset(test, classes), backbone=args.backbone, seed=args.seed,
    )
    m = res.test_metrics
    print(f"test accuracy {m['accuracy']:.3f} (95% CI {m['ci_low']:.3f}-{m['ci_high']:.3f}, n={m['n']})")


if __name__ == "__main__":
    main()
