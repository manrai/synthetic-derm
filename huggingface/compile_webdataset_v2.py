"""Compile synderm 0.2 generation folders into WebDataset shards named by label, generator and method.

Input layout (from ``synderm.generation.generate_synthetic_dataset``):
    <input_dir>/<label>/<id>.png + <id>.json   (json has ``generator``/``backend``, ``method``, ``ref_id`` ...)
Output:
    <output_dir>/shard-{label}-{generator}-{method}-{index:05d}.tar

The 2024 shards (``compile_webdataset.py``) keep their original names; this script only adds new
shard groups next to them.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import webdataset as wds
from tqdm.auto import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=5000)
    args = ap.parse_args()
    inp, out = Path(args.input_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for png in sorted(inp.rglob("*.png")):
        meta_path = png.with_suffix(".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"{png} has no JSON metadata")
        meta = json.loads(meta_path.read_text())
        generator = meta.get("generator") or meta.get("backend")
        if not generator or not meta.get("method") or not meta.get("label"):
            raise ValueError(f"{meta_path} lacks generator/method/label")
        groups[(meta["label"], generator, meta["method"])].append((png, meta_path))

    for (label, generator, method), files in groups.items():
        prefix = f"shard-{label}-{generator}-{method}".replace("_", "-")
        for n, start in enumerate(range(0, len(files), args.shard_size)):
            shard = out / f"{prefix}-{n:05d}.tar"
            with wds.TarWriter(str(shard)) as tar:
                for png, meta_path in tqdm(files[start:start + args.shard_size], desc=shard.name, leave=False):
                    tar.write({"__key__": png.stem, "png": png.read_bytes(), "json": meta_path.read_bytes()})
            print(f"wrote {shard} ({min(args.shard_size, len(files) - start)} samples)")


if __name__ == "__main__":
    main()
