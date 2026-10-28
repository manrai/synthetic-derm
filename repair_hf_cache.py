#!/usr/bin/env python3
"""Rebuild the symlinks in a Hugging Face hub cache whose snapshot entries became empty files.

The hub cache stores each file under ``snapshots/<revision>/<path>`` as a symlink into
``blobs/<sha>``. On an S3-backed FUSE mount (geesefs on /workspace) the symlinks were lost on
remount and came back as 0-byte regular files while the blobs survived. This script asks the Hub
for each repo's file list at the cached revision, maps every file to its blob (LFS sha256, or the
git blob id for small files) and recreates the symlink when the blob is present.

    python repair_hf_cache.py            # all repos under $HF_HOME/hub
    python repair_hf_cache.py --dry-run
"""
import argparse, os, sys
from pathlib import Path
from huggingface_hub import HfApi


def repo_from_dir(name: str):
    kind, rest = name.split("--", 1)
    repo_id = rest.replace("--", "/")
    return {"models": "model", "datasets": "dataset", "spaces": "space"}[kind], repo_id


def repair(hub: Path, dry_run=False) -> int:
    api = HfApi()
    broken_total = 0
    for repo_dir in sorted(hub.glob("*--*")):
        if not (repo_dir / "snapshots").is_dir():
            continue
        kind, repo_id = repo_from_dir(repo_dir.name)
        blobs = repo_dir / "blobs"
        for snap in (repo_dir / "snapshots").iterdir():
            revision = snap.name
            try:
                info = (api.model_info if kind == "model" else api.dataset_info)(repo_id, revision=revision, files_metadata=True)
            except Exception as e:
                print(f"!! {repo_id}@{revision[:8]}: cannot list files ({type(e).__name__}: {str(e)[:120]})"); broken_total += 1
                continue
            fixed = kept = missing = 0
            for s in info.siblings:
                target = snap / s.rfilename
                if not target.exists() and not target.is_symlink():
                    continue  # never downloaded (allow_patterns)
                if target.is_symlink() or target.stat().st_size > 0:
                    kept += 1; continue
                blob = blobs / (s.lfs.sha256 if s.lfs else s.blob_id)
                if not blob.exists():
                    print(f"   missing blob for {repo_id}:{s.rfilename}"); missing += 1; continue
                rel = os.path.relpath(blob, target.parent)
                if not dry_run:
                    target.unlink(); target.symlink_to(rel)
                fixed += 1
            broken_total += missing
            print(f"{repo_id}@{revision[:8]}: relinked {fixed}, already ok {kept}, missing blobs {missing}")
    return broken_total


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hub", default=os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"))
    a = ap.parse_args()
    sys.exit(1 if repair(Path(a.hub), a.dry_run) else 0)
