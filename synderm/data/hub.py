"""Access to the published synthetic-derm datasets on Hugging Face (the 2024 release).

``tbuckley/synthetic-derm-1M`` holds per-label WebDataset shards named
``shard-{label}-{method}-{submethod}-{index}.tar`` where ``method`` is one of
``finetune-text-to-image``, ``finetune-inpaint``, ``pretrained-text-to-image``,
``pretrained-inpaint`` and ``submethod`` is ``text-to-image``, ``inpaint`` or
``inpaint-outpaint``. Every sample's JSON has ``md5hash`` (the Fitzpatrick 17k image it was
generated from), ``label``, ``method``, ``submethod`` and ``generation_num``.

:func:`load_2024_shards` downloads shards through the Hugging Face cache and unpacks a bounded
number of samples per label into the synderm generation layout (``<root>/<label>/<id>.png`` +
``.json`` with ``generator``, ``method`` and ``ref_id`` filled in), so the result is an ordinary
:class:`~synderm.data.SyntheticDataset` whose ``ref_id`` values join to real Fitzpatrick 17k ids.
"""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from typing import Optional, Sequence

from huggingface_hub import HfApi, hf_hub_download

from .atlas import default_data_root
from .datasets import SyntheticDataset, normalize_label

REPO_1M = "tbuckley/synthetic-derm-1M"
GENERATOR_2024 = {"finetune": "sd2.1-dreambooth-2024", "pretrained": "sd2.1-pretrained-2024"}
METHOD_2024 = {"text_to_image": "text-to-image", "inpaint": "inpaint", "inpaint_outpaint": "inpaint-outpaint"}


def list_2024_shards(labels: Optional[Sequence[str]] = None, method: str = "finetune-text-to-image",
                     submethod: str = "text-to-image") -> list[str]:
    files = HfApi().list_repo_files(REPO_1M, repo_type="dataset")
    keep = {normalize_label(l) for l in labels} if labels else None
    out = []
    for f in files:
        if not f.endswith(".tar"):
            continue
        name = Path(f).stem  # shard-<label>-<method>-<submethod>-<index>
        suffix = f"-{method}-{submethod}-"
        if suffix not in name:
            continue
        label = name[len("shard-"):name.index(suffix)]
        if keep is None or label in keep:
            out.append(f)
    return sorted(out)


def load_2024_shards(
    labels: Sequence[str],
    method: str = "finetune-text-to-image",
    submethod: str = "text-to-image",
    max_per_label: int = 1000,
    max_shards_per_label: int = 1,
    data_root: str | os.PathLike | None = None,
    ref_ids: Optional[set[str]] = None,
    exclude_ref_ids: Optional[set[str]] = None,
) -> SyntheticDataset:
    """Unpack up to ``max_per_label`` samples per label from the 2024 release.

    ``ref_ids`` restricts samples to those generated from the given real image ids (e.g. the
    vignette sample's hashes), which is what makes the paper's "10 synthetic per real" protocol
    reproducible on a subset. ``exclude_ref_ids`` drops samples generated from the given real
    images (validation or held-out hashes). Shards are 1 to 2.5 GB each and are cached by
    huggingface_hub; the first shard of a label covers source images whose md5 hashes sort first.
    """
    root = Path(data_root) if data_root else default_data_root()
    out_root = root / "synthetic-derm-1M-2024" / f"{method}-{submethod}"
    gen_name = GENERATOR_2024[method.split("-")[0]]
    for label in labels:
        label = normalize_label(label)
        shards = list_2024_shards([label], method, submethod)[:max_shards_per_label]
        if not shards:
            raise FileNotFoundError(f"No {method}/{submethod} shards for label {label!r} in {REPO_1M}")
        label_dir = out_root / label
        marker = label_dir / ".complete"
        key = {"max_per_label": max_per_label, "max_shards": max_shards_per_label,
               "ref_ids": sorted(ref_ids) if ref_ids else None,
               "exclude_ref_ids": sorted(exclude_ref_ids) if exclude_ref_ids else None}
        if marker.exists() and {k: v for k, v in json.loads(marker.read_text()).items() if k != "n"} == key:
            continue
        for old in label_dir.glob("*"):
            old.unlink()
        label_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for shard in shards:
            path = hf_hub_download(REPO_1M, shard, repo_type="dataset")
            with tarfile.open(path) as tf:
                pending: dict[str, dict] = {}
                for m in tf:
                    if n >= max_per_label:
                        break
                    stem, ext = os.path.splitext(m.name)
                    if ext == ".json":
                        meta = json.loads(tf.extractfile(m).read())
                        if ref_ids is not None and meta.get("md5hash") not in ref_ids:
                            continue
                        if exclude_ref_ids is not None and meta.get("md5hash") in exclude_ref_ids:
                            continue
                        pending[stem] = meta
                    elif ext == ".png" and stem in pending:
                        meta = pending.pop(stem)
                        sid = f"{meta['md5hash']}-{submethod}-{int(meta.get('generation_num', 0)):03d}"
                        (label_dir / f"{sid}.png").write_bytes(tf.extractfile(m).read())
                        rec = {"id": sid, "label": normalize_label(meta["label"]), "ref_id": meta["md5hash"],
                               "generator": gen_name, "method": METHOD_2024.get(meta.get("submethod"), submethod),
                               "source": f"{REPO_1M}/{shard}", "original": meta}
                        (label_dir / f"{sid}.json").write_text(json.dumps(rec, indent=1))
                        n += 1
            if n >= max_per_label:
                break
        if n == 0:
            raise RuntimeError(f"No samples extracted for {label!r} (ref_ids filter too strict?)")
        marker.write_text(json.dumps({**key, "n": n}))
    return SyntheticDataset(out_root, labels=labels)
