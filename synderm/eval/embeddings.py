from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import timm
import torch
from torch.utils.data import DataLoader, Dataset

#: CLIP ViT-B/32 image tower (what the paper used, via timm) and DINOv2 ViT-S/14.
CLIP_MODEL = "vit_base_patch32_clip_224.openai"
DINOV2_MODEL = "vit_small_patch14_dinov2.lvd142m"


class ImageEmbedder:
    """Image embeddings from a timm backbone (CLIP by default), L2-normalized."""

    def __init__(self, model_name: str = CLIP_MODEL, device: str = "cuda", batch_size: int = 64, num_workers: int = 4):
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        self.device, self.batch_size, self.num_workers = device, batch_size, num_workers
        self.model_name = model_name

    @torch.no_grad()
    def __call__(self, dataset: Dataset) -> np.ndarray:
        class _Wrap(Dataset):
            def __init__(s, ds, tf): s.ds, s.tf = ds, tf
            def __len__(s): return len(s.ds)
            def __getitem__(s, i): return s.tf(s.ds[i]["image"].convert("RGB"))
        loader = DataLoader(_Wrap(dataset, self.transform), batch_size=self.batch_size, num_workers=self.num_workers)
        feats = []
        for x in loader:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.startswith("cuda")):
                f = self.model(x.to(self.device))
            feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu().numpy())
        return np.concatenate(feats)


def embed_dataset(dataset: Dataset, embedder: Optional[ImageEmbedder] = None, **kw) -> tuple[np.ndarray, list[dict]]:
    """Embeddings plus the per-item metadata (everything but the image)."""
    embedder = embedder or ImageEmbedder(**kw)
    feats = embedder(dataset)
    meta = []
    for i in range(len(dataset)):
        item = dataset[i] if not hasattr(dataset, "items") else None
        if item is None:
            path, m = dataset.items[i]
            item = dict(m) if isinstance(m, dict) else {"label": m, "id": path.stem}
        meta.append({k: v for k, v in item.items() if k != "image"})
    return feats, meta


def matched_vs_unmatched_distances(
    real_feats: np.ndarray, real_ids: Sequence[str], syn_feats: np.ndarray, syn_ref_ids: Sequence[str],
    rng: Optional[np.random.Generator] = None, n_unmatched: int = 5000,
) -> dict[str, np.ndarray]:
    """The paper's Figure 2B: cosine distances for matched (synthetic to its own reference)
    pairs vs random real-synthetic and real-real pairs."""
    rng = rng or np.random.default_rng(0)
    idx = {rid: i for i, rid in enumerate(real_ids)}
    matched = np.array([1 - syn_feats[j] @ real_feats[idx[r]] for j, r in enumerate(syn_ref_ids) if r in idx])
    rs = rng.integers(0, len(real_feats), n_unmatched)
    ss = rng.integers(0, len(syn_feats), n_unmatched)
    real_syn = 1 - np.einsum("ij,ij->i", real_feats[rs], syn_feats[ss])
    r1, r2 = rng.integers(0, len(real_feats), (2, n_unmatched))
    keep = r1 != r2
    real_real = 1 - np.einsum("ij,ij->i", real_feats[r1[keep]], real_feats[r2[keep]])
    return {"matched": matched, "real_synthetic": real_syn, "real_real": real_real}


def nearest_real_neighbors(real_feats: np.ndarray, syn_feats: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """For each synthetic embedding, cosine similarity and index of its nearest real embeddings.

    High similarity (above ~0.95 for CLIP) flags near-duplicates of training images."""
    sims = syn_feats @ real_feats.T
    nn_idx = np.argsort(-sims, axis=1)[:, :k]
    nn_sim = np.take_along_axis(sims, nn_idx, axis=1)
    return nn_sim, nn_idx
