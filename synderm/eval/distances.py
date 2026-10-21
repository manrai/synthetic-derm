"""Distribution distances between embedding sets (FID-style, in any embedding space)."""
from __future__ import annotations

import numpy as np
from scipy import linalg


def frechet_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two feature sets (FID when Inception features)."""
    mu1, mu2 = feats_a.mean(0), feats_b.mean(0)
    s1, s2 = np.cov(feats_a, rowvar=False), np.cov(feats_b, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(s1 @ s2)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps) @ (s2 + eps))
    covmean = np.real(covmean)
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kernel_distance(feats_a: np.ndarray, feats_b: np.ndarray, subsets: int = 50, subset_size: int = 500,
                    degree: int = 3, gamma: float | None = None, coef0: float = 1.0, seed: int = 0) -> tuple[float, float]:
    """Kernel Inception Distance (unbiased MMD² with a polynomial kernel); returns mean and std over subsets."""
    rng = np.random.default_rng(seed)
    d = feats_a.shape[1]
    gamma = gamma or 1.0 / d
    m = min(subset_size, len(feats_a), len(feats_b))
    vals = []
    for _ in range(subsets):
        x = feats_a[rng.choice(len(feats_a), m, replace=False)]
        y = feats_b[rng.choice(len(feats_b), m, replace=False)]
        kxx = (gamma * x @ x.T + coef0) ** degree
        kyy = (gamma * y @ y.T + coef0) ** degree
        kxy = (gamma * x @ y.T + coef0) ** degree
        mmd = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))
