"""Sanity check utilities."""

from __future__ import annotations

import math
from typing import List

import numpy as np


def indexing_roundtrip_ok(crop_to_full: np.ndarray, Lf: int) -> bool:
    rng = np.random.default_rng(0)
    Lc = len(crop_to_full)
    M = rng.standard_normal((Lf, Lf)).astype(np.float32)
    crop = M[np.ix_(crop_to_full, crop_to_full)]
    back = np.zeros_like(M)
    back[np.ix_(crop_to_full, crop_to_full)] = crop
    return np.allclose(back[np.ix_(crop_to_full, crop_to_full)], crop)


def geometry_sanity(rep_xyz_crop: np.ndarray, mol_type_crop: np.ndarray) -> bool:
    is_prot = (mol_type_crop == 0)
    is_dna = (mol_type_crop == 1)
    if not (is_prot.any() and is_dna.any()):
        return False
    coords = rep_xyz_crop
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    pd = D[np.ix_(is_prot, is_dna)]
    pp = D[np.ix_(is_prot, is_prot)]
    dd = D[np.ix_(is_dna, is_dna)]
    med_pd = float(np.median(pd)) if pd.size else math.inf
    med_oth = float(np.median(np.concatenate([pp.flatten(), dd.flatten()])) if (pp.size or dd.size) else np.array([math.inf]))
    return med_pd < med_oth


def pd_correlation(edge_scores: np.ndarray, contact_probs: np.ndarray, aff_mask: np.ndarray) -> float:
    mask = aff_mask & np.isfinite(contact_probs)
    if not mask.any():
        return float("nan")
    x = edge_scores[mask].reshape(-1)
    y = contact_probs[mask].reshape(-1)
    # Spearman approximation via ranks
    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / counts
        return avg[inv]
    rx = _rank(x)
    ry = _rank(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9)
    ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float(np.mean(rx * ry))


__all__ = ["indexing_roundtrip_ok", "geometry_sanity", "pd_correlation"]


