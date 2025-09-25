"""Distance feature construction utilities."""

from __future__ import annotations

import numpy as np
import torch


def build_dist_bins(rep_xyz_crop: np.ndarray,
                    bin_edges: np.ndarray = np.linspace(2.0, 22.0, 64)) -> np.ndarray:
    coords = np.asarray(rep_xyz_crop, dtype=np.float32)
    L = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.maximum(1e-9, np.sum(diff * diff, axis=-1))).astype(np.float32)  # (L,L)
    B = int(len(bin_edges))
    mids = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    cut = np.concatenate([[bin_edges[0] - 1e9], mids, [bin_edges[-1] + 1e9]])
    idx = np.digitize(D, cut) - 1
    idx = np.clip(idx, 0, B - 1)
    bins = np.zeros((L, L, B), dtype=np.float32)
    rows, cols = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    bins[rows, cols, idx] = 1.0
    return bins


def build_dist_rbf(rep_xyz: np.ndarray | torch.Tensor,
                   n_centers: int = 64,
                   dmin: float = 2.0,
                   dmax: float = 22.0,
                   sigma: float = 1.0) -> torch.Tensor:
    if isinstance(rep_xyz, np.ndarray):
        coords = torch.from_numpy(rep_xyz).float()
    else:
        coords = rep_xyz.to(dtype=torch.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    D = torch.linalg.norm(diff, dim=-1)
    centers = torch.linspace(float(dmin), float(dmax), steps=int(n_centers), device=D.device, dtype=D.dtype)
    diff_cent = D[..., None] - centers[None, None, :]
    phi = torch.exp(-0.5 * (diff_cent / max(1e-8, float(sigma))) ** 2)
    phi = phi / (phi.sum(dim=-1, keepdim=True) + 1e-8)
    return phi


__all__ = ["build_dist_bins", "build_dist_rbf"]


