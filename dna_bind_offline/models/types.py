"""Type dataclasses for dna_bind_offline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class CropBundle:
    z: torch.FloatTensor
    s_proxy: torch.FloatTensor
    crop_to_full: np.ndarray
    contact_probs: Optional[np.ndarray]
    pae: Optional[np.ndarray]
    pde: Optional[np.ndarray]
    tm_expected: Optional[float]
    meta: dict


@dataclass
class TokenGeom:
    rep_xyz: np.ndarray           # [Lf, 3]
    mol_type: np.ndarray          # [Lf] int8: 0=PROTEIN,1=DNA,2=OTHER
    token_pad_mask: np.ndarray    # [Lf] bool
    token_meta: List[dict]


@dataclass
class CropMasks:
    rep_xyz_crop: np.ndarray        # [Lc, 3]
    token_pad_mask_crop: np.ndarray # [Lc]
    mol_type_crop: np.ndarray       # [Lc]
    affinity_pair_mask: np.ndarray  # [Lc, Lc]
    pd_token_mask: np.ndarray       # [Lc]
    pd_pairs: np.ndarray            # [K, 2] int64 indices (i,j) where protein->DNA
    pd_flat_idx: np.ndarray         # [K] int64 flattened indices for [Lc*Lc]


__all__ = ["CropBundle", "TokenGeom", "CropMasks"]


