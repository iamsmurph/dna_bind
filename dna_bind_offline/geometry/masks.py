"""Crop mask construction utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..models.types import CropMasks, TokenGeom


def make_crop_masks(crop_to_full: np.ndarray,
                    geom: TokenGeom,
                    contact_probs: Optional[np.ndarray] = None,
                    pae: Optional[np.ndarray] = None,
                    pde: Optional[np.ndarray] = None) -> CropMasks:
    crop_to_full = np.asarray(crop_to_full, dtype=np.int64)
    rep_xyz_crop = geom.rep_xyz[crop_to_full]
    mol_type_crop = geom.mol_type[crop_to_full]
    token_pad_mask_crop = geom.token_pad_mask[crop_to_full]

    is_prot = (mol_type_crop == 0)
    is_dna = (mol_type_crop == 1)
    pd_token_mask = (is_prot | is_dna)
    # One-sided protein→DNA mask to avoid double-counting pairs
    aff_mask = np.outer(is_prot, is_dna)
    aff_mask = aff_mask & np.outer(token_pad_mask_crop, token_pad_mask_crop)

    # Precompute PD indices for sparse ops
    ii, jj = np.where(aff_mask)
    pd_pairs = np.stack([ii.astype(np.int64), jj.astype(np.int64)], axis=1) if ii.size else np.zeros((0, 2), dtype=np.int64)
    Lc = rep_xyz_crop.shape[0]
    pd_flat_idx = (pd_pairs[:, 0] * int(Lc) + pd_pairs[:, 1]).astype(np.int64) if pd_pairs.size else np.zeros((0,), dtype=np.int64)

    # Downselect optional arrays if provided
    def _maybe_crop(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mat is None:
            return None
        if mat.shape[0] == rep_xyz_crop.shape[0]:
            return mat
        return mat[np.ix_(crop_to_full, crop_to_full)]

    if contact_probs is not None and contact_probs.shape[0] != rep_xyz_crop.shape[0]:
        contact_probs = _maybe_crop(contact_probs)
    if pae is not None and pae.shape[0] != rep_xyz_crop.shape[0]:
        pae = _maybe_crop(pae)
    if pde is not None and pde.shape[0] != rep_xyz_crop.shape[0]:
        pde = _maybe_crop(pde)

    return CropMasks(
        rep_xyz_crop=rep_xyz_crop,
        token_pad_mask_crop=token_pad_mask_crop,
        mol_type_crop=mol_type_crop,
        affinity_pair_mask=aff_mask,
        pd_token_mask=pd_token_mask,
        pd_pairs=pd_pairs,
        pd_flat_idx=pd_flat_idx,
    )


__all__ = ["make_crop_masks"]


