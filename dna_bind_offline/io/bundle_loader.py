"""Bundle loading and file discovery utilities."""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..models.types import CropBundle


def parse_prediction_dir_name(dir_path: str) -> Tuple[str, float, str]:
    name = os.path.basename(dir_path.rstrip("/"))
    parts = name.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Unexpected prediction directory name: {name}")
    uniprot = parts[0]
    try:
        score = float(parts[1])
    except Exception:
        score = float("nan")
    sequence = parts[2]
    return uniprot, score, sequence


def locate_file(pred_dir: str, stem: str, ext: str) -> Optional[str]:
    paths = glob.glob(os.path.join(pred_dir, f"{stem}_*.{ext}"))
    return paths[0] if paths else None


def locate_cif(pred_dir: str) -> Optional[str]:
    patterns = ("*_model_0.cif", "*model_0*.cif", "*.cif")
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(pred_dir, pat)))
        if hits:
            return hits[0]
    return None


def locate_any(pred_dir: str, stems: List[str], ext: str = "npz") -> Optional[str]:
    patterns: List[str] = []
    for s in stems:
        patterns.extend([f"{s}_*.{ext}", f"*{s}*.{ext}"])
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(pred_dir, pat)))
        if hits:
            return hits[0]
    return None


def load_npz_safely(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    try:
        data = np.load(path)
        # Try common keys, else first array
        for key in ("contact_probs", "arr_0"):
            if key in data:
                arr = data[key]
                break
        else:
            arr = data[data.files[0]]
        arr = np.asarray(arr)
        # Symmetrize if near-symmetric expected (warn-silent if notably asymmetric)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if not np.allclose(arr, arr.T, atol=1e-3):
                pass
            arr = 0.5 * (arr + arr.T)
        return arr
    except Exception:
        return None


def _torch_load_cpu(path: str) -> Dict[str, object]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj[0] if (0 in obj and isinstance(obj[0], dict)) else obj
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return obj[0]
    raise ValueError(f"Unexpected .pt structure in {path}: {type(obj)}")


def load_crop_bundle(pred_dir: str, device: Optional[torch.device] = None) -> CropBundle:
    """Load crop-aligned pair/single reps and optional signals from a prediction dir."""
    pt_paths = sorted(glob.glob(os.path.join(pred_dir, "cropped_embeddings_*.pt")))
    if not pt_paths:
        raise FileNotFoundError(f"No cropped_embeddings_*.pt in {pred_dir}")
    pt_path = pt_paths[0]
    inner = _torch_load_cpu(pt_path)

    # Pair reps
    z_key = "z_crop" if "z_crop" in inner else ("z" if "z" in inner else None)
    if z_key is None:
        raise KeyError("Missing z or z_crop in cropped embeddings")
    z = inner[z_key]
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    if z.ndim != 3 or z.shape[0] != z.shape[1]:
        raise ValueError(f"z must be square [Lc,Lc,C], got {tuple(z.shape)}")
    Lc, _, _ = z.shape

    # Indices mapping crop->full
    crop_to_full: Optional[np.ndarray] = None
    if "indices" in inner:
        idx = inner["indices"]
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        crop_to_full = np.asarray(idx, dtype=np.int64)
    elif "tok_idx_full" in inner:
        idx = inner["tok_idx_full"]
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        crop_to_full = np.asarray(idx, dtype=np.int64)
    elif "prot_idx_full" in inner and "dna_idx_full" in inner:
        p = inner["prot_idx_full"]
        d = inner["dna_idx_full"]
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        crop_to_full = np.asarray(list(p) + list(d), dtype=np.int64)
    if crop_to_full is None:
        raise KeyError("Missing indices mapping crop->full tokens")

    # Single-stream proxy
    c_single = 384
    if "s_crop" in inner:
        s_proxy = inner["s_crop"]
        if not isinstance(s_proxy, torch.Tensor):
            s_proxy = torch.as_tensor(s_proxy)
        if s_proxy.ndim == 3 and s_proxy.shape[0] == 1:
            s_proxy = s_proxy[0]
        if s_proxy.shape[0] != Lc:
            raise ValueError("s_crop not aligned to crop")
        c_single = s_proxy.shape[-1]
    elif "s_full" in inner:
        s_full = inner["s_full"]
        if not isinstance(s_full, torch.Tensor):
            s_full = torch.as_tensor(s_full)
        if s_full.ndim == 3 and s_full.shape[0] == 1:
            s_full = s_full[0]
        max_needed = int(crop_to_full.max()) + 1
        if s_full.shape[0] < max_needed:
            raise AssertionError("s_full length < max crop_to_full index + 1")
        s_proxy = s_full[crop_to_full]
        c_single = s_proxy.shape[-1]
    else:
        s_proxy = torch.zeros((Lc, c_single), dtype=z.dtype)

    # Optional signals (all expected LxL)
    contact = load_npz_safely(locate_any(pred_dir, ["contact_probs"]))
    pae = load_npz_safely(locate_any(pred_dir, ["pae", "full_pae", "pae_full"]))
    pde = load_npz_safely(locate_any(pred_dir, ["pde", "full_pde", "pde_full"]))
    tm_arr = load_npz_safely(locate_any(pred_dir, ["tm_expected_value"]))
    tm_expected: Optional[float] = None
    if tm_arr is not None:
        try:
            tm_expected = float(np.asarray(tm_arr).reshape(-1)[0])
        except Exception:
            tm_expected = None

    uniprot, score_from_dirname, sequence = parse_prediction_dir_name(pred_dir)
    meta: Dict[str, object] = {
        "pt_path": pt_path,
        "contact_path": locate_any(pred_dir, ["contact_probs"]),
        "pae_path": locate_any(pred_dir, ["pae", "full_pae", "pae_full"]),
        "pde_path": locate_any(pred_dir, ["pde", "full_pde", "pde_full"]),
        "tm_path": locate_any(pred_dir, ["tm_expected_value"]),
        "cif_path": locate_cif(pred_dir),
        "uniprot": uniprot,
        "sequence": sequence,
        "score_from_dirname": score_from_dirname,
    }

    # Device preference
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    if device is not None:
        z = z.to(device=device, dtype=torch.float32)
        s_proxy = s_proxy.to(device=device, dtype=z.dtype)

    return CropBundle(
        z=z,
        s_proxy=s_proxy,
        crop_to_full=crop_to_full,
        contact_probs=contact,
        pae=pae,
        pde=pde,
        tm_expected=tm_expected,
        meta=meta,
    )


__all__ = [
    "load_crop_bundle",
    "locate_file",
    "locate_cif",
    "locate_any",
    "parse_prediction_dir_name",
    "load_npz_safely",
]


