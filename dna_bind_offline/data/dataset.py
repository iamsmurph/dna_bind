"""Dataset and Sample for dna_bind_offline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import collections
import json
import os
import numpy as np
import torch


@dataclass
class Sample:
    z: torch.Tensor
    s_proxy: torch.Tensor
    dist_bins: np.ndarray | torch.Tensor
    masks: object
    y: torch.Tensor
    edge_weights: Optional[torch.Tensor]
    uniprot: str
    sequence: str
    prior_contact: Optional[torch.Tensor] = None
    prior_pae: Optional[torch.Tensor] = None
    prior_pde: Optional[torch.Tensor] = None

    def pin_memory(self):
        self.z = self.z.pin_memory()
        self.s_proxy = self.s_proxy.pin_memory()
        if isinstance(self.dist_bins, torch.Tensor):
            self.dist_bins = self.dist_bins.pin_memory()
        if isinstance(self.edge_weights, torch.Tensor):
            self.edge_weights = self.edge_weights.pin_memory()
        if isinstance(self.prior_contact, torch.Tensor):
            self.prior_contact = self.prior_contact.pin_memory()
        if isinstance(self.prior_pae, torch.Tensor):
            self.prior_pae = self.prior_pae.pin_memory()
        if isinstance(self.prior_pde, torch.Tensor):
            self.prior_pde = self.prior_pde.pin_memory()
        return self


__all__ = ["Sample"]


def downselect_pairwise(mat: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
    if mat is None:
        return None
    if mat.shape[0] == len(idx):
        return mat
    return mat[np.ix_(idx, idx)]


class AffinityDataset:
    def __init__(self,
                 pred_dirs: List[str],
                 labels: Dict[Tuple[str, str], float],
                 split: str,
                 normalize: str = "none",
                 train_stats: Optional[Dict[str, Tuple[float, float]]] = None,
                 dist_feats: str = "rbf",
                 rbf_centers: int = 64,
                 rbf_min: float = 2.0,
                 rbf_max: float = 22.0,
                 rbf_sigma: float = 1.0,
                 cache_dir: Optional[str] = None,
                 cache_format: str = "npz",
                 cache_in_mem: int = 0,
                 cache_z_in_mem: int = 0) -> None:
        self.pred_dirs = pred_dirs
        self.labels = labels
        self.split = split
        self.normalize = normalize
        self.train_stats = train_stats or {}
        self.dist_feats = dist_feats
        self.rbf_centers = int(rbf_centers)
        self.rbf_min = float(rbf_min)
        self.rbf_max = float(rbf_max)
        self.rbf_sigma = float(rbf_sigma)
        self.cache_dir = cache_dir or ""
        self.cache_format = cache_format
        self.cache_in_mem = int(cache_in_mem)
        self.cache_z_in_mem = int(cache_z_in_mem)
        self._mem: "collections.OrderedDict[str, Sample]" = collections.OrderedDict()

    @staticmethod
    def _to_pinned_half(x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.asarray(x))
        return t.to(torch.float16).pin_memory()

    @staticmethod
    def _to_pinned_long(x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x, dtype=np.int64)).pin_memory()

    def _cache_key_for_dir(self, d: str, bundle_meta: Dict[str, object]) -> str:
        import hashlib
        paths = {
            "pt": bundle_meta.get("pt_path", ""),
            "cif": bundle_meta.get("cif_path", ""),
            "contact": bundle_meta.get("contact_path", ""),
            "pae": bundle_meta.get("pae_path", ""),
            "pde": bundle_meta.get("pde_path", ""),
        }
        rec: Dict[str, Dict[str, float]] = {}
        for k, p in paths.items():
            if isinstance(p, str) and p and os.path.exists(p):
                try:
                    rec[k] = {"size": float(os.path.getsize(p)), "mtime": float(os.path.getmtime(p))}
                except Exception:
                    rec[k] = {"size": 0.0, "mtime": 0.0}
            else:
                rec[k] = {"size": 0.0, "mtime": 0.0}
        conf = {
            "paths": rec,
            "dist_mode": self.dist_feats,
            "rbf": {"B": self.rbf_centers, "min": self.rbf_min, "max": self.rbf_max, "sigma": self.rbf_sigma},
            "cache_version": 1,
        }
        s = json.dumps(conf, sort_keys=True).encode()
        return hashlib.sha1(s).hexdigest()

    def __len__(self) -> int:
        return len(self.pred_dirs)

    def __getitem__(self, i: int) -> Sample:
        from types import SimpleNamespace
        from ..io.bundle_loader import load_crop_bundle
        from ..geometry.distance_features import build_dist_bins
        d = self.pred_dirs[i]
        bundle = load_crop_bundle(d, device=torch.device("cpu"))

        use_cache = bool(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True) if use_cache else None
        key = self._cache_key_for_dir(d, bundle.meta)
        if self.cache_in_mem and key in self._mem:
            sample = self._mem.pop(key); self._mem[key] = sample
            return sample

        cache_path = os.path.join(self.cache_dir, f"{key}.npz") if use_cache else ""
        have_cache = use_cache and os.path.exists(cache_path)

        fast_cached = False
        masks = None
        contact_pd_np = None
        pae_pd_np = None
        pde_pd_np = None
        dist_pd_np = None
        if have_cache:
            data = np.load(cache_path)
            L_arr = data.get("L")
            pd_pairs_np = data.get("pd_pairs")
            pd_flat_idx_np = data.get("pd_flat_idx")
            if self.dist_feats == "rbf":
                dist_pd_np = data.get("dist_rbf_pd")
            contact_pd_np = data.get("contact_pd")
            pae_pd_np = data.get("pae_pd")
            pde_pd_np = data.get("pde_pd")
            if (self.dist_feats == "rbf") and isinstance(dist_pd_np, np.ndarray) and isinstance(pd_pairs_np, np.ndarray) and isinstance(L_arr, np.ndarray):
                Lc = int(L_arr.reshape(-1)[0])
                aff_mask = np.zeros((Lc, Lc), dtype=bool)
                if pd_pairs_np.size:
                    ii = pd_pairs_np[:, 0].astype(np.int64)
                    jj = pd_pairs_np[:, 1].astype(np.int64)
                    aff_mask[ii, jj] = True
                masks = SimpleNamespace(
                    rep_xyz_crop=np.zeros((Lc, 3), dtype=np.float32),
                    token_pad_mask_crop=np.ones((Lc,), dtype=bool),
                    mol_type_crop=np.zeros((Lc,), dtype=np.int8),
                    affinity_pair_mask=aff_mask,
                    pd_token_mask=np.ones((Lc,), dtype=bool),
                    pd_pairs=pd_pairs_np.astype(np.int64),
                    pd_flat_idx=(pd_flat_idx_np.astype(np.int64) if isinstance(pd_flat_idx_np, np.ndarray) and pd_flat_idx_np.size else (pd_pairs_np[:,0].astype(np.int64) * int(Lc) + pd_pairs_np[:,1].astype(np.int64))),
                )
                fast_cached = True

        if not fast_cached:
            from ..io.cif_parser import parse_cif_to_token_geom
            from ..geometry.masks import make_crop_masks
            cif = bundle.meta.get("cif_path")
            if not cif or not os.path.exists(cif):
                raise FileNotFoundError(f"Missing CIF for {d}")
            geom = parse_cif_to_token_geom(cif)
            masks = make_crop_masks(bundle.crop_to_full, geom, bundle.contact_probs, bundle.pae, bundle.pde)
            if not np.any(masks.affinity_pair_mask):
                y_dummy = torch.tensor(0.0, dtype=torch.float32)
                return Sample(z=bundle.z, s_proxy=bundle.s_proxy, dist_bins=build_dist_bins(masks.rep_xyz_crop),
                              masks=masks, y=y_dummy, edge_weights=None,
                              uniprot=bundle.meta.get("uniprot", ""), sequence=bundle.meta.get("sequence", ""))

        pd_pairs = masks.pd_pairs
        i_idx = pd_pairs[:, 0]
        j_idx = pd_pairs[:, 1]

        if not fast_cached:
            contact_c = downselect_pairwise(bundle.contact_probs, bundle.crop_to_full)
            pae_c = downselect_pairwise(bundle.pae, bundle.crop_to_full)
            pde_c = downselect_pairwise(bundle.pde, bundle.crop_to_full)
            contact_pd_np = contact_c[i_idx, j_idx].astype(np.float16) if isinstance(contact_c, np.ndarray) else None
            pae_pd_np = pae_c[i_idx, j_idx].astype(np.float16) if isinstance(pae_c, np.ndarray) else None
            pde_pd_np = pde_c[i_idx, j_idx].astype(np.float16) if isinstance(pde_c, np.ndarray) else None
            if self.dist_feats == "rbf":
                coords = np.asarray(masks.rep_xyz_crop, dtype=np.float32)
                D = np.linalg.norm(coords[i_idx] - coords[j_idx], axis=-1).astype(np.float32)
                centers = np.linspace(self.rbf_min, self.rbf_max, self.rbf_centers, dtype=np.float32)
                sigma = float(self.rbf_sigma)
                diff = (D[:, None] - centers[None, :]) / max(1e-8, sigma)
                phi = np.exp(-0.5 * diff * diff)
                phi /= (phi.sum(axis=1, keepdims=True) + 1e-8)
                dist_pd_np = phi.astype(np.float16)
            else:
                dist_pd_np = None
            if use_cache:
                tmp = cache_path + ".tmp"
                arrs = {
                    "L": np.array([masks.rep_xyz_crop.shape[0]], dtype=np.int32),
                    "pd_pairs": pd_pairs.astype(np.int32),
                    "pd_flat_idx": masks.pd_flat_idx.astype(np.int64),
                    "contact_pd": contact_pd_np if contact_pd_np is not None else np.array([], dtype=np.float16),
                    "pae_pd": pae_pd_np if pae_pd_np is not None else np.array([], dtype=np.float16),
                    "pde_pd": pde_pd_np if pae_pd_np is not None else np.array([], dtype=np.float16),
                }
                if dist_pd_np is not None:
                    arrs["dist_rbf_pd"] = dist_pd_np
                # Write via file handle so numpy doesn't append an extra .npz
                with open(tmp, "wb") as f:
                    np.savez_compressed(f, **arrs)
                os.replace(tmp, cache_path)

        if self.dist_feats == "rbf" and isinstance(dist_pd_np, np.ndarray):
            dist_bins_t = self._to_pinned_half(dist_pd_np)
        elif self.dist_feats == "rbf" and not isinstance(dist_pd_np, np.ndarray):
            coords = torch.from_numpy(masks.rep_xyz_crop.astype(np.float32))
            i_t = torch.from_numpy(i_idx.astype(np.int64))
            j_t = torch.from_numpy(j_idx.astype(np.int64))
            D = torch.linalg.norm(coords.index_select(0, i_t) - coords.index_select(0, j_t), dim=-1)
            centers = torch.linspace(float(self.rbf_min), float(self.rbf_max), steps=int(self.rbf_centers), dtype=D.dtype)
            phi = torch.exp(-0.5 * ((D[:, None] - centers[None, :]) / max(1e-8, float(self.rbf_sigma))) ** 2)
            phi = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
            dist_bins_t = phi.to(torch.float16).pin_memory()
        else:
            from ..geometry.distance_features import build_dist_bins as _build_bins
            dist_bins_np = _build_bins(masks.rep_xyz_crop)
            dist_bins_t = torch.from_numpy(dist_bins_np).to(torch.float16).pin_memory()

        c_t = self._to_pinned_half(contact_pd_np) if isinstance(contact_pd_np, np.ndarray) and contact_pd_np.size else None
        pae_t = self._to_pinned_half(pae_pd_np) if isinstance(pae_pd_np, np.ndarray) and pae_pd_np.size else None
        pde_t = self._to_pinned_half(pde_pd_np) if isinstance(pde_pd_np, np.ndarray) and pde_pd_np.size else None

        uniprot = bundle.meta["uniprot"]
        sequence = bundle.meta["sequence"]
        key2 = (uniprot, sequence)
        if key2 not in self.labels:
            # Allow single trailing 'T' discrepancy without warning
            if sequence.endswith("T"):
                alt_key = (uniprot, sequence[:-1])
                if alt_key in self.labels:
                    key2 = alt_key
                else:
                    raise KeyError(f"Label not found for {(uniprot, sequence)}")
            else:
                raise KeyError(f"Label not found for {key2}")
        y_val = float(self.labels[key2])
        if self.split == "train" and self.normalize == "zscore_per_tf" and uniprot in (self.train_stats or {}):
            mu, sigma = (self.train_stats or {}).get(uniprot, (0.0, 1.0))
            y_val = (y_val - mu) / max(sigma, 1e-8)
        y = torch.tensor(y_val, dtype=torch.float32)

        sample = Sample(z=bundle.z,
                        s_proxy=bundle.s_proxy,
                        dist_bins=dist_bins_t,
                        masks=masks,
                        y=y,
                        edge_weights=None,
                        prior_contact=c_t,
                        prior_pae=pae_t,
                        prior_pde=pde_t,
                        uniprot=uniprot,
                        sequence=sequence)
        if self.cache_in_mem and use_cache:
            self._mem[key] = sample
            if len(self._mem) > self.cache_in_mem:
                self._mem.popitem(last=False)
        return sample


__all__ = ["AffinityDataset", "Sample"]


