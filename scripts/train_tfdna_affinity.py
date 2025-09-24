#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob as globlib
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, DefaultDict

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import hashlib
import json
import collections

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from scripts.tfdna_affinity import (
    TFAffinityRegressor,
    load_crop_bundle,
    parse_cif_to_token_geom,
    make_crop_masks,
    build_dist_bins,
    build_dist_rbf,
    compute_edge_weights,
    parse_prediction_dir_name,
)


def read_labels(csv_path: str, seq_col: str = "nt") -> Dict[Tuple[str, str], float]:
    mapping: Dict[Tuple[str, str], float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot = row["uniprot"].strip()
            seq = row.get(seq_col, row.get("sequence", "")).strip()
            y = float(row["intensity_log1p"])
            mapping[(uniprot, seq)] = y
    return mapping


def downselect_pairwise(mat: Optional[np.ndarray], idx: np.ndarray) -> Optional[np.ndarray]:
    if mat is None:
        return None
    if mat.shape[0] == len(idx):
        return mat
    return mat[np.ix_(idx, idx)]


@dataclass
class Sample:
    z: torch.Tensor
    s_proxy: torch.Tensor
    dist_bins: np.ndarray
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
        if isinstance(self.edge_weights, torch.Tensor):
            self.edge_weights = self.edge_weights.pin_memory()
        if isinstance(self.prior_contact, torch.Tensor):
            self.prior_contact = self.prior_contact.pin_memory()
        if isinstance(self.prior_pae, torch.Tensor):
            self.prior_pae = self.prior_pae.pin_memory()
        if isinstance(self.prior_pde, torch.Tensor):
            self.prior_pde = self.prior_pde.pin_memory()
        return self


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
        d = self.pred_dirs[i]
        # Keep tensors on CPU; Lightning will handle device placement in the module
        bundle = load_crop_bundle(d, device=torch.device("cpu"))
        cif = bundle.meta.get("cif_path")
        if not cif or not os.path.exists(cif):
            raise FileNotFoundError(f"Missing CIF for {d}")
        geom = parse_cif_to_token_geom(cif)

        masks = make_crop_masks(bundle.crop_to_full, geom, bundle.contact_probs, bundle.pae, bundle.pde)
        # masks should have at least one PD edge; datasets are prefiltered in setup
        if not np.any(masks.affinity_pair_mask):
            # Return a dummy but consistent sample to avoid DataLoader errors (shouldn't occur)
            # Use zero weights and zero label
            y_dummy = torch.tensor(0.0, dtype=torch.float32)
            return Sample(z=bundle.z, s_proxy=bundle.s_proxy, dist_bins=build_dist_bins(masks.rep_xyz_crop),
                          masks=masks, y=y_dummy, edge_weights=None,
                          uniprot=bundle.meta.get("uniprot", ""), sequence=bundle.meta.get("sequence", ""))
        # Optional disk+mem cache for PD-only features
        use_cache = bool(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True) if use_cache else None
        key = self._cache_key_for_dir(d, bundle.meta)
        if self.cache_in_mem and key in self._mem:
            sample = self._mem.pop(key); self._mem[key] = sample
            return sample

        cache_path = os.path.join(self.cache_dir, f"{key}.npz") if use_cache else ""
        have_cache = use_cache and os.path.exists(cache_path)
        pd_pairs = masks.pd_pairs  # [K,2]
        i_idx = pd_pairs[:, 0]
        j_idx = pd_pairs[:, 1]

        if have_cache:
            data = np.load(cache_path)
            dist_pd_np = data.get("dist_rbf_pd") if (self.dist_feats == "rbf") else None
            contact_pd_np = data.get("contact_pd")
            pae_pd_np = data.get("pae_pd")
            pde_pd_np = data.get("pde_pd")
        else:
            # Build PD-only priors
            contact_c = downselect_pairwise(bundle.contact_probs, bundle.crop_to_full)
            pae_c = downselect_pairwise(bundle.pae, bundle.crop_to_full)
            pde_c = downselect_pairwise(bundle.pde, bundle.crop_to_full)
            contact_pd_np = contact_c[i_idx, j_idx].astype(np.float16) if isinstance(contact_c, np.ndarray) else None
            pae_pd_np = pae_c[i_idx, j_idx].astype(np.float16) if isinstance(pae_c, np.ndarray) else None
            pde_pd_np = pde_c[i_idx, j_idx].astype(np.float16) if isinstance(pde_c, np.ndarray) else None
            # PD-only dist features
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
                    "pde_pd": pde_pd_np if pde_pd_np is not None else np.array([], dtype=np.float16),
                }
                if dist_pd_np is not None:
                    arrs["dist_rbf_pd"] = dist_pd_np
                np.savez_compressed(tmp, **arrs)
                os.replace(tmp, cache_path)

        # Build tensors, pinned where helpful
        if self.dist_feats == "rbf" and (use_cache and have_cache):
            dist_bins_t = self._to_pinned_half(dist_pd_np)
        elif self.dist_feats == "rbf" and not use_cache:
            # compute on the fly to torch
            coords = torch.from_numpy(masks.rep_xyz_crop.astype(np.float32))
            i_t = torch.from_numpy(i_idx.astype(np.int64))
            j_t = torch.from_numpy(j_idx.astype(np.int64))
            D = torch.linalg.norm(coords.index_select(0, i_t) - coords.index_select(0, j_t), dim=-1)
            centers = torch.linspace(float(self.rbf_min), float(self.rbf_max), steps=int(self.rbf_centers), dtype=D.dtype)
            phi = torch.exp(-0.5 * ((D[:, None] - centers[None, :]) / max(1e-8, float(self.rbf_sigma))) ** 2)
            phi = phi / (phi.sum(dim=1, keepdim=True) + 1e-8)
            dist_bins_t = phi.to(torch.float16).pin_memory()
        else:
            # fallback to dense bins if requested; keep original behavior
            dist_bins_np = build_dist_bins(masks.rep_xyz_crop)
            dist_bins_t = torch.from_numpy(dist_bins_np)

        # Priors as pinned half vectors when available
        c_t = self._to_pinned_half(contact_pd_np) if isinstance(contact_pd_np, np.ndarray) and contact_pd_np.size else None
        pae_t = self._to_pinned_half(pae_pd_np) if isinstance(pae_pd_np, np.ndarray) and pae_pd_np.size else None
        pde_t = self._to_pinned_half(pde_pd_np) if isinstance(pde_pd_np, np.ndarray) and pde_pd_np.size else None

        uniprot = bundle.meta["uniprot"]
        sequence = bundle.meta["sequence"]
        key = (uniprot, sequence)
        if key not in self.labels:
            raise KeyError(f"Label not found for {key}")
        y_val = float(self.labels[key])
        # Apply normalization for training split only if requested
        if self.split == "train" and self.normalize == "zscore_per_tf" and uniprot in self.train_stats:
            mu, sigma = self.train_stats[uniprot]
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


def single_collate(batch: List[Sample]) -> Sample:
    # Variable-sized samples; use batch_size=1 and pass through
    return batch[0]


class AffinityDataModule(pl.LightningDataModule):
    def __init__(self,
                 labels_csv: str,
                 pred_glob: str,
                 seq_col: str = "nt",
                 val_frac: float = 0.1,
                 seed: int = 0,
                 num_workers: int = 0,
                 batch_size: int = 1,
                 split_by: str = "tf",
                 normalize: str = "none",
                 test_glob: Optional[str] = None,
                 test_labels_csv: Optional[str] = None,
                 dist_feats: str = "rbf",
                 rbf_centers: int = 64,
                 rbf_min: float = 2.0,
                 rbf_max: float = 22.0,
                 rbf_sigma: float = 1.0,
                 cache_dir: Optional[str] = None,
                 cache_format: str = "npz",
                 cache_in_mem: int = 0,
                 cache_z_in_mem: int = 0,
                 pin_memory: bool = True,
                 prefetch_factor: int = 4) -> None:
        super().__init__()
        self.labels_csv = labels_csv
        self.pred_glob = pred_glob
        self.seq_col = seq_col
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.batch_size = int(batch_size)
        self.split_by = split_by
        self.normalize = normalize
        self.test_glob = test_glob
        self.test_labels_csv = test_labels_csv
        self.dist_feats = dist_feats
        self.rbf_centers = int(rbf_centers)
        self.rbf_min = float(rbf_min)
        self.rbf_max = float(rbf_max)
        self.rbf_sigma = float(rbf_sigma)
        self.cache_dir = cache_dir or ""
        self.cache_format = cache_format
        self.cache_in_mem = int(cache_in_mem)
        self.cache_z_in_mem = int(cache_z_in_mem)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.labels: Dict[Tuple[str, str], float] = {}
        self.train_ds: Optional[AffinityDataset] = None
        self.val_ds: Optional[AffinityDataset] = None
        self.test_ds: Optional[AffinityDataset] = None
        self.train_stats: Dict[str, Tuple[float, float]] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        # Load labels
        self.labels = read_labels(self.labels_csv, seq_col=self.seq_col)
        # Discover and prefilter prediction dirs
        all_dirs = sorted(globlib.glob(os.path.join(self.pred_glob)))
        if not all_dirs:
            raise RuntimeError(f"No prediction dirs matched {self.pred_glob}")

        # Prefilter invalid PD-edge dirs
        valid_dirs: List[str] = []
        for d in all_dirs:
            try:
                bundle = load_crop_bundle(d, device=torch.device("cpu"))
                cif = bundle.meta.get("cif_path")
                if not cif or not os.path.exists(cif):
                    continue
                geom = parse_cif_to_token_geom(cif)
                masks = make_crop_masks(bundle.crop_to_full, geom, bundle.contact_probs, bundle.pae, bundle.pde)
                if np.any(masks.affinity_pair_mask):
                    valid_dirs.append(d)
            except Exception:
                continue

        if not valid_dirs:
            raise RuntimeError("No valid prediction dirs after prefiltering for PD edges")

        # Group by TF (uniprot)
        if self.split_by not in ("random", "tf"):
            self.split_by = "tf"
        rnd = random.Random(self.seed)

        if self.split_by == "random":
            rnd.shuffle(valid_dirs)
            n_val = max(1, int(len(valid_dirs) * self.val_frac))
            val_dirs = valid_dirs[:n_val]
            train_dirs = valid_dirs[n_val:]
        else:
            # group by uniprot
            tf_to_dirs = defaultdict(list)
            for d in valid_dirs:
                try:
                    uniprot, _, seq = parse_prediction_dir_name(d)
                except Exception:
                    # Fallback via bundle meta
                    try:
                        bundle = load_crop_bundle(d, device=torch.device("cpu"))
                        uniprot = bundle.meta.get("uniprot", "")
                    except Exception:
                        uniprot = ""
                tf_to_dirs[uniprot].append(d)
            tf_ids = list(tf_to_dirs.keys())
            rnd.shuffle(tf_ids)
            n_val_tf = max(1, int(len(tf_ids) * self.val_frac))
            val_tfs = set(tf_ids[:n_val_tf])
            train_tfs = set(tf_ids[n_val_tf:])
            train_dirs = [d for tf in train_tfs for d in tf_to_dirs[tf]]
            val_dirs = [d for tf in val_tfs for d in tf_to_dirs[tf]]

        # Compute train normalization stats per TF if requested
        self.train_stats = {}
        if self.normalize == "zscore_per_tf":
            tf_to_vals = defaultdict(list)
            for d in train_dirs:
                try:
                    uniprot, _, seq = parse_prediction_dir_name(d)
                except Exception:
                    # fallback: parse from dir via bundle
                    try:
                        bundle = load_crop_bundle(d, device=torch.device("cpu"))
                        uniprot = bundle.meta.get("uniprot", "")
                        seq = bundle.meta.get("sequence", "")
                    except Exception:
                        continue
                key = (uniprot, seq)
                if key in self.labels:
                    tf_to_vals[uniprot].append(float(self.labels[key]))
            for tf, vals in tf_to_vals.items():
                if len(vals) == 0:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                if not np.isfinite(sigma) or sigma < 1e-8:
                    sigma = 1e-8
                self.train_stats[tf] = (mu, sigma)

        # Datasets
        self.train_ds = AffinityDataset(train_dirs, self.labels, split="train", normalize=self.normalize, train_stats=self.train_stats,
                                        dist_feats=self.dist_feats, rbf_centers=self.rbf_centers, rbf_min=self.rbf_min, rbf_max=self.rbf_max, rbf_sigma=self.rbf_sigma,
                                        cache_dir=self.cache_dir, cache_format=self.cache_format, cache_in_mem=self.cache_in_mem, cache_z_in_mem=self.cache_z_in_mem)
        self.val_ds = AffinityDataset(val_dirs, self.labels, split="val", normalize="none", train_stats=self.train_stats,
                                      dist_feats=self.dist_feats, rbf_centers=self.rbf_centers, rbf_min=self.rbf_min, rbf_max=self.rbf_max, rbf_sigma=self.rbf_sigma,
                                      cache_dir=self.cache_dir, cache_format=self.cache_format, cache_in_mem=self.cache_in_mem, cache_z_in_mem=self.cache_z_in_mem)

        # Optional test set (unseen TFs default)
        self.test_ds = None
        if self.test_glob:
            test_dirs_all = sorted(globlib.glob(os.path.join(self.test_glob)))
            if test_dirs_all:
                # If test labels CSV provided, override labels for test
                test_labels = self.labels if not self.test_labels_csv else read_labels(self.test_labels_csv, seq_col=self.seq_col)
                train_tfs_set = set(self.train_stats.keys()) if self.train_stats else set()
                # train_stats keys are TFs; else infer from train_dirs
                if not train_tfs_set:
                    train_tfs_set = set()
                    for d in train_dirs:
                        try:
                            tf_id, _, _ = parse_prediction_dir_name(d)
                        except Exception:
                            tf_id = ""
                        train_tfs_set.add(tf_id)
                test_dirs = []
                for d in test_dirs_all:
                    try:
                        tf_id, _, _ = parse_prediction_dir_name(d)
                    except Exception:
                        tf_id = ""
                    if tf_id and tf_id not in train_tfs_set:
                        test_dirs.append(d)
                if test_dirs:
                    self.test_ds = AffinityDataset(test_dirs, test_labels, split="test", normalize="none", train_stats=self.train_stats,
                                                   dist_feats=self.dist_feats, rbf_centers=self.rbf_centers, rbf_min=self.rbf_min, rbf_max=self.rbf_max, rbf_sigma=self.rbf_sigma,
                                                   cache_dir=self.cache_dir, cache_format=self.cache_format, cache_in_mem=self.cache_in_mem, cache_z_in_mem=self.cache_z_in_mem)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.train_ds is not None
        kwargs = dict(batch_size=self.batch_size,
                      shuffle=True,
                      num_workers=self.num_workers,
                      collate_fn=single_collate,
                      pin_memory=self.pin_memory,
                      persistent_workers=self.num_workers > 0)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.val_ds is not None
        kwargs = dict(batch_size=self.batch_size,
                      shuffle=False,
                      num_workers=self.num_workers,
                      collate_fn=single_collate,
                      pin_memory=self.pin_memory,
                      persistent_workers=self.num_workers > 0)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_ds is None:
            return None
        kwargs = dict(batch_size=self.batch_size,
                      shuffle=False,
                      num_workers=self.num_workers,
                      collate_fn=single_collate,
                      pin_memory=self.pin_memory,
                      persistent_workers=self.num_workers > 0)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.test_ds, **kwargs)


class AffinityLightningModule(pl.LightningModule):
    def __init__(self,
                 c_pair: int,
                 c_single: int,
                 n_bins: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 corr_w: float = 0.05,
                 use_soft_pool: bool = True,
                 pooling: str = "attention",
                 attn_hidden: int = 128,
                 attn_dropout: float = 0.10,
                 edge_dropout: float = 0.10,
                 pool_temp: float = 4.0,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TFAffinityRegressor(
            c_pair=c_pair,
            c_single=c_single,
            n_bins=n_bins,
            use_soft_pool=use_soft_pool,
            pooling=pooling,
            attn_hidden=attn_hidden,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
            pool_temp=pool_temp,
            noise_std=noise_std,
            prior_w_contact=prior_w_contact,
            prior_w_pae=prior_w_pae,
            prior_w_pde=prior_w_pde,
            prior_eps=prior_eps,
        )
        self.loss_mse = nn.MSELoss()
        self.val_store: Dict[str, List[Tuple[float, float]]] = {}
        self.test_store: Dict[str, List[Tuple[float, float]]] = {}
        self.normalize: str = "none"
        self.train_stats: Dict[str, Tuple[float, float]] = {}

    @staticmethod
    def corr_penalty(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        yh = (yhat - yhat.mean()) / (yhat.std() + 1e-6)
        yt = (y - y.mean()) / (y.std() + 1e-6)
        return 1.0 - (yh * yt).mean()

    def forward(self, sample: Sample) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.device
        z = sample.z.to(device, non_blocking=True)
        s = sample.s_proxy.to(device, non_blocking=True)
        dist_bins = sample.dist_bins
        if isinstance(dist_bins, np.ndarray):
            dist_bins = torch.from_numpy(dist_bins)
        dist_bins = dist_bins.to(device=device, dtype=z.dtype, non_blocking=True)
        edge_w = sample.edge_weights.to(device, non_blocking=True) if sample.edge_weights is not None else None
        prior_c = sample.prior_contact.to(device) if isinstance(sample.prior_contact, torch.Tensor) else None
        prior_pae = sample.prior_pae.to(device) if isinstance(sample.prior_pae, torch.Tensor) else None
        prior_pde = sample.prior_pde.to(device) if isinstance(sample.prior_pde, torch.Tensor) else None
        y_hat, out = self.model(z, s, dist_bins, sample.masks, edge_weights=edge_w,
                                prior_contact=prior_c, prior_pae=prior_pae, prior_pde=prior_pde)
        return y_hat, out

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        y = batch.y.to(self.device).reshape(1)
        y_hat, _ = self.forward(batch)
        loss = self.loss_mse(y_hat, y)
        # metrics
        with torch.no_grad():
            mse = self.loss_mse(y_hat, y)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        y = batch.y.to(self.device).reshape(1)
        y_hat, out = self.forward(batch)
        # De-normalize prediction for metrics if training used z-score
        yhat_cpu = y_hat.detach().cpu().reshape(()).item()
        uniprot = batch.uniprot
        if self.normalize == "zscore_per_tf" and uniprot in self.train_stats:
            mu, sigma = self.train_stats[uniprot]
            yhat_raw = yhat_cpu * sigma + mu
        else:
            yhat_raw = yhat_cpu
        # y provided for val/test is raw
        y_raw = batch.y.detach().cpu().reshape(()).item() if self.normalize == "none" else batch.y.detach().cpu().reshape(()).item()
        self.val_store.setdefault(uniprot, []).append((y_raw, yhat_raw))
        # still log val MSE in raw space (NMSE computed at epoch end)
        mse = torch.tensor((yhat_raw - y_raw) ** 2, dtype=torch.float32)
        self.log("val/mse_point", mse, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
        # Optional attention stats
        try:
            if isinstance(out, dict):
                if ("attn_weights" in out):
                    w = out["attn_weights"]  # [L,L]
                    mask_np = batch.masks.affinity_pair_mask
                    mask_t = torch.from_numpy(mask_np).to(w.device)
                    w_valid = w[mask_t]
                    if w_valid.numel() > 0:
                        ent = -torch.sum(w_valid * torch.log(w_valid.clamp_min(1e-12)))
                        self.log("val/attn_entropy", ent, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
                        k = min(32, w_valid.numel())
                        topk_mass = w_valid.topk(k=k).values.sum()
                        self.log("val/attn_topk32_mass", topk_mass, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
                        coords = torch.from_numpy(batch.masks.rep_xyz_crop).to(dtype=w.dtype, device=w.device)
                        D = torch.linalg.norm(coords[:, None, :] - coords[None, :, :], dim=-1)
                        near = (D < 8.0) & mask_t
                        if near.any():
                            near_mass = w[near].sum()
                            self.log("val/attn_mass_lt8A", near_mass, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
                elif ("attn_weights_pd" in out):
                    w_pd = out["attn_weights_pd"]  # [K]
                    if w_pd.numel() > 0:
                        ent = -torch.sum(w_pd * torch.log(w_pd.clamp_min(1e-12)))
                        self.log("val/attn_entropy", ent, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
                        k = min(32, w_pd.numel())
                        topk_mass = w_pd.topk(k=k).values.sum()
                        self.log("val/attn_topk32_mass", topk_mass, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
                        # <8A mass using PD pairs only
                        coords = torch.from_numpy(batch.masks.rep_xyz_crop).to(dtype=w_pd.dtype)
                        ij = torch.from_numpy(batch.masks.pd_pairs).to(dtype=torch.long)
                        i_idx, j_idx = ij[:,0], ij[:,1]
                        D = torch.linalg.norm(coords.index_select(0, i_idx) - coords.index_select(0, j_idx), dim=-1)
                        near_mask = (D < 8.0)
                        if near_mask.any():
                            near_mass = w_pd[near_mask].sum()
                            self.log("val/attn_mass_lt8A", near_mass, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)
        except Exception:
            pass

    def on_validation_epoch_start(self) -> None:
        self.val_store = {}

    @staticmethod
    def _pearson(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 2:
            return float("nan")
        x = np.asarray(xs, dtype=np.float64)
        y = np.asarray(ys, dtype=np.float64)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x.std() + 1e-9) * (y.std() + 1e-9)
        if not np.isfinite(denom) or denom < 1e-12:
            return float("nan")
        return float(np.mean((x / (x.std() + 1e-9)) * (y / (y.std() + 1e-9))))

    @staticmethod
    def _spearman(xs: List[float], ys: List[float]) -> float:
        if len(xs) < 2:
            return float("nan")
        def rank(v: np.ndarray) -> np.ndarray:
            order = np.argsort(v)
            ranks = np.empty_like(order, dtype=np.float64)
            ranks[order] = np.arange(len(v), dtype=np.float64)
            _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
            sums = np.bincount(inv, ranks)
            avg = sums / counts
            return avg[inv]
        rx = rank(np.asarray(xs, dtype=np.float64))
        ry = rank(np.asarray(ys, dtype=np.float64))
        rx = (rx - rx.mean()) / (rx.std() + 1e-9)
        ry = (ry - ry.mean()) / (ry.std() + 1e-9)
        return float(np.mean(rx * ry))

    def on_validation_epoch_end(self) -> None:
        # Aggregate per-TF metrics
        per_tf_r: List[float] = []
        per_tf_s: List[float] = []
        per_tf_nmse: List[float] = []
        for tf, pairs in self.val_store.items():
            ys, yh = zip(*pairs)
            ys_arr = np.asarray(ys, dtype=np.float64)
            yh_arr = np.asarray(yh, dtype=np.float64)
            r = self._pearson(ys_arr.tolist(), yh_arr.tolist())
            s = self._spearman(ys_arr.tolist(), yh_arr.tolist())
            var_y = float(np.var(ys_arr)) if len(ys_arr) > 1 else 0.0
            mse = float(np.mean((yh_arr - ys_arr) ** 2))
            nmse = float(mse / max(var_y, 1e-8)) if var_y > 0 else float("nan")
            if np.isfinite(r):
                per_tf_r.append(r)
            if np.isfinite(s):
                per_tf_s.append(s)
            if np.isfinite(nmse):
                per_tf_nmse.append(nmse)
        r_mean = float(np.mean(per_tf_r)) if per_tf_r else float("nan")
        s_mean = float(np.mean(per_tf_s)) if per_tf_s else float("nan")
        nmse_mean = float(np.mean(per_tf_nmse)) if per_tf_nmse else float("nan")
        self.log("val/r_by_tf_mean", torch.tensor(r_mean), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/spearman_by_tf_mean", torch.tensor(s_mean), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/nmse_by_tf_mean", torch.tensor(nmse_mean), on_step=False, on_epoch=True, prog_bar=False)
        # also keep simple count
        self.log("val/num_tfs", torch.tensor(len(self.val_store)), on_step=False, on_epoch=True, prog_bar=False)

    def test_step(self, batch: Sample, batch_idx: int) -> None:
        y = batch.y.to(self.device).reshape(1)
        y_hat, _ = self.forward(batch)
        yhat_cpu = y_hat.detach().cpu().reshape(()).item()
        uniprot = batch.uniprot
        if self.normalize == "zscore_per_tf" and uniprot in self.train_stats:
            mu, sigma = self.train_stats[uniprot]
            yhat_raw = yhat_cpu * sigma + mu
        else:
            yhat_raw = yhat_cpu
        y_raw = batch.y.detach().cpu().reshape(()).item()
        self.test_store.setdefault(uniprot, []).append((y_raw, yhat_raw))

    def on_test_epoch_start(self) -> None:
        self.test_store = {}

    def on_test_epoch_end(self) -> None:
        per_tf_r: List[float] = []
        per_tf_s: List[float] = []
        per_tf_nmse: List[float] = []
        for tf, pairs in self.test_store.items():
            ys, yh = zip(*pairs)
            ys_arr = np.asarray(ys, dtype=np.float64)
            yh_arr = np.asarray(yh, dtype=np.float64)
            r = self._pearson(ys_arr.tolist(), yh_arr.tolist())
            s = self._spearman(ys_arr.tolist(), yh_arr.tolist())
            var_y = float(np.var(ys_arr)) if len(ys_arr) > 1 else 0.0
            mse = float(np.mean((yh_arr - ys_arr) ** 2))
            nmse = float(mse / max(var_y, 1e-8)) if var_y > 0 else float("nan")
            if np.isfinite(r):
                per_tf_r.append(r)
            if np.isfinite(s):
                per_tf_s.append(s)
            if np.isfinite(nmse):
                per_tf_nmse.append(nmse)
        r_mean = float(np.mean(per_tf_r)) if per_tf_r else float("nan")
        s_mean = float(np.mean(per_tf_s)) if per_tf_s else float("nan")
        nmse_mean = float(np.mean(per_tf_nmse)) if per_tf_nmse else float("nan")
        self.log("test/r_by_tf_mean", torch.tensor(r_mean), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/spearman_by_tf_mean", torch.tensor(s_mean), on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/nmse_by_tf_mean", torch.tensor(nmse_mean), on_step=False, on_epoch=True, prog_bar=False)

    def configure_optimizers(self):
        use_fused = torch.cuda.is_available()
        try:
            major, _ = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            use_fused = use_fused and major >= 8
        except Exception:
            use_fused = False
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, fused=use_fused)
        return opt

def infer_dims_from_datamodule(dm: AffinityDataModule) -> Tuple[int, int, int]:
    # Ensure setup has run
    if dm.train_ds is None:
        dm.setup()
    assert dm.train_ds is not None
    probe = dm.train_ds[0]
    c_pair = int(probe.z.shape[-1])
    c_single = int(probe.s_proxy.shape[-1])
    n_bins = int(probe.dist_bins.shape[-1])
    return c_pair, c_single, n_bins

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True, help="path to labels CSV with columns: uniprot, nt (or sequence), intensity_log1p")
    ap.add_argument("--seq_col", default="nt", help="CSV column name for DNA sequence (default: nt)")
    ap.add_argument("--pred_glob", required=True, help="glob matching Boltz prediction dirs (e.g., 'runs/*/*')")
    ap.add_argument("--split_by", default="tf", choices=["random", "tf"], help="how to split train/val (default: tf)")
    ap.add_argument("--test_glob", default=None, help="optional glob for an unseen-TF test set")
    ap.add_argument("--test_labels_csv", default=None, help="optional labels CSV for the test set")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--corr_w", type=float, default=0.05)
    ap.add_argument("--out", default="tfdna_affinity.ckpt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 8, 8))
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--pin_memory", type=int, choices=[0,1], default=1)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", type=str, default="16", choices=["32", "16", "bf16"])  # autocast precision
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--normalize", type=str, default="none", choices=["none", "zscore_per_tf"], help="label normalization strategy")
    # pooling & attention
    ap.add_argument("--pooling", type=str, default="attention", choices=["lse", "attention"])  # legacy lse or attention
    ap.add_argument("--attn-hidden", dest="attn_hidden", type=int, default=128)
    ap.add_argument("--attn-dropout", dest="attn_dropout", type=float, default=0.10)
    ap.add_argument("--edge-dropout", dest="edge_dropout", type=float, default=0.10)
    ap.add_argument("--noise-std", dest="noise_std", type=float, default=0.02)
    ap.add_argument("--pool-temp", dest="pool_temp", type=float, default=4.0)
    # distance features
    ap.add_argument("--dist-feats", dest="dist_feats", type=str, default="rbf", choices=["bins", "rbf"])
    ap.add_argument("--dist-rbf-centers", dest="dist_rbf_centers", type=int, default=64)
    ap.add_argument("--dist-rbf-min", dest="dist_rbf_min", type=float, default=2.0)
    ap.add_argument("--dist-rbf-max", dest="dist_rbf_max", type=float, default=22.0)
    ap.add_argument("--dist-rbf-sigma", dest="dist_rbf_sigma", type=float, default=1.0)
    # cache
    ap.add_argument("--cache-dir", type=str, default="runs_cache")
    ap.add_argument("--cache-format", type=str, choices=["npz", "npy"], default="npz")
    ap.add_argument("--cache-in-mem", type=int, default=16)
    ap.add_argument("--cache-z-in-mem", type=int, choices=[0,1], default=0)
    # prior weights
    ap.add_argument("--prior-w-contact", dest="prior_w_contact", type=float, default=1.0)
    ap.add_argument("--prior-w-pae", dest="prior_w_pae", type=float, default=0.25)
    ap.add_argument("--prior-w-pde", dest="prior_w_pde", type=float, default=0.10)
    ap.add_argument("--prior-eps", dest="prior_eps", type=float, default=1e-6)
    # wandb logging
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="tfdna_affinity", help="wandb project name")
    ap.add_argument("--wandb_entity", type=str, default="cellgp", help="wandb entity (team/org)")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="optional wandb run name")
    ap.add_argument("--wandb_group", type=str, default=None, help="optional wandb group")
    ap.add_argument("--wandb_tags", type=str, default="", help="comma-separated wandb tags")
    ap.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online", help="wandb mode")
    ap.add_argument("--wandb_dir", type=str, default="/data/rbg/users/seanmurphy/dna_bind/wandb", help="local directory for wandb files")
    ap.add_argument("--wandb_log_model", action="store_true", help="log model checkpoints to wandb")
    args = ap.parse_args()

    # Data
    dm = AffinityDataModule(labels_csv=args.labels_csv,
                            pred_glob=args.pred_glob,
                            seq_col=args.seq_col,
                            val_frac=args.val_frac,
                            seed=args.seed,
                            num_workers=args.num_workers,
                            batch_size=1,
                            split_by=args.split_by,
                            normalize=args.normalize,
                            test_glob=args.test_glob,
                            test_labels_csv=args.test_labels_csv,
                            dist_feats=args.dist_feats,
                            rbf_centers=args.dist_rbf_centers,
                            rbf_min=args.dist_rbf_min,
                            rbf_max=args.dist_rbf_max,
                            rbf_sigma=args.dist_rbf_sigma,
                            cache_dir=args.cache_dir,
                            cache_format=args.cache_format,
                            cache_in_mem=args.cache_in_mem,
                            cache_z_in_mem=args.cache_z_in_mem,
                            pin_memory=bool(args.pin_memory),
                            prefetch_factor=args.prefetch_factor)
    dm.setup()

    # Model dims
    c_pair, c_single, n_bins = infer_dims_from_datamodule(dm)
    lit = AffinityLightningModule(c_pair=c_pair,
                                  c_single=c_single,
                                  n_bins=n_bins,
                                  lr=args.lr,
                                  weight_decay=args.wd,
                                  corr_w=args.corr_w,
                                  use_soft_pool=True,
                                  pooling=args.pooling,
                                  attn_hidden=args.attn_hidden,
                                  attn_dropout=args.attn_dropout,
                                  edge_dropout=args.edge_dropout,
                                  pool_temp=args.pool_temp,
                                  noise_std=args.noise_std,
                                  prior_w_contact=args.prior_w_contact,
                                  prior_w_pae=args.prior_w_pae,
                                  prior_w_pde=args.prior_w_pde,
                                  prior_eps=args.prior_eps)
    # Thread normalization context to module for de-normalization during metrics
    lit.normalize = args.normalize
    lit.train_stats = dm.train_stats

    # Trainer
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"
    # Prefer PL>=2 strings; fallback to PL<2 if needed
    precision_map = {"32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"}
    strategy = "auto" if not (accelerator == "gpu" and args.devices > 1) else "ddp_find_unused_parameters_false"
    callbacks = [
        ModelCheckpoint(monitor="val/r_by_tf_mean", mode="max", save_top_k=1, filename="best"),
        EarlyStopping(monitor="val/r_by_tf_mean", mode="max", patience=max(5, args.epochs // 10), min_delta=0.002),
    ]
    # Optional wandb logger
    logger_obj = None
    if args.wandb:
        # Respect local dir and mode to avoid network FS issues
        os.makedirs(args.wandb_dir, exist_ok=True)
        # Configure WANDB environment
        os.environ.setdefault("WANDB_DIR", args.wandb_dir)
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(args.wandb_dir, "cache"))
        os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(args.wandb_dir, "config"))
        # Instantiate logger
        run_name = args.wandb_run_name or f"tfdna_affinity_{args.seed}"
        tags = [t for t in (args.wandb_tags.split(",") if args.wandb_tags else []) if t]
        logger_obj = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            save_dir=args.wandb_dir,
            mode=args.wandb_mode,
            log_model=args.wandb_log_model,
        )
        # Log hyperparameters/config
        try:
            logger_obj.experiment.config.update(vars(args), allow_val_change=True)
        except Exception:
            pass

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else 1,
        strategy=strategy,
        callbacks=callbacks,
        precision=precision_map[args.precision],
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip,
        deterministic=False,
        logger=logger_obj,
    )

    # Fit
    trainer.fit(lit, datamodule=dm)

    # Save final/best
    ckpt_path = trainer.checkpoint_callback.best_model_path if hasattr(trainer, "checkpoint_callback") else ""
    if ckpt_path and os.path.exists(ckpt_path):
        # Also copy to user-specified --out path
        state = torch.load(ckpt_path, map_location="cpu")
        torch.save(state, args.out)
    else:
        # Fallback: raw state_dict
        torch.save(lit.state_dict(), args.out)

    # Optional test on unseen TFs
    if dm.test_ds is not None:
        try:
            trainer.test(lit, datamodule=dm)
        except Exception:
            pass


if __name__ == "__main__":
    main()


