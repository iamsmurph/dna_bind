#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob as globlib
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from scripts.tfdna_affinity import (
    TFAffinityRegressor,
    load_crop_bundle,
    parse_cif_to_token_geom,
    make_crop_masks,
    build_dist_bins,
    compute_edge_weights,
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

    def pin_memory(self):
        self.z = self.z.pin_memory()
        self.s_proxy = self.s_proxy.pin_memory()
        if isinstance(self.edge_weights, torch.Tensor):
            self.edge_weights = self.edge_weights.pin_memory()
        return self


class AffinityDataset:
    def __init__(self, pred_dirs: List[str], labels: Dict[Tuple[str, str], float]) -> None:
        self.pred_dirs = pred_dirs
        self.labels = labels

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
        # Skip samples with no PD edges to avoid degenerate pooling
        if not np.any(masks.affinity_pair_mask):
            raise StopIteration
        dist_bins = build_dist_bins(masks.rep_xyz_crop)

        contact_c = downselect_pairwise(bundle.contact_probs, bundle.crop_to_full)
        pae_c = downselect_pairwise(bundle.pae, bundle.crop_to_full)
        pde_c = downselect_pairwise(bundle.pde, bundle.crop_to_full)
        w = compute_edge_weights(contact_c, pae_c, pde_c, mask=masks.affinity_pair_mask)
        w_t = torch.from_numpy(w) if w is not None else None

        key = (bundle.meta["uniprot"], bundle.meta["sequence"])
        if key not in self.labels:
            raise KeyError(f"Label not found for {key}")
        y = torch.tensor(self.labels[key], dtype=torch.float32)

        return Sample(z=bundle.z, s_proxy=bundle.s_proxy, dist_bins=dist_bins, masks=masks, y=y, edge_weights=w_t)


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
                 batch_size: int = 1) -> None:
        super().__init__()
        self.labels_csv = labels_csv
        self.pred_glob = pred_glob
        self.seq_col = seq_col
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.batch_size = int(batch_size)
        self.labels: Dict[Tuple[str, str], float] = {}
        self.train_ds: Optional[AffinityDataset] = None
        self.val_ds: Optional[AffinityDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.labels = read_labels(self.labels_csv, seq_col=self.seq_col)
        pred_dirs = sorted(globlib.glob(os.path.join(self.pred_glob)))
        if not pred_dirs:
            raise RuntimeError(f"No prediction dirs matched {self.pred_glob}")
        rnd = random.Random(self.seed)
        rnd.shuffle(pred_dirs)
        n_val = max(1, int(len(pred_dirs) * self.val_frac))
        val_dirs = pred_dirs[:n_val]
        train_dirs = pred_dirs[n_val:]
        self.train_ds = AffinityDataset(train_dirs, self.labels)
        self.val_ds = AffinityDataset(val_dirs, self.labels)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.train_ds is not None
        return torch.utils.data.DataLoader(self.train_ds,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=self.num_workers,
                                           collate_fn=single_collate,
                                           pin_memory=True,
                                           persistent_workers=self.num_workers > 0)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.val_ds is not None
        return torch.utils.data.DataLoader(self.val_ds,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=self.num_workers,
                                           collate_fn=single_collate,
                                           pin_memory=True,
                                           persistent_workers=self.num_workers > 0)


class AffinityLightningModule(pl.LightningModule):
    def __init__(self,
                 c_pair: int,
                 c_single: int,
                 n_bins: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 corr_w: float = 0.05,
                 use_soft_pool: bool = True) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TFAffinityRegressor(c_pair=c_pair, c_single=c_single, n_bins=n_bins, use_soft_pool=use_soft_pool)
        self.loss_mse = nn.MSELoss()

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
        y_hat, out = self.model(z, s, dist_bins, sample.masks, edge_weights=edge_w)
        return y_hat, out

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        y = batch.y.to(self.device).reshape(1)
        y_hat, _ = self.forward(batch)
        loss = self.loss_mse(y_hat, y) + self.hparams.corr_w * self.corr_penalty(y_hat, y)
        # metrics
        with torch.no_grad():
            mse = self.loss_mse(y_hat, y)
            yh = (y_hat - y_hat.mean()) / (y_hat.std() + 1e-6)
            yt = (y - y.mean()) / (y.std() + 1e-6)
            pearson = (yh * yt).mean()
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log("train/r", pearson, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        y = batch.y.to(self.device).reshape(1)
        y_hat, _ = self.forward(batch)
        mse = self.loss_mse(y_hat, y)
        yh = (y_hat - y_hat.mean()) / (y_hat.std() + 1e-6)
        yt = (y - y.mean()) / (y.std() + 1e-6)
        pearson = (yh * yt).mean()
        self.log("val/mse", mse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, batch_size=1)
        self.log("val/r", pearson, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False, batch_size=1)

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
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--corr_w", type=float, default=0.05)
    ap.add_argument("--out", default="tfdna_affinity.ckpt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", type=str, default="32", choices=["32", "16", "bf16"])  # autocast precision
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    # Data
    dm = AffinityDataModule(labels_csv=args.labels_csv,
                            pred_glob=args.pred_glob,
                            seq_col=args.seq_col,
                            val_frac=args.val_frac,
                            seed=args.seed,
                            num_workers=args.num_workers,
                            batch_size=1)
    dm.setup()

    # Model dims
    c_pair, c_single, n_bins = infer_dims_from_datamodule(dm)
    lit = AffinityLightningModule(c_pair=c_pair,
                                  c_single=c_single,
                                  n_bins=n_bins,
                                  lr=args.lr,
                                  weight_decay=args.wd,
                                  corr_w=args.corr_w,
                                  use_soft_pool=True)

    # Trainer
    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"
    # Prefer PL>=2 strings; fallback to PL<2 if needed
    precision_map = {"32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"}
    strategy = "auto" if not (accelerator == "gpu" and args.devices > 1) else "ddp_find_unused_parameters_false"
    callbacks = [
        ModelCheckpoint(monitor="val/mse", mode="min", save_top_k=1, filename="best"),
        EarlyStopping(monitor="val/mse", mode="min", patience=max(5, args.epochs // 10)),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else 1,
        strategy=strategy,
        callbacks=callbacks,
        precision=precision_map[args.precision],
        log_every_n_steps=1,
        gradient_clip_val=args.grad_clip,
        deterministic=True,
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


if __name__ == "__main__":
    main()


