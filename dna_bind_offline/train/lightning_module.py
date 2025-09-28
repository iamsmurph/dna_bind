"""LightningModule for TF-DNA affinity training."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

from ..data.dataset import Sample
from ..models.regressor import TFAffinityRegressor


class AffinityLightningModule(pl.LightningModule):
    def __init__(self,
                 c_pair: int,
                 c_single: int,
                 n_bins: int,
                 lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 corr_w: float = 0.05,
                 attn_dropout: float = 0.10,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6,
                 heads: int = 8) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = TFAffinityRegressor(
            c_pair=c_pair,
            c_single=c_single,
            n_bins=n_bins,
            attn_dropout=attn_dropout,
            noise_std=noise_std,
            prior_w_contact=prior_w_contact,
            prior_w_pae=prior_w_pae,
            prior_w_pde=prior_w_pde,
            prior_eps=prior_eps,
            heads=heads,
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

    def forward(self, batch: Sample) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        device = self.device
        z = batch.z.to(device, non_blocking=True)
        s = batch.s_proxy.to(device, non_blocking=True)
        dist_bins = batch.dist_bins
        if isinstance(dist_bins, np.ndarray):
            dist_bins = torch.from_numpy(dist_bins)
        dist_bins = dist_bins.to(device=device, dtype=z.dtype, non_blocking=True)
        prior_c = batch.prior_contact.to(device, non_blocking=True) if isinstance(batch.prior_contact, torch.Tensor) else None
        prior_pae = batch.prior_pae.to(device, non_blocking=True) if isinstance(batch.prior_pae, torch.Tensor) else None
        prior_pde = batch.prior_pde.to(device, non_blocking=True) if isinstance(batch.prior_pde, torch.Tensor) else None
        y_hat, out = self.model(z, s, dist_bins, batch.masks,
                                prior_contact=prior_c, prior_pae=prior_pae, prior_pde=prior_pde)
        return y_hat, out

    def training_step(self, batch: Sample, batch_idx: int) -> torch.Tensor:
        y = batch.y.to(self.device).reshape(1)
        y_hat, _ = self.forward(batch)
        loss = self.loss_mse(y_hat, y)
        with torch.no_grad():
            mse = self.loss_mse(y_hat, y)
        self.log("train/mse", mse, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch: Sample, batch_idx: int) -> None:
        y = batch.y.to(self.device).reshape(1)
        y_hat, out = self.forward(batch)
        yhat_cpu = y_hat.detach().cpu().reshape(()).item()
        uniprot = batch.uniprot
        if self.normalize == "zscore_per_tf" and uniprot in self.train_stats:
            mu, sigma = self.train_stats[uniprot]
            yhat_raw = yhat_cpu * sigma + mu
        else:
            yhat_raw = yhat_cpu
        y_raw = batch.y.detach().cpu().reshape(()).item()
        self.val_store.setdefault(uniprot, []).append((y_raw, yhat_raw))
        mse = torch.tensor((yhat_raw - y_raw) ** 2, dtype=torch.float32)
        self.log("val/mse_point", mse, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False, batch_size=1)

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


__all__ = ["AffinityLightningModule"]


