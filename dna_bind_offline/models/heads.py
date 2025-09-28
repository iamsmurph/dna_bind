"""Affinity head module."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .attn_pd import PairBiasCrossAttentionPD


class BoltzAffinityHeadReplica(nn.Module):
    def __init__(self, c_pair: int, c_single: int, b_bins: int,
                 hidden: int = 256,
                 attn_dropout: float = 0.10,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6,
                 heads: int = 8) -> None:
        super().__init__()
        self.c_pair = int(c_pair)
        self.c_single = int(c_single)
        self.b_bins = int(b_bins)
        self.hidden = int(hidden)
        # legacy args removed; keep only used fields
        self.noise_std = float(noise_std)
        self.prior_w_contact = float(prior_w_contact)
        self.prior_w_pae = float(prior_w_pae)
        self.prior_w_pde = float(prior_w_pde)
        self.prior_eps = float(prior_eps)

        # Single projections
        self.proj_u = nn.Linear(self.c_single, self.hidden)
        self.proj_v = nn.Linear(self.c_single, self.hidden)
        # Combine u, v, u*v into pair channels
        self.to_bias = nn.Linear(3 * self.hidden, self.c_pair)
        # Attention block and distance-to-head bias
        self.attn = PairBiasCrossAttentionPD(c_single=self.c_single,
                                             c_pair=self.c_pair,
                                             num_heads=int(heads),
                                             attn_dropout=float(attn_dropout))
        self.num_heads = int(heads)
        self.dist_to_head = nn.Linear(self.b_bins, int(heads), bias=False)
        # Output MLP from pooled token representation to scalar
        self.out_mlp = nn.Sequential(
            nn.LayerNorm(self.c_single),
            nn.Linear(self.c_single, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    def forward(self,
                z: torch.Tensor,
                s_proxy: torch.Tensor,
                dist_bins: torch.Tensor | np.ndarray,
                masks,
                edge_weights: Optional[torch.Tensor] = None,
                prior_contact: Optional[torch.Tensor] = None,
                prior_pae: Optional[torch.Tensor] = None,
                prior_pde: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        Lc = z.shape[0]
        if z.ndim != 3 or z.shape[0] != z.shape[1]:
            raise ValueError("z must be [Lc,Lc,C_pair]")
        if s_proxy.shape[0] != Lc:
            raise ValueError("s_proxy length must equal Lc")
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
            s_proxy = s_proxy + torch.randn_like(s_proxy) * (self.noise_std * 0.5)
        if isinstance(dist_bins, np.ndarray):
            dist_bins_t = torch.from_numpy(dist_bins)
        else:
            dist_bins_t = dist_bins
        device = z.device
        # Densify PD-sparse RBFs [K,B] -> [Lc,Lc,B] when needed
        if dist_bins_t.dim() == 2 or dist_bins_t.shape[:2] != (Lc, Lc):
            if hasattr(masks, "pd_flat_idx"):
                flat = torch.from_numpy(masks.pd_flat_idx).to(device=device, dtype=torch.long)
                dense = torch.zeros((Lc, Lc, self.b_bins), dtype=dist_bins_t.dtype, device=device)
                dense.view(-1, self.b_bins).index_copy_(0, flat, dist_bins_t.to(device=device))
                dist_bins_t = dense
            else:
                raise ValueError("Need masks.pd_flat_idx to densify PD RBF bins")
        dist_bins_t = dist_bins_t.to(device=device, dtype=z.dtype)
        if dist_bins_t.shape[-1] != self.b_bins:
            raise AssertionError("b_bins must equal dist_bins last dim")
        s_proxy = s_proxy.to(device)

        # Single projections (shared for dense/sparse)
        u = self.proj_u(s_proxy)
        v = self.proj_v(s_proxy)

        outputs: Dict[str, torch.Tensor] = {}

        # Dense PD mask
        if hasattr(masks, "affinity_pair_mask"):
            if isinstance(masks.affinity_pair_mask, np.ndarray):
                pd_mask = torch.from_numpy(masks.affinity_pair_mask).to(device=device)
            else:
                pd_mask = masks.affinity_pair_mask.to(device)
        else:
            pd_mask = torch.ones((Lc, Lc), dtype=torch.bool, device=device)

        # Per-head distance bias
        dist_bias_h = self.dist_to_head(dist_bins_t)

        # Prior bias (dense)
        from ..geometry.priors import make_prior_logits

        def _densify(vec: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if vec is None:
                return None
            if isinstance(vec, torch.Tensor) and vec.dim() >= 2:
                return vec.to(device=device)
            if hasattr(masks, "pd_flat_idx") and isinstance(masks.pd_flat_idx, np.ndarray) and isinstance(vec, torch.Tensor):
                flat_idx = torch.from_numpy(masks.pd_flat_idx).to(device=device, dtype=torch.long)
                dense = torch.zeros((Lc, Lc), dtype=vec.dtype, device=device)
                dense.view(-1).index_copy_(0, flat_idx, vec.to(device=device))
                return dense
            return None

        c_dense = _densify(prior_contact)
        pae_dense = _densify(prior_pae)
        pde_dense = _densify(prior_pde)
        prior_bias = make_prior_logits(contact=c_dense, pae=pae_dense, pde=pde_dense,
                                       w_contact=self.prior_w_contact, w_pae=self.prior_w_pae,
                                       w_pde=self.prior_w_pde, eps=self.prior_eps)
        prior_bias = prior_bias.to(device=device)

        # SDPA over PD edges
        y, attn_w = self.attn(s=s_proxy, z=z, pd_mask=pd_mask.bool(), dist_bias_h=dist_bias_h, prior_bias=prior_bias)

        # Pool over protein rows (rows with at least one PD edge)
        prot_mask = pd_mask.any(dim=1)
        y_prot = y[prot_mask] if prot_mask.any() else y
        pooled = y_prot.mean(dim=0)
        aff_scalar = self.out_mlp(pooled).reshape(())
        outputs["attn_weights_pd"] = attn_w
        outputs["pooled_value"] = aff_scalar

        return {"affinity": aff_scalar, **outputs}


__all__ = ["BoltzAffinityHeadReplica"]


