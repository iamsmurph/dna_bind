"""Affinity head module."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from .pooling import AttnPool


class BoltzAffinityHeadReplica(nn.Module):
    def __init__(self, c_pair: int, c_single: int, b_bins: int,
                 hidden: int = 256,
                 use_soft_pool: bool = True,
                 pool_temp: float = 4.0,
                 pooling: str = "attention",
                 attn_hidden: int = 128,
                 attn_dropout: float = 0.10,
                 edge_dropout: float = 0.10,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6) -> None:
        super().__init__()
        self.c_pair = int(c_pair)
        self.c_single = int(c_single)
        self.b_bins = int(b_bins)
        self.hidden = int(hidden)
        self.use_soft_pool = bool(use_soft_pool)
        self.pool_temp = float(pool_temp)
        self.pooling = str(pooling)
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
        # Distogram projection to pair channels
        self.dist_proj = nn.Linear(self.b_bins, self.c_pair)
        # Fusion + MLP
        self.ln = nn.LayerNorm(2 * self.c_pair)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.c_pair, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        # Attention pooling block
        self.attn_pool = AttnPool(in_dim=2 * self.c_pair,
                                  hidden=int(attn_hidden),
                                  attn_dropout=float(attn_dropout),
                                  edge_dropout=float(edge_dropout))

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
        if dist_bins_t.shape[:2] != (Lc, Lc):
            raise ValueError("dist_bins must be [Lc,Lc,B]")
        device = z.device
        dist_bins_t = dist_bins_t.to(device=device, dtype=z.dtype)
        if dist_bins_t.shape[-1] != self.b_bins:
            raise AssertionError("b_bins must equal dist_bins last dim")
        s_proxy = s_proxy.to(device)

        # Single projections (shared for dense/sparse)
        u = self.proj_u(s_proxy)
        v = self.proj_v(s_proxy)

        outputs: Dict[str, torch.Tensor] = {}

        use_sparse = (self.pooling == "attention") and hasattr(masks, "pd_pairs") and getattr(masks, "pd_pairs") is not None and len(getattr(masks, "pd_pairs")) > 0

        if use_sparse:
            z = z.contiguous()
            z_flat = z.view(-1, z.shape[-1])
            pd_flat_idx_np = masks.pd_flat_idx
            pd_pairs_np = masks.pd_pairs
            pd_flat_idx = torch.from_numpy(pd_flat_idx_np).to(device=device, dtype=torch.long)
            ij = torch.from_numpy(pd_pairs_np).to(device=device, dtype=torch.long)
            i_idx, j_idx = ij[:, 0], ij[:, 1]

            z_pd = z_flat.index_select(0, pd_flat_idx)

            u_i = u.index_select(0, i_idx)
            v_j = v.index_select(0, j_idx)
            hadamard = u_i * v_j
            gate_pd = torch.cat([u_i, v_j, hadamard], dim=-1)
            bias_pd = self.to_bias(gate_pd)
            z_hat_pd = z_pd + bias_pd

            if dist_bins_t.dim() == 3 and dist_bins_t.shape[0] == Lc:
                d_pd = dist_bins_t[i_idx, j_idx, :]
            else:
                d_pd = dist_bins_t
            if d_pd.shape[-1] != self.b_bins:
                raise AssertionError("b_bins must equal dist_bins last dim")
            d_proj_pd = self.dist_proj(d_pd.to(device=device, dtype=z_hat_pd.dtype))
            h_pd = torch.cat([z_hat_pd, d_proj_pd], dim=-1)
            h_pd = self.ln(h_pd)

            pri_pd: Optional[torch.Tensor] = None
            if (prior_contact is not None) or (prior_pae is not None) or (prior_pde is not None):
                c_pd = prior_contact[i_idx, j_idx] if isinstance(prior_contact, torch.Tensor) and prior_contact.dim() >= 2 else (prior_contact if isinstance(prior_contact, torch.Tensor) else None)
                pae_pd = prior_pae[i_idx, j_idx] if isinstance(prior_pae, torch.Tensor) and prior_pae.dim() >= 2 else (prior_pae if isinstance(prior_pae, torch.Tensor) else None)
                pde_pd = prior_pde[i_idx, j_idx] if isinstance(prior_pde, torch.Tensor) and prior_pde.dim() >= 2 else (prior_pde if isinstance(prior_pde, torch.Tensor) else None)
                from ..geometry.priors import make_prior_logits
                pri_pd = make_prior_logits(
                    contact=c_pd,
                    pae=pae_pd,
                    pde=pde_pd,
                    w_contact=self.prior_w_contact,
                    w_pae=self.prior_w_pae,
                    w_pde=self.prior_w_pde,
                    eps=self.prior_eps,
                )

            pooled, w_pd = self.attn_pool.forward_sparse(
                edge_feats=h_pd,
                priors_logits=pri_pd,
                temp=self.pool_temp,
                training=self.training,
            )
            outputs["attn_weights_pd"] = w_pd
            outputs["pooled_value"] = pooled
            aff_scalar = pooled
        elif self.pooling == "attention":
            u_i = u[:, None, :].expand(-1, Lc, -1)
            v_j = v[None, :, :].expand(Lc, -1, -1)
            hadamard = u_i * v_j
            gate = torch.cat([u_i, v_j, hadamard], dim=-1)
            bias = self.to_bias(gate)
            z_hat = z + bias

            d_proj = self.dist_proj(dist_bins_t)
            h = torch.cat([z_hat, d_proj], dim=-1)
            h = self.ln(h)

            aff_mask_np = masks.affinity_pair_mask
            if aff_mask_np.shape != (Lc, Lc):
                raise ValueError("affinity_pair_mask shape mismatch")
            aff_mask_t = torch.from_numpy(aff_mask_np).to(device)

            priors_logits: Optional[torch.Tensor] = None
            if (prior_contact is not None) or (prior_pae is not None) or (prior_pde is not None):
                from ..geometry.priors import make_prior_logits
                c_t = prior_contact.to(device) if isinstance(prior_contact, torch.Tensor) else None
                pae_t = prior_pae.to(device) if isinstance(prior_pae, torch.Tensor) else None
                pde_t = prior_pde.to(device) if isinstance(prior_pde, torch.Tensor) else None
                priors_logits = make_prior_logits(
                    contact=c_t,
                    pae=pae_t,
                    pde=pde_t,
                    w_contact=self.prior_w_contact,
                    w_pae=self.prior_w_pae,
                    w_pde=self.prior_w_pde,
                    eps=self.prior_eps,
                )

            pooled, attn_w = self.attn_pool(
                edge_feats=h,
                pd_mask=aff_mask_t,
                priors_logits=priors_logits,
                temp=self.pool_temp,
                training=self.training,
            )
            outputs["attn_weights"] = attn_w
            aff_scalar = pooled
        else:
            u_i = u[:, None, :].expand(-1, Lc, -1)
            v_j = v[None, :, :].expand(Lc, -1, -1)
            hadamard = u_i * v_j
            gate = torch.cat([u_i, v_j, hadamard], dim=-1)
            bias = self.to_bias(gate)
            z_hat = z + bias

            d_proj = self.dist_proj(dist_bins_t)
            h = torch.cat([z_hat, d_proj], dim=-1)
            h = self.ln(h)
            edge_scores_raw = self.mlp(h).squeeze(-1)

            aff_mask_np = masks.affinity_pair_mask
            if aff_mask_np.shape != (Lc, Lc):
                raise ValueError("affinity_pair_mask shape mismatch")
            aff_mask_t = torch.from_numpy(aff_mask_np).to(device)

            edge_scores = edge_scores_raw
            if edge_weights is not None:
                if edge_weights.shape != edge_scores_raw.shape:
                    raise ValueError("edge_weights must be [Lc,Lc]")
                edge_weights = edge_weights.to(device)
                edge_scores = edge_scores + edge_weights

            if self.use_soft_pool:
                neg_inf = torch.finfo(edge_scores.dtype).min
                masked = torch.where(aff_mask_t, edge_scores, torch.tensor(neg_inf, device=device))
                temp = max(1e-6, self.pool_temp)
                vals = masked[aff_mask_t]
                aff_scalar = (temp * torch.logsumexp(vals / temp, dim=0)) if vals.numel() else edge_scores.new_tensor(0.0)
            else:
                denom = aff_mask_t.sum().clamp(min=1)
                aff_scalar = (edge_scores * aff_mask_t).sum() / denom
            outputs["edge_scores"] = edge_scores

        return {"affinity": aff_scalar, **outputs}


__all__ = ["BoltzAffinityHeadReplica"]


