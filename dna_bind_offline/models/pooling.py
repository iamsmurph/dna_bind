"""Attention pooling layers."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple


class AttnPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, attn_dropout: float = 0.1, edge_dropout: float = 0.1) -> None:
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(attn_dropout),
            nn.Linear(hidden, 1)
        )
        self.logit_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(attn_dropout),
            nn.Linear(hidden, 1)
        )
        self.edge_dropout = float(edge_dropout)

    def forward(self,
                edge_feats: torch.Tensor,
                pd_mask: torch.Tensor,
                priors_logits: Optional[torch.Tensor] = None,
                temp: float = 4.0,
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.value_mlp(edge_feats).squeeze(-1)
        logit = self.logit_mlp(edge_feats).squeeze(-1)
        if priors_logits is not None:
            logit = logit + priors_logits.to(device=logit.device, dtype=logit.dtype)
        neg_inf = torch.finfo(logit.dtype).min
        logit = torch.where(pd_mask, logit, torch.tensor(neg_inf, device=logit.device, dtype=logit.dtype))
        if training and self.edge_dropout > 0:
            keep = (torch.rand_like(logit) > self.edge_dropout) | (~pd_mask)
            logit = torch.where(keep, logit, torch.tensor(neg_inf, device=logit.device, dtype=logit.dtype))
        flat = (logit / max(1e-6, float(temp))).reshape(-1)
        w = torch.softmax(flat, dim=-1).reshape_as(logit)
        pooled = (w * v).sum()
        return pooled, w

    def forward_sparse(self,
                       edge_feats: torch.Tensor,
                       priors_logits: Optional[torch.Tensor] = None,
                       temp: float = 4.0,
                       training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.value_mlp(edge_feats).squeeze(-1)
        logit = self.logit_mlp(edge_feats).squeeze(-1)
        if priors_logits is not None:
            logit = logit + priors_logits.to(device=logit.device, dtype=logit.dtype)
        if training and self.edge_dropout > 0:
            keep = (torch.rand_like(logit) > self.edge_dropout)
            neg_inf = torch.finfo(logit.dtype).min
            logit = torch.where(keep, logit, torch.tensor(neg_inf, device=logit.device, dtype=logit.dtype))
        w = torch.softmax(logit / max(1e-6, float(temp)), dim=0)
        pooled = (w * v).sum()
        return pooled, w


# Alias for compatibility
AttentionPooling = AttnPool

__all__ = ["AttnPool", "AttentionPooling"]


