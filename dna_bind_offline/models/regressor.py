"""TF-DNA affinity regressor."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .heads import BoltzAffinityHeadReplica
from ..models.types import CropMasks


class TFAffinityRegressor(nn.Module):
    def __init__(self, c_pair: int, c_single: int, n_bins: int,
                 attn_dropout: float = 0.10,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6,
                 heads: int = 8,
                 ) -> None:
        super().__init__()
        self.head = BoltzAffinityHeadReplica(
            c_pair=c_pair,
            c_single=c_single,
            b_bins=n_bins,
            attn_dropout=attn_dropout,
            noise_std=noise_std,
            prior_w_contact=prior_w_contact,
            prior_w_pae=prior_w_pae,
            prior_w_pde=prior_w_pde,
            prior_eps=prior_eps,
            heads=heads,
        )
        self.calib = nn.Linear(1, 1)
        with torch.no_grad():
            self.calib.weight.fill_(1.0)
            self.calib.bias.zero_()

    def forward(self,
                z: torch.Tensor,
                s_proxy: torch.Tensor,
                dist_bins: torch.Tensor | np.ndarray,
                masks: CropMasks,
                prior_contact: Optional[torch.Tensor] = None,
                prior_pae: Optional[torch.Tensor] = None,
                prior_pde: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.head(
            z, s_proxy, dist_bins, masks,
            prior_contact=prior_contact,
            prior_pae=prior_pae,
            prior_pde=prior_pde,
        )
        aff = out["affinity"].reshape(1, 1)
        y_hat = self.calib(aff).reshape(-1)
        return y_hat, out


__all__ = ["TFAffinityRegressor"]


