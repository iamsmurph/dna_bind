"""Boltz-style cross-attention with additive pair bias restricted to protein→DNA edges.

This module mirrors Boltz's AttentionPairBias pattern:
- Q, K, V from single features `s`
- per-head additive bias from pair features `z`
- optional per-head distance bias and scalar prior bias (added to logits)
- mask applied in logits before softmax
- softmax computed in float32 for numerical stability
- gated output projected back to `C_s`
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairBiasCrossAttentionPD(nn.Module):
    def __init__(self,
                 c_single: int,
                 c_pair: int,
                 num_heads: int = 8,
                 attn_dropout: float = 0.10) -> None:
        super().__init__()
        assert isinstance(c_single, int) and c_single > 0
        assert isinstance(c_pair, int) and c_pair > 0
        assert isinstance(num_heads, int) and num_heads > 0
        self.c_single = int(c_single)
        self.c_pair = int(c_pair)
        self.num_heads = int(num_heads)
        self.attn_dropout = float(attn_dropout)

        if (self.c_single % self.num_heads) != 0:
            raise ValueError("c_single must be divisible by num_heads")
        self.d_head = self.c_single // self.num_heads

        # Projections for Q, K, V
        self.q_proj = nn.Linear(self.c_single, self.c_single)
        self.k_proj = nn.Linear(self.c_single, self.c_single)
        self.v_proj = nn.Linear(self.c_single, self.c_single)
        # Per-head bias from pair features z -> H
        self.z_to_bias_h = nn.Linear(self.c_pair, self.num_heads, bias=False)

        # Output projection and gating
        self.o_proj = nn.Linear(self.c_single, self.c_single)
        self.gate = nn.Linear(self.c_single, self.c_single)
        self.dropout = nn.Dropout(self.attn_dropout)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Lc, C_s] -> [H, Lc, d]
        Lc = x.shape[0]
        H = self.num_heads
        d = self.d_head
        xh = x.view(Lc, H, d).permute(1, 0, 2).contiguous()
        return xh

    def forward(self,
                s: torch.Tensor,             # [Lc, C_s]
                z: torch.Tensor,             # [Lc, Lc, C_z]
                pd_mask: torch.Tensor,       # [Lc, Lc] bool (True where protein->DNA)
                dist_bias_h: Optional[torch.Tensor] = None,  # [Lc, Lc, H]
                prior_bias: Optional[torch.Tensor] = None,   # [Lc, Lc]
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        Lc, Cs = s.shape
        if Cs != self.c_single:
            raise ValueError("s last dim must equal c_single")
        if z.shape[:2] != (Lc, Lc) or z.shape[-1] != self.c_pair:
            raise ValueError("z must be [Lc,Lc,c_pair]")
        if pd_mask.shape != (Lc, Lc):
            raise ValueError("pd_mask must be [Lc,Lc]")

        device = s.device
        dtype = s.dtype

        q = self.q_proj(s)
        k = self.k_proj(s)
        v = self.v_proj(s)

        qh = self._shape_heads(q)  # [H, Lc, d]
        kh = self._shape_heads(k)  # [H, Lc, d]
        vh = self._shape_heads(v)  # [H, Lc, d]

        # Compute logits in fp32 for stability
        with torch.autocast("cuda", enabled=False):
            qh32 = qh.to(dtype=torch.float32)
            kh32 = kh.to(dtype=torch.float32)
            scale = (1.0 / (self.d_head ** 0.5))
            # [H, Lc, Lc]
            logits = torch.einsum("h i d, h j d -> h i j", qh32, kh32) * scale

            # Add per-head bias from z
            zb = self.z_to_bias_h(z.to(dtype=torch.float32))  # [Lc,Lc,H]
            zb = zb.permute(2, 0, 1).contiguous()             # [H,Lc,Lc]
            logits = logits + zb

            if dist_bias_h is not None:
                db = dist_bias_h.to(dtype=torch.float32).permute(2, 0, 1).contiguous()  # [H,Lc,Lc]
                logits = logits + db

            if prior_bias is not None:
                pb = prior_bias.to(dtype=torch.float32)  # [Lc,Lc]
                logits = logits + pb.unsqueeze(0)

            # Mask out non-PD edges
            neg_inf = torch.finfo(torch.float32).min
            mask = pd_mask.to(device=device)
            logits = torch.where(mask.unsqueeze(0), logits, torch.tensor(neg_inf, device=logits.device, dtype=logits.dtype))

            attn = torch.softmax(logits, dim=-1)  # [H, Lc, Lc]

        attn = self.dropout(attn).to(dtype=dtype)

        # Weighted sum over values
        ctx = torch.einsum("h i j, h j d -> h i d", attn, vh)  # [H,Lc,d]
        ctx = ctx.permute(1, 0, 2).contiguous().view(Lc, self.c_single)  # [Lc,C_s]

        # Gated residual-style output
        gate = torch.sigmoid(self.gate(s))
        o = self.o_proj(ctx)
        y = gate * o

        return y, attn


__all__ = ["PairBiasCrossAttentionPD"]


