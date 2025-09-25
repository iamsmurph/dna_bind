"""Inference helpers."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def predict_intensity(model: torch.nn.Module, sample) -> Tuple[float, Dict[str, torch.Tensor]]:
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    z = getattr(sample, "z")
    s = getattr(sample, "s", None)
    if s is None:
        s = getattr(sample, "s_proxy")
    dist_bins = getattr(sample, "dist_bins")
    masks = getattr(sample, "masks")
    edge_w = getattr(sample, "edge_weights", None)

    z = z.to(device)
    s = s.to(device)
    y_hat, out = model(z, s, dist_bins, masks, edge_weights=edge_w)
    return float(y_hat.detach().cpu().reshape(()).item()), out


__all__ = ["predict_intensity"]


