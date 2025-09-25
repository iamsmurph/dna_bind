"""Priors and edge-weight utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def make_prior_logits(contact: Optional[torch.Tensor] = None,
                      pae: Optional[torch.Tensor] = None,
                      pde: Optional[torch.Tensor] = None,
                      w_contact: float = 1.0,
                      w_pae: float = 0.25,
                      w_pde: float = 0.10,
                      eps: float = 1e-6,
                      pae_scale: float = 10.0,
                      pde_scale: float = 10.0) -> torch.Tensor:
    pri = None
    if contact is not None:
        c = contact.to(dtype=torch.float32)
        term = float(w_contact) * torch.log(c.clamp_min(float(eps)))
        pri = term if pri is None else pri + term
    if pae is not None:
        paet = pae.to(dtype=torch.float32)
        term = float(w_pae) * (-(paet / float(pae_scale)))
        pri = term if pri is None else pri + term
    if pde is not None:
        pdet = pde.to(dtype=torch.float32)
        term = float(w_pde) * (-(pdet / float(pde_scale)))
        pri = term if pri is None else pri + term
    if pri is None:
        pri = torch.tensor(0.0)
    return pri


def compute_edge_weights(contact: Optional[np.ndarray],
                         pae: Optional[np.ndarray],
                         pde: Optional[np.ndarray],
                         alpha: float = 2.0,
                         beta: float = 0.2,
                         gamma: float = 0.2,
                         tau: float = 0.2,
                         mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if contact is None and pae is None and pde is None:
        return None
    base_shape = None
    for arr in (contact, pae, pde):
        if arr is not None:
            base_shape = arr.shape
            break
    if base_shape is None:
        return None
    w = np.zeros(base_shape, dtype=np.float32)
    if contact is not None:
        w += alpha * np.maximum(contact.astype(np.float32) - float(tau), 0.0)
    if pae is not None:
        w -= beta * pae.astype(np.float32)
    if pde is not None:
        w -= gamma * pde.astype(np.float32)
    if mask is not None:
        w = np.where(mask, w, 0.0)
    return w


__all__ = ["make_prior_logits", "compute_edge_weights"]


