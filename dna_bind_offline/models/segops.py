from __future__ import annotations

import torch


def segmented_softmax(
    logits: torch.Tensor,
    groups: torch.LongTensor,
    num_groups: int | None = None,
    temp: float = 1.0,
) -> torch.Tensor:
    """Compute softmax over segments (groups) independently for each head/channel.

    Args:
        logits: Tensor of shape [E, H] where E is number of edges and H is number of heads.
        groups: Long tensor of shape [E] with values in [0, num_groups).
        num_groups: Optional total number of groups. If None, inferred from groups.max()+1.
        temp: Temperature for softmax.

    Returns:
        Tensor of shape [E, H] containing per-group, per-head softmax weights that sum to 1 within each group along E.
    """
    if logits.dim() != 2:
        raise ValueError("logits must be [E,H]")
    if groups.dim() != 1 or groups.shape[0] != logits.shape[0]:
        raise ValueError("groups must be [E] and align with logits E dimension")
    E, H = logits.shape
    if num_groups is None:
        num_groups = int(groups.max().item()) + 1 if E > 0 else 0

    # Max per (group, head) for numerical stability
    max_g = torch.full((num_groups, H), float("-inf"), device=logits.device, dtype=logits.dtype)
    max_g.scatter_reduce_(
        0,
        groups[:, None].expand(-1, H),
        logits,
        reduce="amax",
        include_self=True,
    )

    # Center logits by group max and exponentiate with temperature scaling
    temp_val = max(1e-6, float(temp))
    centered = logits - max_g.index_select(0, groups)
    exp = torch.exp(centered / temp_val)

    # Denominator per (group, head)
    denom = torch.zeros_like(max_g)
    denom.scatter_add_(0, groups[:, None].expand(-1, H), exp)

    # Normalize back to edge positions
    weights = exp / denom.index_select(0, groups).clamp_min_(1e-12)
    return weights


__all__ = ["segmented_softmax"]


