from __future__ import annotations
import torch
from boltz.data import const

DEFAULT_MOL_TYPE_BUDGET = {
    const.chain_type_ids["DNA"]: 50,
    const.chain_type_ids["RNA"]: 50,
    const.chain_type_ids["PROTEIN"]: 150,
}


def cutoff_bin_mask(
    distance_cutoff: float = 12,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    n_bins: int = 64,
    device: torch.device | None = None,
):
    """
    Returns a boolean mask over distogram bins: True for distances < cutoff.
    Expects `bin_mids` as produced in model setup (includes +inf as last bin).
    """
    bin_mask = (
        torch.cat(
            [
                torch.linspace(min_dist, max_dist, n_bins - 1),
                torch.tensor([torch.inf]),
            ]
        )
        < distance_cutoff
    )

    if device is not None:
        return (bin_mask < distance_cutoff).to(device)

    return bin_mask


@torch.no_grad()
def contact_logprobs(
    pdistogram: torch.Tensor,  # [B, N, N, 2, K] or [B, N, N, K] if logits pre-selected
    exclude_mask: torch.Tensor,  # [B, N, N] boolean
    bin_mask: torch.Tensor,  # [K] boolean
) -> torch.Tensor:
    """
    Log P(contact) aggregated over bins in `bin_mask`, with exclusions masked to -inf.
    Assumes logits over bins are in pdistogram[..., 0, :] or pdistogram[..., :].
    """
    logits = pdistogram[..., 0] if pdistogram.ndim == 5 else pdistogram  # [B,N,N,K]

    log_numer = torch.logsumexp(logits[..., bin_mask], dim=-1)
    log_denom = torch.logsumexp(logits, dim=-1)
    logprobs = log_numer - log_denom
    logprobs = logprobs.masked_fill(exclude_mask, float("-inf"))
    return logprobs


@torch.no_grad()
def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    # stable log(1 - exp(x)) for x <= 0
    LOG_HALF = -0.6931471805599453  # log(0.5)
    return torch.where(
        x < LOG_HALF, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
    )


@torch.no_grad()
def contact_scores(logprobs: torch.Tensor) -> torch.Tensor:
    """
    Reduce pairwise log P(contact_ij) to a per-token score:
    score_i = log(1 - prod_j (1 - P_ij)) computed stably in log-space.
    """
    log_complements = _log1mexp(logprobs)  # log(1 - P_ij)
    scores = _log1mexp(log_complements.sum(dim=-1))  # sum over j
    return scores  # [B, N]


@torch.no_grad()
def self_interaction_mask(asym_id: torch.Tensor) -> torch.Tensor:
    # [B, N, N]
    return asym_id[..., :, None] == asym_id[..., None, :]


@torch.no_grad()
def intra_mol_type_interaction_mask(
    mol_type: torch.Tensor, mol_type_id: int
) -> torch.Tensor:
    # [B, N, N] True for pairs where both tokens are of mol_type_id
    i = mol_type[..., :, None] == mol_type_id
    j = mol_type[..., None, :] == mol_type_id
    return i & j


def pad_interaction_mask(token_pad_mask: torch.Tensor) -> torch.Tensor:
    # token_pad_mask is false for padded tokens
    return ~token_pad_mask[..., :, None] | ~token_pad_mask[..., None, :]


@torch.no_grad()
def default_exclude_mask(
    token_pad_mask: torch.Tensor,  # [B, N]
    asym_id: torch.Tensor,  # [B, N]
    mol_type: torch.Tensor,  # [B, N]
    *,
    disallow_intra_mol_type_id: int | None = None,
    exclude_self_pairs: bool = True,
) -> torch.Tensor:
    """
    Build an exclude mask for pair scores:
    - mask padded tokens
    - optionally mask self-chain pairs
    - optionally mask intra pairs for a specific mol_type_id (e.g., DNA-DNA)
    """
    pad = ~token_pad_mask.bool()
    pair_pad = pad[..., :, None] | pad[..., None, :]
    out = pair_pad
    if exclude_self_pairs:
        out = out | self_intersection_mask(asym_id)
    if disallow_intra_mol_type_id is not None:
        out = out | intra_mol_type_pair_mask(mol_type, disallow_intra_mol_type_id)
    return out


@torch.no_grad()
def select_with_neighborhood(
    scores: torch.Tensor,  # [N]
    budget: int,
    neighborhood_size: int,
    *,
    allowed: torch.Tensor | None = None,  # [N] bool
    allow_partial: bool = True,
    preselected: torch.Tensor | None = None,  # [N] bool
) -> torch.Tensor:
    """
    Greedy select centers by descending score, then include [i-radius, i+radius] neighbors
    under constraints. Returns sorted LongTensor of indices.
    """
    assert scores.ndim == 1
    N = scores.numel()
    device = scores.device

    selected = torch.zeros(N, dtype=torch.bool, device=device)
    if preselected is not None:
        selected |= preselected.to(device=device, dtype=torch.bool)
    if allowed is None:
        allowed = torch.ones(N, dtype=torch.bool, device=device)
    else:
        allowed = allowed.to(device=device, dtype=torch.bool)

    masked = scores.clone()
    masked[~allowed] = float("-inf")
    order = torch.argsort(masked, descending=True)

    remaining = budget - int(selected.sum().item())
    if remaining <= 0:
        return selected.nonzero(as_tuple=False).flatten()

    radius = int(neighborhood_size / 2)

    for idx in order:
        if remaining <= 0:
            break
        i = int(idx.item())
        if not allowed[i]:
            continue

        start = max(0, i - radius)
        end = min(N, i + radius + 1)
        window = torch.arange(start, end, device=device)
        window = window[allowed[window] & ~selected[window]]
        if window.numel() == 0:
            continue

        if window.numel() > remaining:
            if not allow_partial:
                continue
            d = (window - i).abs()
            keep = d.argsort()[:remaining]
            window = window[keep]

        selected[window] = True
        remaining -= int(window.numel())

    return selected.nonzero(as_tuple=False).flatten().sort().values


@torch.no_grad()
def select_with_mol_type_budget(
    contact_profile: torch.Tensor,  # [B, N] scores
    asym_id: torch.Tensor,  # [B, N]
    mol_type: torch.Tensor,  # [B, N]
    mol_type_budget: dict[int, int],
    neighborhood_size: int = 10,
) -> list[torch.Tensor]:
    """
    Per-batch selection honoring per-chain mol_type budgets. Returns a list of index tensors per batch.
    """
    B, N = contact_profile.shape
    out: list[torch.Tensor] = []
    device = contact_profile.device

    for b in range(B):
        sel = []
        chains = asym_id[b].unique()
        for chain in chains:
            chain_mask = asym_id[b] == chain
            chain_types = mol_type[b][chain_mask].unique()
            assert chain_types.numel() == 1, "Multiple chain types in a single chain"
            chain_type = int(chain_types.item())
            budget = min(int(mol_type_budget.get(chain_type, 0)), N)
            if budget <= 0:
                continue
            idx = select_with_neighborhood(
                contact_profile[b],
                budget=budget,
                neighborhood_size=10,
                allowed=chain_mask,
            )
            sel.append(idx)
        if sel:
            out.append(torch.cat(sel).sort().values.to(device))
        else:
            out.append(torch.empty(0, dtype=torch.long, device=device))
    return out
