# %%

# Plot the contact profile
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pprint import pprint
from boltz.data import const

# %%

out = torch.load("./temp_pdist_feats.pt", weights_only=False)
# %%
feats = out["feats"]
pdistogram = out["pdistogram"]


# %%


@torch.no_grad()
def select_with_neighborhood(
    scores: torch.Tensor,
    budget: int,
    radius: int,
    *,
    allowed: torch.Tensor | None = None,
    allow_partial: bool = True,
    preselected: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Greedy neighborhood selection restricted to a subset of indices.

    Args:
        scores: 1D float tensor of shape [N].
        budget: maximum number of residues to select.
        radius: neighborhood radius r (selects [i-r, i+r]).
        allowed: optional 1D bool tensor [N] — only residues with allowed[i]=True
                 may be selected.
        allow_partial: if last neighborhood would exceed budget, pick closest-to-center
                       subset to fit exactly.
        preselected: optional bool tensor [N] for residues already selected.

    Returns:
        Long tensor of sorted selected indices (ascending).
    """
    N = scores.numel()
    device = scores.device
    assert scores.ndim == 1

    selected = torch.zeros(N, dtype=torch.bool, device=device)
    if preselected is not None:
        selected |= preselected.to(device=device, dtype=torch.bool)

    if allowed is None:
        allowed = torch.ones(N, dtype=torch.bool, device=device)
    else:
        allowed = allowed.to(device=device, dtype=torch.bool)

    # Zero out scores outside allowed region to ensure they sort to the bottom
    masked_scores = scores.clone()
    masked_scores[~allowed] = float("-inf")

    order = torch.argsort(masked_scores, descending=True)
    remaining = budget - int(selected.sum().item())
    if remaining <= 0:
        return selected.nonzero(as_tuple=False).flatten()

    for idx in order:
        if remaining <= 0:
            break
        i = int(idx.item())
        if not allowed[i]:
            continue

        start = max(0, i - radius)
        end = min(N, i + radius + 1)

        window = torch.arange(start, end, device=device)
        # only allowed, not already selected
        window = window[allowed[window] & ~selected[window]]
        if window.numel() == 0:
            continue

        if window.numel() > remaining:
            if not allow_partial:
                continue
            d = (window - i).abs()
            keep = d.argsort()[:remaining]  # closest to center
            window = window[keep]

        selected[window] = True
        remaining -= int(window.numel())

    return selected.nonzero(as_tuple=False).flatten()


def cutoff_bin_mask(
    distance_cutoff: float = 12,
    min_dist: float = 2.0,
    max_dist: float = 22.0,
    n_bins: int = 64,
):
    return (
        torch.cat(
            [
                torch.linspace(min_dist, max_dist, logits.shape[-1] - 1),
                torch.tensor([torch.inf]),
            ]
        )
        < distance_cutoff
    ).to(pdistogram.device)


@torch.no_grad()
def contact_logprobs(
    pdistogram: torch.Tensor, exclude_mask: torch.Tensor, bin_mask: torch.Tensor
):
    """
    Calculates the log probabilities of a given token to interact with any other token in the sequence.

    By choosing exclude_mask, we can exclude things like self-interactions or intra-chain interactions.
    """
    logits = pdistogram[:, :, :, 0]
    logprobs = torch.logsumexp(logits[:, :, :, bin_mask], dim=-1) - torch.logsumexp(
        logits, dim=-1
    )
    logprobs[exclude_mask] = -torch.inf

    return logprobs


@torch.no_grad()
def _log1mexp(x: torch.Tensor) -> torch.Tensor:
    # stable log(1 - exp(x)) for x <= 0
    LOG_HALF = -0.6931471805599453  # log(0.5)
    return torch.where(
        x < LOG_HALF, torch.log1p(-torch.exp(x)), torch.log(-torch.expm1(x))
    )


@torch.no_grad()
def contact_scores(logprobs: torch.Tensor):
    log_complements = _log1mexp(logprobs)
    scores = _log1mexp(log_complements.sum(dim=-1))
    return scores


@torch.no_grad()
def self_intersection_mask(asym_id: torch.Tensor):
    return asym_id[..., :, None] == asym_id[..., None, :]


@torch.no_grad()
def intra_mol_type_interaction_mask(mol_type: torch.Tensor, mol_type_id: int):
    is_dna_i = mol_type[..., :, None] == mol_type_id
    is_dna_j = mol_type[..., None, :] == mol_type_id
    return is_dna_i & is_dna_j


def select_with_mol_type_budget(
    contact_profile: torch.Tensor, mol_type_budget: dict, window: int = 10
):
    token_indices = {}
    for batch_idx in range(contact_profile.shape[0]):
        indices = []
        for chain in feats["asym_id"][batch_idx].unique():
            chain_mask = feats["asym_id"][batch_idx] == chain
            chain_type = feats["mol_type"][batch_idx][chain_mask].unique()
            assert chain_type.numel() == 1, "Multiple chain types found"
            chain_type = chain_type.item()
            chain_budget = mol_type_budget[chain_type]
            selected_indices = select_with_neighborhood(
                contact_profile[batch_idx],
                budget=chain_budget,
                radius=window,
                allowed=chain_mask,
            )
            indices.append(selected_indices)
        token_indices[batch_idx] = torch.cat(indices)
    return token_indices


# %%
exclude_interactions = (
    self_intersection_mask(feats["asym_id"])
    | intra_mol_type_interaction_mask(feats["mol_type"], const.chain_type_ids["DNA"])
) | (~feats["token_pad_mask"].bool())
logprobs = contact_logprobs(
    pdistogram,
    exclude_interactions,
    cutoff_bin_mask(
        distance_cutoff=12, min_dist=2.0, max_dist=22.0, n_bins=pdistogram.shape[-1]
    ),
)
contact_profile = contact_scores(logprobs)

# %%
pdistogram.shape[-1]
# %%
mol_type_budget = {
    const.chain_type_ids["DNA"]: 30,
    const.chain_type_ids["RNA"]: 30,
    const.chain_type_ids["PROTEIN"]: 100,
}
token_indices = select_with_mol_type_budget(contact_profile, mol_type_budget, window=10)

# %%
token_indices

# %%

# Create a simple plot of the 1D scores
plt.figure(figsize=(12, 6))
batch_idx = 0
plt.plot(contact_profile[batch_idx].float().cpu().numpy(), "b-", linewidth=1)
# Find top k highest scores and add dots
top_k_scores = (
    contact_profile[batch_idx][token_indices[batch_idx].cpu().numpy()]
    .float()
    .cpu()
    .numpy()
)

# Add dots on top k points
plt.scatter(
    token_indices[batch_idx].cpu().numpy(),
    top_k_scores,
    color="red",
    s=50,
    zorder=5,
    alpha=0.8,
)

# Add labels for the top k points
for i, (idx, score) in enumerate(
    zip(token_indices[batch_idx].cpu().numpy(), top_k_scores)
):
    plt.annotate(
        f"{idx}",
        (idx, score),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
        alpha=0.7,
    )

# %%
print(logprobs.shape)
print(lse_pooled_scores.shape)

# %%

# Create a 2D heatmap of the scores
plt.figure(figsize=(12, 18))

# Convert scores to numpy for plotting
scores_2d = scores.squeeze().cpu().float().numpy()


plt.subplot(3, 1, 1)
# Plot the 2D heatmap
im = plt.imshow(scores_2d, cmap="viridis", aspect="auto", interpolation="nearest")
plt.colorbar(im, label="Interaction Score")
plt.xlabel("Token Index")
plt.ylabel("Token Index")
plt.title("2D Interaction Scores Heatmap")

# Overlay the top k points with alpha transparency
# Create a mask for the top k points
overlay = np.zeros_like(scores_2d)
for idx in top_k_indices:
    overlay[idx, :] = scores_2d[idx, :]  # row
    overlay[:, idx] = scores_2d[:, idx]  # column

# Create alpha mask - non-zero values get alpha=0.8, zeros get alpha=0
alpha_mask = np.where(overlay != 0, 0.8, 0)

# Overlay the top k rows/columns with red color and alpha
plt.imshow(
    overlay, cmap="Reds", aspect="auto", interpolation="nearest", alpha=alpha_mask
)

# Add grid lines at the top k indices for better visualization
for idx in top_k_indices:
    plt.axhline(y=idx, color="white", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axvline(x=idx, color="white", linestyle="--", alpha=0.3, linewidth=0.5)


plt.subplot(3, 1, 2)
plt.imshow(
    logprobs.squeeze().cpu().float().numpy(),
    cmap="viridis",
    aspect="auto",
    interpolation="nearest",
)
# Overlay the top k rows/columns with red color and alpha
plt.imshow(
    overlay, cmap="Reds", aspect="auto", interpolation="nearest", alpha=alpha_mask
)

# Add grid lines at the top k indices for better visualization
for idx in top_k_indices:
    plt.axhline(y=idx, color="white", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axvline(x=idx, color="white", linestyle="--", alpha=0.3, linewidth=0.5)
plt.colorbar(im, label="Interaction Score")
plt.xlabel("Token Index")
plt.ylabel("Token Index")
plt.title("2D Interaction Scores Heatmap")

plt.subplot(3, 1, 3)
plt.imshow(
    interaction_probs.squeeze().cpu().float().numpy(),
    cmap="viridis",
    aspect="auto",
    interpolation="nearest",
)
# Overlay the top k rows/columns with red color and alpha
plt.imshow(
    overlay, cmap="Reds", aspect="auto", interpolation="nearest", alpha=alpha_mask
)

# Add grid lines at the top k indices for better visualization
for idx in top_k_indices:
    plt.axhline(y=idx, color="white", linestyle="--", alpha=0.3, linewidth=0.5)
    plt.axvline(x=idx, color="white", linestyle="--", alpha=0.3, linewidth=0.5)
plt.colorbar(im, label="Interaction Score")
plt.xlabel("Token Index")
plt.ylabel("Token Index")
plt.title("2D Interaction Scores Heatmap")

plt.tight_layout()
plt.show()


# %%
