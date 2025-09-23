#!/usr/bin/env python3
import argparse
import glob
import os
import random
import sys
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from collections import defaultdict
import warnings


# Pooling hyper-parameters – updated at runtime by run()
POOL_CFG = {
    "topk": 256,
    "alpha": 1.0,
    "beta": 0.5,
    "gamma": 0.2,
    "tau": 0.25,
}


@dataclass
class Record:
    uniprot: str
    sequence: str
    score_from_dirname: float
    z_path: str
    pred_dir: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_labels(csv_path: str) -> Dict[Tuple[str, str], float]:
    """Read CSV, return mapping (uniprot, sequence) -> intensity_log1p.

    The CSV header is expected to contain at least: uniprot, nt, intensity_log1p
    """
    labels: Dict[Tuple[str, str], float] = {}
    # Prefer pandas if available (faster), else fall back to csv module
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        if not {"uniprot", "nt", "intensity_log1p"}.issubset(df.columns):
            missing = {"uniprot", "nt", "intensity_log1p"} - set(df.columns)
            raise ValueError(f"CSV missing required columns: {missing}")
        for _, row in df.iterrows():
            labels[(str(row["uniprot"]), str(row["nt"]))] = float(row["intensity_log1p"])
    except Exception:  # noqa: BLE001
        import csv

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (str(row["uniprot"]), str(row["nt"]))
                labels[key] = float(row["intensity_log1p"])
    return labels


def find_prediction_dirs(pred_root: str) -> List[str]:
    """Return list of prediction directories containing cropped_embeddings_*.pt files."""
    pattern = os.path.join(pred_root, "boltz_results_chunk_*", "predictions", "*")
    return sorted(glob.glob(pattern))


def parse_prediction_dir_name(dir_path: str) -> Tuple[str, float, str]:
    """Parse basename into (uniprot, score, sequence).

    Expected format: UNIPROT_SCORE_SEQUENCE
    """
    name = os.path.basename(dir_path.rstrip("/"))
    parts = name.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Unexpected prediction directory name: {name}")
    uniprot = parts[0]
    try:
        score = float(parts[1])
    except ValueError as e:  # noqa: PERF203
        raise ValueError(f"Cannot parse score from directory name: {name}") from e
    sequence = parts[2]
    return uniprot, score, sequence


def locate_cropped_embeddings_file(pred_dir: str) -> str:
    paths = glob.glob(os.path.join(pred_dir, "cropped_embeddings_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No cropped_embeddings_*.pt in {pred_dir}")
    return paths[0]


def locate_best_cropped_embeddings_file(pred_dir: str) -> str:
    paths = sorted(glob.glob(os.path.join(pred_dir, "cropped_embeddings_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No cropped_embeddings_*.pt in {pred_dir}")
    contact_path = _locate_contact_probs_file(pred_dir)
    if not contact_path:
        # Fallback to first if no contact available
        return paths[0]
    data = np.load(contact_path)
    contact = data.get("contact_probs", None)
    if contact is None:
        contact = data[data.files[0]]
    best_path = paths[0]
    best_mass = -1.0
    for p in paths:
        try:
            inner = _load_cropped_embeddings(p)
            # Build full-length PD mask using indices if needed
            if "token_is_dna" in inner:
                is_dna = inner["token_is_dna"]
                if isinstance(is_dna, torch.Tensor):
                    is_dna = is_dna.detach().cpu().numpy()
                is_dna = is_dna.astype(bool)
                if is_dna.ndim == 1 and is_dna.shape[0] == contact.shape[0]:
                    # already full-length
                    is_prot = ~is_dna
                    mask_full = np.outer(is_prot, is_dna)
                else:
                    # crop-level -> expand via indices
                    if "indices" not in inner:
                        continue
                    indices = inner["indices"]
                    if isinstance(indices, torch.Tensor):
                        indices = indices.detach().cpu().numpy()
                    indices = indices.astype(int)
                    is_prot_crop = (~is_dna).astype(bool)
                    is_dna_crop = is_dna.astype(bool)
                    mask_crop = np.outer(is_prot_crop, is_dna_crop)
                    L = contact.shape[0]
                    mask_full = np.zeros((L, L), dtype=bool)
                    mask_full[np.ix_(indices, indices)] = mask_crop
            else:
                # No metadata: use all contacts
                mask_full = np.ones_like(contact, dtype=bool)
            mass = float(contact[mask_full].sum())
            if mass > best_mass:
                best_mass = mass
                best_path = p
        except Exception:
            continue
    return best_path


def _load_cropped_embeddings(pt_path: str) -> dict:
    obj = torch.load(pt_path, map_location="cpu")
    if isinstance(obj, dict) and 0 in obj and isinstance(obj[0], dict):
        inner = obj[0]
    elif isinstance(obj, dict):
        inner = obj
    else:
        raise ValueError(f"Unexpected .pt structure in {pt_path}")
    if "z" not in inner:
        raise KeyError(f"Missing key 'z' in {pt_path}")
    return inner


def _locate_contact_probs_file(pred_dir: str) -> str:
    paths = glob.glob(os.path.join(pred_dir, "contact_probs_*.npz"))
    return paths[0] if paths else ""


def _locate_file(pred_dir: str, stem: str) -> str:
    paths = glob.glob(os.path.join(pred_dir, f"{stem}_*.npz"))
    return paths[0] if paths else ""


def load_feature_from_pt(pt_path: str) -> np.ndarray:
    """Baseline feature: mean-pool z over (i,j) -> (128,)."""
    inner = _load_cropped_embeddings(pt_path)
    z: torch.Tensor = inner["z"]
    if z.ndim != 3:
        raise ValueError(f"Expected z to be 3D (N, N, C); got shape {tuple(z.shape)} in {pt_path}")
    pooled: torch.Tensor = z.mean(dim=(0, 1))
    return pooled.cpu().numpy().astype(np.float32)


def load_advanced_feature(pred_dir: str, pt_path: str, use_contact_probs: bool, pool_cfg: Optional[dict] = None) -> np.ndarray:
    """Interface-aware top-K pooled feature.

    Returns vector dim: 3*128 + 2*384 + 3 = 1155.
    """
    inner = _load_cropped_embeddings(pt_path)
    z: torch.Tensor = inner["z"]  # (N,N,128)
    if z.ndim != 3:
        raise ValueError("Expected z to be 3D (N,N,C)")
    N = z.shape[0]
    # metadata-based PD mask
    mask_pd = _compute_pd_mask(inner)
    if mask_pd is None:
        # fallback: use entire matrix
        mask_pd = np.ones((N, N), dtype=bool)

    # edge score matrix W (normalized over PD) and raw score/contact
    W = np.zeros((N, N), dtype=np.float32)
    score = None
    contact = None
    pae = None
    pde = None
    # Optional crop indices
    indices_np: Optional[np.ndarray] = None
    if "indices" in inner:
        idx = inner["indices"]
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        indices_np = idx.astype(int)
    if use_contact_probs:
        try:
            contact_path = _locate_file(pred_dir, "contact_probs")
            pae_path = _locate_file(pred_dir, "pae")
            pde_path = _locate_file(pred_dir, "pde")
            if contact_path:
                data = np.load(contact_path)
                contact = data.get("contact_probs", None)
                if contact is None:
                    contact = data[data.files[0]]
            # Maybe crop full matrices to the crop
            def _maybe_crop(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
                if mat is None:
                    return None
                if mat.shape[0] == N and mat.shape[1] == N:
                    return mat
                if indices_np is None:
                    return None
                return mat[np.ix_(indices_np, indices_np)]

            contact_crop = _maybe_crop(contact)
            pae = np.load(pae_path)["arr_0"] if pae_path else None
            pde = np.load(pde_path)["arr_0"] if pde_path else None
            pae_crop = _maybe_crop(pae) if pae is not None else None
            pde_crop = _maybe_crop(pde) if pde is not None else None

            if contact_crop is not None:
                cfg = POOL_CFG if pool_cfg is None else pool_cfg
                alpha = cfg["alpha"]
                beta = cfg["beta"]
                gamma = cfg["gamma"]
                tau = cfg["tau"]
                score = (
                    alpha * (contact_crop - tau)
                    - beta * (pae_crop if pae_crop is not None else 0.0)
                    - gamma * (pde_crop if pde_crop is not None else 0.0)
                ).astype(np.float32)
                mflat = mask_pd.flatten()
                sflat = score.flatten()
                sflat_masked = sflat[mflat]
                if sflat_masked.size > 0:
                    exp_s = np.exp(sflat_masked - np.max(sflat_masked))
                    wflat = exp_s / max(exp_s.sum(), 1e-9)
                    W_flat = np.zeros_like(sflat, dtype=np.float32)
                    W_flat[mflat] = wflat
                    W = W_flat.reshape(N, N)
                else:
                    W[...] = mask_pd.astype(np.float32)
            else:
                W[...] = mask_pd.astype(np.float32)
        except Exception:
            W[...] = mask_pd.astype(np.float32)
    else:
        W[...] = mask_pd.astype(np.float32)

    # Select top-K edges by masked count, using score if available
    mflat = mask_pd.flatten()
    num_masked = int(mflat.sum())
    cfg = POOL_CFG if pool_cfg is None else pool_cfg
    K = min(cfg["topk"], num_masked if num_masked > 0 else N * N)
    if K <= 0:
        return np.zeros((3 * z.shape[-1] + 2 * 384 + 3,), dtype=np.float32)
    base_flat = (score if score is not None else W).flatten()
    masked_positions = np.nonzero(mflat)[0]
    base_flat_masked = base_flat[mflat]
    if base_flat_masked.size == 0:
        return np.zeros((3 * z.shape[-1] + 2 * 384 + 3,), dtype=np.float32)
    top_in_mask = np.argpartition(base_flat_masked, -K)[-K:]
    top_idx = masked_positions[top_in_mask]
    # weights normalized over top-K
    W_flat = W.flatten()
    weights = W_flat[top_idx][:, None]
    weights_sum = float(weights.sum())
    if weights_sum <= 0:
        weights = np.full_like(weights, 1.0 / max(K, 1), dtype=np.float32)
    else:
        weights = weights / weights_sum
    z_flat = z.detach().cpu().numpy().astype(np.float32).reshape(-1, z.shape[-1])  # (N*N,C)
    Zk = z_flat[top_idx]  # (K,C)
    # stats
    z_wmean = (weights * Zk).sum(0)
    z_wmax = Zk.max(0)
    z_wstd = np.sqrt((weights * (Zk - z_wmean) ** 2).sum(0))

    # Token pools using s_full if available
    s_prot_pool = np.zeros((384,), dtype=np.float32)
    s_dna_pool = np.zeros((384,), dtype=np.float32)
    if "s_full" in inner and "indices" in inner and "token_is_dna" in inner:
        s_full = inner["s_full"]
        if isinstance(s_full, torch.Tensor):
            s_full = s_full.detach().cpu().numpy()
        if s_full.ndim == 3 and s_full.shape[0] == 1:
            s_full = s_full[0]
        indices_np = inner["indices"]
        if isinstance(indices_np, torch.Tensor):
            indices_np = indices_np.detach().cpu().numpy()
        indices_np = indices_np.astype(int)
        is_dna_full = inner["token_is_dna"]
        if isinstance(is_dna_full, torch.Tensor):
            is_dna_full = is_dna_full.detach().cpu().numpy()
        is_dna_full = is_dna_full.astype(bool)
        is_prot_full = ~is_dna_full
        rows = top_idx // N
        cols = top_idx % N
        token_set = set(rows.tolist() + cols.tolist())
        token_idx = np.array(sorted(token_set), dtype=int)
        full_idx = indices_np[token_idx]
        prot_mask_tokens = is_prot_full[full_idx]
        dna_mask_tokens = is_dna_full[full_idx]
        if prot_mask_tokens.any():
            s_prot_pool = s_full[full_idx[prot_mask_tokens]].mean(0).astype(np.float32)
        if dna_mask_tokens.any():
            s_dna_pool = s_full[full_idx[dna_mask_tokens]].mean(0).astype(np.float32)

    # mean contact and median pae/pde from selected edges if available
    mean_contact = np.array([0.0], dtype=np.float32)
    rows = top_idx // N
    cols = top_idx % N
    try:
        if 'contact_crop' in locals() and contact_crop is not None:
            mean_contact = np.array([contact_crop[rows, cols].mean()], dtype=np.float32)
    except Exception:
        pass
    # For pae/pde medians
    median_pae = np.array([0.0], dtype=np.float32)
    median_pde = np.array([0.0], dtype=np.float32)
    try:
        if use_contact_probs and 'pae_crop' in locals() and pae_crop is not None:
            median_pae[...] = np.median(pae_crop[rows, cols]).astype(np.float32)
        if use_contact_probs and 'pde_crop' in locals() and pde_crop is not None:
            median_pde[...] = np.median(pde_crop[rows, cols]).astype(np.float32)
    except Exception:
        pass

    feature = np.concatenate([
        z_wmean,
        z_wmax,
        z_wstd,
        s_prot_pool,
        s_dna_pool,
        mean_contact,
        median_pae,
        median_pde,
    ], axis=0).astype(np.float32)
    return feature


def build_dataset(
    pred_root: str,
    labels_map: Dict[Tuple[str, str], float],
    subset: int = 0,
    workers: int = 0,
    advanced_pooling: bool = False,
    use_contact_probs: bool = False,
    pool_cfg: Optional[dict] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Scan prediction dirs, load features, and assemble X, y, groups (uniprot).

    groups is a list aligned with rows of X and y, used for grouped splits.
    """
    records: List[Record] = []
    pred_dirs = find_prediction_dirs(pred_root)
    if subset and subset > 0:
        pred_dirs = pred_dirs[:subset]

    for d in pred_dirs:
        try:
            uniprot, score, sequence = parse_prediction_dir_name(d)
        except Exception:
            continue
        key = (uniprot, sequence)
        if key not in labels_map:
            # skip if we cannot find label
            continue
        try:
            pt_path = locate_best_cropped_embeddings_file(d)
        except FileNotFoundError:
            continue
        records.append(Record(uniprot=uniprot, sequence=sequence, score_from_dirname=score, z_path=pt_path, pred_dir=d))

    if not records:
        raise RuntimeError("No matching records found. Check pred_root and CSV mapping.")

    features: List[np.ndarray] = []
    labels: List[float] = []
    groups: List[str] = []

    # Parallel or serial feature extraction
    if workers and workers > 0:
        feature_slots: List[Optional[np.ndarray]] = [None] * len(records)
        success: List[bool] = [False] * len(records)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            if advanced_pooling:
                cfg_copy = (POOL_CFG if pool_cfg is None else pool_cfg).copy()
                future_to_idx = {ex.submit(load_advanced_feature, r.pred_dir, r.z_path, use_contact_probs, cfg_copy): i for i, r in enumerate(records)}
            else:
                future_to_idx = {ex.submit(load_feature_from_pt, r.z_path): i for i, r in enumerate(records)}
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    feature_slots[i] = fut.result()
                    success[i] = True
                except Exception:
                    success[i] = False
        for i, r in enumerate(records):
            if not success[i] or feature_slots[i] is None:
                continue
            features.append(feature_slots[i])
            labels.append(labels_map[(r.uniprot, r.sequence)])
            groups.append(r.uniprot)
    else:
        for r in records:
            try:
                if advanced_pooling:
                    feat = load_advanced_feature(r.pred_dir, r.z_path, use_contact_probs, pool_cfg)
                else:
                    feat = load_feature_from_pt(r.z_path)
            except Exception:
                continue
            features.append(feat)
            labels.append(labels_map[(r.uniprot, r.sequence)])
            groups.append(r.uniprot)

    if not features:
        raise RuntimeError("Failed to load any features from .pt files.")

    X = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return X, y, groups


def standardize_fit_transform(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    Xn = (X - mean) / std
    return Xn, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def split_groups(groups: Sequence[str], k_folds: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """Simple grouped K-fold: partition unique groups; return list of (train_idx, val_idx).

    groups: per-sample group labels aligned with dataset indices
    """
    unique_groups = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(unique_groups)
    folds: List[List[str]] = [[] for _ in range(k_folds)]
    for i, g in enumerate(unique_groups):
        folds[i % k_folds].append(g)

    indices = list(range(len(groups)))
    result: List[Tuple[List[int], List[int]]] = []
    for i in range(k_folds):
        val_groups = set(folds[i])
        train_idx = [idx for idx in indices if groups[idx] not in val_groups]
        val_idx = [idx for idx in indices if groups[idx] in val_groups]
        if not train_idx or not val_idx:
            continue
        result.append((train_idx, val_idx))
    return result


class LinearHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, D)
        out = self.linear(x)  # (B, output_dim)
        return out


class MLPHead(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, hidden: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(
    model: torch.nn.Module,
    X_t: torch.Tensor,
    y_t: torch.Tensor,
    opt: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    batch_size: int = 256,
    grad_clip: float = 1.0,
    predict_binary: bool = False,
    y_bin: Optional[torch.Tensor] = None,
    bce_weight: float = 0.3,
) -> float:
    """Train for one epoch using mini-batches."""
    model.train()
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    n = X_t.size(0)
    total_loss = 0.0
    nbatches = 0
    for i in range(0, n, batch_size):
        xb = X_t[i : i + batch_size]
        yb = y_t[i : i + batch_size]
        pred = model(xb)
        if pred.ndim == 1:
            pred = pred.unsqueeze(-1)
        reg_pred = pred[:, 0]
        loss = mse_loss_fn(reg_pred, yb)
        if predict_binary and y_bin is not None:
            ybb = y_bin[i : i + batch_size]
            bin_pred = pred[:, 1]
            loss = loss + bce_weight * bce_loss_fn(bin_pred, ybb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        nbatches += 1
    return total_loss / max(1, nbatches)


@torch.inference_mode()
def predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_t = torch.from_numpy(X)
    out = model(X_t)
    if out.ndim == 2 and out.shape[1] >= 1:
        out = out[:, 0]
    out = out.cpu().numpy().astype(np.float32)
    return out


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)


def spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    def rankdata(a: np.ndarray) -> np.ndarray:
        # average ranks for ties
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        # handle ties by averaging indices
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / counts
        return avg[inv]

    rx = rankdata(y_true)
    ry = rankdata(y_pred)
    rx = (rx - rx.mean()) / (rx.std() + 1e-12)
    ry = (ry - ry.mean()) / (ry.std() + 1e-12)
    return float(np.mean(rx * ry))


def _compute_pd_mask(inner: dict) -> Optional[np.ndarray]:
    """Return boolean (N,N) mask where True = protein row & DNA col edges.
    Tries to use metadata; returns None if cannot determine.
    """
    def _as_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x
    if "token_is_dna" in inner:
        is_dna = _as_numpy(inner["token_is_dna"]).astype(bool)
    elif "entity_type" in inner:
        et = _as_numpy(inner["entity_type"])  # assume 0=protein,1=DNA or similar
        is_dna = (et == 1)
    else:
        return None
    is_prot = ~is_dna
    mask = np.outer(is_prot, is_dna)
    return mask


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    print("[INFO] Loading labels from:", args.csv)
    labels_map = read_labels(args.csv)

    print("[INFO] Scanning predictions under:", args.pred_root)
    if args.workers and args.workers > 0:
        print(f"[INFO] Parallel feature extraction with workers={args.workers}")
    # Prepare pooling config from CLI overrides BEFORE building dataset
    POOL_CFG.update({
        "topk": args.topk,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "tau": args.tau,
    })

    X, y, groups = build_dataset(
        args.pred_root,
        labels_map,
        subset=args.subset,
        workers=args.workers,
        advanced_pooling=args.advanced_pooling,
        use_contact_probs=args.use_contact_probs,
        pool_cfg=POOL_CFG,
    )
    print(f"[INFO] Loaded dataset: X={X.shape}, y={y.shape}, groups={len(groups)}")

    # POOL_CFG already updated before dataset build so workers use overrides

    # Determine unique groups and adjust folds if necessary
    unique_groups_count = len(set(groups))
    print(f"[INFO] Unique groups (uniprot): {unique_groups_count}")
    k_folds = min(args.folds, max(1, unique_groups_count))

    folds = split_groups(groups, k_folds, seed=args.seed)
    # Fallback: if grouped folds cannot be constructed (e.g., only 1 group),
    # use a simple random 80/20 split without grouping.
    if not folds:
        print("[WARN] Grouped folds empty; falling back to random 80/20 split without grouping.")
        idx = np.arange(len(groups))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(idx)
        split = max(1, int(0.8 * len(idx)))
        train_idx = idx[:split].tolist()
        val_idx = idx[split:].tolist()
        if not val_idx:
            val_idx = train_idx[-1:]
            train_idx = train_idx[:-1]
        folds = [(train_idx, val_idx)]

    if args.audit_sample:
        sample_keys = random.sample(list(labels_map.keys()), k=min(20, len(labels_map)))
        print("[AUDIT] Sample key-label pairs:")
        for k in sample_keys:
            print(k, labels_map[k])

    all_metrics: List[Tuple[float, float, float]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        if args.standardize:
            X_train, mean, std = standardize_fit_transform(X_train)
            X_val = standardize_apply(X_val, mean, std)

        input_dim = X_train.shape[1]
        output_dim = 2 if args.predict_binary else 1
        model: torch.nn.Module
        if args.model == "linear":
            model = LinearHead(input_dim, output_dim=output_dim)
        elif args.model == "mlp":
            model = MLPHead(input_dim, output_dim=output_dim, hidden=args.hidden_dim, dropout=args.dropout)
        elif args.model == "ridge":
            try:
                from sklearn.linear_model import RidgeCV
            except ImportError:
                raise RuntimeError("scikit-learn is required for --model ridge")
            alphas = np.logspace(-6, 3, 10)
            ridge = RidgeCV(alphas=alphas, cv=3)
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_val).astype(np.float32)
            m_mse = mse(y_val, y_pred)
            m_r2 = r2(y_val, y_pred)
            m_s = spearmanr(y_val, y_pred)
            print(f"[FOLD {fold_idx+1}] Ridge MSE={m_mse:.4f} R2={m_r2:.4f} Spearman={m_s:.4f}")
            all_metrics.append((m_mse, m_r2, m_s))
            continue  # skip torch training path

        # Move data to tensors once
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)

        # Prepare binary labels if needed
        y_bin_train: Optional[np.ndarray] = None
        if args.predict_binary:
            if args.binary_threshold is None:
                threshold = float(np.median(y_train))
            else:
                threshold = float(args.binary_threshold)
            y_bin_train = (y_train >= threshold).astype(np.float32)

        # Optimizer & scheduler outside epoch loop
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        total_steps = args.epochs * math.ceil(len(X_train) / args.batch_size)
        warmup_steps = min(args.warmup_steps, total_steps // 10)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # Cosine decay
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        for epoch in range(args.epochs):
            loss = train_epoch(
                model,
                X_train_t,
                y_train_t,
                opt,
                scheduler=scheduler,
                batch_size=args.batch_size,
                grad_clip=args.grad_clip,
                predict_binary=args.predict_binary,
                y_bin=torch.from_numpy(y_bin_train) if y_bin_train is not None else None,
                bce_weight=args.bce_weight,
            )
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"[FOLD {fold_idx+1}] epoch {epoch+1}/{args.epochs} loss={loss:.4f} lr={current_lr:.2e}")

        y_pred = predict(model, X_val)
        m_mse = mse(y_val, y_pred)
        m_r2 = r2(y_val, y_pred)
        m_s = spearmanr(y_val, y_pred)
        print(f"[FOLD {fold_idx+1}] MSE={m_mse:.4f} R2={m_r2:.4f} Spearman={m_s:.4f}")
        all_metrics.append((m_mse, m_r2, m_s))

        # Per-TF Spearman
        tf_scores = []
        val_groups = np.array(groups)[val_idx]
        for g in sorted(set(val_groups)):
            mask = (val_groups == g)
            if mask.sum() >= 2:
                tf_scores.append(spearmanr(y_val[mask], y_pred[mask]))
        median_tf = float(np.nanmedian(tf_scores)) if tf_scores else float('nan')
        print(f"[FOLD {fold_idx+1}] Median per-TF Spearman={median_tf:.4f}")

    ms, rs, ss = zip(*all_metrics)
    print("\n[RESULT] Mean across folds:")
    print(f"MSE={np.mean(ms):.4f} ± {np.std(ms):.4f}")
    print(f"R2={np.mean(rs):.4f} ± {np.std(rs):.4f}")
    print(f"Spearman={np.mean(ss):.4f} ± {np.std(ss):.4f}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Simple affinity head training using pooled Boltz z embeddings.")
    parser.add_argument(
        "--pred_root",
        type=str,
        default="/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs",
        help="Root directory containing boltz_results_chunk_*/predictions/*",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/data/rbg/users/seanmurphy/dna_bind/data/uniprobe_subset_100tfs.csv",
        help="Path to labels CSV with columns: uniprot, nt, intensity_log1p",
    )
    parser.add_argument("--subset", type=int, default=0, help="Optional limit on number of prediction dirs to load")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5, help="Number of grouped folds")
    parser.add_argument("--model", type=str, choices=["linear", "mlp", "ridge"], default="linear")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--standardize", action="store_true", help="Standardize features per fold")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for feature extraction (0=serial)")
    parser.add_argument("--advanced_pooling", action="store_true", help="Use advanced feature pooling (z_mean, contact-weighted z, s_full mean, mean contact)")
    parser.add_argument("--use_contact_probs", action="store_true", help="Use contact_probs_*.npz to weight pair features if available")
    parser.add_argument("--predict_binary", action="store_true", help="Add a binary binder logit head and BCE loss")
    parser.add_argument("--bce_weight", type=float, default=0.3, help="Weight for BCE loss when --predict_binary is set")
    parser.add_argument("--binary_threshold", type=float, default=None, help="Threshold on y to form binary labels; default=median per fold")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size for the MLP head")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability in MLP head")
    parser.add_argument("--batch_size", type=int, default=512, help="Mini-batch size")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm value")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Linear warmup steps before cosine decay")
    parser.add_argument("--topk", type=int, default=256, help="Top-K edges for PD pooling")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha weight for contact term in edge score")
    parser.add_argument("--beta", type=float, default=0.5, help="Beta weight for PAE term in edge score")
    parser.add_argument("--gamma", type=float, default=0.2, help="Gamma weight for PDE term in edge score")
    parser.add_argument("--tau", type=float, default=0.25, help="Tau contact threshold in edge score")
    parser.add_argument("--audit_sample", action="store_true", help="Print 20 random (uniprot, nt, label) samples for manual audit")
    args = parser.parse_args(argv)

    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()


