#!/usr/bin/env python3
import argparse
import glob
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch


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


def load_feature_from_pt(pt_path: str) -> np.ndarray:
    """Baseline feature: mean-pool z over (i,j) -> (128,)."""
    inner = _load_cropped_embeddings(pt_path)
    z: torch.Tensor = inner["z"]
    if z.ndim != 3:
        raise ValueError(f"Expected z to be 3D (N, N, C); got shape {tuple(z.shape)} in {pt_path}")
    pooled: torch.Tensor = z.mean(dim=(0, 1))
    return pooled.cpu().numpy().astype(np.float32)


def load_advanced_feature(pred_dir: str, pt_path: str, use_contact_probs: bool) -> np.ndarray:
    """Advanced feature using z mean, contact-weighted z mean, s_full mean over crop, and mean contact.

    Returns a vector of size 128 (z_mean) + 128 (z_contact_mean) + 384 (s_mean_crop) + 1 (mean_contact) = 641.
    If contact_probs are unavailable or disabled, the contact-weighted parts are zeros.
    """
    inner = _load_cropped_embeddings(pt_path)
    z: torch.Tensor = inner["z"]  # (N, N, 128)
    if z.ndim != 3:
        raise ValueError(f"Expected z to be 3D (N, N, C); got shape {tuple(z.shape)} in {pt_path}")
    z_mean = z.mean(dim=(0, 1)).cpu().numpy().astype(np.float32)  # (128,)

    # s_full: (1, L, 384); indices: (N,)
    s_mean_crop = np.zeros((384,), dtype=np.float32)
    if "s_full" in inner and "indices" in inner:
        s_full: torch.Tensor = inner["s_full"]  # (1, L, 384)
        indices: torch.Tensor = inner["indices"]  # (N,)
        try:
            s_crop = s_full[0, indices.long(), :]  # (N, 384)
            s_mean_crop = s_crop.mean(dim=0).cpu().numpy().astype(np.float32)
        except Exception:
            s_mean_crop = s_full[0].mean(dim=0).cpu().numpy().astype(np.float32)
    else:
        # fallback: no s_full/indices
        pass

    # Contact-weighted z pooling
    z_contact_mean = np.zeros((z.shape[-1],), dtype=np.float32)
    mean_contact = np.zeros((1,), dtype=np.float32)
    if use_contact_probs and "indices" in inner:
        contact_path = _locate_contact_probs_file(pred_dir)
        if contact_path:
            try:
                data = np.load(contact_path)
                # try common keys, fallback to first array
                if "contact_probs" in data:
                    contact = data["contact_probs"]
                else:
                    # get the first array-like entry
                    first_key = next(k for k in data.files)
                    contact = data[first_key]
                indices_np = inner["indices"].detach().cpu().numpy().astype(np.int64)
                W = contact[np.ix_(indices_np, indices_np)].astype(np.float32)  # (N, N)
                W_sum = float(W.sum())
                mean_contact[...] = np.array([W.mean() if W.size > 0 else 0.0], dtype=np.float32)
                if W_sum > 0:
                    # weight z by W
                    z_np = z.detach().cpu().numpy().astype(np.float32)  # (N, N, C)
                    z_weighted = (z_np * W[:, :, None]).sum(axis=(0, 1)) / W_sum
                    z_contact_mean = z_weighted.astype(np.float32)
            except Exception:
                pass

    feature = np.concatenate([z_mean, z_contact_mean, s_mean_crop, mean_contact], axis=0)
    return feature


def build_dataset(
    pred_root: str,
    labels_map: Dict[Tuple[str, str], float],
    subset: int = 0,
    workers: int = 0,
    advanced_pooling: bool = False,
    use_contact_probs: bool = False,
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
            pt_path = locate_cropped_embeddings_file(d)
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
                future_to_idx = {ex.submit(load_advanced_feature, r.pred_dir, r.z_path, use_contact_probs): i for i, r in enumerate(records)}
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
                    feat = load_advanced_feature(r.pred_dir, r.z_path, use_contact_probs)
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
    def __init__(self, input_dim: int, output_dim: int = 1, hidden: int = 256) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    predict_binary: bool = False,
    y_bin: Optional[np.ndarray] = None,
    bce_weight: float = 0.3,
) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    pred = model(X_t)
    if pred.ndim == 1:
        pred = pred.unsqueeze(-1)
    reg_pred = pred[:, 0]
    loss = mse_loss_fn(reg_pred, y_t)
    if predict_binary and y_bin is not None:
        yb_t = torch.from_numpy(y_bin)
        bin_pred = pred[:, 1]
        loss = loss + bce_weight * bce_loss_fn(bin_pred, yb_t)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.item())


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


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    print("[INFO] Loading labels from:", args.csv)
    labels_map = read_labels(args.csv)

    print("[INFO] Scanning predictions under:", args.pred_root)
    if args.workers and args.workers > 0:
        print(f"[INFO] Parallel feature extraction with workers={args.workers}")
    X, y, groups = build_dataset(
        args.pred_root,
        labels_map,
        subset=args.subset,
        workers=args.workers,
        advanced_pooling=args.advanced_pooling,
        use_contact_probs=args.use_contact_probs,
    )
    print(f"[INFO] Loaded dataset: X={X.shape}, y={y.shape}, groups={len(groups)}")

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
        else:
            model = MLPHead(input_dim, output_dim=output_dim, hidden=args.hidden_dim)

        # Prepare binary labels if needed
        y_bin_train: Optional[np.ndarray] = None
        if args.predict_binary:
            if args.binary_threshold is None:
                threshold = float(np.median(y_train))
            else:
                threshold = float(args.binary_threshold)
            y_bin_train = (y_train >= threshold).astype(np.float32)

        for epoch in range(args.epochs):
            loss = train_epoch(
                model,
                X_train,
                y_train,
                lr=args.lr,
                predict_binary=args.predict_binary,
                y_bin=y_bin_train,
                bce_weight=args.bce_weight,
            )
            if (epoch + 1) % max(1, args.epochs // 5) == 0:
                print(f"[FOLD {fold_idx+1}] epoch {epoch+1}/{args.epochs} loss={loss:.4f}")

        y_pred = predict(model, X_val)
        m_mse = mse(y_val, y_pred)
        m_r2 = r2(y_val, y_pred)
        m_s = spearmanr(y_val, y_pred)
        print(f"[FOLD {fold_idx+1}] MSE={m_mse:.4f} R2={m_r2:.4f} Spearman={m_s:.4f}")
        all_metrics.append((m_mse, m_r2, m_s))

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
    parser.add_argument("--model", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--standardize", action="store_true", help="Standardize features per fold")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for feature extraction (0=serial)")
    parser.add_argument("--advanced_pooling", action="store_true", help="Use advanced feature pooling (z_mean, contact-weighted z, s_full mean, mean contact)")
    parser.add_argument("--use_contact_probs", action="store_true", help="Use contact_probs_*.npz to weight pair features if available")
    parser.add_argument("--predict_binary", action="store_true", help="Add a binary binder logit head and BCE loss")
    parser.add_argument("--bce_weight", type=float, default=0.3, help="Weight for BCE loss when --predict_binary is set")
    parser.add_argument("--binary_threshold", type=float, default=None, help="Threshold on y to form binary labels; default=median per fold")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden size for the MLP head")
    args = parser.parse_args(argv)

    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()


