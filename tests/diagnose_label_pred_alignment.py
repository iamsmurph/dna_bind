"""Diagnose label vs prediction-dir alignment and dataset behavior.

This script mirrors the CLI preflight tolerant matching and inspects the
datamodule's splits to report exactly which prediction directories lack
corresponding labels. It also tries to iterate a few samples to surface
KeyErrors, printing the offending (uniprot, sequence) keys.

Usage (example):
  python -m tests.diagnose_label_pred_alignment \
    --labels_csv /path/to/labels.csv \
    --seq_col nt \
    --pred_glob "/path/to/runs/*/predictions/*" \
    --cache_dir /data/rbg/users/seanmurphy/dna_bind/runs_cache
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Set, Tuple

import csv
import glob

from dna_bind_offline.data.datamodule import AffinityDataModule, read_labels
from dna_bind_offline.io.bundle_loader import parse_prediction_dir_name


def tolerant_normalize(key: Tuple[str, str]) -> Tuple[str, str]:
    u, s = key
    return (u, s[:-1]) if s.endswith("T") else (u, s)


def read_label_keys(labels_csv: str, seq_col: str) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        seq_field = seq_col if seq_col in reader.fieldnames else ("sequence" if "sequence" in reader.fieldnames else None)
        if seq_field is None or "uniprot" not in reader.fieldnames or "intensity_log1p" not in reader.fieldnames:
            raise SystemExit(f"CSV missing required columns; found: {reader.fieldnames}")
        for row in reader:
            keys.add((row["uniprot"].strip(), row[seq_field].strip()))
    return keys


def read_pred_keys(pred_glob: str) -> Set[Tuple[str, str]]:
    keys: Set[Tuple[str, str]] = set()
    for d in glob.glob(pred_glob):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d.rstrip("/"))
        parts = name.split("_", 2)
        if len(parts) >= 3:
            u, _, s = parts[0], parts[1], parts[2]
            keys.add((u, s))
    return keys


def summarize_sets(labels_set: Set[Tuple[str, str]], preds_set: Set[Tuple[str, str]]) -> None:
    labels_norm = {tolerant_normalize(k) for k in labels_set}
    preds_norm = {tolerant_normalize(k) for k in preds_set}
    common = labels_norm & preds_norm
    only_labels = labels_norm - preds_norm
    only_preds = preds_norm - labels_norm
    print(f"[summary] labels={len(labels_set)} preds={len(preds_set)} common(tolerant)={len(common)} only_labels={len(only_labels)} only_preds={len(only_preds)}")
    # Show a few examples for each difference set
    def _head(xs: Iterable[Tuple[str, str]], n: int = 10) -> List[Tuple[str, str]]:
        out = []
        for x in xs:
            out.append(x)
            if len(out) >= n:
                break
        return out
    if only_labels:
        print("[only_labels examples]", _head(only_labels, 10))
    if only_preds:
        print("[only_preds examples]", _head(only_preds, 10))


def split_dirs_by_tf(valid_dirs: List[str], seed: int, val_frac: float) -> Tuple[List[str], List[str]]:
    import random
    from collections import defaultdict
    tf_to_dirs = defaultdict(list)
    for d in valid_dirs:
        try:
            u, _, _ = parse_prediction_dir_name(d)
        except Exception:
            u = ""
        tf_to_dirs[u].append(d)
    tf_ids = list(tf_to_dirs.keys())
    rnd = random.Random(seed)
    rnd.shuffle(tf_ids)
    n_val_tf = max(1, int(len(tf_ids) * val_frac))
    val_tfs = set(tf_ids[:n_val_tf])
    train_tfs = set(tf_ids[n_val_tf:])
    train_dirs = [d for tf in train_tfs for d in tf_to_dirs[tf]]
    val_dirs = [d for tf in val_tfs for d in tf_to_dirs[tf]]
    return train_dirs, val_dirs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--seq_col", default="nt")
    ap.add_argument("--pred_glob", required=True)
    ap.add_argument("--split_by", choices=["tf", "random"], default="tf")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache_dir", default="runs_cache")
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    # 1) Reproduce preflight tolerant summary
    labels_set = read_label_keys(args.labels_csv, args.seq_col)
    preds_set = read_pred_keys(args.pred_glob)
    summarize_sets(labels_set, preds_set)

    # 2) Build datamodule and let it compute valid prediction dirs (PD-edge prefilter)
    dm = AffinityDataModule(labels_csv=args.labels_csv,
                            pred_glob=args.pred_glob,
                            seq_col=args.seq_col,
                            val_frac=args.val_frac,
                            seed=args.seed,
                            num_workers=args.num_workers,
                            batch_size=1,
                            split_by=args.split_by,
                            normalize="none",
                            cache_dir=args.cache_dir,
                            cache_format="npz",
                            cache_in_mem=0,
                            cache_z_in_mem=0,
                            pin_memory=False,
                            prefetch_factor=2,
                            prefilter_cache_refresh=False,
                            prefilter_cache_off=False,
                            prefilter_workers=8,
                            prefilter_progress=True,
                            prefilter_verbose=False)
    dm.setup()

    # 3) For each split, compute which dirs have label keys and which don't
    labels_map = read_labels(args.labels_csv, seq_col=args.seq_col)
    label_keys = set(labels_map.keys())
    def dir_key(d: str) -> Tuple[str, str]:
        try:
            u, _, s = parse_prediction_dir_name(d)
        except Exception:
            # Fallback to directory basename parsing
            name = os.path.basename(d.rstrip("/"))
            parts = name.split("_", 2)
            if len(parts) >= 3:
                u, s = parts[0], parts[2]
            else:
                u, s = "", ""
        return (u, s)

    def report_split(name: str, dirs: List[str]) -> None:
        raw_keys = [dir_key(d) for d in dirs]
        norm_keys = [tolerant_normalize(k) for k in raw_keys]
        label_keys_norm = {tolerant_normalize(k) for k in label_keys}
        has_label = [k in label_keys for k in raw_keys]
        has_label_tol = [nk in label_keys_norm for nk in norm_keys]
        n = len(dirs)
        print(f"[{name}] dirs={n} with_label_raw={sum(has_label)} with_label_tolerant={sum(has_label_tol)}")
        # Show examples that fail even tolerant match
        missing_tol = [rk for rk, ok in zip(raw_keys, has_label_tol) if not ok]
        if missing_tol:
            print(f"[{name}] missing_tolerant examples (up to 10): {missing_tol[:10]}")

    if dm.train_ds is not None:
        report_split("train", dm.train_ds.pred_dirs)  # type: ignore[attr-defined]
    if dm.val_ds is not None:
        report_split("val", dm.val_ds.pred_dirs)  # type: ignore[attr-defined]

    # 4) Try to iterate a few samples from the train dataset and catch KeyErrors
    #    to show the exact offending (uniprot, sequence).
    if dm.train_ds is not None:
        ds = dm.train_ds
        max_checks = min(200, len(ds))
        errors = []
        for i in range(max_checks):
            try:
                _ = ds[i]
            except KeyError as e:
                errors.append(str(e))
            except Exception:
                # Ignore other exceptions for this diagnostic
                pass
        if errors:
            print(f"[dataset] Caught {len(errors)} KeyErrors while sampling first {max_checks} items.")
            print("[dataset] First few KeyErrors:")
            for msg in errors[:10]:
                print("  ", msg)
        else:
            print(f"[dataset] No KeyErrors observed while sampling first {max_checks} items.")


if __name__ == "__main__":
    main()


