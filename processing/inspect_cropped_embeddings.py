#!/usr/bin/env python3
import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def _torch_load_cpu(path: str) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    # Writer sometimes saves a dict keyed by batch indices: {0: {...}, 1: {...}}
    # We normalize to the first sample if that's the case.
    if isinstance(obj, dict) and 0 in obj and isinstance(obj[0], dict):
        return obj[0]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unexpected .pt structure in {path}: {type(obj)}")


def _shape(x: Any) -> Optional[Tuple[int, ...]]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return tuple(int(d) for d in x.shape)
    if isinstance(x, np.ndarray):
        return tuple(int(d) for d in x.shape)
    if isinstance(x, (list, tuple)):
        try:
            return (len(x),)
        except Exception:
            return None
    return None


def _dtype(x: Any) -> Optional[str]:
    if isinstance(x, torch.Tensor):
        return str(x.dtype)
    if isinstance(x, np.ndarray):
        return str(x.dtype)
    return None


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _is_sorted_unique(indices: np.ndarray) -> bool:
    if indices.ndim != 1:
        return False
    diffs = np.diff(indices)
    return np.all(diffs > 0)


@dataclass
class FileSummary:
    chunk: Optional[int]
    file: str
    size_bytes: int
    mtime: str
    keys: List[str]
    z_key: Optional[str]
    z_dtype: Optional[str]
    z_shape: Optional[Tuple[int, ...]]
    z_is_square: Optional[bool]
    s_key: Optional[str]
    s_dtype: Optional[str]
    s_shape: Optional[Tuple[int, ...]]
    indices_present: bool
    indices_len: Optional[int]
    indices_min: Optional[int]
    indices_max: Optional[int]
    indices_sorted_unique: Optional[bool]
    token_is_dna_present: bool
    token_is_dna_len: Optional[int]
    token_is_dna_true: Optional[int]
    token_pad_mask_present: bool
    token_pad_mask_len: Optional[int]
    affinity_mapping: Dict[str, Any]


def summarize_file(path: str, chunk: Optional[int]) -> FileSummary:
    stat = os.stat(path)
    try:
        inner = _torch_load_cpu(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")

    keys = sorted(list(inner.keys()))

    # Detect z and s keys
    z_key = "z_crop" if "z_crop" in inner else ("z" if "z" in inner else None)
    s_key = "s_crop" if "s_crop" in inner else ("s_full" if "s_full" in inner else ("s" if "s" in inner else None))

    z = inner.get(z_key) if z_key else None
    s = inner.get(s_key) if s_key else None

    z_shape = _shape(z)
    s_shape = _shape(s)
    z_dtype = _dtype(z)
    s_dtype = _dtype(s)
    z_is_square = None
    if z_shape is not None and len(z_shape) == 3:
        z_is_square = (z_shape[0] == z_shape[1])

    # Indices checks
    indices_present = "indices" in inner
    indices_len = None
    indices_min = None
    indices_max = None
    indices_sorted_unique = None
    if indices_present:
        idx = _to_numpy(inner["indices"])
        idx = idx.astype(int).reshape(-1)
        indices_len = int(idx.shape[0])
        indices_min = int(idx.min()) if idx.size > 0 else None
        indices_max = int(idx.max()) if idx.size > 0 else None
        indices_sorted_unique = _is_sorted_unique(idx)

    # token_is_dna / token_pad_mask metadata if present
    token_is_dna_present = "token_is_dna" in inner
    token_is_dna_len = None
    token_is_dna_true = None
    if token_is_dna_present:
        dna = _to_numpy(inner["token_is_dna"]).astype(bool).reshape(-1)
        token_is_dna_len = int(dna.shape[0])
        token_is_dna_true = int(dna.sum())

    token_pad_mask_present = "token_pad_mask" in inner
    token_pad_mask_len = None
    if token_pad_mask_present:
        tpm = _to_numpy(inner["token_pad_mask"]).astype(bool).reshape(-1)
        token_pad_mask_len = int(tpm.shape[0])

    # Affinity head mapping notes
    affinity_mapping: Dict[str, Any] = {}
    affinity_mapping["head_input_z"] = {
        "source": z_key,
        "expected_by_head": "[L, L, token_z]",
        "observed_shape": z_shape,
        "square_pair_dim": z_is_square,
    }
    affinity_mapping["head_masks_from_feats"] = {
        "required_feats": ["token_pad_mask", "mol_type", "affinity_token_mask"],
        "present_in_file": [k for k in ["token_pad_mask", "mol_type", "affinity_token_mask"] if k in inner],
        "note": "Masks are used to build cross_pair_mask over z; typically not saved in cropped embeddings. If token_is_dna is present, DNA vs protein can be approximated.",
    }
    if token_is_dna_present:
        Lc = z_shape[0] if z_shape else None
        pd = int(token_is_dna_true) if token_is_dna_true is not None else None
        affinity_mapping["approx_roles"] = {
            "using_token_is_dna": True,
            "crop_length_Lc": Lc,
            "num_dna_tokens_in_crop": pd,
            "num_protein_tokens_in_crop": (int(token_is_dna_len - token_is_dna_true) if token_is_dna_len is not None and token_is_dna_true is not None else None),
            "cross_pairs_crop_mask_dim": "[Lc, Lc]",
        }
    # Map s to AffinityModule s_inputs (pre-head interaction). This file usually stores s_full.
    affinity_mapping["s_inputs_mapping"] = {
        "source": s_key,
        "expected_by_module": "[L_full, token_s] (before z conditioning)",
        "observed_shape": s_shape,
    }

    return FileSummary(
        chunk=chunk,
        file=path,
        size_bytes=int(stat.st_size),
        mtime=datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        keys=keys,
        z_key=z_key,
        z_dtype=z_dtype,
        z_shape=z_shape,
        z_is_square=z_is_square,
        s_key=s_key,
        s_dtype=s_dtype,
        s_shape=s_shape,
        indices_present=indices_present,
        indices_len=indices_len,
        indices_min=indices_min,
        indices_max=indices_max,
        indices_sorted_unique=indices_sorted_unique,
        token_is_dna_present=token_is_dna_present,
        token_is_dna_len=token_is_dna_len,
        token_is_dna_true=token_is_dna_true,
        token_pad_mask_present=token_pad_mask_present,
        token_pad_mask_len=token_pad_mask_len,
        affinity_mapping=affinity_mapping,
    )


def _iter_pt_files(root: str, chunks: Iterable[int], max_files_per_chunk: Optional[int]) -> Iterable[Tuple[Optional[int], str]]:
    for c in chunks:
        base = os.path.join(root, f"boltz_results_chunk_{c}", "predictions")
        # Files can be nested one level deeper by prediction ID directories
        patterns = [
            os.path.join(base, "*", "cropped_embeddings_*.pt"),
            os.path.join(base, "cropped_embeddings_*.pt"),
        ]
        matched: List[str] = []
        for p in patterns:
            matched.extend(glob.glob(p))
        matched = sorted(set(matched))
        if max_files_per_chunk is not None:
            matched = matched[: max_files_per_chunk]
        for m in matched:
            yield c, m


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cropped_embeddings_*.pt contents and map to Affinity head inputs")
    parser.add_argument("--root", type=str, default="/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs", help="Root directory containing boltz_results_chunk_*/predictions")
    parser.add_argument("--chunks", type=str, default="0,1,2,3,4", help="Comma-separated chunk indices to search")
    parser.add_argument("--max-files-per-chunk", type=int, default=30, help="Limit number of files per chunk (to inspect dozens, increase as needed)")
    parser.add_argument("--out-jsonl", type=str, default="cropped_embeddings_inspection.jsonl", help="Path to write JSONL summary")
    parser.add_argument("--out-csv", type=str, default="cropped_embeddings_inspection.csv", help="Optional CSV summary path")
    args = parser.parse_args()

    chunks = [int(x) for x in args.chunks.split(",") if x.strip()]

    rows: List[FileSummary] = []
    num_errors = 0

    for chunk, pt_path in _iter_pt_files(args.root, chunks, args.max_files_per_chunk):
        try:
            rows.append(summarize_file(pt_path, chunk))
        except Exception as e:
            num_errors += 1
            print(f"[WARN] {e}")  # noqa: T201

    # Write JSONL
    with open(args.out_jsonl, "w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    # Optional CSV (selected columns)
    if args.out_csv:
        try:
            import csv

            with open(args.out_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "chunk",
                        "file",
                        "size_bytes",
                        "mtime",
                        "z_key",
                        "z_dtype",
                        "z_shape",
                        "s_key",
                        "s_dtype",
                        "s_shape",
                        "indices_present",
                        "indices_len",
                        "indices_min",
                        "indices_max",
                        "indices_sorted_unique",
                        "token_is_dna_present",
                        "token_is_dna_len",
                        "token_is_dna_true",
                        "token_pad_mask_present",
                        "token_pad_mask_len",
                    ]
                )
                for r in rows:
                    writer.writerow(
                        [
                            r.chunk,
                            r.file,
                            r.size_bytes,
                            r.mtime,
                            r.z_key,
                            r.z_dtype,
                            "x".join(str(d) for d in (r.z_shape or ())) or None,
                            r.s_key,
                            r.s_dtype,
                            "x".join(str(d) for d in (r.s_shape or ())) or None,
                            r.indices_present,
                            r.indices_len,
                            r.indices_min,
                            r.indices_max,
                            r.indices_sorted_unique,
                            r.token_is_dna_present,
                            r.token_is_dna_len,
                            r.token_is_dna_true,
                            r.token_pad_mask_present,
                            r.token_pad_mask_len,
                        ]
                    )
        except Exception as e:
            print(f"[WARN] Failed to write CSV: {e}")  # noqa: T201

    # Human-readable summary
    print(f"[INFO] Wrote {len(rows)} summaries to {args.out_jsonl}; errors: {num_errors}")  # noqa: T201
    if rows:
        # Show a small preview
        preview = rows[: min(5, len(rows))]
        for r in preview:
            print(  # noqa: T201
                f"chunk={r.chunk} Lc={(r.z_shape[0] if r.z_shape else None)} zC={(r.z_shape[2] if (r.z_shape and len(r.z_shape)==3) else None)} s_dim={(r.s_shape[-1] if r.s_shape else None)} keys={r.keys} file={os.path.basename(r.file)}"
            )


if __name__ == "__main__":
    main()


