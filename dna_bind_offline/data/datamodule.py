"""Lightning DataModule for dna_bind_offline."""

from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import pytorch_lightning as pl

from .dataset import AffinityDataset, Sample
from ..io.bundle_loader import parse_prediction_dir_name, load_crop_bundle, locate_any, locate_cif
from ..geometry.masks import make_crop_masks
from ..io.cif_parser import parse_cif_to_token_geom
from ..utils.hashing_cache import (compute_dir_signature, stable_global_hash,
                                   atomic_write_json, read_json,
                                   try_acquire_lock, release_lock, wait_for_file)


def read_labels(csv_path: str, seq_col: str = "nt") -> Dict[Tuple[str, str], float]:
    import csv
    mapping: Dict[Tuple[str, str], float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uniprot = row["uniprot"].strip()
            seq = row.get(seq_col, row.get("sequence", "")).strip()
            y = float(row["intensity_log1p"])
            mapping[(uniprot, seq)] = y
    return mapping


def single_collate(batch: List[Sample]) -> Sample:
    # Return the Sample as-is. When pin_memory=True, DataLoader will pin this
    # object in the main process by calling its .pin_memory() method.
    return batch[0]


class AffinityDataModule(pl.LightningDataModule):
    def __init__(self,
                 labels_csv: str,
                 pred_glob: str,
                 seq_col: str = "nt",
                 val_frac: float = 0.1,
                 seed: int = 0,
                 num_workers: int = 0,
                 batch_size: int = 1,
                 split_by: str = "tf",
                 normalize: str = "none",
                 test_glob: Optional[str] = None,
                 test_labels_csv: Optional[str] = None,
                 dist_feats: str = "rbf",
                 rbf_centers: int = 64,
                 rbf_min: float = 2.0,
                 rbf_max: float = 22.0,
                 rbf_sigma: float = 1.0,
                 cache_dir: Optional[str] = None,
                 cache_format: str = "npz",
                 cache_in_mem: int = 0,
                 cache_z_in_mem: int = 0,
                 pin_memory: bool = True,
                 prefetch_factor: int = 4,
                 prefilter_cache_refresh: bool = False,
                 prefilter_cache_off: bool = False,
                 prefilter_workers: int = 8,
                 prefilter_progress: bool = True,
                 prefilter_verbose: bool = False) -> None:
        super().__init__()
        self.labels_csv = labels_csv
        self.pred_glob = pred_glob
        self.seq_col = seq_col
        self.val_frac = float(val_frac)
        self.seed = int(seed)
        self.num_workers = int(num_workers)
        self.batch_size = int(batch_size)
        self.split_by = split_by
        self.normalize = normalize
        self.test_glob = test_glob
        self.test_labels_csv = test_labels_csv
        self.dist_feats = dist_feats
        self.rbf_centers = int(rbf_centers)
        self.rbf_min = float(rbf_min)
        self.rbf_max = float(rbf_max)
        self.rbf_sigma = float(rbf_sigma)
        self.cache_dir = cache_dir or ""
        self.cache_format = cache_format
        self.cache_in_mem = int(cache_in_mem)
        self.cache_z_in_mem = int(cache_z_in_mem)
        self.pin_memory = bool(pin_memory)
        self.prefetch_factor = int(prefetch_factor)
        self.prefilter_cache_refresh = bool(prefilter_cache_refresh)
        self.prefilter_cache_off = bool(prefilter_cache_off)
        self.prefilter_workers = int(max(1, prefilter_workers))
        self.prefilter_progress = True
        self.prefilter_verbose = bool(prefilter_verbose)
        self.labels: Dict[Tuple[str, str], float] = {}
        self.train_ds: Optional[AffinityDataset] = None
        self.val_ds: Optional[AffinityDataset] = None
        self.test_ds: Optional[AffinityDataset] = None
        self.train_stats: Dict[str, Tuple[float, float]] = {}

        # Pipeline assumes single-sample batches due to single_collate; guard against silent misuse
        if self.batch_size != 1:
            raise ValueError("This pipeline currently assumes batch_size=1 due to single_collate; got batch_size!=1")

    def prepare_data(self) -> None:
        # Place for one-time, rank-0 work (e.g., building/refreshing prefilter index on disk)
        # Our setup() builds/refreshes the index as needed; leaving this as a no-op keeps behavior unchanged
        # while allowing Lightning to call it only once per run on rank 0.
        return

    def setup(self, stage: Optional[str] = None) -> None:
        # Idempotent: if datasets are already constructed, avoid re-running prefilter/index
        if self.train_ds is not None and self.val_ds is not None:
            return
        import glob as globlib
        self.labels = read_labels(self.labels_csv, seq_col=self.seq_col)
        all_dirs = sorted(globlib.glob(os.path.join(self.pred_glob)))
        if not all_dirs:
            raise RuntimeError(f"No prediction dirs matched {self.pred_glob}")

        # Prefilter index path
        index_path = os.path.join(self.cache_dir or ".", "prefilter_index.json") if self.cache_dir else ""
        use_index = bool(self.cache_dir) and not self.prefilter_cache_off

        def _quick_files_for_dir(pred_dir: str) -> Dict[str, Optional[str]]:
            # Cheap file discovery via glob; avoids loading arrays
            pt_glob = sorted(__import__("glob").glob(os.path.join(pred_dir, "cropped_embeddings_*.pt")))
            pt_path = pt_glob[0] if pt_glob else None
            return {
                "pt": pt_path,
                "contact": locate_any(pred_dir, ["contact_probs"]),
                "pae": locate_any(pred_dir, ["pae", "full_pae", "pae_full"]),
                "pde": locate_any(pred_dir, ["pde", "full_pde", "pde_full"]),
                "cif": locate_cif(pred_dir),
            }

        def _heavy_validate(pred_dir: str) -> bool:
            try:
                bundle = load_crop_bundle(pred_dir, device=torch.device("cpu"))
                cif = bundle.meta.get("cif_path")
                if not cif or not os.path.exists(cif):
                    if self.prefilter_verbose:
                        print(f"[prefilter] invalid(no_cif): {pred_dir}")
                    return False
                try:
                    geom = parse_cif_to_token_geom(cif)
                except Exception as e:
                    if self.prefilter_verbose:
                        print(f"[prefilter] invalid(parse_error): {pred_dir} err={e!r}")
                    return False
                try:
                    masks = make_crop_masks(bundle.crop_to_full, geom, bundle.contact_probs, bundle.pae, bundle.pde)
                except Exception as e:
                    if self.prefilter_verbose:
                        print(f"[prefilter] invalid(mask_error): {pred_dir} err={e!r}")
                    return False
                ok = bool(np.any(masks.affinity_pair_mask))
                if (not ok) and self.prefilter_verbose:
                    print(f"[prefilter] invalid(no_pd_edges): {pred_dir}")
                return ok
            except Exception as e:
                if self.prefilter_verbose:
                    print(f"[prefilter] invalid(loader_error): {pred_dir} err={e!r}")
                return False

        valid_dirs: List[str] = []
        if not use_index:
            # Legacy full scan
            # Parallel I/O-bound validation with a modest pool
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                workers = max(1, min(self.prefilter_workers, 32))
                pbar = None
                if self.prefilter_progress:
                    try:
                        from tqdm import tqdm  # type: ignore
                        pbar = tqdm(total=len(all_dirs), desc="prefilter: validate", smoothing=0.1)
                    except Exception:
                        pbar = None
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    fut_to_dir = {ex.submit(_heavy_validate, d): d for d in all_dirs}
                    for fut in as_completed(fut_to_dir):
                        d = fut_to_dir[fut]
                        ok = False
                        try:
                            ok = bool(fut.result())
                        except Exception:
                            ok = False
                        if ok:
                            valid_dirs.append(d)
                        if pbar is not None:
                            pbar.update(1)
                if pbar is not None:
                    pbar.close()
            except Exception:
                for d in all_dirs:
                    if _heavy_validate(d):
                        valid_dirs.append(d)
        else:
            # Try to load and validate existing index
            index = read_json(index_path) if (not self.prefilter_cache_refresh) else None
            dirs_state: Dict[str, Dict[str, object]] = {}
            if isinstance(index, dict) and "dirs" in index and "meta" in index:
                meta = index.get("meta", {})
                saved_dirs = index.get("dirs", {})
                # Recompute signatures cheaply and decide which dirs changed
                changed: List[str] = []
                # Compute signatures in parallel for speed
                try:
                    from concurrent.futures import ThreadPoolExecutor
                    workers = max(1, min(self.prefilter_workers, 32))
                    def _sig_task(d: str):
                        files = _quick_files_for_dir(d)
                        sig = compute_dir_signature(files)
                        return d, files, sig
                    pbar = None
                    if self.prefilter_progress:
                        try:
                            from tqdm import tqdm  # type: ignore
                            pbar = tqdm(total=len(all_dirs), desc="prefilter: signatures", smoothing=0.1)
                        except Exception:
                            pbar = None
                    with ThreadPoolExecutor(max_workers=workers) as ex:
                        for d, files, sig in ex.map(_sig_task, all_dirs):
                            prev = saved_dirs.get(d)
                            prev_sig = prev.get("sig") if isinstance(prev, dict) else None
                            if prev_sig != sig:
                                changed.append(d)
                            dirs_state[d] = {"sig": sig, "files": files, "valid": bool(prev.get("valid")) if isinstance(prev, dict) else False}
                            if pbar is not None:
                                pbar.update(1)
                    if pbar is not None:
                        pbar.close()
                except Exception:
                    for d in all_dirs:
                        files = _quick_files_for_dir(d)
                        sig = compute_dir_signature(files)
                        prev = saved_dirs.get(d)
                        prev_sig = prev.get("sig") if isinstance(prev, dict) else None
                        if prev_sig != sig:
                            changed.append(d)
                        dirs_state[d] = {"sig": sig, "files": files, "valid": bool(prev.get("valid")) if isinstance(prev, dict) else False}
                # If counts differ or many changed, we still can incrementally validate
                # Heavy-validate only changed/new dirs
                n_changed = len(changed)
                if n_changed:
                    # Validate changed dirs in parallel
                    try:
                        from concurrent.futures import ThreadPoolExecutor
                        workers = max(1, min(self.prefilter_workers, 32))
                        pbar = None
                        if self.prefilter_progress:
                            try:
                                from tqdm import tqdm  # type: ignore
                                pbar = tqdm(total=len(changed), desc="prefilter: revalidate", smoothing=0.1)
                            except Exception:
                                pbar = None
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            for d, ok in ex.map(lambda x: (x, _heavy_validate(x)), changed):
                                dirs_state[d]["valid"] = bool(ok)
                                if pbar is not None:
                                    pbar.update(1)
                        if pbar is not None:
                            pbar.close()
                    except Exception:
                        for d in changed:
                            is_valid = _heavy_validate(d)
                            dirs_state[d]["valid"] = bool(is_valid)
                # Compute global hash
                sig_map = {d: dirs_state[d]["sig"] for d in dirs_state}
                new_hash = stable_global_hash(sig_map)
                global_ok = (meta.get("pred_glob") == self.pred_glob) and (int(meta.get("dir_count", -1)) == len(all_dirs))
                if not global_ok:
                    # Force rebuild all entries incrementally (changed includes all)
                    # Validate all dirs in parallel
                    try:
                        from concurrent.futures import ThreadPoolExecutor
                        workers = max(1, min(self.prefilter_workers, 32))
                        pbar = None
                        if self.prefilter_progress:
                            try:
                                from tqdm import tqdm  # type: ignore
                                pbar = tqdm(total=len(all_dirs), desc="prefilter: validate(all)", smoothing=0.1)
                            except Exception:
                                pbar = None
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            for d, ok in ex.map(lambda x: (x, _heavy_validate(x)), all_dirs):
                                dirs_state[d]["valid"] = bool(ok)
                                if pbar is not None:
                                    pbar.update(1)
                        if pbar is not None:
                            pbar.close()
                    except Exception:
                        for d in all_dirs:
                            is_valid = _heavy_validate(d)
                            dirs_state[d]["valid"] = bool(is_valid)
                    new_hash = stable_global_hash({d: dirs_state[d]["sig"] for d in dirs_state})
                # Write back updated index atomically with a lock
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                lock_dir = os.path.join(os.path.dirname(index_path), ".prefilter.lock")
                if try_acquire_lock(lock_dir):
                    try:
                        obj = {
                            "meta": {
                                "pred_glob": self.pred_glob,
                                "dir_count": len(all_dirs),
                                "global_hash": new_hash,
                                "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                                "tool_version": 1,
                            },
                            "dirs": {d: {"sig": dirs_state[d]["sig"], "valid": bool(dirs_state[d]["valid"]), "files": dirs_state[d]["files"]} for d in dirs_state},
                        }
                        atomic_write_json(index_path, obj)
                    finally:
                        release_lock(lock_dir)
                else:
                    # Another process is writing; wait for file then read
                    wait_for_file(index_path, timeout_s=1800.0)
                    index = read_json(index_path)
                    if isinstance(index, dict) and "dirs" in index:
                        saved_dirs = index.get("dirs", {})
                        for d in all_dirs:
                            prev = saved_dirs.get(d)
                            if isinstance(prev, dict):
                                dirs_state[d] = {"sig": prev.get("sig"), "files": prev.get("files"), "valid": bool(prev.get("valid"))}
                valid_dirs = [d for d in all_dirs if bool(dirs_state.get(d, {}).get("valid", False))]
            else:
                # No usable index; build from scratch with lock
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                lock_dir = os.path.join(os.path.dirname(index_path), ".prefilter.lock")
                if try_acquire_lock(lock_dir):
                    try:
                        dirs_state = {}
                        try:
                            from concurrent.futures import ThreadPoolExecutor
                            workers = max(1, min(self.prefilter_workers, 32))
                            def _full_task(d: str):
                                files = _quick_files_for_dir(d)
                                sig = compute_dir_signature(files)
                                ok = _heavy_validate(d)
                                return d, files, sig, bool(ok)
                            pbar = None
                            if self.prefilter_progress:
                                try:
                                    from tqdm import tqdm  # type: ignore
                                    pbar = tqdm(total=len(all_dirs), desc="prefilter: full build", smoothing=0.1)
                                except Exception:
                                    pbar = None
                            with ThreadPoolExecutor(max_workers=workers) as ex:
                                for d, files, sig, ok in ex.map(_full_task, all_dirs):
                                    dirs_state[d] = {"sig": sig, "files": files, "valid": bool(ok)}
                                    if pbar is not None:
                                        pbar.update(1)
                            if pbar is not None:
                                pbar.close()
                        except Exception:
                            for d in all_dirs:
                                files = _quick_files_for_dir(d)
                                sig = compute_dir_signature(files)
                                is_valid = _heavy_validate(d)
                                dirs_state[d] = {"sig": sig, "files": files, "valid": bool(is_valid)}
                        sig_map = {d: dirs_state[d]["sig"] for d in dirs_state}
                        new_hash = stable_global_hash(sig_map)
                        obj = {
                            "meta": {
                                "pred_glob": self.pred_glob,
                                "dir_count": len(all_dirs),
                                "global_hash": new_hash,
                                "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
                                "tool_version": 1,
                            },
                            "dirs": {d: {"sig": dirs_state[d]["sig"], "valid": bool(dirs_state[d]["valid"]), "files": dirs_state[d]["files"]} for d in dirs_state},
                        }
                        atomic_write_json(index_path, obj)
                    finally:
                        release_lock(lock_dir)
                    valid_dirs = [d for d in all_dirs if bool(dirs_state.get(d, {}).get("valid", False))]
                else:
                    # Another process is building; wait then reuse
                    wait_for_file(index_path, timeout_s=1800.0)
                    index = read_json(index_path)
                    saved_dirs = index.get("dirs", {}) if isinstance(index, dict) else {}
                    valid_dirs = [d for d in all_dirs if bool(saved_dirs.get(d, {}).get("valid", False))]

        if not valid_dirs:
            raise RuntimeError("No valid prediction dirs after prefiltering for PD edges")

        if self.split_by not in ("random", "tf"):
            self.split_by = "tf"
        rnd = random.Random(self.seed)

        if self.split_by == "random":
            rnd.shuffle(valid_dirs)
            n_val = max(1, int(len(valid_dirs) * self.val_frac))
            val_dirs = valid_dirs[:n_val]
            train_dirs = valid_dirs[n_val:]
        else:
            tf_to_dirs = defaultdict(list)
            for d in valid_dirs:
                try:
                    uniprot, _, seq = parse_prediction_dir_name(d)
                except Exception:
                    try:
                        bundle = load_crop_bundle(d, device=torch.device("cpu"))
                        uniprot = bundle.meta.get("uniprot", "")
                    except Exception:
                        uniprot = ""
                tf_to_dirs[uniprot].append(d)
            tf_ids = list(tf_to_dirs.keys())
            rnd.shuffle(tf_ids)
            n_val_tf = max(1, int(len(tf_ids) * self.val_frac))
            val_tfs = set(tf_ids[:n_val_tf])
            train_tfs = set(tf_ids[n_val_tf:])
            train_dirs = [d for tf in train_tfs for d in tf_to_dirs[tf]]
            val_dirs = [d for tf in val_tfs for d in tf_to_dirs[tf]]

        # Filter dirs to those with labels (with trailing-T tolerant normalization)
        def _normalize_key(k: Tuple[str, str]) -> Tuple[str, str]:
            u, s = k
            return (u, s[:-1]) if s.endswith("T") else (u, s)

        label_keys_raw = set(self.labels.keys())
        label_keys_tol = {_normalize_key(k) for k in label_keys_raw}

        def _dir_key(d: str) -> Tuple[str, str]:
            try:
                u, _, s = parse_prediction_dir_name(d)
                return (u, s)
            except Exception:
                try:
                    bundle = load_crop_bundle(d, device=torch.device("cpu"))
                    return (str(bundle.meta.get("uniprot", "")), str(bundle.meta.get("sequence", "")))
                except Exception:
                    name = os.path.basename(d.rstrip("/"))
                    parts = name.split("_", 2)
                    if len(parts) >= 3:
                        return (parts[0], parts[2])
                    return ("", "")

        def _filter_dirs(dirs_in: List[str], split_name: str) -> List[str]:
            keys = [_dir_key(d) for d in dirs_in]
            keys_tol = [_normalize_key(k) for k in keys]
            keep = [((k in label_keys_raw) or (kt in label_keys_tol)) for k, kt in zip(keys, keys_tol)]
            kept = [d for d, ok in zip(dirs_in, keep) if ok]
            dropped = len(dirs_in) - len(kept)
            if dropped:
                print(f"[dataset] {split_name}: kept={len(kept)} dropped={dropped} (from {len(dirs_in)})")
            else:
                print(f"[dataset] {split_name}: kept={len(kept)} dropped=0 (from {len(dirs_in)})")
            return kept

        train_dirs = _filter_dirs(train_dirs, "train")
        val_dirs = _filter_dirs(val_dirs, "val")

        self.train_stats = {}
        if self.normalize == "zscore_per_tf":
            tf_to_vals = defaultdict(list)
            for d in train_dirs:
                try:
                    uniprot, _, seq = parse_prediction_dir_name(d)
                except Exception:
                    try:
                        bundle = load_crop_bundle(d, device=torch.device("cpu"))
                        uniprot = bundle.meta.get("uniprot", "")
                        seq = bundle.meta.get("sequence", "")
                    except Exception:
                        continue
                key = (uniprot, seq)
                if key in self.labels:
                    tf_to_vals[uniprot].append(float(self.labels[key]))
            for tf, vals in tf_to_vals.items():
                if len(vals) == 0:
                    continue
                mu = float(np.mean(vals))
                sigma = float(np.std(vals))
                if not np.isfinite(sigma) or sigma < 1e-8:
                    sigma = 1e-8
                self.train_stats[tf] = (mu, sigma)

        self.train_ds = AffinityDataset(train_dirs, self.labels, split="train", normalize=self.normalize, train_stats=self.train_stats,
                                        dist_feats=self.dist_feats, rbf_centers=self.rbf_centers, rbf_min=self.rbf_min, rbf_max=self.rbf_max, rbf_sigma=self.rbf_sigma,
                                        cache_dir=self.cache_dir, cache_format=self.cache_format, cache_in_mem=self.cache_in_mem, cache_z_in_mem=self.cache_z_in_mem)
        self.val_ds = AffinityDataset(val_dirs, self.labels, split="val", normalize="none", train_stats=self.train_stats,
                                      dist_feats=self.dist_feats, rbf_centers=self.rbf_centers, rbf_min=self.rbf_min, rbf_max=self.rbf_max, rbf_sigma=self.rbf_sigma,
                                      cache_dir=self.cache_dir, cache_format=self.cache_format, cache_in_mem=self.cache_in_mem, cache_z_in_mem=self.cache_z_in_mem)

        self.test_ds = None
        if self.test_glob:
            import glob as globlib
            test_dirs_all = sorted(globlib.glob(os.path.join(self.test_glob)))
            if test_dirs_all:
                test_labels = self.labels if not self.test_labels_csv else read_labels(self.test_labels_csv, seq_col=self.seq_col)
                train_tfs_set = set(self.train_stats.keys()) if self.train_stats else set()
                if not train_tfs_set:
                    train_tfs_set = set()
                    for d in train_dirs:
                        try:
                            tf_id, _, _ = parse_prediction_dir_name(d)
                        except Exception:
                            tf_id = ""
                        train_tfs_set.add(tf_id)
                test_dirs = []
                for d in test_dirs_all:
                    try:
                        tf_id, _, _ = parse_prediction_dir_name(d)
                    except Exception:
                        tf_id = ""
                    if tf_id and tf_id not in train_tfs_set:
                        test_dirs.append(d)
                if test_dirs:
                    self.test_ds = AffinityDataset(test_dirs, test_labels, split="test", normalize="none", train_stats=self.train_stats,
                                                   dist_feats=self.dist_feats, rbf_centers=self.rbf_centers, rbf_min=self.rbf_min, rbf_max=self.rbf_max, rbf_sigma=self.rbf_sigma,
                                                   cache_dir=self.cache_dir, cache_format=self.cache_format, cache_in_mem=self.cache_in_mem, cache_z_in_mem=self.cache_z_in_mem)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.train_ds is not None
        kwargs = dict(batch_size=self.batch_size,
                      shuffle=True,
                      num_workers=self.num_workers,
                      collate_fn=single_collate,
                      pin_memory=self.pin_memory,
                      persistent_workers=self.num_workers > 0)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.train_ds, **kwargs)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.val_ds is not None
        kwargs = dict(batch_size=self.batch_size,
                      shuffle=False,
                      num_workers=self.num_workers,
                      collate_fn=single_collate,
                      pin_memory=self.pin_memory,
                      persistent_workers=self.num_workers > 0)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.val_ds, **kwargs)

    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        if self.test_ds is None:
            return None
        kwargs = dict(batch_size=self.batch_size,
                      shuffle=False,
                      num_workers=self.num_workers,
                      collate_fn=single_collate,
                      pin_memory=self.pin_memory,
                      persistent_workers=self.num_workers > 0)
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return torch.utils.data.DataLoader(self.test_ds, **kwargs)


__all__ = ["AffinityDataModule", "read_labels"]


