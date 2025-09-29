"""Argparse-based CLI to train the TF-DNA affinity model."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from ..data.datamodule import AffinityDataModule
from ..utils.hashing_cache import read_json
from .lightning_module import AffinityLightningModule


def infer_dims_from_datamodule(dm: AffinityDataModule) -> Tuple[int, int, int]:
    if dm.train_ds is None:
        dm.setup()
    assert dm.train_ds is not None
    probe = dm.train_ds[0]
    c_pair = int(probe.z.shape[-1])
    c_single = int(probe.s_proxy.shape[-1])
    n_bins = int(probe.dist_bins.shape[-1])
    return c_pair, c_single, n_bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true", help="run a label-vs-pred mapping summary before training")
    ap.add_argument("--preflight-only", action="store_true", help="only run preflight summary and exit")
    ap.add_argument("--labels_csv", default="/data/rbg/users/seanmurphy/dna_bind/datasets/uniprobe_subset_100tfs.csv", help="path to labels CSV with columns: uniprot, nt (or sequence), intensity_log1p")
    ap.add_argument("--seq_col", default="nt", help="CSV column name for DNA sequence (default: nt)")
    ap.add_argument("--pred_glob", default="/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*", help="glob matching Boltz prediction dirs (e.g., 'runs/*/*')")
    ap.add_argument("--split_by", default="tf", choices=["random", "tf"], help="how to split train/val (default: tf)")
    ap.add_argument("--test_glob", default=None, help="optional glob for an unseen-TF test set")
    ap.add_argument("--test_labels_csv", default=None, help="optional labels CSV for the test set")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--out", default="tfdna_affinity.ckpt")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=min(os.cpu_count() or 16, 16))
    ap.add_argument("--prefetch_factor", type=int, default=4)
    ap.add_argument("--pin_memory", type=int, choices=[0,1], default=1)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", type=str, default="auto", choices=["auto", "32", "16", "bf16"])  # autocast precision
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--log_every_n_steps", type=int, default=1)
    ap.add_argument("--check_val_every_n_epoch", type=int, default=1)
    ap.add_argument("--normalize", type=str, default="none", choices=["none", "zscore_per_tf"], help="label normalization strategy")
    # attention head
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--attn-dropout", dest="attn_dropout", type=float, default=0.10)
    ap.add_argument("--noise-std", dest="noise_std", type=float, default=0.02)
    # distance features (RBF-only)
    ap.add_argument("--dist-rbf-centers", dest="dist_rbf_centers", type=int, default=64)
    ap.add_argument("--dist-rbf-min", dest="dist_rbf_min", type=float, default=2.0)
    ap.add_argument("--dist-rbf-max", dest="dist_rbf_max", type=float, default=22.0)
    ap.add_argument("--dist-rbf-sigma", dest="dist_rbf_sigma", type=float, default=1.0)
    # cache
    ap.add_argument("--cache-dir", type=str, default="runs_cache")
    ap.add_argument("--cache-format", type=str, choices=["npz", "npy"], default="npz")
    ap.add_argument("--cache-in-mem", type=int, default=128)
    ap.add_argument("--cache-z-in-mem", action="store_true", help="keep z tensors in memory cache for speed")
    # prefilter index cache control
    ap.add_argument("--prefilter-cache-refresh", action="store_true", help="force rebuild the prefilter index")
    ap.add_argument("--prefilter-cache-off", action="store_true", help="disable prefilter index and run full scan")
    ap.add_argument("--prefilter-workers", type=int, default=min(os.cpu_count() or 8, 8), help="parallel workers for prefilter/index build (I/O-bound)")
    # default progress on; no toggle
    ap.add_argument("--prefilter-verbose", type=int, choices=[0,1], default=0, help="print first parse/mask error per invalid dir during prefilter")
    # priors and correlation penalty
    ap.add_argument("--corr-w", type=float, default=0.0, help="weight for correlation penalty (requires minibatches)")
    ap.add_argument("--prior-w-contact", type=float, default=1.0)
    ap.add_argument("--prior-w-pae", type=float, default=0.25)
    ap.add_argument("--prior-w-pde", type=float, default=0.10)
    ap.add_argument("--prior-eps", type=float, default=1e-6)
    # wandb logging
    ap.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    ap.add_argument("--wandb_project", type=str, default="tfdna_affinity", help="wandb project name")
    ap.add_argument("--wandb_entity", type=str, default="cellgp", help="wandb entity (team/org)")
    ap.add_argument("--wandb_run_name", type=str, default=None, help="optional wandb run name")
    ap.add_argument("--wandb_group", type=str, default=None, help="optional wandb group")
    ap.add_argument("--wandb_tags", type=str, default="", help="comma-separated wandb tags")
    ap.add_argument("--wandb_mode", type=str, choices=["online", "offline", "disabled"], default="online", help="wandb mode")
    ap.add_argument("--wandb_dir", type=str, default="/data/rbg/users/seanmurphy/dna_bind/wandb", help="local directory for wandb files")
    ap.add_argument("--wandb_log_model", action="store_true", help="log model checkpoints to wandb")
    args = ap.parse_args()
    if args.preflight or args.preflight_only:
        import csv as _csv, glob as _glob
        labels_set = set()
        with open(args.labels_csv, newline="") as _f:
            _r = _csv.DictReader(_f)
            _seq_col = args.seq_col if args.seq_col in _r.fieldnames else ("sequence" if "sequence" in _r.fieldnames else None)
            if _seq_col is None or "uniprot" not in _r.fieldnames or "intensity_log1p" not in _r.fieldnames:
                print("[preflight] CSV missing required columns; found:", _r.fieldnames)
            else:
                for _row in _r:
                    labels_set.add((_row["uniprot"].strip(), _row[_seq_col].strip()))
        pred_dirs = [d for d in _glob.glob(args.pred_glob) if os.path.isdir(d)]
        preds_set = set()
        for d in pred_dirs:
            _name = os.path.basename(d.rstrip("/"))
            _parts = _name.split("_", 2)
            if len(_parts) >= 3:
                u, _, s = _parts[0], _parts[1], _parts[2]
                preds_set.add((u, s))
        # trailing-T tolerant intersect
        def normalize(k):
            u, s = k
            return (u, s[:-1]) if s.endswith("T") else (u, s)
        labels_norm = {normalize(k) for k in labels_set}
        preds_norm = {normalize(k) for k in preds_set}
        common = labels_norm & preds_norm
        print(f"[preflight] labels={len(labels_set)} preds={len(preds_set)} common(tolerant)={len(common)} only_labels={len(labels_norm - preds_norm)} only_preds={len(preds_norm - labels_norm)}")
        if args.preflight_only:
            return

    dm = AffinityDataModule(labels_csv=args.labels_csv,
                            pred_glob=args.pred_glob,
                            seq_col=args.seq_col,
                            val_frac=args.val_frac,
                            seed=args.seed,
                            num_workers=args.num_workers,
                            batch_size=1,
                            split_by=args.split_by,
                            normalize=args.normalize,
                            test_glob=args.test_glob,
                            test_labels_csv=args.test_labels_csv,
                            dist_feats="rbf",
                            rbf_centers=args.dist_rbf_centers,
                            rbf_min=args.dist_rbf_min,
                            rbf_max=args.dist_rbf_max,
                            rbf_sigma=args.dist_rbf_sigma,
                            cache_dir=args.cache_dir,
                            cache_format=args.cache_format,
                            cache_in_mem=args.cache_in_mem,
                            cache_z_in_mem=1 if args.cache_z_in_mem else 0,
                            prefilter_cache_refresh=bool(args.prefilter_cache_refresh),
                            prefilter_cache_off=bool(args.prefilter_cache_off),
                            prefilter_workers=int(args.prefilter_workers),
                            prefilter_progress=True,
                            prefilter_verbose=bool(args.prefilter_verbose),
                            pin_memory=bool(args.pin_memory),
                            prefetch_factor=args.prefetch_factor)
    dm.setup()

    # If we're only refreshing the prefilter cache, summarize and exit early
    if args.prefilter_cache_refresh:
        index_path = os.path.join(args.cache_dir or ".", "prefilter_index.json") if args.cache_dir else ""
        idx = read_json(index_path) if index_path else None
        if isinstance(idx, dict) and "dirs" in idx and "meta" in idx:
            dirs_map = idx.get("dirs", {})
            total = int(idx.get("meta", {}).get("dir_count", len(dirs_map)))
            valid = sum(1 for v in dirs_map.values() if isinstance(v, dict) and bool(v.get("valid", False)))
            changed_note = ""
            print(f"[prefilter] index refreshed at {index_path}: valid={valid} / total={total}")
        else:
            print("[prefilter] index refresh requested, but index not found or invalid")
        return

    c_pair, c_single, n_bins = infer_dims_from_datamodule(dm)
    # Validate heads divisibility; fail fast with a clear suggestion
    if (c_single % args.heads) != 0:
        divisors = [d for d in range(1, min(c_single, args.heads) + 1) if c_single % d == 0]
        fallback = max(divisors) if divisors else 1
        raise SystemExit(
            f"[config] c_single={c_single} not divisible by heads={args.heads}. "
            f"Try --heads {fallback} or any divisor of {c_single}."
        )
    lit = AffinityLightningModule(c_pair=c_pair,
                                  c_single=c_single,
                                  n_bins=n_bins,
                                  lr=args.lr,
                                  weight_decay=args.wd,
                                  corr_w=args.corr_w,
                                  attn_dropout=args.attn_dropout,
                                  noise_std=args.noise_std,
                                  prior_w_contact=args.prior_w_contact,
                                  prior_w_pae=args.prior_w_pae,
                                  prior_w_pde=args.prior_w_pde,
                                  prior_eps=args.prior_eps,
                                  heads=args.heads)
    lit.normalize = args.normalize
    lit.train_stats = dm.train_stats

    pl.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    accelerator = "cpu" if args.cpu or not torch.cuda.is_available() else "gpu"
    # Select precision automatically when requested
    major = 0
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
    except Exception:
        pass
    if args.precision == "auto":
        prec_key = "bf16" if (torch.cuda.is_available() and major >= 8) else "16"
    else:
        prec_key = args.precision
    precision_map = {"32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"}
    strategy = "auto" if not (accelerator == "gpu" and args.devices > 1) else "ddp_find_unused_parameters_false"
    mc = ModelCheckpoint(monitor="val/r_by_tf_mean", mode="max", save_top_k=1, filename="best")
    callbacks = [
        mc,
        EarlyStopping(monitor="val/r_by_tf_mean", mode="max", patience=max(5, args.epochs // 10), min_delta=0.002),
    ]
    logger_obj = None
    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", args.wandb_dir)
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        os.environ.setdefault("WANDB_CACHE_DIR", os.path.join(args.wandb_dir, "cache"))
        os.environ.setdefault("WANDB_CONFIG_DIR", os.path.join(args.wandb_dir, "config"))
        run_name = args.wandb_run_name or f"tfdna_affinity_{args.seed}"
        tags = [t for t in (args.wandb_tags.split(",") if args.wandb_tags else []) if t]
        logger_obj = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            save_dir=args.wandb_dir,
            mode=args.wandb_mode,
            log_model=args.wandb_log_model,
        )
        try:
            logger_obj.experiment.config.update(vars(args), allow_val_change=True)
        except Exception:
            pass

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=args.devices if accelerator == "gpu" else 1,
        strategy=strategy,
        callbacks=callbacks,
        precision=precision_map[prec_key],
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.grad_clip,
        accumulate_grad_batches=16,
        deterministic=False,
        logger=logger_obj,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    trainer.fit(lit, datamodule=dm)

    ckpt_path = mc.best_model_path
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        torch.save(state, args.out)
    else:
        torch.save(lit.state_dict(), args.out)

    if dm.test_ds is not None:
        try:
            trainer.test(lit, datamodule=dm)
        except Exception:
            pass


if __name__ == "__main__":
    main()


