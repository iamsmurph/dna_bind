### Offline TF–DNA affinity model (using Boltz 2 embeddings)

This module trains a lightweight TF–DNA affinity regressor directly from precomputed Boltz 2 prediction outputs. It assumes you have already run Boltz 2 and saved cropped embeddings and CIFs for your TF–DNA complexes.

### What you need

- **Boltz 2 prediction dirs** containing at least:
  - `cropped_embeddings_*.pt` with keys: `z` or `z_crop`, and `indices` (crop→full mapping). Optional: `s_crop` or `s_full`.
  - `*_model_0.cif` (required; used to build geometry/masks).
  - Optional priors: `contact_probs_*.npz`, `pae_*.npz`, `pde_*.npz`, `tm_expected_value_*.npz`.
- **Labels CSV** with columns: `uniprot`, `nt` (or `sequence`), `intensity_log1p`.
- **GPU** recommended. Mixed precision is enabled by default; set `--devices` to the number of GPUs. No internet is required once embeddings are saved.

Example Boltz 2 invocation used to produce embeddings (for reference):

```bash
boltz predict --devices 8 \
  --cache /storage/ujp/.cache/huggingface \
  --diffusion_samples 1 \
  --write_full_pde --write_full_pae --write_contact_probs \
  --write_tm_expected_value --write_cropped_embeddings \
  ../../boltz_runs/uniprobe_subset_100tfs/chunk_0
```

Resulting prediction dirs live under a root like:

```
/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*
```

### Quickstart (most streamlined path)

1) Optional sanity check (counts only):

```bash
python -m dna_bind_offline.train.cli \
  --preflight \
  --labels_csv /data/rbg/users/seanmurphy/dna_bind/datasets/uniprobe_subset_100tfs.csv \
  --pred_glob '/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*'
```

2) Optional: build/refresh prefilter index (fast subsequent starts):

```bash
python -m dna_bind_offline.train.cli \
  --prefilter-cache-refresh \
  --cache-dir /data/rbg/users/seanmurphy/dna_bind/runs_cache \
  --labels_csv /data/rbg/users/seanmurphy/dna_bind/datasets/uniprobe_subset_100tfs.csv \
  --pred_glob '/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*'
```

3) Train (single GPU, mixed precision):

```bash
python -m dna_bind_offline.train.cli \
  --labels_csv /data/rbg/users/seanmurphy/dna_bind/datasets/uniprobe_subset_100tfs.csv \
  --seq_col nt \
  --pred_glob '/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*' \
  --normalize zscore_per_tf \
  --epochs 50 \
  --lr 1e-3 \
  --val_frac 0.1 \
  --devices 1 \
  --precision 16 \
  --num_workers 8 \
  --pin_memory 1 \
  --cache-dir /data/rbg/users/seanmurphy/dna_bind/runs_cache \
  --out /data/rbg/users/seanmurphy/dna_bind/checkpoints/tfdna_affinity.ckpt
```

Multi‑GPU (DDP) is enabled automatically when `--devices > 1`:

```bash
python -m dna_bind_offline.train.cli \
  --labels_csv /data/rbg/users/seanmurphy/dna_bind/datasets/uniprobe_subset_100tfs.csv \
  --pred_glob '/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*' \
  --devices 8 --precision 16 --num_workers 8 --pin_memory 1 \
  --cache-dir /data/rbg/users/seanmurphy/dna_bind/runs_cache \
  --out /data/rbg/users/seanmurphy/dna_bind/checkpoints/tfdna_affinity.ckpt
```

Optional: add a disjoint‑TF test set after training by passing both `--test_glob` and `--test_labels_csv` (same schema as train labels). The CLI will run `.test()` if test data is provided.

### Paths and flags that matter

- `--labels_csv`: training labels file; `--seq_col` chooses the sequence column if not `nt`.
- `--pred_glob`: glob that matches prediction directories. Each dir name must be of the form `UNIPROT_SCORE_SEQUENCE` (parsed automatically).
- `--cache-dir`: directory used for a persistent prefilter index and feature caches; speeds up repeated runs.
- `--normalize`: `zscore_per_tf` generally helps when evaluating correlations across TFs.
- `--devices` and `--precision`: keep GPU on (omit `--cpu`) and use `16` or `bf16` for speed.
- `--num_workers`, `--pin_memory`, `--prefetch_factor`: set for your I/O; defaults are sensible.

### How the loader uses your saved files

- The trainer loads each prediction dir via `cropped_embeddings_*.pt` and reconstructs a single‑token stream (`s_proxy`) from either `s_crop` or `s_full[indices]`.
- CIF is required to build geometry and a protein→DNA pair mask; directories without a valid CIF are filtered out pre‑training.
- If contact/PAE/PDE are present, they are read and can be fused as weak priors by the model; if absent, the model trains without them.

### Troubleshooting

- "No prediction dirs matched": verify your `--pred_glob` and that each directory contains `cropped_embeddings_*.pt` and a `*_model_0.cif`.
- "Missing z or indices": ensure your Boltz export wrote `z`/`z_crop` and `indices` in the cropped embeddings file.
- Slow startup: run step (2) once with `--cache-dir` to build the prefilter index.

### WandB (optional)

Enable and configure logging:

```bash
python -m dna_bind_offline.train.cli \
  --labels_csv ... --pred_glob ... \
  --wandb --wandb_project tfdna_affinity --wandb_entity cellgp \
  --wandb_dir /data/rbg/users/seanmurphy/dna_bind/wandb --wandb_log_model
```


