"""Sample prediction dirs to validate CIF parsing and PD mask presence.

Usage example:
  python processing/check_prefilter_cifs.py \
    --pred_glob '/data/rbg/users/ujp/dnabind/boltz_runs/uniprobe_subset_100tfs/boltz_results_chunk_*/predictions/*' \
    --sample 20
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple


def add_project_root_to_syspath() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(here, os.pardir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def sample_indices(n: int, k: int) -> List[int]:
    if n <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [0]
    return sorted({int(i * (n - 1) / (k - 1)) for i in range(k)})


def find_cif_path(pred_dir: str) -> str | None:
    for pat in ("*_model_0.cif", "*model_0*.cif", "*.cif"):
        hits = sorted(glob.glob(os.path.join(pred_dir, pat)))
        if hits:
            return hits[0]
    return None


def main(argv: List[str] | None = None) -> int:
    add_project_root_to_syspath()

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_glob", required=True, help="glob of prediction directories to sample")
    ap.add_argument("--sample", type=int, default=20, help="number of dirs to sample (evenly spaced)")
    args = ap.parse_args(argv)

    pred_dirs = sorted(glob.glob(args.pred_glob))
    print(f"total_pred_dirs={len(pred_dirs)}")
    if not pred_dirs:
        return 0

    idxs = sample_indices(len(pred_dirs), max(1, int(args.sample)))

    torch_ok = False
    gemmi_ok = False
    biopy_ok = False

    try:
        import torch  # type: ignore
        torch_ok = True
    except Exception:
        torch_ok = False
    try:
        import gemmi  # type: ignore
        gemmi_ok = True
    except Exception:
        gemmi_ok = False
    try:
        import Bio  # type: ignore
        biopy_ok = True
    except Exception:
        biopy_ok = False

    print(f"env: torch={'yes' if torch_ok else 'no'} gemmi={'yes' if gemmi_ok else 'no'} biopython={'yes' if biopy_ok else 'no'}")

    # Imports that depend on project modules
    try:
        from dna_bind_offline.io.cif_parser import parse_cif_to_token_geom  # type: ignore
        from dna_bind_offline.geometry.masks import make_crop_masks  # type: ignore
        from dna_bind_offline.io.bundle_loader import load_crop_bundle  # type: ignore
        project_ok = True
    except Exception as e:
        print(f"project_import_error={e!r}")
        project_ok = False

    num_with_pt = 0
    num_with_cif = 0
    num_with_dna = 0
    num_mask_any = 0

    for i in idxs:
        d = pred_dirs[i]
        pt_hits = sorted(glob.glob(os.path.join(d, "cropped_embeddings_*.pt")))
        cif_path = find_cif_path(d)
        print(f"\nDIR: {d}")
        print(f"  PT: {pt_hits[0] if pt_hits else None}")
        print(f"  CIF: {cif_path}")
        if pt_hits:
            num_with_pt += 1
        if cif_path and os.path.exists(cif_path):
            num_with_cif += 1
        if project_ok and cif_path and os.path.exists(cif_path):
            try:
                geom = parse_cif_to_token_geom(cif_path)
                mt = geom.mol_type
                prot = int((mt == 0).sum())
                dna = int((mt == 1).sum())
                other = int((mt == 2).sum())
                print(f"  tokens: total={len(mt)} prot={prot} dna={dna} other={other}")
                if dna > 0:
                    num_with_dna += 1
            except Exception as e:
                print(f"  parse_error: {e!r}")
        if project_ok and torch_ok and pt_hits and cif_path and os.path.exists(cif_path):
            try:
                # Force CPU for safety
                import torch  # type: ignore
                bundle = load_crop_bundle(d, device=torch.device("cpu"))
                geom = parse_cif_to_token_geom(cif_path)
                masks = make_crop_masks(bundle.crop_to_full, geom, bundle.contact_probs, bundle.pae, bundle.pde)
                has_any = bool(masks.affinity_pair_mask.any())
                num_mask_any += 1 if has_any else 0
                print(f"  affinity_pair_mask.any={has_any} pairs={int(masks.pd_pairs.shape[0])}")
            except Exception as e:
                print(f"  mask_error: {e!r}")

    print("\nSummary:")
    print(f"  sampled={len(idxs)} with_pt={num_with_pt} with_cif={num_with_cif} with_dna_tokens={num_with_dna} pd_mask_any={num_mask_any}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


