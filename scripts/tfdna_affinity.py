#!/usr/bin/env python3
from __future__ import annotations

import os
import glob
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import gemmi  # type: ignore
    _HAS_GEMMI = True
except Exception:
    _HAS_GEMMI = False


# =============================
# Data Models
# =============================


@dataclass
class CropBundle:
    z: torch.FloatTensor
    s_proxy: torch.FloatTensor
    crop_to_full: np.ndarray
    contact_probs: Optional[np.ndarray]
    pae: Optional[np.ndarray]
    pde: Optional[np.ndarray]
    tm_expected: Optional[float]
    meta: dict


@dataclass
class TokenGeom:
    rep_xyz: np.ndarray           # [Lf, 3]
    mol_type: np.ndarray          # [Lf] int8: 0=PROTEIN,1=DNA,2=OTHER
    token_pad_mask: np.ndarray    # [Lf] bool
    token_meta: List[dict]


@dataclass
class CropMasks:
    rep_xyz_crop: np.ndarray        # [Lc, 3]
    token_pad_mask_crop: np.ndarray # [Lc]
    mol_type_crop: np.ndarray       # [Lc]
    affinity_pair_mask: np.ndarray  # [Lc, Lc]
    pd_token_mask: np.ndarray       # [Lc]
    pd_pairs: np.ndarray            # [K, 2] int64 indices (i,j) where protein->DNA
    pd_flat_idx: np.ndarray         # [K] int64 flattened indices for [Lc*Lc]


# =============================
# Utilities: discovery & parsing
# =============================


def parse_prediction_dir_name(dir_path: str) -> Tuple[str, float, str]:
    name = os.path.basename(dir_path.rstrip("/"))
    parts = name.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Unexpected prediction directory name: {name}")
    uniprot = parts[0]
    try:
        score = float(parts[1])
    except Exception:
        score = float("nan")
    sequence = parts[2]
    return uniprot, score, sequence


def locate_file(pred_dir: str, stem: str, ext: str) -> Optional[str]:
    paths = glob.glob(os.path.join(pred_dir, f"{stem}_*.{ext}"))
    return paths[0] if paths else None


def locate_cif(pred_dir: str) -> Optional[str]:
    patterns = ("*_model_0.cif", "*model_0*.cif", "*.cif")
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(pred_dir, pat)))
        if hits:
            return hits[0]
    return None


def locate_any(pred_dir: str, stems: List[str], ext: str = "npz") -> Optional[str]:
    patterns: List[str] = []
    for s in stems:
        patterns.extend([f"{s}_*.{ext}", f"*{s}*.{ext}"])
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(pred_dir, pat)))
        if hits:
            return hits[0]
    return None


def load_npz_safely(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    try:
        data = np.load(path)
        # Try common keys, else first array
        for key in ("contact_probs", "arr_0"):
            if key in data:
                arr = data[key]
                break
        else:
            arr = data[data.files[0]]
        arr = np.asarray(arr)
        # Symmetrize if near-symmetric expected (warn-silent if notably asymmetric)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            if not np.allclose(arr, arr.T, atol=1e-3):
                pass  # optionally warn
            arr = 0.5 * (arr + arr.T)
        return arr
    except Exception:
        return None


def _torch_load_cpu(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        return obj[0] if (0 in obj and isinstance(obj[0], dict)) else obj
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], dict):
        return obj[0]
    raise ValueError(f"Unexpected .pt structure in {path}: {type(obj)}")


# =============================
# (a) Loader for saved artifacts
# =============================


def load_crop_bundle(pred_dir: str, device: Optional[torch.device] = None) -> CropBundle:
    """Load crop-aligned pair/ single reps and optional signals from a Boltz prediction directory.

    Expects files like:
      - cropped_embeddings_*.pt (with keys z or z_crop; s_full or s_crop; indices)
      - contact_probs_*.npz (optional)
      - pae_*.npz, pde_*.npz (optional)
      - tm_expected_value_*.npz (optional)
      - *_model_0.cif (for geometry; parsed separately)
    """
    pt_paths = sorted(glob.glob(os.path.join(pred_dir, "cropped_embeddings_*.pt")))
    if not pt_paths:
        raise FileNotFoundError(f"No cropped_embeddings_*.pt in {pred_dir}")
    pt_path = pt_paths[0]
    inner = _torch_load_cpu(pt_path)

    # Pair reps
    z_key = "z_crop" if "z_crop" in inner else ("z" if "z" in inner else None)
    if z_key is None:
        raise KeyError("Missing z or z_crop in cropped embeddings")
    z: torch.Tensor = inner[z_key]
    if not isinstance(z, torch.Tensor):
        z = torch.as_tensor(z)
    if z.ndim != 3 or z.shape[0] != z.shape[1]:
        raise ValueError(f"z must be square [Lc,Lc,C], got {tuple(z.shape)}")
    Lc, _, c_pair = z.shape

    # Indices mapping crop->full
    crop_to_full: Optional[np.ndarray] = None
    if "indices" in inner:
        idx = inner["indices"]
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        crop_to_full = np.asarray(idx, dtype=np.int64)
    elif "tok_idx_full" in inner:
        idx = inner["tok_idx_full"]
        if isinstance(idx, torch.Tensor):
            idx = idx.detach().cpu().numpy()
        crop_to_full = np.asarray(idx, dtype=np.int64)
    elif "prot_idx_full" in inner and "dna_idx_full" in inner:
        p = inner["prot_idx_full"]
        d = inner["dna_idx_full"]
        if isinstance(p, torch.Tensor):
            p = p.detach().cpu().numpy()
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy()
        crop_to_full = np.asarray(list(p) + list(d), dtype=np.int64)
    if crop_to_full is None:
        raise KeyError("Missing indices mapping crop->full tokens")

    # Single-stream proxy
    c_single = 384
    s_proxy: torch.Tensor
    if "s_crop" in inner:
        s_proxy = inner["s_crop"]
        if isinstance(s_proxy, torch.Tensor):
            s_proxy = s_proxy
        else:
            s_proxy = torch.as_tensor(s_proxy)
        if s_proxy.ndim == 3 and s_proxy.shape[0] == 1:
            s_proxy = s_proxy[0]
        if s_proxy.shape[0] != Lc:
            raise ValueError("s_crop not aligned to crop")
        c_single = s_proxy.shape[-1]
    elif "s_full" in inner:
        s_full = inner["s_full"]
        if isinstance(s_full, torch.Tensor):
            s_full = s_full
        else:
            s_full = torch.as_tensor(s_full)
        if s_full.ndim == 3 and s_full.shape[0] == 1:
            s_full = s_full[0]
        max_needed = int(crop_to_full.max()) + 1
        if s_full.shape[0] < max_needed:
            raise AssertionError("s_full length < max crop_to_full index + 1")
        s_proxy = s_full[crop_to_full]
        c_single = s_proxy.shape[-1]
    else:
        # unlikely; create zeros with guessed dim
        s_proxy = torch.zeros((Lc, c_single), dtype=z.dtype)

    # Optional signals (all expected LxL; downselect if needed happens in masks step)
    contact = load_npz_safely(locate_any(pred_dir, ["contact_probs"]))
    pae = load_npz_safely(locate_any(pred_dir, ["pae", "full_pae", "pae_full"]))
    pde = load_npz_safely(locate_any(pred_dir, ["pde", "full_pde", "pde_full"]))
    tm_arr = load_npz_safely(locate_any(pred_dir, ["tm_expected_value"]))
    tm_expected: Optional[float] = None
    if tm_arr is not None:
        try:
            tm_expected = float(np.asarray(tm_arr).reshape(-1)[0])
        except Exception:
            tm_expected = None

    uniprot, score_from_dirname, sequence = parse_prediction_dir_name(pred_dir)
    meta = {
        "pt_path": pt_path,
        "contact_path": locate_any(pred_dir, ["contact_probs"]),
        "pae_path": locate_any(pred_dir, ["pae", "full_pae", "pae_full"]),
        "pde_path": locate_any(pred_dir, ["pde", "full_pde", "pde_full"]),
        "tm_path": locate_any(pred_dir, ["tm_expected_value"]),
        "cif_path": locate_cif(pred_dir),
        "uniprot": uniprot,
        "sequence": sequence,
        "score_from_dirname": score_from_dirname,
    }

    # Device preference
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    if device is not None:
        z = z.to(device=device, dtype=torch.float32)
        s_proxy = s_proxy.to(device=device, dtype=z.dtype)

    return CropBundle(
        z=z,
        s_proxy=s_proxy,
        crop_to_full=crop_to_full,
        contact_probs=contact,
        pae=pae,
        pde=pde,
        tm_expected=tm_expected,
        meta=meta,
    )


# =====================================
# (b) Parse CIF -> per-token coordinates
# =====================================


def parse_cif_to_token_geom(cif_path: str, expected_Lf: Optional[int] = None) -> TokenGeom:
    if not _HAS_GEMMI:
        # Fallback to Biopython
        try:
            from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # type: ignore
        except Exception as e:
            raise ImportError("Neither gemmi nor Biopython available for CIF parsing") from e
        d = MMCIF2Dict(cif_path)
        chains = d.get("_atom_site.label_asym_id", [])
        poly_types = d.get("_entity_poly.type", [])
        # Build map from entity id to poly type if available, else infer from comp ids
        xs = np.asarray(d.get("_atom_site.Cartn_x", []), dtype=float)
        ys = np.asarray(d.get("_atom_site.Cartn_y", []), dtype=float)
        zs = np.asarray(d.get("_atom_site.Cartn_z", []), dtype=float)
        comp_ids = d.get("_atom_site.label_comp_id", [])
        atom_ids = d.get("_atom_site.label_atom_id", [])
        res_ids = d.get("_atom_site.label_seq_id", [])
        # Group by (chain, res_id)
        key = [f"{c}:{r}" for c, r in zip(chains, res_ids)]
        order, index = np.unique(key, return_inverse=True)
        Lf = len(order)
        rep = np.zeros((Lf, 3), dtype=np.float32)
        mol_type = np.zeros((Lf,), dtype=np.int8)
        pad = np.ones((Lf,), dtype=bool)
        meta: List[dict] = [{} for _ in range(Lf)]
        # Simple heuristic: protein if comp_id in 20 AA set; DNA if in nucleotides
        aa = set(["ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"])
        dna = set(["DA","DT","DG","DC","A","T","G","C","U","DU"])
        for i in range(Lf):
            mask = (index == i)
            comp_i = [comp_ids[j] for j in np.nonzero(mask)[0]]
            atom_i = [atom_ids[j] for j in np.nonzero(mask)[0]]
            xi = xs[mask]
            yi = ys[mask]
            zi = zs[mask]
            if any(c in aa for c in comp_i):
                mol_type[i] = 0
                # prefer CA then CB; else centroid of heavy
                sel = [j for j, a in enumerate(atom_i) if a.strip() == "CA"]
                if not sel:
                    sel = [j for j, a in enumerate(atom_i) if a.strip() == "CB"]
                if not sel:
                    sel = list(range(len(atom_i)))
                j0 = sel[0]
                rep[i] = np.array([xi[j0], yi[j0], zi[j0]], dtype=np.float32)
            elif any(c in dna for c in comp_i):
                mol_type[i] = 1
                sel = [j for j, a in enumerate(atom_i) if a.strip() in ("C4'","P")]
                if not sel:
                    sel = list(range(len(atom_i)))
                j0 = sel[0]
                rep[i] = np.array([xi[j0], yi[j0], zi[j0]], dtype=np.float32)
            else:
                mol_type[i] = 2
                if len(xi) > 0:
                    rep[i] = np.array([xi.mean(), yi.mean(), zi.mean()], dtype=np.float32)
                else:
                    rep[i] = 0.0
        # Sanity checks
        if not np.isfinite(rep).all():
            raise ValueError("NaNs in representative coordinates")
        if expected_Lf is not None and Lf != expected_Lf:
            # allow mismatch; caller may not always know Lf
            pass
        # crude max distance sanity
        if Lf >= 2:
            # sample
            a = rep[0]
            b = rep[-1]
            if float(np.linalg.norm(a - b)) > 200.0:
                raise ValueError("Unreasonable inter-token distance > 200 Å")
        return TokenGeom(rep_xyz=rep, mol_type=mol_type, token_pad_mask=pad, token_meta=meta)

    # gemmi path (prefer polymer typing; fallback above handles non-gemmi case)
    st = gemmi.make_structure_from_block(gemmi.cif.read(cif_path).sole_block())
    st.setup_entities()
    model = st[0]
    PROT, DNA, OTHER = 0, 1, 2
    rep: List[tuple] = []
    typ: List[int] = []
    pad: List[bool] = []
    meta: List[dict] = []
    for chain in model:
        pt = chain.get_polymer_type()
        if pt in {gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideLike}:
            mtype = PROT
        elif pt == gemmi.PolymerType.DNA:
            mtype = DNA
        else:
            mtype = OTHER
        for res in chain.get_polymer():
            prefer = ["C4'", "C4*"] if mtype == DNA else ["CA"]
            fallback = ["P", "C3'", "C3*"] if mtype == DNA else ["CB", "CA"]
            xyz = None
            for nm in prefer + fallback:
                a = next((a for a in res if a.name == nm), None)
                if a is not None:
                    xyz = (a.pos.x, a.pos.y, a.pos.z)
                    break
            if xyz is None:
                coords = [(a.pos.x, a.pos.y, a.pos.z) for a in res if a.element.name != "H"]
                if coords:
                    xyz = tuple(np.mean(np.asarray(coords, np.float64), axis=0))
            if xyz is not None:
                rep.append(xyz)
                typ.append(mtype)
                pad.append(True)
                meta.append({"chain": chain.name, "resname": res.name, "resseq": res.seqid.num})

    rep_arr = np.asarray(rep, dtype=np.float32)
    mol_type_arr = np.asarray(typ, dtype=np.int8)
    pad_arr = np.asarray(pad, dtype=bool)

    if not np.isfinite(rep_arr).all():
        raise ValueError("NaNs in representative coordinates")
    if expected_Lf is not None and len(rep_arr) != expected_Lf:
        pass
    if len(rep_arr) >= 2:
        a = rep_arr[0]
        b = rep_arr[-1]
        if float(np.linalg.norm(a - b)) > 200.0:
            raise ValueError("Unreasonable inter-token distance > 200 Å")

    return TokenGeom(rep_xyz=rep_arr, mol_type=mol_type_arr, token_pad_mask=pad_arr, token_meta=meta)


# =====================================================
# (c) Rebuild crop-space masks and optional edge weights
# =====================================================


def make_crop_masks(crop_to_full: np.ndarray,
                    geom: TokenGeom,
                    contact_probs: Optional[np.ndarray] = None,
                    pae: Optional[np.ndarray] = None,
                    pde: Optional[np.ndarray] = None) -> CropMasks:
    crop_to_full = np.asarray(crop_to_full, dtype=np.int64)
    rep_xyz_crop = geom.rep_xyz[crop_to_full]
    mol_type_crop = geom.mol_type[crop_to_full]
    token_pad_mask_crop = geom.token_pad_mask[crop_to_full]

    is_prot = (mol_type_crop == 0)
    is_dna = (mol_type_crop == 1)
    pd_token_mask = (is_prot | is_dna)
    # One-sided protein→DNA mask to avoid double-counting pairs
    aff_mask = np.outer(is_prot, is_dna)
    aff_mask = aff_mask & np.outer(token_pad_mask_crop, token_pad_mask_crop)

    # Precompute PD indices for sparse ops
    ii, jj = np.where(aff_mask)
    pd_pairs = np.stack([ii.astype(np.int64), jj.astype(np.int64)], axis=1) if ii.size else np.zeros((0, 2), dtype=np.int64)
    Lc = rep_xyz_crop.shape[0]
    pd_flat_idx = (pd_pairs[:, 0] * int(Lc) + pd_pairs[:, 1]).astype(np.int64) if pd_pairs.size else np.zeros((0,), dtype=np.int64)

    # Optional: if provided arrays are full-length, downselect here to crop for later sanity checks
    def _maybe_crop(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if mat is None:
            return None
        if mat.shape[0] == rep_xyz_crop.shape[0]:
            return mat
        return mat[np.ix_(crop_to_full, crop_to_full)]

    if contact_probs is not None and contact_probs.shape[0] != rep_xyz_crop.shape[0]:
        contact_probs = _maybe_crop(contact_probs)
    if pae is not None and pae.shape[0] != rep_xyz_crop.shape[0]:
        pae = _maybe_crop(pae)
    if pde is not None and pde.shape[0] != rep_xyz_crop.shape[0]:
        pde = _maybe_crop(pde)

    # Store as attributes in returned structure if needed later; here we only build masks
    return CropMasks(
        rep_xyz_crop=rep_xyz_crop,
        token_pad_mask_crop=token_pad_mask_crop,
        mol_type_crop=mol_type_crop,
        affinity_pair_mask=aff_mask,
        pd_token_mask=pd_token_mask,
        pd_pairs=pd_pairs,
        pd_flat_idx=pd_flat_idx,
    )


# ==================================
# (d.1) Distogram binning from coords
# ==================================


def build_dist_bins(rep_xyz_crop: np.ndarray,
                    bin_edges: np.ndarray = np.linspace(2.0, 22.0, 64)) -> np.ndarray:
    coords = np.asarray(rep_xyz_crop, dtype=np.float32)
    L = coords.shape[0]
    # pairwise distances
    # (x - x')^2 via broadcasting
    diff = coords[:, None, :] - coords[None, :, :]
    D = np.sqrt(np.maximum(1e-9, np.sum(diff * diff, axis=-1))).astype(np.float32)  # (L,L)
    B = int(len(bin_edges))
    # one-hot bins: assign to nearest edge index (clamped)
    # Use midpoints between edges for binning
    mids = (bin_edges[:-1] + bin_edges[1:]) * 0.5
    # extend for edges
    cut = np.concatenate([[bin_edges[0] - 1e9], mids, [bin_edges[-1] + 1e9]])
    idx = np.digitize(D, cut) - 1
    idx = np.clip(idx, 0, B - 1)
    bins = np.zeros((L, L, B), dtype=np.float32)
    rows, cols = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    bins[rows, cols, idx] = 1.0
    return bins


def build_dist_rbf(rep_xyz: np.ndarray | torch.Tensor,
                   n_centers: int = 64,
                   dmin: float = 2.0,
                   dmax: float = 22.0,
                   sigma: float = 1.0) -> torch.Tensor:
    """RBF features over pairwise token distances.

    Returns tensor of shape [L, L, n_centers].
    """
    if isinstance(rep_xyz, np.ndarray):
        coords = torch.from_numpy(rep_xyz).float()
    else:
        coords = rep_xyz.to(dtype=torch.float32)
    diff = coords[:, None, :] - coords[None, :, :]
    D = torch.linalg.norm(diff, dim=-1)
    centers = torch.linspace(float(dmin), float(dmax), steps=int(n_centers), device=D.device, dtype=D.dtype)
    diff_cent = D[..., None] - centers[None, None, :]
    phi = torch.exp(-0.5 * (diff_cent / max(1e-8, float(sigma))) ** 2)
    phi = phi / (phi.sum(dim=-1, keepdim=True) + 1e-8)
    return phi


# =====================================
# (d.2) Replica affinity head (PyTorch)
# =====================================


class BoltzAffinityHeadReplica(nn.Module):
    def __init__(self, c_pair: int, c_single: int, b_bins: int,
                 hidden: int = 256,
                 use_soft_pool: bool = True,
                 pool_temp: float = 4.0,
                 pooling: str = "attention",
                 attn_hidden: int = 128,
                 attn_dropout: float = 0.10,
                 edge_dropout: float = 0.10,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6) -> None:
        super().__init__()
        self.c_pair = int(c_pair)
        self.c_single = int(c_single)
        self.b_bins = int(b_bins)
        self.hidden = int(hidden)
        self.use_soft_pool = bool(use_soft_pool)
        self.pool_temp = float(pool_temp)
        self.pooling = str(pooling)
        self.noise_std = float(noise_std)
        self.prior_w_contact = float(prior_w_contact)
        self.prior_w_pae = float(prior_w_pae)
        self.prior_w_pde = float(prior_w_pde)
        self.prior_eps = float(prior_eps)

        # Single projections
        self.proj_u = nn.Linear(self.c_single, self.hidden)
        self.proj_v = nn.Linear(self.c_single, self.hidden)
        # Combine u, v, u*v into pair channels
        self.to_bias = nn.Linear(3 * self.hidden, self.c_pair)
        # Distogram projection to pair channels
        self.dist_proj = nn.Linear(self.b_bins, self.c_pair)
        # Fusion + MLP
        self.ln = nn.LayerNorm(2 * self.c_pair)
        self.mlp = nn.Sequential(
            nn.Linear(2 * self.c_pair, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )
        # Attention pooling block
        self.attn_pool = AttnPool(in_dim=2 * self.c_pair,
                                  hidden=int(attn_hidden),
                                  attn_dropout=float(attn_dropout),
                                  edge_dropout=float(edge_dropout))

    def forward(self,
                z: torch.Tensor,
                s_proxy: torch.Tensor,
                dist_bins: torch.Tensor | np.ndarray,
                masks: CropMasks,
                edge_weights: Optional[torch.Tensor] = None,
                prior_contact: Optional[torch.Tensor] = None,
                prior_pae: Optional[torch.Tensor] = None,
                prior_pde: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Shapes and device alignment
        Lc = z.shape[0]
        if z.ndim != 3 or z.shape[0] != z.shape[1]:
            raise ValueError("z must be [Lc,Lc,C_pair]")
        if s_proxy.shape[0] != Lc:
            raise ValueError("s_proxy length must equal Lc")
        # Optional feature noise (training only)
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
            s_proxy = s_proxy + torch.randn_like(s_proxy) * (self.noise_std * 0.5)
        if isinstance(dist_bins, np.ndarray):
            dist_bins_t = torch.from_numpy(dist_bins)
        else:
            dist_bins_t = dist_bins
        if dist_bins_t.shape[:2] != (Lc, Lc):
            raise ValueError("dist_bins must be [Lc,Lc,B]")
        device = z.device
        dist_bins_t = dist_bins_t.to(device=device, dtype=z.dtype)
        # Ensure dist bins last dim matches configured b_bins
        if dist_bins_t.shape[-1] != self.b_bins:
            raise AssertionError("b_bins must equal dist_bins last dim")
        s_proxy = s_proxy.to(device)

        # Single projections (shared for dense/sparse)
        u = self.proj_u(s_proxy)  # [Lc,H]
        v = self.proj_v(s_proxy)  # [Lc,H]

        outputs: Dict[str, torch.Tensor] = {}

        use_sparse = (self.pooling == "attention") and hasattr(masks, "pd_pairs") and getattr(masks, "pd_pairs") is not None and len(getattr(masks, "pd_pairs")) > 0

        if use_sparse:
            # PD-only path: gather z and features on K edges
            z = z.contiguous()
            z_flat = z.view(-1, z.shape[-1])  # [Lc*Lc, C_pair]
            pd_flat_idx_np = masks.pd_flat_idx  # numpy int64 [K]
            pd_pairs_np = masks.pd_pairs       # numpy int64 [K,2]
            pd_flat_idx = torch.from_numpy(pd_flat_idx_np).to(device=device, dtype=torch.long)
            ij = torch.from_numpy(pd_pairs_np).to(device=device, dtype=torch.long)
            i_idx, j_idx = ij[:, 0], ij[:, 1]

            z_pd = z_flat.index_select(0, pd_flat_idx)  # [K, C_pair]

            # PD-only bias from single-stream projections
            u_i = u.index_select(0, i_idx)  # [K,H]
            v_j = v.index_select(0, j_idx)  # [K,H]
            hadamard = u_i * v_j
            gate_pd = torch.cat([u_i, v_j, hadamard], dim=-1)  # [K, 3H]
            bias_pd = self.to_bias(gate_pd)  # [K, C_pair]
            z_hat_pd = z_pd + bias_pd

            # Dist features on PD edges only
            if dist_bins_t.dim() == 3 and dist_bins_t.shape[0] == Lc:
                d_pd = dist_bins_t[i_idx, j_idx, :]  # [K, B]
            else:
                # assume already [K,B]
                d_pd = dist_bins_t
            if d_pd.shape[-1] != self.b_bins:
                raise AssertionError("b_bins must equal dist_bins last dim")
            d_proj_pd = self.dist_proj(d_pd.to(device=device, dtype=z_hat_pd.dtype))  # [K,C_pair]
            h_pd = torch.cat([z_hat_pd, d_proj_pd], dim=-1)  # [K, 2*C_pair]
            h_pd = self.ln(h_pd)

            # PD-only priors
            pri_pd: Optional[torch.Tensor] = None
            if (prior_contact is not None) or (prior_pae is not None) or (prior_pde is not None):
                c_pd = prior_contact[i_idx, j_idx] if isinstance(prior_contact, torch.Tensor) and prior_contact.dim() >= 2 else (prior_contact if isinstance(prior_contact, torch.Tensor) else None)
                pae_pd = prior_pae[i_idx, j_idx] if isinstance(prior_pae, torch.Tensor) and prior_pae.dim() >= 2 else (prior_pae if isinstance(prior_pae, torch.Tensor) else None)
                pde_pd = prior_pde[i_idx, j_idx] if isinstance(prior_pde, torch.Tensor) and prior_pde.dim() >= 2 else (prior_pde if isinstance(prior_pde, torch.Tensor) else None)
                pri_pd = make_prior_logits(
                    contact=c_pd,
                    pae=pae_pd,
                    pde=pde_pd,
                    w_contact=self.prior_w_contact,
                    w_pae=self.prior_w_pae,
                    w_pde=self.prior_w_pde,
                    eps=self.prior_eps,
                )

            pooled, w_pd = self.attn_pool.forward_sparse(
                edge_feats=h_pd,
                priors_logits=pri_pd,
                temp=self.pool_temp,
                training=self.training,
            )
            outputs["attn_weights_pd"] = w_pd
            outputs["pooled_value"] = pooled
            aff_scalar = pooled
        elif self.pooling == "attention":
            # Dense attention path (fallback)
            # Build pair-bias densely
            u_i = u[:, None, :].expand(-1, Lc, -1)  # [Lc,Lc,H]
            v_j = v[None, :, :].expand(Lc, -1, -1)  # [Lc,Lc,H]
            hadamard = u_i * v_j
            gate = torch.cat([u_i, v_j, hadamard], dim=-1)
            bias = self.to_bias(gate)  # [Lc,Lc,C_pair]
            z_hat = z + bias

            # Dist projection
            d_proj = self.dist_proj(dist_bins_t)  # [Lc,Lc,C_pair]
            h = torch.cat([z_hat, d_proj], dim=-1)
            h = self.ln(h)

            # Mask PD edges
            aff_mask_np = masks.affinity_pair_mask
            if aff_mask_np.shape != (Lc, Lc):
                raise ValueError("affinity_pair_mask shape mismatch")
            aff_mask_t = torch.from_numpy(aff_mask_np).to(device)

            # Build priors logits from available signals
            priors_logits: Optional[torch.Tensor] = None
            if (prior_contact is not None) or (prior_pae is not None) or (prior_pde is not None):
                c_t = prior_contact.to(device) if isinstance(prior_contact, torch.Tensor) else None
                pae_t = prior_pae.to(device) if isinstance(prior_pae, torch.Tensor) else None
                pde_t = prior_pde.to(device) if isinstance(prior_pde, torch.Tensor) else None
                priors_logits = make_prior_logits(
                    contact=c_t,
                    pae=pae_t,
                    pde=pde_t,
                    w_contact=self.prior_w_contact,
                    w_pae=self.prior_w_pae,
                    w_pde=self.prior_w_pde,
                    eps=self.prior_eps,
                )

            pooled, attn_w = self.attn_pool(
                edge_feats=h,
                pd_mask=aff_mask_t,
                priors_logits=priors_logits,
                temp=self.pool_temp,
                training=self.training,
            )
            outputs["attn_weights"] = attn_w
            aff_scalar = pooled
        else:
            # Legacy LSE or mean pooling
            # Build pair-bias densely for LSE path
            u_i = u[:, None, :].expand(-1, Lc, -1)  # [Lc,Lc,H]
            v_j = v[None, :, :].expand(Lc, -1, -1)  # [Lc,Lc,H]
            hadamard = u_i * v_j
            gate = torch.cat([u_i, v_j, hadamard], dim=-1)
            bias = self.to_bias(gate)  # [Lc,Lc,C_pair]
            z_hat = z + bias

            d_proj = self.dist_proj(dist_bins_t)
            h = torch.cat([z_hat, d_proj], dim=-1)
            h = self.ln(h)
            edge_scores_raw = self.mlp(h).squeeze(-1)  # [Lc,Lc]

            aff_mask_np = masks.affinity_pair_mask
            if aff_mask_np.shape != (Lc, Lc):
                raise ValueError("affinity_pair_mask shape mismatch")
            aff_mask_t = torch.from_numpy(aff_mask_np).to(device)

            edge_scores = edge_scores_raw
            if edge_weights is not None:
                if edge_weights.shape != edge_scores_raw.shape:
                    raise ValueError("edge_weights must be [Lc,Lc]")
                edge_weights = edge_weights.to(device)
                edge_scores = edge_scores + edge_weights

            if self.use_soft_pool:
                neg_inf = torch.finfo(edge_scores.dtype).min
                masked = torch.where(aff_mask_t, edge_scores, torch.tensor(neg_inf, device=device))
                temp = max(1e-6, self.pool_temp)
                vals = masked[aff_mask_t]
                aff_scalar = (temp * torch.logsumexp(vals / temp, dim=0)) if vals.numel() else edge_scores.new_tensor(0.0)
            else:
                denom = aff_mask_t.sum().clamp(min=1)
                aff_scalar = (edge_scores * aff_mask_t).sum() / denom
            outputs["edge_scores"] = edge_scores

        return {"affinity": aff_scalar, **outputs}


# =====================
# Sanity check utilities
# =====================


class TFAffinityRegressor(nn.Module):
    def __init__(self, c_pair: int, c_single: int, n_bins: int,
                 use_soft_pool: bool = True,
                 pooling: str = "attention",
                 attn_hidden: int = 128,
                 attn_dropout: float = 0.10,
                 edge_dropout: float = 0.10,
                 pool_temp: float = 4.0,
                 noise_std: float = 0.02,
                 prior_w_contact: float = 1.0,
                 prior_w_pae: float = 0.25,
                 prior_w_pde: float = 0.10,
                 prior_eps: float = 1e-6) -> None:
        super().__init__()
        self.head = BoltzAffinityHeadReplica(
            c_pair=c_pair,
            c_single=c_single,
            b_bins=n_bins,
            use_soft_pool=use_soft_pool,
            pool_temp=pool_temp,
            pooling=pooling,
            attn_hidden=attn_hidden,
            attn_dropout=attn_dropout,
            edge_dropout=edge_dropout,
            noise_std=noise_std,
            prior_w_contact=prior_w_contact,
            prior_w_pae=prior_w_pae,
            prior_w_pde=prior_w_pde,
            prior_eps=prior_eps,
        )
        self.calib = nn.Linear(1, 1)
        # Start near identity so early losses are sane
        with torch.no_grad():
            self.calib.weight.fill_(1.0)
            self.calib.bias.zero_()

    def forward(self,
                z: torch.Tensor,
                s_proxy: torch.Tensor,
                dist_bins: torch.Tensor | np.ndarray,
                masks: CropMasks,
                edge_weights: Optional[torch.Tensor] = None,
                prior_contact: Optional[torch.Tensor] = None,
                prior_pae: Optional[torch.Tensor] = None,
                prior_pde: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.head(
            z, s_proxy, dist_bins, masks,
            edge_weights=edge_weights,
            prior_contact=prior_contact,
            prior_pae=prior_pae,
            prior_pde=prior_pde,
        )
        aff = out["affinity"].reshape(1, 1)
        y_hat = self.calib(aff).reshape(-1)  # (1,)
        return y_hat, out


class AttnPool(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, attn_dropout: float = 0.1, edge_dropout: float = 0.1) -> None:
        super().__init__()
        self.value_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(attn_dropout),
            nn.Linear(hidden, 1)
        )
        self.logit_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(attn_dropout),
            nn.Linear(hidden, 1)
        )
        self.edge_dropout = float(edge_dropout)

    def forward(self,
                edge_feats: torch.Tensor,
                pd_mask: torch.Tensor,
                priors_logits: Optional[torch.Tensor] = None,
                temp: float = 4.0,
                training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        v = self.value_mlp(edge_feats).squeeze(-1)
        logit = self.logit_mlp(edge_feats).squeeze(-1)
        if priors_logits is not None:
            logit = logit + priors_logits.to(device=logit.device, dtype=logit.dtype)
        neg_inf = torch.finfo(logit.dtype).min
        logit = torch.where(pd_mask, logit, torch.tensor(neg_inf, device=logit.device, dtype=logit.dtype))
        if training and self.edge_dropout > 0:
            keep = (torch.rand_like(logit) > self.edge_dropout) | (~pd_mask)
            logit = torch.where(keep, logit, torch.tensor(neg_inf, device=logit.device, dtype=logit.dtype))
        flat = (logit / max(1e-6, float(temp))).reshape(-1)
        w = torch.softmax(flat, dim=-1).reshape_as(logit)
        pooled = (w * v).sum()
        return pooled, w

    def forward_sparse(self,
                       edge_feats: torch.Tensor,
                       priors_logits: Optional[torch.Tensor] = None,
                       temp: float = 4.0,
                       training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # edge_feats: [K, D]
        v = self.value_mlp(edge_feats).squeeze(-1)  # [K]
        logit = self.logit_mlp(edge_feats).squeeze(-1)  # [K]
        if priors_logits is not None:
            logit = logit + priors_logits.to(device=logit.device, dtype=logit.dtype)
        if training and self.edge_dropout > 0:
            keep = (torch.rand_like(logit) > self.edge_dropout)
            neg_inf = torch.finfo(logit.dtype).min
            logit = torch.where(keep, logit, torch.tensor(neg_inf, device=logit.device, dtype=logit.dtype))
        w = torch.softmax(logit / max(1e-6, float(temp)), dim=0)  # [K]
        pooled = (w * v).sum()
        return pooled, w


def make_prior_logits(contact: Optional[torch.Tensor] = None,
                      pae: Optional[torch.Tensor] = None,
                      pde: Optional[torch.Tensor] = None,
                      w_contact: float = 1.0,
                      w_pae: float = 0.25,
                      w_pde: float = 0.10,
                      eps: float = 1e-6,
                      pae_scale: float = 10.0,
                      pde_scale: float = 10.0) -> torch.Tensor:
    pri = None
    if contact is not None:
        c = contact.to(dtype=torch.float32)
        term = float(w_contact) * torch.log(c.clamp_min(float(eps)))
        pri = term if pri is None else pri + term
    if pae is not None:
        paet = pae.to(dtype=torch.float32)
        term = float(w_pae) * (-(paet / float(pae_scale)))
        pri = term if pri is None else pri + term
    if pde is not None:
        pdet = pde.to(dtype=torch.float32)
        term = float(w_pde) * (-(pdet / float(pde_scale)))
        pri = term if pri is None else pri + term
    if pri is None:
        # default zero log-prior
        pri = torch.tensor(0.0)
    return pri


def indexing_roundtrip_ok(crop_to_full: np.ndarray, Lf: int) -> bool:
    rng = np.random.default_rng(0)
    Lc = len(crop_to_full)
    M = rng.standard_normal((Lf, Lf)).astype(np.float32)
    crop = M[np.ix_(crop_to_full, crop_to_full)]
    # now expand back
    back = np.zeros_like(M)
    back[np.ix_(crop_to_full, crop_to_full)] = crop
    # equality on the submatrix
    return np.allclose(back[np.ix_(crop_to_full, crop_to_full)], crop)


def geometry_sanity(rep_xyz_crop: np.ndarray, mol_type_crop: np.ndarray) -> bool:
    is_prot = (mol_type_crop == 0)
    is_dna = (mol_type_crop == 1)
    if not (is_prot.any() and is_dna.any()):
        return False
    coords = rep_xyz_crop
    D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    pd = D[np.ix_(is_prot, is_dna)]
    pp = D[np.ix_(is_prot, is_prot)]
    dd = D[np.ix_(is_dna, is_dna)]
    med_pd = float(np.median(pd)) if pd.size else math.inf
    med_oth = float(np.median(np.concatenate([pp.flatten(), dd.flatten()])) if (pp.size or dd.size) else np.array([math.inf]))
    return med_pd < med_oth


def pd_correlation(edge_scores: np.ndarray, contact_probs: np.ndarray, aff_mask: np.ndarray) -> float:
    mask = aff_mask & np.isfinite(contact_probs)
    if not mask.any():
        return float("nan")
    x = edge_scores[mask].reshape(-1)
    y = contact_probs[mask].reshape(-1)
    # Spearman approximation via ranks
    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, ranks)
        avg = sums / counts
        return avg[inv]
    rx = _rank(x)
    ry = _rank(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-9)
    ry = (ry - ry.mean()) / (ry.std() + 1e-9)
    return float(np.mean(rx * ry))


def compute_edge_weights(contact: Optional[np.ndarray],
                         pae: Optional[np.ndarray],
                         pde: Optional[np.ndarray],
                         alpha: float = 2.0,
                         beta: float = 0.2,
                         gamma: float = 0.2,
                         tau: float = 0.2,
                         mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    if contact is None and pae is None and pde is None:
        return None
    base_shape = None
    for arr in (contact, pae, pde):
        if arr is not None:
            base_shape = arr.shape
            break
    if base_shape is None:
        return None
    w = np.zeros(base_shape, dtype=np.float32)
    if contact is not None:
        w += alpha * np.maximum(contact.astype(np.float32) - float(tau), 0.0)
    if pae is not None:
        w -= beta * pae.astype(np.float32)
    if pde is not None:
        w -= gamma * pde.astype(np.float32)
    if mask is not None:
        w = np.where(mask, w, 0.0)
    return w


@torch.no_grad()
def predict_intensity(model: nn.Module, sample) -> Tuple[float, Dict[str, torch.Tensor]]:
    """Predict intensity_log1p for a single sample using a trained regressor.

    Expects `sample` to provide attributes: z, s or s_proxy, dist_bins, masks.
    Moves tensors to the model's device before running inference.
    """
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    z = getattr(sample, "z")
    s = getattr(sample, "s", None)
    if s is None:
        s = getattr(sample, "s_proxy")
    dist_bins = getattr(sample, "dist_bins")
    masks = getattr(sample, "masks")
    edge_w = getattr(sample, "edge_weights", None)

    z = z.to(device)
    s = s.to(device)
    # head will move dist_bins to z.device; leave as-is if numpy/tensor

    y_hat, out = model(z, s, dist_bins, masks, edge_weights=edge_w)
    return float(y_hat.detach().cpu().reshape(()).item()), out


__all__ = [
    "CropBundle",
    "TokenGeom",
    "CropMasks",
    "load_crop_bundle",
    "parse_cif_to_token_geom",
    "make_crop_masks",
    "build_dist_bins",
    "build_dist_rbf",
    "BoltzAffinityHeadReplica",
    "TFAffinityRegressor",
    "AttnPool",
    "make_prior_logits",
    "predict_intensity",
    "indexing_roundtrip_ok",
    "geometry_sanity",
    "pd_correlation",
    "compute_edge_weights",
]


