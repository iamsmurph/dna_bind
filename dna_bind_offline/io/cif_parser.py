"""CIF parsing utilities using gemmi or Biopython fallback."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..models.types import TokenGeom

try:
    import gemmi  # type: ignore
    _HAS_GEMMI = True
except Exception:
    _HAS_GEMMI = False


def parse_cif_to_token_geom(cif_path: str, expected_Lf: Optional[int] = None) -> TokenGeom:
    if not _HAS_GEMMI:
        # Fallback to Biopython
        try:
            from Bio.PDB.MMCIF2Dict import MMCIF2Dict  # type: ignore
        except Exception as e:
            raise ImportError("Neither gemmi nor Biopython available for CIF parsing") from e
        d = MMCIF2Dict(cif_path)
        chains = d.get("_atom_site.label_asym_id", [])
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
            pass
        if Lf >= 2:
            a = rep[0]
            b = rep[-1]
            if float(np.linalg.norm(a - b)) > 200.0:
                raise ValueError("Unreasonable inter-token distance > 200 Å")
        return TokenGeom(rep_xyz=rep, mol_type=mol_type, token_pad_mask=pad, token_meta=meta)

    # gemmi path
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


__all__ = ["parse_cif_to_token_geom"]


