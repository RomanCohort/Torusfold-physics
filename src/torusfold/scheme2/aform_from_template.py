"""aform_from_template.py - reconstructs all-atom RNA from 1EHZ tRNA crystal standard residues.

Replaces the hand-derived templates in allatom_reconstruct.py. The old
hand-built geometry (P-O5'=1.6Å, etc.) deviates from the amber14 OL3 force-field
equilibrium, so after minimization amber_field stayed positive (+70,000 kJ/mol,
which is physically unreasonable). We now build the template from the real
experimental coordinates of 1EHZ (yeast tRNA^Phe, 1.93Å high-resolution crystal):

  1. Take the four standard A/U/G/C residue coordinates from aform_template.npz (real crystal)
  2. For each CG P point, take the standard residue of the matching base
  3. Align the standard residue onto the CG local frame with a three-point (P + C1' + C4') Kabsch fit
  4. Intra-residue coordinates are then real crystal geometry, so the amber energy starts negative

BSJ closure: the O3'/P of the first and last circRNA residues is closed by
amber_refine's HarmonicBondForce restraint (after Kabsch superposition the
terminal O3'-P distance is already near the real ~1.6Å, so a tiny force-field
adjustment closes it).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Reuse the old AllAtomStructure interface so predictors need no changes
from .allatom_reconstruct import AllAtomStructure, Atom


_TEMPLATE_PATH = Path(__file__).parent / "aform_template.npz"
_templates: Dict[str, Dict] = {}


def _load_templates() -> Dict[str, Dict]:
    """Lazily load the 1EHZ standard-residue templates (A/U/G/C)."""
    if _templates:
        return _templates
    data = np.load(_TEMPLATE_PATH, allow_pickle=True)
    for base in "AUGC":
        names = [str(n) for n in data[f"{base}_names"]]
        coords = np.asarray(data[f"{base}_coords"], dtype=np.float32)
        _templates[base] = {"names": names, "coords": coords}
    return _templates


def _kabsch_align(
    src_three: np.ndarray, dst_three: np.ndarray,
    src_all: np.ndarray,
) -> np.ndarray:
    """Three-point Kabsch: transform src_all into the frame that maps src_three -> dst_three.

    Args:
        src_three: (3, 3) the three template anchor points (P, C1', C4') as row vectors
        dst_three: (3, 3) the three target anchor points (P, C1', C4' derived from CG)
        src_all:   (N, 3) all template atom coordinates
    Returns:
        (N, 3) transformed coordinates (translation + rotation, affine alignment)
    """
    # Center
    src_c = src_three.mean(axis=0)
    dst_c = dst_three.mean(axis=0)
    s = src_three - src_c
    d = dst_three - dst_c
    # Kabsch: R = argmin ||R @ s - d||, solved by SVD
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    # Reflection correction: keep R free of mirroring (det=-1); the variable is
    # named refl to avoid clashing with the d above
    refl = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, refl])
    R = Vt.T @ D @ U.T
    # Transform: center the template at the origin, rotate, then translate to the target center
    aligned = (src_all - src_c) @ R.T + dst_c
    return aligned.astype(np.float32)


def reconstruct_all_atom(
    p_coords: np.ndarray, sequence: str,
) -> AllAtomStructure:
    """CG P coordinates -> all-atom RNA (1EHZ crystal template).

    Args:
        p_coords: (L, 3) Å, one P atom per nucleotide (CG solver output)
        sequence: ACGU string of length L
    Returns:
        AllAtomStructure whose per-residue all-atom coordinates are Kabsch-superposed
        1EHZ standard residues.
    """
    # Normalize the sequence: case + T→U
    sequence = sequence.upper().replace("T", "U")
    if p_coords.ndim != 2 or p_coords.shape[1] != 3:
        raise ValueError(f"p_coords has unexpected shape {p_coords.shape}, expected (L,3)")
    L = len(sequence)
    if p_coords.shape[0] != L:
        raise ValueError(f"sequence length {L} != P count {p_coords.shape[0]}")
    bad = [c for c in sequence if c not in "ACGU"]
    if bad:
        raise ValueError(f"sequence contains invalid letters {set(bad)}; only ACGU allowed")

    templates = _load_templates()
    centroid = p_coords.mean(axis=0)
    structure = AllAtomStructure(sequence=sequence)

    serial = 0
    for i in range(L):
        base = sequence[i]
        tmpl = templates[base]
        names = tmpl["names"]
        tcoords = tmpl["coords"]  # (N, 3) template coordinates

        # Find the four anchors P / C1' / C4' / O3' in the template
        # P1 fix: add the O3' anchor (phosphate-bridge geometry) so the rebuilt O3'
        # position is consistent with the P of the neighboring residue, avoiding the
        # catastrophic geometry (C3'-O3'-P bent ~70°) that arose when O3' was inherited
        # from the template right next to the following residue's P
        idx_P = names.index("P")
        idx_C1 = names.index("C1'")
        idx_C4 = names.index("C4'")
        idx_O3 = names.index("O3'")
        src_anchors = np.stack([tcoords[idx_P], tcoords[idx_C1],
                                tcoords[idx_C4], tcoords[idx_O3]])

        # CG only supplies P[i]; the C1'/C4'/O3' targets are inferred in the local frame (approximate A-form geometry):
        #   backbone direction b = P[i+1] - P[i] (the last residue uses P[0]-P[L-1])
        #   radial r = P[i] - centroid (bases point outward)
        #   C1' lies +5.5Å along the backbone and +1.5Å radially from P (A-form statistics)
        #   C4' lies +4.2Å along the backbone and 0 radially from P
        #   O3' is placed 1.6Å back from P[i+1] (A-form O3'-P bond length of 1.6Å)
        nxt = p_coords[(i + 1) % L]
        b = nxt - p_coords[i]
        bn = np.linalg.norm(b)
        b = b / bn if bn > 1e-6 else np.array([1.0, 0.0, 0.0])
        r = p_coords[i] - centroid
        rn = np.linalg.norm(r)
        r = r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0])
        r = r - np.dot(r, b) * b  # orthogonalize into the plane normal to b
        rn = np.linalg.norm(r)
        r = r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0])

        c1_dst = p_coords[i] + b * 5.5 + r * 1.5
        c4_dst = p_coords[i] + b * 4.2
        o3_dst = nxt - b * 1.6  # O3'[i] consistent with the geometry of P[i+1]
        dst_anchors = np.stack([p_coords[i], c1_dst, c4_dst, o3_dst])

        # Kabsch superposition (4-point least squares; one more O3' constraint than the 3-point version)
        aligned = _kabsch_align(src_anchors, dst_anchors, tcoords)

        # Populate the structure
        res_name = base
        res_seq = i + 1
        atom_index: Dict[str, int] = {}
        start = len(structure.atoms)
        for k, name in enumerate(names):
            element = name[0]
            if name[0] == "O" and len(name) > 1 and name[1].isdigit():
                element = "O"
            structure.atoms.append(Atom(
                serial=serial, res_seq=res_seq, res_name=res_name,
                atom_name=name, element=element, xyz=aligned[k],
            ))
            atom_index[name] = serial
            serial += 1
        end = len(structure.atoms)
        structure.residue_atom_spans.append((start, end))
        structure.residue_atom_index.append(atom_index)

    return structure


if __name__ == "__main__":
    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    print(f"L={L} atoms={len(s.atoms)} per_residue={len(s.atoms)/L:.1f}")
    print(f"Residue 0 atoms: {[a.atom_name for a in s.atoms[:s.residue_atom_spans[0][1]]]}")
