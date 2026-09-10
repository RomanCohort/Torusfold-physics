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


# 1EHZ-measured anchor offsets in a full local frame
#   b = P[i] -> P[i+1] (unit)
#   r = direction to the base-pair partner, orthogonalized against b (unit)
#   n = b x r
# Offset = (along b, along r, along n) of (X - P[i]), Angstrom.
# Measured on 45 paired standard residues of 1EHZ chain A (scripts/measure_frame3d.py).
# C4-prime sits mostly along n (-2.2) and hardly along r (0.9), while C1-prime is spread
# over both - which is why one shared perpendicular axis could not place both.
# Per-base constants were tried and are not better (2.867 A vs 2.820 A overall on the
# 62-residue 1EHZ test), so the pooled means are used.
_ANCHOR_OFFSETS = {
    "C1'": (3.36, 2.63, -2.35),
    "C4'": (2.97, 0.65, -2.07),
}

# Half-width (in residues) of the P-trace window whose centroid is used as a local
# axis point for residues without a base-pair partner. Set to >= L to recover the old
# global-centroid behaviour.
_FALLBACK_WINDOW = 2

# Sign of the fallback radial axis. The offsets were measured with r pointing at the
# base-pair partner, i.e. inward across the helix; the axis-radial direction for an
# unpaired base points outward. Measured on the 1EHZ test, the unpaired group goes from
# 6.24 A (sign +1) to 4.44 A (sign -1), so the two conventions are indeed opposed.
_FALLBACK_SIGN = -1.0


def real_cg_beads(p_coords: np.ndarray, sequence: str, pairs=None) -> np.ndarray:
    """Per-residue (P, C4', N9 or N1) beads in Angstrom, shape (L, 3, 3).

    The CG force field lays its beads out as P / C4' / N per residue. Where those beads
    are fabricated -- a backbone-direction offset, or a random perturbation of P -- they
    sit on the backbone axis and carry no base identity, so no base-specific quantity can
    be expressed on them and a CG-level statistical potential has nothing to attach to.
    This returns them from the 1EHZ template reconstruction instead.

    Note the force field's own intra-bead targets are already right: it restrains
    |P-C4'| to 3.90 A and |C4'-N| to 3.35 A (torch_cgsim.py:103,124,125) against 1EHZ
    measurements of 3.887 +/- 0.081 and 3.428 +/- 0.266 A.

    Raises rather than falling back: a missing or non-ACGU sequence would silently give
    the caller fabricated beads again.
    """
    if sequence is None:
        raise ValueError("real_cg_beads needs the sequence to choose N9 (purine) or N1")
    seq = sequence.upper().replace("T", "U")
    bad = sorted({c for c in seq if c not in "ACGU"})
    if bad:
        raise ValueError(f"real_cg_beads: only ACGU is supported, got {bad}")

    structure = reconstruct_all_atom(p_coords, seq, pairs=pairs)
    L = len(seq)
    out = np.zeros((L, 3, 3), dtype=np.float64)
    for i in range(L):
        idx = structure.residue_atom_index[i]
        gly = "N9" if seq[i] in "AG" else "N1"
        for k, nm in enumerate(("P", "C4'", gly)):
            out[i, k] = np.asarray(structure.atoms[idx[nm]].xyz, dtype=np.float64)
    return out


def reconstruct_all_atom(
    p_coords: np.ndarray, sequence: str, pairs=None,
) -> AllAtomStructure:
    """CG P coordinates -> all-atom RNA (1EHZ crystal template).

    Args:
        p_coords: (L, 3) Å, one P atom per nucleotide (CG solver output)
        sequence: ACGU string of length L
        pairs: optional base pairs, any of (i, j) / (i, j, w). When given, a paired
            residue's perpendicular anchor axis points at its partner, so the base
            faces the base it pairs with instead of radiating from the centroid.
            Unpaired residues keep the radial fallback.
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

    # base-pair partner map (first partner wins); absent pairs -> radial fallback
    partner_of: Dict[int, int] = {}
    if pairs:
        for pr in pairs:
            try:
                i0, j0 = int(pr[0]), int(pr[1])
            except (TypeError, IndexError, ValueError):
                continue
            if 0 <= i0 < L and 0 <= j0 < L and i0 != j0:
                partner_of.setdefault(i0, j0)
                partner_of.setdefault(j0, i0)

    templates = _load_templates()
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

        # CG supplies only P[i]; the C1'/C4'/O3' targets are inferred in the local frame.
        #   backbone direction b = P[i+1] - P[i] (the last residue uses P[0]-P[L-1])
        #   perpendicular r = direction to the base-pair partner, projected off b;
        #     unpaired residues fall back to the radial direction P[i] - centroid
        # The C1'/C4' offsets are the 1EHZ-measured means decomposed along b
        # (61 standard residues, chain A; see docs/reconstruction_anchor_audit.md):
        #   C1': 3.25 along + 4.16 perpendicular  (|C1'-P| = 5.33 +/- 0.19)
        #   C4': 2.79 along + 2.63 perpendicular  (|C4'-P| = 3.90 +/- 0.04)
        # The previous constants (5.5/1.5 and 4.2/0.0) put every anchor on the backbone
        # axis and made two of the four targets coincide with O3' at P[i+1]-1.6b:
        # 0.01 A apart against a real |C4'-O3'| of 2.44 +/- 0.04 A. With a degenerate
        # target tetrahedron the roll about b is undetermined, which misplaced the base
        # atoms by 6-9 A.
        nxt = p_coords[(i + 1) % L]
        b = nxt - p_coords[i]
        bn = np.linalg.norm(b)
        b = b / bn if bn > 1e-6 else np.array([1.0, 0.0, 0.0])

        r = None
        j = partner_of.get(i)
        if j is not None and 0 <= j < L:
            d = p_coords[j] - p_coords[i]
            d = d - np.dot(d, b) * b
            dn = np.linalg.norm(d)
            if dn > 1e-6:
                r = d / dn
        if r is None:
            # Unpaired residue: use the centroid of a local P-trace window as a local
            # axis point. The previous rule used the global centroid, which is the
            # special case _FALLBACK_WINDOW = L and is only a crude proxy for the local
            # helix axis.
            _w = min(max(1, int(_FALLBACK_WINDOW)), L)
            _c = p_coords[[(i + k) % L for k in range(-_w, _w + 1)]].mean(axis=0)
            r = (p_coords[i] - _c) * _FALLBACK_SIGN
            rn = np.linalg.norm(r)
            r = r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0])
            r = r - np.dot(r, b) * b  # orthogonalize into the plane normal to b
            rn = np.linalg.norm(r)
            r = r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0])

        n_axis = np.cross(b, r)  # r is already orthogonal to b and unit
        _o1, _o4 = _ANCHOR_OFFSETS["C1'"], _ANCHOR_OFFSETS["C4'"]
        c1_dst = p_coords[i] + b * _o1[0] + r * _o1[1] + n_axis * _o1[2]
        c4_dst = p_coords[i] + b * _o4[0] + r * _o4[1] + n_axis * _o4[2]
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
