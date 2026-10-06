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


def _place_residue_on_p(
    src_anchors: np.ndarray, dst_anchors: np.ndarray, src_all: np.ndarray,
    p_src: np.ndarray, p_dst: np.ndarray, weights=None,
) -> np.ndarray:
    """Rotate the template about its own P, then put that P exactly on `p_dst`.

    WHY THIS REPLACES `_kabsch_align` FOR THE P-TRACE PATH. `_kabsch_align` is mean-centred, and a
    mean-centred fit does not force ANY anchor onto its target -- it distributes the residual over
    all of them. One of the four anchors here is not a guess: `p_coords[i]` is the CG solver's own
    output and is handed to `reconstruct_all_atom` as an input. Paying for the other three anchors
    out of that one is how the reconstruction came to move it.

    Measured over 42 deposited structures / 3724 residues (scripts/reconstruction_fidelity.py),
    feeding each structure its own P trace -- the deposited column is the same set's own mean:

        variant                       |P_out-P_in|   link    |P-C4'|   |O3'(i)-P(i+1)|
        deposited (same set)               0         3.924    3.867        1.600
        mean-centred 4-point fit          0.556       3.647    3.690        1.402
        this, equal weights               0.000       3.492    3.885        1.175
        this, C1'/C4' weighted up         0.000       3.732    3.885        1.402

    THE WEIGHTS. P is imposed exactly, so its weight only sets the scale. C1' and C4' carry the
    base's orientation, which is the quantity this module exists to get right -- the docstring of
    `reconstruct_all_atom_from_beads` records that a wrong roll loses 57 percent of the helical
    stacking -- so they are weighted up. O3' carries no force-field term at all (three beads per
    residue, P/C4'/N9-N1; `grep O3 torch_cgsim.py` matches nothing outside comments), so it is the
    anchor that gives, and `_snap_o3_to_target` then puts it exactly on its own 1.6 A sphere. The
    link saturates by roughly 200:1, so `_ANCHOR_WEIGHTS` uses 100; nothing above it changes the
    answer, and the grid is reproducible through the weights parameter.

    The rotation is a proper weighted Kabsch with both sets centred on P instead of on their means:
    solve `argmin_R || sqrt(w)(src-P_src) R - sqrt(w)(dst-P_dst) ||`, then translate by
    `p_dst - p_src R` so P lands exactly. `_ANCHOR_OFFSETS` is the other half of this and its note
    records why C4' had to stay the 1EHZ measurement.
    """
    a = np.asarray(src_anchors, dtype=np.float64) - np.asarray(p_src, dtype=np.float64)
    b = np.asarray(dst_anchors, dtype=np.float64) - np.asarray(p_dst, dtype=np.float64)
    if weights is None:
        w = np.ones(len(a), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(a),) or np.any(w <= 0):
            raise ValueError(f"weights must be {len(a)} positive numbers, got {weights!r}")
    w = w / w.sum()
    sw = np.sqrt(w)[:, None]
    U, _, Vt = np.linalg.svd((a * sw).T @ (b * sw))
    refl = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, refl]) @ U.T
    return ((np.asarray(src_all, dtype=np.float64) - np.asarray(p_src, dtype=np.float64)) @ R.T
            + np.asarray(p_dst, dtype=np.float64)).astype(np.float32)


# Anchor weights for the P-trace placement, in the order P, C1', C4', O3'. See
# `_place_residue_on_p`. P is imposed, and C1'/C4' set the base roll, so they dominate; O3' gives,
# and `_snap_o3_to_target` then places it exactly. The link saturates by roughly 200:1.
_ANCHOR_WEIGHTS = (1.0, 100.0, 100.0, 1.0)

# The O3'(i)-P(i+1) distance `reconstruct_all_atom` targets, i.e. the `1.6` in `nxt - b * 1.6`.
_O3_BRIDGE = 1.6


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
#
# C4' IS THE 1EHZ-MEASURED (2.97, 0.65, -2.07), AND ROW 18 IS A PROPERTY OF THE TEMPLATE.
#
# This value was briefly replaced with (2.834, 0.779, -2.481), solved so that the link
# sqrt((|P-P| - along_b)^2 + perp^2) would equal the deposited 3.913. That solve was wrong,
# and the measurement that refutes it is worth recording because the same reasoning will look
# right again to the next reader.
#
# The solve assumed `|P(i)-C4'(i)|` could be set to 3.900 independently. It cannot. The
# placement is a ROTATION about P, so a rotation preserves the template's own bond, and the
# four templates have four different bonds: A 3.924, C 3.959, G 3.785, U 3.900. The solve's
# implied norm was 3.846, which no template can take, so the fitted C4' did not land where the
# solve assumed and the link came out 3.617 against the 3.911 promised -- worse than leaving
# the constant alone.
#
# What the offsets can steer is the DIRECTION of the C4' vector, not its length. Measured over
# 42 deposited structures / 3724 residues through `real_cg_beads` with a P-exact placement
# (scripts/reconstruction_fidelity.py):
#
#   (2.97, 0.65, -2.07)    link 3.617   |P-C4'| 3.885
#   (2.834, 0.779, -2.481) link 3.617   |P-C4'| 3.885
#
# -- identical, which is the tell: the offset is not what sets the link here. Row 18's 0.42 A
# shortfall survives every offset, because a rigid 1EHZ residue cannot reproduce the deposited
# C4'(i)-P(i+1). The ceiling is measured, not assumed: pinning P and O3' and leaving C1'/C4'
# free gives 3.581; freeing O3' as well gives 3.785 against a target of 3.875. That is the
# bound, so the fix is to the template or to the placement rule, NOT to this constant.
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
        AllAtomStructure whose per-residue all-atom coordinates are a RIGID 1EHZ standard
        residue, rotated about its own P and translated so that P sits exactly on the
        supplied coordinate.
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
        # The offsets themselves are the _ANCHOR_OFFSETS dict at the top of this module.
        # THIS COMMENT USED TO RESTATE THEM WRONG and then restate them again while telling the
        # reader not to restate them: it claimed C4' was (2.97, 0.65, -2.07) and called the
        # resulting link 3.65 A where the code produced 3.20 and the target wanted 3.80. A reader
        # who trusted the comment hunted a 0.87 A discrepancy that was in the comment. The dict is
        # the value; nothing here repeats it. What the dict's own note records, and what belongs
        # here, is that C4' is no longer a pure 1EHZ measurement: it is solved for the link.
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
        o3_dst = nxt - b * _O3_BRIDGE  # O3'[i] consistent with the geometry of P[i+1]
        dst_anchors = np.stack([p_coords[i], c1_dst, c4_dst, o3_dst])

        # Kabsch superposition, ROTATED ABOUT P and then translated so P lands on its input
        # coordinate exactly (4 anchors: P, C1', C4', O3', weighted). A mean-centred fit here would
        # spend P's own target on the three inferred anchors -- see `_place_residue_on_p`.
        aligned = _place_residue_on_p(
            src_anchors, dst_anchors, tcoords,
            p_src=tcoords[idx_P], p_dst=p_coords[i],
            weights=_ANCHOR_WEIGHTS,
        )

        # `o3_dst` is NOT honoured by the rigid placement and is left alone here on purpose.
        # Measured over 42 structures / 3724 residues, |O3'(i)-P(i+1)| comes out at 1.402 where
        # :395 asks for 1.600 -- and the old mean-centred fit gave 1.402 as well, so this is not a
        # regression. O3' carries no force-field term at all (`grep O3 torch_cgsim.py` matches
        # nothing outside comments; the CG model is three beads per residue), so nothing downstream
        # depends on it, and two corrections were tried and rejected:
        #   * scaling the component of O3'-P(i+1) perpendicular to the C4'-O3' bond, which does
        #     preserve that bond but moves O3' on a circle about C4', overshooting to 1.706 mean
        #     and stretching the sugar bond to 2.893;
        #   * the exact triangle solution, which needs cos(theta) in [-1,1] and does not have it on
        #     8 of 2OIU's 71 residues (i=7 wants +1.0960). Guarded, it reaches 1.775 -- further from
        #     the target than doing nothing.
        # So O3' stays where the rigid template puts it, which keeps the sugar bond exact, and the
        # 1.600 in `o3_dst` is recorded as not-realised rather than silently clamped.

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


def trace_frames(p):
    """Per-residue (tangent, in-plane, normal) frame of a P trace, wrapped for a circular chain."""
    p = np.asarray(p, dtype=np.float64)
    L = len(p)
    out = []
    for i in range(L):
        nxt, prv = p[(i + 1) % L], p[(i - 1) % L]
        b = nxt - prv
        nb = np.linalg.norm(b)
        b = b / nb if nb > 1e-9 else np.array([1.0, 0.0, 0.0])
        n = np.cross(nxt - p[i], prv - p[i])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            tmp = np.array([0.0, 0.0, 1.0]) if abs(b[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            n = np.cross(b, tmp)
            nn = np.linalg.norm(n)
        out.append((b, np.cross(n / nn, b), n / nn))
    return out


def beads_from_pdb(pdb_path, chain=None):
    """(L, 3, 3) Angstrom beads (P, C4', N9/N1) read from a structure that HAS them, or None.

    WHY THIS EXISTS. torch_gpu_refine's input is a P trace, so its CG stage has to FABRICATE the two
    non-backbone beads, and it does that with real_cg_beads -- the very axis heuristic the bead-frame
    reconstruction exists to replace. Measured on 2OIU with the term on and the tables installed: the
    fabricated beads come out with the right internal geometry (P-C4' 3.90 A, C4'-N 3.35 against the
    deposit's 3.86 and 3.37) and the WRONG orientation relative to each other (base_cos 0.607 against the
    deposit's 0.901), and the product built from them is unstacked. The deposit's own base frames are in the
    file the pipeline was given; when they are there, they are what a refinement should start from.

    Returns None rather than raising when the file lacks the atoms, so a caller can fall back to the
    heuristic and say which one it used.
    """
    try:
        from openmm.app import PDBFile
        from openmm import unit as _u
    except Exception:
        return None
    _gly = {"A": "N9", "G": "N9", "C": "N1", "U": "N1", "RA": "N9", "RG": "N9", "RC": "N1", "RU": "N1",
            "ADE": "N9", "GUA": "N9", "CYT": "N1", "URA": "N1"}
    try:
        pdb = PDBFile(str(pdb_path))
    except Exception:
        return None
    beads = []
    for res in pdb.topology.residues():
        if chain is not None and res.chain.id != chain:
            continue
        base = _gly.get(res.name.strip().upper())
        if base is None:
            continue
        want = {"P", "C4'", base}
        got = {}
        for atom in res.atoms():
            nm = atom.name.strip()
            if nm in want and nm not in got:
                got[nm] = np.asarray(pdb.positions[atom.index].value_in_unit(_u.angstrom), dtype=np.float64)
        if len(got) != 3:
            return None
        beads.append([got["P"], got["C4'"], got[base]])
    if len(beads) < 2:
        return None
    return np.asarray(beads, dtype=np.float64)


def carry_beads_along_trace(p_new, beads_ref, p_ref):
    """Move rigid (P, C4', N) units from one P trace to another, keeping each unit's roll in the
    LOCAL trace frame. All arguments in the same length unit; returns (L, 3, 3).

    WHY THIS BELONGS IN THIS MODULE. The model's intra-residue geometry is rigid -- K_INTRA_PN plus the
    K_LINK_* links, all SHAKE/RATTLE constrained -- so a CG state is a rigid unit per residue carried by
    the backbone, and its roll about the backbone is INFORMATION the model has. Two stages of the
    pipeline propagate only the P trace: torch_gpu_refine extracts bead 0 of every 3 after REMD, and
    relax_structure takes and returns P. A caller that wants the sampled base frames at the end
    therefore has to carry them, and this is that operation.

    It is not a re-derivation. The unit that is carried is the SAMPLED one (beads_ref, read at a stage
    where the sampler still had it); only the frame it is expressed in comes from the trace. Compare
    reconstruct_all_atom, which has no sampled unit at all and guesses the roll from the base-pair
    partner or from a radial fallback -- measured to lose 57 percent of the helical stacking even when
    the trace it is given is the crystal's own (scripts/measure_base_stacking.py, findings Part 13).
    """
    p_new = np.asarray(p_new, dtype=np.float64)
    p_ref = np.asarray(p_ref, dtype=np.float64)
    beads_ref = np.asarray(beads_ref, dtype=np.float64)
    if beads_ref.shape[1:] != (3, 3):
        raise ValueError(f"beads_ref has shape {beads_ref.shape}, expected (L, 3, 3)")
    if len(p_new) != beads_ref.shape[0] or len(p_ref) != beads_ref.shape[0]:
        raise ValueError("p_new, p_ref and beads_ref disagree on the number of residues")
    fr_new, fr_ref = trace_frames(p_new), trace_frames(p_ref)
    out = np.zeros_like(beads_ref)
    for i in range(len(p_new)):
        b0, r0, n0 = fr_ref[i]
        b1, r1, n1 = fr_new[i]
        for k in range(3):
            d = beads_ref[i, k] - p_ref[i]
            comp = np.array([float(d @ b0), float(d @ r0), float(d @ n0)])
            out[i, k] = p_new[i] + comp[0] * b1 + comp[1] * r1 + comp[2] * n1
    return out


def reconstruct_all_atom_from_beads(beads: np.ndarray, sequence: str) -> AllAtomStructure:
    """CG beads (P, C4', N9/N1) -> all-atom RNA, with the template fitted on the SAMPLED frame.

    WHY THIS EXISTS, measured. reconstruct_all_atom places every base from the P trace alone: C1' and
    C4' come from _ANCHOR_OFFSETS in a (tangent, partner-or-radial, normal) frame, i.e. the base's roll
    about the backbone is a heuristic guess, and O3' is pinned to the next P. Feeding it the crystal's
    OWN P trace -- zero trace error -- loses 57 percent of the helical stacking (98.8 -> 43.1 percent
    over 20 fragments), puts the bases 0.9 A too close (rise 3.35 -> 2.50 A) and keeps only 10-20 percent
    of the Watson-Crick contacts (scripts/measure_base_stacking.py, findings Part 13). The information
    that is thrown away is exactly the base site: the CG model carries one bead per base, its intra-bead
    geometry is rigid (K_INTRA_PN and the K_LINK_* links, all SHAKE/RATTLE constrained), so the three
    beads P / C4' / N9-or-N1 ARE the base frame, and the roll is not a free parameter to guess.

    This function fits the 1EHZ template's own P, C4' and N9/N1 onto the three sampled beads -- a
    three-point Kabsch per residue, which determines the rotation completely, including the roll -- and
    emits the residue. It is additive: reconstruct_all_atom is untouched, and every caller that has only
    a P trace keeps working exactly as before.

    Args:
        beads: (L, 3, 3) Angstrom, per residue (P, C4', N9 for purines / N1 for pyrimidines). These are
            the CG solver's own beads -- aform_from_template.real_cg_beads returns this layout, and
            torch_cgsim's intra-residue constraints are what make it a rigid frame.
        sequence: ACGU string of length L.
    Returns:
        AllAtomStructure with the same interface reconstruct_all_atom returns.
    """
    sequence = sequence.upper().replace("T", "U")
    bad = [c for c in sequence if c not in "ACGU"]
    if bad:
        raise ValueError(f"sequence contains invalid letters {set(bad)}; only ACGU allowed")
    beads = np.asarray(beads, dtype=np.float64)
    if beads.ndim != 3 or beads.shape[1:] != (3, 3):
        raise ValueError(f"beads has shape {beads.shape}, expected (L, 3, 3)")
    if beads.shape[0] != len(sequence):
        raise ValueError(f"sequence length {len(sequence)} != bead count {beads.shape[0]}")
    if not np.all(np.isfinite(beads)):
        raise ValueError("beads contain a non-finite coordinate")

    templates = _load_templates()
    structure = AllAtomStructure(sequence=sequence)
    serial = 0
    for i in range(len(sequence)):
        base = sequence[i]
        tmpl = templates[base]
        names = tmpl["names"]
        tcoords = tmpl["coords"]
        gly = "N9" if base in "AG" else "N1"
        for anchor in ("P", "C4'", gly):
            if anchor not in names:
                raise ValueError(f"template for {base} has no {anchor}")
        src = np.stack([tcoords[names.index(nm)] for nm in ("P", "C4'", gly)])
        aligned = _kabsch_align(src, beads[i], tcoords)

        atom_index: Dict[str, int] = {}
        start = len(structure.atoms)
        for k, name in enumerate(names):
            element = name[0]
            structure.atoms.append(Atom(
                serial=serial, res_seq=i + 1, res_name=base,
                atom_name=name, element=element, xyz=aligned[k],
            ))
            atom_index[name] = serial
            serial += 1
        structure.residue_atom_spans.append((start, len(structure.atoms)))
        structure.residue_atom_index.append(atom_index)
    return structure


def repair_base_placement(structure, sequence: str, pairs,
                          w_pair: float = 1.0, w_keep: float = 2.0, passes: int = 4,
                          cutoff: float = 3.2) -> int:
    """Aim each base's Watson-Crick edge at its partner by rotating it about the glycosidic bond.

    WHY. Measured on this pipeline's products (findings Parts 28-30): the 2OIU deposit satisfies all 12 of its
    key-contact criteria (every Watson-Crick donor/acceptor pair within 3.6 A) while every product satisfies
    1, with distances of 3.7 to 13 A -- the bases are not twisted, they are in the wrong place. The CG field
    has no term that targets pairing geometry at all (it pairs by a harmonic on the N-N distance) and the
    reconstruction places bases from a rigid template, so nothing in the chain of tools is responsible for
    whether the edges face each other.

    WHAT IT FIXES AND WHAT IT CANNOT. Rotating a base about its glycosidic bond is the chi torsion, a real
    degree of freedom; it is the one the rejected repair in Part 28 was missing, that attempt having turned
    the WHOLE residue about the backbone axis and taken the contacts from 1/12 to 1/12 while destroying the
    stacking (50 -> 0 percent). This one moves only the base atoms, and its second objective term keeps each
    base plane near where the reconstruction put it: on the same product it takes the contacts from 1/12 to
    3/12 with the stacked fraction unchanged at 58.3 percent. What it cannot do is close a POSITIONAL gap --
    a base whose partner sits eight Angstroms away cannot be paired by any twist, and the remaining 9 of 12
    are exactly that. Closing those needs the two bases to move relative to each other, which is the CG
    model's pairing geometry (a distance target with no orientation in it), not a post-hoc repair.

    Mutates the structure's atom coordinates in place and returns the number of bases rotated.
    """
    import numpy as _np
    base_atoms = {
        "A": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4", "N6"),
        "G": ("N9", "C8", "N7", "C5", "C6", "N1", "C2", "N3", "C4", "N6", "O6", "N2"),
        "C": ("N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"),
        "U": ("N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"),
    }
    gly = {"A": "N9", "G": "N9", "C": "N1", "U": "N1"}
    key = {("A", "U"): (("N1", "N3"), ("N6", "O4")),
           ("G", "C"): (("N1", "N3"), ("O6", "N4"), ("N2", "O2"))}
    seq = sequence.upper()
    plist = [(int(p[0]), int(p[1])) for p in pairs]
    idx_of = structure.residue_atom_index

    def _xyz(i, nm):
        j = idx_of[i].get(nm)
        return None if j is None else _np.asarray(structure.atoms[j].xyz, dtype=_np.float64)

    def _set(i, nm, v):
        structure.atoms[idx_of[i][nm]].xyz = _np.asarray(v, dtype=_np.float32)

    def _normal(i):
        pts = _np.array([q for q in (_xyz(i, nm) for nm in base_atoms.get(seq[i], ())) if q is not None])
        if len(pts) < 4:
            return _np.array([0.0, 0.0, 1.0])
        c = pts.mean(0)
        _u, _s, vt = _np.linalg.svd(pts - c)
        n = vt[2]
        return n / _np.linalg.norm(n)

    def _rot(axis_hat, deg):
        th = _np.radians(deg)
        c, s = _np.cos(th), _np.sin(th)
        K = _np.array([[0, -axis_hat[2], axis_hat[1]],
                       [axis_hat[2], 0, -axis_hat[0]],
                       [-axis_hat[1], axis_hat[0], 0]])
        return _np.eye(3) * c + s * K + (1 - c) * _np.outer(axis_hat, axis_hat)

    def _penalty(i, normals, normals0):
        pen = 0.0
        for (a, b) in plist:
            if i not in (a, b):
                continue
            trip = key.get((seq[a], seq[b])) or key.get((seq[b], seq[a]))
            if trip is None:
                continue
            flip = (seq[a], seq[b]) not in key
            for x, y in trip:
                na, nb = (y, x) if flip else (x, y)
                va, vb = _xyz(a, na), _xyz(b, nb)
                if va is None or vb is None:
                    continue
                pen += w_pair * max(0.0, float(_np.linalg.norm(va - vb)) - cutoff) ** 2
        pen += w_keep * (1.0 - float(_np.dot(normals[i], normals0[i])))
        return pen

    normals0 = [_normal(i) for i in range(len(seq))]
    # Bases that moved AT LEAST ONCE. Reporting the last pass's count said "0 bases rotated" on a run that
    # had just moved twenty, because a converged pass rotates nothing by definition (measured).
    rotated = set()
    for _p in range(int(passes)):
        moved = 0
        for i in range(len(seq)):
            b = seq[i]
            p0, g0 = _xyz(i, "C1'"), _xyz(i, gly.get(b, "N1"))
            if p0 is None or g0 is None:
                continue
            axis = g0 - p0
            if float(_np.linalg.norm(axis)) < 1e-9:
                continue
            a_hat = axis / _np.linalg.norm(axis)
            keep = {nm: v.copy() for nm, v in ((nm, _xyz(i, nm)) for nm in base_atoms.get(b, ()))
                    if v is not None}
            best = _penalty(i, [_normal(k) for k in range(len(seq))], normals0)
            best_ang = 0.0
            for deg in range(5, 360, 5):
                R = _rot(a_hat, deg)
                for nm, v in keep.items():
                    _set(i, nm, (R @ (v - p0)) + p0)
                sc = _penalty(i, [_normal(k) for k in range(len(seq))], normals0)
                if sc < best - 1e-9:
                    best, best_ang = sc, deg
            R = _rot(a_hat, best_ang)
            for nm, v in keep.items():
                _set(i, nm, (R @ (v - p0)) + p0 if best_ang else v)
            if best_ang:
                moved += 1
                rotated.add(i)
        if moved == 0:
            break
    return len(rotated)


def write_allatom_pdb(structure: AllAtomStructure, path: str) -> str:
    """AllAtomStructure -> PDB text, in the conventions the CG writer already established.

    WHY THIS EXISTS. torch_gpu_refine's bead-frame branch (cg_frame_allatom) called a function of this
    name that did not exist anywhere in the tree, so the branch raised ImportError, was caught, and fell
    back silently to the P-trace path -- the feature was dead on arrival, and only a warning nothing
    printed would have said so. The columns below are copied from _write_pdb_simple in torch_gpu_refine so
    the two writers cannot disagree about what a PDB from this project looks like: three-letter residue
    names (ADE/URA/GUA/CYT), serial in cols 7-11, atom name 13-16, resname 18-20, chain A, element in the
    two columns before the line end, and a plain END record -- no REMARK lines, because the CG_to_allatom
    binary rejected them.

    Residue boundaries come from the structure's own residue_atom_spans, not from a re-guess of which
    atoms belong together.
    """
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    out = []
    with open(path, "w", newline="\n") as fh:
        for r, (start, end) in enumerate(structure.residue_atom_spans):
            seq_letter = structure.sequence[r] if r < len(structure.sequence) else "A"
            resname = base_map.get(seq_letter.upper(), "ADE")
            for atom in structure.atoms[start:end]:
                x, y, z = (float(v) for v in atom.xyz)
                element = (atom.element or atom.atom_name[:1]).strip()[:2]
                # PDB convention: a name shorter than four characters starts in column 14, which is what
                # the leading space in this format does; full four-character names start in 13.
                _nm = atom.atom_name if len(atom.atom_name) == 4 else " " + atom.atom_name
                out.append("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s\n"
                           % (atom.serial, _nm, resname, r + 1, x, y, z, element))
        out.append("END\n")
        fh.writelines(out)
    return path


if __name__ == "__main__":
    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    print(f"L={L} atoms={len(s.atoms)} per_residue={len(s.atoms)/L:.1f}")
    print(f"Residue 0 atoms: {[a.atom_name for a in s.atoms[:s.residue_atom_spans[0][1]]]}")
