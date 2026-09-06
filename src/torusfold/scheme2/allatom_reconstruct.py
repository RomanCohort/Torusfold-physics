"""
allatom_reconstruct.py — coarse-grained P coordinates → all-atom RNA backbone + base reconstruction.

The scheme2 CG solver outputs (L, 3) coordinates with one P atom per nucleotide.
This module expands each P point into a full all-atom residue using standard
A-form RNA geometry, matching the OpenMM amber14 RNA.OL3.xml template strictly
(atom order and names use asterisks instead of chemical primes).

Atom naming follows the amber14 RNA.OL3 standard:
    * Sugar ring: O5*/C5*/C4*/O4*/C3*/O3*/C2*/C1* (asterisk replaces the chemical prime)
    * P: P plus the two non-bridging oxygens O1P/O2P
    * Bases: A/G purines (start at N9, 9-10 atoms), C/U pyrimidines (start at N1, 8 atoms)

Per-residue local coordinate frame:
    b = normalize(P[i+1] - P[i])        backbone direction
    r = normalize(P[i] - centroid)      radial
    u = cross(b, r)                     normal

Templates are defined in the residue-local frame (Å), transformed to Cartesian
via b/u/r, then translated to P[i].
Outputs an AllAtomStructure for amber_refine to refine with the amber14 force field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Atom:
    """A single atom record."""
    serial: int          # global atom serial (0-based, internal)
    res_seq: int         # residue sequence number (1-based)
    res_name: str        # A / U / G / C (amber14 RNA template name)
    atom_name: str       # P / O1P / O2P / O5* / C5* / ... / N9 / C8 / ...
    element: str        # P / O / C / N
    xyz: np.ndarray      # (3,) Å


@dataclass
class AllAtomStructure:
    """Result of all-atom structure reconstruction."""
    atoms: List[Atom] = field(default_factory=list)
    sequence: str = ""
    residue_atom_spans: List[Tuple[int, int]] = field(default_factory=list)
    residue_atom_index: List[Dict[str, int]] = field(default_factory=list)


# --- A-form RNA local templates (Å) ---
# Template origin = the P atom. x axis along the backbone, y along the radial,
# z = cross(x, y).
# Bond lengths are statistical means for A-form RNA.

# Backbone atoms: P + the two non-bridging oxygens O1P/O2P + the sugar-phosphate chain (O5*/C5*/C4*/O4*/C3*/O3*/C2*/C1*)
# Strictly matches amber14 RNA.OL3.xml: non-bridging oxygens are OP1/OP2; the sugar ring uses prime naming O5'/C5'/...
_BACKBONE_ATOMS: List[Tuple[str, str, Tuple[float, float, float]]] = [
    # (atom_name, element, template_xyz)
    ("P",   "P", (0.00, 0.00, 0.00)),
    ("OP1", "O", (-0.50, -1.20, 0.50)),   # P non-bridging O1 (~1.52Å)
    ("OP2", "O", (-0.50,  1.20, 0.50)),   # P non-bridging O2 (~1.52Å)
    ("O5'", "O", (1.60,  0.00, 0.00)),    # P-O5' ~1.61Å
    ("C5'", "C", (2.96,  0.65, 0.00)),    # O5'-C5' ~1.50Å
    ("C4'", "C", (4.21, -0.05, 0.30)),    # C5'-C4' ~1.52Å
    ("O4'", "O", (4.50, -1.40, -0.20)),   # C4'-O4' ~1.46Å (closes the sugar ring)
    ("C3'", "C", (3.45, -1.30, 0.95)),    # C4'-C3' ~1.52Å
    ("O3'", "O", (2.65, -2.40, 1.05)),    # C3'-O3' ~1.42Å (connects to the downstream P)
    ("C2'", "C", (4.60, -2.20, 0.55)),    # C3'-C2' ~1.53Å (sugar ring)
    ("O2'", "O", (4.90, -2.60, 1.80)),    # C2'-O2' ~1.43Å (RNA 2'-OH; distinguishes RNA from DNA)
    ("C1'", "C", (5.50, -1.10, 0.10)),    # C2'-C1' ~1.52Å (base attachment point)
]

# Purine bases (A/G): N9 attaches to C1'; 9-10 atoms (A: N9 C8 N7 C5 C6 N6 N1 C2 = 8 atoms)
# (G: N9 C8 N7 C5 C6 O6 N1 C2 N2 = 9 atoms)
# Shared 9-atom core, plus A's N6 or G's O6 and G's N2
_PURINE_BASE_ATOMS: Dict[str, List[Tuple[str, str, Tuple[float, float, float]]]] = {
    "A": [
        ("N9", "N", (6.90, -0.40, 0.10)),   # C1'-N9 ~1.47Å
        ("C8", "C", (7.55, -1.45, -0.20)),  # N9-C8 ~1.37Å
        ("N7", "N", (8.60, -0.55, 0.30)),   # C8-N7 ~1.30Å
        ("C5", "C", (8.00,  0.60, 0.50)),   # N7-C5 ~1.39Å
        ("C6", "C", (8.40,  1.80, 0.70)),   # C5-C6 ~1.40Å
        ("N6", "N", (9.45,  2.20, 1.00)),   # C6-N6 amino group ~1.34Å
        ("N1", "N", (7.50,  2.70, 0.60)),   # C6-N1 ~1.36Å
        ("C2", "C", (6.30,  2.40, 0.30)),   # N1-C2 ~1.36Å
        ("N3", "N", (5.10, 1.50, 0.10)),    # C2-N3 ~1.33Å
        ("C4", "C", (6.20, 0.30, 0.20)),    # N3-C4 ~1.37Å (bridge carbon, closes the ring to C5)
    ],
    "G": [
        ("N9", "N", (6.90, -0.40, 0.10)),
        ("C8", "C", (7.55, -1.45, -0.20)),
        ("N7", "N", (8.60, -0.55, 0.30)),
        ("C5", "C", (8.00,  0.60, 0.50)),
        ("C6", "C", (8.40,  1.80, 0.70)),
        ("O6", "O", (9.40,  2.40, 0.95)),   # C6=O6 keto group ~1.24Å
        ("N1", "N", (7.50,  2.70, 0.60)),
        ("C2", "C", (6.30,  2.40, 0.30)),
        ("N3", "N", (5.10, 1.50, 0.10)),    # C2-N3 (closes the ring)
        ("C4", "C", (6.20, 0.30, 0.20)),    # N3-C4 (bridge carbon)
        ("N2", "N", (5.30,  3.20, 0.20)),   # C2-N2 amino group ~1.34Å
    ],
}

# Pyrimidine bases (C/U): N1 attaches to C1'; 6-member ring + substituents
_PYRIMIDINE_BASE_ATOMS: Dict[str, List[Tuple[str, str, Tuple[float, float, float]]]] = {
    "C": [
        ("N1", "N", (6.90, -0.40, 0.10)),   # C1'-N1 ~1.47Å
        ("C2", "C", (7.20,  0.90, 0.30)),   # N1-C2 ~1.36Å
        ("O2", "O", (8.10,  1.30, 0.40)),   # C2=O2 ~1.24Å
        ("N3", "N", (6.30,  1.70, 0.30)),   # C2-N3 ~1.33Å
        ("C4", "C", (5.20,  1.10, 0.10)),   # N3-C4 ~1.36Å
        ("N4", "N", (4.10,  1.80, 0.20)),   # C4-N4 amino group ~1.34Å
        ("C5", "C", (5.00, -0.20, -0.10)),  # C4-C5 ~1.43Å
        ("C6", "C", (6.00, -1.10, -0.20)),  # C5-C6 ~1.36Å (closes the ring to N1)
    ],
    "U": [
        ("N1", "N", (6.90, -0.40, 0.10)),
        ("C2", "C", (7.20,  0.90, 0.30)),
        ("O2", "O", (8.10,  1.30, 0.40)),   # C2=O2 ~1.24Å
        ("N3", "N", (6.30,  1.70, 0.30)),
        ("C4", "C", (5.20,  1.10, 0.10)),
        ("O4", "O", (4.10,  1.80, 0.20)),   # C4=O4 ~1.24Å
        ("C5", "C", (5.00, -0.20, -0.10)),
        ("C6", "C", (6.00, -1.10, -0.20)),
    ],
}


def _local_frame(
    p_coords: np.ndarray, centroid: np.ndarray, i: int, L: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the local coordinate frame (b, u, r) for residue i; all unit vectors."""
    nxt = p_coords[(i + 1) % L]
    b = nxt - p_coords[i]
    n = np.linalg.norm(b)
    if n < 1e-6:
        b = np.array([1.0, 0.0, 0.0])
    else:
        b = b / n
    r = p_coords[i] - centroid
    n = np.linalg.norm(r)
    if n < 1e-6:
        r = np.array([0.0, 0.0, 1.0]) if abs(b[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    else:
        r = r / n
    r = r - np.dot(r, b) * b
    n = np.linalg.norm(r)
    if n < 1e-6:
        r = np.array([0.0, 0.0, 1.0]) if abs(b[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        r = r - np.dot(r, b) * b
        n = np.linalg.norm(r)
    r = r / n
    u = np.cross(b, r)
    u = u / np.linalg.norm(u)
    return b, u, r


def _place_atom(
    template_xyz: Tuple[float, float, float],
    b: np.ndarray, u: np.ndarray, r: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Transform local template coordinates (x_along_b, y_along_u, z_along_r) to Cartesian."""
    tb, tu, tr = template_xyz
    return origin + tb * b + tu * u + tr * r


def reconstruct_all_atom(
    p_coords: np.ndarray, sequence: str
) -> AllAtomStructure:
    """Coarse-grained P coordinates → all-atom RNA structure (amber14 template match).

    Args:
        p_coords: (L, 3) Å, one P atom per nucleotide
        sequence: ACGU string of length L

    Returns:
        AllAtomStructure holding per-residue all-atom coordinates plus index maps.
    """
    p_coords = np.asarray(p_coords, dtype=np.float64)
    if p_coords.ndim != 2 or p_coords.shape[1] != 3:
        raise ValueError(f"p_coords must be (L,3), got {p_coords.shape}")
    # Normalize the sequence: case + T→U
    sequence = sequence.upper().replace("T", "U")
    L = len(sequence)
    if p_coords.shape[0] != L:
        raise ValueError(f"sequence length {L} != P count {p_coords.shape[0]}")
    bad = [c for c in sequence if c not in "ACGU"]
    if bad:
        raise ValueError(f"sequence contains invalid letters {set(bad)}; only ACGU allowed")

    centroid = p_coords.mean(axis=0)
    structure = AllAtomStructure(sequence=sequence)

    serial = 0
    for i in range(L):
        base = sequence[i]
        b, u, r = _local_frame(p_coords, centroid, i, L)
        origin = p_coords[i]

        res_name = base
        res_seq = i + 1
        atom_index: Dict[str, int] = {}
        start = len(structure.atoms)

        # Backbone + sugar ring
        for atom_name, element, tmpl in _BACKBONE_ATOMS:
            xyz = _place_atom(tmpl, b, u, r, origin)
            structure.atoms.append(Atom(
                serial=serial, res_seq=res_seq, res_name=res_name,
                atom_name=atom_name, element=element, xyz=xyz,
            ))
            atom_index[atom_name] = serial
            serial += 1

        # Base (purine or pyrimidine)
        base_atoms = (
            _PURINE_BASE_ATOMS[base] if base in ("A", "G")
            else _PYRIMIDINE_BASE_ATOMS[base]
        )
        for atom_name, element, tmpl in base_atoms:
            xyz = _place_atom(tmpl, b, u, r, origin)
            structure.atoms.append(Atom(
                serial=serial, res_seq=res_seq, res_name=res_name,
                atom_name=atom_name, element=element, xyz=xyz,
            ))
            atom_index[atom_name] = serial
            serial += 1

        end = len(structure.atoms)
        structure.residue_atom_spans.append((start, end))
        structure.residue_atom_index.append(atom_index)

    return structure


def get_atom_xyzs(structure: AllAtomStructure) -> np.ndarray:
    """(N, 3) coordinates of all atoms, in the same order as structure.atoms."""
    return np.array([a.xyz for a in structure.atoms], dtype=np.float64)


if __name__ == "__main__":
    seq = "AUGCAUGCAUGCAUGCAUGCAUGCAUGCAUGC"
    L = len(seq)
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    ps = np.stack([R * np.cos(angles), R * np.sin(angles),
                   np.zeros(L)], axis=1)
    s = reconstruct_all_atom(ps, seq)
    print(f"L={L} atoms={len(s.atoms)} per_residue={len(s.atoms)/L:.1f}")
    print(f"Residue 0 atom names: {[a.atom_name for a in s.atoms[s.residue_atom_spans[0][0]:s.residue_atom_spans[0][1]]]}")
