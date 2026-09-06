"""p_to_5bead.py — Rebuild an IsRNAcirc-style 5-bead CG representation from P-only CG coordinates.

Input: (L, 3) P-only coordinates
Output: (5L, 3) 5-bead CG coordinates [P, S, B1, B2, B3] per nucleotide

IsRNAcirc-style 5-bead definition:
  P  — phosphate group, taken directly from the input coordinates
  S  — sugar ring center (C4' position)
  B1 — base ring major-groove side (C5'/C6 for pyrimidines, C4/C5 for purines)
  B2 — base ring minor-groove side (C2 for pyrimidines, C2/C3 for purines)
  B3 — base ring center / glycosidic N (N1 for pyrimidine, N9 for purine)

Offsets are based on A-form RNA crystal structures (average of 1EHZ/1M3N).
"""
from __future__ import annotations

import numpy as np

# A-form RNA canonical offsets (Å): P at origin
# Average 5-bead offsets from the 1EHZ tRNA^Phe crystal structure
# P → S (C4' sugar center): offset along the backbone tangent
_OFFSET_P_TO_S = np.array([1.85, 0.60, 0.30], dtype=np.float64)

# S → B3 (glycosidic N): points from the sugar toward the base
_OFFSET_S_TO_B3 = np.array([-0.20, -0.85, 0.65], dtype=np.float64)

# B3 → B1 (major groove): points from N toward the C5/C6 side
_OFFSET_B3_TO_B1 = np.array([0.50, -0.60, 0.30], dtype=np.float64)

# B3 → B2 (minor groove): points from N toward the C2 side
_OFFSET_B3_TO_B2 = np.array([-0.40, 0.50, 0.25], dtype=np.float64)


def _kabsch_rotation(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix mapping unit vector v1 onto unit vector v2.

    Uses Rodrigues rotation: axis = v1 × v2, angle = arccos(v1·v2).
    """
    a = v1 / (np.linalg.norm(v1) + 1e-8)
    b = v2 / (np.linalg.norm(v2) + 1e-8)
    cross = np.cross(a, b)
    dot = np.dot(a, b)

    if abs(dot - 1.0) < 1e-6:
        return np.eye(3, dtype=np.float64)
    if abs(dot + 1.0) < 1e-6:
        perp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(a, perp)) > 0.9:
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = perp / (np.linalg.norm(perp) + 1e-8)
        c = -1.0
        s = 0.0
        cross = axis
    else:
        axis = cross / (np.linalg.norm(cross) + 1e-8)
        c = dot
        s = np.sqrt(1.0 - dot * dot + 1e-8)

    # skew-symmetric matrix [axis]_×
    K = np.zeros((3, 3), dtype=np.float64)
    K[0, 1] = -axis[2]; K[0, 2] = axis[1]
    K[1, 0] = axis[2];  K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] = axis[0]
    return np.eye(3) + K * s + np.outer(axis, axis) * (1.0 - c)


def p_to_5bead(p_coords: np.ndarray) -> np.ndarray:
    """P-only CG → 5-bead CG (IsRNAcirc format).

    Args:
        p_coords: (L, 3) P atom coordinates (Å)

    Returns:
        (5L, 3) coordinates in order [P_0, S_0, B1_0, B2_0, B3_0, P_1, S_1, ...]
    """
    L = len(p_coords)
    coords_5bead = np.zeros((5 * L, 3), dtype=np.float64)

    for i in range(L):
        p = p_coords[i]
        coords_5bead[5 * i] = p  # P bead

        # backbone tangent (5'→3' direction)
        if i < L - 1:
            tangent = p_coords[i + 1] - p
        else:
            tangent = p - p_coords[i - 1]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-8:
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            tangent = tangent / tangent_norm

        # reference direction for rotation
        ref_dir = np.array([1.0, 0.0, 0.0])
        R = _kabsch_rotation(ref_dir, tangent)

        # S (sugar/C4') bead: P → S
        s = p + R @ _OFFSET_P_TO_S
        coords_5bead[5 * i + 1] = s

        # B3 (glycosidic N) bead: S → B3 (sequential offset)
        b3 = s + R @ _OFFSET_S_TO_B3
        coords_5bead[5 * i + 4] = b3

        # B1 (major groove) bead: B3 → B1 (sequential offset)
        b1 = b3 + R @ _OFFSET_B3_TO_B1
        coords_5bead[5 * i + 2] = b1

        # B2 (minor groove) bead: B3 → B2 (sequential offset)
        b2 = b3 + R @ _OFFSET_B3_TO_B2
        coords_5bead[5 * i + 3] = b2

    return coords_5bead


def split_5bead_coords(coords_5bead: np.ndarray):
    """Split (5L, 3) 5-bead coords into 5 × (L, 3) arrays.

    Returns:
        (P_coords, S_coords, B1_coords, B2_coords, B3_coords)
    """
    L = len(coords_5bead) // 5
    P = coords_5bead[0::5].copy()
    S = coords_5bead[1::5].copy()
    B1 = coords_5bead[2::5].copy()
    B2 = coords_5bead[3::5].copy()
    B3 = coords_5bead[4::5].copy()
    return P, S, B1, B2, B3
