"""p_to_5bead.py — 从 P-only CG 坐标重建 IsRNAcirc 式 5-bead CG 表示。

输入: (L, 3) P-only 坐标
输出: (5L, 3) 5-bead CG 坐标 [P, S, B1, B2, B3] per nucleotide

IsRNAcirc 式 5-bead 定义:
  P  — 磷酸基 (phosphate), 直接用输入坐标
  S  — sugar ring 中心 (C4' 位置)
  B1 — base ring major groove 侧 (C5'/C6 对于嘧啶, C4/C5 对于嘌呤)
  B2 — base ring minor groove 侧 (C2 对于嘧啶, C2/C3 对于嘌呤)
  B3 — base ring中心 / glycosidic N (N1 for pyrimidine, N9 for purine)

偏移基于 A-form RNA 晶体结构 (1EHZ/1M3N 平均值)。
"""
from __future__ import annotations

import numpy as np

# A-form RNA canonical offsets (Å): P at origin
# 基于 1EHZ tRNA^Phe 晶体结构的 5-bead 平均偏移
# P → S (C4' sugar center): 沿骨架切线偏移
_OFFSET_P_TO_S = np.array([1.85, 0.60, 0.30], dtype=np.float64)

# S → B3 (glycosidic N): 从 sugar 指向 base
_OFFSET_S_TO_B3 = np.array([-0.20, -0.85, 0.65], dtype=np.float64)

# B3 → B1 (major groove): 从 N 指向 C5/C6 侧
_OFFSET_B3_TO_B1 = np.array([0.50, -0.60, 0.30], dtype=np.float64)

# B3 → B2 (minor groove): 从 N 指向 C2 侧
_OFFSET_B3_TO_B2 = np.array([-0.40, 0.50, 0.25], dtype=np.float64)


def _kabsch_rotation(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """返回把 unit vector v1 旋转到 unit vector v2 方向的 3x3 旋转矩阵。

    用 Rodrigues 旋转: axis = v1 × v2, angle = arccos(v1·v2)。
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
    """P-only CG → 5-bead CG (IsRNAcirc 格式)。

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

        # S (sugar/C4') bead
        s_offset = R @ _OFFSET_P_TO_S
        s = p + s_offset
        coords_5bead[5 * i + 1] = s

        # B3 (glycosidic N) bead
        b3_offset = R @ (_OFFSET_P_TO_S + _OFFSET_S_TO_B3)
        b3 = p + b3_offset
        coords_5bead[5 * i + 4] = b3

        # B1 (major groove) bead
        b1_offset = R @ (_OFFSET_P_TO_S + _OFFSET_S_TO_B3 + _OFFSET_B3_TO_B1)
        b1 = p + b1_offset
        coords_5bead[5 * i + 2] = b1

        # B2 (minor groove) bead
        b2_offset = R @ (_OFFSET_P_TO_S + _OFFSET_S_TO_B3 + _OFFSET_B3_TO_B2)
        b2 = p + b2_offset
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
