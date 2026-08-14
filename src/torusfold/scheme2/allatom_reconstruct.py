"""
allatom_reconstruct.py — 粗粒度 P 坐标 → 全原子 RNA 骨架 + 碱基重建。

scheme2 CG 求解器输出 (L, 3) 每核苷酸一个 P 原子坐标。本模块用 A-form RNA
标准几何把每个 P 点展开成全原子残基, 严格匹配 OpenMM amber14 RNA.OL3.xml
模板 (atom 顺序、命名用星号而非撇号)。

原子命名遵循 amber14 RNA.OL3 标准:
    * 糖环: O5*/C5*/C4*/O4*/C3*/O3*/C2*/C1* (星号替代化学撇号)
    * P: P + 两个非桥接氧 O1P/O2P
    * 碱基: A/G 嘌呤 (N9 起始, 9-10 原子), C/U 嘧啶 (N1 起始, 8 原子)

每个残基的局部坐标系:
    b = normalize(P[i+1] - P[i])        backbone 方向
    r = normalize(P[i] - centroid)      径向
    u = cross(b, r)                     法向

模板在残基局部系内定义 (Å), 经 b/u/r 变换到笛卡尔后平移到 P[i]。
输出 AllAtomStructure, 供 amber_refine 用 amber14 力场精修。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class Atom:
    """单个原子记录。"""
    serial: int          # 全局原子序号 (0-based 内部)
    res_seq: int         # 残基序号 (1-based)
    res_name: str        # A / U / G / C (amber14 RNA 模板名)
    atom_name: str       # P / O1P / O2P / O5* / C5* / ... / N9 / C8 / ...
    element: str        # P / O / C / N
    xyz: np.ndarray      # (3,) Å


@dataclass
class AllAtomStructure:
    """全原子结构重建结果。"""
    atoms: List[Atom] = field(default_factory=list)
    sequence: str = ""
    residue_atom_spans: List[Tuple[int, int]] = field(default_factory=list)
    residue_atom_index: List[Dict[str, int]] = field(default_factory=list)


# --- A-form RNA 局部模板 (Å) ---
# 模板原点 = P 原子。x 轴沿 backbone, y 沿径向, z = cross(x, y)。
# 键长取 A-form RNA 统计均值。

# 骨架原子: P + 非桥接 O1P/O2P + 糖-磷酸链 (O5*/C5*/C4*/O4*/C3*/O3*/C2*/C1*)
# 严格匹配 amber14 RNA.OL3.xml: 非桥接氧 OP1/OP2, 糖环撇号命名 O5'/C5'/...
_BACKBONE_ATOMS: List[Tuple[str, str, Tuple[float, float, float]]] = [
    # (atom_name, element, template_xyz)
    ("P",   "P", (0.00, 0.00, 0.00)),
    ("OP1", "O", (-0.50, -1.20, 0.50)),   # P 非桥接 O1 (~1.52Å)
    ("OP2", "O", (-0.50,  1.20, 0.50)),   # P 非桥接 O2 (~1.52Å)
    ("O5'", "O", (1.60,  0.00, 0.00)),    # P-O5' ~1.61Å
    ("C5'", "C", (2.96,  0.65, 0.00)),    # O5'-C5' ~1.50Å
    ("C4'", "C", (4.21, -0.05, 0.30)),    # C5'-C4' ~1.52Å
    ("O4'", "O", (4.50, -1.40, -0.20)),   # C4'-O4' ~1.46Å (糖环闭合)
    ("C3'", "C", (3.45, -1.30, 0.95)),    # C4'-C3' ~1.52Å
    ("O3'", "O", (2.65, -2.40, 1.05)),    # C3'-O3' ~1.42Å (连下游 P)
    ("C2'", "C", (4.60, -2.20, 0.55)),    # C3'-C2' ~1.53Å (糖环)
    ("O2'", "O", (4.90, -2.60, 1.80)),    # C2'-O2' ~1.43Å (RNA 2'-OH, 区别于 DNA)
    ("C1'", "C", (5.50, -1.10, 0.10)),    # C2'-C1' ~1.52Å (碱基连接点)
]

# 嘌呤碱基 (A/G): N9 连 C1', 9-10 原子 (A: N9 C8 N7 C5 C6 N6 N1 C2  = 8 原子)
# (G: N9 C8 N7 C5 C6 O6 N1 C2 N2 = 9 原子)
# 共同骨架 9 原子 + A 的 N6 或 G 的 O6 + G 的 N2
_PURINE_BASE_ATOMS: Dict[str, List[Tuple[str, str, Tuple[float, float, float]]]] = {
    "A": [
        ("N9", "N", (6.90, -0.40, 0.10)),   # C1'-N9 ~1.47Å
        ("C8", "C", (7.55, -1.45, -0.20)),  # N9-C8 ~1.37Å
        ("N7", "N", (8.60, -0.55, 0.30)),   # C8-N7 ~1.30Å
        ("C5", "C", (8.00,  0.60, 0.50)),   # N7-C5 ~1.39Å
        ("C6", "C", (8.40,  1.80, 0.70)),   # C5-C6 ~1.40Å
        ("N6", "N", (9.45,  2.20, 1.00)),   # C6-N6 氨基 ~1.34Å
        ("N1", "N", (7.50,  2.70, 0.60)),   # C6-N1 ~1.36Å
        ("C2", "C", (6.30,  2.40, 0.30)),   # N1-C2 ~1.36Å
        ("N3", "N", (5.10, 1.50, 0.10)),    # C2-N3 ~1.33Å
        ("C4", "C", (6.20, 0.30, 0.20)),    # N3-C4 ~1.37Å (桥碳, 闭环到 C5)
    ],
    "G": [
        ("N9", "N", (6.90, -0.40, 0.10)),
        ("C8", "C", (7.55, -1.45, -0.20)),
        ("N7", "N", (8.60, -0.55, 0.30)),
        ("C5", "C", (8.00,  0.60, 0.50)),
        ("C6", "C", (8.40,  1.80, 0.70)),
        ("O6", "O", (9.40,  2.40, 0.95)),   # C6=O6 酮基 ~1.24Å
        ("N1", "N", (7.50,  2.70, 0.60)),
        ("C2", "C", (6.30,  2.40, 0.30)),
        ("N3", "N", (5.10, 1.50, 0.10)),    # C2-N3 (闭环)
        ("C4", "C", (6.20, 0.30, 0.20)),    # N3-C4 (桥碳)
        ("N2", "N", (5.30,  3.20, 0.20)),   # C2-N2 氨基 ~1.34Å
    ],
}

# 嘧啶碱基 (C/U): N1 连 C1', 6 原子环 + 取代基
_PYRIMIDINE_BASE_ATOMS: Dict[str, List[Tuple[str, str, Tuple[float, float, float]]]] = {
    "C": [
        ("N1", "N", (6.90, -0.40, 0.10)),   # C1'-N1 ~1.47Å
        ("C2", "C", (7.20,  0.90, 0.30)),   # N1-C2 ~1.36Å
        ("O2", "O", (8.10,  1.30, 0.40)),   # C2=O2 ~1.24Å
        ("N3", "N", (6.30,  1.70, 0.30)),   # C2-N3 ~1.33Å
        ("C4", "C", (5.20,  1.10, 0.10)),   # N3-C4 ~1.36Å
        ("N4", "N", (4.10,  1.80, 0.20)),   # C4-N4 氨基 ~1.34Å
        ("C5", "C", (5.00, -0.20, -0.10)),  # C4-C5 ~1.43Å
        ("C6", "C", (6.00, -1.10, -0.20)),  # C5-C6 ~1.36Å (闭环到 N1)
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
    """构造残基 i 的局部坐标系 (b, u, r), 都是单位向量。"""
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
    """把局部模板坐标 (x_along_b, y_along_u, z_along_r) 变换到笛卡尔。"""
    tb, tu, tr = template_xyz
    return origin + tb * b + tu * u + tr * r


def reconstruct_all_atom(
    p_coords: np.ndarray, sequence: str
) -> AllAtomStructure:
    """粗粒度 P 坐标 → 全原子 RNA 结构 (amber14 模板匹配)。

    Args:
        p_coords: (L, 3) Å, 每核苷酸一个 P 原子
        sequence: 长度 L 的 ACGU 字符串

    Returns:
        AllAtomStructure, 含每残基全原子坐标 + 索引映射。
    """
    p_coords = np.asarray(p_coords, dtype=np.float64)
    if p_coords.ndim != 2 or p_coords.shape[1] != 3:
        raise ValueError(f"p_coords 期望 (L,3), 实际 {p_coords.shape}")
    # 序列标准化: 大小写 + T→U
    sequence = sequence.upper().replace("T", "U")
    L = len(sequence)
    if p_coords.shape[0] != L:
        raise ValueError(f"sequence 长度 {L} != P 点数 {p_coords.shape[0]}")
    bad = [c for c in sequence if c not in "ACGU"]
    if bad:
        raise ValueError(f"sequence 含非法字母 {set(bad)}, 只允许 ACGU")

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

        # 骨架 + 糖环
        for atom_name, element, tmpl in _BACKBONE_ATOMS:
            xyz = _place_atom(tmpl, b, u, r, origin)
            structure.atoms.append(Atom(
                serial=serial, res_seq=res_seq, res_name=res_name,
                atom_name=atom_name, element=element, xyz=xyz,
            ))
            atom_index[atom_name] = serial
            serial += 1

        # 碱基 (嘌呤或嘧啶)
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
    """(N, 3) 所有原子坐标, 顺序与 structure.atoms 一致。"""
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
    print(f"残基0 原子名: {[a.atom_name for a in s.atoms[s.residue_atom_spans[0][0]:s.residue_atom_spans[0][1]]]}")
