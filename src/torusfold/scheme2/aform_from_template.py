"""aform_from_template.py - 从 1EHZ tRNA 晶体标准残基重建全原子 RNA。

替代 allatom_reconstruct.py 的手算模板。旧版手算几何 (P-O5'=1.6Å 等) 与
amber14 OL3 力场平衡值有偏差, 最小化后 amber_field 恒正 (+7 万 kJ/mol,
学长指出不合理)。改用 1EHZ (酵母 tRNA^Phe, 1.93Å 高分辨率晶体) 的真实
实验坐标做模板:

  1. 从 aform_template.npz 取 A/U/G/C 四种标准残基坐标 (真实晶体)
  2. 对 CG 的每个 P 点, 取对应碱基的标准残基
  3. 用 P + C1' + C4' 三点 Kabsch 对齐, 把标准残基叠加到 CG 局部坐标
  4. 残基内坐标 = 真实晶体几何, amber 力场能量从负开始

BSJ 闭合: circRNA 首末残基的 O3'/P 靠 amber_refine 的 HarmonicBondForce
约束 (Kabsch 叠加后首末残基 O3'-P 距离接近真实 ~1.6Å, 力场微调即闭合)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# 复用旧 AllAtomStructure 接口, predictor 不用改
from .allatom_reconstruct import AllAtomStructure, Atom


_TEMPLATE_PATH = Path(__file__).parent / "aform_template.npz"
_templates: Dict[str, Dict] = {}


def _load_templates() -> Dict[str, Dict]:
    """惰性加载 1EHZ 标准残基模板 (A/U/G/C)。"""
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
    """三点 Kabsch: 把 src_all 变换到使 src_three -> dst_three 的坐标系。

    Args:
        src_three: (3, 3) 模板的三个锚点 (P, C1', C4') 行向量
        dst_three: (3, 3) 目标的三个锚点 (CG 推出的 P, C1', C4')
        src_all:   (N, 3) 模板全部原子坐标
    Returns:
        (N, 3) 变换后的坐标 (平移+旋转, 仿射对齐)
    """
    # 中心化
    src_c = src_three.mean(axis=0)
    dst_c = dst_three.mean(axis=0)
    s = src_three - src_c
    d = dst_three - dst_c
    # Kabsch: R = argmin ||R @ s - d||, 用 SVD
    H = s.T @ d
    U, _, Vt = np.linalg.svd(H)
    # 反射修正: 防止 R 含镜像 (det=-1), 用变量名 refl 避免与上面的 d 撞名
    refl = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, refl])
    R = Vt.T @ D @ U.T
    # 变换: 先中心化模板到原点, 旋转, 平移到目标中心
    aligned = (src_all - src_c) @ R.T + dst_c
    return aligned.astype(np.float32)


def reconstruct_all_atom(
    p_coords: np.ndarray, sequence: str,
) -> AllAtomStructure:
    """CG P 坐标 -> 全原子 RNA (1EHZ 晶体模板)。

    Args:
        p_coords: (L, 3) Å, 每核苷酸一个 P 原子 (CG 求解输出)
        sequence: ACGU 字符串, 长度 L
    Returns:
        AllAtomStructure, 每残基全原子坐标 = 1EHZ 标准残基 Kabsch 叠加。
    """
    # 序列标准化: 大小写 + T→U
    sequence = sequence.upper().replace("T", "U")
    if p_coords.ndim != 2 or p_coords.shape[1] != 3:
        raise ValueError(f"p_coords 形状异常 {p_coords.shape}, 期望 (L,3)")
    L = len(sequence)
    if p_coords.shape[0] != L:
        raise ValueError(f"sequence 长度 {L} != P 点数 {p_coords.shape[0]}")
    bad = [c for c in sequence if c not in "ACGU"]
    if bad:
        raise ValueError(f"sequence 含非法字母 {set(bad)}, 只允许 ACGU")

    templates = _load_templates()
    centroid = p_coords.mean(axis=0)
    structure = AllAtomStructure(sequence=sequence)

    serial = 0
    for i in range(L):
        base = sequence[i]
        tmpl = templates[base]
        names = tmpl["names"]
        tcoords = tmpl["coords"]  # (N, 3) 模板坐标

        # 找模板的 P / C1' / C4' / O3' 四个锚点
        # P1 修: 加 O3' 锚点 (磷酸桥几何), 让重建时 O3' 位置跟相邻残基 P 协调,
        #        避免 O3' 被模板继承时跟下一残基 P 压成灾难性几何 (C3'-O3'-P 偏 70°)
        idx_P = names.index("P")
        idx_C1 = names.index("C1'")
        idx_C4 = names.index("C4'")
        idx_O3 = names.index("O3'")
        src_anchors = np.stack([tcoords[idx_P], tcoords[idx_C1],
                                tcoords[idx_C4], tcoords[idx_O3]])

        # CG 只给 P[i], C1'/C4'/O3' 的目标位置用局部坐标系推 (近似 A-form 几何):
        #   backbone 方向 b = P[i+1] - P[i] (末位用 P[0]-P[L-1])
        #   径向 r = P[i] - centroid (碱基朝外)
        #   C1' 在 P 沿 backbone 方向 +5.5Å、径向 +1.5Å 处 (A-form 统计)
        #   C4' 在 P 沿 backbone +4.2Å、径向 0 处
        #   O3' 在 P[i+1] 反推 -1.6Å (A-form O3'-P 键长 1.6Å)
        nxt = p_coords[(i + 1) % L]
        b = nxt - p_coords[i]
        bn = np.linalg.norm(b)
        b = b / bn if bn > 1e-6 else np.array([1.0, 0.0, 0.0])
        r = p_coords[i] - centroid
        rn = np.linalg.norm(r)
        r = r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0])
        r = r - np.dot(r, b) * b  # 正交到 b 法平面
        rn = np.linalg.norm(r)
        r = r / rn if rn > 1e-6 else np.array([0.0, 0.0, 1.0])

        c1_dst = p_coords[i] + b * 5.5 + r * 1.5
        c4_dst = p_coords[i] + b * 4.2
        o3_dst = nxt - b * 1.6  # O3'[i] 跟 P[i+1] 几何协调
        dst_anchors = np.stack([p_coords[i], c1_dst, c4_dst, o3_dst])

        # Kabsch 叠加 (4 点最小二乘, 比 3 点多一个 O3' 约束)
        aligned = _kabsch_align(src_anchors, dst_anchors, tcoords)

        # 写入 structure
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
    print(f"残基0 原子: {[a.atom_name for a in s.atoms[:s.residue_atom_spans[0][1]]]}")
