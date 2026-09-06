"""
boundary_constraints.py — Level 1 边界约束注入

从全局配对矩阵 (bpp) 提取跨 segment 边界的配对信息,
作为 RhoFold+ 预测的硬约束, 解决分段预测的边界拓扑断裂问题.

公开 API:
  build_boundary_pairs()  — 构建每 segment 的边界约束对
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ── 常量 ──
BOUNDARY_MARGIN = 50   # 边界窗口: 前后 50nt 内的配对视为边界约束
WC_DIST_TARGET = 20.0  # Watson-Crick C1'-C1' 目标距离 (Å)
EDGE_TYPES = ("wc", "non_wc")  # wc=Watson-Crick, non_wc=非 WC 配对


def build_boundary_pairs(
    global_bpp: np.ndarray,
    segments: List[Dict],
    seq_len: int,
    bpp_threshold: float = 0.3,
    boundary_margin: int = BOUNDARY_MARGIN,
) -> List[List[Tuple[int, int, str]]]:
    """从全局 bpp 矩阵提取每个 segment 的边界约束配对.

    边界定义: segment 前后 boundary_margin 个 nt 内、跨越 segment 边界的配对.
    即: pair 的一个残基在当前 segment 的边界窗口内, 另一个残基在相邻 segment 中.

    Args:
        global_bpp: (L, L) 全局配对概率矩阵 (对称, 上三角)
        segments: split_sequence() 输出的 segment 列表
        seq_len: 完整序列长度
        bpp_threshold: 配对概率阈值, 低于此值的 pair 不纳入约束
        boundary_margin: 边界窗口大小 (nt)

    Returns:
        长度 == len(segments), 每个元素是该 segment 的边界约束对列表.
        每个约束对: (global_i, global_j, edge_type)
        edge_type: "wc" (Watson-Crick) 或 "non_wc" (非 WC)
    """
    L = seq_len
    n_seg = len(segments)
    all_boundary_pairs: List[List[Tuple[int, int, str]]] = [[] for _ in range(n_seg)]

    if global_bpp.shape != (L, L):
        raise ValueError(
            f"global_bpp shape {global_bpp.shape} != ({L}, {L})"
        )

    for idx, seg in enumerate(segments):
        seg_start = seg["start"]
        seg_end = seg["end"]  # exclusive

        # 边界窗口: segment 两端各 boundary_margin nt
        window_start = max(0, seg_start - boundary_margin)
        window_end = min(L, seg_end + boundary_margin)

        # 遍历窗口内的配对
        seen = set()
        for i in range(window_start, window_end):
            for j in range(i + 1, window_end):
                if j >= L:
                    break
                prob = global_bpp[i, j]
                if prob < bpp_threshold:
                    continue

                # 检查是否跨越 segment 边界:
                # pair 的一个在 segment 内, 另一个在 segment 外
                i_in_seg = seg_start <= i < seg_end
                j_in_seg = seg_start <= j < seg_end

                crosses = (i_in_seg and not j_in_seg) or (j_in_seg and not i_in_seg)
                if not crosses:
                    continue

                key = (i, j)
                if key in seen:
                    continue
                seen.add(key)

                # 判断 edge type
                edge_type = _classify_pair_type(i, j, L)
                all_boundary_pairs[idx].append((i, j, edge_type))

    return all_boundary_pairs


def _classify_pair_type(i: int, j: int, seq_len: int) -> str:
    """简单判断配对类型 (用于约束分类).

    这里用距离启发式判断: WC 配对在理想构象中 C1'-C1' ~10.5Å,
    non_wc 更远. 但 bpp 矩阵不直接区分, 所以统一标记为 "wc"
    (RhoFold+ 的距离约束对两类都适用, 只是目标距离不同).

    TODO: 后续可用 ViennaRNA 的 pair type 区分
    """
    return "wc"


def apply_boundary_constraints_to_coords(
    coords: np.ndarray,
    boundary_pairs: List[Tuple[int, int, str]],
    n_steps: int = 2000,
) -> np.ndarray:
    """用边界约束弛豫初始坐标.

    对边界配对施加距离约束, 让跨 segment 边界的配对在 3D 空间中
    满足合理的几何关系.

    Args:
        coords: (L, 3) 初始 P/C1' 坐标
        boundary_pairs: build_boundary_pairs() 输出的约束对列表
        n_steps: 能量最小化步数

    Returns:
        (L, 3) 约束弛豫后的坐标
    """
    if not boundary_pairs:
        return coords

    try:
        import openmm
        from openmm import app, unit

        L = len(coords)
        system = openmm.System()

        # 拓扑
        topology = app.Topology()
        chain = topology.addChain()
        res = topology.addResidue("RNA", chain)
        for i in range(L):
            topology.addAtom(f"P{i}", app.Element.getBySymbol("P"), res)

        # 粒子质量
        for i in range(L):
            system.addParticle(110.0)

        # 键长约束 (harmonic, 保持 backbone)
        bond_force = openmm.HarmonicBondForce()
        for i in range(L - 1):
            bond_force.addBond(
                i, i + 1,
                5.9 * unit.angstrom,
                100.0 * unit.kilocalorie_per_mole / unit.angstrom**2,
            )
        system.addForce(bond_force)

        # 边界配对距离约束
        pair_force = openmm.HarmonicBondForce()
        for gi, gj, edge_type in boundary_pairs:
            if gi >= L or gj >= L:
                continue
            target = WC_DIST_TARGET  # 统一目标距离
            pair_force.addBond(
                gi, gj,
                target * unit.angstrom,
                50.0 * unit.kilocalorie_per_mole / unit.angstrom**2,
            )
        system.addForce(pair_force)

        # 设置坐标
        positions = [
            openmm.Vec3(coords[i, 0], coords[i, 1], coords[i, 2]) * unit.angstrom
            for i in range(L)
        ]

        # 能量最小化
        integrator = openmm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond,
        )
        context = openmm.Context(system, integrator)
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=n_steps)

        state = context.getState(getPositions=True)
        positions = state.getPositions()
        result = np.array([
            [positions[i].x, positions[i].y, positions[i].z]
            for i in range(L)
        ])
        return result

    except ImportError:
        return coords
    except Exception:
        return coords


def format_boundary_pdb_remarks(
    boundary_pairs: List[Tuple[int, int, str]],
) -> List[str]:
    """将边界约束格式化为 PDB REMARK 行.

    方便在 PDB 中记录哪些配对是硬约束, 调试时可追溯.

    Returns:
        PDB REMARK 行列表 (含换行符)
    """
    lines = [
        "REMARK   1 Boundary constraints from global bpp\n",
        "REMARK   1 Format: RESIDUE_I RESIDUE_J EDGE_TYPE PROBABILITY\n",
    ]
    for gi, gj, edge_type in boundary_pairs:
        lines.append(
            f"REMARK   1 BOUNDARY {gi+1:5d} {gj+1:5d} {edge_type:8s}\n"
        )
    return lines
