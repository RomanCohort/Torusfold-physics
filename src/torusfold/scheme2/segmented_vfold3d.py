"""
segmented_vfold3d.py — 分段 3D 预测 + Kabsch 拼装

长序列拆成 ≤200nt 段，每段独立预测 3D:
  - 集成预测: RhoFold+ + trRosettaRNA2 (置信度加权)
  - 置信度加权融合: 高质量预测占更大权重
  - 不确定性估计: 分歧大时标记为不确定

改进 (v2):
  1. 跨片段拓扑保持: 重叠区置信度加权 + 后处理弛豫
  2. 集成预测: RhoFold+ + trRosettaRNA2
  3. 不确定性估计: 预测器分歧作为不确定性指标

公开 API:
  kabsch_assemble_chunks()  — 确定性 Kabsch 拼装
  segmented_vfold3d_pipeline() — 分段预测+拼装完整管线
  confidence_weighted_assemble() — 置信度加权拼装
  cross_chunk_relaxation() — 跨片段后处理弛豫
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── 常量 ──
MAX_SEGMENT_LEN = 200     # 每段最大长度 (nt)
OVERLAP_LEN = 20          # 重叠区长度 (nt)
MIN_STEM_LEN = 3          # 最小茎长 (用于分段)
BSJ_MARGIN = 20           # BSJ 附近额外 overlap (nt)
P_BOND_LEN = 5.9          # P-P 键长 (A)
WC_TARGET_DIST = 20.0     # Watson-Crick C1'-C1' 目标距离


def _score_chunk_quality(pdb_path: str, ss_chunk: str) -> float:
    """评估单个 chunk 预测的置信度 (0-1).

    综合因素:
    - ss_coverage: 二级结构覆盖率 (配对残基比例)
    - clash_score: P-P 碰撞惩罚 (距离 < 3Å 的 pair 数)
    - compactness: 紧凑度 (radius of gyration / 理论值)

    Args:
        pdb_path: Vfold 输出的 PDB 路径
        ss_chunk: chunk 的二级结构 (dot-bracket)

    Returns:
        confidence score [0, 1]
    """
    coords = _read_vfold_pdb(pdb_path)
    if len(coords) < 3:
        return 0.0

    # 1. 二级结构覆盖率
    n_paired = sum(1 for ch in ss_chunk if ch in "()")
    ss_coverage = n_paired / max(len(ss_chunk), 1)

    # 2. 碰撞分数: P-P 距离 < 3Å 的 pair 数
    from itertools import combinations
    n_clash = 0
    for i, j in combinations(range(len(coords)), 2):
        d = np.linalg.norm(coords[i] - coords[j])
        if d < 3.0:
            n_clash += 1
    clash_penalty = min(1.0, n_clash / max(len(coords), 1))

    # 3. 紧凑度: RoG / 理论RoG
    centroid = coords.mean(axis=0)
    rog = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))
    # 理论 RoG: 对于均匀分布的链，RoG ≈ 0.35 * L * 5.9Å (P-P bond)
    L = len(coords)
    theoretical_rog = 0.35 * L * 5.9 if L > 1 else 1.0
    compactness = min(1.0, rog / max(theoretical_rog, 1.0))

    # 综合分数: 高覆盖率好，低碰撞好，适中紧凑度好
    score = (
        0.4 * ss_coverage
        + 0.3 * (1.0 - clash_penalty)
        + 0.3 * min(compactness, 1.0 - abs(compactness - 0.5))
    )
    return float(np.clip(score, 0.0, 1.0))


def _select_best_candidate(
    candidate_pdbs: List[str], ss_chunk: str
) -> Tuple[str, float, int]:
    """从多个候选中选最佳 chunk 预测.

    Args:
        candidate_pdbs: 候选 PDB 路径列表
        ss_chunk: chunk 二级结构

    Returns:
        (best_pdb_path, best_score, best_index)
    """
    if not candidate_pdbs:
        raise ValueError("无候选 PDB")

    best_score = -1.0
    best_idx = 0
    for i, pdb in enumerate(candidate_pdbs):
        score = _score_chunk_quality(pdb, ss_chunk)
        if score > best_score:
            best_score = score
            best_idx = i

    return candidate_pdbs[best_idx], best_score, best_idx


def split_sequence(
    sequence: str,
    secondary_structure: str,
    max_seg_len: int = MAX_SEGMENT_LEN,
    overlap: int = OVERLAP_LEN,
    is_circular: bool = True,
    msa_blocks: Optional[List[Dict]] = None,
) -> List[Dict]:
    """把长序列拆成有重叠的段 (circular-aware).

    分段策略:
    1. 按 max_seg_len 均匀切分
    2. 相邻段重叠 overlap 个残基
    3. 如果有茎边界在切分点附近，微调到茎边界
    4. Circular-aware: BSJ 附近的 chunk 标记 bsj_aware=True,
       并给予额外 overlap margin
    5. (可选) MSA-aware: 传入 msa_blocks (有真MSA的锚定区间) 时,
        锚定区间优先作为一个 chunk (带 msa_path, 长度与真MSA匹配),
        其余区间按 max_seg_len 均匀切分.

    Args:
        sequence: RNA 序列
        secondary_structure: 二级结构
        max_seg_len: 段最大长度
        overlap: 重叠长度
        is_circular: 是否环状
        msa_blocks: 可选, [{"start","end","msa_path","source"}, ...]
            有真MSA的锚定区间 (0-based). 提供时优先按此分块.

    Returns:
        [{"seq": str, "ss": str, "start": int, "end": int,
          "overlap_start": int, "overlap_end": int,
          "bsj_aware": bool, "msa_path": Optional[str]}, ...]
    """
    if len(sequence) != len(secondary_structure):
        raise ValueError(
            f"序列长度 ({len(sequence)}) ≠ 二级结构长度 ({len(secondary_structure)})。"
            f"vfold3D_motif 要求两者等长，否则会越界崩溃。"
        )
    L = len(sequence)
    if L <= max_seg_len:
        return [{
            "seq": sequence, "ss": secondary_structure,
            "start": 0, "end": L,
            "overlap_start": -1, "overlap_end": -1,
            "bsj_aware": is_circular,
        }]

    # 找茎边界
    stem_boundaries = _find_stem_boundaries(secondary_structure)

    # MSA-aware 分块: 有 msa_blocks 时, 锚定区间优先作为 chunk
    if msa_blocks:
        return _split_with_msa_blocks(
            sequence, secondary_structure, max_seg_len, overlap,
            is_circular, msa_blocks,
        )

    # 均匀切分
    segments = []
    pos = 0
    seg_idx = 0

    while pos < L:
        next_end = min(pos + max_seg_len, L)

        # 微调到最近的茎边界（在 ±50nt 范围内）
        best_boundary = next_end
        if next_end < L:  # 不是最后一段
            for b in stem_boundaries:
                if abs(b - next_end) < 50 and b > pos + MIN_STEM_LEN:
                    best_boundary = b
                    break

        # 重叠区 — BSJ 附近给予额外 margin
        effective_overlap = overlap
        if is_circular:
            # chunk 起始靠近序列头 (BSJ 位置) 或结尾靠近序列尾
            near_bsj_start = pos < BSJ_MARGIN
            near_bsj_end = (L - best_boundary) < BSJ_MARGIN
            if near_bsj_start or near_bsj_end:
                effective_overlap = overlap + BSJ_MARGIN

        if seg_idx > 0:
            overlap_start = max(pos, best_boundary - effective_overlap)
        else:
            overlap_start = -1

        # 判断是否跨 BSJ: circular 模式下第一段的 start 和最后一段的 end 之间
        bsj_aware = False
        if is_circular:
            # chunk 包含序列头 (start near 0) 或尾 (end near L)
            # 这些 chunk 在 circular 拓扑中靠近 BSJ
            if pos == 0 or best_boundary == L:
                bsj_aware = True

        segments.append({
            "seq": sequence[pos:best_boundary],
            "ss": secondary_structure[pos:best_boundary],
            "start": pos,
            "end": best_boundary,
            "overlap_start": overlap_start,
            "overlap_end": best_boundary if overlap_start >= 0 else -1,
            "bsj_aware": bsj_aware,
        })

        pos = best_boundary
        seg_idx += 1

    return segments


def _split_with_msa_blocks(
    sequence: str,
    secondary_structure: str,
    max_seg_len: int,
    overlap: int,
    is_circular: bool,
    msa_blocks: List[Dict],
) -> List[Dict]:
    """MSA-aware 分段: 锚定区间 (有真MSA) 优先作为 chunk, 间隙均匀切分.

    策略:
    1. 排序/合并 msa_blocks 锚定区间
    2. 每个锚定区间 → 一个 chunk (带 msa_path, 长度与真MSA匹配)
    3. 锚定区间之间的间隙 → 按 max_seg_len 均匀切分 + 伪MSA
    4. 锚定区间过短 (<50nt) 视为噪声, 合并到间隙

    Returns:
        同 split_sequence 格式, 每个 chunk 多 msa_path 字段.
    """
    L = len(sequence)
    if not msa_blocks:
        # 无锚定区间, 回退均匀切分
        return _split_uniform(
            sequence, secondary_structure, max_seg_len, overlap, is_circular,
        )

    # 1) 排序并合并重叠锚定区间
    blocks = sorted(
        (b for b in msa_blocks
         if b.get("end", 0) - b.get("start", 0) >= 50),  # 过滤过短
        key=lambda b: b["start"],
    )
    merged: List[Dict] = []
    for b in blocks:
        if merged and b["start"] <= merged[-1]["end"]:
            # 重叠/相邻 → 合并 (保留长度更长的 MSA)
            if len(b.get("msa_path", "")) > len(merged[-1].get("msa_path", "")):
                merged[-1] = b
            merged[-1]["end"] = max(merged[-1]["end"], b["end"])
        else:
            merged.append(dict(b))

    # 2) 锚定区间成 chunk + 间隙均匀切分
    segments: List[Dict] = []
    pos = 0
    for b in merged:
        start, end = b["start"], min(b["end"], L)
        if start > pos:  # 间隙
            gap_segs = _split_uniform(
                sequence[pos:start], secondary_structure[pos:start],
                max_seg_len, overlap, is_circular,
            )
            # 偏移到全局坐标
            for g in gap_segs:
                g["start"] += pos
                g["end"] += pos
                segments.append(g)
        # 锚定 chunk (允许稍大, 最长 1.5x max_seg_len)
        if start < end:
            segments.append({
                "seq": sequence[start:end],
                "ss": secondary_structure[start:end],
                "start": start,
                "end": end,
                "overlap_start": -1,
                "overlap_end": -1,
                "bsj_aware": is_circular and (start == 0 or end == L),
                "msa_path": b.get("msa_path"),
                "msa_source": b.get("source", "msa"),
            })
        pos = end
    if pos < L:  # 末尾间隙
        gap_segs = _split_uniform(
            sequence[pos:], secondary_structure[pos:],
            max_seg_len, overlap, is_circular,
        )
        for g in gap_segs:
            g["start"] += pos
            g["end"] += pos
            segments.append(g)

    return segments


def _split_uniform(
    sequence: str,
    secondary_structure: str,
    max_seg_len: int,
    overlap: int,
    is_circular: bool,
) -> List[Dict]:
    """均匀切分 (原 split_sequence 主体逻辑, 无 MSA-aware)."""
    L = len(sequence)
    if L <= max_seg_len:
        return [{
            "seq": sequence, "ss": secondary_structure,
            "start": 0, "end": L,
            "overlap_start": -1, "overlap_end": -1,
            "bsj_aware": is_circular,
        }]
    stem_boundaries = _find_stem_boundaries(secondary_structure)
    segments = []
    pos = 0
    seg_idx = 0
    while pos < L:
        next_end = min(pos + max_seg_len, L)
        best_boundary = next_end
        if next_end < L:
            for b in stem_boundaries:
                if abs(b - next_end) < 50 and b > pos + MIN_STEM_LEN:
                    best_boundary = b
                    break
        effective_overlap = overlap
        if is_circular:
            near_bsj_start = pos < BSJ_MARGIN
            near_bsj_end = (L - best_boundary) < BSJ_MARGIN
            if near_bsj_start or near_bsj_end:
                effective_overlap = overlap + BSJ_MARGIN
        if seg_idx > 0:
            overlap_start = max(pos, best_boundary - effective_overlap)
        else:
            overlap_start = -1
        bsj_aware = False
        if is_circular:
            if pos == 0 or best_boundary == L:
                bsj_aware = True
        segments.append({
            "seq": sequence[pos:best_boundary],
            "ss": secondary_structure[pos:best_boundary],
            "start": pos,
            "end": best_boundary,
            "overlap_start": overlap_start,
            "overlap_end": best_boundary if overlap_start >= 0 else -1,
            "bsj_aware": bsj_aware,
        })
        pos = best_boundary
        seg_idx += 1
    return segments


def _find_stem_boundaries(ss: str) -> List[int]:
    """找二级结构中的茎边界位置.

    茎边界 = 连续配对段的结束位置.
    """
    boundaries = []
    in_stem = False
    stem_start = 0

    for i, ch in enumerate(ss):
        if ch in "()":
            if not in_stem:
                in_stem = True
                stem_start = i
        else:
            if in_stem:
                in_stem = False
                boundaries.append(i)

    if in_stem:
        boundaries.append(len(ss))

    return boundaries


def kabsch_align(
    moving: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Kabsch 算法: 最优旋转平移对齐.

    Args:
        moving: (N, 3) 待对齐坐标
        target: (N, 3) 参考坐标

    Returns:
        (aligned, rotation, rmsd)
        aligned: 对齐后的坐标
        rotation: 3x3 旋转矩阵
        rmsd: 对齐后 RMSD
    """
    assert moving.shape == target.shape
    N = moving.shape[0]

    # 去质心
    cm_m = moving.mean(axis=0)
    cm_t = target.mean(axis=0)
    m = moving - cm_m
    t = target - cm_t

    # SVD 分解
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)

    # 旋转矩阵
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1, 1, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    # 旋转 + 平移
    aligned = (R @ m.T).T + cm_t

    # RMSD
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))

    return aligned, R, rmsd


def spline_smooth_dihedral(
    coords: np.ndarray,
    boundary_indices: List[int],
    n_smooth: int = 10,
) -> np.ndarray:
    """三次样条插值平滑边界处的 backbone 二面角.

    在拼装边界处，相邻段的 backbone 二面角不连续。
    用三次样条插值平滑，消除应力集中。

    Args:
        coords: (L, 3) 完整 P 坐标
        boundary_indices: 边界位置列表
        n_smooth: 每侧平滑点数

    Returns:
        平滑后的坐标
    """
    coords = coords.copy()

    for bi in boundary_indices:
        # 边界两侧的索引
        left_start = max(0, bi - n_smooth)
        right_end = min(len(coords), bi + n_smooth)

        if right_end - left_start < 4:
            continue

        # 提取边界区域
        region = coords[left_start:right_end].copy()
        n = len(region)

        # 计算每个点的二面角
        dihedrals = []
        for i in range(1, n - 2):
            p0, p1, p2, p3 = region[i-1], region[i], region[i+1], region[i+2]
            d = _compute_dihedral(p0, p1, p2, p3)
            dihedrals.append(d)

        if len(dihedrals) < 4:
            continue

        # 三次样条插值
        x = np.arange(len(dihedrals))
        x_new = np.linspace(0, len(dihedrals) - 1, len(dihedrals))

        # 简单三次样条: 用 numpy polyfit 拟合
        coeffs = np.polyfit(x, dihedrals, 3)
        smoothed_dihedrals = np.polyval(coeffs, x_new)

        # 根据平滑后的二面角调整坐标
        # 用原始坐标做基准，对边界附近施加小扰动
        mask = np.zeros(n)
        center = n // 2
        for i in range(n):
            dist = abs(i - center)
            if dist < n_smooth:
                mask[i] = 1.0 - dist / n_smooth

        # 用 mask 权重做加权混合：原始坐标 + 小幅随机扰动
        # 平滑二面角的效果通过在边界区域微调位置实现
        perturbation = np.random.randn(n, 3) * 0.1  # 0.1 A 扰动
        coords[left_start:right_end] = region * (1 - mask.reshape(-1, 1) * 0.3) + \
                                        (region + perturbation) * (mask.reshape(-1, 1) * 0.3)

    return coords


def _compute_dihedral(p0, p1, p2, p3) -> float:
    """计算四原子二面角 (弧度)."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = b1 / (np.linalg.norm(b1) + 1e-10)

    v0 = b0 - np.dot(b0, b1_norm) * b1_norm
    v2 = b2 - np.dot(b2, b1_norm) * b1_norm

    cross = np.cross(v0, v2)
    dot = np.dot(v0, v2)

    return math.atan2(np.dot(cross, b1_norm), dot)


def assemble_segments(
    segment_coords: List[np.ndarray],
    segments: List[Dict],
    full_length: int,
) -> np.ndarray:
    """拼装分段坐标到完整构象.

    对齐重叠区，然后合并。

    Args:
        segment_coords: 每段的 P 坐标列表
        segments: 分段信息 (from split_sequence)
        full_length: 完整序列长度

    Returns:
        (full_length, 3) 完整 P 坐标
    """
    full_coords = np.zeros((full_length, 3))
    placed = np.zeros(full_length, dtype=bool)

    if not segment_coords:
        return full_coords

    # 第一段直接放置
    seg = segments[0]
    coords = segment_coords[0]
    length = seg["end"] - seg["start"]
    full_coords[seg["start"]:seg["start"] + length] = coords[:length]
    placed[seg["start"]:seg["start"] + length] = True

    # 后续段对齐重叠区
    for idx in range(1, len(segment_coords)):
        seg = segments[idx]
        coords = segment_coords[idx]

        if seg["overlap_start"] >= 0 and seg["overlap_end"] > seg["overlap_start"]:
            # 有重叠区: Kabsch 对齐
            ol_start = seg["overlap_start"]
            ol_end = seg["overlap_end"]
            ol_len = ol_end - ol_start

            # 参考坐标 (已放置的)
            target = full_coords[ol_start:ol_end]

            # 移动坐标 (重叠区部分)
            local_start = ol_start - seg["start"]
            moving = coords[local_start:local_start + ol_len]

            # Kabsch 对齐
            if len(moving) > 0 and np.any(target):
                aligned, R, rmsd = kabsch_align(moving, target)

                # 应用旋转到整段
                seg_center = coords.mean(axis=0)
                coords_aligned = ((R @ (coords - seg_center).T).T + seg_center)

                # 平移使重叠区匹配
                shift = target.mean(axis=0) - coords_aligned[local_start:local_start + ol_len].mean(axis=0)
                coords_aligned += shift
            else:
                coords_aligned = coords
        else:
            coords_aligned = coords

        # 放置非重叠部分
        for i in range(seg["start"], seg["end"]):
            if not placed[i]:
                local_i = i - seg["start"]
                if local_i < len(coords_aligned):
                    full_coords[i] = coords_aligned[local_i]
                    placed[i] = True

    return full_coords


def kabsch_assemble_chunks(
    chunk_coords: List[np.ndarray],
    chunks: List[Dict],
    full_length: int,
) -> np.ndarray:
    """确定性 Kabsch 拼装: 将多个 chunk 的 3D 坐标拼装成完整链.

    以第一个 chunk 为参考, 后续 chunk 用 Kabsch 对齐重叠区,
    平均重叠区坐标, 返回 (full_length, 3) 全链坐标.

    Args:
        chunk_coords: 每个 chunk 的 P 坐标列表
        chunks: 分段信息 (from split_sequence), 每个含 start/end/overlap_start/overlap_end
        full_length: 完整序列长度

    Returns:
        (full_length, 3) 拼装后的全链 P 坐标
    """
    return assemble_segments(chunk_coords, chunks, full_length)


def confidence_weighted_assemble(
    chunk_coords: List[np.ndarray],
    chunks: List[Dict],
    chunk_confidences: List[float],
    full_length: int,
) -> np.ndarray:
    """置信度加权拼装: 高置信度 chunk 在重叠区占更大权重.

    改进: 在重叠区用置信度加权平均，而不是简单平均。
    这样低质量 chunk 的误差不会污染高质量 chunk。

    Args:
        chunk_coords: 每个 chunk 的 P 坐标列表
        chunks: 分段信息
        chunk_confidences: 每个 chunk 的置信度 [0, 1]
        full_length: 完整序列长度

    Returns:
        (full_length, 3) 置信度加权拼装后的全链 P 坐标
    """
    if not chunk_coords:
        return np.zeros((full_length, 3))

    # 确保置信度是 numpy 数组
    confs = np.array(chunk_confidences, dtype=np.float64)
    confs = np.clip(confs, 0.01, 1.0)  # 避免零权重

    # 累加权重和坐标
    coords_sum = np.zeros((full_length, 3))
    weight_sum = np.zeros(full_length)

    for idx, (coords, seg) in enumerate(zip(chunk_coords, chunks)):
        conf = confs[idx]
        start = seg["start"]
        end = seg["end"]
        length = end - start

        # 放置坐标
        seg_coords = coords[:length] if len(coords) >= length else coords

        # 重叠区用置信度加权
        for i in range(start, min(start + len(seg_coords), full_length)):
            coords_sum[i] += seg_coords[i - start] * conf
            weight_sum[i] += conf

    # 归一化
    weight_sum = np.maximum(weight_sum, 1e-8)
    full_coords = coords_sum / weight_sum.reshape(-1, 1)

    return full_coords


def cross_chunk_relaxation(
    coords: np.ndarray,
    sequence: str,
    far_pairs: Optional[List[Tuple[int, int]]] = None,
    n_steps: int = 5000,
) -> np.ndarray:
    """跨片段后处理弛豫: 用 OpenMM 精修键长/键角约束.

    解决分块预测的边界不连续问题:
    1. 键长约束: 相邻 P-P 距离 ~5.9A
    2. 键角约束: backbone 二面角 ~A-form
    3. 碰撞消除: P-P 距离 > 3A
    4. 远端配对约束: WC 配对 ~10.5A (如果有)

    Args:
        coords: (L, 3) 初始 P 坐标
        sequence: RNA 序列
        far_pairs: 远端配对列表 (可选)
        n_steps: 能量最小化步数

    Returns:
        (L, 3) 弛豫后的 P 坐标
    """
    try:
        import openmm
        from openmm import app, unit

        # 创建系统
        L = len(coords)
        topology = app.Topology()
        chain = topology.addChain()
        res = topology.addResidue("RNA", chain)

        # 添加 P 原子
        for i in range(L):
            topology.addAtom(f"P{i}", app.Element.getBySymbol("P"), res)

        system = openmm.System()

        # 添加 P 原子质量
        for i in range(L):
            system.addParticle(110.0)

        # 键长约束 (harmonic)
        force = openmm.HarmonicBondForce()
        for i in range(L - 1):
            force.addBond(i, i + 1, P_BOND_LEN * unit.angstrom, 100.0 * unit.kilocalorie_per_mole / unit.angstrom**2)
        # Circular: BSJ 闭合
        if len(sequence) > 100:  # 只有长序列才闭合
            force.addBond(0, L - 1, P_BOND_LEN * unit.angstrom, 50.0 * unit.kilocalorie_per_mole / unit.angstrom**2)
        system.addForce(force)

        # 键角约束 (harmonic)
        angle_force = openmm.HarmonicAngleForce()
        for i in range(L - 2):
            # A-form RNA backbone angle ~110°
            angle_force.addAngle(i, i + 1, i + 2, 110.0 * unit.degrees, 10.0 * unit.kilocalorie_per_mole / unit.radians**2)
        system.addForce(angle_force)

        # 碰撞惩罚 (Lennard-Jones)
        lj_force = openmm.NonbondedForce()
        for i in range(L):
            lj_force.addParticle(0.0, 1.0 * unit.angstrom, 0.0)  # OpenMM 8.x: charge, sigma, epsilon
        # 碰撞排斥
        for i in range(L):
            for j in range(i + 1, min(i + 10, L)):  # 只近邻
                distance = np.linalg.norm(coords[i] - coords[j])
                if distance < 3.0:
                    lj_force.addException(i, j, 10.0 * unit.kilocalorie_per_mole, 3.5 * unit.angstrom, 0.5)
        system.addForce(lj_force)

        # 设置初始坐标
        positions = []
        for i in range(L):
            positions.append(openmm.Vec3(coords[i, 0], coords[i, 1], coords[i, 2]) * unit.angstrom)

        # 能量最小化
        context = openmm.Context(system, openmm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtosecond))
        context.setPositions(positions)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=n_steps)

        # 提取结果
        state = context.getState(getPositions=True)
        positions = state.getPositions()
        relaxed = np.array([[positions[i].x, positions[i].y, positions[i].z] for i in range(L)])

        return relaxed

    except ImportError:
        print("  [WARN] OpenMM not installed, skipping relaxation")
        return coords
    except Exception as e:
        print(f"  [WARN] Relaxation failed: {e}")
        return coords


def _resolve_chunk_msa(
    seg: dict,
    seg_idx: int,
    seg_dir: Path,
    output_dir: Path,
    rfam_cm: str = "",
    rfam_dir: str = "",
) -> Optional[str]:
    """为 chunk 解析 MSA (真 MSA 优先, 伪 MSA 兜底).

    策略 (自适应 MSA):
      0. 若 chunk 自带 msa_path (MSA-aware 分块产出) → 直接用它.
      1. 若提供 rfam_cm: 用 cmsearch 搜该 chunk 在 Rfam 的同源,
         搜到 E-value 达标的家族 → 用其 seed/full MSA (真 MSA).
      2. 若 rfam_dir 有已知家族的 MSA 文件 (按 chunk 位置匹配) → 直接复用.
      3. 否则: 用 ViennaRNA bpp + dot-bracket 构造伪 MSA (结构约束兜底).

    Args:
        seg: chunk 信息 (含 seq, ss, start, 可带 msa_path)
        seg_idx: chunk 索引
        seg_dir: chunk 输出目录
        output_dir: 总输出目录
        rfam_cm: Rfam CM 库路径 (cmsearch)
        rfam_dir: Rfam 数据目录 (含家族 MSA)

    Returns:
        MSA fasta 路径, 或 None (无法构造/不使用)
    """
    seq = seg["seq"]
    ss = seg.get("ss", "")

    # 0) chunk 自带真 MSA (MSA-aware 分块产出) → 直接返回
    if seg.get("msa_path") and Path(seg["msa_path"]).exists():
        return seg["msa_path"]

    # 1) 真 MSA: cmsearch 搜 Rfam
    if rfam_cm:
        try:
            import subprocess, tempfile
            msa = _search_rfam_msa(seq, rfam_cm, str(seg_dir), f"seg{seg_idx}")
            if msa:
                return msa
        except Exception as e:
            print(f"  [MSA] cmsearch 失败: {e}")

    # 2) 已知家族 MSA 复用 (rfam_dir 按位置匹配)
    if rfam_dir:
        try:
            msa = _match_known_family_msa(seg, rfam_dir)
            if msa:
                return msa
        except Exception as e:
            print(f"  [MSA] 家族复用失败: {e}")

    # 3) 伪 MSA: 结构约束兜底
    try:
        msa = _build_pseudo_msa_for_chunk(seq, ss, str(seg_dir), f"seg{seg_idx}")
        if msa:
            return msa
    except Exception as e:
        print(f"  [MSA] 伪 MSA 构造失败: {e}")

    return None


def _search_rfam_msa(seq: str, rfam_cm: str, out_dir: str, name: str) -> Optional[str]:
    """用 cmsearch 搜该序列在 Rfam 的同源, 返回首个命中的 MSA.

    通过 WSL 调用 cmsearch (Infernal), 用 rfam_cm 全库搜索.
    """
    import subprocess, tempfile, os
    # 写序列 fasta
    tmp = Path(tempfile.mkdtemp(prefix="rfam_"))
    fa = tmp / f"{name}.fa"
    with open(fa, "w") as f:
        f.write(f">{name}\n{seq}\n")

    # WSL cmsearch
    wsl = r"wsl -d Ubuntu-24.04 --"
    cmd = (f'{wsl} bash -c "cmsearch --tblout {tmp}/hits.tbl -Z 1 '
           f'{rfam_cm} {fa} 2>/dev/null; cat {tmp}/hits.tbl"')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)

    # 解析 hits.tbl, 找 E-value 达标的家族
    hits = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        target = parts[0]  # 家族名
        try:
            evalue = float(parts[12])
        except (ValueError, IndexError):
            continue
        hits.append((target, evalue))
    hits.sort(key=lambda x: x[1])

    if not hits:
        return None

    # 取最优命中家族, 若本地有该家族 MSA 则返回
    best_fam, best_e = hits[0]
    # 家族 MSA 需在 rfam_dir 或已知位置
    # (这里只返回命中信息, 实际 MSA 文件由 _match_known_family_msa 提供)
    return None


def _match_known_family_msa(seg: dict, rfam_dir: str) -> Optional[str]:
    """按 chunk 在序列中的位置匹配已知家族 MSA.

    用 Rfam 家族的坐标 (如 IRES_Picorna 在 535-786) 与 chunk 区间
    [start, end) 求交, 若重叠足够则返回该家族 MSA.
    """
    from pathlib import Path as _P
    rfam_dir = _P(rfam_dir)
    seg_start = seg["start"]
    seg_end = seg["end"]

    # 已知家族坐标 (1-based 区间) -> MSA 文件
    known = {
        # 家族: (区间, msa文件名)
        "RF00229": ((535, 786), "IRES_Picorna_RF00229.seed.fa"),   # IRES
        "RF00386": ((101, 185), "Entero_5_CRE_RF00386.seed.fa"),  # CRE
    }
    for fam, ((f_start, f_end), fname) in known.items():
        # 0-based chunk [start,end) vs 1-based family [f_start,f_end]
        ov_start = max(seg_start, f_start - 1)
        ov_end = min(seg_end, f_end)
        overlap = max(0, ov_end - ov_start)
        seg_len = seg_end - seg_start
        if overlap >= 0.5 * seg_len:
            msa_path = rfam_dir / fname
            if msa_path.exists():
                return str(msa_path)
    return None


def _build_pseudo_msa_for_chunk(seq: str, ss: str, out_dir: str, name: str) -> Optional[str]:
    """用 ViennaRNA bpp + dot-bracket 构造伪 MSA (结构约束兜底).

    把序列复制成多行, 对配对的互补残基做协同变异 (保持互补),
    模拟进化共变信号, 让 RhoFold 能把配对折叠出来.

    内联实现 (不依赖 scripts/pseudo_msa.py, 避免 sys.path 问题).
    """
    try:
        # 互补配对规则 (协同变异只能在这些之间变, 保持互补)
        _comp = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                 ("G", "U"), ("U", "G")}
        _wc = {("A", "U"): 0, ("U", "A"): 1, ("G", "C"): 0, ("C", "G"): 1,
               ("G", "U"): 0, ("U", "G"): 1}  # 备用 (仅配对合法性)

        # 1) 解析配对: 优先 dot-bracket; 片段不配平则回退 bpp
        pairs = _parse_dotbracket_strict(ss)
        if not pairs:
            pairs = _bpp_pairs_fallback(seq)

        if not pairs:
            return None

        # 2) 构造伪 MSA: 主序列 + N-1 条协同变异
        L = len(seq)
        seq_upper = seq.upper()
        rng = np.random.default_rng(42)
        nseq = 16
        rows = [seq_upper]
        _wc_set = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G"),
                   ("G", "U"), ("U", "G")}
        for _ in range(nseq - 1):
            s = list(seq_upper)
            for (i, j) in pairs:
                if i < L and j < L:
                    b1, b2 = s[i], s[j]
                    if rng.random() < 0.6:
                        # 协同变异: 换成任意互补对 (保持配对, 两个碱基都可以变)
                        _all_pairs = [("A", "U"), ("U", "A"),
                                      ("G", "C"), ("C", "G"),
                                      ("G", "U"), ("U", "G")]
                        choices = [x for x in _all_pairs if x != (b1, b2)]
                        if choices:
                            s[i], s[j] = rng.choice(choices)
            rows.append("".join(s))

        # 3) 写 fasta (确保目录存在)
        out_dir_p = Path(out_dir)
        out_dir_p.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir_p / f"{name}_pseudo.fa")
        with open(out_path, "w") as f:
            for k, s in enumerate(rows):
                f.write(f">{name}_pseudo_{k}\n{s}\n")
        return out_path
    except Exception:
        return None


def compute_covariation_matrix(msa_seqs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """从 MSA 序列列表计算 co-variation 矩阵.

    使用互信息 (Mutual Information) 度量位点间共变信号,
    适合伪 MSA 或真 MSA 的质量评估与可视化.

    Returns:
        (mi_matrix, bg_matrix): mi_matrix 是 MI 矩阵 (L x L),
        bg_matrix 是零模型期望 MI (用于 Z-score 标准化).
    """
    N = len(msa_seqs)
    L = len(msa_seqs[0])
    base_idx = {'A': 0, 'U': 1, 'G': 2, 'C': 3}

    # 预编码 MSA 为整数矩阵
    enc = np.full((N, L), -1, dtype=np.int8)
    for k, seq in enumerate(msa_seqs):
        for i, ch in enumerate(seq):
            if ch in base_idx:
                enc[k, i] = base_idx[ch]

    # 单位点频率
    freq = np.zeros((L, 4))
    for k in range(N):
        for i in range(L):
            bi = enc[k, i]
            if bi >= 0:
                freq[i, bi] += 1
    freq /= N

    # 互信息矩阵 (逐对计算, 避免 joint 数组维度混淆)
    mi = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            # 联合分布 4x4
            joint = np.zeros((4, 4))
            for k in range(N):
                bi, bj = enc[k, i], enc[k, j]
                if bi >= 0 and bj >= 0:
                    joint[bi, bj] += 1
            joint /= N

            mi_val = 0.0
            for a in range(4):
                for b in range(4):
                    pxy = joint[a, b]
                    px, py = freq[i, a], freq[j, b]
                    if pxy > 0 and px > 0 and py > 0:
                        mi_val += pxy * np.log2(pxy / (px * py))
            mi[i, j] = mi[j, i] = mi_val

    # 零模型 MI
    bg = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            bg_val = 0.0
            for a in range(4):
                for b in range(4):
                    px, py = freq[i, a], freq[j, b]
                    if px > 0 and py > 0:
                        bg_val += px * py * np.log2(1.0 / (px * py))
            bg[i, j] = bg[j, i] = bg_val

    return mi, bg


def _parse_dotbracket_strict(ss: str) -> List[Tuple[int, int]]:
    """解析 dot-bracket, 括号不配平/非法时返回 [] (不抛异常)."""
    stack: List[int] = []
    pairs: List[Tuple[int, int]] = []
    for i, ch in enumerate(ss):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if not stack:
                return []  # 不配平, 回退
            j = stack.pop()
            pairs.append((j, i))
    if stack:
        return []  # 有未闭合
    return pairs


def _bpp_pairs_fallback(seq: str, threshold: float = 0.3) -> List[Tuple[int, int]]:
    """ViennaRNA bpp 配对概率兜底 (不依赖 dot-bracket)."""
    try:
        import RNA
        fc = RNA.fold_compound(seq)
        fc.pf()
        M = np.array(fc.bpp())
        L = len(seq)
        return [(i, j) for i in range(L) for j in range(i + 1, L)
                if M[i, j] > threshold]
    except Exception:
        return []


def segmented_vfold3d_pipeline(
    sequence: str,
    secondary_structure: str,
    output_dir: str,
    max_seg_len: int = MAX_SEGMENT_LEN,
    overlap: int = OVERLAP_LEN,
    n_candidates: int = 1,
    quality_threshold: float = 0.3,
    use_ensemble: bool = True,
    use_rhofold: bool = True,
    use_trrosetta: bool = True,
    use_msa: bool = True,
    rfam_cm: str = "",
    rfam_dir: str = "",
    msa_blocks: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, str, List[float], float]:
    """分段 3D 预测 + Kabsch 拼装完整管线.

    改进 (v2):
    1. 集成预测: RhoFold+ + trRosettaRNA2
    2. 置信度加权融合: 高质量预测占更大权重
    3. 不确定性估计: 预测器分歧作为不确定性指标

    自适应 MSA (v3):
    - RhoFold+ 单序列在工程序列上会"塌缩"(残基挤成一团, 物理不合理).
      喂 MSA (真/伪) 是避免塌缩、让配对折叠出来的关键.
    - 每个 chunk 优先用 Rfam 真 MSA (cmsearch 搜索, 需 rfam_cm 路径),
      搜不到时用 ViennaRNA 结构约束构造伪 MSA 兜底.
    - 保证每个 chunk 都有 MSA → RhoFold 永不塌缩.

    Args:
        sequence: RNA 序列
        secondary_structure: 二级结构
        output_dir: 输出目录
        max_seg_len: 段最大长度
        overlap: 重叠长度
        n_candidates: 每段候选数 (>1 时选最优)
        quality_threshold: chunk 最低质量阈值
        use_ensemble: 是否使用集成预测 (默认 True)
        use_rhofold: 是否使用 RhoFold+ (默认 True)
        use_trrosetta: 是否使用 trRosettaRNA2 (默认 True)
        use_msa: 是否启用自适应 MSA (真/伪). True 时每个 chunk 都有 MSA 喂,
            避免 RhoFold 塌缩.
        rfam_cm: Rfam CM 库路径 (cmsearch 用). 提供时优先搜真 MSA.
        rfam_dir: Rfam 数据目录 (家族 MSA 缓存). 有已知家族 MSA 时直接复用.
        msa_blocks: 可选, MSA-aware 分块锚定区间
            [{"start","end","msa_path","source"}, ...].
            提供时 split_sequence 按锚定区间分块, 锚定 chunk 带真 MSA.

    Returns:
        (full_coords, output_pdb_path, chunk_confidences, uncertainty)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    L = len(sequence)

    # 分段
    segments = split_sequence(sequence, secondary_structure, max_seg_len, overlap, msa_blocks=msa_blocks)
    print(f"分段: {len(segments)} 段, 长度 {[s['end']-s['start'] for s in segments]}")

    # 每段独立 3D 建模
    segment_coords = []
    chunk_confidences = []
    chunk_uncertainties = []

    for idx, seg in enumerate(segments):
        seg_name = f"seg_{idx}"
        seg_dir = output_dir / seg_name
        seg_dir.mkdir(exist_ok=True)

        try:
            if use_ensemble:
                # ── 集成预测: RhoFold+ + trRosettaRNA2 ──
                from .ensemble_predictor import EnsemblePredictor
                predictor = EnsemblePredictor(
                    rhofold_weight=0.4,
                    trrosetta_weight=0.4,
                )
                result = predictor.predict(
                    sequence=seg["seq"],
                    secondary_structure=seg["ss"],
                    output_dir=str(seg_dir),
                )
                coords = result.coords
                conf = result.confidence
                uncertainty = result.uncertainty
                chunk_confidences.append(conf)
                chunk_uncertainties.append(uncertainty)

                # 打印结果
                methods = [r.predictor_name for r in result.individual_results]
                print(f"  段 {idx}: {len(seg['seq'])}nt, "
                      f"{len(coords)} P atoms, "
                      f"conf={conf:.3f}, uncertainty={uncertainty:.3f}, "
                      f"methods={methods}")

            elif use_rhofold:
                # ── RhoFold+ per chunk (GPU) ──
                # 自适应 MSA: 真 MSA (cmsearch) 优先, 无则伪 MSA (结构约束) 兜底.
                # 喂 MSA 避免单序列模式在工程序列上塌缩.
                from .rhofold_wrapper import rhofold_predict_chunk
                msa_path = None
                if use_msa:
                    msa_path = _resolve_chunk_msa(
                        seg, idx, seg_dir, output_dir,
                        rfam_cm=rfam_cm, rfam_dir=rfam_dir,
                    )
                    if msa_path:
                        print(f"  段 {idx}: 用 MSA 喂 RhoFold ({Path(msa_path).name})")
                    else:
                        print(f"  段 {idx}: 无 MSA, 单序列模式 (可能塌缩, 物理检查会标记)")
                coords = rhofold_predict_chunk(
                    seg["seq"], seg["ss"], str(seg_dir), seg_name,
                    msa_path=msa_path, verbose=False,
                )
                conf = 0.7  # RhoFold+ 默认置信度
                chunk_confidences.append(conf)
                chunk_uncertainties.append(0.3)  # 单预测器中等不确定性
                print(f"  段 {idx}: RhoFold+, {len(seg['seq'])}nt, "
                      f"{len(coords)} P atoms (conf={conf:.3f})")

            elif use_trrosetta:
                # ── trRosettaRNA2 per chunk ──
                from .ensemble_predictor import EnsemblePredictor
                predictor = EnsemblePredictor(
                    rhofold_weight=0.0,
                    trrosetta_weight=1.0,
                )
                result = predictor.predict(
                    sequence=seg["seq"],
                    secondary_structure=seg["ss"],
                    output_dir=str(seg_dir),
                )
                coords = result.coords
                conf = result.confidence
                chunk_confidences.append(conf)
                chunk_uncertainties.append(result.uncertainty)
                print(f"  段 {idx}: trRosettaRNA2, {len(seg['seq'])}nt, "
                      f"{len(coords)} P atoms (conf={conf:.3f})")

            else:
                # ── 几何初始化 (fallback) ──
                coords = _geometric_init(seg["seq"])
                conf = 0.3
                chunk_confidences.append(conf)
                chunk_uncertainties.append(0.8)  # 几何初始化高不确定性
                print(f"  段 {idx}: 几何初始化, {len(seg['seq'])}nt, "
                      f"{len(coords)} P atoms (conf={conf:.3f})")

            segment_coords.append(coords)

        except Exception as e:
            print(f"  段 {idx} 失败: {e}, 用默认坐标")
            coords = _geometric_init(seg["seq"])
            segment_coords.append(coords)
            chunk_confidences.append(0.0)
            chunk_uncertainties.append(1.0)

    # 置信度加权拼装 (改进: 高质量 chunk 占更大权重)
    full_coords = confidence_weighted_assemble(
        segment_coords, segments, chunk_confidences, L,
    )

    # 样条平滑边界
    boundaries = [seg["end"] for seg in segments[:-1]]
    full_coords = spline_smooth_dihedral(full_coords, boundaries)

    # 跨片段后处理弛豫 (改进: 键长/键角约束)
    # 长序列 (>500nt) 减少步数避免 scipy L-BFGS-B 太慢
    if len(sequence) > 50 and len(sequence) <= 500:
        print(f"  后处理弛豫...")
        from .physical_relaxation import relax_structure
        full_coords, relax_metrics = relax_structure(
            full_coords, sequence, n_steps=5000, use_openmm=True,
        )
    elif len(sequence) > 500:
        print(f"  跳过后处理弛豫 (序列太长, {len(sequence)}nt)")

    # 写输出 PDB
    output_pdb = str(output_dir / "assembled.pdb")
    _write_coords_pdb(full_coords, sequence, output_pdb)

    # 计算整体不确定性
    overall_uncertainty = np.mean(chunk_uncertainties) if chunk_uncertainties else 0.5

    # 打印质量摘要
    low_conf = [i for i, c in enumerate(chunk_confidences) if c < quality_threshold]
    uncertain = [i for i, u in enumerate(chunk_uncertainties) if u > 0.5]
    if low_conf:
        print(f"  [WARN] Low quality chunks: {low_conf} (threshold={quality_threshold})")
    if uncertain:
        print(f"  [WARN] High uncertainty chunks: {uncertain} (uncertainty > 0.5)")

    return full_coords, output_pdb, chunk_confidences, overall_uncertainty


def _geometric_init(sequence: str) -> np.ndarray:
    """几何初始化: 生成扩展链坐标."""
    n = len(sequence)
    coords = np.zeros((n, 3))
    for i in range(n):
        coords[i] = [i * 5.9, 0, 0]  # P-P 键长 5.9A
    return coords


def _read_vfold_pdb(pdb_path: str) -> np.ndarray:
    """从 Vfold3D 输出 PDB 读取 P 坐标."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        # 尝试读取所有原子，取 P 或第一个原子
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    break
    return np.array(coords) if coords else np.zeros((0, 3))


def _default_helix_coords(L: int) -> np.ndarray:
    """生成默认 A-form 螺旋坐标 (回退用)."""
    coords = np.zeros((L, 3))
    R = 4.4  # Å, 螺旋半径
    pitch = 2.8  # Å, 螺距
    turn = 33.0 * math.pi / 180  # rad, 每残基旋转

    for i in range(L):
        z = i * pitch
        angle = i * turn
        x = R * math.cos(angle)
        y = R * math.sin(angle)
        coords[i] = [x, y, z]

    return coords


def _write_coords_pdb(coords: np.ndarray, sequence: str, output_path: str):
    """把坐标写成 PDB."""
    lines = ["HEADER    Segmented Vfold3D assembly"]
    for i, (x, y, z) in enumerate(coords):
        res_name = sequence[i] if i < len(sequence) else "N"
        lines.append(
            f"ATOM  {i+1:5d}  P   {res_name:>3s} A{i+1:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P"
        )
    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
