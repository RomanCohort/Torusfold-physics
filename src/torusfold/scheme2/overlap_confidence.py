"""
overlap_confidence.py — Level 1 重叠区置信度评估

评估分段预测中相邻 chunk 重叠区的一致性:
  - 两 chunk 在重叠区的 RMSD → 置信度分数
  - 多次采样一致性 → 柔性区检测
  - 逐残基置信度图 → Level 2.3 力场参数化

置信度分级:
  RMSD < 3A  → 高置信 (force_scale=1.0)
  3-8A       → 中置信 (force_scale=0.5)
  > 8A       → 柔性区 (force_scale=0.1, 交给力场)

公开 API:
  evaluate_overlap_confidence()  — 单对 chunk 重叠区评估
  multi_sample_confidence()      — 多次采样一致性评估
  segment_confidence_map()       — 逐残基置信度图
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ── 常量 ──
HIGH_CONF_THRESHOLD = 3.0      # RMSD < 3A → 高置信
FLEXIBLE_THRESHOLD = 8.0       # RMSD > 8A → 柔性区
DEFAULT_CONFIDENCE_TEMP = 3.0  # sigmoid 温度参数


@dataclass
class OverlapConfidence:
    """重叠区置信度评估结果.

    Attributes:
        rmsd: 重叠区 RMSD (Angstrom)
        confidence: 置信度 [0, 1], 1.0/(1+rmsd/temp)
        is_flexible: 是否为柔性区 (rmsd > 8A)
        force_scale: 力场缩放因子 (高/中/柔性)
        n_residues: 重叠区残基数
    """
    rmsd: float
    confidence: float
    is_flexible: bool
    force_scale: float
    n_residues: int

    @property
    def tier(self) -> str:
        """置信度等级: high / medium / flexible."""
        if self.rmsd < HIGH_CONF_THRESHOLD:
            return "high"
        elif self.rmsd < FLEXIBLE_THRESHOLD:
            return "medium"
        else:
            return "flexible"


def _kabsch_rmsd(moving: np.ndarray, target: np.ndarray) -> float:
    """Kabsch 对齐后计算 RMSD.

    Args:
        moving: (N, 3) 待对齐坐标
        target: (N, 3) 参考坐标

    Returns:
        对齐后的 RMSD (Angstrom)
    """
    assert moving.shape == target.shape, (
        f"Shape mismatch: {moving.shape} vs {target.shape}"
    )
    n = moving.shape[0]
    if n == 0:
        return 0.0

    # 中心化
    centroid_m = moving.mean(axis=0)
    centroid_t = target.mean(axis=0)
    m = moving - centroid_m
    t = target - centroid_t

    # SVD 求最优旋转
    H = m.T @ t
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, d])
    R = Vt.T @ sign_matrix @ U.T

    # 对齐 + RMSD
    m_aligned = (R @ m.T).T
    diff = m_aligned - t
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return float(rmsd)


def evaluate_overlap_confidence(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    overlap_indices: np.ndarray,
    temperature: float = DEFAULT_CONFIDENCE_TEMP,
) -> OverlapConfidence:
    """评估两个 chunk 在重叠区的一致性.

    将 coords_a 和 coords_b 的重叠区坐标提取出来,
    做 Kabsch 最优对齐后计算 RMSD, 再映射到置信度.

    Args:
        coords_a: chunk_a 的预测坐标 (L_a, 3), 包含重叠区
        coords_b: chunk_b 的预测坐标 (L_b, 3), 包含重叠区
        overlap_indices: 重叠区在全链中的全局索引
        temperature: sigmoid 温度, 越小越严格

    Returns:
        OverlapConfidence 评估结果
    """
    coords_a = np.asarray(coords_a, dtype=np.float64)
    coords_b = np.asarray(coords_b, dtype=np.float64)
    overlap_indices = np.asarray(overlap_indices, dtype=np.int64)

    n_residues = len(overlap_indices)
    if n_residues < 3:
        # 重叠区太短, 无法可靠评估
        return OverlapConfidence(
            rmsd=0.0, confidence=1.0, is_flexible=False,
            force_scale=1.0, n_residues=n_residues,
        )

    # 提取重叠区坐标 — 这里假设传入的 coords 已经是对应 chunk 的坐标
    # overlap_indices 是全局索引, 但 coords_a/b 是各自 chunk 局部坐标
    # 调用方需要确保 overlap_indices 映射到正确的局部索引
    # 如果 overlap_indices 是全局索引且 chunk 从某个 start 开始:
    # 这里直接把 overlap_indices 当作两个 coords 中的有效索引
    # 实际使用中, 传入前应做映射

    # 如果 coords_a 和 coords_b 已经是对齐到同一重叠区的坐标:
    if coords_a.shape[0] == coords_b.shape[0]:
        ol_a = coords_a
        ol_b = coords_b
    else:
        # 按 overlap_indices 索引 (需要 coords 足够长)
        max_idx = max(overlap_indices.max() + 1, 0)
        if coords_a.shape[0] >= max_idx and coords_b.shape[0] >= max_idx:
            ol_a = coords_a[overlap_indices]
            ol_b = coords_b[overlap_indices]
        else:
            # 降级: 用 min 长度对齐
            common = min(coords_a.shape[0], coords_b.shape[0], n_residues)
            ol_a = coords_a[:common]
            ol_b = coords_b[:common]

    # Kabsch RMSD
    rmsd = _kabsch_rmsd(ol_a, ol_b)

    # 置信度: sigmoid 映射
    confidence = 1.0 / (1.0 + rmsd / max(temperature, 1e-6))

    # 柔性判定
    is_flexible = rmsd > FLEXIBLE_THRESHOLD

    # 力场缩放因子
    if rmsd < HIGH_CONF_THRESHOLD:
        force_scale = 1.0
    elif rmsd < FLEXIBLE_THRESHOLD:
        # 线性插值: 3A→1.0, 8A→0.5
        force_scale = 1.0 - 0.5 * (rmsd - HIGH_CONF_THRESHOLD) / (
            FLEXIBLE_THRESHOLD - HIGH_CONF_THRESHOLD
        )
    else:
        force_scale = 0.1

    return OverlapConfidence(
        rmsd=float(rmsd),
        confidence=float(confidence),
        is_flexible=is_flexible,
        force_scale=float(force_scale),
        n_residues=n_residues,
    )


def multi_sample_confidence(
    sequence: str,
    predict_fn: Callable[[str, int], np.ndarray],
    overlap_indices: np.ndarray,
    n_samples: int = 5,
    temperature: float = DEFAULT_CONFIDENCE_TEMP,
) -> OverlapConfidence:
    """多次采样评估重叠区一致性.

    对同一序列用不同 random seed 做 n_samples 次预测,
    计算重叠区坐标的标准差, 一致性高 → 高置信, 低 → 柔性区.

    原理: 如果一个区域结构确定 (如茎区), 多次预测应该收敛;
    如果是柔性区 (如环区), 多次预测会发散.

    Args:
        sequence: RNA 序列
        predict_fn: 预测函数 (sequence, seed) → (L, 3) 坐标
        overlap_indices: 重叠区全局索引
        n_samples: 采样次数
        temperature: sigmoid 温度

    Returns:
        OverlapConfidence 基于采样一致性的评估结果
    """
    overlap_indices = np.asarray(overlap_indices, dtype=np.int64)
    n_residues = len(overlap_indices)

    if n_residues < 3 or n_samples < 2:
        return OverlapConfidence(
            rmsd=0.0, confidence=1.0, is_flexible=False,
            force_scale=1.0, n_residues=n_residues,
        )

    # 多次采样
    samples: List[np.ndarray] = []
    for i in range(n_samples):
        seed = 42 + i  # 固定种子保证可复现, 不同 seed 间有差异
        coords = predict_fn(sequence, seed)
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape[0] > overlap_indices.max():
            samples.append(coords[overlap_indices])
        elif coords.shape[0] > 0:
            # 坐标不够长, 取有效部分
            valid = min(coords.shape[0], n_residues)
            samples.append(coords[:valid])

    if len(samples) < 2:
        return OverlapConfidence(
            rmsd=0.0, confidence=0.5, is_flexible=False,
            force_scale=0.5, n_residues=n_residues,
        )

    # 计算两两 RMSD
    rmsds: List[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            min_len = min(samples[i].shape[0], samples[j].shape[0])
            if min_len >= 3:
                rmsd = _kabsch_rmsd(samples[i][:min_len], samples[j][:min_len])
                rmsds.append(rmsd)

    if not rmsds:
        return OverlapConfidence(
            rmsd=0.0, confidence=0.5, is_flexible=False,
            force_scale=0.5, n_residues=n_residues,
        )

    # 用平均 RMSD 作为一致性指标
    mean_rmsd = float(np.mean(rmsds))
    confidence = 1.0 / (1.0 + mean_rmsd / max(temperature, 1e-6))
    is_flexible = mean_rmsd > FLEXIBLE_THRESHOLD

    if mean_rmsd < HIGH_CONF_THRESHOLD:
        force_scale = 1.0
    elif mean_rmsd < FLEXIBLE_THRESHOLD:
        force_scale = 1.0 - 0.5 * (mean_rmsd - HIGH_CONF_THRESHOLD) / (
            FLEXIBLE_THRESHOLD - HIGH_CONF_THRESHOLD
        )
    else:
        force_scale = 0.1

    return OverlapConfidence(
        rmsd=mean_rmsd,
        confidence=confidence,
        is_flexible=is_flexible,
        force_scale=force_scale,
        n_residues=n_residues,
    )


def segment_confidence_map(
    chunks: List[Dict],
    chunk_coords: List[np.ndarray],
    overlaps: List[Dict],
    full_length: int,
) -> np.ndarray:
    """生成逐残基置信度图.

    遍历所有 chunk 和重叠区, 计算每个残基的置信度:
    - 非重叠区: 使用 chunk 自身置信度 (取自 _score_chunk_quality)
    - 重叠区: 使用两 chunk 的 RMSD 映射置信度
    - 无 chunk 覆盖: 默认 0.0

    输出用于 Level 2.3 力场参数化: force_scale per residue.

    Args:
        chunks: 分段信息列表 (from split_sequence)
        chunk_coords: 每个 chunk 的 P 坐标
        overlaps: 重叠区信息列表, 每个含:
            - "indices": 重叠区全局索引
            - "chunk_a": 左 chunk 索引
            - "chunk_b": 右 chunk 索引
        full_length: 完整序列长度

    Returns:
        (full_length,) 逐残基置信度 [0, 1]
    """
    confidence = np.zeros(full_length, dtype=np.float64)
    weight = np.zeros(full_length, dtype=np.float64)

    # 1. 非重叠区: chunk 自身质量分
    for idx, (seg, coords) in enumerate(zip(chunks, chunk_coords)):
        start = seg["start"]
        end = seg["end"]
        length = end - start
        seg_coords = coords[:length] if len(coords) >= length else coords

        # 简单质量指标: 坐标方差越小越确定
        if seg_coords.shape[0] >= 3:
            # 用相邻残基距离的方差作为确定性指标
            diffs = np.diff(seg_coords, axis=0)
            bond_lengths = np.linalg.norm(diffs, axis=1)
            # 正常 P-P 键长 ~5.9A, 方差越小越确定
            if len(bond_lengths) > 1:
                bl_var = float(np.var(bond_lengths))
                # 映射: 方差 0→1.0, 方差 5→0.3
                chunk_conf = max(0.3, 1.0 - bl_var / 5.0)
            else:
                chunk_conf = 0.5
        else:
            chunk_conf = 0.5

        for i in range(start, min(start + len(seg_coords), full_length)):
            confidence[i] += chunk_conf
            weight[i] += 1.0

    # 2. 重叠区: 两 chunk RMSD → 置信度 (权重更高)
    for ol in overlaps:
        ol_indices = np.asarray(ol["indices"], dtype=np.int64)
        idx_a = ol["chunk_a"]
        idx_b = ol["chunk_b"]

        if idx_a >= len(chunk_coords) or idx_b >= len(chunk_coords):
            continue

        coords_a = chunk_coords[idx_a]
        coords_b = chunk_coords[idx_b]
        seg_a = chunks[idx_a]
        seg_b = chunks[idx_b]

        # 映射全局重叠索引到 chunk 局部坐标
        ol_local_a = ol_indices - seg_a["start"]
        ol_local_b = ol_indices - seg_b["start"]

        # 提取重叠区坐标
        valid_a = (ol_local_a >= 0) & (ol_local_a < coords_a.shape[0])
        valid_b = (ol_local_b >= 0) & (ol_local_b < coords_b.shape[0])
        valid = valid_a & valid_b

        if valid.sum() < 3:
            continue

        ol_coords_a = coords_a[ol_local_a[valid]]
        ol_coords_b = coords_b[ol_local_b[valid]]

        ol_conf = evaluate_overlap_confidence(
            ol_coords_a, ol_coords_b,
            np.arange(valid.sum()),
        )

        # 重叠区置信度用 2x 权重覆盖
        for i, global_idx in enumerate(ol_indices[valid]):
            confidence[global_idx] = ol_conf.confidence
            weight[global_idx] = 2.0  # 重叠区权重更高

    # 归一化
    weight = np.maximum(weight, 1e-8)
    confidence = confidence / weight

    return np.clip(confidence, 0.0, 1.0)


def force_scale_from_confidence(confidence: float) -> float:
    """从置信度分数映射到力场缩放因子.

    Args:
        confidence: [0, 1] 置信度

    Returns:
        force_scale: 0.1 (柔性) ~ 1.0 (高置信)
    """
    if confidence >= 0.75:
        return 1.0
    elif confidence >= 0.3:
        # 线性插值: 0.3→0.5, 0.75→1.0
        return 0.5 + 0.5 * (confidence - 0.3) / 0.45
    else:
        return 0.1
