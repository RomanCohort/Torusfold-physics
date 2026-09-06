# -*- coding: utf-8 -*-
"""trrna2_calibrator.py — trRNA2 距离矩阵校准 RhoFold 坐标输出。

核心思路:
  trRNA2 输出距离矩阵 (L×L)，RhoFold 输出 3D 坐标。
  用 trRNA2 的距离矩阵作为约束，通过最小化距离差异来校准 RhoFold 坐标。

公开 API:
  calibrate_with_distance_matrix() — 坐标优化校准
  trrna2_rhofold_ensemble()       — 双引擎集成预测
  integrated_predict_chunk()      — 统一接口
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ── 常量 ──
P_P_BOND = 5.9          # P-P 键长 (Angstrom)
WC_TARGET_DIST = 20.0   # Watson-Crick C1'-C1' 目标距离
WC_TOLERANCE = 3.0      # WC 距离容差
MAX_BOND_DEVIATION = 1.5  # 键长最大偏差 (Angstrom)


# ── 数据类 ──
@dataclass
class CalibrationResult:
    """校准结果。"""
    coords: np.ndarray          # (L, 3) 校准后坐标
    initial_coords: np.ndarray  # (L, 3) 原始 RhoFold 坐标
    dist_matrix: np.ndarray     # (L, L) trRNA2 距离矩阵
    rmse_before: float          # 校准前距离 RMSE
    rmse_after: float           # 校准后距离 RMSE
    n_iterations: int           # 实际迭代次数
    success: bool               # 优化是否成功
    confidence: float           # 校准置信度


# ── 工具函数 ──

def _pairwise_distance(coords: np.ndarray) -> np.ndarray:
    """计算 pairwise 距离矩阵 (L, L)。"""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def _find_wc_pairs_from_ss(ss: str) -> List[Tuple[int, int]]:
    """从二级结构字符串找 WC 配对位置 (0-based)。

    标准括号匹配: ( 与 ) 配对, 支持嵌套。
    """
    pairs = []
    stack = []
    for i, ch in enumerate(ss):
        if ch == '(':
            stack.append(i)
        elif ch == ')' and stack:
            j = stack.pop()
            pairs.append((j, i))
    return pairs


def _find_wc_pairs(sequence: str) -> List[Tuple[int, int]]:
    """从序列中推断 WC 配对 (仅当序列为 dot-bracket 格式时)。

    如果序列全是 ATGCNU 等碱基, 返回空列表。
    """
    if any(ch in '().' for ch in sequence):
        return _find_wc_pairs_from_ss(sequence)
    return []


def _rmse_distance(coords: np.ndarray, target_dist: np.ndarray) -> float:
    """计算坐标距离矩阵与目标距离矩阵的 RMSE。"""
    pred_dist = _pairwise_distance(coords)
    return float(np.sqrt(np.mean((pred_dist - target_dist) ** 2)))


# ── 核心校准 ──

def calibrate_with_distance_matrix(
    coords_rhofold: np.ndarray,
    dist_trrna2: np.ndarray,
    sequence: str,
    secondary_structure: Optional[str] = None,
    n_iterations: int = 200,
    bond_weight: float = 200.0,
    wc_weight: float = 50.0,
    dist_weight: float = 1.0,
    verbose: bool = False,
) -> CalibrationResult:
    """用 trRNA2 距离矩阵校准 RhoFold 坐标。

    通过 scipy.optimize.minimize 优化坐标，使 |d_ij(coords) - dist_trrna2[i,j]|
    最小化，同时保持骨架键长和二级结构约束。

    Args:
        coords_rhofold: (L, 3) RhoFold 预测坐标 (Angstrom)
        dist_trrna2: (L, L) trRNA2 预测距离矩阵 (Angstrom)
        sequence: RNA 序列
        secondary_structure: 二级结构 (dot-bracket). None 时从序列括号推断.
        n_iterations: 最大优化迭代次数
        bond_weight: 键长约束权重
        wc_weight: WC 配对约束权重
        dist_weight: 距离矩阵拟合权重
        verbose: 打印优化过程

    Returns:
        CalibrationResult 包含校准后坐标和质量指标
    """
    L = len(sequence)
    if coords_rhofold.shape != (L, 3):
        raise ValueError(f"coords shape {coords_rhofold.shape} != ({L}, 3)")
    if dist_trrna2.shape != (L, L):
        raise ValueError(f"dist shape {dist_trrna2.shape} != ({L}, {L})")

    # 找 WC 配对
    if secondary_structure:
        wc_pairs = _find_wc_pairs_from_ss(secondary_structure)
    else:
        wc_pairs = _find_wc_pairs(sequence)

    # 计算初始 RMSE
    rmse_before = _rmse_distance(coords_rhofold, dist_trrna2)

    # 如果已经很准, 直接返回避免退化
    if rmse_before < 0.5:
        return CalibrationResult(
            coords=coords_rhofold.copy(),
            initial_coords=coords_rhofold.copy(),
            dist_matrix=dist_trrna2.copy(),
            rmse_before=rmse_before,
            rmse_after=rmse_before,
            n_iterations=0,
            success=True,
            confidence=0.95,
        )

    # 目标键长 (相邻残基 P-P 距离)
    bond_targets = np.full(L - 1, P_P_BOND)

    # WC 目标距离
    wc_targets = np.array([WC_TARGET_DIST] * len(wc_pairs))

    def objective_stage1(x):
        """Stage 1: 距离矩阵拟合 (无约束)。"""
        coords = x.reshape(L, 3)
        pred_dist = _pairwise_distance(coords)
        return np.mean((pred_dist - dist_trrna2) ** 2)

    def objective_stage2(x):
        """Stage 2: 约束精修 (距离 + 键长 + WC)。"""
        coords = x.reshape(L, 3)

        # 1. 距离矩阵拟合
        pred_dist = _pairwise_distance(coords)
        dist_loss = np.mean((pred_dist - dist_trrna2) ** 2)

        # 2. 键长约束 (相邻残基) — 4次方惩罚
        bond_dists = np.sqrt(np.sum((coords[1:] - coords[:-1]) ** 2, axis=1))
        bond_dev = bond_dists - bond_targets
        bond_loss = np.mean(bond_dev ** 4)

        # 3. WC 配对约束
        wc_loss = 0.0
        if wc_pairs:
            wc_dists = np.array([
                np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
                for i, j in wc_pairs
            ])
            wc_dev = wc_dists - wc_targets
            wc_loss = np.mean(wc_dev ** 4)

        return dist_weight * dist_loss + bond_weight * bond_loss + wc_weight * wc_loss

    # L-BFGS-B 带边界约束
    # L-BFGS-B 边界
    bounds = []
    for i in range(L):
        for j in range(3):
            center = coords_rhofold[i, j]
            bounds.append((center - 30.0, center + 30.0))

    x0 = coords_rhofold.flatten().copy()

    # Stage 1: 距离矩阵拟合 (快速收敛到大致位置)
    s1_iters = max(n_iterations // 2, 50)
    result1 = minimize(
        objective_stage1, x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': s1_iters, 'ftol': 1e-6, 'gtol': 1e-4},
    )

    # Stage 2: 约束精修 (在 stage1 结果上施加键长/WC 约束)
    s2_iters = max(n_iterations // 2, 50)
    result2 = minimize(
        objective_stage2, result1.x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': s2_iters, 'ftol': 1e-6, 'gtol': 1e-4},
    )

    calibrated_coords = result2.x.reshape(L, 3)
    rmse_after = _rmse_distance(calibrated_coords, dist_trrna2)
    total_nit = result1.nit + result2.nit

    # 置信度: RMSE 改善比例
    if rmse_before > 1e-6:
        improvement = 1.0 - (rmse_after / rmse_before)
        confidence = min(0.95, max(0.1, 0.5 + 0.5 * improvement))
    else:
        improvement = 0.0
        confidence = 0.9

    if verbose:
        print(f"  [Calibrate] RMSE: {rmse_before:.2f} -> {rmse_after:.2f} "
              f"({improvement*100:.1f}% improved), iter={total_nit}, "
              f"conf={confidence:.3f}")

    return CalibrationResult(
        coords=calibrated_coords,
        initial_coords=coords_rhofold.copy(),
        dist_matrix=dist_trrna2.copy(),
        rmse_before=rmse_before,
        rmse_after=rmse_after,
        n_iterations=total_nit,
        success=result2.success,
        confidence=confidence,
    )


# ── 双引擎集成 ──

def trrna2_rhofold_ensemble(
    sequence: str,
    secondary_structure: Optional[str] = None,
    msa_path: Optional[str] = None,
    n_trrna2_samples: int = 1,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    verbose: bool = False,
    device: str = "auto",
) -> Tuple[np.ndarray, float, float]:
    """先用 RhoFold 跑一次得坐标，再用 trRNA2 跑得距离矩阵，校准后输出。

    Args:
        sequence: RNA 序列
        secondary_structure: 二级结构 (dot-bracket)
        msa_path: MSA 文件路径 (可选)
        n_trrna2_samples: trRNA2 采样次数 (取平均距离矩阵)
        output_dir: 输出目录
        name: 输出名称前缀
        verbose: 打印详细信息
        device: 设备 ("auto", "cuda", "cpu")

    Returns:
        (coords, confidence, dist_quality)
        coords: (L, 3) 校准后坐标
        confidence: 校准置信度
        dist_quality: 距离矩阵质量指标 (0-1)
    """
    from .rhofold_wrapper import rhofold_predict_chunk
    from .trrna2_wrapper import trrna2_predict_chunk

    L = len(sequence)

    # Step 1: RhoFold 预测
    if verbose:
        print("  [Ensemble] Step 1: RhoFold+ prediction...")

    rf_coords, rf_conf = rhofold_predict_chunk(
        sequence, secondary_structure,
        output_dir=output_dir, name=f"{name}_rhofold" if name else None,
        msa_path=msa_path, verbose=verbose, device=device,
    )

    # Step 2: trRNA2 距离矩阵 (多次采样取平均)
    if verbose:
        print(f"  [Ensemble] Step 2: trRNA2 distance matrix ({n_trrna2_samples} samples)...")

    dist_accum = None
    tr_conf_accum = 0.0

    for s in range(n_trrna2_samples):
        tr_result = trrna2_predict_chunk(
            sequence,
            output_dir=output_dir,
            name=f"{name}_tr{s}" if name else f"tr{s}",
            num_recycles=3,
            verbose=verbose,
        )

        if tr_result.dist is not None:
            if dist_accum is None:
                dist_accum = tr_result.dist.copy()
            else:
                dist_accum += tr_result.dist
            tr_conf_accum += tr_result.confidence

    if dist_accum is None:
        # trRNA2 未返回距离矩阵, 退化为 RhoFold 独立预测
        if verbose:
            print("  [Ensemble] trRNA2 no distance matrix, using RhoFold alone")
        return rf_coords, rf_conf, 0.0

    dist_avg = dist_accum / n_trrna2_samples
    tr_conf_avg = tr_conf_accum / n_trrna2_samples

    # Step 3: 校准
    if verbose:
        print("  [Ensemble] Step 3: Distance matrix calibration...")

    cal_result = calibrate_with_distance_matrix(
        rf_coords, dist_avg, sequence, secondary_structure,
        verbose=verbose,
    )

    # 综合置信度: RhoFold + trRNA2 + 校准改善
    combined_conf = 0.4 * rf_conf + 0.3 * tr_conf_avg + 0.3 * cal_result.confidence

    # 距离质量: RMSE 改善比例
    if cal_result.rmse_before > 0:
        dist_quality = min(1.0, 1.0 - cal_result.rmse_after / cal_result.rmse_before)
    else:
        dist_quality = 0.5

    if verbose:
        print(f"  [Ensemble] Done: conf={combined_conf:.3f}, "
              f"dist_quality={dist_quality:.3f}")

    return cal_result.coords, combined_conf, dist_quality


# ── 统一接口 ──

def integrated_predict_chunk(
    sequence: str,
    secondary_structure: Optional[str] = None,
    msa_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    name: Optional[str] = None,
    use_calibration: bool = True,
    n_trrna2_samples: int = 1,
    verbose: bool = False,
    device: str = "auto",
    boundary_pairs: Optional[List[Tuple[int, int, str]]] = None,
) -> Tuple[np.ndarray, float, dict]:
    """统一预测接口，同时调 RhoFold 和 trRNA2 进行校准。

    兼容 segmented_vfold3d.py 的调用方式。

    Args:
        sequence: RNA 序列
        secondary_structure: 二级结构 (dot-bracket)
        msa_path: MSA 文件路径 (可选)
        output_dir: 输出目录
        name: 输出名称前缀
        use_calibration: 是否启用 trRNA2 校准 (False 时退化为纯 RhoFold)
        n_trrna2_samples: trRNA2 采样次数
        verbose: 打印详细信息
        device: 设备
        boundary_pairs: Level 1 边界约束对列表

    Returns:
        (coords, confidence, metadata)
        coords: (L, 3) C1' 坐标 (Angstrom)
        confidence: 综合置信度 [0, 1]
        metadata: dict 包含详细信息
            - method: 使用的方法
            - rmse_before: 校准前 RMSE (仅校准模式)
            - rmse_after: 校准后 RMSE (仅校准模式)
            - dist_quality: 距离质量 (仅校准模式)
    """
    metadata = {"method": "rhofold"}

    if not use_calibration:
        # 纯 RhoFold 模式
        from .rhofold_wrapper import rhofold_predict_chunk
        coords, conf = rhofold_predict_chunk(
            sequence, secondary_structure,
            output_dir=output_dir, name=name,
            msa_path=msa_path, verbose=verbose, device=device,
            boundary_pairs=boundary_pairs,
        )
        metadata["method"] = "rhofold_only"
        return coords, conf, metadata

    # 校准模式
    try:
        coords, conf, dist_quality = trrna2_rhofold_ensemble(
            sequence, secondary_structure,
            msa_path=msa_path,
            n_trrna2_samples=n_trrna2_samples,
            output_dir=output_dir, name=name,
            verbose=verbose, device=device,
        )
        metadata["method"] = "rhofold+trrna2_calibrated"
        metadata["dist_quality"] = dist_quality
    except Exception as e:
        if verbose:
            print(f"  [Integrated] Calibration failed: {e}, falling back to RhoFold")
        from .rhofold_wrapper import rhofold_predict_chunk
        coords, conf = rhofold_predict_chunk(
            sequence, secondary_structure,
            output_dir=output_dir, name=name,
            msa_path=msa_path, verbose=verbose, device=device,
            boundary_pairs=boundary_pairs,
        )
        metadata["method"] = "rhofold_fallback"
        metadata["fallback_reason"] = str(e)

    # Level 1 边界约束弛豫 (校准后仍可应用)
    if boundary_pairs and metadata["method"] != "rhofold_only":
        try:
            from .boundary_constraints import apply_boundary_constraints_to_coords
            coords = apply_boundary_constraints_to_coords(coords, boundary_pairs)
            if verbose:
                print(f"  [Boundary] Applied {len(boundary_pairs)} boundary constraints post-calibration")
        except Exception as e:
            if verbose:
                print(f"  [Boundary] Post-calibration constraint failed: {e}")

    return coords, conf, metadata
