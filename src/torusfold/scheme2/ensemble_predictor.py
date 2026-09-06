# -*- coding: utf-8 -*-
"""ensemble_predictor.py — 三预测器集成: RhoFold+ + trRNA2 + RNAbpFlow

方案 C: 加权融合 + 距离校准

策略:
  1. RhoFold+ → 坐标 (几何精度高)
  2. trRNA2 → 距离矩阵 (非典型配对强)
  3. RNAbpFlow → 坐标+距离 (生成式, ensemble)
  4. 距离共识: trRNA2 + RNAbpFlow 取平均
  5. 距离校准: 用共识距离校准 RhoFold 坐标
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class EnsembleResult:
    """三预测器集成结果."""
    coords: np.ndarray          # (L, 3) 校准后 P 坐标
    confidence: float           # 集成置信度
    dist_consensus: np.ndarray  # (L, L) 共识距离矩阵
    per_predictor: dict         # 各预测器独立结果
    bond_quality: float         # 键长质量 (越接近 5.9 越好)
    calibration_rmse: float     # 校准前后 RMSE 变化


def _bond_score(coords: np.ndarray) -> float:
    """评估骨架键长质量 (0-1). 理想值 5.9A."""
    if coords is None or len(coords) < 2:
        return 0.0
    diffs = np.diff(coords, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    # 高斯评分: 均值越接近 5.9A, 标准差越小 → 分越高
    mean_dev = abs(np.mean(dists) - 5.9) / 5.9
    std_dev = np.std(dists) / 5.9
    return max(0.0, 1.0 - mean_dev - std_dev)


def _extract_p_coords(allatom_coords: np.ndarray, n_res: int) -> np.ndarray:
    """从全原子坐标提取 P 原子 (atom 0 per residue)."""
    if allatom_coords is None:
        return None
    if allatom_coords.ndim == 3:
        # 格式: (n_atoms_per_res, L, 3) — trRNA2 输出
        return allatom_coords[0, :n_res].astype(np.float32)
    elif allatom_coords.ndim == 2 and allatom_coords.shape[1] == 3:
        # 已经是 (L, 3)
        return allatom_coords[:n_res].astype(np.float32)
    return None


# ── 区域类型动态权重 ──
# stem: WC配对密集, RhoFold+ 表现好
# bsj:  配对稀疏, 需要非典型配对信息
# loop: 非典型配对多, trRNA2/RNAbpFlow 有优势
REGION_WEIGHTS = {
    "stem": {"rhofold": 0.50, "trrna2": 0.20, "rnabpflow": 0.30},
    "bsj":  {"rhofold": 1.00, "trrna2": 0.00, "rnabpflow": 0.10},  # 坐标只用RhoFold, 但距离约束用RNAbpFlow
    "loop": {"rhofold": 0.30, "trrna2": 0.40, "rnabpflow": 0.30},
}

# auto 模式密度阈值
_DENSITY_HIGH = 0.3   # > 0.3 → stem
_DENSITY_LOW = 0.1    # < 0.1 → bsj
# 其他 → loop


def _estimate_pairing_density(
    sequence: str,
    secondary_structure: str = None,
) -> float:
    """估计序列区域的配对密度 (pairs / length).

    优先用二级结构; 无 ss 时用 ViennaRNA bpp 兜底.

    Args:
        sequence: RNA 序列
        secondary_structure: dot-bracket, 可选

    Returns:
        配对密度 [0, 1]
    """
    L = len(sequence)
    if L == 0:
        return 0.0

    n_pairs = 0
    if secondary_structure:
        n_pairs = sum(1 for ch in secondary_structure if ch in "()") // 2
    else:
        # ViennaRNA bpp 兜底
        try:
            import RNA
            ss, mfe = RNA.fold(sequence)
            n_pairs = sum(1 for ch in ss if ch in "()") // 2
        except Exception:
            return 0.0

    return n_pairs / max(L, 1)


def _classify_region(
    pairing_density: float,
    bsj_aware: bool = False,
) -> str:
    """根据配对密度和 BSJ 标记分类区域类型.

    Args:
        pairing_density: 配对密度 [0, 1]
        bsj_aware: chunk 是否在 BSJ 附近

    Returns:
        "stem", "bsj", 或 "loop"
    """
    if bsj_aware:
        return "bsj"
    if pairing_density > _DENSITY_HIGH:
        return "stem"
    if pairing_density < _DENSITY_LOW:
        return "bsj"
    return "loop"


def _select_predictors(region_type, seq_len, gc_content):
    """根据区域类型和序列特征选择预测器。

    Returns: dict of {predictor_name: enabled}
    """
    if region_type == "stem":
        # 高GC stem: 优先RhoFold+ (pLDDT高, 几何精度好)
        return {"rhofold": True, "trrna2": False, "rnabpflow": False}

    elif region_type == "bsj":
        # BSJ区: 优先RNAbpFlow距离矩阵 + RhoFold坐标骨架
        return {"rhofold": True, "trrna2": False, "rnabpflow": True}

    elif region_type == "loop":
        # loop: trRNA2 + RNAbpFlow (非典型配对多)
        if seq_len <= 200:
            # 短序列: 全部跑 (快)
            return {"rhofold": True, "trrna2": True, "rnabpflow": True}
        else:
            # 长序列: 跳过trRNA2 (CPU太慢), 用RhoFold+RNAbpFlow
            return {"rhofold": True, "trrna2": False, "rnabpflow": True}

    else:
        # auto/未知: 全部跑
        return {"rhofold": True, "trrna2": True, "rnabpflow": True}


def ensemble_predict(
    sequence: str,
    secondary_structure: str = None,
    output_dir: str = None,
    verbose: bool = True,
    use_rhofold: bool = True,
    use_trrna2: bool = True,
    use_rnabpflow: bool = True,
    n_rnabpflow_samples: int = 1,
    region_type: str = None,
) -> EnsembleResult:
    """三预测器集成预测.

    Args:
        sequence: RNA 序列
        secondary_structure: 二级结构 (dot-bracket), 可选
        output_dir: 输出目录
        verbose: 打印进度
        use_rhofold: 是否用 RhoFold+
        use_trrna2: 是否用 trRNA2
        use_rnabpflow: 是否用 RNAbpFlow
        n_rnabpflow_samples: RNAbpFlow 采样次数 (ensemble)
        region_type: 区域类型 ("stem"/"bsj"/"loop"/"auto"/None).
            None 或不传 → 退化为原始固定权重 (confidence*bond_score).
            "auto" → 根据配对密度自动判断.

    Returns:
        EnsembleResult
    """
    L = len(sequence)
    t_start = time.time()
    per_pred = {}

    # 预测器选择 (根据区域类型)
    if region_type:
        selected = _select_predictors(region_type, L, 0.0)
        use_rhofold = selected.get("rhofold", use_rhofold)
        use_trrna2 = selected.get("trrna2", use_trrna2)
        use_rnabpflow = selected.get("rnabpflow", use_rnabpflow)
        if verbose:
            enabled = [k for k, v in selected.items() if v]
            print(f"  [Ensemble] 区域={region_type}, 选择: {', '.join(enabled)}")

    # ── 1. RhoFold+ ──
    coords_rh = None
    conf_rh = 0.0
    if use_rhofold:
        try:
            if verbose:
                print("  [Ensemble] RhoFold+ ...")
            t0 = time.time()
            from .rhofold_wrapper import rhofold_predict_chunk
            coords_rh, conf_rh = rhofold_predict_chunk(
                sequence, verbose=False)
            t_rh = time.time() - t0
            per_pred["rhofold"] = {
                "coords": coords_rh, "confidence": conf_rh, "time": t_rh,
            }
            if verbose:
                print(f"    RhoFold+: {coords_rh.shape}, conf={conf_rh:.3f}, {t_rh:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    RhoFold+ 失败: {e}")
            # 降级: 尝试用 trRNA2 的 P 坐标替代
            if use_trrna2:
                try:
                    if verbose:
                        print("  [降级] RhoFold+ 失败, 尝试 trRNA2 P 坐标替代 ...")
                    from .trrna2_wrapper import trrna2_predict_chunk
                    tr_fb = trrna2_predict_chunk(sequence, name="fallback", num_recycles=3)
                    if tr_fb.coords is not None:
                        coords_rh = _extract_p_coords(tr_fb.coords, L)
                        if coords_rh is not None and len(coords_rh) == L:
                            conf_rh = tr_fb.confidence if tr_fb.confidence else 0.0
                            per_pred["rhofold"] = {
                                "coords": coords_rh, "confidence": conf_rh,
                                "time": 0.0, "fallback": "trrna2",
                            }
                            if verbose:
                                print(f"    [降级] 用 trRNA2 P 坐标替代 RhoFold+, conf={conf_rh:.3f}")
                        else:
                            if verbose:
                                print("    [降级] trRNA2 P 坐标长度不匹配, 放弃")
                    else:
                        if verbose:
                            print("    [降级] trRNA2 无坐标输出, 放弃")
                except Exception as e2:
                    if verbose:
                        print(f"    [降级] trRNA2 替代也失败: {e2}")

    # ── 2. trRNA2 ──
    dist_tr = None
    coords_tr_p = None
    if use_trrna2:
        try:
            if verbose:
                print("  [Ensemble] trRNA2 ...")
            t0 = time.time()
            from .trrna2_wrapper import trrna2_predict_chunk
            tr = trrna2_predict_chunk(sequence, name="ensemble", num_recycles=3)
            t_tr = time.time() - t0
            if tr.coords is not None:
                coords_tr_p = _extract_p_coords(tr.coords, L)
            dist_tr = tr.dist
            per_pred["trrna2"] = {
                "coords": coords_tr_p, "dist": dist_tr,
                "confidence": tr.confidence, "time": t_tr,
            }
            if verbose:
                shape_str = str(coords_tr_p.shape) if coords_tr_p is not None else "None"
                dist_str = str(dist_tr.shape) if dist_tr is not None else "None"
                print(f"    trRNA2: P={shape_str}, dist={dist_str}, {t_tr:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    trRNA2 失败: {e}")

    # ── 3. RNAbpFlow ──
    coords_rna = None
    dist_rna = None
    if use_rnabpflow:
        try:
            if verbose:
                print("  [Ensemble] RNAbpFlow ...")
            t0 = time.time()
            coords_rna, dist_rna = _run_rnabpflow(sequence, output_dir, verbose)
            t_rna = time.time() - t0
            per_pred["rnabpflow"] = {
                "coords": coords_rna, "dist": dist_rna, "time": t_rna,
            }
            if verbose:
                shape_str = str(coords_rna.shape) if coords_rna is not None else "None"
                dist_str = str(dist_rna.shape) if dist_rna is not None else "None"
                print(f"    RNAbpFlow: coords={shape_str}, dist={dist_str}, {t_rna:.1f}s")
        except Exception as e:
            if verbose:
                print(f"    RNAbpFlow 失败: {e}")

    # ── 4. 距离共识 ──
    dist_consensus = None
    if dist_tr is not None and dist_rna is not None:
        # 取平均 (两者长度可能不同, 取交集)
        L_min = min(dist_tr.shape[0], dist_rna.shape[0])
        dist_consensus = (dist_tr[:L_min, :L_min] + dist_rna[:L_min, :L_min]) / 2.0
        if verbose:
            rmse = np.sqrt(np.mean((dist_tr[:L_min, :L_min] - dist_rna[:L_min, :L_min]) ** 2))
            print(f"    距离共识: {dist_consensus.shape}, trRNA2<->RNAbp RMSE={rmse:.2f}A")
    elif dist_tr is not None:
        dist_consensus = dist_tr
    elif dist_rna is not None:
        dist_consensus = dist_rna

    # ── 5. 区域类型判断 ──
    effective_region = region_type
    if region_type == "auto":
        density = _estimate_pairing_density(sequence, secondary_structure)
        effective_region = _classify_region(density, bsj_aware=False)
        if verbose:
            print(f"    区域分类: auto → {effective_region} (density={density:.3f})")
    elif region_type is not None and region_type not in REGION_WEIGHTS:
        if verbose:
            print(f"    [WARN] 未知 region_type='{region_type}', 退化为固定权重")
        effective_region = None

    # ── 6. 坐标加权融合 ──
    if coords_rh is not None:
        final_coords = coords_rh.copy()
        all_coords = [coords_rh]

        if coords_tr_p is not None and len(coords_tr_p) == L:
            all_coords.append(coords_tr_p)

        if coords_rna is not None and len(coords_rna) == L:
            all_coords.append(coords_rna)

        if len(all_coords) > 1:
            if effective_region == "bsj":
                # BSJ: 坐标骨架直接用 RhoFold+, 跳过加权平均
                # RNAbpFlow 距离矩阵已保存到 dist_consensus, 供下游弱约束
                if verbose:
                    print(f"    坐标融合 [bsj]: 直接用 RhoFold+ 坐标骨架"
                          + (", RNAbpFlow 距离矩阵已传递" if dist_rna is not None else ""))
            elif effective_region is not None and effective_region in REGION_WEIGHTS:
                # 动态权重模式: 根据区域类型直接使用预定义权重,
                # 再乘以各预测器可用性 (不可用的权重重分配)
                rw = REGION_WEIGHTS[effective_region]
                raw_w = []
                avail_keys = []
                if coords_rh is not None:
                    raw_w.append(rw["rhofold"])
                    avail_keys.append("rhofold")
                if coords_tr_p is not None and len(coords_tr_p) == L:
                    raw_w.append(rw["trrna2"])
                    avail_keys.append("trrna2")
                if coords_rna is not None and len(coords_rna) == L:
                    raw_w.append(rw["rnabpflow"])
                    avail_keys.append("rnabpflow")
                weights = np.array(raw_w, dtype=np.float64)
                weights = weights / max(weights.sum(), 1e-8)
                if verbose:
                    print(f"    坐标融合 [{effective_region}]: {len(all_coords)} 个预测器, "
                          f"权重={weights.round(3)} (来源={avail_keys})")
            else:
                # 原始固定权重模式 (region_type=None 时退化)
                weights = []
                weights.append(conf_rh * _bond_score(coords_rh))
                if coords_tr_p is not None and len(coords_tr_p) == L:
                    w_tr = per_pred.get("trrna2", {}).get("confidence", 0.5) * _bond_score(coords_tr_p)
                    weights.append(w_tr)
                if coords_rna is not None and len(coords_rna) == L:
                    w_rna = 0.7 * _bond_score(coords_rna)
                    weights.append(w_rna)
                weights = np.array(weights, dtype=np.float64)
                weights = weights / max(weights.sum(), 1e-8)
                if verbose:
                    print(f"    坐标融合 [fixed]: {len(all_coords)} 个预测器, "
                          f"权重={weights.round(3)}")

            # 加权平均 (BSJ 已在上面跳过, final_coords 保持 coords_rh)
            if effective_region != "bsj":
                final_coords = np.zeros_like(coords_rh)
                for c, w in zip(all_coords, weights):
                    final_coords += c * w
    else:
        # 没有 RhoFold, 用其他
        final_coords = coords_rna if coords_rna is not None else (
            coords_tr_p if coords_tr_p is not None else np.zeros((L, 3)))

    # ── 7. 距离校准 ──
    calibration_rmse = 0.0
    if effective_region == "bsj":
        # BSJ: 跳过 ensemble 内部校准, 距离约束交给 segmented_vfold3d 下游弱约束
        if verbose:
            print(f"    [BSJ] 跳过 ensemble 内部距离校准, dist_consensus 传递到下游")
    elif dist_consensus is not None and len(final_coords) >= 2:
        try:
            from .trrna2_calibrator import calibrate_with_distance_matrix
            result = calibrate_with_distance_matrix(
                final_coords, dist_consensus, sequence, n_iterations=200)
            if result.rmse_before > 0:
                calibration_rmse = result.rmse_before - result.rmse_after
                final_coords = result.coords
                if verbose:
                    print(f"    距离校准: RMSE {result.rmse_before:.2f} -> {result.rmse_after:.2f}A"
                          f" (改善 {calibration_rmse:.2f}A)")
        except Exception as e:
            if verbose:
                print(f"    距离校准跳过: {e}")

    # ── 8. 全失败兜底: 几何初始化坐标 ──
    if final_coords is None or len(final_coords) == 0:
        if verbose:
            print("  [降级] 所有预测器失败, 用几何初始化坐标")
        L = len(sequence)
        coords_list = []
        for i in range(L):
            angle = 2 * np.pi * i / L
            r = L * 5.9 / (2 * np.pi)  # P-P bond 5.9A, 环形
            coords_list.append([r * np.cos(angle), r * np.sin(angle), 0.0])
        final_coords = np.array(coords_list, dtype=np.float32)
        confidence = 0.0
        bond_quality = 0.0

    # ── 9. 最终质量评估 ──
    bond_score = _bond_score(final_coords)
    confidence = min(1.0, (conf_rh + bond_score) / 2.0) if coords_rh is not None else bond_score

    total_time = time.time() - t_start
    if verbose:
        print(f"  [Ensemble] 完成: {total_time:.1f}s, "
              f"bond={bond_score:.3f}, conf={confidence:.3f}")

    return EnsembleResult(
        coords=final_coords,
        confidence=confidence,
        dist_consensus=dist_consensus,
        per_predictor=per_pred,
        bond_quality=bond_score,
        calibration_rmse=calibration_rmse,
    )


def _run_rnabpflow(
    sequence: str, output_dir: str, verbose: bool
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """调用 RNAbpFlow 推理 (subprocess)."""
    import subprocess
    import tempfile
    import pickle

    checkpoint = os.path.join(os.environ.get("RNABPFLOW_ROOT", ""), "checkpoint", "RNA3DB.ckpt")
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"RNAbpFlow checkpoint not found: {checkpoint}")

    # 创建 RNAbpFlow 输入目录结构 (Inputs/seq/ 下放 FASTA + map.pkl + npy)
    tmp_dir = os.path.join(tempfile.gettempdir(), "rnabpflow_input")
    inp_dir = os.path.join(tmp_dir, "Inputs", "seq")
    os.makedirs(inp_dir, exist_ok=True)

    # list.txt (必须在 Inputs/ 下)
    with open(os.path.join(tmp_dir, "Inputs", "list.txt"), "w") as f:
        f.write("seq 1\n")

    # FASTA (必须在 Inputs/seq/seq.fasta)
    with open(os.path.join(inp_dir, "seq.fasta"), "w") as f:
        f.write(f">seq\n{sequence}\n")

    # 写 map.pkl (needed by dataset)
    L = len(sequence)
    aa_map = {"A": 0, "U": 1, "G": 2, "C": 3}
    onehot = np.zeros((L, 5), dtype=np.float32)
    for i, ch in enumerate(sequence):
        onehot[i, aa_map.get(ch.upper(), 4)] = 1.0
    mapfeat = {
        "onehot": onehot,
        "native_chis": np.zeros((L, 5), dtype=np.float32),
        "dist_to_ft": np.ones((L, 1), dtype=np.float32),
        "map": np.zeros((L, L), dtype=np.float32),
    }
    with open(os.path.join(inp_dir, "map.pkl"), "wb") as f:
        pickle.dump(mapfeat, f)

    # 写 .npy maps
    for name in ["map1.npy", "map2.npy", "map3.npy"]:
        np.save(os.path.join(inp_dir, name), np.zeros((L, L), dtype=np.float32))

    # 调用 inference_rocm.py
    script = os.path.join(os.environ.get("RNABPFLOW_ROOT", ""), "inference_rocm.py")
    out_dir = os.path.join(output_dir or tempfile.gettempdir(), "rnabpflow_out")

    result = subprocess.run(
        [os.environ.get("RNABPFLOW_PYTHON", "python"),
         script, "--name", "seq", "--checkpoint", checkpoint, "--device", "cuda"],
        capture_output=True, text=True, timeout=None,
        cwd=tmp_dir,
    )

    # 读取输出 PDB
    pdb_path = os.path.join(tmp_dir, "Predictions", "seq", "Sample_0.pdb")
    if not os.path.exists(pdb_path):
        # 尝试其他路径
        for d in [tmp_dir, out_dir]:
            p = os.path.join(d, "Predictions", "seq", "Sample_0.pdb")
            if os.path.exists(p):
                pdb_path = p
                break

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"RNAbpFlow output not found: {pdb_path}")

    # 解析 PDB → P 坐标
    coords = _parse_pdb_p_coords(pdb_path, L)
    # 计算距离矩阵
    dist = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1).astype(np.float32)

    return coords, dist


def _parse_pdb_p_coords(pdb_path: str, expected_L: int) -> np.ndarray:
    """从 PDB 提取 P 原子坐标."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P  " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    coords = np.array(coords, dtype=np.float32)
    if len(coords) >= expected_L:
        return coords[:expected_L]
    # 如果 P 原子不够, 尝试每 21 个原子取一个
    if len(coords) > expected_L:
        step = len(coords) // expected_L
        return coords[::step][:expected_L].astype(np.float32)
    return coords
