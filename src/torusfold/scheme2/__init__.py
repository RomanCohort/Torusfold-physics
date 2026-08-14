"""
TorusFold-Scheme2 — circRNA 3D 结构预测 (IsRNAcirc 架构 + RL 增强)

核心架构 (对标 IsRNAcirc):
  序列 → ViennaRNA 二级结构 → pair_graph 补远端配对
  → [RL MCTS 优化初始构象] (方案3)
  → 3-bead CG 分段折叠 (茎区A-form/环区松散 + RL远端引导力)
  → [RL 远端配对引导退火] (方案1)
  → 1EHZ 全原子重建 → Amber14 OL3 精修

改进 (v2):
  1. 修饰感知输入: 支持 m6A, Ψ, m1A, 2'-O-Me, m5C
  2. 物理弛豫后处理: 强制键长/键角/碰撞约束
  3. 不确定性估计: 基于结构指标的置信度

IsRNAcirc 关键参数:
  - 10 副本 REMD 280-460K (我们 8 副本 300-460K)
  - BSJ k 渐进 0.001→5.0 kcal/mol/Å² (我们 0.1→5.0)
  - 配对退火 0.01→0.1→1.0 (三步)
  - 100ns MD (我们 ~0.4ns)
"""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .refine import (
    vienna_pair_probs,
    scheme2_initial_coords,
    openmm_refine,
    predict_3d,
    BOND_LEN,
    PAIR_DIST,
    CLASH_DIST,
)
from .amber_refine import amber_refine
from .rl_optimizer import optimize_far_pairs
from .pair_graph import (
    parse_case_annotation,
    build_full_pair_graph,
    extract_stem_blocks,
)

# RL 默认权重路径
_DEFAULT_RL_POLICY_PATH = str(Path(__file__).resolve().parents[3]
                              / "data" / "bc_policy_big.pt")
_DEFAULT_DPO_POLICY_PATH = str(Path(__file__).resolve().parents[3]
                               / "data" / "dpo_policy_v3_compat.pt")

# 统计势路径
_STAT_POT_PATH = str(Path(__file__).resolve().parents[3]
                     / "data" / "cg_statistical_potential.pkl")

__all__ = [
    "predict_3d_allatom",
    "predict_3d_allatom_v2",
    "BOND_LEN",
    "PAIR_DIST",
    "CLASH_DIST",
]


def predict_3d_allatom(
    sequence: str,
    *,
    pair_threshold: float = 0.5,
    platform_name: str = "CPU",
    max_iterations: int = 3000,
    use_3bead: bool = True,
    use_rl: bool = False,
    use_rest2: bool = False,
    rl_policy_path: str = None,
    rl_n_simulations: int = 50,
    rl_use_defaults: bool = True,
    rl_dpo_weight: float = 0.0,
    rl_dpo_policy_path: str = None,
    rl_dpo_rollout: bool = False,
    rl_dpo_simulate: bool = False,
    coding_mask=None,
    # 新增参数 (v2)
    known_modifications: Optional[List[Dict]] = None,
    use_relaxation: bool = True,
    relaxation_steps: int = 5000,
):
    """端到端: 序列 → 全原子 RNA 结构 (IsRNAcirc 架构).

    链路 (方案1+3):
      ViennaRNA 配对 → pair_graph 补远端
      → [RL MCTS 优化初始构象] (方案3: RL 先拉远端)
      → 3-bead CG 分段折叠 + [RL 远端引导力] (方案1: RL 指导物理)
      → 1EHZ 全原子重建 → Amber14 OL3 精修

    Args:
        use_3bead: True=3-bead 分段折叠 (默认), False=旧1-bead (兼容)
        use_rl: 启用 RL 远端配对优化
        use_rest2: 启用 REST2 增强采样 (8副本 T-REMD)
        rl_use_defaults: 自动加载训练好的 BC prior + DPO 打分器
        known_modifications: 已知修饰位点列表 (可选)
        use_relaxation: 启用物理弛豫后处理 (默认 True)
        relaxation_steps: 物理弛豫步数 (默认 5000)

    Returns:
        dict: 包含结构坐标、能量、指纹等信息
    """
    pairs, bpp = vienna_pair_probs(sequence, pair_threshold)

    # ── 修饰感知: 检测/编码化学修饰 ──
    modification_features = None
    detected_modifications = []
    if known_modifications is not None or len(sequence) < 1000:
        try:
            from .modification_aware import detect_modifications, encode_modifications
            detected_modifications = detect_modifications(sequence, known_modifications=known_modifications)
            modification_features = encode_modifications(sequence, detected_modifications)
            print(f"  检测到 {len(detected_modifications)} 个修饰位点")
        except Exception as e:
            print(f"  [WARN] Modification detection failed: {e}")

    # ── 方案 1+3: RL 先优化初始构象 + 远端配对引导物理退火 ──
    rl_info = None
    far_pairs = []
    cg_coords_for_amber = None

    if use_rl:
        if rl_use_defaults:
            if rl_policy_path is None:
                rl_policy_path = _DEFAULT_RL_POLICY_PATH
            if rl_dpo_policy_path is None and rl_dpo_weight <= 0:
                rl_dpo_policy_path = _DEFAULT_DPO_POLICY_PATH
                rl_dpo_weight = 5.0
            if not rl_dpo_rollout and not rl_dpo_simulate:
                rl_dpo_simulate = True
        # pair_graph: 补 ViennaRNA 漏掉的长程配对 + 标记远端配对
        _, scan_pairs, far_pairs = build_full_pair_graph(
            sequence, pairs, do_scan=True,
        )
        stem_blocks = extract_stem_blocks(pairs, scan_pairs)

    if use_3bead:
        # ── 3-bead 管线: cg_forcefield 全残基力场 (已验证 2.4A RMSD) ──
        from .cg_forcefield import refine_3bead
        L = len(sequence)
        p_init = scheme2_initial_coords(sequence, pairs, n_samples=8)
        if p_init is None:
            raise RuntimeError(f"Scheme2 CG 几何求解失败 (L={L})")

        # 方案 3: RL 先优化初始构象, 作为物理退火起点
        if use_rl and far_pairs:
            opt_p, cg_orig, rl_info = optimize_far_pairs(
                p_init, sequence, far_pairs, stem_blocks,
                policy_path=rl_policy_path,
                n_simulations=rl_n_simulations,
                coding_mask=coding_mask,
                dpo_weight=rl_dpo_weight,
                dpo_policy_path=rl_dpo_policy_path,
                dpo_rollout=rl_dpo_rollout,
                dpo_simulate=rl_dpo_simulate,
            )
            p_init = opt_p
            cg_coords_for_amber = cg_orig
        elif use_rl:
            rl_info = {"skipped": True, "reason": "no_far_pairs"}

        # 3-bead CG 折叠 (cg_forcefield.refine_3bead)
        cg_coords, e0_cg, e1_cg = refine_3bead(
            p_init, pairs, platform_name, n_anneal=200,
            stat_pot_path=_STAT_POT_PATH if Path(_STAT_POT_PATH).exists() else None,
            sequence=sequence)
    else:
        # 旧 1-bead 管线 (兼容)
        init = scheme2_initial_coords(sequence, pairs, n_samples=8)
        if init is None:
            raise RuntimeError(f"Scheme2 CG 几何求解失败 (L={len(sequence)})")
        cg_coords, e0_cg, e1_cg = openmm_refine(init, pairs, platform_name)

    if cg_coords_for_amber is None:
        cg_coords_for_amber = cg_coords

    # ── 物理弛豫后处理: 强制键长/键角/碰撞约束 ──
    relaxation_metrics = None
    if use_relaxation:
        try:
            from .physical_relaxation import relax_structure
            cg_coords, relaxation_metrics = relax_structure(
                cg_coords, sequence,
                far_pairs=far_pairs if far_pairs else None,
                n_steps=relaxation_steps,
                use_openmm=True,
            )
            print(f"  物理弛豫: 碰撞 {relaxation_metrics['initial']['clash_count']} → "
                  f"{relaxation_metrics['final']['clash_count']}")
        except Exception as e:
            print(f"  [WARN] Physical relaxation failed: {e}")

    # ── 全原子重建 + Amber 精修 ──
    from .aform_from_template import reconstruct_all_atom as reconstruct_from_template
    structure = reconstruct_from_template(cg_coords, sequence)

    cg_coords_nm = None
    if use_rl and cg_coords_for_amber is not None:
        cg_coords_nm = np.asarray(cg_coords_for_amber, dtype=np.float64) / 10.0
    coords_aa, e0_aa, e1_aa, amber_info = amber_refine(
        structure, pairs,
        platform_name=platform_name,
        max_iterations=max_iterations,
        coding_mask=coding_mask,
        cg_coords=cg_coords_nm,
    )

    # 结构指纹 + 信号
    from .immune_heuristic import (
        compute_immune_fingerprints, compute_structure_signals,
    )
    bsj_dist = float(np.linalg.norm(cg_coords[0] - cg_coords[-1]))
    immune_fingerprints = compute_immune_fingerprints(
        coords_aa, structure, pairs, sequence,
    )
    structure_signals = compute_structure_signals(
        coords_aa, structure, pairs, bpp, sequence,
        e1_aa=e1_aa, bsj_dist=bsj_dist, cg_coords=cg_coords,
    )

    # ── 不确定性估计 ──
    uncertainty = _estimate_uncertainty(
        cg_coords, pairs, far_pairs, bsj_dist, relaxation_metrics,
    )

    return {
        "coords_cg": cg_coords,
        "pairs": pairs,
        "pair_probs": bpp,
        "e0_cg": e0_cg,
        "e1_cg": e1_cg,
        "atoms": structure,
        "coords_aa": coords_aa,
        "e0_aa": e0_aa,
        "e1_aa": e1_aa,
        "amber_info": amber_info,
        "rl_info": rl_info,
        "structure_method": "scheme2_allatom",
        "available": True,
        "immune_fingerprints": immune_fingerprints,
        "structure_signals": structure_signals,
        # 新增 (v2)
        "detected_modifications": detected_modifications,
        "modification_features": modification_features,
        "relaxation_metrics": relaxation_metrics,
        "uncertainty": uncertainty,
    }


def _estimate_uncertainty(
    cg_coords: np.ndarray,
    pairs: List[Tuple[int, int]],
    far_pairs: List[Tuple[int, int]],
    bsj_dist: float,
    relaxation_metrics: Optional[Dict],
) -> float:
    """估计预测不确定性 [0, 1].

    基于多个指标:
    1. BSJ 闭合距离偏差
    2. 远端配对距离偏差
    3. 碰撞数量
    4. 键长违规数

    Args:
        cg_coords: CG P 坐标
        pairs: 配对列表
        far_pairs: 远端配对列表
        bsj_dist: BSJ 闭合距离
        relaxation_metrics: 弛豫指标

    Returns:
        uncertainty [0, 1]: 0=高置信, 1=低置信
    """
    uncertainties = []

    # 1. BSJ 闭合偏差 (理想 ~5.9A)
    bsj_deviation = abs(bsj_dist - BOND_LEN) / BOND_LEN
    uncertainties.append(min(1.0, bsj_deviation))

    # 2. 远端配对距离偏差 (理想 ~20A)
    if far_pairs:
        wc_devs = []
        for i, j in far_pairs:
            if i < len(cg_coords) and j < len(cg_coords):
                d = np.linalg.norm(cg_coords[i] - cg_coords[j])
                wc_devs.append(abs(d - PAIR_DIST) / PAIR_DIST)
        if wc_devs:
            uncertainties.append(min(1.0, np.mean(wc_devs)))

    # 3. 碰撞数量
    if relaxation_metrics:
        final_clashes = relaxation_metrics.get("final", {}).get("clash_count", 0)
        uncertainties.append(min(1.0, final_clashes / 10.0))

    # 4. 键长违规
    if relaxation_metrics:
        bond_violations = relaxation_metrics.get("final", {}).get("bond_violations", 0)
        uncertainties.append(min(1.0, bond_violations / 20.0))

    return float(np.mean(uncertainties)) if uncertainties else 0.5
