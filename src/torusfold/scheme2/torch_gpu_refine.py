"""
torch_gpu_refine.py — GPU 加速精修 (替代 openmm_gpu_refiner).

用 BatchedREMD2D (torch GPU) 替代 OpenMM CPU REMD,
用 relax_structure (torch GPU) 替代 OpenMM 物理弛豫.

接口与 openmm_gpu_refine 完全兼容, 可直接替换.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def torch_gpu_refine(
    input_pdb: str,
    output_dir: str,
    sequence: str,
    secondary_structure: str,
    name: str = "refine",
    nstep: int = 100000,
    nstep_close: int = 1000,
    nstru: int = 3,
    platform_name: str = "auto",
    use_remd: bool = True,
    remd_n_replicas: int = 12,
    remd_n_steps: int = 100000,
    verbose: bool = True,
    skip_cg_to_allatom: bool = False,
    use_physical_relax: bool = True,
    bpp_matrix: Optional[np.ndarray] = None,
    bpp_weight: float = 0.5,
    use_multistage_remd: bool = True,
    use_potential_refine: bool = True,
    skip_minimal_fold: bool = False,
    use_trirnasp: bool = False,
    use_trirnasp_force: bool = False,
    trirnasp_energy_dir: str = None,
    trirnasp_scale: float = 0.002,  # 统一默认值: 0.002 (Tri/CG ≈ 11%)
    trirnasp_update_freq: int = 10,
    use_adaptive_tri_weight: bool = False,
    use_staged_tri: bool = False,
    tri_stage_config: Optional[dict] = None,
    lambdas: Optional[Tuple[float, ...]] = None,  # 新增: 自定义 λ 值
) -> Tuple[str, float, dict]:
    """torch GPU 加速精修 (openmm_gpu_refine 兼容接口).

    替代路径:
    1. 读 PDB → 提取 P 坐标 + 远端配对
    2. Pre-fold: 骨架键松弛 (400K→300K, 6级×2000步)
    3. BatchedREMD2D (torch GPU) 多轮 REMD (8轮×5000步)
    4. relax_structure (torch GPU) 物理弛豫
    5. CG → 全原子 (复用 isrnacirc_wrapper)
    6. 输出精修后 PDB

    Returns:
        (output_pdb_path, final_energy, diag_dict)
    """
    from .openmm_gpu_refiner import (
        _read_p_coords, _dotbracket_to_pairs,
        _sanitize_p_coords, _generate_compact_coords,
        BOND_P_NEXT,
    )
    from .torch_cgsim import BatchedREMD2D, cg_energy_forces
    from .physical_relaxation import relax_structure

    t0 = time.time()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 读 P 坐标
    p_coords = _read_p_coords(input_pdb)
    L = len(p_coords)
    if verbose:
        print(f"  [Torch GPU] 序列长度: {L} nt")

    if L < 3:
        raise ValueError(f"序列太短 ({L} nt)")

    p_coords = _sanitize_p_coords(p_coords)
    pairs = _dotbracket_to_pairs(secondary_structure)

    # bpp 远端配对
    if bpp_matrix is not None and bpp_matrix.shape[0] == L:
        from .openmm_gpu_refiner import discover_far_pairs_from_bpp
        far = discover_far_pairs_from_bpp(
            bpp_matrix, sequence, min_gap=24, bpp_threshold=0.01,
            top_k=50, existing_pairs=pairs)
        if far:
            pairs = pairs + far
            if verbose:
                print(f"  [bpp] 发现 {len(far)} 个远端配对, 总计 {len(pairs)} 对")

    # 单位检查
    if L > 1:
        avg_pp = float(np.mean(np.linalg.norm(
            p_coords[1:] - p_coords[:-1], axis=1)[:min(L - 1, 500)]))
        if avg_pp < 1.5:
            p_coords *= 10.0
            avg_pp *= 10.0
        use_compact = (not np.isfinite(avg_pp)) or avg_pp > 20.0 or avg_pp < 1.0
        if use_compact:
            if verbose:
                print(f"  [Torch GPU] P-P 键长异常 (avg={avg_pp:.2f}A), 生成紧凑坐标")
            p_coords = _generate_compact_coords(L, pairs)
    else:
        use_compact = False

    # 2. Pre-fold: 骨架键松弛 (400K→300K, 6级×2000步)
    final_e = float("inf")
    final_p_coords = p_coords.copy()

    if use_potential_refine and not skip_minimal_fold:
        if verbose:
            print(f"  [Torch GPU] 预折叠: 400K→300K, 6级×2000步")
        try:
            import torch
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            L = len(final_p_coords)
            # P-only → 3-bead (P, C4', N): C4' 偏移 +0.34nm 沿骨架方向, N 偏移 -0.15nm
            pos_p = torch.tensor(final_p_coords, dtype=torch.float64, device=dev) / 10.0
            # 骨架方向向量
            diffs = torch.zeros_like(pos_p)
            diffs[1:] = pos_p[1:] - pos_p[:-1]
            diffs[0] = diffs[1] if L > 1 else torch.zeros(3, device=dev)
            bb_dir = diffs / (diffs.norm(dim=1, keepdim=True).clamp(min=1e-6))
            pos_c4 = pos_p + bb_dir * 0.034   # C4' +0.34nm 沿骨架
            pos_n = pos_p + bb_dir * (-0.015)  # N -0.15nm 沿骨架
            # 交错排列: (P0, C4'0, N0, P1, C4'1, N1, ...)
            pos_3bead = torch.stack([pos_p, pos_c4, pos_n], dim=1).reshape(1, 3 * L, 3)

            pairs_t = torch.tensor([(i, j) for i, j, _ in pairs], dtype=torch.long, device=dev) if pairs else torch.zeros(0, 2, dtype=torch.long, device=dev)
            pw = torch.tensor([w for _, _, w in pairs], dtype=torch.float64, device=dev) if pairs else torch.zeros(0, dtype=torch.float64, device=dev)

            # 6级温度退火
            fold_temps = [400.0, 350.0, 325.0, 310.0, 300.0, 300.0]
            for stage, T in enumerate(fold_temps):
                pos_3bead.requires_grad_(True)
                opt = torch.optim.Adam([pos_3bead], lr=1e-3)
                for step in range(2000):
                    opt.zero_grad()
                    e, _ = cg_energy_forces(pos_3bead, pairs_t, pw)
                    if not torch.isfinite(e).all():
                        raise RuntimeError(f"预折叠第 {stage + 1} 级能量非有限")
                    e.sum().backward()
                    if pos_3bead.grad is None or not torch.isfinite(pos_3bead.grad).all():
                        raise RuntimeError(f"预折叠第 {stage + 1} 级梯度非有限")
                    opt.step()
                    pos_3bead.data.clamp_(-1.0, 10.0)
                    if not torch.isfinite(pos_3bead).all():
                        raise RuntimeError(f"预折叠第 {stage + 1} 级坐标非有限")
                pos_3bead = pos_3bead.detach()

            # 提取 P 坐标 (每3个珠子取第0个)
            with torch.no_grad():
                final_e_fold, _ = cg_energy_forces(pos_3bead, pairs_t, pw)
                final_e_fold = final_e_fold.item()
            final_p_coords = pos_3bead[:, 0::3, :].squeeze(0).cpu().numpy() * 10.0  # nm→Å
            if not np.isfinite(final_e_fold) or not np.all(np.isfinite(final_p_coords)):
                raise RuntimeError("预折叠结果含非有限能量或坐标")

            if verbose:
                print(f"  [Torch GPU] 预折叠完成: E={final_e_fold:.0f}")
            final_e = final_e_fold
        except Exception as e:
            if verbose:
                print(f"  [Torch GPU] 预折叠失败: {e}")

    # 3. 多轮 REMD (8轮×5000步, 每轮重起 Langevin)
    if use_remd:
        n_rounds = 8 if use_multistage_remd else 1
        steps_per_round = 5000 if use_multistage_remd else remd_n_steps

        if verbose:
            print(f"  [Torch GPU] BatchedREMD2D: {remd_n_replicas}副本 × {n_rounds}轮 × {steps_per_round}步")

        # 8T × 8λ: 300-1000K, λ=1.0→0.65 (64 副本)
        # Bug 8 修复: 使用自定义 lambdas 或默认值
        if lambdas is None:
            lambdas = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65)
        n_lam = len(lambdas)
        n_t = remd_n_replicas // n_lam
        _lambdas = lambdas

        all_diags = []
        backup_p_coords = final_p_coords.copy()  # NaN 恢复备份
        prev_3bead_state = None  # 跨轮保留完整 3-bead 状态
        backup_3bead_state = None  # Bug 7 修复: 3-bead 状态备份
        _global_step_acc = 0  # 跨轮累积全局步数
        prev_vel = None  # Bug 3 修复: 跨轮传递速度

        # Bug 10 修复: 创建一次 BatchedREMD2D 实例, 多次调用 run()
        remd = BatchedREMD2D(
            n_t=n_t,
            t_lo=300.0, t_hi=1000.0,
            lambdas=_lambdas,
            exchange_interval=500,
            use_trirnasp=use_trirnasp,
            use_trirnasp_force=use_trirnasp_force,
            trirnasp_scale=trirnasp_scale,
            sequence=sequence,
            trirnasp_energy_dir=trirnasp_energy_dir,
            force_refresh_freq=500,
            use_adaptive_tri_weight=use_adaptive_tri_weight,
            use_staged_tri=use_staged_tri,
            tri_stage_config=tri_stage_config,
        )

        for round_idx in range(n_rounds):
            try:
                # Bug 10 修复: 传入跨轮状态, 而不是重新创建实例
                best_coords, best_e, diag = remd.run(
                    final_p_coords, pairs, n_steps=steps_per_round,
                    verbose=verbose, initial_pos_3bead=prev_3bead_state,
                    initial_global_step=_global_step_acc,
                    initial_velocities=prev_vel)
                all_diags.append(diag)

                # NaN 恢复: 检测 NaN → 回退到上一步
                if not np.isfinite(best_e) or not np.all(np.isfinite(best_coords)):
                    if verbose:
                        print(f"  [Torch GPU] 轮 {round_idx+1} NaN, 回退到上一步")
                    final_p_coords = backup_p_coords.copy()
                    # Bug 7 修复: 从备份恢复 3-bead 状态 (不设为 None)
                    prev_3bead_state = backup_3bead_state
                    prev_vel = None  # Bug 3 修复: 速度也失效
                    continue

                backup_p_coords = final_p_coords.copy()  # 保存备份
                # 无条件更新: 每轮用上一轮输出作为下一轮输入
                final_p_coords = best_coords
                # 保留完整 3-bead 状态 (P + C4'/N), 避免下轮重新初始化
                prev_3bead_state = diag.get("best_pos_3bead")
                # Bug 7 修复: 保存 3-bead 状态备份
                backup_3bead_state = prev_3bead_state
                # Bug 3 修复: 保留速度状态
                prev_vel = diag.get("velocities")
                # 累积全局步数 (分阶段策略跨轮)
                _global_step_acc += steps_per_round
                if not np.isfinite(final_p_coords).all():
                    if verbose:
                        print(f"  [Torch GPU] 轮 {round_idx+1} 输出非有限, 回退到上一步")
                    final_p_coords = backup_p_coords.copy()
                    continue
                if best_e < final_e:
                    final_e = best_e

                if verbose:
                    print(f"  [Torch GPU] 轮 {round_idx+1}/{n_rounds}: E={best_e:.0f}")
            except Exception as e:
                import traceback
                if verbose:
                    print(f"  [Torch GPU] 轮 {round_idx+1} 失败: {e}")
                    traceback.print_exc()
                final_p_coords = backup_p_coords.copy()
                prev_3bead_state = None
                continue

        if verbose and all_diags:
            print(f"  [Torch GPU] REMD 完成: 最终 E={final_e:.0f}, "
                  f"T-acc={np.mean(all_diags[-1]['acceptance_T']):.0%}")

    # 3. 物理弛豫 (torch GPU)
    if use_physical_relax and L >= 10:
        try:
            # 弛豫: 传入完整配对列表 (包含假结候选和 BPP 加权)
            relaxed, relax_m = relax_structure(
                final_p_coords, sequence,
                far_pairs=None,
                n_steps=5000,
                pairs_all=pairs)
            # 弛豫后能量必须合理
            if np.all(np.isfinite(relaxed)):
                _avg = float(np.mean(np.linalg.norm(
                    relaxed[1:] - relaxed[:-1], axis=1)[:100]))
                if _avg > 3.0 and _avg < 12.0:
                    final_p_coords = relaxed
                    if verbose:
                        print(f"  [Torch GPU] 物理弛豫: "
                              f"clash {relax_m['initial']['clash_count']}"
                              f"→{relax_m['final']['clash_count']}, "
                              f"bond_viol {relax_m['final']['bond_violations']}")
        except Exception as e:
            if verbose:
                print(f"  [Torch GPU] 物理弛豫失败: {e}")

    # 4. 写 CG PDB + CG→全原子
    cg_pdb = str(out_path / f"{name}_cg.pdb")
    _write_pdb_simple(cg_pdb, final_p_coords, sequence)

    if skip_cg_to_allatom:
        aa_pdb = cg_pdb
    else:
        aa_pdb = str(out_path / f"{name}.pdb")
        try:
            from .isrnacirc_wrapper import cg_to_allatom
            cg_to_allatom(cg_pdb, aa_pdb, sequence)
            if verbose:
                print(f"  [Torch GPU] CG→全原子: {aa_pdb}")
        except Exception as e:
            if verbose:
                print(f"  [Torch GPU] CG→全原子失败: {e}, 用 CG PDB")
            aa_pdb = cg_pdb

    elapsed = time.time() - t0
    if verbose:
        print(f"  [Torch GPU] 完成: {elapsed:.1f}s, E={final_e:.0f}")

    diag = {
        "success": True,
        "energy": final_e,
        "elapsed": elapsed,
        "remd_rounds": n_rounds if use_remd else 0,
        "use_remd": use_remd,
        "use_multistage_remd": use_multistage_remd,
    }
    return aa_pdb, final_e, diag


def _write_pdb_simple(pdb_path: str, coords_A: np.ndarray, sequence: str):
    """写简单 CG PDB (P-only, 与 CG_to_allatom.exe 兼容).

    PDB ATOM 格式: col 31-38 x, 39-46 y, 47-54 z (8.3f each, 必须空格分隔).
    """
    base_map = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT"}
    L = len(coords_A)
    with open(pdb_path, "w", newline="\n") as f:
        # CG_to_allatom.exe 要求第一行直接是 ATOM (不认 REMARK)
        for i in range(L):
            x, y, z = coords_A[i]
            resname = base_map.get(sequence[i].upper(), "ADE")
            # PDB ATOM 标准: col 13-16 atom name " P  ", col 17-19 resname
            f.write(f"ATOM  {i+1:5d}  P   {resname} A{i+1:4d}"
                    f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
        f.write("END\n")
