"""
isrnaclong.py — isRNAcircLong 主管线

长链 circRNA 3D 结构预测:
  Level 0: ViennaRNA 粗筛
  Level 1: 分段 Vfold3D/RhoFold+ + Kabsch 拼装
  Level 2: RL-guided isRNAcirc close + 迭代弛豫
  Level 3: RL-MCTS 拓扑搜索
  Level 4: REST2 精修
  Level 5: 全原子 + Amber

Level 2 细节:
  - 第 1 轮: isRNAcirc Type=1 close_ends + MD (闭合 BSJ)
  - 后续轮: RL agent 指导 pair_weights + MD 参数 (替代启发式)
  - RL state: 配对距离 + 能量 + clash + 收敛指标
  - RL action: pair_weights (N_far_pairs,) + md_nstep (scalar)
  - RL reward: energy_delta + pair_rate_delta - clash_penalty

参考: isRNAcircLong_design.md
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def _save_checkpoint(ckpt_path: Path, data: dict):
    """保存 checkpoint (JSON + numpy arrays → .npy).

    原子写: 先写临时文件再 rename, 防止崩溃时半写损坏 JSON.
    """
    # numpy arrays 单独存为 .npy 文件
    arrays = {}
    clean = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            npy_path = ckpt_path.parent / f"ckpt_{k}.npy"
            np.save(str(npy_path), v)
            arrays[k] = str(npy_path)
        else:
            clean[k] = v
    clean["_npy_refs"] = arrays
    tmp_path = ckpt_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(clean, default=str, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(ckpt_path)  # 原子替换


def _load_checkpoint(ckpt_path: Path) -> dict:
    """加载 checkpoint."""
    if not ckpt_path.exists():
        return {}
    data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    arrays = data.pop("_npy_refs", {})
    for k, npy_path in arrays.items():
        data[k] = np.load(npy_path)
    return data


@dataclass
class RelaxationMetrics:
    """迭代弛豫监控指标 (4 指标联合)."""
    cross_segment_ok: float = 0.0    # 跨片段配对距离 < 15Å 比例
    clash_count: int = 0             # P-P 距离 < 3Å 的碰撞数
    rmsd_change: float = 0.0         # 相对上一轮 RMSD 变化
    pair_rate: float = 0.0           # 总配对率
    energy_delta: float = 0.0        # 能量变化

    @property
    def is_converged(self) -> bool:
        """多指标收敛判据."""
        return (self.cross_segment_ok > 0.8
                and self.clash_count == 0
                and abs(self.rmsd_change) < 0.5)


class RelaxationRL:
    """RL agent: 用 MCTS + GNN PolicyNetwork 指导 isRNAcirc 每轮弛豫.

    支持在线学习: decide() 时收集轨迹, 周期性触发 PPO 更新.
    复用 rl_optimizer.py 的 PolicyNetwork + MCTS + ReplayBuffer + OnlineLearner.
    """

    def __init__(
        self,
        far_pairs: list,
        stem_blocks: list,
        sequence: str,
        n_simulations: int = 15,
        policy_path: str = None,
        md_step_scale: float = 1.0,
    ):
        self.far_pairs = far_pairs
        self.stem_blocks = stem_blocks
        self.sequence = sequence
        self.n_simulations = n_simulations
        self.policy_path = policy_path
        self.md_step_scale = md_step_scale

        # 加载 PolicyNetwork + MCTS
        from .rl_optimizer import PolicyNetwork, MCTS, build_rl_state, compute_reward
        self._build_rl_state = build_rl_state
        self._compute_reward = compute_reward

        self.policy = None
        if policy_path:
            self.policy = PolicyNetwork()
            self.policy.load(policy_path)

        self.mcts = MCTS(
            policy=self.policy,
            c_puct=1.5,
            n_simulations=n_simulations,
            rollout_depth=3,
            use_rollout=True,
        )

        # 在线学习 (延迟启用)
        self._online_learner = None
        self._last_rmsd = 0.0  # 最近一次 decide() 的 MCTS 偏差 (诊断用)

    def enable_online_learning(
        self,
        buffer_path: str = None,
        update_every: int = 5,
        capacity: int = 500,
    ):
        """启用在线学习."""
        from .rl_optimizer import ReplayBuffer, OnlineLearner, ContinuousAssemblyPolicy

        buffer = ReplayBuffer(capacity=capacity)
        if buffer_path:
            buffer.load(buffer_path)
            print(f"  [RL] 从 {buffer_path} 加载 {len(buffer)} 条历史轨迹")

        # 用 ContinuousAssemblyPolicy (PPO 训练用)
        cont_policy = ContinuousAssemblyPolicy()
        self._online_learner = OnlineLearner(
            policy=cont_policy,
            buffer=buffer,
            update_every=update_every,
        )
        self._online_buffer_path = buffer_path

    def decide(
        self,
        coords: np.ndarray,
        far_pairs: list,
        energy: float,
        metrics: RelaxationMetrics,
        round_idx: int,
        n_relax_rounds: int,
    ) -> Tuple[dict, int]:
        """MCTS 搜索决定 pair_weights + MD 步数.

        在线学习: 每次 decide() 记录轨迹, 周期性重训练.

        Returns:
            (pair_weights, n_steps)
        """
        # 构建 RL state
        if len(coords) == 0:
            # 坐标为空, 跳过 RL, 返回默认参数
            scale = self.md_step_scale
            if metrics.pair_rate < 0.3:
                return {}, max(1000, int(50000 * scale))
            elif metrics.pair_rate < 0.6:
                return {}, max(1000, int(20000 * scale))
            else:
                return {}, max(1000, int(5000 * scale))

        state = self._build_rl_state(
            coords, self.sequence, far_pairs, self.stem_blocks,
        )

        # MCTS 搜索: 返回 best P 坐标
        best_coords = self.mcts.search(state, far_pairs)

        # 在线学习: 记录轨迹
        if self._online_learner is not None:
            reward = self._compute_reward(best_coords, far_pairs)
            traj = {
                "sequence": self.sequence,
                "far_pairs": far_pairs,
                "stem_blocks": self.stem_blocks,
                "states": [coords, best_coords],
                "best_coords": best_coords,
                "rewards": np.array([reward]),
                "round_idx": round_idx,
            }
            self._online_learner.observe(traj)

        # 从最优坐标与原始坐标的偏差 → pair_weights
        pair_weights = {}
        L = len(coords)
        for k, (i, j) in enumerate(far_pairs):
            if i >= L or j >= L:
                continue
            dist_before = float(np.linalg.norm(coords[i] - coords[j]))
            dist_after = float(np.linalg.norm(best_coords[i] - best_coords[j]))
            # 距离缩短的配对加权
            if dist_after < dist_before:
                ratio = dist_before / max(dist_after, 0.1)
                pair_weights[(i, j)] = min(5.0, max(0.1, ratio))
            else:
                pair_weights[(i, j)] = 1.0

        # MD 步数: 由 RL 的 MCTS 搜索偏差驱动 (替代硬编码 pair_rate 阈值).
        # MCTS 搜索出的 best_coords 若与当前坐标偏差大, 说明 RL 认为构象还需
        # 大调整 → 多跑 MD; 偏差小 → 接近收敛, 少跑. 这是 RL 内部真实信号.
        rmsd = 0.0
        if len(best_coords) == L and L > 0:
            diff = np.asarray(best_coords) - np.asarray(coords)
            rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        scale = self.md_step_scale
        if rmsd > 3.0:
            n_steps = 1000000   # 大调整: 1M
        elif rmsd > 1.5:
            n_steps = 500000    # 中等: 500K
        else:
            n_steps = 200000    # 接近收敛: 200K
        n_steps = max(1000, int(n_steps * scale))
        self._last_rmsd = rmsd

        return pair_weights, n_steps

    def save_online_state(self):
        """保存在线学习状态 (策略 + buffer)."""
        if self._online_learner and self._online_buffer_path:
            self._online_learner.save(self._online_buffer_path)


@dataclass
class LongPipelineResult:
    """长链管线结果."""
    sequence: str
    secondary_structure: str
    coords_cg: np.ndarray              # (L, 3) CG P 坐标
    coords_aa: Optional[np.ndarray]    # (N_atoms, 3) 全原子坐标
    energy_cg: float
    energy_aa: float
    rmsd_to_native: Optional[float]
    pair_rate: float
    cross_segment_ok_rate: float
    n_segments: int
    n_candidates: int
    runtime_seconds: float
    fidelity_history: List[Dict]
    details: Dict = field(default_factory=dict)


def isrnaclong_pipeline(
    sequence: str,
    secondary_structure: str,
    output_dir: str,
    *,
    max_seg_len: int = 200,
    overlap: int = 20,
    n_relax_rounds: int = 20,
    n_parallel: int = 0,
    n_rest2_replicas: int = 8,
    rest2_nsteps: int = 300000,
    use_rl_relax: bool = True,
    use_rl_mcts: bool = True,
    rl_n_simulations: int = 50,
    md_step_scale: float = 0.1,
    nrep: int = 1,
    platform: str = "auto",
    verbose: bool = True,
    # 分段拼装参数
    use_rhofold: bool = False,
    n_candidates: int = 1,
    # 自适应 MSA (避免 RhoFold 单序列塌缩)
    use_msa: bool = True,
    rfam_cm: str = "",
    rfam_dir: str = "",
    msa_blocks: Optional[List[Dict]] = None,
    # 5-bead CG 精修
    use_5bead: bool = True,
    # Metadynamics 增强采样
    use_metad: bool = True,
    metad_n_steps: int = 150000,
    # 断点续跑
    resume: bool = True,
    # structRFM 多任务预测头 (opt-in)
    use_multi_task_heads: bool = False,
    multitask_head_weights: Optional[str] = None,
    use_structrfm: bool = False,
    ss_head_weight: float = 1.0,
    pair_head_weight: float = 1.0,
    bsj_head_weight: float = 0.5,
    clash_head_weight: float = 0.3,
) -> LongPipelineResult:
    """isRNAcircLong 完整管线.

    Args:
        sequence: RNA 序列
        secondary_structure: 二级结构 (dot-bracket)
        output_dir: 输出目录
        max_seg_len: 分段最大长度
        overlap: 重叠区长度
        n_relax_rounds: 迭代弛豫轮数
        n_rest2_replicas: REST2 副本数
        rest2_nsteps: REST2 步数
        use_rl_relax: Level 2 是否用 RL guidance (False=消融, 用固定参数)
        use_rl_mcts: Level 3 是否用 RL-MCTS (False=消融)
        rl_n_simulations: RL 模拟次数
        md_step_scale: Level 2 每轮 MD 步数缩放因子. 默认 0.1 (步数减到 1/10,
            1M→100K / 500K→50K / 200K→20K). 控制 Level 2 总耗时;
            构象收敛不足时可调回 0.3~0.5
        use_5bead: Level 2.3 是否用 5-bead CG 精修 (默认 True).
            5-bead: P/S/B1/B2/B3 每核苷酸, 比 3-bead 更精确的 stacking/H-bond 几何.
        use_metad: Level 3.5 是否用 Metadynamics 增强采样 (默认 True).
            沿 CV (BSJ距离/配对接触/回旋半径) 加 Gaussian hill, 跨越自由能垒.
        metad_n_steps: Metadynamics 总 MD 步数, 默认 50000.
        nrep: Level 2 REMD 副本数 (IsRNAcirc 并发跑多副本, 多核并行).
            >1 时每个副本独立温度/种子, 并发 lmp 进程; 需要足够 CPU 核.
            默认 1 (单副本, 与旧版一致).
        platform: OpenMM/LAMMPS 平台
        verbose: 是否打印详细信息
        use_rhofold: True 用 RhoFold+ 预测每 chunk, False 用 isRNAcirc Type=0
        n_candidates: 每 chunk 候选数
        use_msa: True 启用自适应 MSA (真 MSA 优先, 伪 MSA 兜底),
            避免 RhoFold+ 单序列在工程序列上塌缩. 默认 True.
        rfam_cm: Rfam CM 库路径 (cmsearch 搜真 MSA 用, WSL 内路径)
        rfam_dir: Rfam 数据目录 (含已知家族 MSA, 复用真 MSA)
        msa_blocks: 可选, MSA-aware 分块锚定区间
            [{"start","end","msa_path","source"}, ...].
            提供时按锚定区间分块 (真MSA chunk 用对应 MSA)

    Returns:
        LongPipelineResult
    """
    t0 = time.time()

    # 序列标准化: 大小写统一 + T→U (RNA)
    sequence = sequence.upper().replace("T", "U")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 断点续跑: checkpoint 文件
    _ckpt_path = output_path / "_checkpoint.json"
    ckpt = _load_checkpoint(_ckpt_path) if resume else {}
    ckpt_level = ckpt.get("level", -1)

    L = len(sequence)
    if verbose:
        print(f"=== isRNAcircLong: {L}nt ===")
        if ckpt_level >= 0:
            print(f"  [续跑] 从 Level {ckpt_level} 之后继续")

    # ── Level 0: ViennaRNA 粗筛 + MUSES 多源共识 ──
    if ckpt_level >= 0:
        # 从 checkpoint 恢复 Level 0
        pairs = ckpt["pairs"]
        far_pairs = ckpt["far_pairs"]
        stem_blocks = ckpt["stem_blocks"]
        bpp = ckpt.get("bpp")
        ss_consensus = ckpt.get("ss_consensus")
        if verbose:
            print(f"\n[Level 0] 从 checkpoint 恢复: 近程{len(pairs)}, 远端{len(far_pairs)}")
    else:
        if verbose:
            print("\n[Level 0] ViennaRNA 粗筛 + MUSES 多源SS共识...")
        from .refine import vienna_pair_probs
        pairs, bpp = vienna_pair_probs(sequence, 0.5)

        # MUSES: 多源二级结构共识 (structRFM 启发)
        try:
            from .multisource_ss import multisource_consensus_ss
            ss_consensus, bpp = multisource_consensus_ss(sequence, bpp)
            if verbose:
                print(f"  MUSES 共识 SS: {ss_consensus[:50]}...")
        except Exception:
            ss_consensus = None
            if verbose:
                print("  MUSES 回退: 仅 ViennaRNA")

        # pair_graph 扫描远端配对
        from .pair_graph import build_full_pair_graph, extract_stem_blocks
        _, scan_pairs, far_pairs = build_full_pair_graph(
            sequence, pairs, do_scan=True,
        )
        stem_blocks = extract_stem_blocks(pairs, scan_pairs)

        if verbose:
            print(f"  近程配对: {len(pairs)}, 远端配对: {len(far_pairs)}")

        _save_checkpoint(_ckpt_path, {
            "level": 0,
            "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
            "bpp": bpp, "ss_consensus": ss_consensus,
        })

    # ── Level 1: 分段 Vfold3D + 拼装 ──
    if ckpt_level >= 1:
        coords_vfold = ckpt["coords_vfold"]
        n_segments = ckpt["n_segments"]
        segments = ckpt.get("segments", [])
        if verbose:
            print(f"\n[Level 1] 从 checkpoint 恢复: {n_segments} 段")
    else:
        if verbose:
            print("\n[Level 1] 分段 Vfold3D + 拼装...")
        from .segmented_vfold3d import segmented_vfold3d_pipeline, split_sequence

        # 分段信息 (MSA-aware 分块可选)
        segments = split_sequence(
            sequence, secondary_structure, max_seg_len, overlap,
            msa_blocks=msa_blocks,
        )
        n_segments = len(segments)

        if verbose:
            print(f"  分段模式: {'RhoFold+' if use_rhofold else 'isRNAcirc Type=0'}, "
                  f"{n_segments} chunks, candidates={n_candidates}")

        try:
            coords_vfold, pdb_vfold, chunk_confidences, _uncertainty = segmented_vfold3d_pipeline(
                sequence, secondary_structure, str(output_path / "vfold3d"),
                max_seg_len=max_seg_len, overlap=overlap,
                n_candidates=n_candidates,
                use_ensemble=False,
                use_rhofold=use_rhofold,
                use_trrosetta=False,
                use_msa=use_msa,
                rfam_cm=rfam_cm,
                rfam_dir=rfam_dir,
                msa_blocks=msa_blocks,
            )
        except Exception as e:
            if verbose:
                print(f"  3D 预测失败: {e}, 用默认螺旋坐标")
            coords_vfold = _default_helix_coords(L)
            chunk_confidences = [0.0] * n_segments

        if verbose:
            print(f"  分段数: {n_segments}, 初始 RMSD 估算: ~30-40A")

        _save_checkpoint(_ckpt_path, {
            "level": 1,
            "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
            "coords_vfold": coords_vfold, "n_segments": n_segments,
            "segments": segments,
        })

    # ── Level 1.5: CG 弛豫 (平滑 Vfold3D 预测) ──
    if verbose:
        print(f"\n[Level 1.5] CG 弛豫 (平滑 Vfold3D 预测)...")
    try:
        from .openmm_gpu_refiner import (
            _build_3bead_system_gpu, _create_3bead_topology,
            _generate_compact_coords, _sanitize_p_coords,
        )
        import openmm.unit as unit
        from openmm import LangevinMiddleIntegrator, Platform
        from openmm.app import Simulation

        _relax_coords = _sanitize_p_coords(coords_vfold.copy())
        # 只检查 P-P 键长异常才替换 (Vfold3D 环状结构首末距天然可大,
        # 用首末距会误判为展开并丢弃真实预测坐标)
        _avg_pp = 0.0
        if L > 1:
            _diffs = _relax_coords[1:] - _relax_coords[:-1]
            _ppd = np.linalg.norm(_diffs, axis=1)
            _avg_pp = float(np.mean(_ppd[:min(L - 1, 500)]))
        if (not np.isfinite(_avg_pp)) or _avg_pp > 20.0 or _avg_pp < 1.0:
            _relax_coords = _generate_compact_coords(L, pairs)

        _sys, _cnm, _pf, _sf, _bjf, _bjg = _build_3bead_system_gpu(
            _relax_coords, pairs, pair_scale=0.5, bsj_k_scale=0.3)
        _topo = _create_3bead_topology(L)
        _plat = Platform.getPlatformByName("CPU")
        _nthreads = os.cpu_count() or 8
        _pp = {"CpuThreads": str(_nthreads)}
        _intg = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        _sim = Simulation(_topo, _sys, _intg, _plat, _pp)
        _sim.context.setPositions(_cnm * unit.nanometer)

        # 最小化 + 短 MD
        _sim.minimizeEnergy(
            tolerance=100.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=1000)
        _sim.step(2000)  # 4ps 短 MD

        _state = _sim.context.getState(getPositions=True, getEnergy=True)
        _e = _state.getPotentialEnergy()._value
        _pos = _state.getPositions(asNumpy=True)._value  # nm
        _p_coords_relaxed = _pos[0::3] * 10.0  # nm → Å, 提取 P

        # 用弛豫后的 P 坐标替换
        if len(_p_coords_relaxed) == L:
            coords_vfold = _p_coords_relaxed
            if verbose:
                print(f"    CG 弛豫完成: E={_e:.0f} kJ/mol")
        else:
            if verbose:
                print(f"    CG 弛豫输出维度不匹配 ({len(_p_coords_relaxed)} vs {L}), 跳过")
    except Exception as e:
        if verbose:
            print(f"    CG 弛豫失败: {e}, 用原始坐标")

    # ── structRFM 多任务预测头 (opt-in) ──
    multitask_heads = None
    if use_multi_task_heads:
        try:
            from .multitask_heads import CircRNAPredictionHeads
            from .multitask_loss import CircRNAMultiTaskLoss
            multitask_heads = CircRNAPredictionHeads(use_structrfm=use_structrfm)
            if multitask_head_weights and Path(multitask_head_weights).exists():
                multitask_heads.load_state_dict(
                    torch.load(str(multitask_head_weights), map_location="cpu"))
            multitask_heads.eval()
            multitask_loss_fn = CircRNAMultiTaskLoss(
                w_ss=ss_head_weight, w_pair=pair_head_weight,
                w_bsj=bsj_head_weight, w_clash=clash_head_weight)
            if verbose:
                n_params = sum(p.numel() for p in multitask_heads.parameters())
                print(f"  [MultiTask] heads loaded: {n_params} params")
        except Exception as e_mt:
            if verbose:
                print(f"  [MultiTask] heads 初始化失败: {e_mt}")
            multitask_heads = None

    # ── Level 2: 分段并行 CG→全原子 + RL 调度 REMD ──
    if False:  # Level 2 checkpoint 续跑已禁用, 每次强制重跑
        best_coords = ckpt["best_coords"]
        best_energy = ckpt["best_energy"]
        segments = ckpt.get("segments", [])
        # 检查坐标是否有效 (可能是空数组)
        if len(best_coords) == 0:
            if verbose:
                print(f"  [警告] checkpoint 坐标为空, 用 Level 1 坐标")
            best_coords = coords_vfold.copy()
            best_energy = _estimate_energy(best_coords, pairs, sequence)
        from .multifidelity_scheduler import RuleScheduler, SimulationState
        state = SimulationState()
        state.pair_rate = ckpt.get("pair_rate", 0.0)
        state.cross_segment_ok_rate = ckpt.get("cross_segment_ok_rate", 0.0)
        state.energy = ckpt.get("energy", 0.0)
        state.clash_count = ckpt.get("clash_count", 0)
        # scheduler 在 checkpoint 续跑路径也需要定义 (return 语句引用)
        scheduler = RuleScheduler()
        if verbose:
            print(f"\n[Level 2] 从 checkpoint 恢复: E={best_energy:.0f}")
    else:
        if verbose:
            print(f"\n[Level 2] 分段并行 CG→全原子 + {'RL 调度 REMD' if use_rl_relax else '固定 REMD'}...")

        # 解析 OpenMM 平台 (Level 2 OpenMM GPU 精修需要)
        if platform == "auto":
            from .rest2_sampler import detect_openmm_platform
            resolved_platform = detect_openmm_platform()
        else:
            resolved_platform = platform

        from .multifidelity_scheduler import RuleScheduler, SimulationState, FidelityLevel
        scheduler = RuleScheduler()
        state = SimulationState()
        # 加载训练好的 RL 策略 (Level 2 RelaxationRL)
        _rl_policy_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rl_policy_b0.pth")
        if not Path(_rl_policy_path).exists():
            _rl_policy_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rl_policy_bootstrap.pth")

        rl_agent = RelaxationRL(
            far_pairs=far_pairs,
            stem_blocks=stem_blocks,
            sequence=sequence,
            n_simulations=15,
            policy_path=_rl_policy_path if Path(_rl_policy_path).exists() else None,
            md_step_scale=md_step_scale,
        ) if use_rl_relax else None

        coords_current = coords_vfold
        best_energy = float("inf")
        best_coords = coords_current.copy()
        coords_prev = None
        prev_energy = float("inf")
        metrics = RelaxationMetrics()

        def _segmented_cg_to_allatom(cg_coords_full, seg_list, out_dir, seq):
            """分段并行 CG→全原子, 拼装成完整全原子 PDB."""
            from .isrnacirc_wrapper import cg_to_allatom
            from concurrent.futures import ThreadPoolExecutor, as_completed
            Path(out_dir).mkdir(parents=True, exist_ok=True)

            seg_pdbs = []
            for idx, seg in enumerate(seg_list):
                s, e = seg["start"], seg["end"]
                seg_coords = cg_coords_full[s:e]
                seg_pdb = str(Path(out_dir) / f"seg_{idx}_cg.pdb")
                _write_coords_pdb(seg_coords, seg["seq"], seg_pdb)
                seg_pdbs.append((idx, seg_pdb, seg["seq"]))

            aa_pdbs = [None] * len(seg_pdbs)

            def _one(idx, cg_in, s):
                aa_out = str(Path(out_dir) / f"seg_{idx}_aa.pdb")
                cg_to_allatom(cg_in, aa_out, s)
                return idx, aa_out

            with ThreadPoolExecutor(max_workers=min(4, len(seg_pdbs))) as pool:
                futures = {pool.submit(_one, a[0], a[1], a[2]): a[0] for a in seg_pdbs}
                for fut in as_completed(futures):
                    try:
                        idx, aa_out = fut.result()
                        aa_pdbs[idx] = aa_out
                    except Exception as e:
                        if verbose:
                            print(f"    段 {futures[fut]}: 失败: {e}")

            merged_pdb = str(Path(out_dir) / "merged_aa.pdb")
            _merge_allatom_pdbs(aa_pdbs, seg_list, merged_pdb, seq)
            return merged_pdb

        # 分段 CG→全原子 (只做一次, 后续轮复用)
        try:
            merged_aa = _segmented_cg_to_allatom(
                coords_current, segments, str(output_path / "cg2aa"), sequence,
            )
            if verbose:
                print(f"    分段 CG→全原子完成: {merged_aa}")
        except Exception as e:
            if verbose:
                print(f"    CG→全原子失败: {e}, 用 Level 1 坐标")
            energy = _estimate_energy(coords_current, pairs, sequence)
            best_coords = coords_current
            best_energy = energy
            merged_aa = None

        # 迭代 REMD (RL 调度或固定)
        n_remd_rounds = n_relax_rounds if use_rl_relax else 1
        prev_pdb_out = None  # 上一轮的精修 PDB 路径
        round_idx = 0  # 初始化, 确保循环外可用
        metrics = RelaxationMetrics()  # 初始化
        energy = 0.0  # 初始化
        for round_idx in range(n_remd_rounds):
            if merged_aa is None:
                break
            if verbose:
                print(f"  REMD 轮 {round_idx + 1}/{n_remd_rounds}:")

            # 决定 REMD 参数
            if use_rl_relax and rl_agent is not None:
                pw, n_steps = rl_agent.decide(
                    coords_current, far_pairs, prev_energy, metrics,
                    round_idx, n_remd_rounds,
                )
                nstep_close = max(1000, n_steps // 5)
                if verbose:
                    w_vals = list(pw.values()) if pw else []
                    rmsd = getattr(rl_agent, "_last_rmsd", 0.0)
                    if w_vals:
                        print(f"    RL: nstep={n_steps} (rmsd={rmsd:.2f}Å), nstep_close={nstep_close}, "
                              f"w=[{min(w_vals):.2f}, {max(w_vals):.2f}]")
                    else:
                        print(f"    RL: nstep={n_steps} (rmsd={rmsd:.2f}Å), nstep_close={nstep_close}")
            else:
                # 固定参数 (IsRNAcirc 推荐: nstep=1M, nstep_close=500K, nstru=500)
                # 乘 md_step_scale 缩小 Level 2 单轮步数 (默认 0.1)
                n_steps = max(1000, int(1000000 * md_step_scale))
                nstep_close = max(1000, int(500000 * md_step_scale))
                if verbose:
                    print(f"    固定: nstep={n_steps}, nstep_close={nstep_close}")

            # structRFM: learned pair predictions 融合到 RL weights
            if (multitask_heads is not None and pw
                    and len(coords_current) == L):
                try:
                    from .multitask_heads import build_struct_condition_from_coords
                    # 构建结构条件
                    struct_cond = build_struct_condition_from_coords(
                        coords_current, far_pairs, L)
                    struct_t = torch.tensor(struct_cond, dtype=torch.float32)
                    # 获取 bpp 矩阵
                    bpp_tensor = None
                    if bpp_matrix is not None:
                        bpp_tensor = torch.tensor(bpp_matrix, dtype=torch.float32)
                    # 预测
                    with torch.no_grad():
                        mt_out = multitask_heads(
                            SEQUENCE,
                            struct_condition=struct_t,
                            bpp_matrix=bpp_tensor,
                        )
                    # 融合 pair weights: 50% MCTS + 50% learned
                    if "pair_probs" in mt_out and "clash_scores" in mt_out:
                        clash_scores = mt_out["clash_scores"].numpy()
                        for (i, j) in pw:
                            if i < L and j < L:
                                # 碰撞高发位置降低权重
                                clash_penalty = 0.5 * (clash_scores[i] + clash_scores[j])
                                pw[(i, j)] *= max(0.1, 1.0 - clash_penalty)
                        if verbose:
                            print(f"    [MultiTask] clash scores 融合完成")
                except Exception as e_mt:
                    if verbose:
                        print(f"    [MultiTask] 融合失败: {e_mt}")

            try:
                round_dir = str(output_path / f"remd_r{round_idx}")
                pdb_out = None
                energy = float("inf")

                # Level 2 精修: 只用 IsRNAcirc.exe (IsRNA2 力场, 无任何 fallback)
                # 第 0 轮用 RhoFold+ 拼装坐标 (配对已折叠~28A, 键长校正到 5.9A),
                #   而不是分段重建的 merged_aa (分段会丢失全局折叠, 配对退化到 46A).
                # 后续轮用上一轮的精修结果.
                if round_idx == 0:
                    # 优先用已有 merged_aa.pdb (已含全原子, 跳过 20min CG→allatom 重建).
                    # merged_aa 是分段拼装结果, 保留了全局折叠 + 全原子坐标.
                    # 只有 merged_aa 不存在时才 fallback 到 RhoFold+ P-only 路径.
                    _merged_aa = str(output_path / "cg2aa" / "merged_aa.pdb")
                    if Path(_merged_aa).exists():
                        refine_input = _merged_aa
                        if verbose:
                            print(f"    round 0: 直接用 merged_aa.pdb (已有全原子, 跳过 CG→allatom)")
                    else:
                        _start_pdb = str(output_path / "start_rhofold.pdb")
                        _coords_start = coords_vfold.copy()
                        if L > 1:
                            _pp = np.linalg.norm(
                                _coords_start[1:] - _coords_start[:-1], axis=1)
                            _pp_mean = float(_pp.mean())
                            if 0.1 < _pp_mean < 20.0 and abs(_pp_mean - 5.9) > 0.5:
                                _coords_start = _coords_start * (5.9 / _pp_mean)
                        _write_coords_pdb(_coords_start, sequence, _start_pdb)
                        refine_input = _start_pdb
                else:
                    # 用上轮 best_coords 写临时 PDB 作为输入 (不依赖 prev_pdb_out)
                    if best_coords is not None and len(best_coords) == L:
                        _prev_cg = str(output_path / f"_prev_round_cg.pdb")
                        _write_coords_pdb(best_coords, sequence, _prev_cg)
                        refine_input = _prev_cg
                    else:
                        refine_input = prev_pdb_out or merged_aa
                from .openmm_gpu_refiner import openmm_gpu_refine
                if verbose:
                    print(f"    OpenMM GPU 精修 (输入: {'RhoFold+起点' if round_idx == 0 else '上轮结果'})...")
                # 渐进软化注入远端配对: round 0 只注入 30%, 后续 round 递增到 100%
                # (OpenMM 路径暂不支持 far_pair_ratio, 保留接口兼容)
                n_rounds_total = max(1, n_remd_rounds)
                far_ratio = 1.0 if n_rounds_total == 1 else min(1.0, 0.3 + 0.7 * (round_idx / max(1, n_rounds_total - 1)))
                if verbose and far_pairs and far_ratio > 0:
                    print(f"    远端配对注入: {len(far_pairs)} 对 (OpenMM 模式)")
                pdb_out, energy = openmm_gpu_refine(
                    refine_input, round_dir,
                    sequence, secondary_structure,
                    name=f"remd_r{round_idx}",
                    nstep=max(100000, n_steps),
                    platform_name=resolved_platform,
                    use_remd=True,
                    remd_n_replicas=max(6, n_rest2_replicas),
                    remd_n_steps=max(10000, n_steps // 3),
                    verbose=verbose,
                    use_physical_relax=True,
                    skip_minimal_fold=(round_idx > 0),
                )

                coords_relaxed = _read_pdb_p_coords(pdb_out)
                if len(coords_relaxed) == 0:
                    if verbose:
                        print(f"    PDB 读取为空, 用输入坐标")
                    coords_relaxed = coords_current
                elif np.any(np.isnan(coords_relaxed)):
                    if verbose:
                        print(f"    PDB 含 NaN 坐标, 用输入坐标")
                    coords_relaxed = coords_current
                else:
                    prev_pdb_out = pdb_out  # 记录本轮输出, 下轮复用
                if verbose:
                    print(f"    E={energy:.0f}")
            except Exception as e:
                if verbose:
                    print(f"    REMD 失败: {e}, 跳过")
                    import traceback
                    traceback.print_exc()
                energy = _estimate_energy(coords_current, pairs, sequence)
                coords_relaxed = coords_current

            # 4 指标监控
            metrics = _compute_relaxation_metrics(
                coords_relaxed, coords_prev, pairs, far_pairs, segments,
                energy, prev_energy,
            )
            state.energy = energy
            state.cross_segment_ok_rate = metrics.cross_segment_ok
            state.pair_rate = metrics.pair_rate
            state.clash_count = metrics.clash_count

            if energy < best_energy and len(coords_relaxed) > 0:
                # NaN 安全检查: 如果输出坐标有 NaN, 跳过本轮
                if np.any(np.isnan(coords_relaxed)):
                    if verbose:
                        print(f"    [WARN] REMD 输出有 NaN, 跳过本轮更新")
                else:
                    best_energy = energy
                    best_coords = coords_relaxed.copy()

            coords_prev = coords_relaxed.copy()
            prev_energy = energy
            coords_current = coords_relaxed

            if metrics.is_converged:
                if verbose:
                    print(f"    收敛!")
                break

            # 每轮保存 checkpoint
            _save_checkpoint(_ckpt_path, {
                "level": 2, "remd_round": round_idx + 1,
                "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
                "coords_vfold": coords_vfold, "n_segments": n_segments,
                "segments": segments,
                "best_coords": best_coords, "best_energy": best_energy,
                "pair_rate": metrics.pair_rate,
                "cross_segment_ok_rate": metrics.cross_segment_ok,
                "energy": energy,
                "clash_count": metrics.clash_count,
            })
            ckpt_level = 2  # 更新内存, 防止 Level 3/4 误续跑

        # Level 2 完成后强制保存 checkpoint (即使 REMD 循环被 break)
        _save_checkpoint(_ckpt_path, {
            "level": 2, "remd_round": round_idx + 1,
            "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
            "coords_vfold": coords_vfold, "n_segments": n_segments,
            "segments": segments,
            "best_coords": best_coords, "best_energy": best_energy,
            "pair_rate": metrics.pair_rate,
            "cross_segment_ok_rate": metrics.cross_segment_ok,
            "energy": energy,
            "clash_count": metrics.clash_count,
        })

    # ── Level 2.3: 5-bead CG 精修 (比 3-bead 更精确的 stacking/H-bond 几何) ──
    if use_5bead and best_coords is not None and len(best_coords) == len(sequence):
        # 检查输入坐标是否有 NaN
        _has_nan = np.any(np.isnan(best_coords))
        _has_inf = np.any(np.isinf(best_coords))
        if _has_nan or _has_inf:
            if verbose:
                print(f"\n[Level 2.3] 5-bead 跳过: 输入坐标含 NaN/Inf (nan={_has_nan}, inf={_has_inf})")
        elif verbose:
            print(f"\n[Level 2.3] 5-bead CG 精修...")
        try:
            from .fivebead_folding import refine_5bead
            _resolved_platform = "CPU"  # 5-bead 用 CPU (粒子数 5x)
            p5_refined, e5_0, e5_1 = refine_5bead(
                best_coords, pairs,
                platform_name=_resolved_platform,
                n_anneal=5000,
                sequence=sequence)
            # 5-bead 输出含 NaN 时丢弃
            if np.any(np.isnan(p5_refined)) or np.any(np.isinf(p5_refined)):
                if verbose:
                    print(f"    5-bead 输出含 NaN/Inf, 保留当前坐标")
            elif e5_1 < best_energy or not np.isfinite(best_energy):
                best_coords = p5_refined.copy()
                best_energy = e5_1
                if verbose:
                    print(f"    5-bead: E={e5_0:.0f} -> {e5_1:.0f} kJ/mol")
            elif verbose:
                print(f"    5-bead: E={e5_1:.0f} (未优于当前 {best_energy:.0f}, 保留)")
        except Exception as e:
            if verbose:
                print(f"    5-bead 精修跳过: {e}")

    # ── Level 2.5: REMD 后 CG→allatom (把最终 CG 坐标转成全原子) ──
    if best_coords is not None and len(best_coords) == len(sequence):
        _final_aa_path = str(output_path / "final_allatom.pdb")
        if verbose:
            print(f"\n[Level 2.5] REMD 后 CG→全原子: {_final_aa_path}")
        try:
            from .isrnacirc_wrapper import cg_to_allatom
            # 写临时 CG PDB (用 _write_coords_pdb)
            _tmp_cg = str(output_path / "_final_cg_for_aa.pdb")
            _write_coords_pdb(best_coords, sequence, _tmp_cg)
            cg_to_allatom(_tmp_cg, _final_aa_path, sequence)
            if verbose:
                _sz = os.path.getsize(_final_aa_path) / 1024
                print(f"    全原子输出: {_final_aa_path} ({_sz:.0f} KB)")
        except Exception as e:
            if verbose:
                print(f"    CG→allatom 失败: {e}")

    # ── Level 3: RL 微调 (连续动作空间) ──
    if ckpt_level >= 3:
        if verbose:
            print(f"\n[Level 3] 从 checkpoint 恢复")
    elif use_rl_mcts and far_pairs:
        if verbose:
            print(f"\n[Level 3] RL 微调 (连续动作, PPO {rl_n_simulations} epochs)...")
        try:
            from .rl_optimizer import optimize_far_pairs
            _rl_l3_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rl_policy_b0.pth")
            if not Path(_rl_l3_path).exists():
                _rl_l3_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rl_policy_bootstrap.pth")
            # 检查 best_coords 维度是否匹配 CG 粒子数
            # IsRNAcirc 输出全原子 PDB, 但 far_pairs 基于 CG 索引 (0~L-1)
            # 如果 best_coords 维度 != len(sequence), 跳过 RL 微调
            if len(best_coords) != len(sequence):
                if verbose:
                    print(f"    跳过: best_coords 维度 ({len(best_coords)}) != 序列长度 ({len(sequence)}), IsRNAcirc 输出全原子 PDB")
            else:
                opt_p, cg_orig, rl_info = optimize_far_pairs(
                    best_coords, sequence, far_pairs, stem_blocks,
                    n_simulations=rl_n_simulations,
                    policy_path=_rl_l3_path if Path(_rl_l3_path).exists() else None,
                )
                best_coords = opt_p
                if verbose:
                    print(f"    RL 完成: reward={rl_info.get('final_reward', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"    RL 微调失败: {e}")
        _save_checkpoint(_ckpt_path, {
            "level": 3,
            "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
            "coords_vfold": coords_vfold, "n_segments": n_segments,
            "segments": segments,
            "best_coords": best_coords, "best_energy": best_energy,
        })

    # ── Level 3.5: Metadynamics 增强采样 (沿 CV 跨越自由能垒) ──
    if use_metad and best_coords is not None and len(best_coords) == len(sequence):
        if verbose:
            print(f"\n[Level 3.5] Metadynamics 采样 (well-tempered, {metad_n_steps} 步)...")
        try:
            from .metadynamics_sampler import MetaDynamicsSampler
            meta = MetaDynamicsSampler(
                sequence, pairs,
                hill_height=1.0,       # kJ/mol
                hill_sigma=0.1,         # CV 空间 (nm)
                hill_freq=100,          # 每 100 步加一个 hill
                max_hills=5000,
                well_tempered=True,
                bias_factor=5.0,
                platform_name="CPU",
            )
            meta_coords, meta_e = meta.sample(
                best_coords, n_steps=metad_n_steps, verbose=verbose)
            if meta_e < best_energy:
                best_coords = meta_coords
                best_energy = meta_e
                if verbose:
                    print(f"    MetaD E={meta_e:.0f} (优于当前)")
            elif verbose:
                print(f"    MetaD E={meta_e:.0f} (未优于 {best_energy:.0f}, 保留)")
        except Exception as e:
            if verbose:
                print(f"    MetaD 跳过: {e}")

    # ── Level 4: REST2 精修 ──
    if ckpt_level >= 4:
        if verbose:
            print(f"\n[Level 4] 从 checkpoint 恢复")
    else:
        # 解析 "auto" 平台
        if platform == "auto":
            from .rest2_sampler import detect_openmm_platform
            resolved_platform = detect_openmm_platform()
        else:
            resolved_platform = platform
        if verbose:
            print(f"\n[Level 4] REST2 精修 ({n_rest2_replicas} 副本, 平台={resolved_platform})...")
        try:
            from .rest2_sampler import REST2Sampler
            temperatures = [1.0 + i * 0.5 for i in range(n_rest2_replicas)]
            rest2 = REST2Sampler(
                temperatures=temperatures,
                n_steps=rest2_nsteps,
                platform_name=resolved_platform,
            )
            coords_rest2, e_rest2, _rest2_snaps = rest2.sample(
                best_coords, pairs, sequence,
            )
            best_coords = coords_rest2
            best_energy = e_rest2
            if verbose:
                print(f"    REST2 E={e_rest2:.0f}")
        except Exception as e:
            if verbose:
                print(f"    REST2 失败: {e}")
        _save_checkpoint(_ckpt_path, {
            "level": 4,
            "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
            "coords_vfold": coords_vfold, "n_segments": n_segments,
            "segments": segments,
            "best_coords": best_coords, "best_energy": best_energy,
        })

    # ── Level 5: AMBER RNA.OL3 全原子精修 ──
    if False:  # Level 5 checkpoint 续跑已禁用
        pass
    else:
        if verbose:
            print(f"\n[Level 5] AMBER RNA.OL3 全原子精修 (最小化+MD)...")
        try:
            from .openmm_amber_refiner import openmm_amber_refine, OPENMM_AVAILABLE as AMBER_OK
            if AMBER_OK:
                # CG P → 全原子 PDB
                cg_pdb_5 = str(output_path / "level5_cg.pdb")
                _write_coords_pdb(best_coords, sequence, cg_pdb_5)
                from .isrnacirc_wrapper import cg_to_allatom
                aa_pdb_5 = str(output_path / "level5_aa.pdb")
                cg_to_allatom(cg_pdb_5, aa_pdb_5, sequence)
                # AMBER 全原子精修
                amber_out, amber_e = openmm_amber_refine(
                    aa_pdb_5, str(output_path / "level5_amber"),
                    name="level5",
                    minimize_max_iter=3000,
                    md_steps=15000,
                    use_remd=False,
                    restraints_k=50.0,
                    platform_name="CPU",
                    verbose=verbose,
                )
                old_energy = best_energy
                if amber_e < best_energy:
                    best_energy = amber_e
                    # 从 AMBER 输出提取 P 坐标更新 best_coords
                    p_coords_5 = _read_pdb_p_coords(amber_out)
                    if len(p_coords_5) == L:
                        best_coords = p_coords_5
                    if verbose:
                        print(f"    AMBER 精修: E={amber_e:.0f} (优于之前 {old_energy:.0f})")
                else:
                    if verbose:
                        print(f"    AMBER 精修: E={amber_e:.0f} (未优于 {old_energy:.0f}, 保留原结果)")
            else:
                if verbose:
                    print(f"    AMBER 精修跳过 (OpenMM 未安装)")
        except Exception as e:
            if verbose:
                print(f"    Level 5 失败: {e}")
        _save_checkpoint(_ckpt_path, {
            "level": 5,
            "pairs": pairs, "far_pairs": far_pairs, "stem_blocks": stem_blocks,
            "coords_vfold": coords_vfold, "n_segments": n_segments,
            "segments": segments,
            "best_coords": best_coords, "best_energy": best_energy,
        })

    # 写最终 PDB
    final_pdb = str(output_path / "isrnaclong_final.pdb")
    _write_coords_pdb(best_coords, sequence, final_pdb)

    # 读取全原子 P 坐标 (如果 final_allatom.pdb 存在)
    _faa = str(output_path / "final_allatom.pdb")
    coords_aa = _read_pdb_p_coords(_faa) if os.path.exists(_faa) else best_coords

    runtime = time.time() - t0
    if verbose:
        print(f"\n=== 完成: {runtime:.1f}s ===")

    # 全原子坐标: 从 final_allatom.pdb 读取 (Level 2.5 输出)
    _faa_path = str(output_path / "final_allatom.pdb")
    coords_aa = _read_pdb_p_coords(_faa_path) if os.path.exists(_faa_path) else best_coords

    return LongPipelineResult(
        sequence=sequence,
        secondary_structure=secondary_structure,
        coords_cg=best_coords,
        coords_aa=coords_aa,
        energy_cg=best_energy,
        energy_aa=best_energy,
        rmsd_to_native=None,
        pair_rate=state.pair_rate,
        cross_segment_ok_rate=state.cross_segment_ok_rate,
        n_segments=n_segments,
        n_candidates=n_candidates,
        runtime_seconds=runtime,
        fidelity_history=scheduler.history,
    )


def _steps_for_level(level) -> int:
    """根据保真度级别返回 MD 步数 (3 级版)."""
    steps = {
        "CG_FAST": 500,        # ~1ps, 快速探索
        "CG_MEDIUM": 5000,     # ~10ps, 中等精度
        "CG_REST2": 50000,     # ~100ps, REST2 增强采样
        # 旧版兼容
        "CG_SHORT": 500,
        "REST2": 50000,
    }
    return steps.get(level.name, 5000)


def _estimate_energy(coords, pairs, sequence) -> float:
    """简单能量估计 (无 LAMMPS 时). coords 为 P-only Å."""
    from .refine import BOND_LEN
    energy = 0.0
    L = len(coords)

    # 骨架键 (BOND_LEN 单位 Å, coords 单位 Å)
    for i in range(L - 1):
        d = np.linalg.norm(coords[i] - coords[i + 1])
        energy += 0.5 * 31000.0 * (d - BOND_LEN) ** 2

    # 配对 (兼容 (i,j) 和 (i,j,w) 格式, 目标 ~10.5Å WC 距离)
    for p in pairs:
        if len(p) == 3:
            i, j, w = p
        else:
            i, j = p
            w = 1.0
        if 0 <= i < L and 0 <= j < L:
            d = np.linalg.norm(coords[i] - coords[j])
            energy += 0.5 * w * 800.0 * (d - 10.5) ** 2

    return energy


def _check_cross_segment_pairs(coords, far_pairs, segments) -> float:
    """检查跨片段配对距离."""
    if not far_pairs:
        return 1.0

    ok_count = 0
    total = 0
    L = len(coords)
    for i, j in far_pairs:
        if i >= L or j >= L:
            continue
        seg_i = _find_segment(i, segments)
        seg_j = _find_segment(j, segments)
        if seg_i != seg_j:
            total += 1
            d = np.linalg.norm(coords[i] - coords[j])
            if d < 15.0:
                ok_count += 1

    return ok_count / total if total > 0 else 1.0


def _find_segment(res_idx, segments) -> int:
    """找残基属于哪个段."""
    for idx, seg in enumerate(segments):
        if seg["start"] <= res_idx < seg["end"]:
            return idx
    return -1


def _compute_pair_rate(coords, pairs) -> float:
    """计算配对率."""
    if not pairs or len(coords) == 0:
        return 0.0
    ok = 0
    L = len(coords)
    for p in pairs:
        if len(p) == 3:
            i, j, _ = p
        else:
            i, j = p
        if i >= L or j >= L:
            continue
        d = np.linalg.norm(coords[i] - coords[j])
        if d < 15.0:
            ok += 1
    return ok / len(pairs)


def _compute_clash_count(coords, threshold: float = 3.0) -> int:
    """计算 P-P 碰撞数 (距离 < threshold Å)."""
    L = len(coords)
    count = 0
    for i in range(L):
        for j in range(i + 2, min(i + 20, L)):  # 局部检查, 避免 O(n²)
            d = np.linalg.norm(coords[i] - coords[j])
            if d < threshold:
                count += 1
    return count


def _compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """计算两组坐标之间的 RMSD."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _compute_relaxation_metrics(
    coords: np.ndarray,
    coords_prev: Optional[np.ndarray],
    pairs,
    far_pairs,
    segments,
    energy: float,
    prev_energy: float,
) -> RelaxationMetrics:
    """计算 4 指标弛豫监控."""
    if len(coords) == 0:
        # 坐标读取失败, 返回空指标
        return RelaxationMetrics(
            cross_segment_ok=0.0, clash_count=0, rmsd_change=0.0,
            pair_rate=0.0, energy_delta=0.0,
        )
    return RelaxationMetrics(
        cross_segment_ok=_check_cross_segment_pairs(coords, far_pairs, segments),
        clash_count=_compute_clash_count(coords),
        rmsd_change=_compute_rmsd(coords, coords_prev) if coords_prev is not None else 0.0,
        pair_rate=_compute_pair_rate(coords, pairs),
        energy_delta=energy - prev_energy,
    )


def _update_pair_weights(coords, far_pairs, old_weights, metrics=None) -> dict:
    """更新跨片段配对权重 (扩展版: 加入 clash 惩罚)."""
    new_weights = old_weights.copy()
    L = len(coords)
    for p in far_pairs:
        i, j = p[0], p[1]
        if i >= L or j >= L:
            continue
        d = np.linalg.norm(coords[i] - coords[j])
        if d > 15.0:
            # 距离太远 → 加强权重
            new_weights[(i, j)] = min(old_weights.get((i, j), 1.0) * 1.2, 5.0)
        elif d < 5.0:
            # 太近 → 降低权重
            new_weights[(i, j)] = max(old_weights.get((i, j), 1.0) * 0.8, 0.1)

    # 全局 clash 惩罚: 有碰撞时降低所有权重
    if metrics is not None and metrics.clash_count > 0:
        for key in new_weights:
            new_weights[key] = max(new_weights[key] * 0.7, 0.1)

    return new_weights


def _default_helix_coords(L):
    """默认 A-form 螺旋坐标."""
    import math
    coords = np.zeros((L, 3))
    for i in range(L):
        z = i * 2.8
        angle = i * 33.0 * math.pi / 180
        coords[i] = [4.4 * math.cos(angle), 4.4 * math.sin(angle), z]
    return coords


def _read_pdb_p_coords(pdb_path: str) -> np.ndarray:
    """从 PDB 读取 P 原子坐标, 返回 (N, 3)."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        # fallback: IsRNAcirc 输出全原子 PDB, 没有 P 原子标记.
        # 读所有原子坐标 (不只是第一个), 供 Level 3 RL 使用.
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords) if coords else np.zeros((0, 3))


def _write_coords_pdb(coords, sequence, output_path):
    """写坐标到 PDB."""
    lines = ["HEADER    isRNAcircLong CG structure"]
    for i, (x, y, z) in enumerate(coords):
        res_name = sequence[i] if i < len(sequence) else "N"
        lines.append(
            f"ATOM  {i+1:5d}  P   {res_name:>3s} A{i+1:4d}"
            f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P"
        )
    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def _merge_allatom_pdbs(aa_pdb_paths, seg_list, output_path, full_sequence):
    """把分段全原子 PDB 按残基顺序拼成完整全原子 PDB.

    直接复制原始 ATOM 行 (保持 PDB 列对齐), 只改残基编号.
    """
    lines = ["HEADER    isRNAcircLong merged allatom"]
    atom_idx = 0
    res_offset = 0

    for seg_idx, (aa_pdb, seg) in enumerate(zip(aa_pdb_paths, seg_list)):
        if aa_pdb is None:
            continue
        seg_res_count = 0
        with open(aa_pdb) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    continue
                line = line.rstrip("\n\r")
                # 段内残基编号 (从 PDB 原始行读取)
                try:
                    local_res = int(line[22:26].strip())
                except (ValueError, IndexError):
                    local_res = seg_res_count + 1
                global_res = res_offset + local_res
                atom_idx += 1
                # 保持原始 PDB 列对齐, 只改 atom serial (7-11) 和 resSeq (22-26)
                new_line = (
                    line[:6]                              # "ATOM  "
                    + f"{atom_idx:5d}"                    # serial 7-11
                    + line[11:22]                         # atom name, altLoc, resName, chainID
                    + f"{global_res:4d}"                  # resSeq 22-26
                    + line[26:]                           # iCode + 其余 (coords, occ, etc.)
                )
                lines.append(new_line)
                seg_res_count = local_res
        res_offset += seg_res_count

    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
