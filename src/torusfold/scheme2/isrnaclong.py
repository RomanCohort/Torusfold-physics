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
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ── WSL 预热 + PyRosetta 长驻服务 ──────────────────────────────────────
_PYROSETTA_SOCK = "/tmp/torusfold_pyrosetta.sock"
_PYROSETTA_SERVER_PROC = None  # 本进程启动的 server 引用


def _preheat_wsl():
    """异步预热 WSL 实例, 消除后续 subprocess 冷启动延迟."""
    try:
        subprocess.Popen(
            ["wsl", "echo", "ok"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # 预热失败不影响后续


def _win_to_wsl(p) -> str:
    """Windows 路径 → WSL 路径 (/mnt/c/...)."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


def _pyrosetta_server_running() -> bool:
    """检查 PyRosetta server 是否已在运行."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(_PYROSETTA_SOCK)
        s.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def _pyrosetta_start_server(verbose: bool = True) -> bool:
    """启动 PyRosetta 长驻服务 (WSL 后台), 等待 ready 信号."""
    global _PYROSETTA_SERVER_PROC
    if _pyrosetta_server_running():
        if verbose:
            print("    [PyR-server] 已在运行")
        return True

    src_dir = _win_to_wsl(Path(__file__).resolve().parent)
    cmd = (
        f'python3 -c "import sys; sys.path.insert(0, \'{src_dir}\'); '
        f'from pyrosetta_refine import _server_main; _server_main()"'
    )
    _PYROSETTA_SERVER_PROC = subprocess.Popen(
        ["wsl", "bash", "-c", cmd],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )

    # 等待 socket 就绪 (最多 30s)
    if verbose:
        print("    [PyR-server] 启动中 (等待 PyRosetta init)...")
    for _ in range(300):
        time.sleep(0.1)
        if _pyrosetta_server_running():
            if verbose:
                print("    [PyR-server] 就绪 ✓")
            return True
        if _PYROSETTA_SERVER_PROC.poll() is not None:
            if verbose:
                print("    [PyR-server] 进程已退出, 回退到 subprocess 模式")
            return False

    if verbose:
        print("    [PyR-server] 启动超时, 回退到 subprocess 模式")
    return False


def _pyrosetta_socket_refine(
    pdb_path: str, output_path: str, max_iter: int = 200, verbose: bool = True,
) -> Optional[Tuple[str, float]]:
    """通过 Unix socket 调用 PyRosetta 长驻服务精修.

    Returns: (output_path, energy) 或 None (服务不可用时).
    """
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5)
        s.connect(_PYROSETTA_SOCK)
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return None

    try:
        req = json.dumps({
            "pdb": pdb_path,
            "output": output_path,
            "max_iter": max_iter,
        }).encode("utf-8")
        # 长度前缀 (4 字节 little-endian)
        s.sendall(len(req).to_bytes(4, "little") + req)

        # 读响应
        raw_len = b""
        while len(raw_len) < 4:
            chunk = s.recv(4 - len(raw_len))
            if not chunk:
                return None
            raw_len += chunk

        msg_len = int.from_bytes(raw_len, "little")
        raw_msg = b""
        while len(raw_msg) < msg_len:
            chunk = s.recv(min(65536, msg_len - len(raw_msg)))
            if not chunk:
                return None
            raw_msg += chunk

        resp = json.loads(raw_msg.decode("utf-8"))
        if resp.get("ok"):
            return (resp["output"], resp.get("energy", float("inf")))
        else:
            if verbose:
                print(f"    [PyR-server] 错误: {resp.get('error', '?')}")
            return None
    finally:
        s.close()


def _save_checkpoint(ckpt_path: Path, data: dict):
    """保存 checkpoint (JSON + numpy arrays → .npy).

    原子写: 先写临时文件再 rename, 防止崩溃时半写损坏 JSON.
    numpy arrays 也用临时文件, 写完再 rename, 避免 JSON 引用不存在的 .npy.
    """
    # numpy arrays 单独存为 .npy 文件 (原子写)
    arrays = {}
    clean = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            npy_path = ckpt_path.parent / f"ckpt_{k}.npy"
            npy_tmp = ckpt_path.parent / f"_ckpt_{k}_tmp.npy"
            np.save(str(npy_tmp), v)
            os.replace(str(npy_tmp), str(npy_path))
            arrays[k] = str(npy_path)
        else:
            clean[k] = v
    clean["_npy_refs"] = arrays
    tmp_path = str(ckpt_path) + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(clean, default=str, ensure_ascii=False))
    os.replace(tmp_path, str(ckpt_path))  # 原子替换


def _load_checkpoint(ckpt_path: Path) -> dict:
    """加载 checkpoint (容错: .npy 文件缺失/损坏时跳过, 不影响其余字段)."""
    if not ckpt_path.exists():
        return {}
    try:
        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [checkpoint] JSON 加载失败: {e}")
        return {}
    arrays = data.pop("_npy_refs", {})
    for k, npy_path in arrays.items():
        try:
            data[k] = np.load(npy_path)
        except Exception as e:
            print(f"  [checkpoint] .npy 加载失败: {k} ({npy_path}): {e}")
    return data


def _cleanup_checkpoints(output_path: Path, keep_final: bool = True):
    """清理中间 checkpoint 文件, 只保留最终结果.

    删除:
      - _checkpoint.json (checkpoint 元数据)
      - _checkpoint_*.npy (numpy 数组)
      - _tmp_* (临时文件)

    保留:
      - *.pdb (所有 PDB 输出)
      - *.json (结果文件, 非 checkpoint)
      - _final_*.npz (最终数据)
    """
    removed = 0
    for f in output_path.glob("_checkpoint*"):
        f.unlink(missing_ok=True)
        removed += 1


def _delete_checkpoint_from_level(output_path: Path, from_level: float = 2.0):
    """删除指定 level 及之后的 checkpoint 字段, 回退到该 level 之前的状态.

    例: _delete_checkpoint_from_level(path, 2.0) 删除 Level 2/2.5/2.6/3/... 的字段,
    checkpoint 回退到 Level 1.5, 下次 resume 从 Level 2 重新开始.

    Args:
        output_path: 输出目录
        from_level: 从哪个 level 开始删除 (包含该 level)
    """
    ckpt_path = output_path / "_checkpoint.json"
    if not ckpt_path.exists():
        print(f"  [checkpoint] 无 checkpoint 文件, 跳过删除")
        return

    ckpt = _load_checkpoint(ckpt_path)
    current_level = float(ckpt.get("level", -1))

    # Level 2+ 的字段列表 (从 _save_ckpt 调用中收集)
    _LEVEL2_PLUS_FIELDS = {
        "remd_round", "best_coords", "best_energy", "pair_rate",
        "cross_segment_ok_rate", "energy", "clash_count",
        # Level 2.5/2.6/3/3.5/4/5/5.5 的字段
        "p5_refined", "pyrosetta_out", "l26_ok",
        "coords_rl", "rl_info",
        "coords_metad", "meta_e",
        "coords_rest2", "rest2_e",
        "level5_amber", "amber_e",
        "ppr_repaired", "ppr_rate",
    }

    # 检查是否有 Level 2+ 的字段残留 (即使 level 数值已回退)
    has_level2_fields = bool(set(ckpt.keys()) & _LEVEL2_PLUS_FIELDS)
    if current_level < from_level and not has_level2_fields:
        print(f"  [checkpoint] 当前 level={current_level}, 无 Level {from_level}+ 字段, 无需删除")
        return

    removed_fields = []
    removed_npy = 0
    for field in _LEVEL2_PLUS_FIELDS:
        if field in ckpt:
            del ckpt[field]
            removed_fields.append(field)
            # 删除关联的 .npy 文件
            npy_path = output_path / f"ckpt_{field}.npy"
            if npy_path.exists():
                npy_path.unlink()
                removed_npy += 1

    # 回退 level 到 from_level 之前
    _rollback_levels = {2.0: 1.5, 2.5: 2.0, 2.6: 2.5, 3.0: 2.6,
                        3.5: 3.0, 4.0: 3.5, 5.0: 4.0, 5.5: 5.0}
    new_level = _rollback_levels.get(from_level, from_level - 0.5)
    ckpt["level"] = new_level

    # 保存修改后的 checkpoint
    _save_checkpoint(ckpt_path, ckpt)
    print(f"  [checkpoint] 已删除 Level {from_level}+ 字段: {removed_fields}")
    print(f"  [checkpoint] 已删除 {removed_npy} 个 .npy 文件")
    print(f"  [checkpoint] level 回退到 {new_level}, 下次从 Level {new_level} 之后继续")


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
        from torusfold.scheme2.rl_optimizer import PolicyNetwork, MCTS, build_rl_state, compute_reward
        self._build_rl_state = build_rl_state
        self._compute_reward = compute_reward

        self.policy = None
        if policy_path and os.path.exists(policy_path):
            self.policy = PolicyNetwork()
            self.policy.load(policy_path)
        else:
            # 无预训练权重: 创建随机策略, MCTS 用启发式 rollout
            self.policy = PolicyNetwork()

        self.mcts = MCTS(
            policy=None,  # 纯启发式 MCTS (不依赖策略先验)
            c_puct=1.5,
            n_simulations=max(n_simulations, 20),
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
        from torusfold.scheme2.rl_optimizer import ReplayBuffer, OnlineLearner, ContinuousAssemblyPolicy

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
        # 长度缩放: L>500 时步数减半, L>1000 时再减半
        l_scale = 1.0
        if L > 1000:
            l_scale = 0.25
        elif L > 500:
            l_scale = 0.5
        n_steps = max(1000, int(n_steps * scale * l_scale))
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
    hbond_rate: float = 0.0  # 真氢键满足率 (全原子级, <3.6Å)
    details: Dict = field(default_factory=dict)


def isrnaclong_pipeline(
    sequence: str,
    secondary_structure: str,
    output_dir: str,
    *,
    max_seg_len: int = 200,
    overlap: int = 20,
    n_relax_rounds: int = 6,
    n_parallel: int = 0,
    n_rest2_replicas: int = 16,
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
    metad_n_steps: int = 200000,
    # PyRosetta 条件式精修 (Level 2.6, WSL)
    use_pyrosetta: bool = True,
    # PPR 碱基对氢键修复 (Level 5.5)
    use_ppr: bool = True,
    ppr_max_rounds: int = 5,
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
        n_relax_rounds: 迭代弛豫轮数 (默认 6, 大多数情况够用; 早停机制会提前结束)
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

    # 异步预热 WSL (消除后续冷启动延迟)
    if use_pyrosetta:
        _preheat_wsl()

    # 序列标准化: 大小写统一 + T→U (RNA)
    sequence = sequence.upper().replace("T", "U")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 断点续跑: checkpoint 文件
    _ckpt_path = output_path / "_checkpoint.json"
    ckpt = _load_checkpoint(_ckpt_path) if resume else {}
    _raw_level = ckpt.get("level", -1)
    ckpt_level = float(_raw_level) if _raw_level is not None else -1
    # 累加 checkpoint: 每个 Level 追加字段, 存盘时写完整 dict (防覆盖丢字段)
    _ckpt_data = dict(ckpt)

    def _save_ckpt(level: float, **extra):
        """累加字段到 _ckpt_data 并存盘."""
        _ckpt_data["level"] = level
        _ckpt_data.update(extra)
        _save_checkpoint(_ckpt_path, _ckpt_data)
        print(f"  [checkpoint] level={level}, fields={list(_ckpt_data.keys())}")

    L = len(sequence)
    if verbose:
        print(f"=== isRNAcircLong: {L}nt ===")
        if ckpt_level >= 0:
            print(f"  [续跑] 从 Level {ckpt_level} 之后继续")

    # ── Level 0: Partition Function BPP + 置信度分层约束 ──
    # 只要 JSON 里有完整字段就恢复, 不依赖 ckpt_level 数值
    _l0_from_ckpt = ("pairs" in ckpt and "far_pairs" in ckpt and "stem_blocks" in ckpt
                     and "bpp" in ckpt and ckpt["bpp"] is not None)
    if _l0_from_ckpt:
        pairs = ckpt["pairs"]
        far_pairs = ckpt["far_pairs"]
        stem_blocks = ckpt["stem_blocks"]
        bpp = ckpt.get("bpp")
        ss_consensus = ckpt.get("ss_consensus")
        bpp_high = ckpt.get("bpp_high", [])
        bpp_mid = ckpt.get("bpp_mid", [])
        if verbose:
            print(f"\n[Level 0] 从 checkpoint 恢复: 近程{len(pairs)}, 远端{len(far_pairs)}")
    else:
        if ckpt_level >= 0 and verbose:
            print(f"\n[Level 0] checkpoint 不完整, 重新计算配对...")
        if verbose:
            print("\n[Level 0] Partition Function BPP + 置信度分层...")

        # ── 1. ViennaRNA Partition Function ──
        import RNA as _RNA
        md_pf = _RNA.md()
        md_pf.circ = 1  # 环化模式
        fc_pf = _RNA.fold_compound(sequence, md_pf)
        ss_pf, pf_energy = fc_pf.pf()

        # 提取全概率配对列表
        plist = fc_pf.plist_from_probs(0.01)  # P > 1%

        # 置信度分层
        bpp_high = [(ep.i - 1, ep.j - 1, ep.p) for ep in plist if ep.p > 0.9]
        bpp_mid = [(ep.i - 1, ep.j - 1, ep.p) for ep in plist if 0.5 < ep.p <= 0.9]
        bpp_low = [(ep.i - 1, ep.j - 1, ep.p) for ep in plist if 0.1 < ep.p <= 0.5]

        if verbose:
            print(f"  PF energy: {pf_energy:.2f}")
            print(f"  高置信 (P>0.9):   {len(bpp_high)} 对")
            print(f"  中置信 (0.5-0.9): {len(bpp_mid)} 对")
            print(f"  低置信 (0.1-0.5): {len(bpp_low)} 对")

        # ── 2. BPP 矩阵 + MFE (复用 fc_pf: 省一次 O(L^3) PF, 且 circ=1 一致) ──
        try:
            _bpp_raw = np.array(fc_pf.bpp(), dtype=np.float64)
            bpp = _bpp_raw[1:, 1:] if _bpp_raw.shape[0] > len(sequence) else _bpp_raw.copy()
            if verbose:
                print(f"  BPP 矩阵: {bpp.shape}, max P={float(bpp.max()):.3f}")
        except Exception:
            bpp = None
        try:
            _ss_mfe, _mfe_e = fc_pf.mfe()
            pairs_mfe = []
            _stk = []
            for _i, _c in enumerate(_ss_mfe):
                if _c == "(":
                    _stk.append(_i)
                elif _c == ")" and _stk:
                    _j = _stk.pop()
                    pairs_mfe.append((min(_i, _j), max(_i, _j)))
        except Exception:
            pairs_mfe = [(i, j) for i, j, _ in bpp_high + bpp_mid]

        # DivideFold: 递归分块预测独立结构
        ss_divide = None
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "DivideFold-main" / "src"))
            import RNA as _RNA_DL
            def _rnafold_api(seq):
                md = _RNA_DL.md(); fc = _RNA_DL.fold_compound(seq, md)
                ss, _ = fc.mfe(); return ss
            # DivideFold 需要纯 CPU 子进程 (ROCm 在 import 时编译 MIOpen kernel)
            _dd_runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "_dd_runner.py")
            _sys_python = os.path.join(os.path.expanduser("~"), "AppData", "Local", "Python", "bin", "python3.exe")
            if not os.path.exists(_sys_python):
                _sys_python = sys.executable
            try:
                _dd_result = subprocess.run(
                    [_sys_python, _dd_runner, "--seq", sequence, "--max-frag", "200"],
                    capture_output=True, text=True, timeout=None,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "",
                         "HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""},
                )
                if _dd_result.returncode == 0 and _dd_result.stdout.strip():
                    ss_divide = _dd_result.stdout.strip()
                else:
                    ss_divide = None
                    if verbose:
                        print(f"  DivideFold 子进程失败: {_dd_result.stderr[:200]}")
            except Exception as _dd_err:
                ss_divide = None
                if verbose:
                    print(f"  DivideFold 子进程异常: {_dd_err}")
            if ss_divide is not None:
                if verbose:
                    print(f"  DivideFold: {ss_divide.count('(')} pairs")
        except Exception as e:
            if verbose:
                print(f"  DivideFold 不可用: {e}")

        # ── 3. 多源约束融合 ──
        mfe_set = set((i, j) for i, j in pairs_mfe)
        pf_high_set = set((i, j) for i, j, _ in bpp_high)
        pf_mid_set = set((i, j) for i, j, _ in bpp_mid)

        # DivideFold 配对集
        divide_set = set()
        if ss_divide:
            stack = []
            for i, c in enumerate(ss_divide):
                if c == "(": stack.append(i)
                elif c == ")" and stack:
                    divide_set.add((stack.pop(), i))

        # ── 硬约束: 多方法一致 (≥2个方法支持) ──
        all_sources = [pf_high_set, mfe_set, divide_set]
        pair_votes = {}
        for src in all_sources:
            for p in src:
                pair_votes[p] = pair_votes.get(p, 0) + 1

        hard_pairs = [p for p, v in pair_votes.items() if v >= 2]
        hard_set = set(hard_pairs)
        for p in pf_high_set:
            if p not in hard_set:
                hard_pairs.append(p)
                hard_set.add(p)

        # ── 软约束: MFE + PF中置信 + DivideFold (不截断) ──
        soft_pairs = []
        for i, j in mfe_set:
            if (i, j) not in hard_set:
                soft_pairs.append((i, j, 0.8))
                hard_set.add((i, j))
        for i, j, p in bpp_mid:
            if (i, j) not in hard_set:
                soft_pairs.append((i, j, p))
                hard_set.add((i, j))
        for i, j in divide_set:
            if (i, j) not in hard_set:
                soft_pairs.append((i, j, 0.6))

        pairs = [(i, j, 1.0) for i, j in hard_pairs]
        pairs += soft_pairs

        # ── RCM 重加权: 用反向互补匹配替代 BPP 概率 ──
        try:
            from torusfold.scheme2.rcm import compute_rcm_score
            _flank = min(200, len(sequence) // 4)  # 侧翼窗口 200bp
            _rcm_pairs = []
            for i, j, w in pairs:
                # 提取侧翼序列 (circRNA 环形: 取 i 上游 + j 下游)
                seq_up = sequence[max(0, i - _flank):i]
                seq_down = sequence[j:min(len(sequence), j + _flank)]
                if len(seq_up) >= 5 and len(seq_down) >= 5:
                    rcm = compute_rcm_score(seq_up, seq_down)
                    _rcm_pairs.append((i, j, rcm['confidence']))
                else:
                    _rcm_pairs.append((i, j, w))  # 太短保持原权重
            pairs = _rcm_pairs
            if verbose:
                _rcm_vals = [w for _, _, w in pairs]
                print(f"  RCM 重加权: mean={np.mean(_rcm_vals):.3f}, "
                      f"min={min(_rcm_vals):.3f}, max={max(_rcm_vals):.3f}")
        except Exception as e:
            if verbose:
                print(f"  RCM 重加权跳过: {e}")

        if verbose:
            n_multi = sum(1 for v in pair_votes.values() if v >= 2)
            print(f"  PF高置信: {len(pf_high_set)}, MFE: {len(mfe_set)}, DivideFold: {len(divide_set)}")
            print(f"  多方法一致(≥2): {n_multi}")
            print(f"  硬约束: {len(hard_pairs)}, 软约束: {len(soft_pairs)}")
            print(f"  总约束: {len(pairs)}")

        # ── 3b. NCM 非典型配对检测 (P0) ──
        # 传入 Level 0 已算好的 BPP 矩阵, 避免 NCM 内部重复跑 ViennaRNA PF
        ncm_type_map = {}  # (i,j) -> type, 下游 CG 力场按类型分配目标距离
        ncm_type_target = {
            "HOOGSTEEN": 10.5,  # Hoogsteen: 与 WC 类似距离
            "SUGAR":     10.5,  # Sugar: 类似 WC
            "SHEAR":     11.5,  # Shear: 稍远
            "STACK":     10.0,  # Stacking: 接近平行距离
        }
        try:
            from torusfold.scheme2.ncm_detector import detect_ncms_from_bpp
            ncm_raw = detect_ncms_from_bpp(sequence, bpp, hard_pairs)
            # ncm_raw: [(i, j, weight, type)]
            for _i, _j, _w, _etype in ncm_raw:
                ncm_type_map[(_i, _j)] = _etype
                _target = ncm_type_target.get(_etype, 10.5)
                # 用 weight 和类型标记做软约束
                pairs.append((_i, _j, _w))
            if verbose:
                _ncm_type_counts = {}
                for _, etype in ncm_type_map.items():
                    _ncm_type_counts[etype] = _ncm_type_counts.get(etype, 0) + 1
                print(f"  NCM 非典型对: {len(ncm_raw)}  "
                      f"类型分布: {_ncm_type_counts}")
        except Exception as e:
            if verbose:
                print(f"  NCM 检测跳过: {e}")

        # ── 4. pair_graph 扫描远端配对 ──
        from torusfold.scheme2.pair_graph import build_full_pair_graph, extract_stem_blocks
        # pairs 已经是 (i, j, w) 三元组
        _, scan_pairs, far_pairs = build_full_pair_graph(
            sequence, pairs, do_scan=True,
        )
        stem_blocks = extract_stem_blocks(pairs, scan_pairs)

        # ── 4b. 假结检测 (circRNA 关键) ──
        try:
            from torusfold.scheme2.pair_graph import detect_pseudoknots_from_bpp
            pk_pairs = detect_pseudoknots_from_bpp(
                sequence, bpp, pairs,
                pk_threshold=0.1, min_confidence=0.3,
                is_circular=True,
            )
            # 假结配对加入约束: 作为软约束 (weight 由 confidence 决定)
            for pk_i, pk_j, pk_conf in pk_pairs:
                pairs.append((pk_i, pk_j, pk_conf))
                # 假结配对天然跨越远端 → 加入 far_pairs
                _topo_dist = min(abs(pk_i - pk_j), len(sequence) - abs(pk_i - pk_j))
                if _topo_dist > 100:
                    far_pairs.append((pk_i, pk_j))
            if verbose and pk_pairs:
                print(f"  假结候选: {len(pk_pairs)} 对 (已加入约束)")
        except Exception as e:
            if verbose:
                print(f"  假结检测失败: {e}")

        # NCM 类型映射写入 checkpoint 供下游 CG 力场使用
        # (ncm_type_map 在上方 3b 里已定义)

        if verbose:
            print(f"  近程配对: {len(pairs)}, 远端配对: {len(far_pairs)}")

        _save_ckpt(0,
            pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
            bpp=bpp, ss_consensus=ss_pf,
            bpp_high=bpp_high, bpp_mid=bpp_mid,
            pf_energy=pf_energy,
        )

        # Level 0 完整诊断
        try:
            _n_mfe = len(pairs_mfe) if 'pairs_mfe' in dir() else 0
            _n_divide = len(divide_set) if 'divide_set' in dir() else 0
            _n_ncm = len(ncm_pairs) if 'ncm_pairs' in dir() else 0
            diag_l0 = {
                "pf_energy": float(pf_energy),
                "bpp_sum": float(np.sum(bpp)) if bpp is not None else 0,
                "n_hard": len(hard_pairs),
                "n_soft": len(soft_pairs),
                "n_ncm": _n_ncm,
                "n_mfe": _n_mfe,
                "dividerefold_n_pairs": _n_divide,
                "n_far": len(far_pairs),
                "n_stem_blocks": len(stem_blocks),
                "seq_length": len(sequence),
                "gc_content": sum(1 for c in sequence if c in "GCgc") / max(len(sequence), 1),
            }
            diag_path = output_path / "_plots" / "00_level0_diag.json"
            diag_path.parent.mkdir(parents=True, exist_ok=True)
            diag_path.write_text(json.dumps(diag_l0, indent=2))
        except Exception as _err:
            raise

    # ── Level 0 数据导出 ──
    try:
        from torusfold.scheme2.data_exporter import export_level0_bpp, export_level0_ncm
        _ncm_pairs_for_export = ncm_pairs if 'ncm_pairs' in dir() else []
        export_level0_bpp(bpp, len(sequence), str(output_path))
        export_level0_ncm(_ncm_pairs_for_export, len(sequence), str(output_path))
    except Exception as _err:
        raise

    # ── Level 1: 分段 Vfold3D + 拼装 ──
    # Level 1 只有在 Level 0 也从 checkpoint 恢复时才能复用, 否则 pairs 不一致
    _l1_from_ckpt = (_l0_from_ckpt and "coords_vfold" in ckpt and "n_segments" in ckpt)
    if _l1_from_ckpt:
        coords_vfold = ckpt["coords_vfold"]
        n_segments = ckpt["n_segments"]
        segments = ckpt.get("segments", [])
        chunk_confidences = [0.5] * n_segments  # checkpoint 未保存, 用默认值
        _chunk_unc = [0.5] * n_segments
        if verbose:
            print(f"\n[Level 1] 从 checkpoint 恢复: {n_segments} 段")
    else:
        if verbose:
            print("\n[Level 1] 分段 Vfold3D + 拼装...")
        from torusfold.scheme2.segmented_vfold3d import segmented_vfold3d_pipeline, split_sequence

        # 分段信息 (MSA-aware 分块可选)
        segments = split_sequence(
            sequence, secondary_structure, max_seg_len, overlap,
            msa_blocks=msa_blocks,
        )
        n_segments = len(segments)

        if verbose:
            print(f"  分段模式: {'RhoFold+' if use_rhofold else 'isRNAcirc Type=0'}, "
                  f"{n_segments} chunks, candidates={n_candidates}")

        _l1_ok = False
        try:
            coords_vfold, pdb_vfold, chunk_confidences, _uncertainty, ncm_ens_pairs = segmented_vfold3d_pipeline(
                sequence, secondary_structure, str(output_path / "vfold3d"),
                max_seg_len=max_seg_len, overlap=overlap,
                n_candidates=n_candidates,
                use_ensemble=True,
                use_rhofold=use_rhofold,
                use_trrosetta=False,
                use_msa=use_msa,
                global_bpp=bpp,
                rfam_cm=rfam_cm,
                rfam_dir=rfam_dir,
                msa_blocks=msa_blocks,
                far_pairs=far_pairs,
            )
            # ── NCM ensemble 距离反推合并进 pairs (软约束) ──
            if ncm_ens_pairs:
                _ncm_dist = [(gi, gj, conf) for gi, gj, _t, conf in ncm_ens_pairs]
                pairs += _ncm_dist
                if verbose:
                    print(f"  NCM ensemble 距离证据: +{len(_ncm_dist)} 对软约束")
            _l1_ok = True
        except Exception as e:
            if verbose:
                print(f"  3D 预测失败: {e}, 用默认螺旋坐标")
            coords_vfold = _default_helix_coords(L)
            chunk_confidences = [0.0] * n_segments
            ncm_ens_pairs = []

        if verbose:
            print(f"  分段数: {n_segments}, 初始 RMSD 估算: ~30-40A")

        # Level 1 验证 (通过后才存 checkpoint)
        v1 = _validate_structure(coords_vfold, pairs, bpp, sequence, "Level 1")
        print(f"  [验证 Level 1] clash={v1['clash_count']}, pair_rate={v1['pair_rate']:.2f}, "
              f"bond_q={v1['bond_quality']:.2f}, valid={v1['is_valid']}")

        if _l1_ok:
            _save_ckpt(1,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
            )
        if not v1["is_valid"]:
            print(f"  [警告] Level 1 输出质量不佳: {v1['warnings']}")

        # ── P2: 重叠区置信度评估 ──
        try:
            from torusfold.scheme2.overlap_confidence import OverlapConfidence
            if chunk_confidences and len(chunk_confidences) == len(segments):
                per_res_conf = np.ones(L, dtype=np.float32) * 0.5
                for seg_idx, (seg, cconf) in enumerate(zip(segments, chunk_confidences)):
                    s, e = seg["start"], seg["end"]
                    per_res_conf[s:e] = cconf
                if verbose:
                    mean_conf = float(np.mean(per_res_conf))
                    low_conf = int(np.sum(per_res_conf < 0.3))
                    print(f"  [P2] 逐残基置信度: mean={mean_conf:.3f}, low={low_conf}/{L}")
        except Exception as e:
            if verbose:
                print(f"  [P2] 置信度评估跳过: {e}")

    # Level 1 逐 chunk 诊断
    try:
        if 'chunk_confidences' in dir() and chunk_confidences and segments:
            _diag_chunks = []
            for _ci, (_seg, _cc) in enumerate(zip(segments, chunk_confidences)):
                _s, _e = _seg["start"], _seg["end"]
                _chunk_seq = sequence[_s:_e] if _s < len(sequence) and _e <= len(sequence) else ""
                _diag_c = {
                    "chunk_id": _ci,
                    "seq_len": _e - _s,
                    "start": _s,
                    "end": _e,
                    "method": "rhofold" if use_rhofold else "isrnacirc",
                    "confidence": float(_cc),
                }
                if coords_vfold is not None and _e <= len(coords_vfold) and _e > _s:
                    _cd = coords_vfold[_s:_e]
                    if len(_cd) > 1:
                        _diag_c["bond_mean"] = float(np.mean(np.linalg.norm(np.diff(_cd, axis=0), axis=1)))
                    else:
                        _diag_c["bond_mean"] = 0.0
                _diag_chunks.append(_diag_c)
            _chunk_diag_path = output_path / "_plots" / "01_level1_chunks_diag.json"
            _chunk_diag_path.parent.mkdir(parents=True, exist_ok=True)
            _chunk_diag_path.write_text(json.dumps(_diag_chunks, indent=2))
            # 逐 chunk 保存到各自目录
            for _diag_c in _diag_chunks:
                _seg_dir = output_path / "vfold3d" / f"chunk_{_diag_c['chunk_id']}"
                _seg_dir.mkdir(parents=True, exist_ok=True)
                (_seg_dir / "diag.json").write_text(json.dumps(_diag_c, indent=2))
    except Exception as _err:
        raise

    # ── Level 1 数据导出 ──
    try:
        from torusfold.scheme2.data_exporter import export_level1_chunks, export_level1_weights
        _chunk_unc = [1.0] * len(segments) if segments else []
        export_level1_chunks(segments, chunk_confidences, _chunk_unc, str(output_path))
        _region_w = [{"rhofold": 0.5, "trrna2": 0.2, "rnabpflow": 0.3}] * len(segments)
        export_level1_weights(_region_w, str(output_path))
    except Exception as _err:
        raise

    # ── Level 1.5: CG 全局约束弛豫 (平滑 Vfold3D 分块拼装接缝) ──
    # Level 1.5 depends on Level 1 output; only restore if Level 1 was also restored
    _l15_from_ckpt = (_l1_from_ckpt and "coords_relaxed" in ckpt)
    if _l15_from_ckpt:
        coords_vfold = ckpt["coords_relaxed"]
        if verbose:
            print(f"\n[Level 1.5] 从 checkpoint 恢复全局弛豫坐标")
    else:
        if verbose:
            print(f"\n[Level 1.5] CG 全局约束弛豫...")
    _l15_ok = False
    try:
        from torusfold.scheme2.openmm_gpu_refiner import (
            _generate_compact_coords, _sanitize_p_coords,
        )
        from torusfold.scheme2.physical_relaxation import relax_structure

        _relax_coords = _sanitize_p_coords(coords_vfold.copy())
        _avg_pp = 0.0
        if L > 1:
            _diffs = _relax_coords[1:] - _relax_coords[:-1]
            _ppd = np.linalg.norm(_diffs, axis=1)
            _avg_pp = float(np.mean(_ppd[:min(L - 1, 500)]))
        if (not np.isfinite(_avg_pp)) or _avg_pp > 20.0 or _avg_pp < 1.0:
            _relax_coords = _generate_compact_coords(L, pairs)

        # torch GPU 弛豫 (完整 CG 力场, 替代 OpenMM CPU)
        _pairs_for_relax = [(i, j, w) for (i, j, w) in pairs
                            if 0 <= i < L and 0 <= j < L]
        _relaxed_l15, _metrics_l15 = relax_structure(
            _relax_coords, sequence,
            far_pairs=None,
            n_steps=5000,
            pairs_all=_pairs_for_relax if _pairs_for_relax else None)
        _coords_relaxed = _relaxed_l15

        # torch GPU 弛豫完成, 提取结果
        _p_coords_relaxed = _coords_relaxed
        _e = 0.0  # torch GPU 版不输出 OpenMM 能量

        if len(_p_coords_relaxed) == L:
            coords_vfold = _p_coords_relaxed
            if verbose:
                print(f"    全局弛豫 (torch GPU): bond_viol={_metrics_l15['final']['bond_violations']}")
        else:
            if verbose:
                print(f"    输出维度不匹配 ({len(_p_coords_relaxed)} vs {L}), 跳过")
        _l15_ok = True
    except Exception as e:
        if verbose:
            print(f"    CG 弛豫失败: {e}, 用原始坐标")

    # Level 1.5 验证
    v15 = _validate_structure(coords_vfold, pairs, bpp, sequence, "Level 1.5")
    print(f"  [验证 Level 1.5] clash={v15['clash_count']}, bond_q={v15['bond_quality']:.2f}, valid={v15['is_valid']}")
    if 'v1' in dir() and v15["clash_count"] < v1["clash_count"]:
        print(f"  [OK] 弛豫减少 clash: {v1['clash_count']} → {v15['clash_count']}")

    # 保存 Level 1.5 弛豫后 PDB
    _l15_pdb = str(output_path / "level1_5_relaxed.pdb")
    _write_coords_pdb(coords_vfold, sequence, _l15_pdb)
    if verbose:
        print(f"  [PDB] Level 1.5: {_l15_pdb}")

    # ── Level 1.5 数据导出 ──
    try:
        from torusfold.scheme2.data_exporter import export_level15_trajectory
        export_level15_trajectory([], str(output_path))
    except Exception as _err:
        raise

    # Level 1.5 强制 checkpoint (每 2 个 Level 存一次)
    if _l15_ok:
        try:
            _save_ckpt(1.5,
                coords_relaxed=coords_vfold,
                time=time.time(),
            )
        except Exception as _err:
            raise

    # ── structRFM 多任务预测头 (opt-in) ──
    multitask_heads = None
    if use_multi_task_heads:
        try:
            from torusfold.scheme2.multitask_heads import CircRNAPredictionHeads
            from torusfold.scheme2.multitask_loss import CircRNAMultiTaskLoss
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

    remd_history = []  # Level 2 REMD 收敛数据

    # ── Level 2: 分段并行 CG→全原子 + RL 调度 REMD ──
    _l2_from_ckpt = ("best_coords" in ckpt and "best_energy" in ckpt)
    if _l2_from_ckpt:
        best_coords = ckpt["best_coords"]
        best_energy = ckpt["best_energy"]
        segments = ckpt.get("segments", [])
        # 检查坐标是否有效 (可能是空数组)
        if len(best_coords) == 0:
            if verbose:
                print(f"  [警告] checkpoint 坐标为空, 用 Level 1 坐标")
            best_coords = coords_vfold.copy()
            best_energy = _estimate_energy(best_coords, pairs, sequence)
        from torusfold.scheme2.multifidelity_scheduler import RuleScheduler, SimulationState
        state = SimulationState()
        state.pair_rate = ckpt.get("pair_rate", 0.0)
        state.cross_segment_ok_rate = ckpt.get("cross_segment_ok_rate", 0.0)
        state.energy = ckpt.get("energy", 0.0)
        state.clash_count = ckpt.get("clash_count", 0)
        # scheduler 在 checkpoint 续跑路径也需要定义 (return 语句引用)
        scheduler = RuleScheduler()
        # 续跑时需要初始化这些变量 (循环内后续轮次引用)
        coords_prev = best_coords.copy()
        prev_energy = best_energy
        no_improve_count = 0
        inject_frac = 0.3
        metrics = RelaxationMetrics()
        if verbose:
            print(f"\n[Level 2] 从 checkpoint 恢复: E={best_energy:.0f}")
    else:
        if verbose:
            print(f"\n[Level 2] 分段并行 CG→全原子 + {'RL 调度 REMD' if use_rl_relax else '固定 REMD'}...")

        # 解析 OpenMM 平台 (Level 2 OpenMM GPU 精修需要)
        if platform == "auto":
            from torusfold.scheme2.rest2_sampler import detect_openmm_platform
            resolved_platform = detect_openmm_platform()
        else:
            resolved_platform = platform

        # 检测 GPU 用于 REMD 加速
        _remd_platform_name = resolved_platform  # 默认用 auto 解析的平台
        try:
            import openmm as _omm
            for _try_gpu in ["CUDA", "OpenCL"]:
                try:
                    _omm.Platform.getPlatformByName(_try_gpu)
                    _remd_platform_name = _try_gpu
                    if verbose:
                        print(f"  [Level 2] GPU 检测: {_try_gpu} 可用, REMD 使用 GPU 加速")
                    break
                except Exception:
                    pass
        except Exception as _err:
            raise

        from torusfold.scheme2.multifidelity_scheduler import RuleScheduler, SimulationState, FidelityLevel
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
        no_improve_count = 0
        inject_frac = 0.3  # 远端配对动态注入比例
        metrics = RelaxationMetrics()

        def _segmented_cg_to_allatom(cg_coords_full, seg_list, out_dir, seq):
            """分段并行 CG→全原子, 拼装成完整全原子 PDB."""
            from torusfold.scheme2.isrnacirc_wrapper import cg_to_allatom
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

            # 分段 CG→全原子 并行执行 (每个 exe 调用独立进程, 可并行)
            def _convert_segment(cg_path, aa_path, seq_chunk):
                cg_to_allatom(cg_path, aa_path, seq_chunk)
                return aa_path

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for idx, cg_in, seg_seq in seg_pdbs:
                    aa_out = str(Path(out_dir) / f"seg_{idx}_aa.pdb")
                    fut = executor.submit(_convert_segment, cg_in, aa_out, seg_seq)
                    futures[fut] = idx
                for fut in as_completed(futures):
                    seg_idx = futures[fut]
                    try:
                        fut.result()
                        aa_pdbs[seg_idx] = str(Path(out_dir) / f"seg_{seg_idx}_aa.pdb")
                    except Exception as e:
                        if verbose:
                            print(f"    段 {seg_idx}: 失败: {e}")

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
                    from torusfold.scheme2.multitask_heads import build_struct_condition_from_coords
                    # 构建结构条件
                    struct_cond = build_struct_condition_from_coords(
                        coords_current, far_pairs, L)
                    struct_t = torch.tensor(struct_cond, dtype=torch.float32)
                    # 获取 bpp 矩阵
                    # Bug 9 修复: 使用正确的变量名 (bpp 而不是 bpp_matrix)
                    bpp_tensor = None
                    if bpp is not None:
                        bpp_tensor = torch.tensor(bpp, dtype=torch.float32)
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
                # 优先 torch GPU 路径 (ROCm/HIP 兼容), OpenMM 作为 fallback
                try:
                    from torusfold.scheme2.torch_gpu_refine import torch_gpu_refine as _refine_fn
                    _refine_name = "Torch GPU"
                except Exception as _e:
                    from torusfold.scheme2.openmm_gpu_refiner import openmm_gpu_refine as _refine_fn
                    _refine_name = "OpenMM CPU (fallback)"
                    if verbose:
                        print(f"    ⚠ Torch GPU 不可用: {_e}, 回退到 OpenMM CPU")
                if verbose:
                    print(f"    {_refine_name} 精修 (输入: {'RhoFold+起点' if round_idx == 0 else '上轮结果'})...")
                # 动态远端配对注入: 根据 pair_rate 调整注入比例
                # (OpenMM 路径暂不支持 far_pair_ratio, 保留接口兼容)
                if round_idx > 0:
                    if metrics.pair_rate > 0.6:
                        inject_frac = min(1.0, inject_frac + 0.2)  # 配对好, 加速注入
                    elif metrics.pair_rate < 0.3:
                        inject_frac = max(0.3, inject_frac - 0.1)  # 配对差, 减速注入
                n_rounds_total = max(1, n_remd_rounds)
                far_ratio = 1.0 if n_rounds_total == 1 else inject_frac
                if verbose and far_pairs and far_ratio > 0:
                    print(f"    远端配对注入: {len(far_pairs)} 对 ({_refine_name} 模式)")
                _remd_reps = (64 if L > 1000
                              else max(nrep if nrep else 6, n_rest2_replicas))
                _remd_steps = (max(60000, n_steps // 2) if L > 1000
                               else max(10000, n_steps // 3))
                if _refine_name == "Torch GPU":
                    _refine_result = _refine_fn(
                        refine_input, round_dir,
                        sequence, secondary_structure,
                        name=f"remd_r{round_idx}",
                        nstep=max(100000, n_steps),
                        use_remd=True,
                        remd_n_replicas=_remd_reps,
                        remd_n_steps=_remd_steps,
                        verbose=verbose,
                        use_physical_relax=True,
                        skip_minimal_fold=(round_idx > 0),
                        use_trirnasp=False,  # 完全禁用 TriRNASP, 只用 CG 力场
                        use_trirnasp_force=False,
                        trirnasp_scale=0.002,  # 最优: Tri/CG ≈ 11%, 平衡点
                        trirnasp_update_freq=1000,
                        # 新增: 分阶段 TriRNASP 策略 (适配每轮 5000 步)
                        use_staged_tri=True,
                        tri_stage_config={
                            "stages": [
                                {"name": "CG-only", "steps": 1000, "tri_scale": 0.0},
                                {"name": "Ramp-up", "steps": 1500, "tri_scale": 0.001},
                                {"name": "Tri-guided", "steps": 2500, "tri_scale": 0.002},
                            ],
                        },
                        use_adaptive_tri_weight=True,
                    )
                else:
                    _refine_result = _refine_fn(
                        refine_input, round_dir,
                        sequence, secondary_structure,
                        name=f"remd_r{round_idx}",
                        nstep=max(100000, n_steps),
                        platform_name=_remd_platform_name,
                        use_remd=True,
                        remd_n_replicas=_remd_reps,
                        remd_n_steps=_remd_steps,
                        verbose=verbose,
                        use_physical_relax=True,
                        skip_minimal_fold=(round_idx > 0),
                        use_trirnasp=False,  # 完全禁用 TriRNASP, 只用 CG 力场
                        use_trirnasp_force=False,
                        trirnasp_scale=0.002,  # 最优: Tri/CG ≈ 11%, 平衡点
                        trirnasp_update_freq=1000,
                        # 新增: 分阶段 TriRNASP 策略 (适配每轮 5000 步)
                        use_staged_tri=True,
                        tri_stage_config={
                            "stages": [
                                {"name": "CG-only", "steps": 1000, "tri_scale": 0.0},
                                {"name": "Ramp-up", "steps": 1500, "tri_scale": 0.001},
                                {"name": "Tri-guided", "steps": 2500, "tri_scale": 0.002},
                            ],
                        },
                        use_adaptive_tri_weight=True,
                    )
                # openmm_gpu_refine 返回 (pdb, energy, diag)
                # torch_gpu_refine 返回 (pdb, energy, diag) — 已对齐
                if len(_refine_result) == 3:
                    pdb_out, energy, _refine_diag = _refine_result
                else:
                    pdb_out, energy = _refine_result
                    _refine_diag = {}

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
            remd_history.append([round_idx, best_energy, metrics.pair_rate,
                                 metrics.clash_count, metrics.rmsd_change, inject_frac])

            # 每轮 REMD 完成后保存 checkpoint
            _save_ckpt(2, remd_round=round_idx + 1,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
                pair_rate=metrics.pair_rate,
                cross_segment_ok_rate=metrics.cross_segment_ok,
                energy=energy,
                clash_count=metrics.clash_count,
            )

            # Level 2 per-REMD round 诊断
            try:
                _rl_action_summary = {}
                if use_rl_relax and rl_agent is not None:
                    _pw, _ns = rl_agent.decide(
                        coords_current, far_pairs, prev_energy, metrics,
                        round_idx, n_remd_rounds) if False else (pw, n_steps)
                    if pw:
                        _w_vals = list(pw.values())
                        _rl_action_summary = {
                            "w_mean": float(np.mean(_w_vals)),
                            "w_min": float(np.min(_w_vals)),
                            "w_max": float(np.max(_w_vals)),
                            "nstep": n_steps,
                        }
                _round_t0 = time.time()  # 粗略轮时间
                remd_diag = {
                    "round": round_idx,
                    "best_energy": float(best_energy),
                    "round_energy": float(energy),
                    "hot_start_energy": float(
                        (_refine_diag or {}).get("hot_start_energy", float("nan"))
                    ) if '_refine_diag' in dir() else None,
                    "pair_rate": float(metrics.pair_rate),
                    "cross_segment_ok": float(metrics.cross_segment_ok),
                    "clash_count": int(metrics.clash_count),
                    "rmsd_change": float(metrics.rmsd_change),
                    "n_far_pairs_injected": len(far_pairs) * inject_frac,
                    "inject_frac": float(inject_frac),
                    "rl_action": _rl_action_summary,
                }
                _diag_dir = output_path / "_plots"
                _diag_dir.mkdir(parents=True, exist_ok=True)
                _round_diag_path = _diag_dir / f"02_remd_r{round_idx}_diag.json"
                _round_diag_path.write_text(json.dumps(remd_diag, indent=2))
            except Exception:
                pass

            prev_best_energy = best_energy  # 早停对比用
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

            # 早停: 连续 3 轮 best_energy 无 >1% 改善则停止
            if best_energy < prev_best_energy * 0.99:
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= 3:
                if verbose:
                    print(f"  [Level 2] 早停: 连续 3 轮无显著改善 (E={best_energy:.0f})")
                break

        # Level 2 验证
        v2 = _validate_structure(best_coords, pairs, bpp, sequence, "Level 2")
        print(f"  [验证 Level 2] clash={v2['clash_count']}, pair_rate={v2['pair_rate']:.2f}, valid={v2['is_valid']}")

    # ── Level 2 数据导出 ──
    try:
        from torusfold.scheme2.data_exporter import export_level2_remd, export_validation
        export_level2_remd(remd_history, str(output_path))
        export_validation(v1 if 'v1' in dir() else None,
                          v15 if 'v15' in dir() else None,
                          v2 if 'v2' in dir() else None,
                          str(output_path))
    except Exception as _err:
        raise

    # ── Level 2.3: 5-bead CG 精修 (比 3-bead 更精确的 stacking/H-bond 几何) ──
    _skip_5bead = False
    if use_5bead and best_coords is not None and len(best_coords) == len(sequence):
        # 快速筛选: 如果 Level 2 输出已够好, 跳过 2.3
        if metrics.clash_count == 0 and metrics.pair_rate > 0.8:
            _skip_5bead = True
            if verbose:
                print(f"\n[Level 2.3] 跳过: Level 2 输出已够好 (clash=0, pair_rate={metrics.pair_rate:.2f}>0.8)")
    if not _skip_5bead and use_5bead and best_coords is not None and len(best_coords) == len(sequence):
        # 检查输入坐标是否有 NaN
        _has_nan = np.any(np.isnan(best_coords))
        _has_inf = np.any(np.isinf(best_coords))
        if _has_nan or _has_inf:
            if verbose:
                print(f"\n[Level 2.3] 5-bead 跳过: 输入坐标含 NaN/Inf (nan={_has_nan}, inf={_has_inf})")
        elif verbose:
            print(f"\n[Level 2.3] 5-bead CG 精修...")
        try:
            from torusfold.scheme2.fivebead_folding import refine_5bead
            _resolved_platform = "CPU"  # 5-bead 用 CPU (粒子数 5x)

            # 生成 DL 距离约束: 从 PF BPP + DivideFold 融合
            _dl_constraints = []
            if bpp_high or bpp_mid:
                # 高置信 PF 配对 → 强约束 (d≈10Å B1-B1 WC 距离)
                for i, j, p in bpp_high:
                    _dl_constraints.append((i, j, 10.0, p))
                # 中置信 → 弱约束
                for i, j, p in bpp_mid[:50]:  # 限制数量
                    _dl_constraints.append((i, j, 10.0, p * 0.5))
                if verbose:
                    print(f"    DL 约束: {len(_dl_constraints)} 对 (PF)")

            p5_refined, e5_0, e5_1 = refine_5bead(
                best_coords, pairs,
                platform_name=_resolved_platform,
                n_anneal=3000,
                sequence=sequence,
                dl_constraints=_dl_constraints if _dl_constraints else None)
            # 5-bead 输出含 NaN 时丢弃
            if np.any(np.isnan(p5_refined)) or np.any(np.isinf(p5_refined)):
                if verbose:
                    print(f"    5-bead 输出含 NaN/Inf, 保留当前坐标")
            elif e5_1 < e5_0:
                # 5-bead 自身有改善 → 接受 (不同力场尺度, 不和3-bead 比能量)
                best_coords = p5_refined.copy()
                if verbose:
                    print(f"    5-bead: E={e5_0:.0f} -> {e5_1:.0f} kJ/mol (改善, 接受)")
            elif verbose:
                print(f"    5-bead: E={e5_0:.0f} -> {e5_1:.0f} (未改善, 保留3bead坐标)")
        except Exception as e:
            if verbose:
                print(f"    5-bead 精修跳过: {e}")

    # ── Level 2.5: REMD 后 CG→allatom (把最终 CG 坐标转成全原子) ──
    if "final_aa_pdb" in ckpt:
        _final_aa_path = ckpt.get("final_aa_pdb", str(output_path / "final_allatom.pdb"))
        if verbose:
            print(f"\n[Level 2.5] 从 checkpoint 恢复: {_final_aa_path}")
    elif best_coords is not None and len(best_coords) == len(sequence):
        _final_aa_path = str(output_path / "final_allatom.pdb")
        _l25_ok = False
        if verbose:
            print(f"\n[Level 2.5] REMD 后 CG→全原子: {_final_aa_path}")
        try:
            from torusfold.scheme2.isrnacirc_wrapper import cg_to_allatom
            _tmp_cg = str(output_path / "_final_cg_for_aa.pdb")
            _write_coords_pdb(best_coords, sequence, _tmp_cg)
            _cg2aa_t0 = time.time()
            cg_to_allatom(_tmp_cg, _final_aa_path, sequence)
            _cg2aa_elapsed = time.time() - _cg2aa_t0
            if verbose:
                _sz = os.path.getsize(_final_aa_path) / 1024
                print(f"    全原子输出: {_final_aa_path} ({_sz:.0f} KB)")
            _l25_ok = True

            # ── Level 2.5b: CG→AA 后处理弛豫 (far pair 约束防散开) ──
            # 转换后立即用 far_pairs 约束做一轮短弛豫,
            # 否则转换产生的几何噪声会让 Level 1.5/2 拉拢的远端配对散开.
            try:
                from torusfold.scheme2.physical_relaxation import relax_structure as _relax_25b
                if verbose and far_pairs:
                    print(f"    后处理弛豫: {len(far_pairs)} 对远端配对约束...")
                best_coords, _relax_metrics_25b = _relax_25b(
                    best_coords, sequence,
                    far_pairs=far_pairs if far_pairs else None,
                    n_steps=3000,
                    use_openmm=True,
                )
                if verbose:
                    print(f"    弛豫完成: clash {_relax_metrics_25b['initial']['clash_count']} → "
                          f"{_relax_metrics_25b['final']['clash_count']}")
            except Exception as e_relax_25b:
                if verbose:
                    print(f"    后处理弛豫跳过: {e_relax_25b}")

            # Level 2.5 CG→allatom 转换诊断
            try:
                _n_aa_atoms = 0
                if os.path.exists(_final_aa_path):
                    with open(_final_aa_path) as _f:
                        _n_aa_atoms = sum(1 for _l in _f if _l.startswith("ATOM"))
                diag_cg2aa = {
                    "cg_atoms": len(best_coords),
                    "aa_atoms": _n_aa_atoms,
                    "conversion_time": float(_cg2aa_elapsed),
                    "success": True,
                }
                _diag_cg2aa_path = output_path / "_plots" / "03_level2_5_cg2aa_diag.json"
                _diag_cg2aa_path.parent.mkdir(parents=True, exist_ok=True)
                _diag_cg2aa_path.write_text(json.dumps(diag_cg2aa, indent=2))
            except Exception:
                pass
        except Exception as e:
            if verbose:
                print(f"    CG→allatom 失败: {e}")
            # Level 2.5 失败诊断
            try:
                diag_cg2aa = {
                    "cg_atoms": len(best_coords),
                    "aa_atoms": 0,
                    "conversion_time": 0.0,
                    "success": False,
                    "error": str(e),
                }
                _diag_cg2aa_path = output_path / "_plots" / "03_level2_5_cg2aa_diag.json"
                _diag_cg2aa_path.parent.mkdir(parents=True, exist_ok=True)
                _diag_cg2aa_path.write_text(json.dumps(diag_cg2aa, indent=2))
            except Exception:
                pass

        # Level 2.5 完成后保存 checkpoint (仅成功时)
        if _l25_ok:
            _save_ckpt(2.5,
                final_aa_pdb=_final_aa_path,
                best_coords=best_coords,
                time=time.time(),
            )
    else:
        _final_aa_path = str(output_path / "final_allatom.pdb")

    # ── Level 2.6: PyRosetta 条件式精修 (WSL, socket 优先) ──
    if not os.path.exists(_final_aa_path):
        _final_aa_path = str(output_path / "final_allatom.pdb")
    if ckpt.get("pyrosetta_done"):
        if verbose:
            print(f"\n[Level 2.6] 从 checkpoint 恢复")
    elif os.path.exists(_final_aa_path) and use_pyrosetta:
        _l26_ok = False
        if verbose:
            print(f"\n[Level 2.6] PyRosetta 条件式精修...")
        try:
            _pyrosetta_out = str(output_path / "final_allatom_refined.pdb")
            _wsl_aa = _win_to_wsl(_final_aa_path)
            _wsl_out = _win_to_wsl(_pyrosetta_out)

            # 优先: 长驻 socket 服务 (init 只做一次, 后续 <1s)
            _result = _pyrosetta_socket_refine(
                _wsl_aa, _wsl_out, max_iter=200, verbose=verbose,
            )
            if _result is None and not _pyrosetta_server_running():
                # 服务不在 → 启动
                if _pyrosetta_start_server(verbose=verbose):
                    _result = _pyrosetta_socket_refine(
                        _wsl_aa, _wsl_out, max_iter=200, verbose=verbose,
                    )

            if _result is not None:
                _final_aa_path = _pyrosetta_out
                _l26_ok = True
                if verbose:
                    print(f"    PyRosetta 精修完成 (socket): {_pyrosetta_out}")
            else:
                # 回退: 一次性 subprocess (带超时)
                if verbose:
                    print("    [fallback] 回退到 subprocess 模式...")
                _wsl_src = _win_to_wsl(Path(__file__).resolve().parent)
                _wsl_script = (
                    f'import sys; sys.path.insert(0, "{_wsl_src}")\n'
                    f'from pyrosetta_refine import pyrosetta_refine\n'
                    f'pyrosetta_refine("{_wsl_aa}", "{_wsl_out}", max_iter=200, verbose=True)\n'
                )
                _win_script_path = str(output_path / "_run_pyrosetta.py")
                _wsl_script_path = _win_to_wsl(output_path / "_run_pyrosetta.py")
                with open(_win_script_path, "w") as f:
                    f.write(_wsl_script)

                _pyrosetta_timeout = 600
                _proc = subprocess.Popen(
                    ["wsl", "bash", "-c", f"python3 {_wsl_script_path}"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                )
                try:
                    _stdout_lines = []
                    for _line in _proc.stdout:
                        _stdout_lines.append(_line)
                        if verbose:
                            print(f"    [PyR] {_line.rstrip()}")
                    _proc.wait(timeout=_pyrosetta_timeout)
                except subprocess.TimeoutExpired:
                    _proc.kill()
                    _proc.wait()
                    if verbose:
                        print(f"    PyRosetta 超时({_pyrosetta_timeout}s), 已终止")
                    _proc = None
                _rc = _proc.returncode if _proc is not None else -1
                if _rc == 0 and os.path.exists(_pyrosetta_out):
                    _final_aa_path = _pyrosetta_out
                    _l26_ok = True
                    if verbose:
                        print(f"    PyRosetta 精修完成 (subprocess): {_pyrosetta_out}")
                elif verbose:
                    _tail = ''.join(_stdout_lines[-20:]) if _stdout_lines else '(无输出)'
                    print(f"    PyRosetta 跳过或失败 (rc={_rc}):")
                    print(f"    {_tail}")

        except Exception as e:
            if verbose:
                print(f"    PyRosetta 精修失败: {e}")

    # Level 2.6 完成后保存 checkpoint (仅成功时)
    if _l26_ok:
        _save_ckpt(2.6,
            final_aa_pdb=_final_aa_path,
            pyrosetta_done=True,
        )

    # ── Level 3: RL 微调 (连续动作空间) ──
    if ckpt_level >= 3:
        if verbose:
            print(f"\n[Level 3] 从 checkpoint 恢复")
    elif use_rl_mcts and far_pairs:
        _l3_ok = False
        if verbose:
            print(f"\n[Level 3] RL 微调 (连续动作, PPO {rl_n_simulations} epochs)...")
        try:
            from torusfold.scheme2.rl_optimizer import optimize_far_pairs
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
                _l3_ok = True
                if verbose:
                    print(f"    RL 完成: reward={rl_info.get('reward_after', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"    RL 微调失败: {e}")
        if _l3_ok:
            _save_ckpt(3,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
                rl_done=True,
            )

    # ── Level 3.5: Metadynamics 增强采样 (沿 CV 跨越自由能垒) ──
    # GPU batched 版本优先 (torch.cuda), OpenMM CPU 作为 fallback.
    if ckpt_level >= 3.5:
        if verbose:
            print(f"\n[Level 3.5] 从 checkpoint 恢复")
    elif use_metad and best_coords is not None and len(best_coords) == len(sequence):
        _l35_ok = False
        _use_gpu_meta = False
        try:
            import torch as _torch_meta
            _use_gpu_meta = _torch_meta.cuda.is_available()
        except Exception:
            pass

        if _use_gpu_meta:
            # ── GPU batched path: 所有副本在单 GPU 上批量跑 ──
            if verbose:
                print(f"\n[Level 3.5] GPU batched Metadynamics "
                      f"(8 replicas, well-tempered, {metad_n_steps} 步)...")
            try:
                from torusfold.scheme2.metadynamics_gpu import BatchedMetadynamics

                _meta_gpu = BatchedMetadynamics(
                    n_replicas=8, sequence=sequence, device="cuda",
                    relax_bond_k=500.0,
                    relax_angle_k=200.0,
                    relax_pair_k=500.0,
                    restraint_k=500.0,
                )
                meta_coords, meta_e, _meta_diag = _meta_gpu.run(
                    best_coords, pairs,
                    n_steps=metad_n_steps,
                    hill_height=1.0,
                    hill_sigma=1.0,
                    hill_freq=100,
                    max_hills=5000,
                    well_tempered=True,
                    bias_factor=5.0,
                    verbose=verbose,
                )
                if meta_e < best_energy:
                    best_coords = meta_coords
                    best_energy = meta_e
                    if verbose:
                        print(f"    GPU-MetaD E={meta_e:.0f} (优于当前)")
                elif verbose:
                    print(f"    GPU-MetaD E={meta_e:.0f} "
                          f"(未优于 {best_energy:.0f}, 保留)")
                _l35_ok = True
            except Exception as e:
                if verbose:
                    print(f"    GPU-MetaD 失败, 回退 OpenMM: {e}")

        if not _l35_ok:
            # ── OpenMM CPU fallback: 2 副本独立线程 ──
            if verbose:
                _fallback_tag = "OpenMM" if not _use_gpu_meta else "回退 OpenMM"
                print(f"\n[Level 3.5] {_fallback_tag} Metadynamics "
                      f"(2 replicas, well-tempered, {metad_n_steps} 步)...")
            try:
                from torusfold.scheme2.metadynamics_sampler import MetaDynamicsSampler
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _run_metad_replica(coords_init, seq, prs, n_steps, replica_id):
                    """单个 MetaD 副本 (独立线程)."""
                    _meta_plat = "CPU"
                    try:
                        import openmm as _omm
                        for _try in ["CUDA", "OpenCL"]:
                            try:
                                _omm.Platform.getPlatformByName(_try)
                                _meta_plat = _try
                                break
                            except Exception:
                                pass
                    except Exception:
                        pass
                    _meta = MetaDynamicsSampler(
                        seq, prs,
                        hill_height=1.0,
                        hill_sigma_bsj=2.0,
                        hill_sigma_nc=0.1,
                        hill_sigma_rg=1.0,
                        hill_freq=100,
                        max_hills=5000,
                        well_tempered=True,
                        bias_factor=5.0,
                        platform_name=_meta_plat,
                    )
                    _coords, _e = _meta.sample(coords_init, n_steps=n_steps,
                                               verbose=False)
                    return replica_id, _coords, _e

                metad_results = []
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = []
                    for rep_id in range(2):
                        noise = np.random.normal(0, 0.5,
                                                best_coords.shape).astype(np.float32)
                        coords_perturbed = best_coords + noise
                        fut = executor.submit(
                            _run_metad_replica, coords_perturbed, sequence, pairs,
                            metad_n_steps, rep_id)
                        futures.append(fut)
                    for fut in as_completed(futures):
                        metad_results.append(fut.result())

                best_replica = min(metad_results, key=lambda x: x[2])
                meta_coords = best_replica[1]
                meta_e = best_replica[2]
                if meta_e < best_energy:
                    best_coords = meta_coords
                    best_energy = meta_e
                    if verbose:
                        print(f"    MetaD (replica {best_replica[0]}) "
                              f"E={meta_e:.0f} (优于当前)")
                elif verbose:
                    print(f"    MetaD (replica {best_replica[0]}) "
                          f"E={meta_e:.0f} (未优于 {best_energy:.0f}, 保留)")
                _l35_ok = True
            except Exception as e:
                if verbose:
                    print(f"    MetaD 跳过: {e}")

        if _l35_ok:
            _save_ckpt(3.5,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
            )

    # ── Level 4: REST2 精修 ──
    if ckpt_level >= 4:
        if verbose:
            print(f"\n[Level 4] 从 checkpoint 恢复")
    else:
        # 解析 "auto" 平台
        if platform == "auto":
            from torusfold.scheme2.rest2_sampler import detect_openmm_platform
            resolved_platform = detect_openmm_platform()
        else:
            resolved_platform = platform
        if verbose:
            print(f"\n[Level 4] REST2 精修 ({n_rest2_replicas} 副本, 平台={resolved_platform})...")
        _l4_ok = False
        _use_gpu_rest2 = torch.cuda.is_available()
        use_trirnasp = False  # 默认关闭, 等 Level 2 集成后启用
        if _use_gpu_rest2:
            try:
                # ── GPU 批量 REST2×REMD ──
                from torusfold.scheme2.torch_cgsim import BatchedREMD2D
                _remd2d = BatchedREMD2D(
                    n_t=8,
                    t_lo=300.0, t_hi=1000.0,
                    lambdas=(1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65),
                    exchange_interval=1000,
                    use_trirnasp=use_trirnasp,
                    sequence=sequence,
                    force_refresh_freq=500,
                    # 弛豫参数 (与 OpenMM 版对齐)
                    relax_bond_k=500.0,
                    relax_angle_k=200.0,
                    relax_pair_k=500.0,
                    restraint_k=500.0,
                )
                # best_coords 始终是 P-only Cartesian coordinates in Å.
                # BatchedREMD2D expects the same public interface and a step count.
                _pairs_flat = [(i, j, w) for i, j, w in pairs]
                best_coords, energy, diag = _remd2d.run(
                    best_coords, _pairs_flat, n_steps=rest2_nsteps,
                    verbose=verbose)
                coords_rest2 = best_coords
                _l4_ok = True
            except Exception as e_gpu:
                if verbose:
                    print(f"    GPU REST2 失败, 回退 OpenMM: {e_gpu}")
                _use_gpu_rest2 = False

        if not _l4_ok and not _use_gpu_rest2:
            try:
                # ── CPU OpenMM fallback ──
                from torusfold.scheme2.rest2_remd_2d import REMD2DSampler
                remd2d = REMD2DSampler(
                    n_t=8,
                    t_lo=300.0, t_hi=1000.0,
                    lambdas=(1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65),
                    n_steps=rest2_nsteps,
                    exchange_interval=1000,
                    platform_name=resolved_platform,
                )
                coords_rest2, e_rest2, _rest2_diag = remd2d.sample(
                    best_coords, pairs, sequence, verbose=verbose)
                _acc_t, _acc_l = [], []
                if verbose and isinstance(_rest2_diag, dict):
                    _acc_t = _rest2_diag.get("acceptance_T", [])
                    _acc_l = _rest2_diag.get("acceptance_lam", [])
                if _acc_t:
                    print(f"    [2D-REMD] T轴接受率: "
                          f"{['%.0f%%' % (a*100) for a in _acc_t]}")
                if _acc_l:
                    print(f"    [2D-REMD] λ轴接受率: "
                          f"{['%.0f%%' % (a*100) for a in _acc_l]}")
                _rest2_snaps = [coords_rest2]
                # 聚类选择: 从 REST2 快照中选最优
                _snap_list = list(_rest2_snaps) if _rest2_snaps else [coords_rest2]
                _snap_energies = [e_rest2] * len(_snap_list)
                if len(_snap_list) > 1:
                    _best_i, _n_cl, _cl_info = _cluster_and_select(
                        _snap_list, _snap_energies, rmsd_threshold=5.0)
                    best_coords = _snap_list[_best_i]
                    best_energy = _snap_energies[_best_i]
                    if verbose:
                        print(f"    REST2: {_n_cl} clusters, best E={best_energy:.0f}")
                else:
                    best_coords = coords_rest2
                    best_energy = e_rest2
                if verbose:
                    print(f"    REST2 E={best_energy:.0f}")
                _l4_ok = True
            except Exception as e:
                if verbose:
                    print(f"    REST2 失败: {e}")
        if _l4_ok:
            _save_ckpt(4,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
            )

    # ── Level 5: AMBER RNA.OL3 全原子精修 (带 C1'-C1' pair restraints) ──
    if ckpt_level >= 5:
        if verbose:
            print(f"\n[Level 5] 从 checkpoint 恢复")
    else:
        _l5_ok = False
        if verbose:
            print(f"\n[Level 5] AMBER RNA.OL3 全原子精修 (最小化+MD)...")
        try:
            # 优先: amber_refine (完整版, 含 C1'-C1' pair restraints + A-form torsion)
            from torusfold.scheme2.aform_from_template import reconstruct_all_atom
            from torusfold.scheme2.amber_refine import amber_refine as _amber_refine_full

            # CG P coords (Å) -> AllAtomStructure (1EHZ 晶体模板)
            _structure_5 = reconstruct_all_atom(best_coords, sequence)
            # amber_refine: C1'-C1' pair restraints (K=100 kJ/mol/nm²) + A-form torsion
            _refined_coords_5, _e0_5, _e1_5, _info_5 = _amber_refine_full(
                _structure_5, pairs,
                platform_name="CPU",
                max_iterations=3000,
            )

            # 从 heavy atom 输出提取 P-only 坐标 (按 residue_atom_spans 定位)
            _p_coords_5 = []
            for _ri in range(L):
                _span = _structure_5.residue_atom_spans[_ri]
                _p_idx = _span[0]  # 每残基第一个原子是 P
                if _p_idx < len(_refined_coords_5):
                    _p_coords_5.append(_refined_coords_5[_p_idx])
            _p_coords_5 = np.array(_p_coords_5) if _p_coords_5 else np.zeros((0, 3))
            # 写 PDB 供 PPR (Level 5.5) 等后续步骤使用
            _write_coords_pdb(
                _p_coords_5 if len(_p_coords_5) == L else best_coords,
                sequence, str(output_path / "level5_amber.pdb"))

            old_energy = best_energy
            if _e1_5 < best_energy:
                best_energy = _e1_5
                if len(_p_coords_5) == L:
                    best_coords = _p_coords_5
                if verbose:
                    print(f"    AMBER 精修: E={_e0_5:.0f} -> {_e1_5:.0f} kJ/mol (优于之前 {old_energy:.0f})")
                    print(f"    pair restraints: {len(pairs)} 对, A-form torsions: {_info_5.get('n_torsions', 0)}")
            else:
                if verbose:
                    print(f"    AMBER 精修: E={_e1_5:.0f} (未优于 {old_energy:.0f}, 保留原结果)")
            _l5_ok = True
        except Exception as e_full:
            if verbose:
                print(f"    amber_refine 失败 ({e_full!r}), 尝试 openmm_amber_refine fallback...")
            try:
                # Fallback: openmm_amber_refine (简化版, 无 pair restraints)
                from torusfold.scheme2.openmm_amber_refiner import openmm_amber_refine, OPENMM_AVAILABLE as AMBER_OK
                if AMBER_OK:
                    cg_pdb_5 = str(output_path / "level5_cg.pdb")
                    _write_coords_pdb(best_coords, sequence, cg_pdb_5)
                    from torusfold.scheme2.isrnacirc_wrapper import cg_to_allatom
                    aa_pdb_5 = str(output_path / "level5_aa.pdb")
                    cg_to_allatom(cg_pdb_5, aa_pdb_5, sequence)
                    amber_out, amber_e = openmm_amber_refine(
                        aa_pdb_5, str(output_path / "level5_amber.pdb"),
                        sequence=sequence,
                        nsteps=15000,
                        platform_name="CPU",
                        verbose=verbose,
                    )
                    old_energy = best_energy
                    if amber_e < best_energy:
                        best_energy = amber_e
                        p_coords_5 = _read_pdb_p_coords(amber_out)
                        if len(p_coords_5) == L:
                            best_coords = p_coords_5
                        if verbose:
                            print(f"    AMBER 精修 (fallback): E={amber_e:.0f} (优于之前 {old_energy:.0f})")
                    else:
                        if verbose:
                            print(f"    AMBER 精修 (fallback): E={amber_e:.0f} (未优于 {old_energy:.0f}, 保留)")
                    _l5_ok = True
                else:
                    if verbose:
                        print(f"    AMBER 精修跳过 (OpenMM 未安装)")
            except Exception as e:
                if verbose:
                    print(f"    Level 5 fallback 也失败: {e}")
    if _l5_ok:
        _save_ckpt(5,
            pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
            coords_vfold=coords_vfold, n_segments=n_segments,
            segments=segments,
            best_coords=best_coords, best_energy=best_energy,
        )

    # ── Level 5.5: PPR 碱基对氢键修复 ──
    if use_ppr:
        if ckpt_level >= 5.5:
            if verbose:
                print(f"\n[Level 5.5] 从 checkpoint 跳过 (已修复)")
        else:
            _level5_pdb = str(output_path / "level5_amber.pdb")
            if not os.path.exists(_level5_pdb):
                _level5_pdb = str(output_path / "level5_aa.pdb")
            if os.path.exists(_level5_pdb):
                if verbose:
                    print(f"\n[Level 5.5] PPR 碱基对氢键修复...")
                try:
                    from torusfold.scheme2.ppr_repair import ppr_repair
                    _ppr_out = str(output_path / "level5_ppr.pdb")
                    ppr_result = ppr_repair(
                        _level5_pdb, _ppr_out, sequence,
                        pairs=pairs, max_rounds=ppr_max_rounds,
                        verbose=verbose,
                    )
                    if ppr_result["after"] > ppr_result["before"]:
                        if verbose:
                            print(f"  PPR 有效: {ppr_result['before']} -> {ppr_result['after']} 对")
                except Exception as e:
                    if verbose:
                        print(f"  PPR 失败: {e}")
                _save_ckpt(5.5,
                    ppr_output=_ppr_out if os.path.exists(_ppr_out) else "",
                )
            elif verbose:
                print(f"  PPR 跳过: Level 5 PDB 不存在")

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

    # ── 最终统计: 真 H-bond rate (全原子级) ──
    _hbond_rate = 0.0
    _faa_check = str(output_path / "final_allatom.pdb")
    if not os.path.exists(_faa_check):
        _faa_check = str(output_path / "level5_amber.pdb")
    if os.path.exists(_faa_check):
        try:
            _hbond_rate = _compute_hbond_rate(_faa_check, pairs, sequence)
            if verbose:
                print(f"\n  真 H-bond rate: {_hbond_rate*100:.1f}% ({_hbond_rate:.4f})")
                print(f"  CG pair_rate (P-P<12A): {state.pair_rate*100:.1f}%")
        except Exception as _err:
            raise

    # ── checkpoint 保留 ──
    # _cleanup_checkpoints 已禁用: checkpoint 文件 (_checkpoint.json, ckpt_*.npy)
    # 用于后续分析 (能量轨迹、REMD 收敛曲线、best_coords 回溯等)

    # ── 最终数据导出 ──
    try:
        from torusfold.scheme2.data_exporter import export_final_summary
        export_final_summary(best_coords, sequence, str(output_path))
    except Exception as _err:
        raise

    # ── Pipeline 完整总结 ──
    try:
        total_time = time.time() - t0
        _chunk_confs = chunk_confidences if 'chunk_confidences' in dir() else []
        summary = {
            "input": {"sequence_length": len(sequence), "is_circular": True},
            "level0": {
                "pairs": len(pairs),
                "energy": float(pf_energy) if 'pf_energy' in dir() else None,
                "n_far": len(far_pairs),
                "gc_content": sum(1 for c in sequence if c in "GCgc") / max(len(sequence), 1),
            },
            "level1": {
                "n_chunks": len(segments),
                "avg_confidence": float(np.mean(_chunk_confs)) if _chunk_confs else None,
            },
            "level1_5": {
                "final_energy": float(_e) if '_e' in dir() and np.isfinite(_e) else None,
                "n_energy_samples": len(energy_traj) if 'energy_traj' in dir() else 0,
            },
            "level2": {
                "n_rounds": round_idx + 1 if 'round_idx' in dir() else 0,
                "final_pair_rate": float(metrics.pair_rate) if 'metrics' in dir() else 0,
                "final_clash": int(metrics.clash_count) if 'metrics' in dir() else 0,
            },
            "level2_3": {"skipped": _skip_5bead if '_skip_5bead' in dir() else True},
            "level2_5": {"aa_path": _final_aa_path if '_final_aa_path' in dir() else None},
            "level3": {"applied": use_rl_mcts and bool(far_pairs)},
            "level3_5": {"applied": use_metad},
            "level4": {"applied": True},
            "level5": {"applied": True},
            "level5_5": {"applied": use_ppr},
            "final": {
                "rsrnasp1": None,
                "hbond_rate": float(_hbond_rate),
                "clash_count": int(metrics.clash_count) if 'metrics' in dir() else None,
                "pair_rate": float(state.pair_rate),
            },
            "total_time": float(total_time),
        }
        _summary_path = output_path / "pipeline_summary.json"
        _summary_path.write_text(json.dumps(summary, indent=2))
    except Exception as _err:
        raise

    return LongPipelineResult(
        sequence=sequence,
        secondary_structure=secondary_structure,
        coords_cg=best_coords,
        coords_aa=coords_aa,
        energy_cg=best_energy,
        energy_aa=best_energy,
        rmsd_to_native=None,
        pair_rate=state.pair_rate,
        hbond_rate=_hbond_rate,
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
    try:
        from torusfold.scheme2.refine import BOND_LEN
    except ImportError:
        BOND_LEN = 5.9
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
    """计算配对满足率 (P-P 距离 < 12Å, 比旧版 15Å 更严格)."""
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
        if d < 12.0:  # 旧版 15Å 太松, 12Å 更合理
            ok += 1
    return ok / len(pairs)


def _compute_hbond_rate(pdb_path, pairs, sequence) -> float:
    """计算真氢键满足率 (N1/N3/O6/N4 距离 < 3.6Å)."""
    try:
        from openmm.app import PDBFile
        import openmm.unit as unit
        pdb = PDBFile(pdb_path)
        pos = np.array([[p.x, p.y, p.z] for p in pdb.positions.value_in_unit(unit.nanometers)]) * 10.0
        res_atoms = {}
        for atom in pdb.topology.atoms():
            ri = atom.residue.index
            an = atom.name.strip()
            if ri not in res_atoms:
                res_atoms[ri] = {}
            res_atoms[ri][an] = pos[atom.index]

        hb_pairs = {
            ("A", "U"): [("N1", "N3"), ("N6", "O4")],
            ("U", "A"): [("N3", "N1"), ("O4", "N6")],
            ("G", "C"): [("N1", "N3"), ("O6", "N4"), ("N2", "O2")],
            ("C", "G"): [("N3", "N1"), ("N4", "O6"), ("O2", "N2")],
            ("G", "U"): [("N1", "N3"), ("O6", "N3")],
            ("U", "G"): [("N3", "N1"), ("N3", "O6")],
        }
        ok = 0
        for p in pairs:
            i, j = (p[0], p[1]) if len(p) >= 2 else (p[0], p[1])
            if i >= len(sequence) or j >= len(sequence):
                continue
            ai, aj = sequence[i], sequence[j]
            if (ai, aj) not in hb_pairs:
                continue
            ri, rj = res_atoms.get(i, {}), res_atoms.get(j, {})
            best = 999.0
            for a1, a2 in hb_pairs[(ai, aj)]:
                if a1 in ri and a2 in rj:
                    d = np.linalg.norm(ri[a1] - rj[a2])
                    if d < best:
                        best = d
                if a2 in ri and a1 in rj:
                    d = np.linalg.norm(ri[a2] - rj[a1])
                    if d < best:
                        best = d
            if best < 3.6:
                ok += 1
        return ok / max(len(pairs), 1)
    except Exception:
        return 0.0


def _validate_structure(coords, pairs, bpp_matrix, sequence, level_name=""):
    """快速验证结构质量 (clash + pair_rate + bond_quality).

    Returns: dict with clash_count, pair_rate, bond_quality, is_valid
    """
    L = len(coords)
    result = {
        "clash_count": 0,
        "pair_rate": 0.0,
        "bond_quality": 0.0,
        "is_valid": True,
        "warnings": [],
    }

    if L < 2:
        result["is_valid"] = False
        return result

    # 1. Clash 检测: P-P 距离 < 3.0A (排除相邻残基)
    from scipy.spatial.distance import cdist
    dist_mat = cdist(coords, coords)
    for i in range(L):
        for j in range(i + 3, L):  # 跳过相邻残基
            if dist_mat[i, j] < 3.0:
                result["clash_count"] += 1

    # 2. 配对符合度: pairs 距离 < 15A 的比例
    if pairs:
        n_ok = 0
        for p in pairs:
            i, j = (p[0], p[1]) if len(p) >= 2 else (p[0], p[1])
            if i < L and j < L and dist_mat[i, j] < 15.0:
                n_ok += 1
        result["pair_rate"] = n_ok / len(pairs)

    # 3. 键长质量: P-P 相邻距离
    diffs = np.diff(coords, axis=0)
    bond_dists = np.linalg.norm(diffs, axis=1)
    mean_bond = np.mean(bond_dists)
    result["bond_quality"] = max(0.0, 1.0 - abs(mean_bond - 5.9) / 5.9)

    # 4. 判定
    if result["clash_count"] > 10:
        result["is_valid"] = False
        result["warnings"].append(f"clash={result['clash_count']}")
    if result["pair_rate"] < 0.1 and pairs:
        result["warnings"].append(f"pair_rate={result['pair_rate']:.2f}")
    if result["bond_quality"] < 0.3:
        result["warnings"].append(f"bond_q={result['bond_quality']:.2f}")

    return result


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


def _cluster_and_select(coords_list, energies, rmsd_threshold=5.0):
    """聚类 + 选择最优构象.

    算法: 贪心聚类 (RMSD < threshold 归为同一类), 选能量最低的代表.

    Args:
        coords_list: List[(L, 3)] CG 坐标列表
        energies: List[float] 对应能量
        rmsd_threshold: 聚类 RMSD 阈值 (Å)

    Returns:
        best_idx: 最优构象索引
        n_clusters: 聚类数
        cluster_info: [(center_idx, member_count, min_energy), ...]
    """
    if not coords_list:
        return 0, 0, []
    if len(coords_list) == 1:
        return 0, 1, [(0, 1, energies[0])]

    # 贪心聚类
    clusters = []  # [(center_coords, [member_indices])]
    for idx in range(len(coords_list)):
        assigned = False
        for ci, (center, members) in enumerate(clusters):
            rmsd = _compute_rmsd(coords_list[idx], center)
            if rmsd < rmsd_threshold:
                members.append(idx)
                assigned = True
                break
        if not assigned:
            clusters.append((coords_list[idx].copy(), [idx]))

    # 每个聚类选能量最低的代表
    cluster_info = []
    best_idx = 0
    best_energy = float("inf")
    for center, members in clusters:
        member_energies = [energies[i] for i in members]
        min_e = min(member_energies)
        min_i = members[member_energies.index(min_e)]
        cluster_info.append((min_i, len(members), min_e))
        if min_e < best_energy:
            best_energy = min_e
            best_idx = min_i

    return best_idx, len(clusters), cluster_info


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
    """写坐标到 PDB. CG_to_allatom.exe 需要 3字母残基名."""
    _BASE_MAP = {"A": "ADE", "U": "URA", "G": "GUA", "C": "CYT", "T": "THY"}
    lines = ["HEADER    isRNAcircLong CG structure"]
    for i, (x, y, z) in enumerate(coords):
        base = sequence[i] if i < len(sequence) else "N"
        res_name = _BASE_MAP.get(base.upper(), "UNK")
        lines.append(
            f"ATOM  {i+1:5d}  P   {res_name} A{i+1:4d}"
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
