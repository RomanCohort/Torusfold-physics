"""
rl_optimizer.py - RL 远端配对优化器 (MCTS + 策略网络)。

RL 不替代物理管线, 只补 circRNA 长程配对的局部最优盲区。在 CG 粒度
(P 坐标) 上用 MCTS 探索跳出局部解, 把远端配对拉拢到 WC 几何 (C1'-C1'
~10.5 Å), 再交给已有物理管线 (1EHZ 重建 + amber 精修) 收敛局部几何。

架构 (见 docs/scheme2_rl_design.md):
  - 状态: 远端配对块小图 (节点=茎块, 边=块间拓扑距离)
  - 策略网络: 3 层 GNN + 动作头 (π_block, π_dir, π_step)
  - 动作: (块索引, 6 方向, 3 步长) 离散
  - reward: Σ exp(-|d_C1'C1' - 10.5| / 2) over 远端配对
  - MCTS: policy 先验 + rollout 跑短 CG 精修评估

训练: PPO + GAE (training/ 单独脚本)。
推理: 加载权重, MCTS 搜索输出优化后 P 坐标。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# torch 惰性导入 (rl_optimizer 可能在不训练时被 import, 避免强依赖)
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------- 常量 ----------
# 动作空间 (离散)
N_DIRECTIONS = 6   # ±x ±y ±z
N_STEPS = 3        # 步长档: 0.5, 2.0, 5.0 Å
STEP_SIZES = (0.5, 2.0, 5.0)
DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.float32)

WC_TARGET_DIST = 10.5  # Å, Watson-Crick C1'-C1' 目标距离


# ---------- 状态表示 ----------
@dataclass
class BlockState:
    """一个远端配对茎块的状态。"""
    block_idx: int                 # 块在远端列表中的索引
    residues_i: List[int]          # 块内 i 侧残基索引
    residues_j: List[int]          # 块内 j 侧残基索引
    centroid_i: np.ndarray         # i 侧质心 P 坐标 (3,)
    centroid_j: np.ndarray         # j 侧质心 P 坐标 (3,)
    current_deviation: float       # 当前 C1'-C1' 平均偏差 (Å)


@dataclass
class RLOptimizerState:
    """RL 优化的完整状态。"""
    p_coords: np.ndarray            # (L, 3) CG P 坐标
    sequence: str
    far_blocks: List[BlockState]   # 远端配对块列表
    far_pairs: List[Tuple[int, int]]  # 远端配对 (i, j) 列表
    # 块间邻接 (稀疏): [(block_a, block_b, topo_dist), ...]
    block_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    # coding mask: 透传给下游 amber 精修, coding 区残基钉死
    coding_mask: Optional[np.ndarray] = None  # shape (L,) bool, True=coding


def build_rl_state(
    p_coords: np.ndarray,
    sequence: str,
    far_pairs: List[Tuple[int, int]],
    stem_blocks: List[List[Tuple[int, int]]],
    coding_mask: Optional[np.ndarray] = None,
) -> RLOptimizerState:
    """从 CG P 坐标 + 远端配对 + 茎块构建 RL 状态。

    Args:
        p_coords: (L, 3) CG 求解输出的 P 坐标
        sequence: ACGU 字符串
        far_pairs: 远端配对 [(i, j), ...] (来自 pair_graph.far_end_pairs)
        stem_blocks: 茎块 [[(i, j), ...], ...] (来自 pair_graph.extract_stem_blocks)
        coding_mask: 可选 coding 标注 (来自 pair_graph.parse_case_annotation)
            透传给下游, 不影响 RL 动作空间 (RL 全序列可动)
    """
    # 筛出远端茎块 (块内配对都在 far_pairs 里)
    far_set = set((min(i, j), max(i, j)) for i, j in far_pairs)
    far_blocks: List[BlockState] = []
    for bidx, block in enumerate(stem_blocks):
        # 块内配对是否都在远端集
        in_far = all((min(i, j), max(i, j)) in far_set for i, j in block)
        if not in_far:
            continue
        res_i = [i for i, _ in block]
        res_j = [j for _, j in block]
        ci = p_coords[res_i].mean(axis=0)
        cj = p_coords[res_j].mean(axis=0)
        # 当前偏差: 块内配对 C1'-C1' 均值 (用 P 近似, CG 粒度无 C1')
        dev = float(np.mean([
            np.linalg.norm(p_coords[i] - p_coords[j]) for i, j in block
        ]))
        far_blocks.append(BlockState(
            block_idx=bidx, residues_i=res_i, residues_j=res_j,
            centroid_i=ci, centroid_j=cj, current_deviation=dev,
        ))

    # 块间邻接: 拓扑距离 < 100 的块对 (稀疏边)
    block_edges: List[Tuple[int, int, float]] = []
    for a in range(len(far_blocks)):
        for b in range(a + 1, len(far_blocks)):
            # 块间距离 = 两块质心最近距离 (近似)
            d_ij = np.linalg.norm(far_blocks[a].centroid_i - far_blocks[b].centroid_j)
            d_ji = np.linalg.norm(far_blocks[a].centroid_j - far_blocks[b].centroid_i)
            d = min(d_ij, d_ji)
            if d < 100.0:
                block_edges.append((a, b, float(d)))

    return RLOptimizerState(
        p_coords=p_coords, sequence=sequence,
        far_blocks=far_blocks, far_pairs=far_pairs,
        block_edges=block_edges, coding_mask=coding_mask,
    )


# ---------- Reward ----------
# 正则系数 (防作弊: 拉拢远端配对时不能撞原子/扭曲骨架)
# 实测 λ2=0.1 太强 (单残基平移破坏邻居骨架键, R_distort 暴增压过 R_pair,
# MCTS 不敢动)。降到 0.01 让 R_pair 主导, 正则只在严重扭曲时介入。
LAMBDA_CLASH = 0.05   # 非键 P-P 太近惩罚
LAMBDA_DISTORT = 0.01  # 相邻 P-P 偏离 5.9Å 惩罚 (弱, 不压过 R_pair)
CLASH_THRESH = 3.0    # P-P < 此值算位障 (CG 粒度近似)
BOND_LEN_CG = 5.9     # CG 相邻 P-P 目标距离
BOND_TOL = 1.0        # 相邻 P-P 偏离 5.9±1.0 算扭曲


def compute_reward(
    p_coords: np.ndarray,
    far_pairs: List[Tuple[int, int]],
    *,
    use_regularization: bool = True,
) -> float:
    """远端配对 reward + 正则 (防作弊)。

    R = R_pair - λ1·R_clash - λ2·R_distort

    R_pair    = Σ [exp(-|d_C1'C1' - 10.5|/2) - 0.01·|d-10.5|]  远端配对到位
    R_clash   = Σ max(0, CLASH_THRESH - d_P-P)                  非键 P 位障
    R_distort = Σ max(0, |d_adj_P-P - 5.9| - BOND_TOL)          骨架扭曲

    分布解 + reward 自驱: 防止 RL 靠"硬拉远端配对但崩骨架"作弊。
    use_regularization=False 时退回纯 R_pair (验证阶段用)。
    """
    L = len(p_coords)
    if not far_pairs:
        return 0.0

    # R_pair (主目标)
    r_pair = 0.0
    for (i, j) in far_pairs:
        d = np.linalg.norm(p_coords[i] - p_coords[j])
        dev = abs(d - WC_TARGET_DIST)
        r_pair += np.exp(-dev / 2.0) - 0.01 * dev

    if not use_regularization or L < 3:
        return float(r_pair)

    # R_clash: 非键 P-P 太近 (O(L^2), 长序列要优化, 先正确后优化)
    r_clash = 0.0
    # 只检查远端配对块内 + 相邻区, 避免全 O(L^2)
    far_residues = set()
    for (i, j) in far_pairs:
        far_residues.add(i); far_residues.add(j)
    for r in far_residues:
        # 检查该残基到所有其他 P 的距离 (O(L) per residue)
        dists = np.linalg.norm(p_coords - p_coords[r], axis=1)
        # 排除自身和相邻残基 (相邻本来就近)
        for k in range(L):
            if k == r or abs(k - r) <= 1 or (min(k, r), max(k, r)) in {(min(i, j), max(i, j)) for i, j in far_pairs}:
                continue
            if dists[k] < CLASH_THRESH:
                r_clash += CLASH_THRESH - dists[k]

    # R_distort: 相邻 P-P 偏离 5.9Å (骨架扭曲)
    r_distort = 0.0
    for k in range(L - 1):
        d_adj = np.linalg.norm(p_coords[k + 1] - p_coords[k])
        dev_bond = abs(d_adj - BOND_LEN_CG)
        if dev_bond > BOND_TOL:
            r_distort += dev_bond - BOND_TOL
    # BSJ 闭合
    d_bsj = np.linalg.norm(p_coords[0] - p_coords[-1])
    dev_bsj = abs(d_bsj - BOND_LEN_CG)
    if dev_bsj > BOND_TOL:
        r_distort += dev_bsj - BOND_TOL

    return float(r_pair - LAMBDA_CLASH * r_clash - LAMBDA_DISTORT * r_distort)


# ---------- 策略网络 ----------
class PolicyNetwork:
    """块 GNN 策略网络 (torch, 手写消息传递, 不依赖 torch_geometric)。

    输入: RLOptimizerState
    输出: π_block (softmax over 块), π_dir (6), π_step (3)

    架构: 块节点特征 -> node_enc -> K 层消息传递 (block_edges 邻接) ->
          块嵌入 -> 3 个动作头

    消息传递 (GCN 式): h_i <- ReLU(W·h_i + W·Σ_{j∈N(i)} h_j / |N(i)|)
    边来自 state.block_edges (块间质心距<100 的稀疏邻接)。
    """
    def __init__(self, hidden_dim: int = 128, n_mp_layers: int = 3):
        torch = _get_torch()
        self.hidden_dim = hidden_dim
        self.n_mp_layers = n_mp_layers
        # 节点特征维度: [block_len, centroid_i(3), centroid_j(3), deviation,
        #                mean_pos(3)] = 11
        self.node_feat_dim = 11
        # 节点编码 (特征 -> hidden)
        self.node_enc = torch.nn.Sequential(
            torch.nn.Linear(self.node_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        # 消息传递层 (每层一个 Linear, 残差连接)
        self.mp_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_mp_layers)
        ])
        # 动作头
        self.head_block = torch.nn.Linear(hidden_dim, 1)  # 每块打分, softmax over 块
        self.head_dir = torch.nn.Linear(hidden_dim, N_DIRECTIONS)
        self.head_step = torch.nn.Linear(hidden_dim, N_STEPS)
        # value 头 (PPO GAE 用, 输出标量 V(s))。从块均值嵌入出, 代表整图状态价值。
        self.head_value = torch.nn.Linear(hidden_dim, 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _message_passing(self, h, edge_index, edge_weight):
        """K 层消息传递。h: (n, hidden), edge_index: (2, E) tensor。

        GCN 式归一化聚合, 边权 = 1/(1+d) (块间质心越近影响越大)。
        无邻居的孤立节点只过自身变换 (保留信息)。
        """
        torch = _get_torch()
        n = h.shape[0]
        for layer in self.mp_layers:
            # 聚合邻居 (scatter_add, 边权加权)
            if edge_index is not None and edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                # 每个目标节点收到的加权消息
                agg = torch.zeros_like(h)
                msg = h[src] * edge_weight.unsqueeze(-1)
                agg = agg.index_add(0, dst, msg)
                # 按度归一化 (加 1 防零, 自身算一个邻居)
                deg = torch.zeros(n, dtype=h.dtype, device=h.device)
                deg = deg.index_add(0, dst, edge_weight)
                agg = agg / (deg + 1.0).unsqueeze(-1)
                h_new = torch.relu(layer(h + agg))
            else:
                # 无边: 只过自身变换 (退化为 MLP, 保留旧路径)
                h_new = torch.relu(layer(h))
            h = h_new + h  # 残差
        return h

    @staticmethod
    def _edges_to_tensor(state: RLOptimizerState):
        """把 state.block_edges 转成 (edge_index, edge_weight) tensor。

        block_edges: [(a, b, topo_dist), ...] (无向, 存一份, 消息传递时双向)
        返回 edge_index (2, 2E) 双向, edge_weight (2E,) = 1/(1+d)。
        """
        torch = _get_torch()
        if not state.block_edges:
            return None, None
        src_l, dst_l, w_l = [], [], []
        for a, b, d in state.block_edges:
            w = 1.0 / (1.0 + d)
            src_l.extend([a, b])
            dst_l.extend([b, a])
            w_l.extend([w, w])
        edge_index = torch.tensor([src_l, dst_l], dtype=torch.long)
        edge_weight = torch.tensor(w_l, dtype=torch.float32)
        return edge_index, edge_weight

    def _embed(self, state: RLOptimizerState):
        """共享嵌入: 状态 -> 块嵌入 h (n_blocks, hidden)。forward/value 共用。"""
        torch = _get_torch()
        if not state.far_blocks:
            return None
        feats = []
        for b in state.far_blocks:
            f = np.concatenate([
                [len(b.residues_i)],
                b.centroid_i, b.centroid_j,
                [b.current_deviation],
                (b.centroid_i + b.centroid_j) / 2.0,
            ]).astype(np.float32)
            feats.append(f)
        x = torch.tensor(np.stack(feats), dtype=torch.float32)
        h = self.node_enc(x)
        edge_index, edge_weight = self._edges_to_tensor(state)
        h = self._message_passing(h, edge_index, edge_weight)
        return h

    def forward(self, state: RLOptimizerState, *, return_value: bool = False):
        """返回 (π_block, π_dir, π_step[, V])。

        return_value=False (推理默认): 三元组, MCTS 用。
        return_value=True (训练用): 四元组, 多一个标量 V(s)。
        """
        torch = _get_torch()
        h = self._embed(state)
        if h is None:
            return (None, None, None, None) if return_value else (None, None, None)
        # π_block: 每块打分后 softmax
        block_scores = self.head_block(h).squeeze(-1)  # (n_blocks,)
        pi_block = self.softmax(block_scores)
        # π_dir / π_step: 用平均嵌入 (块选择独立于方向/步长)
        h_mean = h.mean(dim=0, keepdim=True)
        pi_dir = self.softmax(self.head_dir(h_mean)).squeeze(0)
        pi_step = self.softmax(self.head_step(h_mean)).squeeze(0)
        if return_value:
            # V(s): 从均值嵌入出, 代表整图价值
            v = self.head_value(h_mean).squeeze(0).squeeze(-1)  # 标量
            return pi_block, pi_dir, pi_step, v
        return pi_block, pi_dir, pi_step

    def value(self, state: RLOptimizerState):
        """单独算 V(s) (GAE bootstrap 用)。"""
        torch = _get_torch()
        h = self._embed(state)
        if h is None:
            return None
        h_mean = h.mean(dim=0, keepdim=True)
        return self.head_value(h_mean).squeeze(0).squeeze(-1)

    def parameters(self):
        torch = _get_torch()
        params = list(self.node_enc.parameters()) + \
                 list(self.head_block.parameters()) + \
                 list(self.head_dir.parameters()) + \
                 list(self.head_step.parameters()) + \
                 list(self.head_value.parameters())
        for layer in self.mp_layers:
            params += list(layer.parameters())
        return params

    def save(self, path: str):
        torch = _get_torch()
        sd = {
            "node_enc": self.node_enc.state_dict(),
            "mp_layers": self.mp_layers.state_dict(),
            "head_block": self.head_block.state_dict(),
            "head_dir": self.head_dir.state_dict(),
            "head_step": self.head_step.state_dict(),
            "head_value": self.head_value.state_dict(),
            "hidden_dim": self.hidden_dim,
            "n_mp_layers": self.n_mp_layers,
        }
        torch.save(sd, path)

    def load(self, path: str):
        torch = _get_torch()
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.hidden_dim = sd["hidden_dim"]
        self.n_mp_layers = sd["n_mp_layers"]
        # 兼容旧权重 (无 mp_layers): 重建空 ModuleList, 消息传递退化为自身变换
        if "mp_layers" in sd:
            self.mp_layers = torch.nn.ModuleList([
                torch.nn.Linear(self.hidden_dim, self.hidden_dim)
                for _ in range(self.n_mp_layers)
            ])
            self.mp_layers.load_state_dict(sd["mp_layers"])
        self.node_enc.load_state_dict(sd["node_enc"])
        self.head_block.load_state_dict(sd["head_block"])
        self.head_dir.load_state_dict(sd["head_dir"])
        self.head_step.load_state_dict(sd["head_step"])
        # 兼容旧权重 (无 head_value): 随机初始化, 训练前不影响推理
        if "head_value" in sd:
            self.head_value.load_state_dict(sd["head_value"])


# ---------- 动作执行 ----------
def apply_action(
    state: RLOptimizerState,
    block_idx: int,
    dir_idx: int,
    step_idx: int,
) -> np.ndarray:
    """执行动作: 平移指定块的 i 侧残基 (j 侧不动), 改变 i-j 相对距离。

    只动 i 侧: 配对距离 = |P[i] - P[j]|, 移动 i 会改变这个距离。
    旧版 i/j 同向平移, 相对距离不变 (bug, 已修)。
    """
    new_p = state.p_coords.copy()
    b = state.far_blocks[block_idx]
    direction = DIRECTIONS[dir_idx]
    step = STEP_SIZES[step_idx]
    delta = direction * step
    for r in b.residues_i:
        new_p[r] = new_p[r] + delta
    return new_p


# ---------- MCTS ----------
@dataclass
class MCTSNode:
    """MCTS 搜索节点。"""
    p_coords: np.ndarray
    reward: float
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action_taken: Optional[Tuple[int, int, int]] = None


class MCTS:
    """Monte Carlo Tree Search with policy prior.

    策略网络给先验概率, Simulation 阶段可选:
      - use_rollout=False (先验版): 叶节点估值直接用当前 reward (快, 但短视)
      - use_rollout=True  (默认): 叶节点后再走 rollout_depth 步启发式 rollout,
        用终点 reward 估值 (多看几步, 评估更准但慢 rollout_depth 倍)
    rollout 用启发式 (偏差大的块优先, 朝 j 侧方向拉), 不用 policy (policy 是
    待训练对象, 训练前不能用来评估自己, 否则 reward 信号有偏)。
    """
    def __init__(
        self,
        policy: Optional[PolicyNetwork] = None,
        c_puct: float = 1.5,
        n_simulations: int = 50,
        rollout_depth: int = 5,
        use_rollout: bool = True,
    ):
        self.policy = policy
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.rollout_depth = rollout_depth
        self.use_rollout = use_rollout

    def _heuristic_action(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
    ) -> Tuple[int, int, int]:
        """启发式选动作 (rollout 和无策略 fallback 共用)。

        块: 偏差大的块概率高 (softmax over deviation);
        方向: 块 i 侧朝 j 侧的向量量化到 6 方向, 70% 选它 30% 随机;
        步长: 偏差大用大步, 偏小用小步。
        """
        n_blocks = len(state.far_blocks)
        deviations = [abs(b.current_deviation - WC_TARGET_DIST) for b in state.far_blocks]
        probs = np.array(deviations) + 1e-6
        probs = probs / probs.sum()
        bidx = int(np.random.choice(n_blocks, p=probs))
        selected = state.far_blocks[bidx]
        target_dir = selected.centroid_j - selected.centroid_i
        norm = np.linalg.norm(target_dir)
        if norm > 1e-6:
            target_dir = target_dir / norm
            dots = DIRECTIONS @ target_dir
            best_dir = int(np.argmax(dots))
            didx = best_dir if np.random.random() < 0.7 else int(np.random.randint(N_DIRECTIONS))
        else:
            didx = int(np.random.randint(N_DIRECTIONS))
        dev = abs(selected.current_deviation - WC_TARGET_DIST)
        if dev > 15:
            sidx = 2
        elif dev > 5:
            sidx = 1
        else:
            sidx = 0
        return bidx, didx, sidx

    def _rollout(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
    ) -> float:
        """从叶节点启发式走 rollout_depth 步, 返回终点 reward。

        纯 numpy (不建树), 速度快。中间状态用 _rebuild_blocks 更新质心/偏差。
        """
        p = state.p_coords.copy()
        for _ in range(self.rollout_depth):
            tmp = RLOptimizerState(
                p_coords=p, sequence=state.sequence,
                far_blocks=_rebuild_blocks(state, p),
                far_pairs=far_pairs, block_edges=state.block_edges,
            )
            bidx, didx, sidx = self._heuristic_action(tmp, far_pairs)
            p = apply_action(tmp, bidx, didx, sidx)
        return compute_reward(p, far_pairs)


    def search(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
    ) -> np.ndarray:
        """MCTS 搜索, 返回 reward 最高的 P 坐标。

        Selection 用 UCB1 (含 policy prior), Expansion 每次加一个子节点,
        Simulation 用当前 reward 直接评估 (no rollout, 先验版),
        Backprop 沿父链更新 visit/value。
        """
        root_reward = compute_reward(state.p_coords, far_pairs)
        root = MCTSNode(p_coords=state.p_coords, reward=root_reward)
        best = root

        n_blocks = len(state.far_blocks)
        if n_blocks == 0:
            return state.p_coords

        for sim in range(self.n_simulations):
            # --- Selection: 沿树下行, UCB1 选子节点 ---
            node = root
            cur_p = state.p_coords.copy()
            while node.children:
                # UCB1 = value/visits + c_puct * prior * sqrt(ln(parent_visits)/visits)
                best_child = None
                best_ucb = -np.inf
                for c in node.children:
                    if c.visits == 0:
                        ucb = np.inf
                    else:
                        exploit = c.value / c.visits
                        explore = self.c_puct * np.sqrt(
                            np.log(node.visits + 1) / c.visits
                        )
                        ucb = exploit + explore
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = c
                if best_child is None:
                    break
                node = best_child
                cur_p = best_child.p_coords

            # --- Expansion: 从 node 展开一个新子节点 (policy prior 选动作) ---
            tmp_state = RLOptimizerState(
                p_coords=cur_p, sequence=state.sequence,
                far_blocks=_rebuild_blocks(state, cur_p),
                far_pairs=far_pairs, block_edges=state.block_edges,
            )
            pi_block, pi_dir, pi_step = (None, None, None)
            if self.policy is not None:
                pi_block, pi_dir, pi_step = self.policy.forward(tmp_state)

            if pi_block is not None:
                bidx = int(np.random.choice(n_blocks, p=pi_block.detach().numpy()))
                didx = int(np.random.choice(N_DIRECTIONS, p=pi_dir.detach().numpy()))
                sidx = int(np.random.choice(N_STEPS, p=pi_step.detach().numpy()))
            else:
                # 无策略: 启发式选动作 (与 rollout 共用 _heuristic_action)
                bidx, didx, sidx = self._heuristic_action(tmp_state, far_pairs)

            new_p = apply_action(tmp_state, bidx, didx, sidx)
            r_exp = compute_reward(new_p, far_pairs)

            # --- Simulation: 叶节点估值 (可选 rollout 多看几步) ---
            if self.use_rollout:
                # 从新展开的子节点出发跑 rollout, 用终点 reward 估值
                roll_state = RLOptimizerState(
                    p_coords=new_p, sequence=state.sequence,
                    far_blocks=_rebuild_blocks(state, new_p),
                    far_pairs=far_pairs, block_edges=state.block_edges,
                )
                r = self._rollout(roll_state, far_pairs)
            else:
                # 先验版: 直接用当前 reward (短视但快)
                r = r_exp

            child = MCTSNode(p_coords=new_p, reward=r, parent=node,
                             action_taken=(bidx, didx, sidx))
            node.children.append(child)

            # --- Backprop: 沿父链更新 visit/value ---
            cur = child
            while cur is not None:
                cur.visits += 1
                cur.value += r
                cur = cur.parent

            # best 用即时 reward (不卷入 rollout 估值, 避免 rollout 随机性污染最优解)
            if r_exp > best.reward:
                best = MCTSNode(p_coords=new_p, reward=r_exp,
                                parent=None, action_taken=(bidx, didx, sidx))

        return best.p_coords


def _rebuild_blocks(state: RLOptimizerState, new_p: np.ndarray) -> List[BlockState]:
    """用新 P 坐标重建块状态 (更新质心和偏差)。"""
    new_blocks = []
    for b in state.far_blocks:
        ci = new_p[b.residues_i].mean(axis=0)
        cj = new_p[b.residues_j].mean(axis=0)
        dev = float(np.mean([
            np.linalg.norm(new_p[i] - new_p[j]) for i, j in zip(b.residues_i, b.residues_j)
        ]))
        new_blocks.append(BlockState(
            block_idx=b.block_idx, residues_i=b.residues_i, residues_j=b.residues_j,
            centroid_i=ci, centroid_j=cj, current_deviation=dev,
        ))
    return new_blocks


# ---------- 端到端入口 ----------
def optimize_far_pairs(
    p_coords: np.ndarray,
    sequence: str,
    far_pairs: List[Tuple[int, int]],
    stem_blocks: List[List[Tuple[int, int]]],
    *,
    policy_path: Optional[str] = None,
    n_simulations: int = 50,
    coding_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """端到端: CG P 坐标 + 远端配对 -> RL 优化后 P 坐标 + CG 原坐标。

    Args:
        p_coords: (L, 3) CG 求解输出
        sequence: ACGU 字符串
        far_pairs: 远端配对 [(i, j), ...]
        stem_blocks: 茎块 [[(i, j), ...], ...]
        policy_path: 策略网络权重路径 (None 用随机策略, 训练前 baseline)
        n_simulations: MCTS 模拟次数
        coding_mask: 可选 coding 标注 (L,) bool。透传进 state, 并在输出
            里一并返回, 供下游 amber 精修时钉死 coding 区残基。

    Returns:
        (optimized_p, cg_coords, info)
        optimized_p: (L, 3) RL 优化后 P 坐标
        cg_coords: (L, 3) CG 原坐标副本 (给下游 amber 钉死用)
        info: {reward_before, reward_after, improvement, n_blocks,
               n_far_pairs, n_simulations, policy_loaded, coding_mask}
    """
    state = build_rl_state(
        p_coords, sequence, far_pairs, stem_blocks,
        coding_mask=coding_mask,
    )
    reward_before = compute_reward(p_coords, far_pairs)
    # 保存 CG 原坐标副本 (apply_action 会改 p_coords 引用指向的数组, 这里先 copy)
    cg_coords = p_coords.copy()

    policy = None
    if policy_path is not None:
        try:
            policy = PolicyNetwork()
            policy.load(policy_path)
        except Exception as exc:
            print(f"[rl_optimizer] 策略权重加载失败, 用随机策略: {exc!r}")
            policy = None

    mcts = MCTS(policy=policy, n_simulations=n_simulations)
    optimized_p = mcts.search(state, far_pairs)
    reward_after = compute_reward(optimized_p, far_pairs)

    info = {
        "reward_before": float(reward_before),
        "reward_after": float(reward_after),
        "improvement": float(reward_after - reward_before),
        "n_blocks": len(state.far_blocks),
        "n_far_pairs": len(far_pairs),
        "n_simulations": n_simulations,
        "policy_loaded": policy is not None,
        "coding_mask": coding_mask,  # 透传给下游
    }
    return optimized_p, cg_coords, info


if __name__ == "__main__":
    # 自测: 合成远端配对, 验证 RL 能拉拢
    np.random.seed(42)
    L = 100
    # 构造 CG P 坐标 (环形), 远端配对 (10, 60) 故意拉远
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    p = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)
    # 远端配对 (10, 60): 环距 50, 真实 P-P 距离 ~2R*sin(25°) 偏离 10.5
    far_pairs = [(10, 60)]
    # 茎块: (10, 60) 单配对 (凑成 4 连续)
    stem_blocks = [[(10, 60), (11, 59), (12, 58), (13, 57)]]
    # 但 (11,59) 等不在 far_pairs, build_rl_state 会跳过 -- 直接造远端块
    # 简化: 让 far_pairs 包含整块
    far_pairs = [(10, 60), (11, 59), (12, 58), (13, 57)]

    d_before = np.linalg.norm(p[10] - p[60])
    print(f"优化前: pair(10,60) P-P = {d_before:.2f} Å (目标 ~10.5)")

    opt_p, _cg_coords, info = optimize_far_pairs(p, "A" * L, far_pairs, [far_pairs],
                                                   n_simulations=30)
    d_after = np.linalg.norm(opt_p[10] - opt_p[60])
    print(f"优化后: pair(10,60) P-P = {d_after:.2f} Å")
    print(f"info: {info}")
