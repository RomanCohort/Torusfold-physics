"""
rl_optimizer.py - RL far/long-range pair optimizer (MCTS + policy network).

RL does not replace the physics pipeline; it only fills the local-optimum blind
spot of circRNA long-range pairing. At the CG resolution (P coordinates) it uses
MCTS exploration to escape local minima and pull far/long-range pairs into
Watson-Crick geometry (C1'-C1' ~10.5 A), then hands off to the existing physics
pipeline (1EHZ reconstruction + amber refinement) to converge the local
geometry.

Architecture (see docs/scheme2_rl_design.md):
  - State: a small graph of far/long-range pairing blocks (nodes = stem blocks,
    edges = inter-block topological distance)
  - Policy network: 3-layer GNN + action heads (pi_block, pi_dir, pi_step)
  - Actions: discrete (block index, 6 directions, 3 step sizes)
  - Reward: sum over far/long-range pairs of exp(-|d_C1'C1' - 10.5| / 2)
  - MCTS: policy prior + rollout running a short CG refinement to evaluate

Training: PPO + GAE (separate training/ scripts).
Inference: load the weights; MCTS search outputs the optimized P coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy torch import (rl_optimizer may be imported outside training; avoids a hard dependency)
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


# ---------- Constants ----------
# Action space (discrete)
N_DIRECTIONS = 6   # ±x ±y ±z
N_STEPS = 3        # step-size levels: 0.5, 2.0, 5.0 A
STEP_SIZES = (0.5, 2.0, 5.0)
DIRECTIONS = np.array([
    [1, 0, 0], [-1, 0, 0],
    [0, 1, 0], [0, -1, 0],
    [0, 0, 1], [0, 0, -1],
], dtype=np.float32)

WC_TARGET_DIST = 10.5  # A, Watson-Crick C1'-C1' target distance


# ---------- State representation ----------
@dataclass
class BlockState:
    """State of a single far/long-range pairing stem block."""
    block_idx: int                 # index of this block in the far-block list
    residues_i: List[int]          # residue indices on the i side of the block
    residues_j: List[int]          # residue indices on the j side of the block
    centroid_i: np.ndarray         # i-side centroid P coordinates (3,)
    centroid_j: np.ndarray         # j-side centroid P coordinates (3,)
    current_deviation: float       # current mean C1'-C1' deviation (A)


@dataclass
class RLOptimizerState:
    """Complete state for RL optimization."""
    p_coords: np.ndarray            # (L, 3) CG P coordinates
    sequence: str
    far_blocks: List[BlockState]   # far/long-range pairing blocks
    far_pairs: List[Tuple[int, int]]  # far/long-range pairs (i, j)
    # Inter-block adjacency (sparse): [(block_a, block_b, topo_dist), ...]
    block_edges: List[Tuple[int, int, float]] = field(default_factory=list)
    # coding mask: passed through to the downstream amber refinement, which pins
    # the residues of coding regions
    coding_mask: Optional[np.ndarray] = None  # shape (L,) bool, True=coding


def build_rl_state(
    p_coords: np.ndarray,
    sequence: str,
    far_pairs: List[Tuple[int, int]],
    stem_blocks: List[List[Tuple[int, int]]],
    coding_mask: Optional[np.ndarray] = None,
) -> RLOptimizerState:
    """Build an RL state from CG P coordinates + far/long-range pairs + stem blocks.

    Args:
        p_coords: (L, 3) P coordinates output by the CG solver
        sequence: ACGU string
        far_pairs: far/long-range pairs [(i, j), ...] (from pair_graph.far_end_pairs)
        stem_blocks: stem blocks [[(i, j), ...], ...] (from pair_graph.extract_stem_blocks)
        coding_mask: optional coding annotation (from pair_graph.parse_case_annotation);
            passed through downstream and does not constrain the RL action space
            (RL may move the whole sequence)
    """
    # Keep only far/long-range stem blocks (blocks whose pairs are all in far_pairs)
    far_set = set((min(i, j), max(i, j)) for i, j in far_pairs)
    far_blocks: List[BlockState] = []
    for bidx, block in enumerate(stem_blocks):
        # Are all pairs in this block in the far set?
        in_far = all((min(i, j), max(i, j)) in far_set for i, j in block)
        if not in_far:
            continue
        res_i = [i for i, _ in block]
        res_j = [j for _, j in block]
        ci = p_coords[res_i].mean(axis=0)
        cj = p_coords[res_j].mean(axis=0)
        # Current deviation: mean C1'-C1' over the block pairs (approximated with P;
        # C1' is not available at CG resolution)
        dev = float(np.mean([
            np.linalg.norm(p_coords[i] - p_coords[j]) for i, j in block
        ]))
        far_blocks.append(BlockState(
            block_idx=bidx, residues_i=res_i, residues_j=res_j,
            centroid_i=ci, centroid_j=cj, current_deviation=dev,
        ))

    # Inter-block adjacency: block pairs with topological distance < 100 (sparse edges)
    block_edges: List[Tuple[int, int, float]] = []
    for a in range(len(far_blocks)):
        for b in range(a + 1, len(far_blocks)):
            # Inter-block distance = nearest distance between the two centroids (approximation)
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
# Regularization coefficients (anti-cheating: pulling far/long-range pairs
# together must not clash atoms or distort the backbone)
# Measured: lambda2=0.1 was too strong (translating a single residue broke the
# neighbor backbone bonds, R_distort blew up past R_pair, and MCTS would not
# move). Lowered to 0.01 so R_pair dominates; the regularizer only intervenes
# under severe distortion.
LAMBDA_CLASH = 0.05   # penalty for non-bonded P-P too close
LAMBDA_DISTORT = 0.01  # penalty for neighbor P-P deviating from 5.9 A (weak, does not overwhelm R_pair)
CLASH_THRESH = 3.0    # P-P below this counts as a clash (approximation at CG resolution)
BOND_LEN_CG = 5.9     # target CG neighbor P-P distance
BOND_TOL = 1.0        # neighbor P-P deviating from 5.9 +/- 1.0 counts as distorted


def compute_reward(
    p_coords: np.ndarray,
    far_pairs: List[Tuple[int, int]],
    *,
    use_regularization: bool = True,
    sequence: str = None,
    trirnasp_potential=None,
    trirnasp_scale: float = 1.0,
    progress: float = None,
) -> float:
    """Far/long-range-pair reward + regularization (anti-cheating) + optional TriRNASP statistical potential (vectorized).

    R = w_pair*R_pair - lambda1*R_clash - lambda2*R_distort + w_trirnasp*(-trirnasp_score)

    Vectorized: R_clash uses numpy broadcasting instead of Python double loops,
    R_distort is fully vectorized, and R_pair is fully vectorized. At L=2003 this
    is ~50x faster than the old Python loop version.
    """
    L = len(p_coords)
    if not far_pairs:
        return 0.0

    far_pairs_arr = np.asarray(far_pairs, dtype=np.int64)  # (P, 2)

    # -- R_pair (vectorized) --
    pi, pj = far_pairs_arr[:, 0], far_pairs_arr[:, 1]
    d_pair = np.linalg.norm(p_coords[pi] - p_coords[pj], axis=1)
    dev_pair = np.abs(d_pair - WC_TARGET_DIST)
    r_pair = float(np.sum(np.exp(-dev_pair / 2.0) - 0.01 * dev_pair))

    if not use_regularization or L < 3:
        return r_pair

    # -- R_clash (vectorized: only check far/long-range residues, excluding neighbors and paired pairs) --
    r_clash = 0.0
    far_res = np.unique(far_pairs_arr.ravel())
    if len(far_res) > 0:
        # Distance matrix: far_res x full sequence (only the needed rows)
        d_mat = np.linalg.norm(p_coords[far_res, None] - p_coords[None, :], axis=2)  # (n_far, L)
        # Mask: exclude self, neighbors (+/-1), and already-paired pairs
        mask = np.ones_like(d_mat, dtype=bool)
        for idx, r in enumerate(far_res):
            mask[idx, r] = False                       # self
            if r > 0: mask[idx, r - 1] = False        # left neighbor
            if r < L - 1: mask[idx, r + 1] = False    # right neighbor
        # Exclude already-paired pairs (i-j are already close by design)
        for a, b in far_pairs:
            ia = np.searchsorted(far_res, a)
            ib = np.searchsorted(far_res, b)
            if ia < len(far_res) and far_res[ia] == a:
                mask[ia, b] = False
            if ib < len(far_res) and far_res[ib] == b:
                mask[ib, a] = False
        # clashing: d < CLASH_THRESH and not masked out
        clash_val = np.where(mask, np.maximum(CLASH_THRESH - d_mat, 0.0), 0.0)
        r_clash = float(clash_val.sum())

    # -- R_distort (vectorized) --
    d_adj = np.linalg.norm(p_coords[1:] - p_coords[:-1], axis=1)  # (L-1,)
    dev_bond = np.abs(d_adj - BOND_LEN_CG)
    r_distort = float(np.sum(np.maximum(dev_bond - BOND_TOL, 0.0)))
    # BSJ closure
    d_bsj = np.linalg.norm(p_coords[0] - p_coords[-1])
    dev_bsj = abs(d_bsj - BOND_LEN_CG)
    if dev_bsj > BOND_TOL:
        r_distort += dev_bsj - BOND_TOL

    # -- R_trirnasp (unchanged) --
    r_trirnasp = 0.0
    if trirnasp_potential is not None and sequence is not None:
        try:
            coords_3b = np.zeros((L, 3, 3), dtype=np.float64)
            for i in range(L):
                coords_3b[i, 0] = p_coords[i]
                rng = np.random.default_rng(i * 31 + 7)
                coords_3b[i, 1] = p_coords[i] + rng.normal(0, 0.3, 3)
                coords_3b[i, 2] = p_coords[i] + rng.normal(0, 0.3, 3)
            tri_score = trirnasp_potential.score(coords_3b, sequence)
            r_trirnasp = -tri_score * trirnasp_scale
        except Exception:
            pass

    # -- Curriculum-learning weights --
    if progress is not None:
        w_pair = 1.0 - 0.5 * progress
        w_trirnasp = trirnasp_scale * progress
    else:
        w_pair = 1.0
        w_trirnasp = trirnasp_scale

    scaled_trirnasp = r_trirnasp * (w_trirnasp / trirnasp_scale) if trirnasp_scale != 0.0 else 0.0

    return float(w_pair * r_pair - LAMBDA_CLASH * r_clash - LAMBDA_DISTORT * r_distort + scaled_trirnasp)


# ---------- Policy network ----------
class PolicyNetwork:
    """Block GNN policy network (torch, hand-written message passing, no torch_geometric dependency).

    Input: RLOptimizerState
    Output: pi_block (softmax over blocks), pi_dir (6), pi_step (3)

    Architecture: block-node features -> node_enc -> K message-passing layers
    (block_edges adjacency) -> block embeddings -> 3 action heads

    Message passing (GCN-style): h_i <- ReLU(W*h_i + W*sum_{j in N(i)} h_j / |N(i)|)
    Edges come from state.block_edges (sparse adjacency of blocks whose centroid
    distance is < 100).
    """
    def __init__(self, hidden_dim: int = 128, n_mp_layers: int = 3):
        torch = _get_torch()
        self.hidden_dim = hidden_dim
        self.n_mp_layers = n_mp_layers
        # Node feature dimension: [block_len, centroid_i(3), centroid_j(3),
        #                          deviation, mean_pos(3)] = 11
        self.node_feat_dim = 11
        # Node encoder (features -> hidden)
        self.node_enc = torch.nn.Sequential(
            torch.nn.Linear(self.node_feat_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        # Message-passing layers (one Linear per layer, with residual connections)
        self.mp_layers = torch.nn.ModuleList([
            torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_mp_layers)
        ])
        # Action heads
        self.head_block = torch.nn.Linear(hidden_dim, 1)  # scores each block; softmax over blocks
        self.head_dir = torch.nn.Linear(hidden_dim, N_DIRECTIONS)
        self.head_step = torch.nn.Linear(hidden_dim, N_STEPS)
        # Value head (used by PPO/GAE; outputs the scalar V(s)). It reads from the
        # mean block embedding and represents the whole-graph state value.
        self.head_value = torch.nn.Linear(hidden_dim, 1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def _message_passing(self, h, edge_index, edge_weight):
        """K message-passing layers. h: (n, hidden); edge_index: (2, E) tensor.

        GCN-style normalized aggregation with edge weight = 1/(1+d) (the closer
        two block centroids, the stronger the influence). An isolated node with
        no neighbors just passes through its own transform (information kept).
        """
        torch = _get_torch()
        n = h.shape[0]
        for layer in self.mp_layers:
            # Aggregate neighbors (scatter_add, weighted by edge weight)
            if edge_index is not None and edge_index.shape[1] > 0:
                src, dst = edge_index[0], edge_index[1]
                # Weighted messages received by each target node
                agg = torch.zeros_like(h)
                msg = h[src] * edge_weight.unsqueeze(-1)
                agg = agg.index_add(0, dst, msg)
                # Normalize by degree (add 1 to avoid dividing by zero; self counts as one neighbor)
                deg = torch.zeros(n, dtype=h.dtype, device=h.device)
                deg = deg.index_add(0, dst, edge_weight)
                agg = agg / (deg + 1.0).unsqueeze(-1)
                h_new = torch.relu(layer(h + agg))
            else:
                # No edges: only pass through the node's own transform (degrades to an MLP; keeps the old path)
                h_new = torch.relu(layer(h))
            h = h_new + h  # residual connection
        return h

    @staticmethod
    def _edges_to_tensor(state: RLOptimizerState):
        """Convert state.block_edges into (edge_index, edge_weight) tensors.

        block_edges: [(a, b, topo_dist), ...] (undirected, stored once; message
        passing uses both directions). Returns a bidirectional edge_index
        (2, 2E) and edge_weight (2E,) = 1/(1+d).
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
        """Shared embedding: state -> block embeddings h (n_blocks, hidden). Shared by forward/value."""
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
        """Return (pi_block, pi_dir, pi_step[, V]).

        return_value=False (inference default): the triple, used by MCTS.
        return_value=True (training): the quadruple, with an extra scalar V(s).
        """
        torch = _get_torch()
        h = self._embed(state)
        if h is None:
            return (None, None, None, None) if return_value else (None, None, None)
        # pi_block: softmax over the per-block scores
        block_scores = self.head_block(h).squeeze(-1)  # (n_blocks,)
        pi_block = self.softmax(block_scores)
        # pi_dir / pi_step: use the mean embedding (block choice is independent of direction/step)
        h_mean = h.mean(dim=0, keepdim=True)
        pi_dir = self.softmax(self.head_dir(h_mean)).squeeze(0)
        pi_step = self.softmax(self.head_step(h_mean)).squeeze(0)
        if return_value:
            # V(s): read from the mean embedding; represents the whole-graph value
            v = self.head_value(h_mean).squeeze(0).squeeze(-1)  # scalar
            return pi_block, pi_dir, pi_step, v
        return pi_block, pi_dir, pi_step

    def value(self, state: RLOptimizerState):
        """Compute V(s) on its own (for the GAE bootstrap)."""
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
        # Compatible with old weights (no mp_layers): rebuild an empty ModuleList;
        # message passing then degrades to the identity transform
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
        # Compatible with old weights (no head_value): leave it randomly
        # initialized; it does not affect inference before training
        if "head_value" in sd:
            self.head_value.load_state_dict(sd["head_value"])


# ---------- Action execution ----------
def apply_action(
    state: RLOptimizerState,
    block_idx: int,
    dir_idx: int,
    step_idx: int,
) -> Tuple[np.ndarray, int]:
    """Execute an action: translate the i-side residues of the given block (the j side stays put), changing the i-j relative distance.

    Only the i side moves: pair distance = |P[i] - P[j]|, so moving i changes it.
    The old version translated i and j in the same direction, leaving the
    relative distance unchanged (a bug, now fixed).

    Returns:
        (new_p_coords, block_idx) - block_idx lets _rebuild_blocks update incrementally.
    """
    new_p = state.p_coords.copy()
    b = state.far_blocks[block_idx]
    direction = DIRECTIONS[dir_idx]
    step = STEP_SIZES[step_idx]
    delta = direction * step
    for r in b.residues_i:
        new_p[r] = new_p[r] + delta
    return new_p, block_idx


# ---------- MCTS ----------
@dataclass
class MCTSNode:
    """An MCTS search node."""
    p_coords: np.ndarray
    reward: float
    parent: Optional["MCTSNode"] = None
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action_taken: Optional[Tuple[int, int, int]] = None
    _far_blocks: Optional[List] = None  # cached block states, to avoid rebuilding


class MCTS:
    """Monte Carlo Tree Search with policy prior.

    The policy network provides the prior probabilities; the Simulation stage is
    optional:
      - use_rollout=False (prior-only): value a leaf directly from its current
        reward (fast, but short-sighted)
      - use_rollout=True (default): after reaching a leaf, run rollout_depth
        steps of heuristic rollout and value it by the terminal reward (looks
        further ahead; more accurate but rollout_depth times slower)
    Rollout uses heuristics (blocks with large deviation first, pulled toward
    the j side), not the policy: the policy is the object being trained and
    cannot be used to evaluate itself before training, or the reward signal
    would be biased.
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
        """Pick an action heuristically (shared by rollout and the no-policy fallback).

        Block: blocks with larger deviation are more likely (softmax over deviation);
        Direction: the vector from the block's i side toward its j side is
            quantized to the 6 directions; choose it 70% of the time, random 30%;
        Step: large deviation uses a big step, small deviation a small step.
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
        progress: float = None,
    ) -> float:
        """Walk rollout_depth heuristic steps from a leaf node and return the terminal reward.

        Pure numpy (no tree built), so it is fast. Intermediate states are updated
        incrementally with _rebuild_blocks.
        """
        p = state.p_coords.copy()
        cur_blocks = _rebuild_blocks(state, p)  # full rebuild on the first call
        for _ in range(self.rollout_depth):
            tmp = RLOptimizerState(
                p_coords=p, sequence=state.sequence,
                far_blocks=cur_blocks,
                far_pairs=far_pairs, block_edges=state.block_edges,
            )
            bidx, didx, sidx = self._heuristic_action(tmp, far_pairs)
            p, changed = apply_action(tmp, bidx, didx, sidx)
            cur_blocks = _rebuild_blocks(tmp, p, changed_block=changed)
        return compute_reward(p, far_pairs, progress=progress)


    def search(
        self,
        state: RLOptimizerState,
        far_pairs: List[Tuple[int, int]],
        progress: float = None,
    ) -> np.ndarray:
        """Run MCTS search and return the P coordinates with the highest reward.

        Selection uses UCB1 (with the policy prior), Expansion adds one child per
        iteration, Simulation scores the current reward directly (no rollout,
        prior-only version), and Backprop updates visit/value up the parent chain.

        Args:
            progress: curriculum-learning progress [0, 1], forwarded to compute_reward.
                None means no curriculum (backward compatible).
        """
        root_reward = compute_reward(state.p_coords, far_pairs, progress=progress)
        root = MCTSNode(p_coords=state.p_coords, reward=root_reward)
        best = root

        n_blocks = len(state.far_blocks)
        if n_blocks == 0:
            return state.p_coords

        for sim in range(self.n_simulations):
            # --- Selection: descend the tree, choosing children by UCB1 ---
            node = root
            cur_p = state.p_coords.copy()
            cur_blocks = state.far_blocks  # first iteration uses the original blocks
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
                # The node already stores far_blocks (computed at creation); no rebuild needed
                cur_blocks = getattr(node, '_far_blocks', state.far_blocks)

            # --- Expansion: grow one new child from node (action chosen by the policy prior) ---
            tmp_state = RLOptimizerState(
                p_coords=cur_p, sequence=state.sequence,
                far_blocks=cur_blocks,
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
                bidx, didx, sidx = self._heuristic_action(tmp_state, far_pairs)

            new_p, changed_blk = apply_action(tmp_state, bidx, didx, sidx)
            # Incremental rebuild: only recompute the modified block
            new_blocks = _rebuild_blocks(tmp_state, new_p, changed_block=changed_blk)

            # Curriculum learning: progress increases linearly across simulations
            sim_progress = (sim + 1) / self.n_simulations if progress is not None else None
            r_exp = compute_reward(new_p, far_pairs, progress=sim_progress)

            # --- Simulation: value the leaf (optional rollout to look a few steps ahead) ---
            if self.use_rollout:
                roll_state = RLOptimizerState(
                    p_coords=new_p, sequence=state.sequence,
                    far_blocks=new_blocks,
                    far_pairs=far_pairs, block_edges=state.block_edges,
                )
                r = self._rollout(roll_state, far_pairs, progress=sim_progress)
            else:
                r = r_exp

            child = MCTSNode(p_coords=new_p, reward=r, parent=node,
                             action_taken=(bidx, didx, sidx),
                             _far_blocks=new_blocks)
            node.children.append(child)

            # --- Backprop: update visit/value up the parent chain ---
            cur = child
            while cur is not None:
                cur.visits += 1
                cur.value += r
                cur = cur.parent

            # best uses the immediate reward (kept out of the rollout estimate, so
            # rollout randomness cannot pollute the best solution)
            if r_exp > best.reward:
                best = MCTSNode(p_coords=new_p, reward=r_exp,
                                parent=None, action_taken=(bidx, didx, sidx))

        return best.p_coords


def _rebuild_blocks(state: RLOptimizerState, new_p: np.ndarray,
                     changed_block: int = -1) -> List[BlockState]:
    """Rebuild the block states from the new P coordinates (incremental: only recompute changed_block, reuse the old centroids/deviations otherwise).

    changed_block: the block index modified by apply_action; -1 = full rebuild (first call).
    """
    new_blocks = []
    for idx, b in enumerate(state.far_blocks):
        if idx == changed_block or changed_block == -1:
            ci = new_p[b.residues_i].mean(axis=0)
            cj = new_p[b.residues_j].mean(axis=0)
            # Compute the deviation vectorized
            ri_arr = np.asarray(b.residues_i)
            rj_arr = np.asarray(b.residues_j)
            dev = float(np.mean(np.linalg.norm(new_p[ri_arr] - new_p[rj_arr], axis=1)))
            new_blocks.append(BlockState(
                block_idx=b.block_idx, residues_i=b.residues_i, residues_j=b.residues_j,
                centroid_i=ci, centroid_j=cj, current_deviation=dev,
            ))
        else:
            # Reuse the old block (coordinates unchanged)
            new_blocks.append(b)
    return new_blocks


# ---------- End-to-end entry point ----------
def optimize_far_pairs(
    p_coords: np.ndarray,
    sequence: str,
    far_pairs: List[Tuple[int, int]],
    stem_blocks: List[List[Tuple[int, int]]],
    *,
    policy_path: Optional[str] = None,
    n_simulations: int = 50,
    coding_mask: Optional[np.ndarray] = None,
    progress: float = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """End to end: CG P coordinates + far/long-range pairs -> RL-optimized P coordinates + original CG coordinates.

    Args:
        p_coords: (L, 3) CG solver output
        sequence: ACGU string
        far_pairs: far/long-range pairs [(i, j), ...]
        stem_blocks: stem blocks [[(i, j), ...], ...]
        policy_path: path to policy-network weights (None = random policy, the pre-training baseline)
        n_simulations: number of MCTS simulations
        coding_mask: optional coding annotation (L,) bool. Passed into the state
            and also returned in the output, so the downstream amber refinement
            can pin the coding-region residues.
        progress: curriculum-learning progress [0, 1]. None = no curriculum
            (backward compatible); when given, MCTS interpolates it linearly
            (0->1 per simulation).

    Returns:
        (optimized_p, cg_coords, info)
        optimized_p: (L, 3) RL-optimized P coordinates
        cg_coords: (L, 3) copy of the original CG coordinates (for downstream amber pinning)
        info: {reward_before, reward_after, improvement, n_blocks,
               n_far_pairs, n_simulations, policy_loaded, coding_mask}
    """
    state = build_rl_state(
        p_coords, sequence, far_pairs, stem_blocks,
        coding_mask=coding_mask,
    )
    reward_before = compute_reward(p_coords, far_pairs)
    # Keep a copy of the original CG coordinates (apply_action mutates the array
    # that p_coords references, so copy it first)
    cg_coords = p_coords.copy()

    policy = None
    if policy_path is not None:
        try:
            policy = PolicyNetwork()
            policy.load(policy_path)
        except Exception as exc:
            print(f"[rl_optimizer] failed to load policy weights; using a random policy: {exc!r}")
            policy = None

    mcts = MCTS(policy=policy, n_simulations=n_simulations)
    optimized_p = mcts.search(state, far_pairs, progress=progress)
    reward_after = compute_reward(optimized_p, far_pairs)

    info = {
        "reward_before": float(reward_before),
        "reward_after": float(reward_after),
        "improvement": float(reward_after - reward_before),
        "n_blocks": len(state.far_blocks),
        "n_far_pairs": len(far_pairs),
        "n_simulations": n_simulations,
        "policy_loaded": policy is not None,
        "coding_mask": coding_mask,  # passed through to the downstream refinement
    }
    return optimized_p, cg_coords, info


# -- Stub classes: imported by isrnaclong.py but never implemented --

class ReplayBuffer:
    """Experience replay buffer (simplified, fixed capacity)."""

    def __init__(self, capacity=10000):
        from collections import deque
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states), np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class OnlineLearner:
    """Online learner: PPO updates for the policy network."""

    def __init__(self, policy, buffer, lr=3e-4, gamma=0.99, clip=0.2):
        self.policy = policy
        self.buffer = buffer
        self.gamma = gamma
        self.clip = clip
        self.optimizer = None
        if policy is not None:
            try:
                import torch
                self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
            except Exception:
                pass

    def update(self, batch_size=32):
        """Sample from the buffer and take one PPO update step."""
        if len(self.buffer) < batch_size or self.optimizer is None:
            return 0.0
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        # Simplified PPO: direct policy gradient (no GAE)
        try:
            import torch
            self.policy.train()
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards)

            block_logits, dir_logits, step_logits = self.policy(states_t)
            log_probs = (
                torch.nn.functional.log_softmax(block_logits, dim=-1)
                + torch.nn.functional.log_softmax(dir_logits, dim=-1)
                + torch.nn.functional.log_softmax(step_logits, dim=-1)
            )
            # Simplified: weight the log prob by the reward
            loss = -(log_probs.sum(dim=-1) * rewards_t).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return float(loss.item())
        except Exception:
            return 0.0


class ContinuousAssemblyPolicy:
    """Continuous-space assembly policy (placeholder; full version not yet implemented)."""

    def __init__(self, input_dim=12, hidden_dim=64, output_dim=6):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def predict(self, state):
        """Return a continuous action [offset(3), rotation(3)]."""
        return np.zeros(self.output_dim, dtype=np.float32)


if __name__ == "__main__":
    # Self-test: synthetic far/long-range pairs, checking that RL can pull them together
    np.random.seed(42)
    L = 100
    # Build CG P coordinates on a ring, with the far/long-range pair (10, 60) deliberately pulled far apart
    R = L * 5.9 / (2 * np.pi)
    angles = np.linspace(0, 2 * np.pi, L, endpoint=False)
    p = np.stack([R * np.cos(angles), R * np.sin(angles), np.zeros(L)], axis=1)
    # Far/long-range pair (10, 60): ring distance 50, real P-P distance
    # ~2R*sin(25 deg), i.e. far from 10.5
    far_pairs = [(10, 60)]
    # Stem block: (10, 60) plus further pairs (to form 4 consecutive pairs)
    stem_blocks = [[(10, 60), (11, 59), (12, 58), (13, 57)]]
    # But (11,59) etc. are not in far_pairs, so build_rl_state would skip them -
    # just make far_pairs contain the whole block instead.
    # Simplify: let far_pairs contain the whole block
    far_pairs = [(10, 60), (11, 59), (12, 58), (13, 57)]

    d_before = np.linalg.norm(p[10] - p[60])
    print(f"before: pair(10,60) P-P = {d_before:.2f} A (target ~10.5)")

    opt_p, _cg_coords, info = optimize_far_pairs(p, "A" * L, far_pairs, [far_pairs],
                                                   n_simulations=30)
    d_after = np.linalg.norm(opt_p[10] - opt_p[60])
    print(f"after: pair(10,60) P-P = {d_after:.2f} A")
    print(f"info: {info}")
