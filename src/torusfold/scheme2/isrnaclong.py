"""
isrnaclong.py — isRNAcircLong main pipeline

Long circRNA 3D structure prediction:
  Level 0: ViennaRNA coarse screen
  Level 1: segmented Vfold3D/RhoFold+ + Kabsch assembly
  Level 2: RL-guided isRNAcirc close + iterative relaxation
  Level 3: RL-MCTS topology search
  Level 4: REST2 refinement
  Level 5: all-atom + Amber

Level 2 details:
  - Round 1: isRNAcirc Type=1 close_ends + MD (closes the BSJ)
  - Later rounds: the RL agent guides pair_weights + MD parameters (replaces heuristics)
  - RL state: pairing distance + energy + clash + convergence indicators
  - RL action: pair_weights (N_far_pairs,) + md_nstep (scalar)
  - RL reward: energy_delta + pair_rate_delta - clash_penalty

Reference: isRNAcircLong_design.md
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


# ── WSL preheat + PyRosetta long-lived server ─────────────────────────
_PYROSETTA_SOCK = "/tmp/torusfold_pyrosetta.sock"
_PYROSETTA_SERVER_PROC = None  # reference to the server started by this process


def _preheat_wsl():
    """Preheat the WSL instance asynchronously to avoid subprocess cold-start latency."""
    try:
        subprocess.Popen(
            ["wsl", "echo", "ok"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass  # failed preheat does not affect the rest


def _win_to_wsl(p) -> str:
    """Convert a Windows path to a WSL path (/mnt/c/...)."""
    s = str(p).replace("\\", "/")
    if len(s) >= 2 and s[1] == ":":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


def _pyrosetta_server_running() -> bool:
    """Check whether the PyRosetta server is already running."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect(_PYROSETTA_SOCK)
        s.close()
        return True
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        return False


def _pyrosetta_start_server(verbose: bool = True) -> bool:
    """Start the long-lived PyRosetta server (WSL background) and wait for the ready signal."""
    global _PYROSETTA_SERVER_PROC
    if _pyrosetta_server_running():
        if verbose:
            print("    [PyR-server] already running")
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

    # wait for the socket to be ready (up to 30s)
    if verbose:
        print("    [PyR-server] starting (waiting for PyRosetta init)...")
    for _ in range(300):
        time.sleep(0.1)
        if _pyrosetta_server_running():
            if verbose:
                print("    [PyR-server] ready [ok]")
            return True
        if _PYROSETTA_SERVER_PROC.poll() is not None:
            if verbose:
                print("    [PyR-server] process exited; falling back to subprocess mode")
            return False

    if verbose:
        print("    [PyR-server] startup timed out; falling back to subprocess mode")
    return False


def _pyrosetta_socket_refine(
    pdb_path: str, output_path: str, max_iter: int = 200, verbose: bool = True,
) -> Optional[Tuple[str, float]]:
    """Refine via the long-lived PyRosetta server over a Unix socket.

    Returns: (output_path, energy), or None when the service is unavailable.
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
        # length prefix (4-byte little-endian)
        s.sendall(len(req).to_bytes(4, "little") + req)

        # read the response
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
                print(f"    [PyR-server] error: {resp.get('error', '?')}")
            return None
    finally:
        s.close()


def _save_checkpoint(ckpt_path: Path, data: dict):
    """Save a checkpoint (JSON + numpy arrays as .npy).

    Atomic write: write a temp file first, then rename, so a crash cannot leave a
    half-written JSON. numpy arrays also go through a temp file and are renamed only
    after saving, so the JSON never references a missing .npy.
    """
    # store numpy arrays separately as .npy files (atomic write)
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
    os.replace(tmp_path, str(ckpt_path))  # atomic replace


def _load_checkpoint(ckpt_path: Path) -> dict:
    """Load a checkpoint (fault-tolerant: missing/corrupt .npy files are skipped without affecting the rest)."""
    if not ckpt_path.exists():
        return {}
    try:
        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [checkpoint] JSON load failed: {e}")
        return {}
    arrays = data.pop("_npy_refs", {})
    for k, npy_path in arrays.items():
        try:
            data[k] = np.load(npy_path)
        except Exception as e:
            print(f"  [checkpoint] .npy load failed: {k} ({npy_path}): {e}")
    return data


def _cleanup_checkpoints(output_path: Path, keep_final: bool = True):
    """Clean up intermediate checkpoint files, keeping only the final results.

    Deletes:
      - _checkpoint.json (checkpoint metadata)
      - _checkpoint_*.npy (numpy arrays)
      - _tmp_* (temporary files)

    Keeps:
      - *.pdb (all PDB outputs)
      - *.json (result files, not checkpoints)
      - _final_*.npz (final data)
    """
    removed = 0
    for f in output_path.glob("_checkpoint*"):
        f.unlink(missing_ok=True)
        removed += 1


def _delete_checkpoint_from_level(output_path: Path, from_level: float = 2.0):
    """Delete checkpoint fields for the given level onward, rolling back to the state before that level.

    Example: _delete_checkpoint_from_level(path, 2.0) removes the Level 2/2.5/2.6/3/... fields,
    the checkpoint rolls back to Level 1.5, and the next resume starts over from Level 2.

    Args:
        output_path: output directory
        from_level: the level from which to start deleting (inclusive)
    """
    ckpt_path = output_path / "_checkpoint.json"
    if not ckpt_path.exists():
        print(f"  [checkpoint] no checkpoint file found, skipping deletion")
        return

    ckpt = _load_checkpoint(ckpt_path)
    current_level = float(ckpt.get("level", -1))

    # field list for Level 2+ (collected from the _save_ckpt calls)
    _LEVEL2_PLUS_FIELDS = {
        "remd_round", "best_coords", "best_energy", "pair_rate",
        "cross_segment_ok_rate", "energy", "clash_count",
        # fields for Level 2.5/2.6/3/3.5/4/5/5.5
        "p5_refined", "pyrosetta_out", "l26_ok",
        "coords_rl", "rl_info",
        "coords_metad", "meta_e",
        "coords_rest2", "rest2_e",
        "level5_amber", "amber_e",
        "ppr_repaired", "ppr_rate",
    }

    # check whether Level 2+ fields are still present (even if the level value was rolled back)
    has_level2_fields = bool(set(ckpt.keys()) & _LEVEL2_PLUS_FIELDS)
    if current_level < from_level and not has_level2_fields:
        print(f"  [checkpoint] current level={current_level}, no Level {from_level}+ fields, nothing to delete")
        return

    removed_fields = []
    removed_npy = 0
    for field in _LEVEL2_PLUS_FIELDS:
        if field in ckpt:
            del ckpt[field]
            removed_fields.append(field)
            # delete the associated .npy files
            npy_path = output_path / f"ckpt_{field}.npy"
            if npy_path.exists():
                npy_path.unlink()
                removed_npy += 1

    # roll the level back to before from_level
    _rollback_levels = {2.0: 1.5, 2.5: 2.0, 2.6: 2.5, 3.0: 2.6,
                        3.5: 3.0, 4.0: 3.5, 5.0: 4.0, 5.5: 5.0}
    new_level = _rollback_levels.get(from_level, from_level - 0.5)
    ckpt["level"] = new_level

    # save the modified checkpoint
    _save_checkpoint(ckpt_path, ckpt)
    print(f"  [checkpoint] deleted Level {from_level}+ fields: {removed_fields}")
    print(f"  [checkpoint] deleted {removed_npy} .npy files")
    print(f"  [checkpoint] level rolled back to {new_level}, resuming after Level {new_level} next time")


@dataclass
class RelaxationMetrics:
    """Joint metrics for monitoring iterative relaxation (4 indicators)."""
    cross_segment_ok: float = 0.0    # fraction of cross-segment pairs < 15Å
    clash_count: int = 0             # number of clashes with P-P distance < 3Å
    rmsd_change: float = 0.0         # RMSD change relative to the previous round
    pair_rate: float = 0.0           # overall base-pairing rate
    energy_delta: float = 0.0        # energy change

    @property
    def is_converged(self) -> bool:
        """Multi-metric convergence criterion."""
        return (self.cross_segment_ok > 0.8
                and self.clash_count == 0
                and abs(self.rmsd_change) < 0.5)


class RelaxationRL:
    """RL agent that uses MCTS + a GNN PolicyNetwork to guide each isRNAcirc relaxation round.

    Supports online learning: trajectories are collected during decide(), and PPO updates
    are triggered periodically. Reuses PolicyNetwork + MCTS + ReplayBuffer + OnlineLearner
    from rl_optimizer.py.
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

        # load PolicyNetwork + MCTS
        from torusfold.scheme2.rl_optimizer import PolicyNetwork, MCTS, build_rl_state, compute_reward
        self._build_rl_state = build_rl_state
        self._compute_reward = compute_reward

        self.policy = None
        if policy_path and os.path.exists(policy_path):
            self.policy = PolicyNetwork()
            self.policy.load(policy_path)
        else:
            # no pretrained weights: create a random policy; MCTS uses heuristic rollouts
            self.policy = PolicyNetwork()

        self.mcts = MCTS(
            policy=None,  # pure-heuristic MCTS (no policy prior)
            c_puct=1.5,
            n_simulations=max(n_simulations, 20),
            rollout_depth=3,
            use_rollout=True,
        )

        # online learning (lazily enabled)
        self._online_learner = None
        self._last_rmsd = 0.0  # deviation of the last decide() from MCTS (diagnostics)

    def enable_online_learning(
        self,
        buffer_path: str = None,
        update_every: int = 5,
        capacity: int = 500,
    ):
        """Enable online learning."""
        from torusfold.scheme2.rl_optimizer import ReplayBuffer, OnlineLearner, ContinuousAssemblyPolicy

        buffer = ReplayBuffer(capacity=capacity)
        if buffer_path:
            buffer.load(buffer_path)
            print(f"  [RL] loaded {len(buffer)} past trajectories from {buffer_path}")

        # use ContinuousAssemblyPolicy (for PPO training)
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
        """Decide pair_weights + MD step count via MCTS search.

        Online learning: every decide() records a trajectory and periodically retrains.

        Returns:
            (pair_weights, n_steps)
        """
        # build the RL state
        if len(coords) == 0:
            # empty coordinates: skip RL and return default parameters
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

        # MCTS search: returns the best P coordinates
        best_coords = self.mcts.search(state, far_pairs)

        # online learning: record a trajectory
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

        # derive pair_weights from the shift between the optimal and original coordinates
        pair_weights = {}
        L = len(coords)
        for k, (i, j) in enumerate(far_pairs):
            if i >= L or j >= L:
                continue
            dist_before = float(np.linalg.norm(coords[i] - coords[j]))
            dist_after = float(np.linalg.norm(best_coords[i] - best_coords[j]))
            # weight pairs whose distance shrank
            if dist_after < dist_before:
                ratio = dist_before / max(dist_after, 0.1)
                pair_weights[(i, j)] = min(5.0, max(0.1, ratio))
            else:
                pair_weights[(i, j)] = 1.0

        # MD step count: driven by the deviation found by RL's MCTS search (replaces the
        # hard-coded pair_rate threshold). If MCTS best_coords deviate strongly from the
        # current coordinates, RL thinks the conformation still needs large adjustments ->
        # run more MD; small deviation -> near convergence, run less. This is RL's internal
        # real signal.
        rmsd = 0.0
        if len(best_coords) == L and L > 0:
            diff = np.asarray(best_coords) - np.asarray(coords)
            rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        scale = self.md_step_scale
        if rmsd > 3.0:
            n_steps = 1000000   # large adjustment: 1M
        elif rmsd > 1.5:
            n_steps = 500000    # medium: 500K
        else:
            n_steps = 200000    # near convergence: 200K
        # length scaling: halve the steps when L>500, halve again when L>1000
        l_scale = 1.0
        if L > 1000:
            l_scale = 0.25
        elif L > 500:
            l_scale = 0.5
        n_steps = max(1000, int(n_steps * scale * l_scale))
        self._last_rmsd = rmsd

        return pair_weights, n_steps

    def save_online_state(self):
        """Save the online-learning state (policy + buffer)."""
        if self._online_learner and self._online_buffer_path:
            self._online_learner.save(self._online_buffer_path)


@dataclass
class LongPipelineResult:
    """Result of the long-chain pipeline."""
    sequence: str
    secondary_structure: str
    coords_cg: np.ndarray              # (L, 3) CG P coordinates
    coords_aa: Optional[np.ndarray]    # (N_atoms, 3) all-atom coordinates
    energy_cg: float
    energy_aa: float
    rmsd_to_native: Optional[float]
    pair_rate: float
    cross_segment_ok_rate: float
    n_segments: int
    n_candidates: int
    runtime_seconds: float
    fidelity_history: List[Dict]
    hbond_rate: float = 0.0  # real hydrogen-bond satisfaction rate (all-atom level, <3.6Å)
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
    # segmented assembly parameters
    use_rhofold: bool = False,
    n_candidates: int = 1,
    # adaptive MSA (avoids RhoFold collapse on a single sequence)
    use_msa: bool = True,
    rfam_cm: str = "",
    rfam_dir: str = "",
    msa_blocks: Optional[List[Dict]] = None,
    # 5-bead CG refinement
    use_5bead: bool = True,
    # Metadynamics enhanced sampling
    use_metad: bool = True,
    metad_n_steps: int = 200000,
    # PyRosetta conditional refinement (Level 2.6, WSL)
    use_pyrosetta: bool = True,
    # PPR base-pair hydrogen-bond repair (Level 5.5)
    use_ppr: bool = True,
    ppr_max_rounds: int = 5,
    # checkpoint resume
    resume: bool = True,
    # structRFM multi-task prediction heads (opt-in)
    use_multi_task_heads: bool = False,
    multitask_head_weights: Optional[str] = None,
    use_structrfm: bool = False,
    ss_head_weight: float = 1.0,
    pair_head_weight: float = 1.0,
    bsj_head_weight: float = 0.5,
    clash_head_weight: float = 0.3,
    # Pair weights: keep the method-agreement weights, or overwrite them with the RCM
    # confidence. Default False; the reason is measured and written out at the block below.
    use_rcm_reweight: bool = False,
) -> LongPipelineResult:
    """Full isRNAcircLong pipeline.

    Args:
        sequence: RNA sequence
        secondary_structure: secondary structure (dot-bracket)
        output_dir: output directory
        max_seg_len: maximum segment length
        overlap: overlap length
        n_relax_rounds: number of iterative relaxation rounds (default 6, enough in most
            cases; the early-stop mechanism may end sooner)
        n_rest2_replicas: number of REST2 replicas
        rest2_nsteps: REST2 step count
        use_rl_relax: whether Level 2 uses RL guidance (False = ablation, fixed parameters)
        use_rl_mcts: whether Level 3 uses RL-MCTS (False = ablation)
        rl_n_simulations: number of RL simulations
        md_step_scale: step-count scaling factor for each Level 2 MD round. Default 0.1
            (reduces steps to 1/10: 1M->100K / 500K->50K / 200K->20K). Controls total Level 2
            runtime; raise back to 0.3~0.5 when the conformation is not converging well.
        use_5bead: whether Level 2.3 uses 5-bead CG refinement (default True).
            5-bead: P/S/B1/B2/B3 per nucleotide, giving more accurate stacking/H-bond
            geometry than 3-bead.
        use_metad: whether Level 3.5 uses Metadynamics enhanced sampling (default True).
            Gaussian hills are added along the CVs (BSJ distance / pairing contacts /
            radius of gyration) to cross free-energy barriers.
        metad_n_steps: total Metadynamics MD steps, default 50000.
        nrep: number of Level 2 REMD replicas (IsRNAcirc runs replicas concurrently on
            multiple cores). When >1 each replica has its own temperature/seed and a
            concurrent LAMMPS process; enough CPU cores are required. Default 1
            (single replica, consistent with the legacy version).
        platform: OpenMM/LAMMPS platform
        verbose: whether to print detailed information
        use_rhofold: True predicts each chunk with RhoFold+, False uses isRNAcirc Type=0
        n_candidates: candidate count per chunk
        use_msa: True enables adaptive MSA (real MSA preferred, pseudo MSA as fallback),
            preventing RhoFold+ from collapsing on engineered single sequences. Default True.
        rfam_cm: path to the Rfam CM library (used by cmsearch to find real MSAs; WSL path)
        rfam_dir: Rfam data directory (contains known-family MSAs, reused as real MSAs)
        msa_blocks: optional, MSA-aware anchored blocking intervals
            [{"start","end","msa_path","source"}, ...].
            When provided, blocks are split along the anchored intervals (each real-MSA
            chunk uses its corresponding MSA).

    Returns:
        LongPipelineResult
    """
    t0 = time.time()

    # preheat WSL asynchronously (avoids later cold-start latency)
    if use_pyrosetta:
        _preheat_wsl()

    # sequence normalization: unify case + T->U (RNA)
    sequence = sequence.upper().replace("T", "U")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # checkpoint resume: checkpoint file
    _ckpt_path = output_path / "_checkpoint.json"
    ckpt = _load_checkpoint(_ckpt_path) if resume else {}
    _raw_level = ckpt.get("level", -1)
    ckpt_level = float(_raw_level) if _raw_level is not None else -1
    # cumulative checkpoint: each Level appends fields; the full dict is written on save
    # (prevents field loss by overwriting)
    _ckpt_data = dict(ckpt)

    def _save_ckpt(level: float, **extra):
        """Append fields to _ckpt_data and save it."""
        _ckpt_data["level"] = level
        _ckpt_data.update(extra)
        _save_checkpoint(_ckpt_path, _ckpt_data)
        print(f"  [checkpoint] level={level}, fields={list(_ckpt_data.keys())}")

    L = len(sequence)
    if verbose:
        print(f"=== isRNAcircLong: {L}nt ===")
        if ckpt_level >= 0:
            print(f"  [resume] resuming after Level {ckpt_level}")

    # ── Level 0: Partition Function BPP + confidence-tiered restraints ──
    # restore as long as the JSON has the complete fields, independent of the ckpt_level value
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
            print(f"\n[Level 0] restored from checkpoint: short-range {len(pairs)}, far {len(far_pairs)}")
    else:
        if ckpt_level >= 0 and verbose:
            print(f"\n[Level 0] checkpoint incomplete, recomputing pairs...")
        if verbose:
            print("\n[Level 0] Partition Function BPP + confidence tiering...")

        # ── 1. ViennaRNA Partition Function ──
        import RNA as _RNA
        md_pf = _RNA.md()
        md_pf.circ = 1  # circular mode
        fc_pf = _RNA.fold_compound(sequence, md_pf)
        ss_pf, pf_energy = fc_pf.pf()

        # extract the full-probability pairing list
        plist = fc_pf.plist_from_probs(0.01)  # P > 1%

        # confidence tiering
        bpp_high = [(ep.i - 1, ep.j - 1, ep.p) for ep in plist if ep.p > 0.9]
        bpp_mid = [(ep.i - 1, ep.j - 1, ep.p) for ep in plist if 0.5 < ep.p <= 0.9]
        bpp_low = [(ep.i - 1, ep.j - 1, ep.p) for ep in plist if 0.1 < ep.p <= 0.5]

        if verbose:
            print(f"  PF energy: {pf_energy:.2f}")
            print(f"  high-confidence (P>0.9):     {len(bpp_high)} pairs")
            print(f"  medium-confidence (0.5-0.9): {len(bpp_mid)} pairs")
            print(f"  low-confidence (0.1-0.5):    {len(bpp_low)} pairs")

        # ── 2. BPP matrix + MFE (reuse fc_pf: saves one O(L^3) PF and stays consistent with circ=1) ──
        try:
            _bpp_raw = np.array(fc_pf.bpp(), dtype=np.float64)
            bpp = _bpp_raw[1:, 1:] if _bpp_raw.shape[0] > len(sequence) else _bpp_raw.copy()
            if verbose:
                print(f"  BPP matrix: {bpp.shape}, max P={float(bpp.max()):.3f}")
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

        # DivideFold: recursively split into blocks and predict independent structures
        ss_divide = None
        try:
            _dd_root = os.environ.get("TF_DIVIDEFOLD_ROOT") or str(Path(__file__).resolve().parents[3] / "DivideFold-main")
            sys.path.insert(0, os.path.join(_dd_root, "src"))
            import RNA as _RNA_DL
            def _rnafold_api(seq):
                md = _RNA_DL.md(); fc = _RNA_DL.fold_compound(seq, md)
                ss, _ = fc.mfe(); return ss
            # DivideFold runs as a pure-CPU subprocess (GPU devices hidden in its env)
            _dd_runner = os.environ.get("TF_DIVIDEFOLD_RUNNER") or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scripts", "_dd_runner.py")
            _sys_python = os.environ.get("TF_DIVIDEFOLD_PYTHON") or sys.executable
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
                        print(f"  DivideFold subprocess failed: {_dd_result.stderr[:200]}")
            except Exception as _dd_err:
                ss_divide = None
                if verbose:
                    print(f"  DivideFold subprocess error: {_dd_err}")
            if ss_divide is not None:
                if verbose:
                    print(f"  DivideFold: {ss_divide.count('(')} pairs")
        except Exception as e:
            if verbose:
                print(f"  DivideFold unavailable: {e}")

        # ── 3. fuse restraints from multiple sources ──
        mfe_set = set((i, j) for i, j in pairs_mfe)
        pf_high_set = set((i, j) for i, j, _ in bpp_high)
        pf_mid_set = set((i, j) for i, j, _ in bpp_mid)

        # DivideFold pairing set
        divide_set = set()
        if ss_divide:
            stack = []
            for i, c in enumerate(ss_divide):
                if c == "(": stack.append(i)
                elif c == ")" and stack:
                    divide_set.add((stack.pop(), i))

        # ── hard restraints: consistent across methods (>=2 sources) ──
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

        # ── soft restraints: MFE + PF medium-confidence + DivideFold (not truncated) ──
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

        # ── RCM reweighting (OFF by default; opt in with use_rcm_reweight=True) ──
        #
        # This overwrites every method-agreement weight built just above with
        # compute_rcm_score(...)['confidence'], a ratio of kmer reverse-complement match counts.
        # That quantity was measured against the Watson-Crick pairs of the deposited-structure
        # database (scripts/measure_pair_weight_quality.py) and it does not carry what this use
        # assumes. Against geometry-matched negatives its AUC is 0.5019, 95 percent CI
        # [0.4939, 0.5093]; its sequence-specific component, real minus shuffled, is +0.0021
        # with a CI spanning zero (p=0.350). It is also not a graded weight: of 2330 true pairs
        # 88.84 percent score exactly 0.0 and only five distinct values occur in total. pair_w
        # multiplies the WC spring stiffness in torch_cgsim.cg_energy_forces, so a confidence of
        # zero switches that restraint off, and the old behaviour was disabling pairing for
        # about 89 percent of the pairs it was handed. The one-bit base complementarity the
        # pipeline already has in hand scores 0.8197 on the same rows.
        #
        # Left in place rather than deleted so earlier runs stay reproducible.
        #
        # The weights this replaces -- 1.0 hard, 1.0/0.8/0.6/bpp_mid soft -- were unmeasured when
        # that note was written. They have since been measured, on the same rows and through the
        # same harness, which now scores them as "method_agree" (scripts/
        # measure_pair_weight_quality.py):
        #
        #   N1, band-matched non-WC: geometry held fixed, chemistry removed
        #       method_agree     AUC 0.7080 [0.6653, 0.7401]  shuffled 0.4986  component +0.2094
        #       rcm_confidence   AUC 0.5044 [0.4801, 0.5269]  shuffled 0.5029  component +0.0015
        #       method_agree - rcm_confidence = +0.2036 [+0.1583, +0.2425]  p = 0.000
        #   N2, separation-matched, any identity, any distance
        #       method_agree     AUC 0.7075 [0.6717, 0.7455]  shuffled 0.5005  component +0.2070
        #       rcm_confidence   AUC 0.5519 [0.5276, 0.5859]  shuffled 0.5006  component +0.0513
        #       method_agree - rcm_confidence = +0.1556 [+0.1129, +0.1973]  p = 0.000
        #
        # So the channel does carry sequence information, far more than the reweight that would
        # replace it, and turning this off is a measured improvement. Three things it does NOT
        # establish, each measured so that nobody re-proposes them:
        #
        #   * Multiplying these weights by the one-bit base complementarity the pipeline already
        #     holds changes NOTHING: ma_x_wc is identical to method_agree on both sets, because
        #     every pair that receives a weight is already Watson-Crick compatible. There is no
        #     chemistry left for a lookup to add.
        #   * Whether the channel adds anything BEYOND chemistry and geometry cannot be measured
        #     on this database. The third negative set that would test it -- separation-matched,
        #     in-band AND base-complementary, so both are held fixed -- is empty BY CONSTRUCTION,
        #     because _chain_residues accepts a pair exactly when it is WC compatible, at least
        #     3 residues apart, and inside the 9.0-11.5 A band. The label IS that conjunction.
        #   * N2 is geometry-dominated: geom_in_band alone scores 0.9833 there with a shuffled
        #     control of 0.9833, i.e. zero sequence content. The one set on which these weights
        #     lose to a rival is therefore not a fair ground for that comparison.
        if use_rcm_reweight:
            try:
                from torusfold.scheme2.rcm import compute_rcm_score
                _flank = min(200, len(sequence) // 4)  # flanking window 200bp
                _rcm_pairs = []
                for i, j, w in pairs:
                    # extract flanking sequence (circular RNA: upstream of i + downstream of j)
                    seq_up = sequence[max(0, i - _flank):i]
                    seq_down = sequence[j:min(len(sequence), j + _flank)]
                    if len(seq_up) >= 5 and len(seq_down) >= 5:
                        rcm = compute_rcm_score(seq_up, seq_down)
                        _rcm_pairs.append((i, j, rcm['confidence']))
                    else:
                        _rcm_pairs.append((i, j, w))  # too short: keep the original weight
                pairs = _rcm_pairs
                if verbose:
                    _rcm_vals = [w for _, _, w in pairs]
                    print(f"  RCM reweighting: mean={np.mean(_rcm_vals):.3f}, "
                          f"min={min(_rcm_vals):.3f}, max={max(_rcm_vals):.3f}")
            except Exception as e:
                if verbose:
                    print(f"  RCM reweighting skipped: {e}")
        elif verbose:
            print("  pair weights: method-agreement (hard 1.0, soft 1.0/0.8/bpp_mid/0.6); "
                  "RCM reweighting off")

        if verbose:
            n_multi = sum(1 for v in pair_votes.values() if v >= 2)
            print(f"  PF high-confidence: {len(pf_high_set)}, MFE: {len(mfe_set)}, DivideFold: {len(divide_set)}")
            print(f"  multi-method consistent (>=2): {n_multi}")
            print(f"  hard restraints: {len(hard_pairs)}, soft restraints: {len(soft_pairs)}")
            print(f"  total restraints: {len(pairs)}")

        # ── 3b. NCM non-canonical pairing detection (P0) ──
        # pass in the BPP matrix already computed at Level 0 so NCM does not rerun
        # ViennaRNA PF
        ncm_type_map = {}  # (i,j) -> type; the downstream CG force field assigns a target distance per type
        ncm_type_target = {
            "HOOGSTEEN": 10.5,  # Hoogsteen: distance similar to WC
            "SUGAR":     10.5,  # Sugar: similar to WC
            "SHEAR":     11.5,  # Shear: slightly farther
            "STACK":     10.0,  # Stacking: near-parallel stacking distance
        }
        try:
            from torusfold.scheme2.ncm_detector import detect_ncms_from_bpp
            ncm_raw = detect_ncms_from_bpp(sequence, bpp, hard_pairs)
            # ncm_raw: [(i, j, weight, type)]
            for _i, _j, _w, _etype in ncm_raw:
                ncm_type_map[(_i, _j)] = _etype
                _target = ncm_type_target.get(_etype, 10.5)
                # use the weight and the type label as a soft restraint
                pairs.append((_i, _j, _w))
            if verbose:
                _ncm_type_counts = {}
                for _, etype in ncm_type_map.items():
                    _ncm_type_counts[etype] = _ncm_type_counts.get(etype, 0) + 1
                print(f"  NCM non-canonical pairs: {len(ncm_raw)}  "
                      f"type distribution: {_ncm_type_counts}")
        except Exception as e:
            if verbose:
                print(f"  NCM detection skipped: {e}")

        # ── 4. scan far/long-range pairs via pair_graph ──
        from torusfold.scheme2.pair_graph import build_full_pair_graph, extract_stem_blocks
        # pairs are already (i, j, w) triples
        _, scan_pairs, far_pairs = build_full_pair_graph(
            sequence, pairs, do_scan=True,
        )
        stem_blocks = extract_stem_blocks(pairs, scan_pairs)

        # ── 4b. pseudoknot detection (critical for circRNA) ──
        try:
            from torusfold.scheme2.pair_graph import detect_pseudoknots_from_bpp
            pk_pairs = detect_pseudoknots_from_bpp(
                sequence, bpp, pairs,
                pk_threshold=0.1, min_confidence=0.3,
                is_circular=True,
            )
            # add pseudoknot pairs as soft restraints (weight determined by confidence)
            for pk_i, pk_j, pk_conf in pk_pairs:
                pairs.append((pk_i, pk_j, pk_conf))
                # pseudoknot pairs inherently span long range -> add to far_pairs
                _topo_dist = min(abs(pk_i - pk_j), len(sequence) - abs(pk_i - pk_j))
                if _topo_dist > 100:
                    far_pairs.append((pk_i, pk_j))
            if verbose and pk_pairs:
                print(f"  pseudoknot candidates: {len(pk_pairs)} pairs (added to restraints)")
        except Exception as e:
            if verbose:
                print(f"  pseudoknot detection failed: {e}")

        # write the NCM type map into the checkpoint for the downstream CG force field
        # (ncm_type_map is defined in 3b above)

        if verbose:
            print(f"  short-range pairs: {len(pairs)}, far pairs: {len(far_pairs)}")

        _save_ckpt(0,
            pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
            bpp=bpp, ss_consensus=ss_pf,
            bpp_high=bpp_high, bpp_mid=bpp_mid,
            pf_energy=pf_energy,
        )

        # full Level 0 diagnostics
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

    # ── Level 0 data export ──
    try:
        from torusfold.scheme2.data_exporter import export_level0_bpp, export_level0_ncm
        _ncm_pairs_for_export = ncm_pairs if 'ncm_pairs' in dir() else []
        export_level0_bpp(bpp, len(sequence), str(output_path))
        export_level0_ncm(_ncm_pairs_for_export, len(sequence), str(output_path))
    except Exception as _err:
        raise

    # ── Level 1: segmented Vfold3D + assembly ──
    # Level 1 can only be reused when Level 0 was also restored from the checkpoint;
    # otherwise the pairs are inconsistent
    _l1_from_ckpt = (_l0_from_ckpt and "coords_vfold" in ckpt and "n_segments" in ckpt)
    if _l1_from_ckpt:
        coords_vfold = ckpt["coords_vfold"]
        n_segments = ckpt["n_segments"]
        segments = ckpt.get("segments", [])
        chunk_confidences = [0.5] * n_segments  # not saved in the checkpoint; use defaults
        _chunk_unc = [0.5] * n_segments
        if verbose:
            print(f"\n[Level 1] restored from checkpoint: {n_segments} segments")
    else:
        if verbose:
            print("\n[Level 1] segmented Vfold3D + assembly...")
        from torusfold.scheme2.segmented_vfold3d import segmented_vfold3d_pipeline, split_sequence

        # segment info (MSA-aware blocking is optional)
        segments = split_sequence(
            sequence, secondary_structure, max_seg_len, overlap,
            msa_blocks=msa_blocks,
        )
        n_segments = len(segments)

        if verbose:
            print(f"  segmentation mode: {'RhoFold+' if use_rhofold else 'isRNAcirc Type=0'}, "
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
            # ── merge NCM ensemble distance evidence into pairs (soft restraints) ──
            if ncm_ens_pairs:
                _ncm_dist = [(gi, gj, conf) for gi, gj, _t, conf in ncm_ens_pairs]
                pairs += _ncm_dist
                if verbose:
                    print(f"  NCM ensemble distance evidence: +{len(_ncm_dist)} soft restraints")
            _l1_ok = True
        except Exception as e:
            if verbose:
                print(f"  3D prediction failed: {e}, using default helix coordinates")
            coords_vfold = _default_helix_coords(L)
            chunk_confidences = [0.0] * n_segments
            ncm_ens_pairs = []

        if verbose:
            print(f"  segments: {n_segments}, estimated initial RMSD: ~30-40A")

        # Level 1 validation (only store the checkpoint once it passes)
        v1 = _validate_structure(coords_vfold, pairs, bpp, sequence, "Level 1")
        print(f"  [validate Level 1] clash={v1['clash_count']}, pair_rate={v1['pair_rate']:.2f}, "
              f"bond_q={v1['bond_quality']:.2f}, valid={v1['is_valid']}")

        if _l1_ok:
            _save_ckpt(1,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
            )
        if not v1["is_valid"]:
            print(f"  [warn] Level 1 output quality is poor: {v1['warnings']}")

        # ── P2: overlap-region confidence assessment ──
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
                    print(f"  [P2] per-residue confidence: mean={mean_conf:.3f}, low={low_conf}/{L}")
        except Exception as e:
            if verbose:
                print(f"  [P2] confidence assessment skipped: {e}")

    # per-chunk Level 1 diagnostics
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
            # save each chunk to its own directory
            for _diag_c in _diag_chunks:
                _seg_dir = output_path / "vfold3d" / f"chunk_{_diag_c['chunk_id']}"
                _seg_dir.mkdir(parents=True, exist_ok=True)
                (_seg_dir / "diag.json").write_text(json.dumps(_diag_c, indent=2))
    except Exception as _err:
        raise

    # ── Level 1 data export ──
    try:
        from torusfold.scheme2.data_exporter import export_level1_chunks, export_level1_weights
        _chunk_unc = [1.0] * len(segments) if segments else []
        export_level1_chunks(segments, chunk_confidences, _chunk_unc, str(output_path))
        _region_w = [{"rhofold": 0.5, "trrna2": 0.2, "rnabpflow": 0.3}] * len(segments)
        export_level1_weights(_region_w, str(output_path))
    except Exception as _err:
        raise

    # ── Level 1.5: CG global restraint relaxation (smooths the seams of Vfold3D block assembly) ──
    # Level 1.5 depends on Level 1 output; only restore if Level 1 was also restored
    _l15_from_ckpt = (_l1_from_ckpt and "coords_relaxed" in ckpt)
    if _l15_from_ckpt:
        coords_vfold = ckpt["coords_relaxed"]
        if verbose:
            print(f"\n[Level 1.5] restored globally relaxed coordinates from checkpoint")
    else:
        if verbose:
            print(f"\n[Level 1.5] CG global restraint relaxation...")
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

        # torch GPU relaxation (full CG force field, replaces OpenMM CPU)
        _pairs_for_relax = [(i, j, w) for (i, j, w) in pairs
                            if 0 <= i < L and 0 <= j < L]
        _relaxed_l15, _metrics_l15 = relax_structure(
            _relax_coords, sequence,
            far_pairs=None,
            n_steps=5000,
            pairs_all=_pairs_for_relax if _pairs_for_relax else None)
        _coords_relaxed = _relaxed_l15

        # torch GPU relaxation finished; extract the result
        _p_coords_relaxed = _coords_relaxed
        _e = 0.0  # the torch GPU version does not output an OpenMM energy

        if len(_p_coords_relaxed) == L:
            coords_vfold = _p_coords_relaxed
            if verbose:
                print(f"    global relaxation (torch GPU): bond_viol={_metrics_l15['final']['bond_violations']}")
        else:
            if verbose:
                print(f"    output dimension mismatch ({len(_p_coords_relaxed)} vs {L}), skipping")
        _l15_ok = True
    except Exception as e:
        if verbose:
            print(f"    CG relaxation failed: {e}, using the original coordinates")

    # Level 1.5 validation
    v15 = _validate_structure(coords_vfold, pairs, bpp, sequence, "Level 1.5")
    print(f"  [validate Level 1.5] clash={v15['clash_count']}, bond_q={v15['bond_quality']:.2f}, valid={v15['is_valid']}")
    if 'v1' in dir() and v15["clash_count"] < v1["clash_count"]:
        print(f"  [OK] relaxation reduced clashes: {v1['clash_count']} -> {v15['clash_count']}")

    # save the PDB after Level 1.5 relaxation
    _l15_pdb = str(output_path / "level1_5_relaxed.pdb")
    _write_coords_pdb(coords_vfold, sequence, _l15_pdb)
    if verbose:
        print(f"  [PDB] Level 1.5: {_l15_pdb}")

    # ── Level 1.5 data export ──
    try:
        from torusfold.scheme2.data_exporter import export_level15_trajectory
        export_level15_trajectory([], str(output_path))
    except Exception as _err:
        raise

    # forced Level 1.5 checkpoint (saved every 2 levels)
    if _l15_ok:
        try:
            _save_ckpt(1.5,
                coords_relaxed=coords_vfold,
                time=time.time(),
            )
        except Exception as _err:
            raise

    # ── structRFM multi-task prediction heads (opt-in) ──
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
                print(f"  [MultiTask] head initialization failed: {e_mt}")
            multitask_heads = None

    remd_history = []  # Level 2 REMD convergence data

    # ── Level 2: parallel segmented CG->all-atom + RL-scheduled REMD ──
    _l2_from_ckpt = ("best_coords" in ckpt and "best_energy" in ckpt)
    if _l2_from_ckpt:
        best_coords = ckpt["best_coords"]
        best_energy = ckpt["best_energy"]
        segments = ckpt.get("segments", [])
        # check whether the coordinates are valid (they may be an empty array)
        if len(best_coords) == 0:
            if verbose:
                print(f"  [warn] checkpoint coordinates are empty; using Level 1 coordinates")
            best_coords = coords_vfold.copy()
            best_energy = _estimate_energy(best_coords, pairs, sequence)
        from torusfold.scheme2.multifidelity_scheduler import RuleScheduler, SimulationState
        state = SimulationState()
        state.pair_rate = ckpt.get("pair_rate", 0.0)
        state.cross_segment_ok_rate = ckpt.get("cross_segment_ok_rate", 0.0)
        state.energy = ckpt.get("energy", 0.0)
        state.clash_count = ckpt.get("clash_count", 0)
        # scheduler must also be defined on the checkpoint-resume path (referenced by the return statement)
        scheduler = RuleScheduler()
        # these variables need initialization when resuming (referenced by later loop rounds)
        coords_prev = best_coords.copy()
        prev_energy = best_energy
        no_improve_count = 0
        inject_frac = 0.3
        metrics = RelaxationMetrics()
        if verbose:
            print(f"\n[Level 2] restored from checkpoint: E={best_energy:.0f}")
    else:
        if verbose:
            print(f"\n[Level 2] parallel segmented CG->all-atom + {'RL-scheduled REMD' if use_rl_relax else 'fixed REMD'}...")

        # resolve the OpenMM platform (needed by the Level 2 OpenMM GPU refinement)
        if platform == "auto":
            from torusfold.scheme2.rest2_sampler import detect_openmm_platform
            resolved_platform = detect_openmm_platform()
        else:
            resolved_platform = platform

        # detect a GPU for REMD acceleration
        _remd_platform_name = resolved_platform  # default to the platform resolved by auto
        try:
            import openmm as _omm
            for _try_gpu in ["CUDA", "OpenCL"]:
                try:
                    _omm.Platform.getPlatformByName(_try_gpu)
                    _remd_platform_name = _try_gpu
                    if verbose:
                        print(f"  [Level 2] GPU detection: {_try_gpu} available, using GPU acceleration for REMD")
                    break
                except Exception:
                    pass
        except Exception as _err:
            raise

        from torusfold.scheme2.multifidelity_scheduler import RuleScheduler, SimulationState, FidelityLevel
        scheduler = RuleScheduler()
        state = SimulationState()
        # load the trained RL policy (Level 2 RelaxationRL)
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
        inject_frac = 0.3  # dynamic far-pair injection ratio
        metrics = RelaxationMetrics()

        def _segmented_cg_to_allatom(cg_coords_full, seg_list, out_dir, seq):
            """Convert segmented CG to all-atom in parallel, assembling a complete all-atom PDB."""
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

            # run segmented CG->all-atom in parallel (each exe call is an independent
            # process, so they can run concurrently)
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
                            print(f"    segment {seg_idx}: failed: {e}")

            merged_pdb = str(Path(out_dir) / "merged_aa.pdb")
            _merge_allatom_pdbs(aa_pdbs, seg_list, merged_pdb, seq)
            return merged_pdb

        # segmented CG->all-atom (done once, reused by later rounds)
        try:
            merged_aa = _segmented_cg_to_allatom(
                coords_current, segments, str(output_path / "cg2aa"), sequence,
            )
            if verbose:
                print(f"    segmented CG->all-atom done: {merged_aa}")
        except Exception as e:
            if verbose:
                print(f"    CG->all-atom failed: {e}, using Level 1 coordinates")
            energy = _estimate_energy(coords_current, pairs, sequence)
            best_coords = coords_current
            best_energy = energy
            merged_aa = None

        # iterative REMD (RL-scheduled or fixed)
        n_remd_rounds = n_relax_rounds if use_rl_relax else 1
        prev_pdb_out = None  # refined PDB path from the previous round
        round_idx = 0  # initialize so it is usable outside the loop
        metrics = RelaxationMetrics()  # initialize
        energy = 0.0  # initialize
        for round_idx in range(n_remd_rounds):
            if merged_aa is None:
                break
            if verbose:
                print(f"  REMD round {round_idx + 1}/{n_remd_rounds}:")

            # decide the REMD parameters
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
                # fixed parameters (IsRNAcirc recommended: nstep=1M, nstep_close=500K, nstru=500)
                # multiply by md_step_scale to shrink per-round Level 2 steps (default 0.1)
                n_steps = max(1000, int(1000000 * md_step_scale))
                nstep_close = max(1000, int(500000 * md_step_scale))
                if verbose:
                    print(f"    fixed: nstep={n_steps}, nstep_close={nstep_close}")

            # structRFM: fuse learned pair predictions into the RL weights
            if (multitask_heads is not None and pw
                    and len(coords_current) == L):
                try:
                    from torusfold.scheme2.multitask_heads import build_struct_condition_from_coords
                    # build the structure condition
                    struct_cond = build_struct_condition_from_coords(
                        coords_current, far_pairs, L)
                    struct_t = torch.tensor(struct_cond, dtype=torch.float32)
                    # get the bpp matrix
                    # Bug 9 fix: use the correct variable name (bpp rather than bpp_matrix)
                    bpp_tensor = None
                    if bpp is not None:
                        bpp_tensor = torch.tensor(bpp, dtype=torch.float32)
                    # predict
                    with torch.no_grad():
                        mt_out = multitask_heads(
                            SEQUENCE,
                            struct_condition=struct_t,
                            bpp_matrix=bpp_tensor,
                        )
                    # fuse pair weights: 50% MCTS + 50% learned
                    if "pair_probs" in mt_out and "clash_scores" in mt_out:
                        clash_scores = mt_out["clash_scores"].numpy()
                        for (i, j) in pw:
                            if i < L and j < L:
                                # lower the weight at clash-prone positions
                                clash_penalty = 0.5 * (clash_scores[i] + clash_scores[j])
                                pw[(i, j)] *= max(0.1, 1.0 - clash_penalty)
                        if verbose:
                            print(f"    [MultiTask] clash-score fusion done")
                except Exception as e_mt:
                    if verbose:
                        print(f"    [MultiTask] fusion failed: {e_mt}")

            try:
                round_dir = str(output_path / f"remd_r{round_idx}")
                pdb_out = None
                energy = float("inf")

                # Level 2 refinement: only use IsRNAcirc.exe (IsRNA2 force field, no
                # fallback at all). Round 0 starts from the RhoFold+ assembled coordinates
                # (pairs already folded to ~28A, bond lengths corrected to 5.9A), rather
                # than the segmentally rebuilt merged_aa (segmentation loses the global
                # fold and pairs regress to 46A). Later rounds start from the previous
                # round's refinement result.
                if round_idx == 0:
                    # Prefer the existing merged_aa.pdb (already all-atom, skips the
                    # 20-min CG->allatom rebuild). merged_aa is the segmented-assembly
                    # result and retains the global fold + all-atom coordinates. Only fall
                    # back to the RhoFold+ P-only path when merged_aa is absent.
                    _merged_aa = str(output_path / "cg2aa" / "merged_aa.pdb")
                    if Path(_merged_aa).exists():
                        refine_input = _merged_aa
                        if verbose:
                            print(f"    round 0: using merged_aa.pdb directly (already all-atom, skipping CG->allatom)")
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
                    # write last round's best_coords to a temporary PDB as input
                    # (independent of prev_pdb_out)
                    if best_coords is not None and len(best_coords) == L:
                        _prev_cg = str(output_path / f"_prev_round_cg.pdb")
                        _write_coords_pdb(best_coords, sequence, _prev_cg)
                        refine_input = _prev_cg
                    else:
                        refine_input = prev_pdb_out or merged_aa
                # prefer the torch GPU path (ROCm/HIP compatible); OpenMM is the fallback
                try:
                    from torusfold.scheme2.torch_gpu_refine import torch_gpu_refine as _refine_fn
                    _refine_name = "Torch GPU"
                except Exception as _e:
                    from torusfold.scheme2.openmm_gpu_refiner import openmm_gpu_refine as _refine_fn
                    _refine_name = "OpenMM CPU (fallback)"
                    if verbose:
                        print(f"    [!] Torch GPU unavailable: {_e}, falling back to OpenMM CPU")
                if verbose:
                    print(f"    {_refine_name} refinement (input: {'RhoFold+ start' if round_idx == 0 else 'previous round result'})...")
                # dynamic far-pair injection: adjust the injection ratio according to pair_rate
                # (the OpenMM path does not yet support far_pair_ratio; the interface is kept for compatibility)
                if round_idx > 0:
                    if metrics.pair_rate > 0.6:
                        inject_frac = min(1.0, inject_frac + 0.2)  # pairing is good, speed up injection
                    elif metrics.pair_rate < 0.3:
                        inject_frac = max(0.3, inject_frac - 0.1)  # pairing is poor, slow down injection
                n_rounds_total = max(1, n_remd_rounds)
                far_ratio = 1.0 if n_rounds_total == 1 else inject_frac
                if verbose and far_pairs and far_ratio > 0:
                    print(f"    far-pair injection: {len(far_pairs)} pairs ({_refine_name} mode)")
                _remd_reps = (64 if L > 1000
                              else max(nrep if nrep else 6, n_rest2_replicas))
                # The Level-2 REMD budget is fixed inside torch_gpu_refine: 8 rounds x 5000
                # steps per replica (torch_gpu_refine.py:176-177, use_multistage_remd=True).
                # A `_remd_steps` computed here used to be passed as remd_n_steps and was
                # silently ignored in that mode; it is removed rather than left behind
                # looking like a control that works.
                if _refine_name == "Torch GPU":
                    _refine_result = _refine_fn(
                        refine_input, round_dir,
                        sequence, secondary_structure,
                        name=f"remd_r{round_idx}",
                        nstep=max(100000, n_steps),
                        use_remd=True,
                        remd_n_replicas=_remd_reps,
                        use_multistage_remd=True,  # 8 rounds x 5000 steps/replica (fixed in torch_gpu_refine)
                        verbose=verbose,
                        use_physical_relax=True,
                        skip_minimal_fold=(round_idx > 0),
                        use_trirnasp=False,  # fully disable TriRNASP, use only the CG force field
                        use_trirnasp_force=False,
                        trirnasp_scale=0.002,  # optimal: Tri/CG ≈ 11%, the balance point
                        trirnasp_update_freq=1000,
                        # new: staged TriRNASP strategy (adapted to 5000 steps per round)
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
                        use_multistage_remd=True,  # 8 rounds x 5000 steps/replica (fixed in torch_gpu_refine)
                        verbose=verbose,
                        use_physical_relax=True,
                        skip_minimal_fold=(round_idx > 0),
                        use_trirnasp=False,  # fully disable TriRNASP, use only the CG force field
                        use_trirnasp_force=False,
                        trirnasp_scale=0.002,  # optimal: Tri/CG ≈ 11%, the balance point
                        trirnasp_update_freq=1000,
                        # new: staged TriRNASP strategy (adapted to 5000 steps per round)
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
                # openmm_gpu_refine returns (pdb, energy, diag)
                # torch_gpu_refine returns (pdb, energy, diag) — aligned
                if len(_refine_result) == 3:
                    pdb_out, energy, _refine_diag = _refine_result
                else:
                    pdb_out, energy = _refine_result
                    _refine_diag = {}

                coords_relaxed = _read_pdb_p_coords(pdb_out)
                if len(coords_relaxed) == 0:
                    if verbose:
                        print(f"    PDB read is empty, using the input coordinates")
                    coords_relaxed = coords_current
                elif np.any(np.isnan(coords_relaxed)):
                    if verbose:
                        print(f"    PDB contains NaN coordinates, using the input coordinates")
                    coords_relaxed = coords_current
                else:
                    prev_pdb_out = pdb_out  # record this round's output for reuse next round
                if verbose:
                    print(f"    E={energy:.0f}")
            except Exception as e:
                if verbose:
                    print(f"    REMD failed: {e}, skipping")
                    import traceback
                    traceback.print_exc()
                energy = _estimate_energy(coords_current, pairs, sequence)
                coords_relaxed = coords_current

            # monitor the 4 metrics
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

            # save a checkpoint after each REMD round
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

            # per-REMD-round Level 2 diagnostics
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
                _round_t0 = time.time()  # rough round timing
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

            prev_best_energy = best_energy  # used for the early-stop comparison
            if energy < best_energy and len(coords_relaxed) > 0:
                # NaN safety check: if the output coordinates contain NaN, skip this round
                if np.any(np.isnan(coords_relaxed)):
                    if verbose:
                        print(f"    [WARN] REMD output contains NaN, skipping this round's update")
                else:
                    best_energy = energy
                    best_coords = coords_relaxed.copy()

            coords_prev = coords_relaxed.copy()
            prev_energy = energy
            coords_current = coords_relaxed

            if metrics.is_converged:
                if verbose:
                    print(f"    converged!")
                break

            # early stop: halt when best_energy has not improved by >1% for 3 consecutive rounds
            if best_energy < prev_best_energy * 0.99:
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= 3:
                if verbose:
                    print(f"  [Level 2] early stop: no significant improvement for 3 rounds (E={best_energy:.0f})")
                break

        # Level 2 validation
        v2 = _validate_structure(best_coords, pairs, bpp, sequence, "Level 2")
        print(f"  [validate Level 2] clash={v2['clash_count']}, pair_rate={v2['pair_rate']:.2f}, valid={v2['is_valid']}")

    # ── Level 2 data export ──
    try:
        from torusfold.scheme2.data_exporter import export_level2_remd, export_validation
        export_level2_remd(remd_history, str(output_path))
        export_validation(v1 if 'v1' in dir() else None,
                          v15 if 'v15' in dir() else None,
                          v2 if 'v2' in dir() else None,
                          str(output_path))
    except Exception as _err:
        raise

    # ── Level 2.3: 5-bead CG refinement (more accurate stacking/H-bond geometry than 3-bead) ──
    _skip_5bead = False
    if use_5bead and best_coords is not None and len(best_coords) == len(sequence):
        # fast filter: skip 2.3 if the Level 2 output is already good enough
        if metrics.clash_count == 0 and metrics.pair_rate > 0.8:
            _skip_5bead = True
            if verbose:
                print(f"\n[Level 2.3] skipping: Level 2 output is already good (clash=0, pair_rate={metrics.pair_rate:.2f}>0.8)")
    if not _skip_5bead and use_5bead and best_coords is not None and len(best_coords) == len(sequence):
        # check whether the input coordinates contain NaN
        _has_nan = np.any(np.isnan(best_coords))
        _has_inf = np.any(np.isinf(best_coords))
        if _has_nan or _has_inf:
            if verbose:
                print(f"\n[Level 2.3] 5-bead skipped: input coordinates contain NaN/Inf (nan={_has_nan}, inf={_has_inf})")
        elif verbose:
            print(f"\n[Level 2.3] 5-bead CG refinement...")
        try:
            from torusfold.scheme2.fivebead_folding import refine_5bead
            _resolved_platform = "CPU"  # 5-bead uses CPU (5x the particle count)

            # build DL distance restraints: fused from the PF BPP + DivideFold
            _dl_constraints = []
            if bpp_high or bpp_mid:
                # high-confidence PF pairs -> strong restraints (d≈10Å B1-B1 WC distance)
                for i, j, p in bpp_high:
                    _dl_constraints.append((i, j, 10.0, p))
                # medium-confidence -> weak restraints
                for i, j, p in bpp_mid[:50]:  # cap the number
                    _dl_constraints.append((i, j, 10.0, p * 0.5))
                if verbose:
                    print(f"    DL restraints: {len(_dl_constraints)} pairs (PF)")

            p5_refined, e5_0, e5_1 = refine_5bead(
                best_coords, pairs,
                platform_name=_resolved_platform,
                sequence=sequence,
                dl_constraints=_dl_constraints if _dl_constraints else None)
            # discard when the 5-bead output contains NaN
            if np.any(np.isnan(p5_refined)) or np.any(np.isinf(p5_refined)):
                if verbose:
                    print(f"    5-bead output contains NaN/Inf, keeping the current coordinates")
            elif e5_1 < e5_0:
                # 5-bead improved on its own -> accept (different force-field scale; not
                # compared against 3-bead energies)
                best_coords = p5_refined.copy()
                if verbose:
                    print(f"    5-bead: E={e5_0:.0f} -> {e5_1:.0f} kJ/mol (improved, accepting)")
            elif verbose:
                print(f"    5-bead: E={e5_0:.0f} -> {e5_1:.0f} (not improved, keeping the 3-bead coordinates)")
        except Exception as e:
            if verbose:
                print(f"    5-bead refinement skipped: {e}")

    # ── Level 2.5: CG->allatom after REMD (convert the final CG coordinates to all-atom) ──
    if "final_aa_pdb" in ckpt:
        _final_aa_path = ckpt.get("final_aa_pdb", str(output_path / "final_allatom.pdb"))
        if verbose:
            print(f"\n[Level 2.5] restored from checkpoint: {_final_aa_path}")
    elif best_coords is not None and len(best_coords) == len(sequence):
        _final_aa_path = str(output_path / "final_allatom.pdb")
        _l25_ok = False
        if verbose:
            print(f"\n[Level 2.5] CG->all-atom after REMD: {_final_aa_path}")
        try:
            from torusfold.scheme2.isrnacirc_wrapper import cg_to_allatom
            _tmp_cg = str(output_path / "_final_cg_for_aa.pdb")
            _write_coords_pdb(best_coords, sequence, _tmp_cg)
            _cg2aa_t0 = time.time()
            cg_to_allatom(_tmp_cg, _final_aa_path, sequence)
            _cg2aa_elapsed = time.time() - _cg2aa_t0
            if verbose:
                _sz = os.path.getsize(_final_aa_path) / 1024
                print(f"    all-atom output: {_final_aa_path} ({_sz:.0f} KB)")
            _l25_ok = True

            # ── Level 2.5b: post-conversion CG->AA relaxation (far-pair restraints prevent drifting apart) ──
            # run one short relaxation with the far_pairs restraints immediately after
            # conversion, otherwise the geometric noise from conversion lets the far pairs
            # pulled in by Level 1.5/2 drift apart again.
            try:
                from torusfold.scheme2.physical_relaxation import relax_structure as _relax_25b
                if verbose and far_pairs:
                    print(f"    post-processing relaxation: {len(far_pairs)} far-pair restraints...")
                best_coords, _relax_metrics_25b = _relax_25b(
                    best_coords, sequence,
                    far_pairs=far_pairs if far_pairs else None,
                    n_steps=3000,
                    use_openmm=True,
                )
                if verbose:
                    print(f"    relaxation done: clash {_relax_metrics_25b['initial']['clash_count']} -> "
                          f"{_relax_metrics_25b['final']['clash_count']}")
            except Exception as e_relax_25b:
                if verbose:
                    print(f"    post-processing relaxation skipped: {e_relax_25b}")

            # Level 2.5 CG->allatom conversion diagnostics
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
                print(f"    CG->allatom failed: {e}")
            # Level 2.5 failure diagnostics
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

        # save the checkpoint after Level 2.5 finishes (only on success)
        if _l25_ok:
            _save_ckpt(2.5,
                final_aa_pdb=_final_aa_path,
                best_coords=best_coords,
                time=time.time(),
            )
    else:
        _final_aa_path = str(output_path / "final_allatom.pdb")

    # ── Level 2.6: PyRosetta conditional refinement (WSL, socket preferred) ──
    if not os.path.exists(_final_aa_path):
        _final_aa_path = str(output_path / "final_allatom.pdb")
    if ckpt.get("pyrosetta_done"):
        if verbose:
            print(f"\n[Level 2.6] restored from checkpoint")
    elif os.path.exists(_final_aa_path) and use_pyrosetta:
        _l26_ok = False
        if verbose:
            print(f"\n[Level 2.6] PyRosetta conditional refinement...")
        try:
            _pyrosetta_out = str(output_path / "final_allatom_refined.pdb")
            _wsl_aa = _win_to_wsl(_final_aa_path)
            _wsl_out = _win_to_wsl(_pyrosetta_out)

            # preferred: long-lived socket service (init runs once; later calls <1s)
            _result = _pyrosetta_socket_refine(
                _wsl_aa, _wsl_out, max_iter=200, verbose=verbose,
            )
            if _result is None and not _pyrosetta_server_running():
                # server is down -> start it
                if _pyrosetta_start_server(verbose=verbose):
                    _result = _pyrosetta_socket_refine(
                        _wsl_aa, _wsl_out, max_iter=200, verbose=verbose,
                    )

            if _result is not None:
                _final_aa_path = _pyrosetta_out
                _l26_ok = True
                if verbose:
                    print(f"    PyRosetta refinement done (socket): {_pyrosetta_out}")
            else:
                # fallback: one-shot subprocess (with timeout)
                if verbose:
                    print("    [fallback] falling back to subprocess mode...")
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
                        print(f"    PyRosetta timed out ({_pyrosetta_timeout}s), terminated")
                    _proc = None
                _rc = _proc.returncode if _proc is not None else -1
                if _rc == 0 and os.path.exists(_pyrosetta_out):
                    _final_aa_path = _pyrosetta_out
                    _l26_ok = True
                    if verbose:
                        print(f"    PyRosetta refinement done (subprocess): {_pyrosetta_out}")
                elif verbose:
                    _tail = ''.join(_stdout_lines[-20:]) if _stdout_lines else '(no output)'
                    print(f"    PyRosetta skipped or failed (rc={_rc}):")
                    print(f"    {_tail}")

        except Exception as e:
            if verbose:
                print(f"    PyRosetta refinement failed: {e}")

    # save the checkpoint after Level 2.6 finishes (only on success)
    if _l26_ok:
        _save_ckpt(2.6,
            final_aa_pdb=_final_aa_path,
            pyrosetta_done=True,
        )

    # ── Level 3: RL fine-tuning (continuous action space) ──
    if ckpt_level >= 3:
        if verbose:
            print(f"\n[Level 3] restored from checkpoint")
    elif use_rl_mcts and far_pairs:
        _l3_ok = False
        if verbose:
            print(f"\n[Level 3] RL fine-tuning (continuous actions, PPO {rl_n_simulations} epochs)...")
        try:
            from torusfold.scheme2.rl_optimizer import optimize_far_pairs
            _rl_l3_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rl_policy_b0.pth")
            if not Path(_rl_l3_path).exists():
                _rl_l3_path = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "rl_policy_bootstrap.pth")
            # check whether best_coords dimensions match the CG particle count
            # IsRNAcirc outputs an all-atom PDB, but far_pairs index into the CG trace (0~L-1)
            # if the best_coords dimension != len(sequence), skip RL fine-tuning
            if len(best_coords) != len(sequence):
                if verbose:
                    print(f"    skipping: best_coords dimension ({len(best_coords)}) != sequence length ({len(sequence)}); IsRNAcirc outputs an all-atom PDB")
            else:
                opt_p, cg_orig, rl_info = optimize_far_pairs(
                    best_coords, sequence, far_pairs, stem_blocks,
                    n_simulations=rl_n_simulations,
                    policy_path=_rl_l3_path if Path(_rl_l3_path).exists() else None,
                )
                best_coords = opt_p
                _l3_ok = True
                if verbose:
                    print(f"    RL done: reward={rl_info.get('reward_after', 0):.4f}")
        except Exception as e:
            if verbose:
                print(f"    RL fine-tuning failed: {e}")
        if _l3_ok:
            _save_ckpt(3,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
                rl_done=True,
            )

    # ── Level 3.5: Metadynamics enhanced sampling (crossing free-energy barriers along the CVs) ──
    # GPU batched version preferred (torch.cuda); OpenMM CPU is the fallback.
    if ckpt_level >= 3.5:
        if verbose:
            print(f"\n[Level 3.5] restored from checkpoint")
    elif use_metad and best_coords is not None and len(best_coords) == len(sequence):
        _l35_ok = False
        _use_gpu_meta = False
        try:
            import torch as _torch_meta
            _use_gpu_meta = _torch_meta.cuda.is_available()
        except Exception:
            pass

        if _use_gpu_meta:
            # ── GPU batched path: all replicas run in batch on a single GPU ──
            if verbose:
                print(f"\n[Level 3.5] GPU batched Metadynamics "
                      f"(8 replicas, well-tempered, {metad_n_steps} steps)...")
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
                        print(f"    GPU-MetaD E={meta_e:.0f} (better than current)")
                elif verbose:
                    print(f"    GPU-MetaD E={meta_e:.0f} "
                          f"(not better than {best_energy:.0f}, keeping)")
                _l35_ok = True
            except Exception as e:
                if verbose:
                    print(f"    GPU-MetaD failed, falling back to OpenMM: {e}")

        if not _l35_ok:
            # ── OpenMM CPU fallback: 2 replicas on independent threads ──
            if verbose:
                _fallback_tag = "OpenMM" if not _use_gpu_meta else "fallback OpenMM"
                print(f"\n[Level 3.5] {_fallback_tag} Metadynamics "
                      f"(2 replicas, well-tempered, {metad_n_steps} steps)...")
            try:
                from torusfold.scheme2.metadynamics_sampler import MetaDynamicsSampler
                from concurrent.futures import ThreadPoolExecutor, as_completed

                def _run_metad_replica(coords_init, seq, prs, n_steps, replica_id):
                    """Run a single MetaD replica (on an independent thread)."""
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
                              f"E={meta_e:.0f} (better than current)")
                elif verbose:
                    print(f"    MetaD (replica {best_replica[0]}) "
                          f"E={meta_e:.0f} (not better than {best_energy:.0f}, keeping)")
                _l35_ok = True
            except Exception as e:
                if verbose:
                    print(f"    MetaD skipped: {e}")

        if _l35_ok:
            _save_ckpt(3.5,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
            )

    # ── Level 4: REST2 refinement ──
    if ckpt_level >= 4:
        if verbose:
            print(f"\n[Level 4] restored from checkpoint")
    else:
        # resolve the "auto" platform
        if platform == "auto":
            from torusfold.scheme2.rest2_sampler import detect_openmm_platform
            resolved_platform = detect_openmm_platform()
        else:
            resolved_platform = platform
        if verbose:
            print(f"\n[Level 4] REST2 refinement ({n_rest2_replicas} replicas, platform={resolved_platform})...")
        _l4_ok = False
        _use_gpu_rest2 = torch.cuda.is_available()
        use_trirnasp = False  # off by default; enabled after Level 2 integration
        if _use_gpu_rest2:
            try:
                # ── batched GPU REST2×REMD ──
                from torusfold.scheme2.torch_cgsim import BatchedREMD2D
                _remd2d = BatchedREMD2D(
                    n_t=8,
                    t_lo=300.0, t_hi=1000.0,
                    lambdas=(1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65),
                    exchange_interval=1000,
                    use_trirnasp=use_trirnasp,
                    sequence=sequence,
                    force_refresh_freq=500,
                    # relaxation parameters (aligned with the OpenMM version)
                    relax_bond_k=500.0,
                    relax_angle_k=200.0,
                    relax_pair_k=500.0,
                    restraint_k=500.0,
                )
                # best_coords is always P-only Cartesian coordinates in Å.
                # BatchedREMD2D expects the same public interface and a step count.
                _pairs_flat = [(i, j, w) for i, j, w in pairs]
                best_coords, energy, diag = _remd2d.run(
                    best_coords, _pairs_flat, n_steps=rest2_nsteps,
                    verbose=verbose)
                coords_rest2 = best_coords
                _l4_ok = True
            except Exception as e_gpu:
                if verbose:
                    print(f"    GPU REST2 failed, falling back to OpenMM: {e_gpu}")
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
                    print(f"    [2D-REMD] T-axis acceptance: "
                          f"{['%.0f%%' % (a*100) for a in _acc_t]}")
                if _acc_l:
                    print(f"    [2D-REMD] lambda-axis acceptance: "
                          f"{['%.0f%%' % (a*100) for a in _acc_l]}")
                _rest2_snaps = [coords_rest2]
                # cluster selection: choose the best from the REST2 snapshots
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
                    print(f"    REST2 failed: {e}")
        if _l4_ok:
            _save_ckpt(4,
                pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
                coords_vfold=coords_vfold, n_segments=n_segments,
                segments=segments,
                best_coords=best_coords, best_energy=best_energy,
            )

    # ── Level 5: AMBER RNA.OL3 all-atom refinement (with C1'-C1' pair restraints) ──
    if ckpt_level >= 5:
        if verbose:
            print(f"\n[Level 5] restored from checkpoint")
    else:
        _l5_ok = False
        if verbose:
            print(f"\n[Level 5] AMBER RNA.OL3 all-atom refinement (minimization + MD)...")
        try:
            # preferred: amber_refine (full version with C1'-C1' pair restraints + A-form torsions)
            from torusfold.scheme2.aform_from_template import reconstruct_all_atom
            from torusfold.scheme2.amber_refine import amber_refine as _amber_refine_full

            # CG P coords (Å) -> AllAtomStructure (1EHZ crystal template)
            _structure_5 = reconstruct_all_atom(best_coords, sequence, pairs=pairs)
            # amber_refine: C1'-C1' pair restraints (K=100 kJ/mol/nm²) + A-form torsions
            _refined_coords_5, _e0_5, _e1_5, _info_5 = _amber_refine_full(
                _structure_5, pairs,
                platform_name="CPU",
                max_iterations=3000,
            )

            # extract P-only coordinates from the heavy-atom output (located via residue_atom_spans)
            _p_coords_5 = []
            for _ri in range(L):
                _span = _structure_5.residue_atom_spans[_ri]
                _p_idx = _span[0]  # the first atom of each residue is P
                if _p_idx < len(_refined_coords_5):
                    _p_coords_5.append(_refined_coords_5[_p_idx])
            _p_coords_5 = np.array(_p_coords_5) if _p_coords_5 else np.zeros((0, 3))
            # write the PDB for downstream steps such as PPR (Level 5.5)
            _write_coords_pdb(
                _p_coords_5 if len(_p_coords_5) == L else best_coords,
                sequence, str(output_path / "level5_amber.pdb"))

            old_energy = best_energy
            if _e1_5 < best_energy:
                best_energy = _e1_5
                if len(_p_coords_5) == L:
                    best_coords = _p_coords_5
                if verbose:
                    print(f"    AMBER refinement: E={_e0_5:.0f} -> {_e1_5:.0f} kJ/mol (better than previous {old_energy:.0f})")
                    print(f"    pair restraints: {len(pairs)} pairs, A-form torsions: {_info_5.get('n_torsions', 0)}")
            else:
                if verbose:
                    print(f"    AMBER refinement: E={_e1_5:.0f} (not better than {old_energy:.0f}, keeping the original result)")
            _l5_ok = True
        except Exception as e_full:
            if verbose:
                print(f"    amber_refine failed ({e_full!r}), trying openmm_amber_refine fallback...")
            try:
                # Fallback: openmm_amber_refine (simplified, no pair restraints)
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
                            print(f"    AMBER refinement (fallback): E={amber_e:.0f} (better than previous {old_energy:.0f})")
                    else:
                        if verbose:
                            print(f"    AMBER refinement (fallback): E={amber_e:.0f} (not better than {old_energy:.0f}, keeping)")
                    _l5_ok = True
                else:
                    if verbose:
                        print(f"    AMBER refinement skipped (OpenMM not installed)")
            except Exception as e:
                if verbose:
                    print(f"    Level 5 fallback also failed: {e}")
    if _l5_ok:
        _save_ckpt(5,
            pairs=pairs, far_pairs=far_pairs, stem_blocks=stem_blocks,
            coords_vfold=coords_vfold, n_segments=n_segments,
            segments=segments,
            best_coords=best_coords, best_energy=best_energy,
        )

    # ── Level 5.5: PPR base-pair hydrogen-bond repair ──
    if use_ppr:
        if ckpt_level >= 5.5:
            if verbose:
                print(f"\n[Level 5.5] skipping from checkpoint (already repaired)")
        else:
            _level5_pdb = str(output_path / "level5_amber.pdb")
            if not os.path.exists(_level5_pdb):
                _level5_pdb = str(output_path / "level5_aa.pdb")
            if os.path.exists(_level5_pdb):
                if verbose:
                    print(f"\n[Level 5.5] PPR base-pair hydrogen-bond repair...")
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
                            print(f"  PPR effective: {ppr_result['before']} -> {ppr_result['after']} pairs")
                except Exception as e:
                    if verbose:
                        print(f"  PPR failed: {e}")
                _save_ckpt(5.5,
                    ppr_output=_ppr_out if os.path.exists(_ppr_out) else "",
                )
            elif verbose:
                print(f"  PPR skipped: Level 5 PDB does not exist")

    # write the final PDB
    final_pdb = str(output_path / "isrnaclong_final.pdb")
    _write_coords_pdb(best_coords, sequence, final_pdb)

    # read the all-atom P coordinates (if final_allatom.pdb exists)
    _faa = str(output_path / "final_allatom.pdb")
    coords_aa = _read_pdb_p_coords(_faa) if os.path.exists(_faa) else best_coords

    runtime = time.time() - t0
    if verbose:
        print(f"\n=== done: {runtime:.1f}s ===")

    # all-atom coordinates: read from final_allatom.pdb (Level 2.5 output)
    _faa_path = str(output_path / "final_allatom.pdb")
    coords_aa = _read_pdb_p_coords(_faa_path) if os.path.exists(_faa_path) else best_coords

    # ── final statistics: real H-bond rate (all-atom level) ──
    _hbond_rate = 0.0
    _faa_check = str(output_path / "final_allatom.pdb")
    if not os.path.exists(_faa_check):
        _faa_check = str(output_path / "level5_amber.pdb")
    if os.path.exists(_faa_check):
        try:
            _hbond_rate = _compute_hbond_rate(_faa_check, pairs, sequence)
            if verbose:
                print(f"\n  real H-bond rate: {_hbond_rate*100:.1f}% ({_hbond_rate:.4f})")
                print(f"  CG pair_rate (P-P<12A): {state.pair_rate*100:.1f}%")
        except Exception as _err:
            raise

    # ── keep the checkpoints ──
    # _cleanup_checkpoints is disabled: the checkpoint files (_checkpoint.json, ckpt_*.npy)
    # are used for later analysis (energy trajectories, REMD convergence curves,
    # best_coords traceback, etc.)

    # ── final data export ──
    try:
        from torusfold.scheme2.data_exporter import export_final_summary
        export_final_summary(best_coords, sequence, str(output_path))
    except Exception as _err:
        raise

    # ── full pipeline summary ──
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
    """Return the MD step count for a fidelity level (3-level version)."""
    steps = {
        "CG_FAST": 500,        # ~1ps, fast exploration
        "CG_MEDIUM": 5000,     # ~10ps, medium accuracy
        "CG_REST2": 50000,     # ~100ps, REST2 enhanced sampling
        # legacy compatibility
        "CG_SHORT": 500,
        "REST2": 50000,
    }
    return steps.get(level.name, 5000)


def _estimate_energy(coords, pairs, sequence) -> float:
    """Simple energy estimate (used when LAMMPS is unavailable). coords are P-only in Å."""
    try:
        from torusfold.scheme2.refine import BOND_LEN
    except ImportError:
        BOND_LEN = 5.9
    energy = 0.0
    L = len(coords)

    # backbone bonds (BOND_LEN in Å, coords in Å)
    for i in range(L - 1):
        d = np.linalg.norm(coords[i] - coords[i + 1])
        energy += 0.5 * 31000.0 * (d - BOND_LEN) ** 2

    # pairs (accepts both (i,j) and (i,j,w) formats; target ~10.5Å WC distance)
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
    """Check the cross-segment pairing distance."""
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
    """Find which segment a residue index belongs to."""
    for idx, seg in enumerate(segments):
        if seg["start"] <= res_idx < seg["end"]:
            return idx
    return -1


def _compute_pair_rate(coords, pairs) -> float:
    """Compute the base-pairing satisfaction rate (P-P distance < 12Å, stricter than the old 15Å)."""
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
        if d < 12.0:  # the old 15Å was too loose; 12Å is more reasonable
            ok += 1
    return ok / len(pairs)


def _compute_hbond_rate(pdb_path, pairs, sequence) -> float:
    """Compute the real hydrogen-bond satisfaction rate (N1/N3/O6/N4 distance < 3.6Å)."""
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
    """Quickly validate structure quality (clash + pair_rate + bond_quality).

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

    # 1. Clash detection: P-P distance < 3.0A (excluding adjacent residues)
    from scipy.spatial.distance import cdist
    dist_mat = cdist(coords, coords)
    for i in range(L):
        for j in range(i + 3, L):  # skip adjacent residues
            if dist_mat[i, j] < 3.0:
                result["clash_count"] += 1

    # 2. pairing satisfaction: fraction of pairs closer than 15A
    if pairs:
        n_ok = 0
        for p in pairs:
            i, j = (p[0], p[1]) if len(p) >= 2 else (p[0], p[1])
            if i < L and j < L and dist_mat[i, j] < 15.0:
                n_ok += 1
        result["pair_rate"] = n_ok / len(pairs)

    # 3. bond-length quality: adjacent P-P distances
    diffs = np.diff(coords, axis=0)
    bond_dists = np.linalg.norm(diffs, axis=1)
    mean_bond = np.mean(bond_dists)
    result["bond_quality"] = max(0.0, 1.0 - abs(mean_bond - 5.9) / 5.9)

    # 4. verdict
    if result["clash_count"] > 10:
        result["is_valid"] = False
        result["warnings"].append(f"clash={result['clash_count']}")
    if result["pair_rate"] < 0.1 and pairs:
        result["warnings"].append(f"pair_rate={result['pair_rate']:.2f}")
    if result["bond_quality"] < 0.3:
        result["warnings"].append(f"bond_q={result['bond_quality']:.2f}")

    return result


def _compute_clash_count(coords, threshold: float = 3.0) -> int:
    """Count P-P clashes (distance < threshold Å)."""
    L = len(coords)
    count = 0
    for i in range(L):
        for j in range(i + 2, min(i + 20, L)):  # local check to avoid O(n²)
            d = np.linalg.norm(coords[i] - coords[j])
            if d < threshold:
                count += 1
    return count


def _compute_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the RMSD between two coordinate sets."""
    if a.shape != b.shape:
        return float("inf")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def _cluster_and_select(coords_list, energies, rmsd_threshold=5.0):
    """Cluster and select the best conformation.

    Algorithm: greedy clustering (RMSD < threshold merges into one cluster), then pick
    the lowest-energy representative.

    Args:
        coords_list: List[(L, 3)] of CG coordinate sets
        energies: List[float] of the corresponding energies
        rmsd_threshold: RMSD threshold for clustering (Å)

    Returns:
        best_idx: index of the best conformation
        n_clusters: number of clusters
        cluster_info: [(center_idx, member_count, min_energy), ...]
    """
    if not coords_list:
        return 0, 0, []
    if len(coords_list) == 1:
        return 0, 1, [(0, 1, energies[0])]

    # greedy clustering
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

    # pick the lowest-energy representative from each cluster
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
    """Compute the 4-metric relaxation monitor."""
    if len(coords) == 0:
        # failed to read coordinates; return empty metrics
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
    """Update the cross-segment pair weights (extended version with a clash penalty)."""
    new_weights = old_weights.copy()
    L = len(coords)
    for p in far_pairs:
        i, j = p[0], p[1]
        if i >= L or j >= L:
            continue
        d = np.linalg.norm(coords[i] - coords[j])
        if d > 15.0:
            # too far -> increase the weight
            new_weights[(i, j)] = min(old_weights.get((i, j), 1.0) * 1.2, 5.0)
        elif d < 5.0:
            # too close -> decrease the weight
            new_weights[(i, j)] = max(old_weights.get((i, j), 1.0) * 0.8, 0.1)

    # global clash penalty: lower every weight when there are clashes
    if metrics is not None and metrics.clash_count > 0:
        for key in new_weights:
            new_weights[key] = max(new_weights[key] * 0.7, 0.1)

    return new_weights


def _default_helix_coords(L):
    """Default A-form helix coordinates."""
    import math
    coords = np.zeros((L, 3))
    for i in range(L):
        z = i * 2.8
        angle = i * 33.0 * math.pi / 180
        coords[i] = [4.4 * math.cos(angle), 4.4 * math.sin(angle), z]
    return coords


def _read_pdb_p_coords(pdb_path: str) -> np.ndarray:
    """Read P-atom coordinates from a PDB file, returning (N, 3)."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and " P " in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        # fallback: IsRNAcirc outputs an all-atom PDB with no P-atom tag.
        # read every atom coordinate (not just the first), for the Level 3 RL.
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    return np.array(coords) if coords else np.zeros((0, 3))


def _write_coords_pdb(coords, sequence, output_path):
    """Write coordinates to a PDB. CG_to_allatom.exe requires 3-letter residue names."""
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
    """Merge the segmented all-atom PDBs into one complete all-atom PDB in residue order.

    Copy the original ATOM lines verbatim (preserving PDB column alignment), changing
    only the residue numbering.
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
                # residue number within the segment (read from the original PDB line)
                try:
                    local_res = int(line[22:26].strip())
                except (ValueError, IndexError):
                    local_res = seg_res_count + 1
                global_res = res_offset + local_res
                atom_idx += 1
                # keep the original PDB column alignment; change only atom serial (7-11)
                # and resSeq (22-26)
                new_line = (
                    line[:6]                              # "ATOM  "
                    + f"{atom_idx:5d}"                    # serial 7-11
                    + line[11:22]                         # atom name, altLoc, resName, chainID
                    + f"{global_res:4d}"                  # resSeq 22-26
                    + line[26:]                           # iCode + the rest (coords, occupancy, etc.)
                )
                lines.append(new_line)
                seg_res_count = local_res
        res_offset += seg_res_count

    lines.append("END")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
