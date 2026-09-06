# -*- coding: utf-8 -*-
"""metadynamics_sampler.py — Well-Tempered Metadynamics for circRNA.

关键改进:
1. 偏置势通过 CustomExternalForce 注入 (不修改坐标)
2. CV-specific sigma (BSJ=2nm, nc=0.1, Rg=1nm)
3. hill_height 按系统大小缩放
4. 平滑 nc CV 梯度 (sigmoid switching)
5. well-tempered 自适应 hill 高度

Level 3.5 GPU 迁移 (2026-08-26):
  _add_bias 内循环 (hills × particles × pairs) 用 torch 向量化.
  OpenMM 集成不变 — numpy 写回 CustomExternalForce.
"""
from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np

try:
    import openmm as mm
    from openmm import unit, LangevinMiddleIntegrator, Platform
    from openmm.app import Simulation
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _detect_torch_device():
    """Detect torch device, preferring CUDA/HIP over CPU."""
    if not TORCH_AVAILABLE:
        return None, False
    if torch.cuda.is_available():
        return torch.device("cuda"), True
    return torch.device("cpu"), False


def _t(np_arr, device=None, dtype=torch.float64):
    """numpy -> torch tensor, optionally to device."""
    t = torch.from_numpy(np.asarray(np_arr, dtype=np.float64)).to(dtype)
    if device is not None:
        t = t.to(device)
    return t


# ═══ torch GPU kernel: batched hill + gradient computation ══════════════

def _compute_bias_forces_gpu(
    pos: "torch.Tensor",     # (L, 3) float64, positions in Å
    hills_tensor: "torch.Tensor",  # (H, 3) CV centers
    hill_weights: "torch.Tensor",  # (H,) hill heights
    sigmas: "torch.Tensor",        # (3,) CV sigmas
    # nc sparse gradient precomputation
    nc_p1: "torch.Tensor",   # (P,) particle indices (left)
    nc_p2: "torch.Tensor",   # (P,) particle indices (right)
    nc_nsign: "torch.Tensor", # (P,) +1 or -1 per pair
    nc_w: "torch.Tensor",    # (P,) pair weights
    nc_nfar: "torch.Tensor", # scalar: total number of active far pairs
    L: int,
    device: "torch.device",
) -> "torch.Tensor":
    """Compute accumulated bias forces for ALL hills in one pass.

    Returns (L, 3) force tensor (kJ/mol/Å).
    """
    if hills_tensor.shape[0] == 0:
        return torch.zeros(L, 3, dtype=torch.float64, device=device)

    H = hills_tensor.shape[0]
    s1, s2, s3 = sigmas[0], sigmas[1], sigmas[2]
    s1_sq, s2_sq, s3_sq = s1 * s1, s2 * s2, s3 * s3

    # ── 1. CV values for current conformation ──
    # BSJ distance (Å)
    diff_bsj = pos[0] - pos[-1]
    dist_bsj = torch.clamp(diff_bsj.norm(), min=1e-8)

    # Radius of gyration (Å)
    center = pos.mean(dim=0)
    rg_sq = ((pos - center) ** 2).sum(dim=1).mean()
    rg = torch.clamp(rg_sq.sqrt(), min=1e-8)

    # NC contact fraction (vectorized)
    nc_val = torch.tensor(0.0, dtype=torch.float64, device=device)
    if nc_p1.numel() > 0:
        nc_diff = pos[nc_p1] - pos[nc_p2]
        nc_dist = torch.clamp(nc_diff.norm(dim=1), min=1e-8)
        nc_d_nm = nc_dist / 10.0
        # Only compute sigmoid where d < 6nm (otherwise saturated to 0)
        nc_mask = nc_d_nm < 6.0
        if nc_mask.any():
            safe_d = nc_d_nm[nc_mask]
            exp_arg = torch.clamp((safe_d - 1.0) / 0.5 * 10, max=50.0)
            # Sparse array stores both (i,j) and (j,i) for each pair, so the
            # sigmoid sum counts each pair twice. Divide by 2*n_far to get the
            # same per-pair average as the original _cv_nc.
            nc_val = (torch.sigmoid(-exp_arg) * nc_w[nc_mask]).sum() / (nc_nfar * 2)

    # Stack CVs: (3,)
    cv_current = torch.stack([dist_bsj / 10.0, nc_val, rg / 10.0])

    # ── 2. Hill evaluation (vectorized over H) ──
    arg = 0.5 * ((cv_current - hills_tensor) / sigmas) ** 2  # (H, 3)
    arg_sum = arg.sum(dim=1)  # (H,)
    arg_clamped = torch.clamp(arg_sum, max=50.0)
    hill_vals = hill_weights * torch.exp(-arg_clamped)  # (H,)
    # Skip negligible hills
    active = hill_vals > 0.01
    if not active.any():
        return torch.zeros(L, 3, dtype=torch.float64, device=device)
    hill_vals = hill_vals[active]  # (H',)
    hills_act = hills_tensor[active]  # (H', 3)
    H_act = hill_vals.shape[0]

    # ── 3. dCV/dc for all hills (H', 3) ──
    dc1 = (cv_current[0] - hills_act[:, 0]) / s1_sq  # (H',)
    dc2 = (cv_current[1] - hills_act[:, 1]) / s2_sq  # (H',)
    dc3 = (cv_current[2] - hills_act[:, 2]) / s3_sq  # (H',)

    # ── 4. Particle-level gradients (L, 3) ──
    bias = torch.zeros(L, 3, dtype=torch.float64, device=device)

    # --- 4a. BSJ gradient: only particles 0 and L-1 ---
    bsj_dir = diff_bsj / dist_bsj  # unit vector (3,)
    # grad BSJ w.r.t. particle 0: -bsj_dir/10
    # grad BSJ w.r.t. particle L-1: +bsj_dir/10
    # force contribution = Σ_hill hill_val * dc1 * (±bsj_dir/10)
    sum_dc1_w = (dc1 * hill_vals).sum()  # scalar
    bias[0] += (-sum_dc1_w / 10.0) * bsj_dir
    bias[L - 1] += (sum_dc1_w / 10.0) * bsj_dir

    # --- 4b. Rg gradient: each particle ---
    # grad Rg w.r.t. particle i = (p_i - center) / (rg * L)
    # (in Å units, CV is rg/10, so divide by 10 more)
    rg_diff = pos - center  # (L, 3)
    sum_dc3_w = (dc3 * hill_vals).sum()  # scalar
    bias += (sum_dc3_w / (rg * L * 10.0)) * rg_diff  # (L, 3)

    # --- 4c. NC gradient: scatter via precomputed sparse indices ---
    if nc_p1.numel() > 0:
        # Compute sigmoid derivatives for current conformation
        nc_diff = pos[nc_p1] - pos[nc_p2]  # (P, 3)
        nc_dist = torch.clamp(nc_diff.norm(dim=1), min=1e-8)
        nc_d_nm = nc_dist / 10.0
        nc_active = nc_d_nm < 6.0
        if nc_active.any():
            idx = torch.where(nc_active)[0]
            safe_d = nc_d_nm[idx]
            exp_arg = torch.clamp((safe_d - 1.0) / 0.5 * 10, max=50.0)
            exp_val = torch.exp(exp_arg)
            s = 1.0 / (1.0 + exp_val)
            ds = -s * s * exp_val * (10.0 / 0.5)  # ds/dd_nm

            # Pair distance gradient: -d_ij/dist_ij (matches original _add_bias)
            pair_dist_grad = -nc_diff[idx] / nc_dist[idx].unsqueeze(1)  # (P', 3)

            # Per-pair force contribution: w * sign * ds * pair_dist_grad
            per_pair = (nc_w[idx] * nc_nsign[idx] * ds).unsqueeze(1) * pair_dist_grad  # (P', 3)

            # Hill-weighted sum: Σ_hill (dc2 * hill_val) * per_pair / n_far
            sum_dc2_w = (dc2 * hill_vals).sum()  # scalar
            nc_force_contrib = (sum_dc2_w / nc_nfar.float()) * per_pair  # (P', 3)

            # Accumulate to particles: scatter_add_
            flat_idx = nc_p1[idx].long()
            bias.scatter_add_(0, flat_idx.unsqueeze(1).expand_as(nc_force_contrib), nc_force_contrib)
            flat_idx2 = nc_p2[idx].long()
            bias.scatter_add_(0, flat_idx2.unsqueeze(1).expand_as(nc_force_contrib), nc_force_contrib)

    # ── 5. Clamp per-particle force magnitude ──
    force_norms = bias.norm(dim=1)  # (L,)
    scale = torch.clamp(force_norms / 500.0, min=1.0)
    bias = bias / scale.unsqueeze(1)

    return bias


def _compute_bias_forces_cpu(
    pos_np, hills_list, L, sigmas_np, nc_p1_np, nc_p2_np, nc_nsign_np, nc_w_np, n_far
):
    """NumPy fallback when torch unavailable."""
    if not hills_list:
        return np.zeros((L, 3), dtype=np.float64)

    s1, s2, s3 = sigmas_np
    s1_sq, s2_sq, s3_sq = s1*s1, s2*s2, s3*s3

    diff_bsj = pos_np[0] - pos_np[-1]
    dist_bsj = max(1e-8, np.linalg.norm(diff_bsj))

    center = np.mean(pos_np, axis=0)
    rg = max(1e-8, np.sqrt(np.mean(np.sum((pos_np - center)**2, axis=1))))

    # NC contact fraction
    nc_val = 0.0
    if len(nc_p1_np) > 0:
        nc_diffs = pos_np[nc_p1_np] - pos_np[nc_p2_np]
        nc_dists = np.maximum(np.linalg.norm(nc_diffs, axis=1), 1e-8)
        nc_d_nm = nc_dists / 10.0
        nc_mask = nc_d_nm < 6.0
        if nc_mask.any():
            exp_arg = np.clip((nc_d_nm[nc_mask] - 1.0) / 0.5 * 10, None, 50.0)
            # Divide by 2*n_far (sparse array has both directions of each pair)
            nc_val = float(np.sum(1.0/(1.0+np.exp(exp_arg)) * nc_w_np[nc_mask])) / max(1, n_far * 2)

    cv = np.array([dist_bsj/10.0, nc_val, rg/10.0])
    bias = np.zeros((L, 3), dtype=np.float64)

    for h1, h2, h3, w in hills_list:
        arg = 0.5*((cv[0]-h1)/s1)**2 + 0.5*((cv[1]-h2)/s2)**2 + 0.5*((cv[2]-h3)/s3)**2
        hill_val = w * math.exp(-min(arg, 50))
        if hill_val < 0.01:
            continue
        dc1 = (cv[0] - h1) / s1_sq
        dc2 = (cv[1] - h2) / s2_sq
        dc3 = (cv[2] - h3) / s3_sq

        grad = np.zeros((L, 3), dtype=np.float64)
        # BSJ
        grad[0] += dc1 * (-diff_bsj/dist_bsj)
        grad[L-1] += dc1 * (diff_bsj/dist_bsj)
        # Rg
        grad += dc3 * (pos_np - center) / (rg * L)

        # NC (divided by n_far to match original normalization)
        if len(nc_p1_np) > 0:
            nc_d = pos_np[nc_p1_np] - pos_np[nc_p2_np]
            nc_dist = np.maximum(np.linalg.norm(nc_d, axis=1), 1e-8)
            nc_d_nm = nc_dist / 10.0
            nc_act = nc_d_nm < 6.0
            if np.any(nc_act):
                ai = np.where(nc_act)[0]
                ea = np.clip((nc_d_nm[ai]-1.0)/0.5*10, None, 50.0)
                ev = np.exp(ea)
                s = 1.0/(1.0+ev)
                ds = -s*s*ev*20.0
                pair_g = -nc_d[ai] / nc_dist[ai, np.newaxis]  # -d_ij/dist_ij (matches original)
                contrib = (nc_w_np[ai] * nc_nsign_np[ai] * ds)[:, np.newaxis] * pair_g
                for k, pi in enumerate(nc_p1_np[ai]):
                    grad[pi] += dc2 * contrib[k] / n_far
                for k, pi in enumerate(nc_p2_np[ai]):
                    grad[pi] += dc2 * contrib[k] / n_far

        fv = grad * hill_val
        fn = np.linalg.norm(fv, axis=1, keepdims=True)
        fv = np.where(fn > 500, fv * 500/fn, fv)
        bias += fv

    return bias


class MetaDynamicsSampler:
    """Well-Tempered Metadynamics.

    每 hill_freq 步:
      1. 读取 P 坐标, 计算3个 CV
      2. 添加 Gaussian hill (存入列表)
      3. 移除旧偏置力, 重建所有 hill 的偏置力
      4. reinitialize (保持坐标+速度)
    """

    def __init__(
        self,
        sequence: str,
        pairs,
        hill_height: float = 100.0,
        hill_sigma_bsj: float = 2.0,
        hill_sigma_nc: float = 0.1,
        hill_sigma_rg: float = 1.0,
        hill_freq: int = 10,
        max_hills: int = 2000,
        well_tempered: bool = True,
        bias_factor: float = 3.0,
        platform_name: str = "CPU",
    ):
        self.sequence = sequence
        self.pairs = list(pairs)
        self.L = len(sequence)
        self.hill_sigma_bsj = hill_sigma_bsj
        self.hill_sigma_nc = hill_sigma_nc
        self.hill_sigma_rg = hill_sigma_rg
        L_scale = max(1.0, self.L / 100.0)
        self.hill_height = hill_height * L_scale
        self.hill_freq = hill_freq
        self.max_hills = max_hills
        self.well_tempered = well_tempered
        self.bias_factor = bias_factor
        self.platform_name = platform_name

        # Torch state (initialized on first use in _add_bias)
        self._torch_device = None
        self._torch_ok = False
        self._hills_tensor = None
        self._hill_weights = None
        self._nc_p1 = None  # precomputed nc sparse indices

    def _init_torch_cache(self):
        """Lazily init torch tensors for GPU-accelerated bias."""
        if not TORCH_AVAILABLE:
            return
        device, ok = _detect_torch_device()
        self._torch_device = device
        self._torch_ok = ok

        # Precompute nc sparse pair indices (once)
        # n_far: total number of far pairs (abs(i-j) >= 5), used to normalize
        # nc gradient — matches original _add_bias's /n_far factor.
        nc_p1, nc_p2, nc_nsign, nc_w = [], [], [], []
        n_far = 0
        for i, j, w in self.pairs:
            if abs(i - j) < 5:
                continue
            n_far += 1
            nc_p1.append(i)
            nc_p2.append(j)
            nc_nsign.append(1.0)
            nc_w.append(w)
            nc_p1.append(j)
            nc_p2.append(i)
            nc_nsign.append(-1.0)
            nc_w.append(w)
        self._nc_p1 = torch.tensor(nc_p1, dtype=torch.long, device=device)
        self._nc_p2 = torch.tensor(nc_p2, dtype=torch.long, device=device)
        self._nc_nsign = torch.tensor(nc_nsign, dtype=torch.float64, device=device)
        self._nc_w = torch.tensor(nc_w, dtype=torch.float64, device=device)
        self._nc_nfar = torch.tensor(max(1, n_far), dtype=torch.float64, device=device)
        self._sigmas = torch.tensor(
            [self.hill_sigma_bsj, self.hill_sigma_nc, self.hill_sigma_rg],
            dtype=torch.float64, device=device,
        )

        # Hills accumulator (will grow dynamically)
        self._hills_tensor = torch.zeros(0, 3, dtype=torch.float64, device=device)
        self._hill_weights = torch.zeros(0, dtype=torch.float64, device=device)

    def sample(self, p_init, n_steps=50000, verbose=True):
        """Run Metadynamics. Returns (best_coords_A, best_energy_kJ)."""
        from .openmm_gpu_refiner import (
            _build_3bead_system_gpu, _create_3bead_topology,
        )

        system, coords_nm, *_ = _build_3bead_system_gpu(
            p_init, self.pairs, pair_scale=1.0, bsj_k_scale=1.0)
        P_idx = [3 * i for i in range(self.L)]
        topo = _create_3bead_topology(self.L)

        # ── 单个累积 bias force ──
        _bias_force_obj = mm.CustomExternalForce("fx*x + fy*y + fz*z")
        _bias_force_obj.addPerParticleParameter("fx")
        _bias_force_obj.addPerParticleParameter("fy")
        _bias_force_obj.addPerParticleParameter("fz")
        for i in range(self.L):
            _bias_force_obj.addParticle(P_idx[i], [0.0, 0.0, 0.0])
        system.addForce(_bias_force_obj)

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        try:
            plat = Platform.getPlatformByName(self.platform_name)
        except Exception:
            plat = Platform.getPlatformByName("CPU")

        sim = Simulation(topo, system, integrator, plat)
        sim.context.setPositions(coords_nm * unit.nanometer)
        sim.minimizeEnergy(maxIterations=2000)

        # Init torch cache (nc sparse indices, sigmas)
        self._init_torch_cache()

        hills = []
        cv_history = []
        best_energy = float("inf")
        best_pos = p_init.copy()

        def _add_bias(sim, hills_list, p_coords_A):
            """Recompute accumulated bias force for all hills."""
            _bias_acc = self._compute_bias_forces(p_coords_A, hills_list)
            for i in range(self.L):
                fx, fy, fz = _bias_acc[i]
                _bias_force_obj.setParticleParameters(
                    i, P_idx[i], [float(fx), float(fy), float(fz)])
            _bias_force_obj.updateParametersInContext(sim.context)

        for step in range(0, n_steps, self.hill_freq):
            sim.step(self.hill_freq)
            state = sim.context.getState(getPositions=True, getEnergy=True)
            pos = state.getPositions(asNumpy=True)._value
            energy = state.getPotentialEnergy()._value
            p_A = pos[P_idx].copy() * 10.0

            cv1, cv2, cv3 = self._compute_cvs(p_A)

            if self.well_tempered and cv_history:
                n_visits = sum(1 for (h1, h2, h3) in cv_history
                               if abs(h1 - cv1) < self.hill_sigma_bsj * 2
                               and abs(h2 - cv2) < self.hill_sigma_nc * 2)
                height = self.hill_height / (1 + n_visits / self.bias_factor)
            else:
                height = self.hill_height

            if len(hills) < self.max_hills and height > 0.1:
                hills.append((cv1, cv2, cv3, height))
                cv_history.append((cv1, cv2, cv3))
                _add_bias(sim, hills, p_A)

            if energy < best_energy:
                best_energy = energy
                best_pos = p_A.copy()

            if verbose and (step // self.hill_freq) % 50 == 0:
                print(f"    MetaD {step}/{n_steps}: "
                      f"BSJ={cv1:.2f}nm nc={cv2:.2f} Rg={cv3:.2f}nm "
                      f"E={energy:.0f} hills={len(hills)}")

        # Final minimization: 清零 bias 力
        for i in range(self.L):
            _bias_force_obj.setParticleParameters(
                i, P_idx[i], [0.0, 0.0, 0.0])
        _bias_force_obj.updateParametersInContext(sim.context)
        sim.minimizeEnergy(maxIterations=5000)
        state = sim.context.getState(getPositions=True, getEnergy=True)
        ef = state.getPotentialEnergy()._value
        pf = state.getPositions(asNumpy=True)._value[P_idx].copy() * 10.0
        if ef < best_energy:
            best_energy = ef
            best_pos = pf

        if verbose:
            print(f"    MetaD 完成: {len(hills)} hills, best E={best_energy:.0f}")
        return best_pos, best_energy

    def _compute_bias_forces(self, p_coords_A, hills_list):
        """Dispatch to torch GPU kernel or numpy CPU fallback.

        GPU 加速只在 hills×L 足够大时有效 (小系统传输开销 > 计算收益).
        阈值: hills × L > 5000 时走 GPU.
        """
        n_hills = len(hills_list)
        # GPU 传输开销 ~1-2ms, 只在计算量足够大时才值得
        # 阈值估算: L=64 需 >500 hills, L=2013 需 >20 hills
        use_gpu = (self._torch_ok and TORCH_AVAILABLE
                   and self._nc_p1 is not None
                   and n_hills * self.L > 20000)
        if use_gpu:
            # GPU path
            hills_np = np.array(hills_list, dtype=np.float64)
            if len(hills_np) > 0:
                self._hills_tensor = torch.from_numpy(hills_np[:, :3]).to(
                    device=self._torch_device, dtype=torch.float64)
                self._hill_weights = torch.from_numpy(hills_np[:, 3]).to(
                    device=self._torch_device, dtype=torch.float64)
            else:
                self._hills_tensor = torch.zeros(0, 3, dtype=torch.float64,
                                                  device=self._torch_device)
                self._hill_weights = torch.zeros(0, dtype=torch.float64,
                                                  device=self._torch_device)

            pos_t = torch.from_numpy(p_coords_A).to(
                dtype=torch.float64, device=self._torch_device)
            bias_t = _compute_bias_forces_gpu(
                pos_t, self._hills_tensor, self._hill_weights,
                self._sigmas,
                self._nc_p1, self._nc_p2, self._nc_nsign, self._nc_w,
                self._nc_nfar, self.L, self._torch_device,
            )
            return bias_t.cpu().numpy()
        else:
            # CPU numpy fallback (tensors may be on GPU — .cpu() first)
            def _to_numpy(t):
                if t is None:
                    return None
                return t.detach().cpu().numpy()
            nc_p1 = _to_numpy(self._nc_p1) if self._nc_p1 is not None else np.array([], dtype=int)
            nc_p2 = _to_numpy(self._nc_p2) if self._nc_p2 is not None else np.array([], dtype=int)
            nc_nsign = _to_numpy(self._nc_nsign) if self._nc_nsign is not None else np.array([], dtype=np.float64)
            nc_w = _to_numpy(self._nc_w) if self._nc_w is not None else np.array([], dtype=np.float64)
            n_far_val = float(_to_numpy(self._nc_nfar)) if self._nc_nfar is not None else 1.0
            sigmas = np.array([self.hill_sigma_bsj, self.hill_sigma_nc, self.hill_sigma_rg],
                              dtype=np.float64)
            return _compute_bias_forces_cpu(
                p_coords_A, hills_list, self.L, sigmas,
                nc_p1, nc_p2, nc_nsign, nc_w, n_far_val,
            )

    def _compute_cvs(self, p_coords_A):
        cv1 = float(np.linalg.norm(p_coords_A[0] - p_coords_A[-1])) / 10.0
        cv2 = self._cv_nc(p_coords_A)
        center = np.mean(p_coords_A, axis=0)
        cv3 = np.sqrt(np.mean(np.sum((p_coords_A - center) ** 2, axis=1))) / 10.0
        return cv1, cv2, cv3

    def _cv_nc(self, p_coords_A):
        if not self.pairs:
            return 0.0
        cnt = 0.0
        tot = 0
        for i, j, w in self.pairs:
            if abs(i - j) < 5:
                continue
            tot += 1
            d = np.linalg.norm(p_coords_A[i] - p_coords_A[j]) / 10.0
            if d > 6.0:
                continue
            exp_arg = min((d - 1.0) / 0.5 * 10, 50.0)
            s = 1.0 / (1.0 + math.exp(exp_arg))
            cnt += s
        return cnt / max(1, tot)


# ══════════════════════════════════════════════════════════════════════
#  冒烟测试 (不需要 OpenMM)
# ══════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Test CV computation and torch vectorized hill force (no OpenMM)."""
    import time

    print("=" * 60)
    print("MetaDynamicsSampler GPU smoke test")
    print("=" * 60)

    # --- 1. Check torch availability ---
    print(f"\n[1] TORCH_AVAILABLE = {TORCH_AVAILABLE}")
    if TORCH_AVAILABLE:
        device, ok = _detect_torch_device()
        print(f"    device = {device}, GPU = {ok}")
    else:
        print("    WARNING: torch not available, CPU-only mode")

    # --- 2. Test CV computation ---
    L = 200  # small test
    seq = "G" * L
    # Random pair list: some near (filtered), some far (active)
    pairs = []
    for i in range(0, L - 10, 3):
        j = min(i + 10 + (i % 7) * 3, L - 1)
        pairs.append((i, j, 1.0))

    sampler = MetaDynamicsSampler(seq, pairs)
    rng = np.random.RandomState(42)
    p_A = rng.randn(L, 3).astype(np.float64) * 5.0

    t0 = time.time()
    cv1, cv2, cv3 = sampler._compute_cvs(p_A)
    t_cv = time.time() - t0
    print(f"\n[2] CV computation (L={L}):")
    print(f"    BSJ = {cv1:.4f} nm")
    print(f"    nc  = {cv2:.4f}")
    print(f"    Rg  = {cv3:.4f} nm")
    print(f"    time = {t_cv*1000:.2f} ms")

    # --- 3. Test GPU bias force ---
    sampler._init_torch_cache()
    H_test = 50
    test_hills = [(float(rng.randn()), float(rng.randn()),
                   float(rng.randn()), 100.0) for _ in range(H_test)]

    t0 = time.time()
    bias_np = sampler._compute_bias_forces(p_A, test_hills)
    t_bias = time.time() - t0
    print(f"\n[3] Bias force (H={H_test}, L={L}):")
    print(f"    bias shape = {bias_np.shape}")
    print(f"    |bias|_max = {np.max(np.abs(bias_np)):.4f} kJ/mol/A")
    print(f"    |bias|_norm per particle = "
          f"{np.mean(np.linalg.norm(bias_np, axis=1)):.4f}")
    print(f"    time = {t_bias*1000:.2f} ms")

    # --- 4. Test with many hills (scaling) ---
    for H_test in [10, 50, 100, 200]:
        test_hills = [(float(rng.randn()), float(rng.randn()),
                       float(rng.randn()), 100.0) for _ in range(H_test)]
        t0 = time.time()
        _ = sampler._compute_bias_forces(p_A, test_hills)
        t_bias = time.time() - t0
        print(f"    H={H_test:>3d}: {t_bias*1000:>8.2f} ms")

    # --- 5. Test with larger system ---
    L_big = 2000
    seq_big = "G" * L_big
    pairs_big = [(i, min(i + 15, L_big - 1), 1.0)
                 for i in range(0, L_big - 20, 5)]
    sampler_big = MetaDynamicsSampler(seq_big, pairs_big)
    sampler_big._init_torch_cache()
    p_big = rng.randn(L_big, 3).astype(np.float64) * 5.0

    for H_test in [50, 200]:
        test_hills = [(float(rng.randn()), float(rng.randn()),
                       float(rng.randn()), 100.0) for _ in range(H_test)]
        t0 = time.time()
        _ = sampler_big._compute_bias_forces(p_big, test_hills)
        t_bias = time.time() - t0
        print(f"\n[4] L={L_big}, H={H_test}: {t_bias*1000:.2f} ms")

    # --- 6. Verify GPU path sanity ---
    if TORCH_AVAILABLE:
        sampler_small = MetaDynamicsSampler("G" * 50,
            [(i, min(i+10, 49), 1.0) for i in range(0, 40, 3)])
        sampler_small._init_torch_cache()
        p_small = rng.randn(50, 3).astype(np.float64) * 5.0
        test_hills = [(1.0, 0.5, 0.3, 100.0), (-1.0, 0.2, 0.1, 80.0)]

        sampler_small._torch_ok = True
        bias_gpu = sampler_small._compute_bias_forces(p_small, test_hills)

        # Verify: forces should be finite and non-trivial
        all_finite = np.all(np.isfinite(bias_gpu))
        non_zero = np.any(np.abs(bias_gpu) > 1e-6)
        reasonable = np.max(np.abs(bias_gpu)) < 1e6
        print(f"\n[5] GPU path sanity (L=50, H=2):")
        print(f"    all finite: {all_finite}")
        print(f"    non-zero:   {non_zero}")
        print(f"    reasonable: {reasonable} (|F|_max={np.max(np.abs(bias_gpu)):.2f})")
        if all_finite and non_zero and reasonable:
            print("    PASS")
        else:
            print("    FAIL")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
