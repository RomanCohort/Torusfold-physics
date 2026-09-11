# -*- coding: utf-8 -*-
"""metadynamics_gpu.py — GPU Batched Well-Tempered Metadynamics for circRNA.

Level 3.5 GPU migration: replace OpenMM CPU metadynamics with PyTorch GPU
batched version. All n_replicas copies run on GPU simultaneously as a single
(B, 3L, 3) tensor.

CV = Radius of Gyration (Rg):
  Rg = sqrt( mean( ||p_i - center||^2 ) ) / 10.0  [nm]

Bias potential: sum of Gaussian hills in CV space
  V(Rg) = sum_h w_h * exp( -(Rg - Rg_h)^2 / (2 * sigma^2) )

Force on particle i from hill h:
  F_i = -dV/dx_i = w_h * (Rg - Rg_h) / (sigma^2 * Rg) * (x_i - center) / L

Well-tempered: delta = w / bias_factor
  hill weight w_h = w_h * gamma / (1 + n_visits / bias_factor)

Uses cg_energy_forces from torch_cgsim.py for unified energy+forces.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

if TORCH_OK:
    from .torch_cgsim import (
        cg_energy_forces, batch_langevin_step, GPUCellList,
        KB_KJ, _safe_zeros, _arange_dev,
    )


def _compute_rg(pos_nm: "torch.Tensor") -> "torch.Tensor":
    """Radius of gyration in nm. pos_nm: (B, N, 3). Returns (B,)."""
    center = pos_nm.mean(dim=1, keepdim=True)  # (B, 1, 3)
    rg_sq = ((pos_nm - center) ** 2).sum(dim=-1).mean(dim=1)  # (B,)
    return rg_sq.sqrt().clamp(min=1e-8)


def _compute_rg_grad(pos_nm: "torch.Tensor") -> "torch.Tensor":
    """dRg/dx_i for all particles. Returns (B, N, 3).

    Rg = sqrt( mean( ||p_i - c||^2 ) ), c = mean(p_i)
    dRg/dx_i = (x_i - c) / (Rg * N)
    """
    B, N, _ = pos_nm.shape
    center = pos_nm.mean(dim=1, keepdim=True)  # (B, 1, 3)
    diff = pos_nm - center  # (B, N, 3)
    rg = _compute_rg(pos_nm).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
    return diff / (rg * N)  # (B, N, 3)


class BatchedMetadynamics:
    """GPU batched well-tempered metadynamics.

    All n_replicas copies run on GPU simultaneously as (B, 3L, 3).
    CV = Radius of Gyration (Rg).

    Args:
        n_replicas: number of parallel replicas
        sequence: RNA sequence (for L = len(sequence))
        device: torch device ("cuda" or "cpu")
    """

    def __init__(
        self,
        n_replicas: int = 8,
        sequence: str = "",
        device: str = "cuda",
        # Relaxation parameters (aligned with OpenMM rest2_remd_2d)
        relax_bond_k: float = 500.0,
        relax_angle_k: float = 200.0,
        relax_pair_k: float = 500.0,
        restraint_k: float = 500.0,
    ):
        assert TORCH_OK, "PyTorch is required for BatchedMetadynamics"
        self.n_replicas = n_replicas
        self.L = len(sequence)
        self.sequence = sequence
        self.device = device if torch.cuda.is_available() else "cpu"
        # Relaxation parameters
        self.relax_bond_k = relax_bond_k
        self.relax_angle_k = relax_angle_k
        self.relax_pair_k = relax_pair_k
        self.restraint_k = restraint_k

    def run(
        self,
        coords_A: np.ndarray,           # (L, 3) initial P coords in Angstrom
        pairs,                           # list of (i, j, w) WC pairs
        n_steps: int = 200000,
        hill_height: float = 1.0,        # kJ/mol (base hill height)
        hill_sigma: float = 1.0,         # nm (width in CV space)
        hill_freq: int = 100,            # steps between hill additions
        max_hills: int = 5000,
        well_tempered: bool = True,
        bias_factor: float = 5.0,
        dt_ps: float = 0.002,
        friction: float = 1.0,
        temperature: float = 300.0,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, float, dict]:
        """Run batched well-tempered metadynamics.

        Args:
            coords_A: (L, 3) initial P coordinates in Angstrom
            pairs: list of (i, j, w) WC pair tuples
            n_steps: total MD steps
            hill_height: base Gaussian hill height (kJ/mol)
            hill_sigma: Gaussian width in CV (Rg) space (nm)
            hill_freq: add a hill every this many steps
            max_hills: maximum number of hills to deposit
            well_tempered: use well-tempered biasing
            bias_factor: well-tempered gamma parameter
            dt_ps: timestep in picoseconds
            friction: Langevin friction coefficient (1/ps)
            temperature: simulation temperature (K)
            verbose: print progress

        Returns:
            (best_coords_A, best_energy_kJ, diagnostics)
        """
        dev = self.device
        L = self.L
        n_rep = self.n_replicas

        # Scale hill height by system size
        L_scale = max(1.0, L / 100.0)
        hill_height_scaled = hill_height * L_scale

        # ── Initialize 3-bead coordinates from P-only input ──
        rng = np.random.default_rng(42)
        p_nm = np.asarray(coords_A, dtype=np.float64) / 10.0
        pos0 = np.zeros((3 * L, 3), dtype=np.float64)
        for i in range(L):
            pos0[3 * i + 0] = p_nm[i]
            pos0[3 * i + 1] = p_nm[i] + rng.normal(0, 0.03, 3)
            pos0[3 * i + 2] = p_nm[i] + rng.normal(0, 0.03, 3)

        # Batch: (n_rep, 3L, 3)
        pos = torch.tensor(pos0, dtype=torch.float32, device=dev)[None].repeat(
            n_rep, 1, 1).contiguous()
        vel = _safe_zeros(pos.shape, dev)

        # ── Pairs tensor ──
        if pairs:
            pairs_t = torch.tensor(
                np.asarray(pairs)[:, :2], dtype=torch.long, device=dev)
            pw = torch.tensor(
                [p[2] for p in pairs], dtype=torch.float32, device=dev)
        else:
            pairs_t = torch.zeros(0, 2, dtype=torch.long, device=dev)
            pw = None

        temps_t = torch.full((n_rep,), temperature, dtype=torch.float32,
                             device=dev)

        # ── Cell list for clash detection ──
        cell_list = GPUCellList(cell_size=1.5)

        # ── Metadynamics state ──
        hills_centers = []   # list of (n_rep,) Rg values per hill
        hills_heights = []   # list of (n_rep,) hill weights per hill
        cv_history = []      # all deposited CV centers (for well-tempered)
        best_energy = float("inf")
        best_pos = coords_A.copy()
        n_deposited = 0

        n_hill_events = n_steps // hill_freq
        if verbose:
            print(f"    [GPU-MetaD] {n_rep} replicas, {n_steps} steps, "
                  f"hill_freq={hill_freq}, max_hills={max_hills}")

        def _total_force(positions: "torch.Tensor") -> "torch.Tensor":
            """The full force at `positions`: CG force field + accumulated-hill bias.

            batch_langevin_step's last B half-kick acts at the post-update coordinates,
            so it must receive the force evaluated there; reusing the force tensor
            computed before the step makes the deterministic part of the map
            non-symplectic and grows phase-space volume on every step. This closure
            must reproduce the FULL force the loop assembles below, bias included --
            the bias is part of the Hamiltonian the integrator is propagating, so
            dropping it here would integrate a different system. hills_centers and
            hills_heights are read at call time, so hills deposited in earlier blocks
            are part of the force exactly as they are in the loop's own f_total.
            """
            with torch.no_grad():
                pos_grad = positions.detach().clone().requires_grad_(True)
            _, f_phys = cg_energy_forces(
                pos_grad, pairs_t, pw, cell_list=cell_list,
                relax_bond_k=self.relax_bond_k, relax_angle_k=self.relax_angle_k,
                relax_pair_k=self.relax_pair_k, restraint_k=self.restraint_k)
            f_bias_new = self._compute_bias_forces(
                positions, hills_centers, hills_heights, hill_sigma)
            return f_phys + f_bias_new

        for step_idx in range(n_hill_events):
            # ── Integrate hill_freq steps ──
            for _ in range(hill_freq):
                # Compute energy + forces from CG force field
                with torch.no_grad():
                    pos_in = pos.detach().clone().requires_grad_(True)
                en, f_cg = cg_energy_forces(
                    pos_in, pairs_t, pw, cell_list=cell_list,
                    relax_bond_k=self.relax_bond_k, relax_angle_k=self.relax_angle_k,
                    relax_pair_k=self.relax_pair_k, restraint_k=self.restraint_k)

                # ── Bias forces from accumulated hills ──
                f_bias = self._compute_bias_forces(
                    pos, hills_centers, hills_heights, hill_sigma)

                # Total force = CG + bias
                f_total = f_cg + f_bias

                # Langevin step. force_fn supplies the tail B half-kick with the
                # force at the post-update coordinates, which is what makes the
                # deterministic part symplectic; it costs one extra force evaluation.
                pos, vel = batch_langevin_step(
                    pos, vel, f_total, temps_t,
                    dt_ps=dt_ps, friction=friction,
                    force_fn=_total_force)

                # Clamp positions to prevent explosion
                pos.data.clamp_(-10.0, 10.0)

            # ── Compute current CV (Rg) for each replica ──
            with torch.no_grad():
                rg_nm = _compute_rg(pos)  # (n_rep,) in nm

            # ── Evaluate energy for best-tracking ──
            with torch.no_grad():
                en_check, _ = cg_energy_forces(
                    pos.detach().clone().requires_grad_(True),
                    pairs_t, pw, cell_list=cell_list,
                    relax_bond_k=self.relax_bond_k, relax_angle_k=self.relax_angle_k,
                    relax_pair_k=self.relax_pair_k, restraint_k=self.restraint_k)
            energies = en_check.cpu().numpy()

            # Add bias energy to get total effective energy
            if hills_centers:
                bias_e = self._compute_bias_energy(
                    rg_nm, hills_centers, hills_heights, hill_sigma)
                total_e = energies + bias_e.cpu().numpy()
            else:
                total_e = energies

            # Track best (lowest CG energy, not bias energy)
            i_min = int(np.argmin(energies))
            if energies[i_min] < best_energy:
                best_energy = float(energies[i_min])
                best_pos = pos[i_min].detach().cpu().numpy()[0::3] * 10.0

            # ── Deposit new hill ──
            if n_deposited < max_hills:
                if well_tempered and cv_history:
                    # Well-tempered: height adapted by visit count
                    # For each replica, check how many past hills are nearby
                    rg_np = rg_nm.cpu().numpy()
                    heights = np.zeros(n_rep, dtype=np.float64)
                    for r in range(n_rep):
                        n_visits = 0
                        for past_rg in cv_history:
                            if abs(past_rg - rg_np[r]) < hill_sigma * 2:
                                n_visits += 1
                        heights[r] = hill_height_scaled / (1.0 + n_visits / bias_factor)
                else:
                    heights = np.full(n_rep, hill_height_scaled, dtype=np.float64)

                # Only deposit if height > threshold
                if np.any(heights > 0.1):
                    hills_centers.append(rg_nm.detach().clone())
                    hills_heights.append(
                        torch.tensor(heights, dtype=torch.float32, device=dev))
                    # Store average Rg for visit counting
                    cv_history.append(float(rg_nm.mean()))
                    n_deposited += 1

            if verbose and step_idx % 50 == 0:
                rg_avg = float(rg_nm.mean())
                rg_std = float(rg_nm.std())
                print(f"    [GPU-MetaD] {step_idx * hill_freq}/{n_steps}: "
                      f"Rg={rg_avg:.2f}+/-{rg_std:.2f}nm "
                      f"E_min={energies.min():.0f} "
                      f"hills={n_deposited}")

        # ── Final: minimize with bias zeroed out ──
        if verbose:
            print(f"    [GPU-MetaD] Deposition done, {n_deposited} hills. "
                  f"Final minimization...")
        # Just minimize the CG energy (no bias) for final structure
        pos_min = pos.detach().clone().requires_grad_(True)
        opt = torch.optim.Adam([pos_min], lr=5e-4)
        for _ in range(200):
            opt.zero_grad()
            e, _ = cg_energy_forces(
                pos_min, pairs_t, pw, cell_list=cell_list,
                relax_bond_k=self.relax_bond_k, relax_angle_k=self.relax_angle_k,
                relax_pair_k=self.relax_pair_k, restraint_k=self.restraint_k)
            e.sum().backward()
            opt.step()
            pos_min.data.clamp_(-10.0, 10.0)

        with torch.no_grad():
            e_final, _ = cg_energy_forces(
                pos_min.detach(), pairs_t, pw, cell_list=cell_list)
        e_final_np = e_final.cpu().numpy()
        i_best_final = int(np.argmin(e_final_np))
        e_final_best = float(e_final_np[i_best_final])
        if e_final_best < best_energy:
            best_energy = e_final_best
            best_pos = pos_min[i_best_final].detach().cpu().numpy()[0::3] * 10.0

        if verbose:
            print(f"    [GPU-MetaD] Done: {n_deposited} hills, "
                  f"best E={best_energy:.0f}")

        diag = {
            "n_hills": n_deposited,
            "cv_history": cv_history,
        }
        return best_pos, best_energy, diag

    def _compute_bias_forces(
        self,
        pos: "torch.Tensor",          # (B, N_tot, 3) 3-bead coords
        hills_centers: list,           # list of (B,) Rg centers
        hills_heights: list,           # list of (B,) hill weights
        sigma: float,
    ) -> "torch.Tensor":
        """Compute accumulated bias forces from all hills.

        Returns (B, N_tot, 3) force tensor.
        """
        if not hills_centers:
            return torch.zeros_like(pos)

        B, N_tot, _ = pos.shape
        dev = pos.device
        L = N_tot // 3
        P_idx = _arange_dev(L, dev) * 3  # P particle indices

        # Current Rg for all replicas: (B,)
        rg = _compute_rg(pos)
        # Current dRg/dx for P particles: (B, L, 3)
        # Only P particles contribute to Rg (C4'/N are virtual)
        pos_p = pos[:, P_idx, :]  # (B, L, 3)
        drg_dx = _compute_rg_grad(pos_p)  # (B, L, 3)

        # Accumulate bias over all hills
        bias = torch.zeros_like(pos)  # (B, N_tot, 3)
        sigma_sq = sigma * sigma

        for center_t, height_t in zip(hills_centers, hills_heights):
            # center_t: (B,), height_t: (B,)
            # Gaussian: w * exp(-(rg - center)^2 / (2*sigma^2))
            delta_rg = rg - center_t  # (B,)
            arg = 0.5 * (delta_rg / sigma) ** 2  # (B,)
            arg = torch.clamp(arg, max=50.0)
            hill_val = height_t * torch.exp(-arg)  # (B,)

            # dV/dRg = -w * (rg - center) / sigma^2 * exp(...)
            dV_drg = -hill_val * delta_rg / sigma_sq  # (B,)

            # Force on particle i: F_i = -dV/dx_i = -dV/dRg * dRg/dx_i
            # = hill_val * (rg - center) / sigma^2 * drg_dx
            # (B,) * (B, L, 3) -> (B, L, 3)
            f_on_p = (dV_drg / (rg + 1e-8)).unsqueeze(-1).unsqueeze(-1) * drg_dx

            # Scale by hill_val explicitly (dV_drg already includes it)
            # Actually dV_drg = -hill_val * delta_rg / sigma_sq, so
            # force = -dV/dx = hill_val * delta_rg / sigma_sq / rg * drg_dx
            # which is what we have above. Correct.

            # Write to full particle array (only P particles affected)
            bias[:, P_idx, :] += f_on_p

        # Clamp per-particle force magnitude
        force_norms = bias.norm(dim=-1)  # (B, N_tot)
        scale = torch.clamp(force_norms / 500.0, min=1.0)
        bias = bias / scale.unsqueeze(-1)

        return bias

    def _compute_bias_energy(
        self,
        rg: "torch.Tensor",            # (B,) current Rg
        hills_centers: list,            # list of (B,) Rg centers
        hills_heights: list,            # list of (B,) hill weights
        sigma: float,
    ) -> "torch.Tensor":
        """Compute total bias energy V(Rg) for each replica. Returns (B,)."""
        if not hills_centers:
            return torch.zeros(rg.shape[0], device=rg.device)

        sigma_sq = sigma * sigma
        total = torch.zeros_like(rg)
        for center_t, height_t in zip(hills_centers, hills_heights):
            delta_rg = rg - center_t
            arg = torch.clamp(0.5 * (delta_rg / sigma) ** 2, max=50.0)
            total = total + height_t * torch.exp(-arg)
        return total


# ══════════════════════════════════════════════════════════════════════
#  Smoke test (no OpenMM needed)
# ══════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Test GPU BatchedMetadynamics with a small system."""
    import time

    print("=" * 60)
    print("BatchedMetadynamics GPU smoke test")
    print("=" * 60)

    if not TORCH_OK:
        print("ERROR: PyTorch not available")
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    if dev == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    L = 60
    seq = ("AUGCAUGC" * 8)[:L]
    pairs = [(i, i + 30, 1.0) for i in range(5)]

    # Initial coords: helix-like
    coords = np.zeros((L, 3))
    for i in range(L):
        ang = i * (5.9 / 11.0) * np.pi * 2 * 0.55
        coords[i] = [8 * np.cos(ang), 8 * np.sin(ang), i * 4.7]

    n_rep = 4
    meta = BatchedMetadynamics(n_replicas=n_rep, sequence=seq, device=dev)

    t0 = time.time()
    best_pos, best_e, diag = meta.run(
        coords, pairs,
        n_steps=2000,
        hill_height=1.0,
        hill_sigma=1.0,
        hill_freq=100,
        max_hills=50,
        well_tempered=True,
        bias_factor=5.0,
        verbose=True,
    )
    elapsed = time.time() - t0

    print(f"\nResults:")
    print(f"  Best energy: {best_e:.0f} kJ/mol")
    print(f"  Hills deposited: {diag['n_hills']}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Output shape: {best_pos.shape}")
    assert best_pos.shape == (L, 3), f"Expected ({L}, 3), got {best_pos.shape}"
    assert np.all(np.isfinite(best_pos)), "Non-finite coordinates"
    assert np.all(np.isfinite(best_e)), "Non-finite energy"

    print("\n" + "=" * 60)
    print("PASS: BatchedMetadynamics GPU smoke test")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
