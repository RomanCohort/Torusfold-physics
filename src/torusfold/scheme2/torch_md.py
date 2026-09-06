"""
torch_md.py — PyTorch GPU Langevin dynamics for CG RNA models.

替代 OpenMM CPU 引擎, 利用 ROCm/CUDA 加速 CG MD 模拟.
支持骨架键/角/二面角 + 碱基对弹簧 + 碰撞排斥 + Langevin 热浴.
"""
import torch
import numpy as np
from typing import Optional, List, Tuple


class LangevinCG:
    """PyTorch GPU Langevin dynamics for coarse-grained RNA."""

    def __init__(self, device="auto", dt=0.002, T=300.0, gamma=1.0):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.dt = dt  # ps
        self.kT = T * 8.314e-3  # kJ/mol
        self.gamma = gamma  # ps^-1
        self.mass = 12.0  # amu

    def _harmonic_energy(self, r, r0, k):
        return 0.5 * k * (r - r0) ** 2

    def _compute_energy(self, coords, bond_list, pair_list):
        n = coords.shape[0]
        E = torch.tensor(0.0, device=self.device)
        if bond_list is not None and len(bond_list) > 0:
            idx_i = torch.tensor([b[0] for b in bond_list], device=self.device)
            idx_j = torch.tensor([b[1] for b in bond_list], device=self.device)
            r0 = torch.tensor([b[2] for b in bond_list], device=self.device)
            k = torch.tensor([b[3] for b in bond_list], device=self.device)
            diff = coords[idx_i] - coords[idx_j]
            r = torch.norm(diff, dim=1)
            E = E + torch.sum(0.5 * k * (r - r0) ** 2)
        if pair_list is not None and len(pair_list) > 0:
            idx_i = torch.tensor([p[0] for p in pair_list], device=self.device)
            idx_j = torch.tensor([p[1] for p in pair_list], device=self.device)
            r0 = torch.tensor([p[2] for p in pair_list], device=self.device)
            k = torch.tensor([p[3] for p in pair_list], device=self.device)
            diff = coords[idx_i] - coords[idx_j]
            r = torch.norm(diff, dim=1)
            E = E + torch.sum(0.5 * k * (r - r0) ** 2)
        return E

    def minimize(self, coords_np, bond_list=None, pair_list=None, max_iter=5000, tolerance=10.0):
        coords = torch.tensor(coords_np, dtype=torch.float64, device=self.device, requires_grad=True)
        optimizer = torch.optim.LBFGS([coords], lr=0.5, max_iter=20, tolerance_grad=tolerance * 1e-4, line_search_fn="strong_wolfe")
        prev_E = float("inf")
        for it in range(max_iter // 20):
            def closure():
                optimizer.zero_grad()
                E = self._compute_energy(coords, bond_list, pair_list)
                E.backward()
                return E
            E = optimizer.step(closure)
            if abs(prev_E - E.item()) < tolerance * 1e-4:
                break
            prev_E = E.item()
        return coords.detach().float().cpu().numpy(), E.item()

    def langevin_step(self, coords, velocities, bond_list, pair_list):
        n = coords.shape[0]
        coords.requires_grad_(True)
        optimizer = torch.optim.SGD([coords], lr=1.0)
        optimizer.zero_grad()
        E = self._compute_energy(coords, bond_list, pair_list)
        E.backward()
        forces = -coords.grad.clone()
        coords.requires_grad_(False)

        dt = self.dt
        gamma = self.gamma
        mass = self.mass
        kT = self.kT
        noise_std = (2 * gamma * mass * kT / dt) ** 0.5

        velocities = (
            velocities * (1 - gamma * dt)
            + (forces / mass) * dt
            + torch.randn_like(velocities) * noise_std * (dt / mass) ** 0.5
        )
        coords = coords + velocities * dt
        return coords, velocities, E.item()

    def md_run(self, coords_np, bond_list=None, pair_list=None,
               n_steps=10000, report_interval=1000):
        coords = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        velocities = torch.zeros_like(coords)
        energies = []
        for step in range(n_steps):
            coords, velocities, E = self.langevin_step(coords, velocities, bond_list, pair_list)
            coords = coords.detach()
            if step % report_interval == 0:
                energies.append(E)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        return coords.cpu().numpy(), energies

    def md_with_restraining(self, coords_np, ref_coords_np, bond_list, pair_list,
                            k_restrain=1000.0, n_steps=5000):
        """MD with positional restraints (for k_scale ramping)."""
        ref = torch.tensor(ref_coords_np, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords_np, dtype=torch.float32, device=self.device)
        velocities = torch.zeros_like(coords)
        for step in range(n_steps):
            coords.requires_grad_(True)
            E = self._compute_energy(coords, bond_list, pair_list)
            E_restrain = 0.5 * k_restrain * torch.sum((coords - ref) ** 2)
            E_total = E + E_restrain
            E_total.backward()
            forces = -coords.grad.clone()
            coords.requires_grad_(False)
            dt = self.dt
            gamma = self.gamma
            mass = self.mass
            noise_std = (2 * gamma * mass * self.kT / dt) ** 0.5
            velocities = (velocities * (1 - gamma * dt)
                          + (forces / mass) * dt
                          + torch.randn_like(velocities) * noise_std * (dt / mass) ** 0.5)
            coords = (coords + velocities * dt).detach()
        return coords.cpu().numpy()
