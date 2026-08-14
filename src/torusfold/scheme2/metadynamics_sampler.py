# -*- coding: utf-8 -*-
"""metadynamics_sampler.py — 环状 RNA Metadynamics 增强采样

沿着集合变量 (CV) 主动加 Gaussian hill 偏置, 强制构象跨越自由能垒.
比 REMD 更高效: 不是"暴力升温", 而是"精确推开".

集合变量:
  CV1 = BSJ 距离 (P0 ↔ P(L-1))
  CV2 = 配对接触分数 (native contacts)
  CV3 = 回旋半径 (radius of gyration)

偏置势:
  U_bias = Σ_i w * exp(-((CV - CV_i)^2) / (2σ^2))
  每 hill_freq 步加一个 hill, well-tempered 模式下 hill 高度随访问次数衰减.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

try:
    import openmm as mm
    from openmm import (
        unit, LangevinMiddleIntegrator, Platform,
    )
    from openmm.app import Simulation, Topology, Element
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False


class MetaDynamicsSampler:
    """环状 RNA Metadynamics 增强采样器.

    实现: Python 侧计算 CV + 偏置梯度, 通过 CustomExternalForce 注入.
    每 hill_freq 步添加一个 Gaussian hill.

    Args:
        sequence: RNA 序列
        pairs: [(i, j, w)] ViennaRNA 配对
        hill_height: Gaussian hill 高度 (kJ/mol), 默认 1.0
        hill_sigma: Gaussian hill 宽度 (CV 空间, nm), 默认 0.1
        hill_freq: 加 hill 的间隔 (步), 默认 100
        max_hills: 最大 hill 数, 默认 5000
        well_tempered: 是否用 well-tempered 模式 (hill 高度衰减)
        bias_factor: well-tempered 偏置因子, 默认 5.0
        platform_name: OpenMM 平台, 默认 "CPU"
    """

    def __init__(
        self,
        sequence: str,
        pairs: List[Tuple[int, int, float]],
        hill_height: float = 1.0,
        hill_sigma: float = 0.1,
        hill_freq: int = 100,
        max_hills: int = 5000,
        well_tempered: bool = True,
        bias_factor: float = 5.0,
        platform_name: str = "CPU",
    ):
        if not OPENMM_AVAILABLE:
            raise ImportError("OpenMM not available")

        self.sequence = sequence
        self.pairs = pairs
        self.L = len(sequence)
        self.hill_height = hill_height
        self.hill_sigma = hill_sigma
        self.hill_freq = hill_freq
        self.max_hills = max_hills
        self.well_tempered = well_tempered
        self.bias_factor = bias_factor
        self.platform_name = platform_name

    def sample(
        self,
        p_init: np.ndarray,
        n_steps: int = 50000,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """执行 Metadynamics 采样.

        Args:
            p_init: (L, 3) P 坐标 (Å)
            n_steps: 总 MD 步数
            verbose: 打印信息

        Returns:
            (best_coords, best_energy): 最佳 P 坐标 (Å) 和能量 (kJ/mol)
        """
        L = self.L
        pairs = self.pairs

        # 构建 3-bead 系统
        from .openmm_gpu_refiner import (
            _build_3bead_system_gpu, _create_3bead_topology,
        )
        system, coords_nm, pair_force, stack_force, bsj_force, bsj_guide = \
            _build_3bead_system_gpu(
                p_init, pairs, pair_scale=1.0, bsj_k_scale=1.0)

        # P bead 索引: 3-bead 拓扑中 P=3*i
        P_indices = [3 * i for i in range(L)]

        topo = _create_3bead_topology(L)
        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)

        try:
            plat = Platform.getPlatformByName(self.platform_name)
        except Exception:
            plat = Platform.getPlatformByName("CPU")

        sim = Simulation(topo, system, integrator, plat)
        sim.context.setPositions(coords_nm * unit.nanometer)

        # ── Metadynamics 循环 ──
        hills = []  # [(cv1, cv2, cv3, height), ...]
        best_energy = float("inf")
        best_pos = p_init.copy()
        cv_history = []

        for step in range(0, n_steps, self.hill_freq):
            # MD 积分
            sim.step(self.hill_freq)

            # 获取当前状态
            state = sim.context.getState(getPositions=True, getEnergy=True)
            pos = state.getPositions(asNumpy=True)._value  # nm
            energy = state.getPotentialEnergy()._value

            p_coords = pos[P_indices].copy() * 10.0  # Å

            # 计算 CV
            cv1 = self._cv_bsj_distance(p_coords)
            cv2 = self._cv_native_contacts(p_coords)
            cv3 = self._cv_radius_of_gyration(p_coords)

            # 计算 hill 高度
            if self.well_tempered and cv_history:
                n_visits = sum(
                    1 for (h1, h2, h3) in cv_history
                    if (abs(h1 - cv1) < self.hill_sigma * 2 and
                        abs(h2 - cv2) < self.hill_sigma * 2))
                height = self.hill_height / (1 + n_visits / self.bias_factor)
            else:
                height = self.hill_height

            if len(hills) < self.max_hills and height > 0.01:
                hills.append((cv1, cv2, cv3, height))
                cv_history.append((cv1, cv2, cv3))

                # 通过 CustomExternalForce 施加偏置梯度
                self._apply_bias(sim, hills, pos, P_indices)

            # 跟踪最佳构象
            if energy < best_energy:
                best_energy = energy
                best_pos = p_coords.copy()

            if verbose and (step // self.hill_freq) % 100 == 0:
                print(f"    MetaD {step}/{n_steps}: "
                      f"BSJ={cv1:.2f}nm nc={cv2:.2f} Rg={cv3:.2f}nm "
                      f"E={energy:.0f} hills={len(hills)}")

        # 终局最小化
        sim.minimizeEnergy(
            tolerance=5.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=5000)
        state = sim.context.getState(getPositions=True, getEnergy=True)
        pos_final = state.getPositions(asNumpy=True)._value
        energy_final = state.getPotentialEnergy()._value

        p_final = pos_final[P_indices].copy() * 10.0
        if energy_final < best_energy:
            best_energy = energy_final
            best_pos = p_final

        if verbose:
            print(f"    MetaD 完成: {len(hills)} hills, best E={best_energy:.0f}")

        return best_pos, best_energy

    # ── CV 计算 ──

    def _cv_bsj_distance(self, p_coords: np.ndarray) -> float:
        """CV1: BSJ 距离 (nm)."""
        return float(np.linalg.norm(p_coords[0] - p_coords[-1])) / 10.0

    def _cv_native_contacts(self, p_coords: np.ndarray) -> float:
        """CV2: native contact 分数 (0-1)."""
        if not self.pairs:
            return 0.0
        d_cut = 1.5  # nm
        count = 0
        total = 0
        for (i, j, w) in self.pairs:
            if abs(i - j) < 5:
                continue
            total += 1
            d = np.linalg.norm(p_coords[i] - p_coords[j]) / 10.0
            if d < d_cut:
                count += 1
        return count / max(1, total)

    def _cv_radius_of_gyration(self, p_coords: np.ndarray) -> float:
        """CV3: 回旋半径 (nm)."""
        center = np.mean(p_coords, axis=0)
        rg = np.sqrt(np.mean(np.sum((p_coords - center) ** 2, axis=1)))
        return rg / 10.0

    # ── 偏置施加 ──

    def _apply_bias(self, sim, hills, current_pos_nm, P_indices):
        """用 CustomExternalForce 施加 metadynamics 偏置梯度.

        简化实现: 计算偏置势的梯度 ∂U/∂x, 直接作为 force 加到 P beads 上.
        """
        state = sim.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)._value  # (N, 3) kJ/mol/nm

        L = self.L
        pos_p = current_pos_nm[P_indices]  # (L, 3) nm

        cv1 = self._cv_bsj_distance(pos_p * 10.0)
        cv2 = self._cv_native_contacts(pos_p * 10.0)
        cv3 = self._cv_radius_of_gyration(pos_p * 10.0)

        sigma = self.hill_sigma
        bias_forces = np.zeros((L, 3), dtype=np.float64)

        center = np.mean(pos_p, axis=0)
        rg = max(1e-8, np.sqrt(np.mean(np.sum((pos_p - center) ** 2, axis=1))))

        # BSJ distance 梯度
        diff_bsj = pos_p[0] - pos_p[-1]
        dist_bsj = max(1e-8, np.linalg.norm(diff_bsj))

        for (h1, h2, h3, w) in hills:
            hill_val = w * math.exp(
                -0.5 * ((cv1 - h1) ** 2 + (cv2 - h2) ** 2 + (cv3 - h3) ** 2)
                / (sigma ** 2))

            dc1 = (cv1 - h1) / (sigma ** 2)
            dc2 = (cv2 - h2) / (sigma ** 2)
            dc3 = (cv3 - h3) / (sigma ** 2)

            for i in range(L):
                grad = np.zeros(3)

                # ∂CV1/∂P(i): BSJ distance gradient
                if i == 0:
                    grad += dc1 * (-diff_bsj / dist_bsj)
                elif i == L - 1:
                    grad += dc1 * (diff_bsj / dist_bsj)

                # ∂CV3/∂P(i): radius of gyration gradient
                grad += dc3 * (pos_p[i] - center) / (rg * L)

                # ∂CV2/∂P(i): native contacts gradient (simplified)
                for (ii, jj, ww) in self.pairs:
                    if abs(ii - jj) < 5:
                        continue
                    if i == ii or i == jj:
                        d_ij = pos_p[ii] - pos_p[jj]
                        dist_ij = max(1e-8, np.linalg.norm(d_ij))
                        sign = 1.0 if i == ii else -1.0
                        grad += dc2 * sign * (-d_ij / dist_ij) / max(1, len(self.pairs))

                bias_forces[i] += grad * hill_val

        # 应用偏置: 通过位置偏移模拟偏置力 (OpenMM 8 不支持 setForces)
        # bias_forces 单位 kJ/mol/nm, 除以 ~1000 kJ/mol/nm 得到 nm 级位移
        state_pos = sim.context.getState(getPositions=True)
        pos_arr = state_pos.getPositions(asNumpy=True)._value  # nm
        for i in range(L):
            pos_arr[P_indices[i]] += bias_forces[i] * 1e-3  # 小步偏移
        sim.context.setPositions(pos_arr * unit.nanometer)
