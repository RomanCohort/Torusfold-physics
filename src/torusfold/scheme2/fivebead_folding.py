"""fivebead_folding.py — IsRNAcirc 式 5-bead CG 力场 + 退火。

5-bead 每核苷酸: P / S(sugar) / B1(major groove) / B2(minor groove) / B3(glycosidic N)
优势: 比 3-bead 多 sugar ring 和 base ring 的独立描述, 更好捕捉 stacking/H-bond 几何。

力场项:
  1. P-P 骨架键 (全环) + BSJ (可调)
  2. P-S 残基内键
  3. S-B3 残基内键 (sugar → glycosidic N)
  4. S-B1, S-B2 残基内键 (sugar → base grooves)
  5. P-P-P 骨架键角 (A-form)
  6. P-P-P-P 二面角 (A-form 螺旋扭转)
  7. S-S 堆叠 LJ (相邻 sugar, 主 stacking)
  8. B1-B1 堆叠 LJ (相邻 base major groove, 辅 stacking)
  9. WC 配对: B1-B1 方向依赖 12-10 H-bond
 10. 非键 clash + 静电
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .p_to_5bead import p_to_5bead, split_5bead_coords
from .refine import BOND_LEN


def build_5bead_system(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    *,
    bsj_k_scale: float = 0.1,
    enabled: Optional[List[bool]] = None,
    stat_pot_path: Optional[str] = None,
    sequence: Optional[str] = None,
):
    """构建 5-bead CG OpenMM system.

    Args:
        p_coords: (L, 3) P-only coordinates (Å)
        pairs: [(i, j, w)] ViennaRNA pairing
        bsj_k_scale: BSJ spring constant multiplier
        enabled: 10 个 bool, 控制各 force 块.
            [bb, bsj, intra, angle, dihedral, stack_s, stack_b1, pair, clash, stat_pot]
        stat_pot_path: 统计势 pkl 路径
        sequence: RNA 序列 (用于统计势)
    """
    from openmm import (
        System, HarmonicBondForce, HarmonicAngleForce,
        CustomBondForce, CustomNonbondedForce, CustomTorsionForce,
    )

    L = len(p_coords)
    coords_5bead = p_to_5bead(p_coords)  # (5L, 3) Å
    coords_nm = coords_5bead / 10.0
    en = enabled if enabled is not None else [True] * 10
    if len(en) != 10:
        raise ValueError("enabled 必须是 10 个 bool")

    system = System()
    for _ in range(5 * L):
        system.addParticle(110.0)

    # Bead index helpers
    def P(i): return 5 * i
    def S(i): return 5 * i + 1
    def B1(i): return 5 * i + 2
    def B2(i): return 5 * i + 3
    def B3(i): return 5 * i + 4

    # 1. P-P 骨架键 + BSJ
    bb_k = 5000.0  # kJ/mol/nm² (CG, 不需要 AA 级别刚度)
    bsj_force = None
    if en[0] or en[1]:
        bond_bb = HarmonicBondForce()
        if en[0]:
            for i in range(L - 1):
                bond_bb.addBond(P(i), P(i + 1), BOND_LEN / 10.0, bb_k)
        system.addForce(bond_bb)
    if en[1]:
        bsj_force = CustomBondForce("0.5*k_bsj*(r-r0)^2")
        bsj_force.addPerBondParameter("k_bsj")
        bsj_force.addPerBondParameter("r0")
        bsj_force.addBond(P(L - 1), P(0), [bsj_k_scale * 500.0, BOND_LEN / 10.0])
        system.addForce(bsj_force)

    # 2. 残基内键 P-S, S-B3, S-B1, S-B2
    #    5-bead 初始坐标精度有限, 用较软力常数避免爆炸
    if en[2]:
        bond_intra = HarmonicBondForce()
        for i in range(L):
            bond_intra.addBond(P(i), S(i), 0.204, 5000.0)   # P-S: 2.04Å
            bond_intra.addBond(S(i), B3(i), 0.109, 5000.0)   # S-B3: 1.09Å
            bond_intra.addBond(S(i), B1(i), 0.115, 5000.0)   # S-B1: 1.15Å
            bond_intra.addBond(S(i), B2(i), 0.110, 5000.0)   # S-B2: 1.10Å
        system.addForce(bond_intra)

    # 3. P-P-P 骨架键角 (A-form 150°)
    if en[3]:
        angle_force = HarmonicAngleForce()
        angle0 = 2.618  # 150°
        angle_k = 100.0  # kJ/mol/rad²
        for i in range(L - 2):
            angle_force.addAngle(P(i), P(i + 1), P(i + 2), angle0, angle_k)
        system.addForce(angle_force)

    # 4. P-P-P-P 二面角 (A-form 螺旋扭转 33°)
    if en[4]:
        dihedral_force = CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
        dihedral_force.addGlobalParameter("k_dih", 500.0)  # kJ/mol/rad²
        dihedral_force.addGlobalParameter("theta0", 33.0 * math.pi / 180.0)
        for i in range(L - 3):
            dihedral_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
        system.addForce(dihedral_force)

    # 5. S-S 堆叠 (主 stacking, 相邻 sugar)
    #    5-bead 距离~3.4Å=LJσ, 全LJ会在r≈σ处爆炸
    #    改用 WCA-like: r<σ 时排斥(软), r>σ 时弱线性吸引
    if en[5]:
        stack_s = CustomBondForce(
            "step(sig-r)*k_rep*(sig-r)^2 - step(r-sig)*eps*(r-sig)/sig")
        stack_s.addPerBondParameter("eps")
        stack_s.addPerBondParameter("sig")
        stack_s.addPerBondParameter("k_rep")
        for i in range(L - 1):
            stack_s.addBond(S(i), S(i + 1), [1.0, 0.34, 500.0])  # ε=1, k_rep=500
        system.addForce(stack_s)

    # 6. B1-B1 堆叠 (辅 stacking, 相邻 base major groove)
    if en[6]:
        stack_b1 = CustomBondForce(
            "step(sig-r)*k_rep*(sig-r)^2 - step(r-sig)*eps*(r-sig)/sig")
        stack_b1.addPerBondParameter("eps")
        stack_b1.addPerBondParameter("sig")
        stack_b1.addPerBondParameter("k_rep")
        for i in range(L - 1):
            stack_b1.addBond(B1(i), B1(i + 1), [0.8, 0.34, 500.0])  # ε=0.8, k_rep=500
        system.addForce(stack_b1)

    # 7. WC 配对: B1-B1 12-10 H-bond
    if en[7] and pairs:
        pair_force = CustomBondForce(
            "pair_k_scale * w_pair * 30 * (5*(r0/r)^12 - 6*(r0/r)^10) * step(r_cut - r)")
        pair_force.addGlobalParameter("pair_k_scale", 1.0)
        pair_force.addGlobalParameter("w_pair", 1.0)
        pair_force.addGlobalParameter("r0", 0.50)   # 5.0Å → 0.50nm
        pair_force.addGlobalParameter("r_cut", 2.0)  # 20Å → 2.0nm
        for i, j, w in pairs:
            pair_force.addBond(B1(i), B1(j), [])
        system.addForce(pair_force)
    else:
        pair_force = None

    # 8. 非键 clash (5-bead 更密集, 需更小 dmin)
    if en[8]:
        clash_force = CustomNonbondedForce(
            "step(dmin-r)*k_clash*(dmin-r)^2")
        clash_force.addPerParticleParameter("q")
        clash_force.addGlobalParameter("dmin", 0.20)  # 2.0Å
        clash_force.addGlobalParameter("k_clash", 1000.0)  # kJ/mol/nm²
        clash_force.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        clash_force.setCutoffDistance(1.2)  # 12Å
        for i in range(5 * L):
            clash_force.addParticle([0.0])
        # Exclusions: intra-residue + backbone neighbors
        for i in range(L):
            # Intra-residue: P-S-B1-B2-B3 all excluded
            for a in range(5):
                for b in range(a + 1, 5):
                    clash_force.addExclusion(5 * i + a, 5 * i + b)
            # Backbone neighbors
            if i > 0:
                for a in range(5):
                    for b in range(5):
                        clash_force.addExclusion(5 * i + a, 5 * (i - 1) + b)
        system.addForce(clash_force)
    else:
        clash_force = None

    return system, coords_nm, pair_force, bsj_force


def refine_5bead(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str = "CPU",
    n_anneal: int = 30,
    stat_pot_path: Optional[str] = None,
    sequence: Optional[str] = None,
):
    """5-bead CG 三阶段退火。

    Args:
        p_coords: (L, 3) P-only 初始坐标 (Å)
        pairs: [(i, j, w)] ViennaRNA 配对
        platform_name: "CPU" 或 "CUDA"
        n_anneal: 每阶段 MD 步数 (× 1000)

    Returns:
        (p_refined, e0, e1): 优化后 P 坐标, 初始能量, 最终能量
    """
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import Simulation, Topology, Element

    L = len(p_coords)
    system, coords_nm, pair_force, bsj_force = \
        build_5bead_system(p_coords, pairs, stat_pot_path=stat_pot_path,
                           sequence=sequence)

    # Topology
    topo = Topology()
    chain = topo.addChain()
    for i in range(L):
        res = topo.addResidue("N", chain)
        for bead_name in ["P", "S", "B1", "B2", "B3"]:
            topo.addAtom(f"{bead_name}{i}", Element.getBySymbol("P"), res)

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond)

    try:
        platform_obj = Platform.getPlatformByName(platform_name)
    except Exception:
        platform_obj = Platform.getPlatformByName("CPU")

    sim = Simulation(topo, system, integrator, platform_obj)
    sim.context.setPositions(coords_nm)
    sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    # Initial minimize — 5-bead 需要更多步 (5L 粒子)
    sim.minimizeEnergy(maxIterations=5000)
    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy()._value

    def set_bsj_k(scale):
        if bsj_force is None:
            return
        bsj_force.setBondParameters(0, 5*(L-1), 0, [scale * 500.0, BOND_LEN / 10.0])
        bsj_force.updateParametersInContext(sim.context)

    def set_pair_k(scale):
        if pair_force is None:
            return
        pair_force.setGlobalParameterDefaultValue(0, scale)
        pair_force.updateParametersInContext(sim.context)

    pre_md = sim.context.getState(getPositions=True, getEnergy=True)

    # Phase 1: 弱 BSJ, 强配对
    set_pair_k(1.0); set_bsj_k(0.1)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal); sim.minimizeEnergy(maxIterations=5000)

    # Phase 2: 中 BSJ, 强配对
    set_pair_k(1.0); set_bsj_k(0.5)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(n_anneal * 3); sim.minimizeEnergy(maxIterations=8000)

    # Phase 3: 强 BSJ, 低温
    set_pair_k(1.0); set_bsj_k(5.0)
    sim.integrator.setTemperature(290 * unit.kelvin)
    sim.step(n_anneal * 2)
    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
                       maxIterations=15000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value
    e1 = state.getPotentialEnergy()._value

    # Safety: only discard if energy went 10x worse (真正的爆炸)
    e_pre = pre_md.getPotentialEnergy()._value
    if e1 > e_pre * 10 and e_pre < 0:
        pos = pre_md.getPositions(asNumpy=True)._value
        e1 = e_pre

    # Extract P-only coordinates from 5-bead
    p_refined = (pos * 10.0)[0::5].copy()  # P bead every 5 atoms
    return p_refined, e0, e1
