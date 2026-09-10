"""fivebead_folding.py — IsRNAcirc-style 5-bead CG force field + annealing.

Each nucleotide is coarse-grained to 5 beads: P / S(sugar) / B1(major groove) /
B2(minor groove) / B3(glycosidic N).
Advantage over 3-bead: the sugar ring and base ring are described independently,
which better captures stacking/H-bond geometry.

Force-field terms:
  1. P-P backbone bond (full ring) + BSJ (adjustable)
  2. P-S intra-residue bond
  3. S-B3 intra-residue bond (sugar → glycosidic N)
  4. S-B1, S-B2 intra-residue bonds (sugar → base grooves)
  5. P-P-P backbone angle (A-form)
  6. P-P-P-P dihedral (A-form helical twist)
  7. B1-B1 stacking LJ (adjacent bases' major-groove edge, primary stacking, ε=1.5)
  8. WC pairing: B1-B1 orientation-dependent 12-10 H-bond
  9. Nonbonded clash + electrostatics
 10. DL Gaussian restraint: B1-B1 distance (optional)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    from .p_to_5bead import p_to_5bead, split_5bead_coords
except ImportError:
    from p_to_5bead import p_to_5bead, split_5bead_coords
try:
    from .refine import BOND_LEN
except ImportError:
    BOND_LEN = 5.9


def build_5bead_system(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    *,
    bsj_k_scale: float = 0.1,
    enabled: Optional[List[bool]] = None,
    stat_pot_path: Optional[str] = None,
    sequence: Optional[str] = None,
    dl_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
    intra_k: float = 5000.0,
):
    """Build a 5-bead CG OpenMM system.

    Args:
        p_coords: (L, 3) P-only coordinates (Å)
        pairs: [(i, j, w)] ViennaRNA pairing
        bsj_k_scale: BSJ spring constant multiplier
        enabled: 10 bools, one per force block.
            [bb, bsj, intra, angle, dihedral, ps_bond, stack_b1, pair, clash, stat_pot]
            Note: block #5 was originally an S-S stacking term (physically unsound);
            it has been replaced by a P-S backbone restraint.
        stat_pot_path: path to the statistical-potential pkl
        sequence: RNA sequence (used by the statistical potential)
        dl_constraints: [(i, j, d_ij_A, w)] DL-predicted base-distance restraints (B1-B1, Å)
            i, j: residue indices (0-indexed)
            d_ij: predicted distance (Å)
            w: confidence weight (0-1), controls the Gaussian amplitude and width
            Should be w=0 within ±50 nt of the BSJ

        Gaussian form: E = A * exp(-(r-μ)²/(2σ²))
            A = 500 * w (kJ/mol)
            σ = 0.3 + 1.2*(1-w) Å  (high confidence → narrow, low confidence → wide)
            μ = d_ij / 10 (nm)
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
        raise ValueError("enabled must be a list of 10 bools")

    system = System()
    for _ in range(5 * L):
        system.addParticle(110.0)

    # Bead index helpers
    def P(i): return 5 * i
    def S(i): return 5 * i + 1
    def B1(i): return 5 * i + 2
    def B2(i): return 5 * i + 3
    def B3(i): return 5 * i + 4

    # 1. P-P backbone bond + BSJ
    bb_k = 5000.0  # kJ/mol/nm² (CG, no need for AA-level stiffness)
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

    # 2. Intra-residue bonds P-S, S-B3, S-B1, S-B2
    if en[2]:
        bond_intra = HarmonicBondForce()
        for i in range(L):
            bond_intra.addBond(P(i), S(i), 0.204, intra_k)   # P-S: 2.04Å
            bond_intra.addBond(S(i), B3(i), 0.109, intra_k)   # S-B3: 1.09Å
            bond_intra.addBond(S(i), B1(i), 0.115, intra_k)   # S-B1: 1.15Å
            bond_intra.addBond(S(i), B2(i), 0.110, intra_k)   # S-B2: 1.10Å
        system.addForce(bond_intra)

    # 3. P-P-P backbone angle (A-form 150°)
    if en[3]:
        angle_force = HarmonicAngleForce()
        angle0 = 2.618  # 150°
        angle_k = 100.0  # kJ/mol/rad²
        for i in range(L - 2):
            angle_force.addAngle(P(i), P(i + 1), P(i + 2), angle0, angle_k)
        system.addForce(angle_force)

    # 4. P-P-P-P dihedral (A-form helical twist 33°)
    if en[4]:
        dihedral_force = CustomTorsionForce("0.5*k_dih*(theta-theta0)^2")
        dihedral_force.addGlobalParameter("k_dih", 500.0)  # kJ/mol/rad²
        dihedral_force.addGlobalParameter("theta0", 33.0 * math.pi / 180.0)
        for i in range(L - 3):
            dihedral_force.addTorsion(P(i), P(i + 1), P(i + 2), P(i + 3))
        system.addForce(dihedral_force)

    # 5. (removed: the P-S bond is already added in the #2 intra-residue block, not repeated)

    # 6. B1-B1 stacking (auxiliary stacking between adjacent bases' major-groove edges)
    if en[6]:
        stack_b1 = CustomBondForce(
            "step(sig-r)*k_rep*(sig-r)^2 - step(r-sig)*eps*(r-sig)/sig")
        stack_b1.addPerBondParameter("eps")
        stack_b1.addPerBondParameter("sig")
        stack_b1.addPerBondParameter("k_rep")
        for i in range(L - 1):
            stack_b1.addBond(B1(i), B1(i + 1), [1.5, 0.34, 500.0])  # ε=1.5 (raised from 0.8 to compensate for S-S removal), k_rep=500
        system.addForce(stack_b1)

    # 7. WC pairing: B1-B1 12-10 H-bond
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

    # 8. Nonbonded clash (5-bead is denser, so a smaller dmin is needed)
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

    # 9. DL distance restraints: B1-B1 Gaussian potential; target distances come from
    #    trRosettaRNA2/RhoFold+/DivideFold
    #    E_DL = Σ A * exp[-(r - μ)² / (2σ²)]
    #    Gaussian advantage: it decays automatically at long range, so it never tears the
    #    structure apart, and it is robust to incorrect DL predictions
    #    w=0 within ±50 nt of the BSJ (DL is unreliable for circularization topology)
    dl_force = None
    if dl_constraints:
        # Gaussian: A * exp(-(r-mu)^2 / (2*sigma^2))
        # OpenMM CustomBondForce uses: exp(-k_gauss * (r - mu)^2)
        # where k_gauss = 1 / (2 * sigma^2)
        dl_force = CustomBondForce(
            "A_dl * exp(-k_gauss * (r - mu)^2)"
        )
        dl_force.addPerBondParameter("A_dl")       # amplitude (kJ/mol)
        dl_force.addPerBondParameter("k_gauss")     # = 1/(2σ²), controls width
        dl_force.addPerBondParameter("mu")           # target distance (nm)
        bsj_zone = max(50, L // 40)  # ±50 nt around the BSJ, or 1/40 of the sequence length
        for i, j, d_ij_a, w in dl_constraints:
            if w <= 0 or i >= L or j >= L:
                continue
            # Downweight the BSJ zone: residues located near position 0 or L
            pos_i = min(i, L - i)
            pos_j = min(j, L - j)
            if pos_i < bsj_zone or pos_j < bsj_zone:
                w *= 0.1  # BSJ-zone weight reduced 10x
            if w < 0.01:
                continue
            mu_nm = d_ij_a / 10.0
            # σ: high confidence → narrow (0.3Å), low confidence → wide (1.5Å)
            sigma_a = 0.3 + 1.2 * (1.0 - w)
            sigma_nm = sigma_a / 10.0
            k_gauss = 1.0 / (2.0 * sigma_nm ** 2)
            A_dl = 500.0 * w  # amplitude × confidence
            dl_force.addBond(B1(i), B1(j), [A_dl, k_gauss, mu_nm])
        if dl_force.getNumBonds() > 0:
            system.addForce(dl_force)

    return system, coords_nm, pair_force, bsj_force


def refine_5bead(
    p_coords: np.ndarray,
    pairs: List[Tuple[int, int, float]],
    platform_name: str = "CPU",
    stat_pot_path: Optional[str] = None,
    sequence: Optional[str] = None,
    dl_constraints: Optional[List[Tuple[int, int, float, float]]] = None,
):
    """Three-stage 5-bead CG annealing.

    Args:
        p_coords: (L, 3) initial P-only coordinates (Å)
        pairs: [(i, j, w)] ViennaRNA pairing
        platform_name: "CPU" or "CUDA"
        (no step-count argument: the three MD phases are 5000*_s, 3000*_s and 5000*_s
         steps at dt = 1 fs, with _s = max(1.0, L / 200.0); L = 2013 -> 130,845 steps total)

    Returns:
        (p_refined, e0, e1): refined P coordinates, initial energy, final energy
    """
    from openmm import LangevinMiddleIntegrator, Platform, unit
    from openmm.app import Simulation, Topology, Element

    L = len(p_coords)

    # ── Build system: intra bonds at full force constant (coordinates now fixed) ──
    system, coords_nm, pair_force, bsj_force = \
        build_5bead_system(p_coords, pairs, stat_pot_path=stat_pot_path,
                           sequence=sequence, dl_constraints=dl_constraints)

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

    # Locate the intra-residue bond force (so it can be ramped up in stages)
    intra_bond_force = None
    for fi in range(system.getNumForces()):
        f = system.getForce(fi)
        if hasattr(f, 'getNumBonds') and f.getNumBonds() == 4 * L:
            intra_bond_force = f
            break

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

    # Step counts auto-scale with the sequence length (L=200 → 1×, L=2009 → 10×)
    _s = max(1.0, L / 200.0)

    # Phase 0: pure minimize (no pairing/BSJ) so the intra bonds reach equilibrium
    set_pair_k(0.0)
    set_bsj_k(0.0)
    sim.minimizeEnergy(maxIterations=int(500000 * _s))
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy()._value
    pre_md = sim.context.getState(getPositions=True, getEnergy=True)

    # Phase 1: weak pairing + MD annealing (350K→300K) + minimize
    set_pair_k(0.1); set_bsj_k(0.0)
    sim.integrator.setTemperature(350 * unit.kelvin)
    sim.step(int(5000 * _s))
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.minimizeEnergy(maxIterations=int(300000 * _s))

    # Phase 2: strong pairing + weak BSJ + MD annealing (320K→300K) + minimize
    set_pair_k(1.0); set_bsj_k(0.1)
    sim.integrator.setTemperature(320 * unit.kelvin)
    sim.step(int(3000 * _s))
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.minimizeEnergy(maxIterations=int(400000 * _s))

    # Phase 3: strong pairing + strong BSJ + MD (300K) + minimize
    set_pair_k(1.0); set_bsj_k(1.0)
    sim.integrator.setTemperature(300 * unit.kelvin)
    sim.step(int(5000 * _s))
    sim.minimizeEnergy(maxIterations=int(500000 * _s))

    # Phase 4: final minimize (tight tolerance)
    sim.minimizeEnergy(tolerance=0.1 * unit.kilojoules_per_mole / unit.nanometer,
                       maxIterations=int(500000 * _s))

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value
    e1 = state.getPotentialEnergy()._value

    # Safety: only discard if the energy got 10x worse (a genuine blow-up)
    e_pre = pre_md.getPotentialEnergy()._value
    if e1 > e_pre * 10 and e_pre < 0:
        pos = pre_md.getPositions(asNumpy=True)._value
        e1 = e_pre

    # Extract P-only coordinates from 5-bead
    p_refined = (pos * 10.0)[0::5].copy()  # P bead every 5 atoms
    return p_refined, e0, e1
