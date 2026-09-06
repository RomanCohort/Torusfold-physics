"""
refine.py — Scheme 2 geometric initial coordinates + OpenMM coarse-grained refinement.

Path B: ViennaRNA pairing probabilities -> GeometricConstraintSolver geometry solve -> OpenMM energy minimization.
One particle per nucleotide (a P atom). Force field:
  1. HarmonicBondForce: adjacent P-P bonds (5.9Å) + BSJ head-to-tail bond (5.9Å) -> closure
  2. CustomBondForce: ViennaRNA pairing distance restraints (10.6Å)
  3. CustomNonbondedForce: soft repulsion (relieves steric clashes) + weak attraction (promotes folding)
  4. Energy minimization (L-BFGS); mid-size sequences (401-1000nt) additionally get MD annealing to escape local minima

Note: BOND_LEN = 5.9 is a P-O3'-P span approximation (CG: 1 particle per nucleotide).
The all-atom amber_field drop from +70k to -10k kJ/mol did NOT come from changing this value; it came
from aform_from_template rebuilding with a 1EHZ crystal template plus a three-stage amber_refine
refinement (loosen P -> anneal -> tighten P). A stale note once claimed "changed to 7.0 to align the
force field", but the code never changed it — 7.0 was a misannotation and has been removed
(corrected 2026-07-22).
"""

from __future__ import annotations
import numpy as np

from .constraint_solver import GeometricConstraintSolver, SolverConfig

# ---------- geometric parameters (Ang) ----------
BOND_LEN = 5.9      # P-P backbone distance (P-O3'-P span approximation, CG 1 particle per nucleotide)
PAIR_DIST = 10.6    # WC pair C1'-C1' (approximated by P here)
CLASH_DIST = 3.0    # minimum nonbonded distance


def _import_rna():
    """Lazily import ViennaRNA, giving a clear error when it is missing."""
    try:
        import RNA  # type: ignore
        return RNA
    except ImportError as e:
        raise ImportError(
            "ViennaRNA (RNA module) is not installed. Install it with: conda install -c bioconda viennarna"
        ) from e


def vienna_pair_probs(sequence: str, threshold: float = 0.3, circ: bool = True):
    """Compute the pairing probability matrix with ViennaRNA -> extract (i, j) pairing restraints.

    circ=True (default): ViennaRNA circular mode (VRNA_OPTION_CIRC), which correctly handles
    head-to-tail pairing of circular RNA. This project is a circRNA pipeline, so circular is the
    default. Pass circ=False explicitly to switch back to linear mode.

    Returns (pairs, bpp): pairs=[(i,j,prob)] is 0-indexed, and bpp is a 0-indexed L x L
    probability matrix.

    Note: ViennaRNA's fc.bpp() returns a 1-indexed (L+1) x (L+1) matrix ([0,:] is padding);
    we slice [1:,1:] here to get back to 0-indexing, consistent with the downstream coords.
    """
    RNA = _import_rna()
    md = RNA.md()
    md.circ = 1 if circ else 0
    fc = RNA.fold_compound(sequence, md)
    fc.pf()
    fc.mfe()
    bpp = np.array(fc.bpp())[1:, 1:]  # 1-indexed -> 0-indexed
    L = bpp.shape[0]
    pairs = []
    for i in range(L):
        for j in range(i + 4, L):
            if bpp[i, j] > threshold:
                pairs.append((i, j, float(bpp[i, j])))
    return pairs, bpp


def scheme2_initial_coords(sequence: str, pairs, n_samples: int = 8):
    """Produce Scheme 2 initial coarse coordinates (P atoms, Ang). Returns None on failure."""

    class CS:
        def __init__(self, L, p):
            self.seq_len = L
            self.pair_constraints = [(i, j, PAIR_DIST, w) for (i, j, w) in p]

    solver = GeometricConstraintSolver(SolverConfig(n_samples=n_samples))
    confs = solver.solve(CS(len(sequence), pairs))
    return confs[0] if confs else None


def predict_3d(sequence: str, n_samples: int = 8, platform_name: str = "CPU",
               pair_threshold: float = 0.3):
    """End-to-end: sequence -> 3D coordinates (N,3) Ang.

    Call chain: ViennaRNA pairing -> Scheme2 geometric solve -> openmm_refine.
    Returns a dict: {coords, pairs, e0, e1, bsj_before, bsj_after}.

    pair_threshold=0.3: the CG-only entry point also keeps weak pairs (the CG force field
    tolerates mismatches well). Contrast predict_3d_allatom, which uses 0.5 (feeding amber
    demands stricter pairs); see __init__.py.
    """
    pairs, _ = vienna_pair_probs(sequence, pair_threshold)
    init = scheme2_initial_coords(sequence, pairs, n_samples)
    if init is None:
        raise RuntimeError(f"Scheme2 geometry solve failed (L={len(sequence)})")
    bsj0 = float(np.linalg.norm(init[0] - init[-1]))
    refined, e0, e1 = openmm_refine(init, pairs, platform_name)
    bsj1 = float(np.linalg.norm(refined[0] - refined[-1]))
    return dict(coords=refined, pairs=pairs, e0=e0, e1=e1,
                bsj_before=bsj0, bsj_after=bsj1)


def build_topology(L: int):
    """OpenMM topology with one particle per nucleotide."""
    from openmm.app import Topology, Element
    topo = Topology()
    chain = topo.addChain()
    res_name = "N"  # generic residue
    for i in range(L):
        res = topo.addResidue(res_name, chain)
        topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)
    return topo


def openmm_refine(coords_angstrom: np.ndarray, pairs, platform_name: str = "CPU"):
    """Optimize (N,3) coordinates (Ang) with an OpenMM coarse-grained force field.

    pairs: [(i, j, weight), ...]
    Returns: (refined_coords (N,3) Ang, e0, e1)
    """
    from openmm import (
        System, Platform, VerletIntegrator, HarmonicBondForce,
        CustomBondForce, CustomNonbondedForce, LangevinMiddleIntegrator,
    )
    from openmm import unit
    from openmm.app import Simulation

    L = len(coords_angstrom)
    coords_nm = coords_angstrom / 10.0  # Ang -> nm

    topo = build_topology(L)

    system = System()
    for _ in range(L):
        system.addParticle(330.0)  # coarse-grained nucleotide mass (~330 Da)

    # 1. backbone bonds + BSJ bond (harmonic, nm)
    bond_k = 50000.0  # kJ/mol/nm^2 (strong, keeps closure)
    bond_force = HarmonicBondForce()
    for i in range(L - 1):
        bond_force.addBond(i, i + 1, BOND_LEN / 10.0, bond_k)
    bond_force.addBond(L - 1, 0, BOND_LEN / 10.0, bond_k)  # BSJ head-to-tail bond - enforces closure
    system.addForce(bond_force)

    # 2. pairing distance restraints (CustomBond, nm)
    pair_k = 2500.0
    pair_force = CustomBondForce("0.5*k*(r-r0)^2")
    pair_force.addPerBondParameter("k")
    pair_force.addPerBondParameter("r0")
    for (i, j, w) in pairs:
        if 0 <= i < L and 0 <= j < L and abs(i - j) > 1 and not (i == 0 and j == L - 1):
            pair_force.addBond(i, j, [pair_k * w, PAIR_DIST / 10.0])
    system.addForce(pair_force)

    # 3. soft repulsion + weak attraction (promote a 3D fold)
    nb = CustomNonbondedForce(
        "step(CLASH-r)*K_rep*(CLASH-r)^2 - K_attr*step(r-CLASH)*exp(-(r-CLASH)/sigma)"
    )
    nb.addGlobalParameter("CLASH", CLASH_DIST / 10.0)
    nb.addGlobalParameter("K_rep", 10000.0)   # set by ablation: strong repulsion (was 2000)
    nb.addGlobalParameter("K_attr", 1.0)
    nb.addGlobalParameter("sigma", 1.0)  # nm
    nb.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
    for i in range(L):
        nb.addParticle([])
    # exclude 1-2 pairs (adjacent + BSJ)
    for i in range(L - 1):
        nb.addExclusion(i, i + 1)
    nb.addExclusion(L - 1, 0)
    system.addForce(nb)

    # --- integrator + platform ---
    # Mid-size sequences (401-1000nt) use Langevin (supports the MD annealing below); the rest use Verlet (pure minimization)
    use_md = 401 <= L <= 1000
    integrator = (LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 0.002 * unit.picosecond)
                  if use_md else VerletIntegrator(0.001 * unit.picosecond))
    try:
        platform = Platform.getPlatformByName(platform_name)
        sim = Simulation(topo, system, integrator, platform)
    except Exception:
        sim = Simulation(topo, system, integrator)

    sim.context.setPositions(coords_nm * unit.nanometer)

    state0 = sim.context.getState(getEnergy=True)
    e0 = state0.getPotentialEnergy()._value

    sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=2000)

    # Mid-size sequences (401-1000nt) only: a light MD annealing pass to escape local minima
    # Ablation conclusion: short (<=400) and long (>1000) sequences need no MD; mid-size
    # sequences get stuck in local minima and need MD to break out
    # Best recipe: 100 steps @500K -> cool to 300K (50 steps) -> re-minimize. Heavier doses
    # (200+ steps / 800K+) over-scatter the structure instead
    # Safety net: MD occasionally diverges (clash count spikes); keep the pre-MD result and
    # take whichever has fewer clashes
    pre_md_state = sim.context.getState(getPositions=True, getEnergy=True) if use_md else None
    if use_md:
        sim.integrator.setTemperature(500 * unit.kelvin)
        sim.step(100)                       # high temperature breaks up local standoffs
        sim.integrator.setTemperature(300 * unit.kelvin)
        sim.step(50)                        # cool down
        sim.minimizeEnergy(tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer, maxIterations=2000)

    state = sim.context.getState(getPositions=True, getEnergy=True)
    pos = state.getPositions(asNumpy=True)._value  # nm
    e1 = state.getPotentialEnergy()._value

    # Safety net: if MD raised the energy sharply (divergence), fall back to the pre-MD pure-minimization result
    if use_md and pre_md_state is not None:
        e_pre = pre_md_state.getPotentialEnergy()._value
        if e1 > e_pre * 0.5 and e_pre < 0:
            pos = pre_md_state.getPositions(asNumpy=True)._value
            e1 = e_pre

    refined_angstrom = pos * 10.0  # nm -> Ang
    return refined_angstrom, e0, e1
