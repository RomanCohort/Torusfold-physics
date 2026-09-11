"""
physical_relaxation.py — physics-based relaxation post-processing

After segment assembly, quickly relax bond lengths and bond angles on torch GPU
to remove the geometric discontinuities at the assembly seams.

=====================================================================
CONVENTION — read this before copying a number out of this file
=====================================================================

Lengths are nanometres (nm), angles radians, energies kJ/mol.  The public API
takes Angstroms and divides by 10 once per path, so every r0 and every force
constant below is in nm.

The torch energy terms (the `_energy` closure in `_relax_torch_gpu`) are
SINGLE-SIDED harmonics:

    E = K * (x - x0)^2          <- NO 1/2

Three neighbouring fields write the 1/2 explicitly and were checked against the
code in the working tree: torch_cgsim.cg_energy (`e_bb = 0.5 * K_BB * (d_bb -
BOND_P_NEXT) ** 2`), openmm_gpu_refiner's CustomBondForce strings
("0.5*k_bsj*(r-r0)^2" and its siblings), and relax_structure's OWN OpenMM
branch below, whose HarmonicBondForce was measured on this box to return
0.5*k*(r-r0)^2 (k = 100 kJ/mol/nm^2, r-r0 = 0.05 / 0.10 / 0.20 nm ->
0.125 / 0.500 / 2.000 kJ/mol).

So the same numeral means TWICE the stiffness here as it does in those files,
and no number may be moved across that boundary without converting.  For scale
on the one coordinate both fields restrain, P(i)-P(i+1) at 0.59 nm: this file's
K_BB = 5000 is a curvature of 5000 kJ/mol/nm^2, while openmm_gpu_refiner.py
-- the file an earlier version of this module's docstring claimed to be aligned
with -- sits at 11.22 kJ/mol/A^2 * 100 * 0.5 = 561, i.e. 8.9x softer.

Whether the missing 1/2 was deliberate is NOT recorded.  This file's entire
history is two commits (an import and a comment translation), it carries no
note on the subject, and no constant here is a factor-of-two twin of one in a
neighbouring file: they are round hand-set restraint strengths (5000 / 5000 /
200 / 200 / 500) annotated "hard restraint, must be satisfied" and "soft
preference", never kBT/sigma^2.  What is certain is that the convention is
load-bearing and undocumented, that the alignment claim was false in both
numeral and form, and that the OpenMM branch of this same file sits on the
other side of the 1/2.  Halving a constant is a trajectory change for every
caller, so it is a decision, not a cleanup.

Numbers defined under `_relax_torch_gpu` (this table IS the code's digits;
tests/test_physical_relaxation_convention.py fails if the two drift apart):

| constant | value    | unit         | appears in                                          |
|----------|----------|--------------|-----------------------------------------------------|
| K_BB     | 5000.0   | kJ/mol/nm^2  | e_bb = (K_BB * (dist_bb - BOND_R0) ** 2).sum()       |
| K_BSJ    | 5000.0   | kJ/mol/nm^2  | e_bsj = K_BSJ * (d_bsj - BOND_R0) ** 2               |
| K_ANGLE  | 200.0    | kJ/mol/rad^2 | e_angle = (K_ANGLE * (angles - ANGLE_0) ** 2).sum()  |
| K_DIH    | 200.0    | kJ/mol/rad^2 | e_dih = (K_DIH * (dihedral - DIH_0) ** 2).sum()      |
| K_PAIR   | 500.0    | kJ/mol/nm^2  | e_pair = (K_PAIR * pr_w * (dist_pr - PAIR_R0) ** 2).sum() |
| K_STACK  | 0.0      | kJ/mol/nm^2  | e_stack = (K_STACK * (dist_st - BOND_R0) ** 2).sum(), and e_stack is NOT in the returned sum |
| K_CLASH  | 5000.0   | kJ/mol/nm^2  | nothing. Dead constant; the clash push-apart uses CLASH_R, not an energy |
| BOND_R0  | 0.59     | nm           | r0 of the backbone bond, the BSJ closure, and the stacking term |
| PAIR_R0  | 1.0      | nm           | r0 of the WC pairing term                            |
| ANGLE_0  | 2.618    | rad          | P-P-P target, 150 deg                                |
| DIH_0    | 0.5759587| rad          | 33 deg, applied to acos(n1.n2) between consecutive plane normals, which is unsigned and lives in [0, pi] |
| CLASH_R  | 0.4      | nm           | phase-3 push-apart trigger (4 A)                     |
| lr_full  | 0.000005 | nm^2 mol/kJ  | phase-2 gradient step size                           |

TWO OTHER PARAMETER SETS LIVE IN THIS SAME FILE, on OpenMM's with-1/2 side:

  - relax_structure's OpenMM branch: bond addBond(i, i+1, 0.59, 31000.0), a
    curvature of 15500 (3.1x the torch path's 5000); far pairs
    addBond(i, j, 1.0, 5000.0), a curvature of 2500.  The far-pair restraint
    exists ONLY here -- _relax_torch_gpu accepts far_pairs and ignores it, and
    is scheduled first whenever CUDA is available, so which restraint set a
    caller actually gets depends on torch.cuda.is_available().
  - _simple_relax (use_openmm=False): no force constant at all, only a
    0.5-damped correction toward 5.9 A -- the one pair of digits in this file
    that agrees with itself (5.9 A = BOND_R0).  No caller in this repository
    passes use_openmm=False, so this branch is reached only through the API.

WHO CALLS THIS MODULE (all through relax_structure; the torch path runs only
when torch.cuda.is_available(), otherwise the OpenMM branch runs):

    isrnaclong.py:1085         isrnaclong_pipeline        Level 1.5 global relaxation, n_steps=5000, pairs_all
    isrnaclong.py:1712         isrnaclong_pipeline        Level 2.5b post-CG->AA, n_steps=3000, far_pairs, no pairs_all
    segmented_vfold3d.py:1592  segmented_vfold3d_pipeline  cross-chunk post-relaxation, far_pairs, no pairs_all
    torch_gpu_refine.py:269    torch_gpu_refine           Level 2 refine, n_steps=5000, pairs_all
    __init__.py:227            predict_3d_allatom         relaxation after CG refinement (use_relaxation flag)
"""
import numpy as np
from typing import Dict, Optional, Tuple

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


def relax_structure(
    coords: np.ndarray,
    sequence: str,
    far_pairs=None,
    n_steps: int = 5000,
    use_openmm: bool = True,
    pairs_all=None,
) -> Tuple[np.ndarray, Dict]:
    """Physics-based relaxation: bond-length/bond-angle restrained minimization + short MD.

    Uses the torch GPU path (full CG force field), with OpenMM as a fallback.

    Dispatch, because the two branches are NOT the same force field:
    use_openmm=False -> _simple_relax; else, if CUDA is available ->
    _relax_torch_gpu (single-sided harmonics, NO 1/2 -- read the module header
    before reading one of its constants); else -> the OpenMM branch below, which
    uses OpenMM's convention (1/2 present) and different numerals.  far_pairs is
    honoured only on the OpenMM branch.

    Args:
        coords: (L, 3) P coordinates (Å)
        sequence: RNA sequence
        far_pairs: optional list of far pairs
        n_steps: number of minimization steps
        use_openmm: whether to use OpenMM (False = simple bond-length correction only)
        pairs_all: full pairing list [(i, j, weight), ...] (used by the torch GPU path)

    Returns:
        (relaxed_coords, metrics_dict)
    """
    L = len(coords)
    metrics = {
        "initial": {"clash_count": 0, "bond_violations": 0},
        "final": {"clash_count": 0, "bond_violations": 0},
    }

    if not use_openmm:
        return _simple_relax(coords, sequence)

    # Prefer the torch GPU path (full CG force field)
    if TORCH_OK and torch.cuda.is_available():
        return _relax_torch_gpu(coords, far_pairs, n_steps, metrics,
                                pairs_all=pairs_all)

    # OpenMM fallback
    try:
        import openmm as mm
        from openmm import unit
        from openmm.app import Simulation, Topology, Element
        from openmm import LangevinMiddleIntegrator, Platform

        topo = Topology()
        chain = topo.addChain()
        for i in range(L):
            res = topo.addResidue("RA", chain)
            topo.addAtom(f"P{i}", Element.getBySymbol("P"), res)

        system = mm.System()
        for _ in range(L):
            system.addParticle(100.0)

        # OpenMM branch: OpenMM's units and OpenMM's convention.  Its
        # HarmonicBondForce computes U = 0.5*k*(r-r0)^2 (measured here: k=100,
        # r-r0 = 0.05/0.10/0.20 nm -> 0.125/0.500/2.000 kJ/mol), so the 1/2 is
        # present on this branch and absent on the torch branch.  The two are not
        # the same parameter set and were not made to agree: 31000 kJ/mol/nm^2
        # here is a curvature of 15500, against the torch path's K_BB = 5000.
        bond_force = mm.HarmonicBondForce()
        for i in range(L - 1):
            bond_force.addBond(i, i + 1, 0.59, 31000.0)
        system.addForce(bond_force)

        if far_pairs:
            # Far-pair restraint: 5000 kJ/mol/nm^2 with the 1/2 -> 2500 of
            # curvature, at r0 = 1.0 nm.  This term exists ONLY on this branch;
            # _relax_torch_gpu takes far_pairs and ignores it.
            pair_force = mm.HarmonicBondForce()
            for (i, j) in far_pairs:
                if 0 <= i < L and 0 <= j < L and abs(i - j) > 1:
                    pair_force.addBond(i, j, 1.0, 5000.0)
            system.addForce(pair_force)

        coords_nm = coords.copy() / 10.0
        plat_name = "CPU"
        for try_name in ["CUDA", "OpenCL"]:
            try:
                Platform.getPlatformByName(try_name)
                plat_name = try_name
                break
            except Exception:
                pass
        plat = Platform.getPlatformByName(plat_name)

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        sim = Simulation(topo, system, integrator, plat)
        sim.context.setPositions(coords_nm * unit.nanometer)

        state = sim.context.getState(getPositions=True)
        pos = state.getPositions(asNumpy=True)._value * 10.0
        metrics["initial"]["clash_count"] = _count_clashes(pos)

        sim.minimizeEnergy(
            tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=n_steps)
        sim.step(min(2000, n_steps // 5))
        sim.minimizeEnergy(
            tolerance=10.0 * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=n_steps // 2)

        state = sim.context.getState(getPositions=True)
        relaxed = state.getPositions(asNumpy=True)._value * 10.0
        metrics["final"]["clash_count"] = _count_clashes(relaxed)
        metrics["final"]["bond_violations"] = _count_bond_violations(relaxed)
        return relaxed, metrics

    except Exception:
        # OpenMM failed; try the torch GPU fallback
        if TORCH_OK and torch.cuda.is_available():
            return _relax_torch_gpu(coords, far_pairs, n_steps, metrics)
        return _simple_relax(coords, sequence)


def _relax_torch_gpu(
    coords: np.ndarray,
    far_pairs=None,
    n_steps: int = 5000,
    metrics: dict = None,
    pairs_all=None,
) -> Tuple[np.ndarray, Dict]:
    """torch GPU relaxation: full CG force field + gradient-descent minimization.

    Functional form -- SINGLE-SIDED harmonics, no 1/2 (module header has units,
    the comparison against the neighbouring fields, and the warning):

        E = K_BB    * (|P(i)-P(i+1)| - BOND_R0)^2          over backbone bonds
          + K_BSJ   * (|P(0)-P(L-1)| - BOND_R0)^2          BSJ closure
          + K_ANGLE * (theta(j) - ANGLE_0)^2                P-P-P angle at j, rad
          + K_DIH   * (phi(i) - DIH_0)^2                    unsigned angle between
                                                            consecutive plane
                                                            normals, rad
          + K_PAIR  * w(i,j) * (|P(i)-P(j)| - PAIR_R0)^2    WC pairs from pairs_all

    The constants below are the ones this function defines and uses.  They are
    not the numbers of openmm_gpu_refiner._build_3bead_system_gpu, even though
    an earlier version of this docstring said they were: it listed K_BB=500,
    K_BSJ=800, K_ANGLE=600, K_DIH=800, K_PAIR=1500, K_STACK=500, K_CLASH=5000
    -- stale numerals belonging to torch_cgsim.py and the OpenMM refiner, none
    of which this file has ever computed with (verified back to the initial
    commit).

    far_pairs is accepted for signature compatibility and IGNORED here: the
    far-pair restraint is implemented only on relax_structure's OpenMM branch.
    A caller that passes far_pairs and lands on this branch gets no far-pair
    term at all (isrnaclong.py:1712, Level 2.5b, is such a caller).
    """
    dev = torch.device("cuda")
    L = len(coords)

    # Force constants: distances in nm, angles in rad, energies in kJ/mol, and
    # SINGLE-SIDED -- E = K*(x-x0)^2 with NO 1/2 (module header has the full warning).
    # These are hand-set restraint strengths for a geometric cleanup, not measured
    # springs: "hard restraint" and "soft preference" below are the author's own words,
    # and none of these is a kBT/sigma^2 value.  The old comment here, "aligned with the
    # OpenMM version", is false in both directions -- the OpenMM branch above uses 31000
    # and 5000 with the 1/2, and openmm_gpu_refiner.py (the file the docstring claimed)
    # uses 11.22 kJ/mol/A^2.  A difference is evidence, not an error to smooth away.
    K_BB = 5000.0   # bond: hard restraint, must be satisfied
    K_BSJ = 5000.0  # BSJ: treated like a bond
    K_ANGLE = 200.0  # angle: soft preference  [kJ/mol/rad^2]
    K_DIH = 200.0   # dihedral: soft preference  [kJ/mol/rad^2]
    K_PAIR = 500.0   # pairing: soft preference (much weaker than bonds); scaled by pr_w
    K_STACK = 0.0    # stacking: not restrained during relaxation.  Dead twice over:
                     # e_stack below is also missing from the returned energy.
    K_CLASH = 5000.0  # DEAD CONSTANT: referenced by nothing in this file.  Clash is
                      # handled by the literal push-apart loop in phase 3 (CLASH_R),
                      # not by an energy term, despite the name and the docstring.
    BOND_R0 = 0.59       # nm
    PAIR_R0 = 1.0        # nm
    ANGLE_0 = 2.618      # rad (150°)
    DIH_0 = 33.0 * np.pi / 180.0  # rad (0.5759587); applied to acos(n1.n2), which is
                                  # unsigned and in [0, pi], not a signed torsion

    pos = torch.tensor(coords / 10.0, dtype=torch.float64, device=dev)

    # ── Precompute indices ──
    # Backbone bonds
    bb_i = torch.arange(L - 1, device=dev)
    bb_j = torch.arange(1, L, device=dev)
    # BSJ
    has_bsj = L >= 3
    # Backbone angles
    if L >= 3:
        ang_i = torch.arange(L - 2, device=dev)
        ang_j = torch.arange(1, L - 1, device=dev)
        ang_k = torch.arange(2, L, device=dev)
    # Dihedrals (four consecutive residues a,b,c,d)
    if L >= 4:
        n_dih = L - 3
        dih_a = torch.arange(n_dih, device=dev)
        dih_b = torch.arange(1, n_dih + 1, device=dev)
        dih_c = torch.arange(2, n_dih + 2, device=dev)
        dih_d = torch.arange(3, n_dih + 3, device=dev)
    # Stacking P(i)-P(i+2)
    if L >= 3:
        stk_i = torch.arange(L - 2, device=dev)
        stk_j = torch.arange(2, L, device=dev)

    # WC pairings (from pairs_all, those with weight > 0)
    if pairs_all:
        valid_pairs = [(i, j, w) for (i, j, w) in pairs_all
                       if 0 <= i < L and 0 <= j < L
                       and abs(i - j) > 1
                       and not (i == 0 and j == L - 1)]
        if valid_pairs:
            pi, pj, pw = zip(*valid_pairs)
            pr_i = torch.tensor(pi, device=dev, dtype=torch.long)
            pr_j = torch.tensor(pj, device=dev, dtype=torch.long)
            pr_w = torch.tensor(pw, device=dev, dtype=torch.float64)
        else:
            pr_i = torch.zeros(0, device=dev, dtype=torch.long)
            pr_j = torch.zeros(0, device=dev, dtype=torch.long)
            pr_w = torch.zeros(0, device=dev, dtype=torch.float64)
    else:
        pr_i = torch.zeros(0, device=dev, dtype=torch.long)
        pr_j = torch.zeros(0, device=dev, dtype=torch.long)
        pr_w = torch.zeros(0, device=dev, dtype=torch.float64)

    def _energy(p):
        """Full CG potential energy."""
        # Backbone bonds
        diff_bb = p[bb_i] - p[bb_j]
        dist_bb = diff_bb.norm(dim=1)
        e_bb = (K_BB * (dist_bb - BOND_R0) ** 2).sum()

        # BSJ
        e_bsj = torch.tensor(0.0, device=dev)
        if has_bsj:
            d_bsj = (p[0] - p[L - 1]).norm()
            e_bsj = K_BSJ * (d_bsj - BOND_R0) ** 2

        # Backbone angles
        e_angle = torch.tensor(0.0, device=dev)
        if L >= 3:
            v1 = p[ang_j] - p[ang_i]
            v2 = p[ang_k] - p[ang_j]
            n1 = v1.norm(dim=1).clamp(min=1e-8)
            n2 = v2.norm(dim=1).clamp(min=1e-8)
            cos_a = (v1 * v2).sum(dim=1) / (n1 * n2)
            cos_a = cos_a.clamp(-1 + 1e-6, 1 - 1e-6)
            angles = torch.acos(cos_a)
            e_angle = (K_ANGLE * (angles - ANGLE_0) ** 2).sum()

        # Dihedrals
        e_dih = torch.tensor(0.0, device=dev)
        if L >= 4:
            b1 = p[dih_b] - p[dih_a]
            b2 = p[dih_c] - p[dih_b]
            b3 = p[dih_d] - p[dih_c]
            n1 = torch.cross(b1, b2, dim=1)
            n2 = torch.cross(b2, b3, dim=1)
            n1_norm = n1.norm(dim=1).clamp(min=1e-8)
            n2_norm = n2.norm(dim=1).clamp(min=1e-8)
            cos_d = (n1 * n2).sum(dim=1) / (n1_norm * n2_norm)
            cos_d = cos_d.clamp(-1 + 1e-6, 1 - 1e-6)
            dihedral = torch.acos(cos_d)
            e_dih = (K_DIH * (dihedral - DIH_0) ** 2).sum()

        # WC pairings
        e_pair = torch.tensor(0.0, device=dev)
        if pr_i.numel() > 0:
            diff_pr = p[pr_i] - p[pr_j]
            dist_pr = diff_pr.norm(dim=1)
            e_pair = (K_PAIR * pr_w * (dist_pr - PAIR_R0) ** 2).sum()

        # Stacking.  NOTE: e_stack is built and then NOT returned -- it is absent from
        # the sum below, which is why K_STACK = 0.0 is dead twice over.  Keeping the
        # omission visible matters: switching K_STACK back on would otherwise look like
        # it re-enables a restraint while changing no energy at all.
        e_stack = torch.tensor(0.0, device=dev)
        if L >= 3:
            diff_st = p[stk_i] - p[stk_j]
            dist_st = diff_st.norm(dim=1)
            e_stack = (K_STACK * (dist_st - BOND_R0) ** 2).sum()

        # e_stack deliberately absent (see the comment above).
        return e_bb + e_bsj + e_angle + e_dih + e_pair

    # Initial metrics
    with torch.no_grad():
        pos_np = pos.cpu().numpy() * 10.0
        if metrics:
            metrics["initial"]["clash_count"] = _count_clashes(pos_np)

    # ── Phase 1: selective bond correction (fix only the most deviated bonds, keeping the 3D structure) ──
    with torch.no_grad():
        p = pos.clone()
        for _ in range(50):
            diff = p[bb_j] - p[bb_i]
            dist = diff.norm(dim=1)                      # (L-1,)
            bad = (dist - BOND_R0).abs() > 0.05          # bonds deviating by more than 0.05 nm
            if not bad.any():
                break
            # Fix only the bad bonds: put atom j at distance BOND_R0 from atom i (along the original direction)
            delta = diff[bad]
            d_norm = delta.norm(dim=1, keepdim=True).clamp(min=1e-8)
            correction = (BOND_R0 - d_norm) / d_norm
            # Gentle correction (50% weight, to avoid overshoot)
            p[bb_j[bad]] = p[bb_i[bad]] + delta * (1 + correction * 0.5)
        pos = p

    # ── Phase 1b: BSJ closure (gentle) ──
    if has_bsj:
        with torch.no_grad():
            p = pos.clone()
            for _ in range(50):
                d = p[0] - p[L - 1]
                dist = d.norm().clamp(min=1e-8)
                if abs(dist.item() - BOND_R0) < 0.05:
                    break
                corr = (BOND_R0 - dist.item()) / dist.item() * 0.3
                p[0] -= d * corr
                p[L - 1] += d * corr
            pos = p

    # ── Phase 2: gradient descent with a tiny step (energy minimization that does not change the structure) ──
    pos_g = pos.detach().requires_grad_(True)
    lr_full = 0.000005
    for step in range(max(50, min(n_steps // 10, 200))):
        pos_g.grad = None
        e = _energy(pos_g)
        e.backward()
        with torch.no_grad():
            gn = pos_g.grad.norm()
            pos_g -= pos_g.grad * lr_full
            pos_g.data.clamp_(-5.0, 5.0)
        if gn < 1e-5:
            break
    pos = pos_g.detach()

    # ── Phase 3: soft-sphere repulsion (iterative push-apart + bond re-calibration) ──
    CLASH_R = 0.4  # nm (4Å, looser than the 3Å used elsewhere)
    # Reading metrics['final']['clash_count'] alongside this: phase 3 pushes apart every
    # pair with |i-j| >= 2 closer than CLASH_R (0.4 nm), while the metric is
    # _count_clashes(): 0.3 nm, |i-j| >= 3, windowed to j < i+30.  So a structure that
    # hits the 100-iteration cap can stay 0.3-0.4 nm dirty without the metric saying so,
    # and a clash further apart than 30 residues is never counted at all.
    with torch.no_grad():
        p = pos.clone()
        for iteration in range(100):
            diff_all = p[:, None] - p[None, :]
            dist_all = diff_all.norm(dim=2)
            mask = torch.triu(torch.ones(L, L, device=dev, dtype=bool), diagonal=2)
            near_mask = torch.zeros(L, L, device=dev, dtype=bool)
            near_mask[bb_i, bb_j] = True; near_mask[bb_j, bb_i] = True
            clash_mask = mask & ~near_mask & (dist_all < CLASH_R)
            if not clash_mask.any():
                break
            ci, cj = torch.nonzero(clash_mask, as_tuple=True)
            d = p[ci] - p[cj]
            dist = dist_all[ci, cj].clamp(min=1e-8)
            push = (CLASH_R - dist) / dist
            disp = torch.zeros_like(p)
            disp.index_add_(0, ci, d * push[:, None] * 0.5)
            disp.index_add_(0, cj, -d * push[:, None] * 0.5)
            p = p + disp
            # Bond re-calibration (only fix bonds whose endpoints shifted a lot, not the whole chain)
            diff_b = p[bb_j] - p[bb_i]
            dist_b = diff_b.norm(dim=1, keepdim=True).clamp(min=1e-8)
            bad = (dist_b - BOND_R0).abs() > 0.01  # only fix bonds deviating by more than 0.01 nm
            if bad.any():
                corr = (BOND_R0 - dist_b) / dist_b * 0.3
                disp2 = torch.zeros_like(p)
                disp2.index_add_(0, bb_i[bad.squeeze()], (-diff_b[bad.squeeze()] * corr[bad.squeeze()]))
                disp2.index_add_(0, bb_j[bad.squeeze()], (diff_b[bad.squeeze()] * corr[bad.squeeze()]))
                p = p + disp2
        pos = p

    relaxed = pos.cpu().numpy() * 10.0
    if metrics:
        metrics["final"]["clash_count"] = _count_clashes(relaxed)
        metrics["final"]["bond_violations"] = _count_bond_violations(relaxed)
    return relaxed, metrics or {}


def _simple_relax(coords: np.ndarray, sequence: str) -> Tuple[np.ndarray, Dict]:
    """Simple bond-length correction (fallback when OpenMM is unavailable)."""
    relaxed = coords.copy()
    target_bond = 5.9  # Å -- the same distance as BOND_R0 = 0.59 nm on the other paths

    for iteration in range(10):
        for i in range(len(relaxed) - 1):
            diff = relaxed[i + 1] - relaxed[i]
            dist = np.linalg.norm(diff)
            if dist > 0:
                correction = (dist - target_bond) / dist * 0.5
                relaxed[i] += diff * correction
                relaxed[i + 1] -= diff * correction

    metrics = {
        "initial": {"clash_count": _count_clashes(coords), "bond_violations": 0},
        "final": {"clash_count": _count_clashes(relaxed), "bond_violations": _count_bond_violations(relaxed)},
    }
    return relaxed, metrics


def _count_clashes(coords: np.ndarray, threshold: float = 3.0) -> int:
    """Count clashes (P-P distances below threshold Å, excluding adjacent residues)."""
    L = len(coords)
    count = 0
    for i in range(L):
        for j in range(i + 3, min(i + 30, L)):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < threshold:
                count += 1
    return count


def _count_bond_violations(coords: np.ndarray, target: float = 5.9, tol: float = 1.0) -> int:
    """Count bond-length violations (deviation from target beyond tol Å)."""
    count = 0
    for i in range(len(coords) - 1):
        d = np.linalg.norm(coords[i + 1] - coords[i])
        if abs(d - target) > tol:
            count += 1
    return count
