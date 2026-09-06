"""
physical_relaxation.py — physics-based relaxation post-processing

After segment assembly, quickly relax bond lengths and bond angles on torch GPU
to remove the geometric discontinuities at the assembly seams.
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

        bond_force = mm.HarmonicBondForce()
        for i in range(L - 1):
            bond_force.addBond(i, i + 1, 0.59, 31000.0)
        system.addForce(bond_force)

        if far_pairs:
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

    Force field (aligned with openmm_gpu_refiner._build_3bead_system_gpu):
      - Backbone P-P bond:     K_BB=500,     r0=0.59nm
      - BSJ closure:           K_BSJ=800,    r0=0.59nm
      - Backbone P-P-P angle:  K_ANGLE=600,  θ0=150°
      - P-P-P-P dihedral:      K_DIH=800,    θ0=33°
      - WC N-N pairing:        K_PAIR=1500,  r0=1.0nm
      - P(i)-P(i+2) stacking:  K_STACK=500,  r0=0.59nm
      - soft-sphere clash:     K_CLASH=5000
    """
    dev = torch.device("cuda")
    L = len(coords)

    # Force constants (nm units, aligned with the OpenMM version)
    K_BB = 5000.0   # bond: hard restraint, must be satisfied
    K_BSJ = 5000.0  # BSJ: treated like a bond
    K_ANGLE = 200.0  # angle: soft preference
    K_DIH = 200.0   # dihedral: soft preference
    K_PAIR = 500.0   # pairing: soft preference (much weaker than bonds)
    K_STACK = 0.0    # stacking: not restrained during relaxation
    K_CLASH = 5000.0
    BOND_R0 = 0.59       # nm
    PAIR_R0 = 1.0        # nm
    ANGLE_0 = 2.618      # rad (150°)
    DIH_0 = 33.0 * np.pi / 180.0  # rad

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

        # Stacking
        e_stack = torch.tensor(0.0, device=dev)
        if L >= 3:
            diff_st = p[stk_i] - p[stk_j]
            dist_st = diff_st.norm(dim=1)
            e_stack = (K_STACK * (dist_st - BOND_R0) ** 2).sum()

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
    target_bond = 5.9  # Å

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
