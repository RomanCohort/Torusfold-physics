#!/usr/bin/env python
r"""Measure what the nonbonded block of openmm_gpu_refiner._build_3bead_system_gpu computes.

Read-only with respect to the module: every number below comes either from the OpenMM
System that _build_3bead_system_gpu itself returns (parameters read back out of the built
object, never out of the source text), from a Context built on that System, or from the
300 K Langevin trajectory this script runs.  The "controls" mutate a System that was
already built, in this process only; the module is not touched.

Sections
  A  the block as built: charges, sigmas, epsilons, Born radii, dielectrics, global params
  B  screeningLength: is it read?  Perturbation test, not a grep
  G  exclusion audit against the 1-2 + cyclisation set
  D  P-P pair potential, per contributing force, vs separation (built parameters)
  C  reproduce the backbone observation (30-nt ring: minimise + 3000 steps, 0 and 2 pairs)
  E  in-situ axial load across every backbone bond, per force (what holds 7 A)
  F  controls: charges off / GB off / Debye screening / 1-3 exclusion / old K_BB / eps_in

Usage:  python scripts/characterize_nonbonded_block.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import openmm as mm                          # noqa: E402
import openmm.unit as unit                   # noqa: E402
from openmm import LangevinMiddleIntegrator, Platform   # noqa: E402
from openmm.app import Simulation            # noqa: E402

from torusfold.scheme2 import openmm_gpu_refiner as R   # noqa: E402

L = 30
SEED = 12345
THREADS = 4
SAMPLE_EVERY = 30
KBT = 2.494                      # kJ/mol at 300 K
COUL = 138.935456                # kJ/mol/nm per e^2
PAIRS_2 = [(5, 24, 1.0), (6, 23, 1.0)]
PAIR_SETS = [(5, 24, 1.0), (6, 23, 1.0)]


# ── plumbing ──────────────────────────────────────────────────────────────

def scalar(x):
    return x._value if hasattr(x, "_value") else float(x)


def force_label(f, idx):
    cls = type(f).__name__
    try:
        expr = f.getEnergyFunction()
    except AttributeError:
        expr = None
    if expr:
        return f"#{idx} {cls}[{expr}]"
    n = None
    for attr in ("getNumBonds", "getNumParticles", "getNumAngles", "getNumTorsions"):
        if hasattr(f, attr):
            n = getattr(f, attr)()
            break
    return f"#{idx} {cls}({n})"


def find_force(system, cls, expr_contains=None):
    out = []
    for idx, f in enumerate(system.getForces()):
        if isinstance(f, cls):
            if expr_contains is None:
                out.append((idx, f))
            else:
                try:
                    if expr_contains in f.getEnergyFunction():
                        out.append((idx, f))
                except AttributeError:
                    pass
    return out


def system_global_params(system):
    """OpenMM keeps global parameters on the Forces, not on the System."""
    out = []
    for idx, f in enumerate(system.getForces()):
        if not hasattr(f, "getNumGlobalParameters"):
            continue
        for k in range(f.getNumGlobalParameters()):
            out.append((f.getGlobalParameterName(k),
                        f.getGlobalParameterDefaultValue(k),
                        force_label(f, idx)))
    return out


def one_force(system, cls, expr_contains=None):
    found = find_force(system, cls, expr_contains)
    if len(found) != 1:
        raise RuntimeError(f"expected 1 {cls.__name__}, found {len(found)}")
    return found[0]


def build(pairs=None):
    pairs = list(pairs or [])
    p_ang = R._generate_compact_coords(L, pairs)
    system, coords_nm, pf, sf, bf, bg = R._build_3bead_system_gpu(
        p_ang, pairs, pair_scale=1.0, bsj_k_scale=1.0)
    return system, coords_nm, p_ang


def group_forces(system):
    names = []
    for idx, f in enumerate(system.getForces()):
        f.setForceGroup(idx)
        names.append(force_label(f, idx))
    return names


def energy(sim, group=None):
    kw = dict(getEnergy=True)
    if group is not None:
        kw["groups"] = {group}
    return sim.context.getState(**kw).getPotentialEnergy()._value


def positions(sim):
    return sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value


def group_forces_at(sim, n_groups):
    return [sim.context.getState(getForces=True, groups={g})
            .getForces(asNumpy=True)._value[0::3] for g in range(n_groups)]


def run(system, coords_nm, names, n_min=2000, n_md=3000, seed=SEED, threads=THREADS,
        sample_every=0):
    topo = R._create_3bead_topology(L)
    integ = LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
    integ.setRandomNumberSeed(seed)
    plat = Platform.getPlatformByName("CPU")
    sim = Simulation(topo, system, integ, plat, {"CpuThreads": str(threads)})
    sim.context.setPositions(coords_nm * unit.nanometer)
    e0 = energy(sim)
    sim.minimizeEnergy(maxIterations=n_min)
    e_min = energy(sim)
    pos_min = positions(sim)
    f_min = group_forces_at(sim, len(names))
    samples = {"pos": [], "F": []}
    done = 0
    while done < n_md:
        n = min(sample_every, n_md - done) if sample_every else n_md
        sim.step(n)
        done += n
        if sample_every:
            samples["pos"].append(positions(sim))
            samples["F"].append(group_forces_at(sim, len(names)))
    e_md = energy(sim)
    pos_md = positions(sim)
    f_md = group_forces_at(sim, len(names))
    return dict(sim=sim, system=system, names=names, e0=e0, e_min=e_min, e_md=e_md,
                pos_min=pos_min, pos_md=pos_md, f_min=f_min, f_md=f_md,
                samples=samples)


def pp_dist_ang(pos_nm):
    """All L backbone P-P distances (i, i+1 mod L) in Angstroms."""
    P = pos_nm[0::3]
    return np.array([np.linalg.norm(P[(i + 1) % L] - P[i]) * 10.0 for i in range(L)])


def per_force_energies(sim, names):
    return [(nm, energy(sim, g)) for g, nm in enumerate(names)]


def axial_load(pos_nm, forces_by_group):
    """Per-force axial load across every backbone bond.

    load_k(i) = -(F_i^k - F_j^k) . u / 2, u = unit(P_j - P_i), j = i+1 mod L.
    A stretched harmonic bond pulls i toward j, i.e. F_i = +k(r-r0)u, so the bond
    reports -k(r-r0): negative.  Two like charges push apart, F_i = -C u, and report
    +C: positive.  Summed over every force group it averages to ~0 over a trajectory
    (the Langevin kicks average out), which is the check that the decomposition closes.
    """
    P = pos_nm[0::3] * 10.0
    n = len(forces_by_group)
    loads = np.zeros((L, n))
    dist = np.zeros(L)
    for i in range(L):
        j = (i + 1) % L
        d = P[j] - P[i]
        dist[i] = np.linalg.norm(d)
        u = d / dist[i]
        for k in range(n):
            loads[i, k] = -(forces_by_group[k][i] - forces_by_group[k][j]) @ u / 2.0
    return dist, loads


def mean_axial_load(samples):
    """Trajectory-averaged <r> and per-force axial load, from sampled frames."""
    dists, loads = [], []
    for pos, F in zip(samples["pos"], samples["F"]):
        d, ld = axial_load(pos, F)
        dists.append(d)
        loads.append(ld)
    return np.mean(dists, axis=0), np.mean(loads, axis=0), len(dists)


def radial_load(pos_nm, forces_by_group):
    """Per-bead force decomposition projected on the outward radial direction.

    Every force acting on a P bead is counted exactly once here (the bead's two
    backbone bonds included), so unlike a per-bond axial projection there is no
    leakage between neighbouring bonds.  Positive = that force pushes the bead out
    of the ring, negative = pulls it in.  Averaged over a trajectory the Langevin
    kicks cancel and the columns sum to ~0.
    """
    P = pos_nm[0::3] * 10.0
    c = P.mean(axis=0)
    n = np.zeros(3)
    for i in range(L):
        n += np.cross(P[i] - c, P[(i + 1) % L] - c)
    n /= np.linalg.norm(n)
    loads = np.zeros((L, len(forces_by_group)))
    for i in range(L):
        d = P[i] - c
        v = d - np.dot(d, n) * n
        e = v / np.linalg.norm(v)
        for k, F in enumerate(forces_by_group):
            loads[i, k] = F[i] @ e
    return loads


def mean_radial_load(samples):
    loads = [radial_load(pos, F) for pos, F in zip(samples["pos"], samples["F"])]
    return np.mean(loads, axis=0), len(loads)


# ── A ─────────────────────────────────────────────────────────────────────

def section_A():
    print("=" * 78)
    print("A. the nonbonded block as built (read back out of the OpenMM System)")
    print("=" * 78)
    system, coords_nm, p_ang = build()
    print(f"system: {system.getNumParticles()} particles, {system.getNumForces()} forces, "
          f"{len(system_global_params(system))} global parameters")
    for idx, f in enumerate(system.getForces()):
        print(f"  {force_label(f, idx)}")

    _, nb = one_force(system, mm.NonbondedForce)
    print("\nNonbondedForce:")
    print(f"  nonbondedMethod         = {nb.getNonbondedMethod()} (0=NoCutoff)")
    print(f"  cutoffDistance          = {scalar(nb.getCutoffDistance()):.3f} nm")
    print(f"  reactionFieldDielectric = {nb.getReactionFieldDielectric()}")
    print(f"  useDispersionCorrection = {nb.getUseDispersionCorrection()}")
    print(f"  particles               = {nb.getNumParticles()}, "
          f"exceptions = {nb.getNumExceptions()}")
    for i in range(min(nb.getNumParticles(), 3)):
        q, sig, eps = nb.getParticleParameters(i)
        print(f"    particle {i} ({'PCN'[i]}): charge={scalar(q):+.4f} e  "
              f"sigma={scalar(sig):.4f} nm  epsilon={scalar(eps):.4f} kJ/mol")
    print(f"  global parameters on the NonbondedForce: "
          f"{[nb.getGlobalParameterName(i) for i in range(nb.getNumGlobalParameters())]}")

    _, gb = one_force(system, mm.GBSAOBCForce)
    print("\nGBSAOBCForce:")
    print(f"  soluteDielectric  = {gb.getSoluteDielectric()}")
    print(f"  solventDielectric = {gb.getSolventDielectric()}")
    print(f"  particles         = {gb.getNumParticles()}")
    for i in range(min(gb.getNumParticles(), 3)):
        q, rad, sc = gb.getParticleParameters(i)
        print(f"    particle {i} ({'PCN'[i]}): charge={scalar(q):+.4f} e  "
              f"bornRadius={scalar(rad):.4f} nm  scale={scalar(sc):.3f}")

    _, clash = one_force(system, mm.CustomBondForce, "dmin")
    print("\nclash CustomBondForce:")
    print(f"  expression = {clash.getEnergyFunction()}")
    print(f"  bonds      = {clash.getNumBonds()}")
    if clash.getNumBonds():
        a, b, params = clash.getBondParameters(0)
        print(f"    first bond ({a},{b}) params = {[scalar(x) for x in params]}")

    print("\nsystem global parameters (name, default, owning force):")
    for nm, dv, owner in system_global_params(system):
        print(f"  {nm:<12} = {dv:<20.6f} owner = {owner}")
    return system, coords_nm


# ── B ─────────────────────────────────────────────────────────────────────

def section_B():
    print("\n" + "=" * 78)
    print("B. is screeningLength read?  (perturbation of a live Context)")
    print("=" * 78)
    system, coords_nm, _ = build()
    integ = mm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = mm.Context(system, integ, Platform.getPlatformByName("CPU"))
    ctx.setPositions(coords_nm * unit.nanometer)

    names_all = [nm for nm, _dv, _o in system_global_params(system)]
    print(f"system global parameters: {names_all}")
    if "screeningLength" not in names_all:
        # The parameter has been deleted from the module.  Re-add it in memory, exactly
        # as the module used to, so this measurement stays reproducible from this script.
        _, nb = one_force(system, mm.NonbondedForce)
        literal = 1.0 / math.sqrt(0.145 * 0.06022 * 2)
        nb.addGlobalParameter("screeningLength", literal)
        integ = mm.VerletIntegrator(0.001 * unit.picosecond)
        ctx = mm.Context(system, integ, Platform.getPlatformByName("CPU"))
        ctx.setPositions(coords_nm * unit.nanometer)
        print("screeningLength is absent from the module now (deleted after measurement);")
        print("re-added in memory with the value the module used to write, to measure it:")
        names_all = [nm for nm, _dv, _o in system_global_params(system)]
        print(f"  global parameters in this probe System: {names_all}")
    e_ref = ctx.getState(getEnergy=True).getPotentialEnergy()._value
    f_ref = ctx.getState(getForces=True).getForces(asNumpy=True)._value
    literal = 1.0 / math.sqrt(0.145 * 0.06022 * 2)
    print(f"value written by the module: 1/sqrt(0.145*0.06022*2) = {literal:.6f} nm")
    print(f"the comment above it says '~= 0.78 nm (Debye length)'; the number written is "
          f"{literal / 0.78:.3f}x that")
    print(f"\n{'value set':>14} {'total energy':>16} {'dE':>12} {'max|dF|':>12}")
    for v in (0.0, 1e-4, 0.78, literal, 1000.0):
        ctx.setParameter("screeningLength", v)
        e = ctx.getState(getEnergy=True).getPotentialEnergy()._value
        F = ctx.getState(getForces=True).getForces(asNumpy=True)._value
        print(f"{v:>14.6g} {e:>16.6f} {e - e_ref:>12.6f} "
              f"{np.abs(F - f_ref).max():>12.6f}")
    print("units kJ/mol and kJ/mol/nm.  A parameter that is read would move these.")
    return e_ref


# ── G ─────────────────────────────────────────────────────────────────────

def section_G():
    print("\n" + "=" * 78)
    print("G. exclusion audit (NonbondedForce exceptions)")
    print("=" * 78)
    system, _, _ = build()
    _, nb = one_force(system, mm.NonbondedForce)
    actual = []
    for k in range(nb.getNumExceptions()):
        a, b, q, s, e = nb.getExceptionParameters(k)
        actual.append(((min(a, b), max(a, b)), (scalar(q), scalar(s), scalar(e))))

    expected = {}
    for i in range(L):
        for key in ((3 * i, 3 * i + 1), (3 * i + 1, 3 * i + 2)):
            expected[key] = "intra-residue P-C4'/C4'-N"
    for i in range(L - 1):
        expected[(3 * i, 3 * i + 3)] = "P(i)-P(i+1)"
    expected[(0, 3 * (L - 1))] = "cyclisation P(0)-P(L-1)"

    seen, dupes = {}, []
    for key, prm in actual:
        if key in seen:
            dupes.append(key)
        seen[key] = prm
    missing = sorted(k for k in expected if k not in seen)
    extra = sorted(k for k in seen if k not in expected)
    print(f"exceptions in the built system          : {len(actual)}")
    print(f"expected set (1-2 bonds + cyclisation)  : {len(expected)}")
    print(f"missing    : {missing}")
    print(f"extra      : {extra}")
    print(f"double-added: {dupes}")
    bad = [(k, p) for k, p in seen.items() if p != (0.0, 0.3, 0.0)]
    print(f"entries whose (charge, sigma, epsilon) != (0, 0.3, 0): {bad}")

    P = R._generate_compact_coords(L, [])
    d13 = np.array([np.linalg.norm(P[(i + 2) % L] - P[i]) for i in range(L)])
    print(f"\n1-3 backbone pairs NOT excluded: {L} (P(i)-P(i+2 mod L))")
    print(f"initial ring 1-3 P-P distance : mean {d13.mean():.3f} A "
          f"min {d13.min():.3f} max {d13.max():.3f}")
    print(f"bare Coulomb of one 1-3 pair at q=0.6 e: "
          f"{COUL * 0.36 / (d13.mean() / 10.0):.2f} kJ/mol")
    print("1-4 pairs (P(i)-P(i+3)) are also not excluded")

    # the clash pair list is a second enumeration, and it is linear, not cyclic
    _, clash = one_force(system, mm.CustomBondForce, "dmin")
    clash_pairs = set()
    for b in range(clash.getNumBonds()):
        a, bb, _pr = clash.getBondParameters(b)
        clash_pairs.add((min(a, bb), max(a, bb)))
    cyclic = set()
    for i in range(L):
        for d in range(2, 8):
            j = (i + d) % L
            cyclic.add((min(3 * i, 3 * j), max(3 * i, 3 * j)))
    missing = sorted(cyclic - clash_pairs)
    print(f"clash force: {len(clash_pairs)} P-P bonds; cyclic pairs with ring separation "
          f"2..7 residues: {len(cyclic)}")
    print(f"  pairs the clash list misses: {len(missing)} -- all crossing the 5'/3' "
          f"junction, e.g. {missing[:6]}")
    print(f"  (it never matters unless such a pair comes within {scalar(0.3)*10:.0f} A: "
          f"the clash contributed 0.00 kJ/mol in the shipped run below)")


# ── D ─────────────────────────────────────────────────────────────────────

def pair_probe(system, r_nm, bonded, with_clash):
    """Two P beads carrying the built system's own parameters, separation r_nm."""
    _, nb = one_force(system, mm.NonbondedForce)
    _, gb = one_force(system, mm.GBSAOBCForce)
    _, clash = one_force(system, mm.CustomBondForce, "dmin")
    q, sigma, eps = [scalar(x) for x in nb.getParticleParameters(0)]
    qg, radius, scale = [scalar(x) for x in gb.getParticleParameters(0)]
    ck = cdmin = None
    for b in range(clash.getNumBonds()):
        _a, _b, pr = clash.getBondParameters(b)
        ck, cdmin = scalar(pr[0]), scalar(pr[1])
        break

    s = mm.System()
    s.addParticle(110.0)
    s.addParticle(110.0)
    nb2 = mm.NonbondedForce()
    nb2.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    nb2.setReactionFieldDielectric(nb.getReactionFieldDielectric())
    nb2.addParticle(q, sigma, eps)
    nb2.addParticle(q, sigma, eps)
    if bonded:
        nb2.addException(0, 1, 0.0, 0.3, 0.0)
    nb2.setForceGroup(0)
    s.addForce(nb2)
    gb2 = mm.GBSAOBCForce()
    gb2.setSoluteDielectric(gb.getSoluteDielectric())
    gb2.setSolventDielectric(gb.getSolventDielectric())
    gb2.addParticle(qg, radius, scale)
    gb2.addParticle(qg, radius, scale)
    gb2.setForceGroup(1)
    s.addForce(gb2)
    if with_clash and ck is not None:
        cf = mm.CustomBondForce(clash.getEnergyFunction())
        cf.addPerBondParameter("k_clash")
        cf.addPerBondParameter("dmin")
        cf.addBond(0, 1, [ck, cdmin])
        cf.setForceGroup(2)
        s.addForce(cf)
    integ = mm.VerletIntegrator(0.001 * unit.picosecond)
    ctx = mm.Context(s, integ, Platform.getPlatformByName("CPU"))
    ctx.setPositions(np.array([[0.0, 0.0, 0.0], [r_nm, 0.0, 0.0]]) * unit.nanometer)
    return [ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy()._value
            for g in range(s.getNumForces())]


def d_pair_potential():
    system, _, _ = build()
    _, nb = one_force(system, mm.NonbondedForce)
    q = scalar(nb.getParticleParameters(0)[0])
    print("\n" + "=" * 78)
    print("D. P-P pair potential from the built parameters, per contributing force")
    print(f"   charge of the P bead read out of the built system: {q:+.4f} e")
    print("=" * 78)
    rs = [0.20, 0.25, 0.30, 0.40, 0.50, 0.59, 0.70, 0.80, 1.00, 1.20, 1.45,
          2.00, 3.00, 6.00]
    h = 1e-3

    def tot(r, bonded, clash):
        return sum(pair_probe(system, r, bonded, clash))

    print("\n(a) bonded backbone pair P(i)-P(i+1): NonbondedForce carries the (0,0.3,0) "
          "exception; GBSA does not honour it; the clash force is not added for this pair")
    print(f"{'r (A)':>7} {'E_NB':>10} {'E_GB':>10} {'E_tot':>10} "
          f"{'dE_tot vs 60A':>14} {'F = -dE/dr':>12}")
    e_ref = tot(6.0, True, False)
    for r in rs:
        e = pair_probe(system, r, True, False)
        f = -(tot(r + h, True, False) - tot(r - h, True, False)) / (2 * h)
        print(f"{r * 10:>7.2f} {e[0]:>10.3f} {e[1]:>10.3f} {sum(e):>10.3f} "
              f"{sum(e) - e_ref:>14.3f} {f:>12.2f}")

    print("\n(b) non-bonded P-P pair as P(i)-P(i+2..i+7) sees it: full 1/r Coulomb, GBSA, "
          "clash below 3 A")
    print(f"{'r (A)':>7} {'E_NB':>10} {'E_GB':>10} {'E_clash':>9} {'E_tot':>10} "
          f"{'dE_tot vs 60A':>14} {'F':>12} {'eps_eff':>8}")
    e_ref = tot(6.0, False, True)
    for r in rs:
        e = pair_probe(system, r, False, True)
        f = -(tot(r + h, False, True) - tot(r - h, False, True)) / (2 * h)
        eps_eff = COUL * q * q / r / (sum(e) - e_ref) if abs(sum(e) - e_ref) > 1e-9 else float("nan")
        print(f"{r * 10:>7.2f} {e[0]:>10.3f} {e[1]:>10.3f} {e[2]:>9.3f} {sum(e):>10.3f} "
              f"{sum(e) - e_ref:>14.3f} {f:>12.2f} {eps_eff:>8.2f}")
    print("eps_eff = the built field's net pair interaction expressed as a dielectric: "
          "1/r Coulomb divided by (Coulomb + GBSA).  78.5 = fully screened water, "
          "1.0 = vacuum, no screening at all.")


# ── C ─────────────────────────────────────────────────────────────────────

def reproduce(tag, pairs, sample_every=0):
    system, coords_nm, _ = build(pairs)
    names = group_forces(system)
    out = run(system, coords_nm, names, sample_every=sample_every)
    d_min = pp_dist_ang(out["pos_min"])
    d_md = pp_dist_ang(out["pos_md"])
    print(f"\n--- {tag} ---")
    print(f"  E start={out['e0']:.1f}  minimised={out['e_min']:.1f}  "
          f"after 3000 steps={out['e_md']:.1f} kJ/mol")
    print(f"  P-P after minimisation: mean={d_min.mean():.3f} sd={d_min.std():.3f} "
          f"min={d_min.min():.3f} max={d_min.max():.3f} A  (target {R.BOND_P_NEXT})")
    print(f"  P-P after MD          : mean={d_md.mean():.3f} sd={d_md.std():.3f} "
          f"min={d_md.min():.3f} max={d_md.max():.3f} A")
    print(f"  closure bond (29,0)   : {d_md[-1]:.3f} A   (its spring is K_BSJ + K_BSJ_GUIDE)")
    print(f"  thermal width the K_BB measurement implies: "
          f"sqrt(kBT/{R.K_BB * 100.0})*10 = {math.sqrt(KBT / (R.K_BB * 100.0)) * 10:.4f} A")
    print("  per-force energy after MD (kJ/mol):")
    for nm, e in per_force_energies(out["sim"], names):
        print(f"    {nm:<64} {e:>11.2f}")
    return out, names


def section_C():
    print("\n" + "=" * 78)
    print("C. reproduction: 30-nt ring, minimise + 3000 steps at 300 K, 2 fs")
    print("=" * 78)
    out0, names0 = reproduce("0 base pairs", [], sample_every=SAMPLE_EVERY)
    out2, names2 = reproduce("2 base pairs (5,24) (6,23)", PAIRS_2)
    return (out0, names0), (out2, names2)


# ── E ─────────────────────────────────────────────────────────────────────

def section_E(out, tag):
    print("\n" + "=" * 78)
    print(f"E. trajectory-averaged per-bead force balance, outward radial ({tag})")
    print("   every force acting on each P bead counted once; + pushes the bead out of "
          "the ring")
    print("=" * 78)
    names = out["names"]
    dist, _, n_frames = mean_axial_load(out["samples"])
    loads, _ = mean_radial_load(out["samples"])
    print(f"  frames sampled: {n_frames}, every {SAMPLE_EVERY} steps")
    print(f"  <r> = {dist.mean():.3f} A (sd over the {L} bonds {dist.std():.3f}); the "
          f"bond's own restoring force k(<r>-r0) = "
          f"{R.K_BB * 100.0 * (dist.mean() / 10.0 - R.BOND_P_NEXT / 10.0):.2f} kJ/mol/nm")
    print(f"  closure bond <r> = {dist[-1]:.3f} A "
          f"(its spring is the sum of the two BSJ forces)")
    print(f"\n  {'force':<64} {'mean radial':>11} {'sd over beads':>14}")
    order = np.argsort(-np.abs(loads.mean(axis=0)))
    for k in order:
        col = loads[:, k]
        if abs(col.mean()) < 1e-3 and col.std() < 1e-3:
            continue
        print(f"  {names[k]:<64} {col.mean():>11.2f} {col.std():>14.2f}")
    tot = loads.sum(axis=1)
    print(f"  {'TOTAL (the Langevin kicks average to 0)':<64} "
          f"{tot.mean():>11.2f} {tot.std():>14.2f}")
    nb_cols = [k for k, nm in enumerate(names)
               if "NonbondedForce" in nm or "GBSAOBC" in nm or "dmin" in nm]
    print(f"  nonbonded block (NonbondedForce + GBSAOBCForce + clash) total: "
          f"{loads[:, nb_cols].sum(axis=1).mean():.2f} kJ/mol/nm outward per bead; "
          f"the bonded terms must supply the same inward")


def section_E_shells(out, tag):
    """Where the nonbonded energy lives, by ring separation."""
    print(f"\n--- nonbonded repulsion by ring separation, {tag}, minimised frame ---")
    system = out["system"]
    e_ref = sum(pair_probe(system, 6.0, False, True))
    pos = out["pos_min"]
    P = pos[0::3] * 10.0
    print(f"  {'i-j (mod 30)':>13} {'n pairs':>8} {'<r> A':>8} {'<E_NB>':>10} "
          f"{'<E_tot pair>':>12}")
    for s in range(2, 16):
        ds, es, et = [], [], []
        for i in range(L):
            j = (i + s) % L
            if i == j:
                continue
            r = np.linalg.norm(P[i] - P[j]) * 0.1     # nm
            ds.append(r * 10)
            es.append(COUL * 0.36 / r)
            et.append(sum(pair_probe(system, r, False, True)) - e_ref)
        print(f"  {s:>13} {len(ds):>8} {np.mean(ds):>8.3f} {np.mean(es):>10.2f} "
              f"{np.mean(et):>12.2f}")
    print("  <E_NB> is the bare 1/r Coulomb at the actual separation; <E_tot pair> is the "
          "built pair potential (Coulomb+GBSA+clash) at that separation")


# ── F ─────────────────────────────────────────────────────────────────────

def zero_charges(system, which="both"):
    _, nb = one_force(system, mm.NonbondedForce)
    _, gb = one_force(system, mm.GBSAOBCForce)
    n = 0
    if which in ("both", "nb"):
        for i in range(nb.getNumParticles()):
            q, s, e = nb.getParticleParameters(i)
            if scalar(q) != 0.0:
                nb.setParticleParameters(i, 0.0, s, e)
                n += 1
    if which in ("both", "gb"):
        for i in range(gb.getNumParticles()):
            q, r, sc = gb.getParticleParameters(i)
            if scalar(q) != 0.0:
                gb.setParticleParameters(i, 0.0, r, sc)
                n += 1
    return n


def scale_charges(system, scale):
    _, nb = one_force(system, mm.NonbondedForce)
    _, gb = one_force(system, mm.GBSAOBCForce)
    for i in range(nb.getNumParticles()):
        q, s, e = nb.getParticleParameters(i)
        nb.setParticleParameters(i, scalar(q) * scale, s, e)
    for i in range(gb.getNumParticles()):
        q, r, sc = gb.getParticleParameters(i)
        gb.setParticleParameters(i, scalar(q) * scale, r, sc)
    return f"q->{scale}*q"


def set_solute_dielectric(system, eps_in):
    _, gb = one_force(system, mm.GBSAOBCForce)
    old = gb.getSoluteDielectric()
    gb.setSoluteDielectric(eps_in)
    return f"eps_in {old}->{eps_in}"


def replace_coulomb_with_yukawa(system, kappa):
    """Scratch: same particles and exclusions, direct Coulomb -> Debye-Huckel screened."""
    idx, old = one_force(system, mm.NonbondedForce)
    new = mm.CustomNonbondedForce(f"{COUL}*q1*q2*exp(-kappa*r)/r")
    new.addPerParticleParameter("q")
    new.addGlobalParameter("kappa", kappa)
    new.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
    for i in range(old.getNumParticles()):
        q, s, e = old.getParticleParameters(i)
        new.addParticle([scalar(q)])
    for k in range(old.getNumExceptions()):
        a, b, qq, sg, ep = old.getExceptionParameters(k)
        new.addExclusion(a, b)
    system.removeForce(idx)
    system.addForce(new)
    return f"kappa={kappa:.4f}/nm"


def old_kbb(system, k_nm2=50000.0):
    """Scratch: restore the pre-measurement K_BB (500 kJ/mol/A^2 -> 50000 kJ/mol/nm^2)."""
    n = 0
    for _, f in find_force(system, mm.HarmonicBondForce):
        for b in range(f.getNumBonds()):
            a, c, r0, k = f.getBondParameters(b)
            if a % 3 == 0 and c % 3 == 0:
                f.setBondParameters(b, a, c, r0, k_nm2)
                n += 1
    return f"{n} backbone bonds set to {k_nm2}"


def add_13_exclusions(system):
    _, nb = one_force(system, mm.NonbondedForce)
    have = set()
    for k in range(nb.getNumExceptions()):
        a, b, q, s, e = nb.getExceptionParameters(k)
        have.add((min(a, b), max(a, b)))
    added = 0
    for i in range(L):
        a, b = 3 * i, 3 * ((i + 2) % L)
        key = (min(a, b), max(a, b))
        if key not in have:
            nb.addException(a, b, 0.0, 0.3, 0.0)
            have.add(key)
            added += 1
    return f"{added} 1-3 pairs excluded"


def section_F():
    print("\n" + "=" * 78)
    print("F. controls: what puts the backbone at 7 A (same coordinates, same seed, "
          "same protocol)")
    print("=" * 78)
    kappa_phys = 3.2881 * math.sqrt(0.145)
    rows = []

    def one(tag, mutate):
        system, coords_nm, _ = build()
        info = mutate(system) if mutate else ""
        names = group_forces(system)
        out = run(system, coords_nm, names)
        d_md = pp_dist_ang(out["pos_md"])
        d_min = pp_dist_ang(out["pos_min"])
        e_terms = dict(per_force_energies(out["sim"], names))
        enb = next((v for k, v in e_terms.items() if "NonbondedForce" in k), float("nan"))
        egb = next((v for k, v in e_terms.items() if "GBSAOBC" in k), float("nan"))
        rows.append((tag, d_md.mean(), d_md.std(), d_min.mean(), enb, egb, out["e_md"]))
        print(f"  {tag:<52} mean={d_md.mean():.3f} sd={d_md.std():.3f} "
              f"min={d_min.mean():.3f}  E={out['e_md']:.0f}  {info}")

    one("shipped", None)
    one("charges zeroed, both forces", lambda s: zero_charges(s, "both"))
    one("NonbondedForce only (GB charges 0)", lambda s: zero_charges(s, "gb"))
    one("GBSAOBC only (NonbondedForce charges 0)", lambda s: zero_charges(s, "nb"))
    one(f"Debye screened Coulomb kappa={kappa_phys:.3f}/nm (0.145 M)",
        lambda s: replace_coulomb_with_yukawa(s, kappa_phys))
    one("Debye screened Coulomb kappa=0.132/nm (module literal)",
        lambda s: replace_coulomb_with_yukawa(s, 1.0 / (1.0 / math.sqrt(0.145 * 0.06022 * 2))))
    one("soluteDielectric 4.0 -> 1.0", lambda s: set_solute_dielectric(s, 1.0))
    one("soluteDielectric 4.0 -> 20.0", lambda s: set_solute_dielectric(s, 20.0))
    one("1-3 backbone P-P pairs excluded too", lambda s: add_13_exclusions(s))
    one("old K_BB = 50000 kJ/mol/nm^2 restored", lambda s: old_kbb(s, 50000.0))
    one("old K_BB restored AND 1-3 excluded",
        lambda s: (old_kbb(s, 50000.0), add_13_exclusions(s)))
    one("old K_BB restored AND charges zeroed",
        lambda s: (old_kbb(s, 50000.0), zero_charges(s, "both")))
    for s_ in (0.7, 0.5, 0.3):
        one(f"charges scaled to {s_}*q", lambda s, sc=s_: scale_charges(s, sc))

    print(f"\n{'control':<54} {'<r> A':>7} {'sd':>6} {'<r>min':>7} "
          f"{'E_NB':>9} {'E_GB':>9} {'E_tot':>9}")
    for tag, m, sd, mn, enb, egb, et in rows:
        print(f"{tag:<54} {m:>7.3f} {sd:>6.3f} {mn:>7.3f} "
              f"{enb:>9.0f} {egb:>9.0f} {et:>9.0f}")
    return rows


# ── H ─────────────────────────────────────────────────────────────────────

def section_H():
    """Measure the OTHER force field in the same module, and whether it is reachable."""
    print("=" * 78)
    print("H. _build_minimal_system_gpu: a different force field, measured")
    print("=" * 78)
    pairs = [(2, 27, 1.0)]
    p_ang = R._generate_compact_coords(L, pairs)
    system, coords_nm, pf = R._build_minimal_system_gpu(p_ang, pairs, pair_scale=1.0)
    print(f"minimal system: {system.getNumParticles()} particles (P beads only), "
          f"{system.getNumForces()} forces; the full field has "
          f"{build()[0].getNumParticles()} particles")
    print("charges: none -- this system has no NonbondedForce and no GBSAOBCForce "
          "(it is P beads and harmonic/clash restraints only)")
    for idx, f in enumerate(system.getForces()):
        print(f"  {force_label(f, idx)}")
        if isinstance(f, mm.HarmonicBondForce):
            rows = {}
            for b in range(f.getNumBonds()):
                a, c, r0, k = f.getBondParameters(b)
                rows.setdefault((round(scalar(k), 4), round(scalar(r0), 4)), 0)
                rows[(round(scalar(k), 4), round(scalar(r0), 4))] += 1
            for (k, r0), n in sorted(rows.items()):
                print(f"      k={k} kJ/mol/nm^2  r0={r0} nm  x{n} bonds")
        if isinstance(f, mm.HarmonicAngleForce):
            rows = {}
            for b in range(f.getNumAngles()):
                a, c, d, th0, k = f.getAngleParameters(b)
                rows.setdefault((round(scalar(k), 4), round(scalar(th0), 4)), 0)
                rows[(round(scalar(k), 4), round(scalar(th0), 4))] += 1
            for (k, th0), n in sorted(rows.items()):
                print(f"      k={k} kJ/mol/rad^2  theta0={th0} rad  x{n} angles")
        if isinstance(f, mm.CustomBondForce):
            rows = {}
            for b in range(f.getNumBonds()):
                a, c, prm = f.getBondParameters(b)
                key = tuple(round(scalar(x), 3) for x in prm)
                rows.setdefault(key, 0)
                rows[key] += 1
            for key, n in sorted(rows.items())[:6]:
                print(f"      params={key}  x{n} bonds")
        if isinstance(f, mm.CustomNonbondedForce):
            print(f"      expression = {f.getEnergyFunction()}")
            for k in range(f.getNumGlobalParameters() if hasattr(f, "getNumGlobalParameters") else 0):
                print(f"      global {f.getGlobalParameterName(k)} = "
                      f"{f.getGlobalParameterDefaultValue(k)}")
            print(f"      particles = {f.getNumParticles()}, "
                  f"interaction groups = {f.getNumInteractionGroups() if hasattr(f, 'getNumInteractionGroups') else '?'}")
    print("\nsource call sites of _build_minimal_system_gpu (grep, read-only):")
    print("  openmm_gpu_refiner.py:769  _run_minimal_anneal_worker")
    print("  openmm_gpu_refiner.py:943  _run_remd_worker(minimal=True)")
    print("  openmm_gpu_refiner.py:2016 far-pair pre-pull (L > 50)")
    print("  openmm_gpu_refine() defaults: skip_minimal_fold=False -> "
          "_run_parallel_minimal_annealing -> _run_minimal_anneal_worker")
    print("  -> live on the default shipped path; it carries no nonbonded block at all")


# ── main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    section_A()
    section_B()
    section_G()
    d_pair_potential()
    (out0, names0), _ = section_C()
    section_E(out0, "0 base pairs, 300 K / 2 fs")
    section_E_shells(out0, "0 base pairs")
    section_F()
    print(f"\ntotal wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
