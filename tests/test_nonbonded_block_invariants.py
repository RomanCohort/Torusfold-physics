r"""Invariants of the nonbonded block of openmm_gpu_refiner._build_3bead_system_gpu.

These lock the two things that characterisation of that block established structurally,
not the physics (which the coordinator owns):

  * every global parameter the built System carries is actually read by the force that
    carries it -- a name written and never consumed is this project's silent-failure
    pattern, and a grep is not a measurement;
  * the NonbondedForce exception set covers every 1-2 (covalently bonded) pair and the
    cyclisation pair exactly once, with the (charge 0, sigma 0.3, epsilon 0) parameters
    the block declares -- so nothing bonded is left carrying a bare 1/r repulsion.

Extra exceptions are deliberately allowed: adding 1-3 exclusions is a live physics
option, and these tests must not forbid it.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.openmm_gpu_refiner as C  # noqa: E402

L = 12
RING_PAIRS = [(2, 9, 1.0)]


def _build(L=L, pairs=RING_PAIRS, **kw):
    p_ang = C._generate_compact_coords(L, pairs)
    return C._build_3bead_system_gpu(p_ang, pairs, pair_scale=1.0, bsj_k_scale=1.0, **kw)


def _scalar(x):
    return x._value if hasattr(x, "_value") else float(x)


def _global_parameters(system):
    out = []
    for f in system.getForces():
        if not hasattr(f, "getNumGlobalParameters"):
            continue
        for k in range(f.getNumGlobalParameters()):
            out.append((f.getGlobalParameterName(k),
                        f.getGlobalParameterDefaultValue(k),
                        type(f).__name__))
    return out


def test_every_global_parameter_in_the_built_system_is_read():
    """Perturb each global parameter in a live Context; an unread one moves nothing."""
    openmm = pytest.importorskip("openmm")
    system, coords_nm, *_ = _build(pair_guide_k=600.0)
    params = _global_parameters(system)
    assert params, "the built system carries no global parameters at all"

    integ = openmm.VerletIntegrator(0.001 * openmm.unit.picosecond)
    ctx = openmm.Context(system, integ,
                         openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(coords_nm * openmm.unit.nanometer)
    e0 = ctx.getState(getEnergy=True).getPotentialEnergy()._value

    dead = []
    for name, default, owner in params:
        seen = False
        for probe in (default + 0.37, default * 1.5 + 0.11):
            ctx.setParameter(name, probe)
            e = ctx.getState(getEnergy=True).getPotentialEnergy()._value
            if not np.isfinite(e) or abs(e - e0) > 1e-6:
                seen = True
                break
        ctx.setParameter(name, default)
        if not seen:
            dead.append(f"{name} (default {default}, owner {owner})")
    assert not dead, (
        "global parameter(s) written into the built System but read by no force: "
        + ", ".join(dead) + ".  A parameter that no energy expression references is "
        "dead weight -- perturbing it in a live Context must move the energy.")


def test_nonbonded_exceptions_cover_every_bonded_pair_exactly_once():
    system, *_ = _build(L=30, pairs=[])
    openmm = pytest.importorskip("openmm")
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]

    actual = []
    for k in range(nb.getNumExceptions()):
        a, b, q, s, e = nb.getExceptionParameters(k)
        actual.append(((min(a, b), max(a, b)), (_scalar(q), _scalar(s), _scalar(e))))

    expected = set()
    for i in range(30):
        expected.add((3 * i, 3 * i + 1))          # P(i)-C4'(i)
        expected.add((3 * i + 1, 3 * i + 2))      # C4'(i)-N(i)
    for i in range(29):
        expected.add((3 * i, 3 * i + 3))          # P(i)-P(i+1)
    expected.add((0, 3 * 29))                     # cyclisation P(0)-P(29)

    keys = [k for k, _ in actual]
    missing = sorted(expected - set(keys))
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not missing, (
        f"bonded pairs left without a NonbondedForce exception (bare 1/r between "
        f"bonded beads): {missing}")
    assert not dupes, f"exception added twice for the same pair: {dupes}"
    for key in sorted(expected):
        prm = dict(actual)[key]
        assert prm == (0.0, 0.3, 0.0), (
            f"exception {key} carries {prm}, not (charge 0, sigma 0.3, epsilon 0)")


def test_exceptions_do_not_reach_gbsaocbcforce():
    """The (0, 0.3, 0) exception zeroes the Coulomb term only; GBSA is a separate force."""
    openmm = pytest.importorskip("openmm")
    system, *_ = _build(L=8, pairs=[])
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    gb = [f for f in system.getForces() if isinstance(f, openmm.GBSAOBCForce)][0]
    q, sigma, eps = [_scalar(x) for x in nb.getParticleParameters(0)]
    qg, radius, scale = (list(gb.getParticleParameters(0)) + [None])[:3]

    s = openmm.System()
    s.addParticle(110.0)
    s.addParticle(110.0)
    nb2 = openmm.NonbondedForce()
    nb2.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    nb2.addParticle(q, sigma, eps)
    nb2.addParticle(q, sigma, eps)
    nb2.addException(0, 1, 0.0, 0.3, 0.0)
    nb2.setForceGroup(0)
    s.addForce(nb2)
    gb2 = openmm.GBSAOBCForce()
    gb2.setSoluteDielectric(gb.getSoluteDielectric())
    gb2.setSolventDielectric(gb.getSolventDielectric())
    gb2.addParticle(gb.getParticleParameters(0)[0],
                    gb.getParticleParameters(0)[1],
                    gb.getParticleParameters(0)[2])
    gb2.addParticle(gb.getParticleParameters(0)[0],
                    gb.getParticleParameters(0)[1],
                    gb.getParticleParameters(0)[2])
    gb2.setForceGroup(1)
    s.addForce(gb2)

    integ = openmm.VerletIntegrator(0.001 * openmm.unit.picosecond)
    ctx = openmm.Context(s, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(np.array([[0.0, 0.0, 0.0], [0.59, 0.0, 0.0]])
                     * openmm.unit.nanometer)
    e_nb = ctx.getState(getEnergy=True, groups={0}).getPotentialEnergy()._value
    e_gb = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy()._value
    assert abs(e_nb) < 1e-9, (
        "the NonbondedForce exception did not zero the pair's Coulomb term")
    assert abs(e_gb) > 1e-9, (
        "GBSAOBCForce returned 0 for a bonded pair; it does not share the "
        "NonbondedForce exception list, so the exclusion set cannot be read as "
        "'this pair has no nonbonded interaction'")
