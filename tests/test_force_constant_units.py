r"""Lock the per-constant unit convention of openmm_gpu_refiner.py, as its header states it.

The trap this file keeps shut: that module declares K_BB and K_STACK in kJ/mol/angstrom^2 and
multiplies them by 100 at their own use site, but hands K_PAIR, K_BSJ and K_BSJ_GUIDE to OpenMM
RAW (no conversion), and multiplies K_CLASH by 10 rather than 100.  The declaration therefore
does not carry the unit.  Reading the block as if one conversion applied to all of it is wrong
by a factor of 100 on three constants -- wrong in a way that still builds a legal System and
raises nothing.

K_INTRA_PC and K_INTRA_CN used to be the two other *100 rows and are deleted.  They are not
missing from the dict below by oversight: P-C4' and C4'-N are System constraints now, and a
constraint has no factor at use and no unit to convert.  The 3-bead model therefore has ONE
HarmonicBondForce, the backbone; the intra-residue pairs appear in getNumConstraints() and in
getConstraintParameters(), which is where this file and test_cpu_force_constants.py read them.

_build_minimal_system_gpu is a second, P-only model with its own literals and no measurement
behind them; it is locked here too, together with the two facts that make it a different model
rather than a simplification of the 3-bead one: it has no NonbondedForce and no GBSAOBCForce.

Every number below is read back out of a System built by the module, the same way
tests/test_cpu_force_constants.py and tests/test_nonbonded_block_invariants.py do, so a use
site silently changing its conversion fails here rather than being noticed as a factor of 100
in an energy.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.openmm_gpu_refiner as C  # noqa: E402

# The conversion each main-field use site applies, exactly as the module header states it.
# The declared value and the value handed to OpenMM are NOT the same number; that is the
# point of the table, and this dict is the second copy of it that fails loudly on drift.
MAIN_FIELD_FACTOR = {
    "K_BB": 100.0,
    "K_STACK": 100.0,
    "K_ANGLE": 1.0,
    "K_DIHEDRAL": 1.0,
    "K_PAIR": 1.0,
    "K_BSJ": 1.0,
    "K_BSJ_GUIDE": 1.0,
    "K_CLASH": 10.0,
}

# The minimal P-only model's literals, as read back from its built System.  They are not
# sourced from the block above, are not derived from the database measurement that block
# cites, and are not to be compared with it.
MINIMAL_BOND_K = 31000.0     # kJ/mol/nm^2, per P(i)-P(i+1)
MINIMAL_BSJ_K = 500.0        # kJ/mol/nm^2, the single (0, L-1) closure
MINIMAL_ANGLE_K = 500.0      # kJ/mol/rad^2
MINIMAL_CLASH_K = 5000.0     # kJ/mol/nm^2
MINIMAL_CLASH_DMIN = 0.3     # nm
MINIMAL_PAIR_K = 30000.0     # kJ/mol/nm^2 before * w * pair_scale * far_boost


def _scalar(x):
    return x._value if hasattr(x, "_value") else float(x)


def _bond(force, b):
    i, j, r0, k = force.getBondParameters(b)
    return (min(i, j), max(i, j)), _scalar(r0), _scalar(k)


def _custom_bond(system, openmm, param_name):
    for f in system.getForces():
        if isinstance(f, openmm.CustomBondForce) and f.getNumPerBondParameters() >= 1 \
                and f.getPerBondParameterName(0) == param_name:
            return f
    raise AssertionError(f"no CustomBondForce with per-bond parameter {param_name!r}")


def _chain(L):
    p = np.zeros((L, 3), dtype=np.float64)
    p[:, 0] = np.arange(L) * C.BOND_P_NEXT
    return p


def test_each_main_field_constant_reaches_openmm_at_its_own_factor():
    """Read every constant of the block back out of the built System, one at a time."""
    openmm = pytest.importorskip("openmm")
    L = 12
    pairs = [(0, 9, 1.0), (1, 8, 0.5)]
    system, _, _, _, _, _ = C._build_3bead_system_gpu(_chain(L), pairs)

    def expect(name):
        return getattr(C, name) * MAIN_FIELD_FACTOR[name]

    # -- the one HarmonicBondForce: the backbone (*100).  The intra-residue pair is a
    #    constraint here, so it has no factor to get wrong; it is asserted below.
    hb = [f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce)]
    assert len(hb) == 1, (
        f"expected exactly one HarmonicBondForce (the backbone), got {len(hb)}. P-C4' and "
        f"C4'-N are System constraints now; a second bonded force means they are springs "
        f"again, and the factor table above would be silently incomplete")
    backbone = hb[0]
    ids = {_bond(backbone, b)[0] for b in range(backbone.getNumBonds())}
    assert (0, 3) in ids, "the single HarmonicBondForce is not the P(i)-P(i+1) backbone"
    assert (0, 1) not in ids and (1, 2) not in ids, (
        "an intra-residue pair is in the bonded force list again")

    _, r0, k = _bond(backbone, 0)
    assert k == pytest.approx(expect("K_BB")), (
        f"K_BB reaches OpenMM as {k}, not K_BB * {MAIN_FIELD_FACTOR['K_BB']}")
    assert r0 == pytest.approx(C.BOND_P_NEXT / 10.0), "BOND_P_NEXT is not divided by 10"

    # -- the intra-residue pair, in the list that has no units: 2 constraints per residue
    assert system.getNumConstraints() == 2 * L, (
        f"expected 2 constraints per residue; got {system.getNumConstraints()}")
    cst = {}
    for c in range(system.getNumConstraints()):
        i, j, d = system.getConstraintParameters(c)
        cst[(min(i, j), max(i, j))] = _scalar(d)
    assert set(cst) == {(3 * i, 3 * i + 1) for i in range(L)} \
        | {(3 * i + 1, 3 * i + 2) for i in range(L)}, sorted(cst)
    assert cst[(0, 1)] == pytest.approx(C.BOND_P_C4 / 10.0)
    assert cst[(1, 2)] == pytest.approx(C.BOND_C4_N / 10.0)
    assert not hasattr(C, "K_INTRA_PC") and not hasattr(C, "K_INTRA_CN"), (
        "K_INTRA_PC/K_INTRA_CN are readable again; they were deleted, so that any surviving "
        "reference raises rather than reading a stiffness that was never calibratable")

    # -- angle (*1) and dihedral (*1)
    ang = [f for f in system.getForces() if isinstance(f, openmm.HarmonicAngleForce)]
    assert len(ang) == 1
    for b in range(ang[0].getNumAngles()):
        i, j, k3, th0, k = ang[0].getAngleParameters(b)
        assert _scalar(k) == pytest.approx(expect("K_ANGLE")), (
            f"K_ANGLE reaches OpenMM as {_scalar(k)}; it is already kJ/mol/rad^2")
        assert _scalar(th0) == pytest.approx(C.ANGLE_PPP)

    dih = [f for f in system.getForces() if isinstance(f, openmm.CustomTorsionForce)]
    assert len(dih) == 1 and dih[0].getGlobalParameterName(0) == "k_dih"
    assert dih[0].getGlobalParameterDefaultValue(0) == pytest.approx(expect("K_DIHEDRAL"))

    # -- stacking (*100)
    stack = _custom_bond(system, openmm, "k_stack")
    assert stack.getNumBonds() == L, "one stacking bond per consecutive N-N pair plus the BSJ"
    for b in range(stack.getNumBonds()):
        i, j, prm = stack.getBondParameters(b)
        assert _scalar(prm[0]) == pytest.approx(expect("K_STACK"))

    # -- pairing: the constant that must NOT be converted
    pair = _custom_bond(system, openmm, "k_pair")
    got = {}
    for b in range(pair.getNumBonds()):
        i, j, prm = pair.getBondParameters(b)
        got[(i, j)] = _scalar(prm[0])
    # N(i) is atom 3i+2; pair (0,9) has w=1.0 and pair (1,8) has w=0.5
    assert got[(2, 29)] == pytest.approx(C.K_PAIR * 1.0 * MAIN_FIELD_FACTOR["K_PAIR"])
    assert got[(5, 26)] == pytest.approx(C.K_PAIR * 0.5 * MAIN_FIELD_FACTOR["K_PAIR"])
    assert got[(2, 29)] != pytest.approx(C.K_PAIR * 100.0), (
        "the built pair constant is K_PAIR*100; K_PAIR is declared in kJ/mol/nm^2 and is "
        "used raw -- converting it here is the factor-100 trap this test exists to catch")

    # -- BSJ and its guide force: also raw
    bsj = _custom_bond(system, openmm, "k_bsj")
    guide = _custom_bond(system, openmm, "k_guide")
    assert _scalar(bsj.getBondParameters(0)[2][0]) == pytest.approx(expect("K_BSJ"))
    assert _scalar(guide.getBondParameters(0)[2][0]) == pytest.approx(expect("K_BSJ_GUIDE"))
    assert _scalar(bsj.getBondParameters(0)[2][0]) != pytest.approx(C.K_BSJ * 100.0)

    # -- clash: *10, and an expression with no 0.5 in it
    clash = _custom_bond(system, openmm, "k_clash")
    expr = clash.getEnergyFunction()
    assert "0.5" not in expr, (
        f"the clash expression is now {expr!r}; it is documented as having no 0.5, which is "
        f"why K_CLASH*10 = {expect('K_CLASH')} counts as {2 * expect('K_CLASH')} in the "
        f"0.5*k convention every other term here uses")
    for b in range(clash.getNumBonds()):
        i, j, prm = clash.getBondParameters(b)
        assert _scalar(prm[0]) == pytest.approx(expect("K_CLASH"))
        assert _scalar(prm[1]) == pytest.approx(0.3)


def test_minimal_p_only_model_is_separate_and_carries_its_own_literals():
    """Read the minimal model's constants back out of its own System."""
    openmm = pytest.importorskip("openmm")
    L = 30
    system, coords_nm, pair_force = C._build_minimal_system_gpu(
        _chain(L), [(0, 20, 1.0), (1, 19, 0.5)])

    assert coords_nm.shape == (L, 3)
    assert system.getNumParticles() == L, "the minimal model is P-only, one bead per nt"
    for i in range(L):
        assert _scalar(system.getParticleMass(i)) == pytest.approx(110.0)

    kinds = [type(f).__name__ for f in system.getForces()]
    assert kinds == ["HarmonicBondForce", "HarmonicAngleForce",
                     "CustomNonbondedForce", "CustomBondForce"], kinds
    assert not any(isinstance(f, openmm.NonbondedForce) for f in system.getForces()), (
        "a NonbondedForce appeared in the minimal P-only model; it is documented as having "
        "neither electrostatics nor implicit solvent, and a model that gained them is no "
        "longer the model whose unmeasured literals this test locks")
    assert not any(isinstance(f, openmm.GBSAOBCForce) for f in system.getForces()), (
        "a GBSAOBCForce appeared in the minimal P-only model; the 3-bead model above has one")

    # -- bonds: 31000 backbone, 500 the BSJ closure, all at r0 = 5.9 A / 10
    hb = [f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce)]
    assert len(hb) == 1
    seen = {}
    for b in range(hb[0].getNumBonds()):
        (a1, a2), r0, k = _bond(hb[0], b)
        seen.setdefault(round(k, 6), []).append((a1, a2))
        assert r0 == pytest.approx(C.BOND_P_NEXT / 10.0)
    assert set(seen) == {MINIMAL_BOND_K, MINIMAL_BSJ_K}, seen
    assert seen[MINIMAL_BSJ_K] == [(0, L - 1)], "the 500.0 bond is the BSJ closure"
    assert len(seen[MINIMAL_BOND_K]) == L - 1

    # -- angles: 500 kJ/mol/rad^2 at the A-form 150 deg
    ang = [f for f in system.getForces() if isinstance(f, openmm.HarmonicAngleForce)][0]
    assert ang.getNumAngles() == L - 2
    for b in range(ang.getNumAngles()):
        i, j, k3, th0, k = ang.getAngleParameters(b)
        assert _scalar(k) == pytest.approx(MINIMAL_ANGLE_K)
        assert _scalar(th0) == pytest.approx(C.ANGLE_PPP)

    # -- clash: a CustomNonbondedForce that DOES carry the 0.5 the 3-bead clash lacks
    clash = [f for f in system.getForces()
             if isinstance(f, openmm.CustomNonbondedForce)][0]
    assert "0.5" in clash.getEnergyFunction(), (
        "the minimal model's clash is documented as carrying the 0.5 that the 3-bead clash "
        f"expression does not; it now reads {clash.getEnergyFunction()!r}")
    gparams = {clash.getGlobalParameterName(g): clash.getGlobalParameterDefaultValue(g)
               for g in range(clash.getNumGlobalParameters())}
    assert gparams == {"k_clash": MINIMAL_CLASH_K, "d_min": MINIMAL_CLASH_DMIN}
    assert clash.getNumParticles() == L
    assert clash.getNumInteractionGroups() == 1

    # -- pairing: 30000 kJ/mol/nm^2 * w, raw
    got = {}
    for b in range(pair_force.getNumBonds()):
        i, j, prm = pair_force.getBondParameters(b)
        got[(i, j)] = (_scalar(prm[0]), _scalar(prm[1]))
    assert got[(0, 20)][0] == pytest.approx(MINIMAL_PAIR_K * 1.0)
    assert got[(1, 19)][0] == pytest.approx(MINIMAL_PAIR_K * 0.5)
    assert got[(0, 20)][1] == pytest.approx(C.BOND_P_NEXT / 10.0)

    # -- the >100 nt far-pair boost, read back rather than inferred
    Lf = 220
    far = C._build_minimal_system_gpu(_chain(Lf), [(0, 110, 1.0), (0, 40, 1.0)])[0]
    kf = {}
    for f in far.getForces():
        if isinstance(f, openmm.CustomBondForce):
            for b in range(f.getNumBonds()):
                i, j, prm = f.getBondParameters(b)
                kf[(min(i, j), max(i, j))] = _scalar(prm[0])
    assert kf[(0, 110)] == pytest.approx(2.0 * MINIMAL_PAIR_K), "no 2x far-pair boost"
    assert kf[(0, 40)] == pytest.approx(MINIMAL_PAIR_K), "boost applied inside 100 nt"

    # -- and the two models must not be made to agree by copying numerals
    assert MINIMAL_BOND_K != pytest.approx(C.K_BB * MAIN_FIELD_FACTOR["K_BB"]), (
        "the minimal model's backbone constant now equals the 3-bead one's; they are "
        "different models and their numbers are not comparable")
