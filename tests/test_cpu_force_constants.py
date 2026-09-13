r"""Lock the CPU-path (OpenMM) bonded constants to the measurement that produced them.

Every constant in the bonded block of openmm_gpu_refiner.py was derived by
scripts/measure_cpu_constants.py from D:\torusfold-cgdata\rsRNASP\Training_set,
with the criterion k = kBT/sigma^2: a harmonic restraint of stiffness k on a
coordinate whose equilibrium spread is sigma produces a spread sqrt(kBT/k), so
reproducing the observed spread requires k = kBT/sigma^2, kBT = 2.494 kJ/mol at 300 K.
The sigmas below are the pooled spreads over 126 gap-free chains, measured with the
same boltzmann_bonded machinery that produced results/boltzmann_tables_clean.npz.

What the database cannot do, stated per constant: it holds ONE conformation per
structure, so the pooled sigma mixes residue-to-residue and conformer-to-conformer
variation into the same number as the thermal fluctuation.  It therefore OVERSTATES
the thermal width and kBT/sigma^2 is a LOWER BOUND on the stiffness, never the
stiffness.  These tests lock each constant to that bound and to the functional form
and unit the constant is used in -- not to a stiffness the database cannot supply.
The mean within-chain sigma is stored alongside each entry as the stiffer (still
bounded) alternative.

Also locked here: this file's units.  The distance constants are declared in
kJ/mol/angstrom^2 and multiplied by 100 at use, while torch_cgsim.py declares the
same kind of constant in kJ/mol/nm^2; the last test reads the numbers back out of a
built OpenMM System, so a numeral silently changing meaning fails the suite.

The two intra-residue distances are the exception that proves it: P-C4' and C4'-N have no
constant at all here any more, so there is no unit for them to be got wrong in.  They are
System constraints, and the test that covers them reads the constraint list instead.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.openmm_gpu_refiner as C  # noqa: E402

KBT = 2.494  # kJ/mol at 300 K

# name -> (pooled sigma in angstrom, n, mean within-chain sigma in angstrom)
#
# K_INTRA_PC (sigma 0.1055 A) and K_INTRA_CN (sigma 0.0814 A) were rows here and are gone,
# along with the constants.  Both sigmas are BELOW the 0.1-0.3 A coordinate-error floor of
# the source structures, so kBT/sigma^2 over them is not a bounded stiffness -- it is the
# inversion of a number that measures refinement restraint rather than thermal motion.
# They are constraints now; test_intra_residue_distances_are_constraints_not_constants below
# locks that, which is the invariant that replaced these two rows.
MEASURED_DISTANCE = {
    "K_BB": (0.4714, 6638, 0.4454),
    "K_STACK": (1.6127, 6638, 1.4959),
}

# name -> (pooled sigma in rad, n, mean within-chain sigma in rad)
MEASURED_ANGLE = {
    "K_ANGLE": (0.3855, 6512, 0.3570),
    "K_DIHEDRAL": (1.0846, 6386, 1.0249),
}


@pytest.mark.parametrize("name", sorted(MEASURED_DISTANCE))
def test_distance_constant_is_kbt_over_sigma_squared(name):
    sigma_A, n, within_A = MEASURED_DISTANCE[name]
    k_expected = KBT / sigma_A ** 2        # kJ/mol/angstrom^2, the declared unit
    k = getattr(C, name)
    assert k == pytest.approx(k_expected, rel=2e-3), (
        f"{name} = {k}, but the measured sigma = {sigma_A} A over {n} observations "
        f"implies kBT/sigma^2 = {k_expected:.4f} kJ/mol/angstrom^2 "
        f"(within-chain sigma {within_A} A would give {KBT / within_A ** 2:.4f})")


@pytest.mark.parametrize("name", sorted(MEASURED_DISTANCE))
def test_distance_constant_reproduces_the_measured_width_in_nm(name):
    sigma_A, _, _ = MEASURED_DISTANCE[name]
    k_nm2 = getattr(C, name) * 100.0       # what the code hands OpenMM
    assert math.sqrt(KBT / k_nm2) == pytest.approx(sigma_A / 10.0, rel=1e-3), (
        f"{name} = {getattr(C, name)} kJ/mol/angstrom^2 = {k_nm2} kJ/mol/nm^2, so its "
        f"thermal width is sqrt(kBT/k) = {math.sqrt(KBT / k_nm2):.6f} nm, not the "
        f"measured {sigma_A / 10.0:.6f} nm")


@pytest.mark.parametrize("name", sorted(MEASURED_ANGLE))
def test_angular_constant_is_kbt_over_sigma_squared_in_radians(name):
    sigma_rad, n, within_rad = MEASURED_ANGLE[name]
    k_expected = KBT / sigma_rad ** 2      # kJ/mol/rad^2
    k = getattr(C, name)
    assert k == pytest.approx(k_expected, rel=2e-3), (
        f"{name} = {k}, but the measured sigma = {sigma_rad} rad over {n} observations "
        f"implies kBT/sigma^2 = {k_expected:.4f} kJ/mol/rad^2 "
        f"(within-chain sigma {within_rad} rad would give {KBT / within_rad ** 2:.4f})")


def test_intra_residue_distances_are_constraints_not_constants():
    """The invariant that replaced the two deleted intra-bond constants.

    The old test asserted K_INTRA_PC != K_INTRA_CN and that their ratio matched the squared
    inverse sigma ratio.  Both were true and neither was evidence: the constants WERE set to
    kBT/sigma^2, so the ratio matched by construction, whatever the sigmas were.  The measured
    thing was the arithmetic.  What is checkable now is that the two names are gone and that
    the two distances are rigid.
    """
    for name in ("K_INTRA_PC", "K_INTRA_CN"):
        assert not hasattr(C, name), (
            f"{name} is readable again. It was deleted, not zeroed, because its sigma "
            f"(0.1055 A / 0.0814 A) is below the 0.1-0.3 A coordinate-error floor, so "
            f"kBT/sigma^2 over it is not a stiffness. A resurrected name is a resurrected "
            f"claim that the number was calibratable.")
    assert not hasattr(C, "K_INTRA"), (
        "a stale shared K_INTRA is still readable; it cannot mean both bonds")


def test_the_two_intra_residue_distances_are_rigid_constraints():
    """2L constraints, at the same 0.390/0.335 nm the torch path holds.

    Read back out of a built System rather than off the source, so the assertion is about
    what OpenMM will actually project onto.  The targets must agree with torch_cgsim's
    make_intra_constraints: two consumers of one 3-bead model disagreeing about a rigid
    distance is a discrepancy nothing else in the suite would catch.
    """
    openmm = pytest.importorskip("openmm")
    L = 6
    p_coords = np.zeros((L, 3), dtype=np.float64)
    p_coords[:, 0] = np.arange(L) * 5.9
    system, _, _, _, _, _ = C._build_3bead_system_gpu(p_coords, [])

    assert system.getNumConstraints() == 2 * L, (
        f"expected 2 constraints per residue (P-C4' and C4'-N), got "
        f"{system.getNumConstraints()}")
    got = {}
    for c in range(system.getNumConstraints()):
        i, j, d = system.getConstraintParameters(c)
        got[(min(i, j), max(i, j))] = _scalar(d)

    for i in range(L):
        assert got[(3 * i, 3 * i + 1)] == pytest.approx(C.BOND_P_C4 / 10.0), (
            f"P({i})-C4'({i}) is constrained to {got.get((3 * i, 3 * i + 1))} nm, "
            f"not BOND_P_C4/10 = {C.BOND_P_C4 / 10.0}")
        assert got[(3 * i + 1, 3 * i + 2)] == pytest.approx(C.BOND_C4_N / 10.0), (
            f"C4'({i})-N({i}) is constrained to {got.get((3 * i + 1, 3 * i + 2))} nm, "
            f"not BOND_C4_N/10 = {C.BOND_C4_N / 10.0}")

    # And the targets must still be live module values, not a frozen copy: perturbing them
    # has to move the System, or BOND_P_C4 is a comment.
    old_pc, old_cn = C.BOND_P_C4, C.BOND_C4_N
    try:
        C.BOND_P_C4, C.BOND_C4_N = 3.80, 3.30
        s2, _, _, _, _, _ = C._build_3bead_system_gpu(p_coords, [])
        d2 = sorted({round(_scalar(s2.getConstraintParameters(c)[2]), 9)
                     for c in range(s2.getNumConstraints())})
        assert d2 == [0.33, 0.38], (
            f"changing BOND_P_C4/BOND_C4_N to 3.80/3.30 left the constraints at {d2}; "
            f"the targets are a frozen copy, so editing the constants would silently do "
            f"nothing")
    finally:
        C.BOND_P_C4, C.BOND_C4_N = old_pc, old_cn


def test_stack_constant_is_nonzero_and_measured():
    sigma_A, n, _ = MEASURED_DISTANCE["K_STACK"]
    assert C.K_STACK > 0.0, (
        "K_STACK is zero, but in this force field N(i) is bonded only to C4'(i), so "
        "N(i)-N(i+1) is positioned by no other term; the identity that makes "
        "torch_cgsim.py's P(i)-P(i+2) stacking term redundant does not apply here")
    assert C.K_STACK == pytest.approx(KBT / sigma_A ** 2, rel=2e-3), (
        f"K_STACK = {C.K_STACK}, measured sigma = {sigma_A} A over {n} observations")


def _scalar(x):
    return x._value if hasattr(x, "_value") else float(x)


def test_built_system_carries_the_measured_constants_in_this_file_units():
    """Read the constants back out of an OpenMM System, atom pair by atom pair."""
    openmm = pytest.importorskip("openmm")
    L = 6
    p_coords = np.zeros((L, 3), dtype=np.float64)
    p_coords[:, 0] = np.arange(L) * 5.9        # 5.9 A apart, the P-P target
    system, _, _, _, _, _ = C._build_3bead_system_gpu(p_coords, [])

    hb = [f for f in system.getForces() if isinstance(f, openmm.HarmonicBondForce)]
    assert len(hb) == 1, (
        "expected exactly one HarmonicBondForce, the backbone. The intra-residue pair used "
        "to be a second one; if a second has reappeared, P-C4'/C4'-N went back to being "
        "springs with constants that cannot be calibrated")
    bonds = {}
    for f in hb:
        for b in range(f.getNumBonds()):
            i, j, r0, kk = f.getBondParameters(b)
            bonds[(min(i, j), max(i, j))] = (_scalar(r0), _scalar(kk))

    r0, kk = bonds[(0, 3)]                      # P(0)-P(1)
    assert kk == pytest.approx(C.K_BB * 100.0), "backbone bond not in kJ/mol/nm^2"
    assert r0 == pytest.approx(C.BOND_P_NEXT / 10.0)

    # P(0)-C4'(0) and C4'(0)-N(0) are NOT here: they are constraints. Locked separately in
    # test_the_two_intra_residue_distances_are_rigid_constraints.
    assert (0, 1) not in bonds and (1, 2) not in bonds, (
        "an intra-residue pair is back in the bonded force list")

    ang = [f for f in system.getForces() if isinstance(f, openmm.HarmonicAngleForce)]
    assert len(ang) == 1
    a_f = ang[0]
    for b in range(a_f.getNumAngles()):
        i, j, k3, theta0, kk = a_f.getAngleParameters(b)
        assert _scalar(kk) == pytest.approx(C.K_ANGLE), "angle constant changed at use"
        assert _scalar(theta0) == pytest.approx(C.ANGLE_PPP)

    dih = [f for f in system.getForces() if isinstance(f, openmm.CustomTorsionForce)]
    assert len(dih) == 1
    d = dih[0]
    assert d.getGlobalParameterName(0) == "k_dih"
    assert d.getGlobalParameterDefaultValue(0) == pytest.approx(C.K_DIHEDRAL)
    assert d.getGlobalParameterName(1) == "theta0"
    assert d.getGlobalParameterDefaultValue(1) == pytest.approx(C.DIH_PPPP)

    stack = None
    for f in system.getForces():
        if isinstance(f, openmm.CustomBondForce) and f.getNumPerBondParameters() == 2 \
                and f.getPerBondParameterName(0) == "k_stack":
            stack = f
    assert stack is not None, "no k_stack CustomBondForce in the built system"
    assert stack.getNumBonds() == L, "stack term is not one per consecutive pair + BSJ"
    seen = set()
    for b in range(stack.getNumBonds()):
        i, j, params = stack.getBondParameters(b)
        seen.add((min(i, j), max(i, j)))
        assert _scalar(params[0]) == pytest.approx(C.K_STACK * 100.0)
        assert _scalar(params[1]) == pytest.approx(C.STACK_R0 / 10.0)
    # the CPU stacks N(i) with N(i+1) -- atom 3i+2 -- not P(i) with P(i+2)
    assert (2, 5) in seen, "the stacking coordinate is no longer N(i)-N(i+1)"
    assert (0, 6) not in seen, (
        "the CPU stacking term now restrains P(i)-P(i+2), the GPU file's coordinate")


def test_a_live_context_holds_the_two_intra_residue_distances():
    """Declaration is not enforcement: run the integrator and measure.

    Everything above reads the built System.  None of it would notice if the constraint were
    declared in the wrong unit, or if the integrator did not apply it.  So this one minimizes
    and then steps, exactly as _run_annealing does, and measures the distances it gets.

    The minimize is not decoration, and it is also not vacuous: measured against
    git HEAD:src/torusfold/scheme2/openmm_gpu_refiner.py -- the OLD model, the two
    HarmonicBondForces at K_INTRA_PC/K_INTRA_CN -- LocalEnergyMinimizer raises "Particle
    coordinate is NaN" from inside this very call, for 3 of 3 seeds at L=30 and again at L=8.
    So with the springs restored this test fails by raising, and if the minimizer were ever
    made to survive it, it would fail on the assertion instead: a spring at kBT/sigma^2 with
    sigma = 0.0106 nm gives a spread three orders of magnitude above the 1e-4 nm bound here.

    The cause is the initial geometry, and it is geometry-independent because it is created in
    the builder: rng.normal(0, 0.3, 3) offsets each C4'/N by ~0.5 A from its P, so P-C4' starts
    near 0.05 nm against a 0.390 nm target.  A constraint projection separates those beads
    before the minimizer ever evaluates the GB pair energy at that separation; a spring does not.

    The NaN is NOT offered as a production fix: the entry point has a retry ladder that catches
    it, and on a compact input HEAD did not NaN at all.  It is recorded here so the behaviour is
    pinned rather than rediscovered.
    """
    openmm = pytest.importorskip("openmm")
    from openmm import unit
    L = 8
    p_coords = np.zeros((L, 3), dtype=np.float64)
    p_coords[:, 0] = np.arange(L) * 5.9
    system, coords_nm, *_ = C._build_3bead_system_gpu(p_coords, [])
    integ = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("CPU"))
    ctx.setPositions(coords_nm * unit.nanometer)

    openmm.LocalEnergyMinimizer.minimize(ctx, maxIterations=2000)
    ctx.setVelocitiesToTemperature(300 * unit.kelvin)
    integ.step(200)

    x = ctx.getState(getPositions=True).getPositions(asNumpy=True)
    x = np.asarray(x.value_in_unit(unit.nanometer)).reshape(L, 3, 3)
    d_pc = np.linalg.norm(x[:, 0] - x[:, 1], axis=-1)
    d_cn = np.linalg.norm(x[:, 1] - x[:, 2], axis=-1)
    assert np.abs(d_pc - C.BOND_P_C4 / 10.0).max() < 1e-4, (
        f"P-C4' drifted to {d_pc.mean():.6f} +/- {d_pc.std():.2e} nm over 200 steps at 300 K, "
        f"target {C.BOND_P_C4 / 10.0}")
    assert np.abs(d_cn - C.BOND_C4_N / 10.0).max() < 1e-4, (
        f"C4'-N drifted to {d_cn.mean():.6f} +/- {d_cn.std():.2e} nm, "
        f"target {C.BOND_C4_N / 10.0}")


def test_base_beads_are_connected_only_by_the_stack_bond():
    """Measured structure of the built force field, and why K_STACK cannot be zero.

    In this file N(i) is attached only to C4'(i) -- now by a rigid constraint rather than by
    a bond, which is why the constraint list is enumerated alongside the bonded forces; the
    P(i)-P(i+2) identity that makes torch_cgsim.py's stacking term redundant does not exist
    for N(i)-N(i+1).  So enumerate every term that references a base bead in a system built
    with no base pairs: the only neighbours of N(i) must be C4'(i) and N(i +/- 1) (+ the BSJ
    closure partner for i = 0 and i = L-1).  With K_STACK = 0 the base beads would hang off
    the backbone attached by one rigid tether each, and nothing else would position them
    relative to one another.
    """
    openmm = pytest.importorskip("openmm")
    L = 6
    p_coords = np.zeros((L, 3), dtype=np.float64)
    p_coords[:, 0] = np.arange(L) * 5.9
    system, _, _, _, _, _ = C._build_3bead_system_gpu(p_coords, [])

    n_beads = {3 * i + 2 for i in range(L)}
    partners = {n: set() for n in n_beads}
    for f in system.getForces():
        if isinstance(f, (openmm.HarmonicBondForce, openmm.CustomBondForce)):
            for b in range(f.getNumBonds()):
                i, j = f.getBondParameters(b)[0], f.getBondParameters(b)[1]
                if i in partners:
                    partners[i].add(j)
                if j in partners:
                    partners[j].add(i)
    for c in range(system.getNumConstraints()):
        i, j = system.getConstraintParameters(c)[0], system.getConstraintParameters(c)[1]
        if i in partners:
            partners[i].add(j)
        if j in partners:
            partners[j].add(i)

    for i in range(L):
        expected = {3 * i + 1}                                    # C4'(i), by constraint
        if i > 0:
            expected.add(3 * (i - 1) + 2)                         # N(i-1)
        if i < L - 1:
            expected.add(3 * (i + 1) + 2)                         # N(i+1)
        if i == 0:
            expected.add(3 * (L - 1) + 2)                         # BSJ stacking partner
        if i == L - 1:
            expected.add(2)                                       # BSJ stacking partner
        assert partners[3 * i + 2] == expected, (
            f"N({i}) bonded to {sorted(partners[3 * i + 2])}, expected "
            f"{sorted(expected)}; with no base pairs nothing else may position the "
            f"base bead, which is why K_STACK is not zero here")

