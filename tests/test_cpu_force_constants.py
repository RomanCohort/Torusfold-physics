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
MEASURED_DISTANCE = {
    "K_BB": (0.4714, 6638, 0.4454),
    "K_INTRA_PC": (0.1055, 6764, 0.0917),
    "K_INTRA_CN": (0.0814, 6764, 0.0736),
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


def test_intra_bond_constants_differ_as_the_measurement_requires():
    pc, _, _ = MEASURED_DISTANCE["K_INTRA_PC"]
    cn, _, _ = MEASURED_DISTANCE["K_INTRA_CN"]
    assert C.K_INTRA_PC != C.K_INTRA_CN, (
        "P-C4' and C4'-N are back on one shared constant; their measured spreads "
        f"({pc} A vs {cn} A over 6764 observations each) differ by {cn / pc:.3f}x")
    assert C.K_INTRA_CN / C.K_INTRA_PC == pytest.approx((pc / cn) ** 2, rel=2e-3), (
        f"C4'-N/P-C4' stiffness ratio {C.K_INTRA_CN / C.K_INTRA_PC:.4f} does not match "
        f"the squared inverse sigma ratio {(pc / cn) ** 2:.4f}")
    assert not hasattr(C, "K_INTRA"), (
        "a stale shared K_INTRA is still readable; it cannot mean both bonds")


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
    assert len(hb) == 2, "expected the backbone bond force and the intra-residue one"
    bonds = {}
    for f in hb:
        for b in range(f.getNumBonds()):
            i, j, r0, kk = f.getBondParameters(b)
            bonds[(min(i, j), max(i, j))] = (_scalar(r0), _scalar(kk))

    r0, kk = bonds[(0, 3)]                      # P(0)-P(1)
    assert kk == pytest.approx(C.K_BB * 100.0), "backbone bond not in kJ/mol/nm^2"
    assert r0 == pytest.approx(C.BOND_P_NEXT / 10.0)

    r0, kk = bonds[(0, 1)]                      # P(0)-C4'(0)
    assert kk == pytest.approx(C.K_INTRA_PC * 100.0)
    assert r0 == pytest.approx(C.BOND_P_C4 / 10.0)

    r0, kk = bonds[(1, 2)]                      # C4'(0)-N(0)
    assert kk == pytest.approx(C.K_INTRA_CN * 100.0)
    assert r0 == pytest.approx(C.BOND_C4_N / 10.0)

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


def test_base_beads_are_connected_only_by_the_stack_bond():
    """Measured structure of the built force field, and why K_STACK cannot be zero.

    In this file N(i) is bonded only to C4'(i); the P(i)-P(i+2) identity that makes
    torch_cgsim.py's stacking term redundant does not exist for N(i)-N(i+1).  So
    enumerate the bonded terms that reference a base bead in a system built with no
    base pairs: the only neighbours of N(i) must be C4'(i) and N(i +/- 1) (+ the BSJ
    closure partner for i = 0 and i = L-1).  With K_STACK = 0 the base beads would
    hang off the backbone attached by one bond each, at an unconstrained distance.
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

    for i in range(L):
        expected = {3 * i + 1}                                    # C4'(i)
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

