"""The stored tables' interpolation gap, pinned, with the arithmetic checked independently.

boltzmann_bonded._sample reads U_i = -kBT ln p_i as the potential AT the bin centre when those
numbers are bin MASSES. The gap is kBT*ln(P_model/P_target) under the interpolant, and because the
tables are what the IBI path actually loads, the gap that matters is the one from
results/boltzmann_tables_clean.npz, not from a fresh fit.

The numbers are larger than an earlier reading of 0.36/0.49/0.29 kBT, and the difference is not a
bug: that reading came from B.fit(load_structures()) -- tables fitted on the fly -- and the stored
npz was produced by a different fit. Both are real; the stored one is the one IBI reads.
"""
import pathlib
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

gap = pytest.importorskip("table_convention_gap")


def test_the_mass_integral_agrees_with_a_brute_force_integration_of_the_same_interpolant():
    """Independent check, so the pinned numbers below are not pinning a bug.

    bin_masses() integrates exp(-U/kBT) in closed form over each half-bin. The same quantity is
    computed here by sampling _sample's own interpolation on a fine grid and summing, which shares
    no code with bin_masses.
    """
    z = np.load(gap.NPZ)
    u = np.asarray(z["angle__U"], dtype=np.float64)
    binw = float(z["angle__binw"])
    lo = float(z["angle__lo"])

    # _sample's mapping, reproduced: u = (q - lo)/binw - 0.5, clamped
    q = np.linspace(lo, lo + binw * len(u), 400001)
    uu = (q - lo) / binw - 0.5
    i0 = np.clip(np.floor(uu).astype(np.int64), 0, len(u) - 2)
    f = np.clip(uu - i0, 0.0, 1.0)
    interp = u[i0] + f * (u[i0 + 1] - u[i0])
    # bin edges in q
    dens = np.exp(-interp / gap.KBT)
    h = q[1] - q[0]
    # a Riemann SUM of the density, not yet an integral: the h factor is the difference
    brute = h * np.array([dens[(q >= lo + i * binw) & (q < lo + (i + 1) * binw)].sum()
                          for i in range(len(u))])
    closed = gap.bin_masses(u, binw)
    rel = np.abs(brute - closed) / np.maximum(closed, 1e-300)
    assert rel.max() < 1e-3, (
        f"the closed-form bin masses disagree with a fine-grid integration of the same "
        f"interpolant by up to {rel.max():.3e} relative; one of the two is wrong")


def test_the_stored_tables_interpolation_gap_is_where_it_was_recorded():
    """Pinned so a change to _sample or a refit of the npz shows up here rather than later.

    Measured from results/boltzmann_tables_clean.npz:
        angle      core max 2.6059 kJ/mol = 1.045 kBT   weighted rms 0.0350 kBT
        dihedral   core max 4.4898 kJ/mol = 1.800 kBT   weighted rms 0.1470 kBT
        stack      core max 0.9127 kJ/mol = 0.366 kBT   weighted rms 0.0330 kBT
    """
    pinned = {"angle": (2.6059, 0.0874), "dihedral": (4.4898, 0.3667),
              "stack": (0.9127, 0.0823)}
    for coord, (mx, rms) in pinned.items():
        got_max, got_rms = gap.gaps(coord)
        assert got_max == pytest.approx(mx, rel=5e-3), (
            f"{coord}: the core maximum gap is {got_max:.4f} kJ/mol against the recorded "
            f"{mx:.4f}; either the stored table or _sample's interpolation changed")
        assert got_rms == pytest.approx(rms, rel=5e-2), f"{coord} weighted rms moved"


def test_the_gap_is_not_zero_so_this_test_is_not_vacuous():
    for coord in gap.TABLED:
        mx, rms = gap.gaps(coord)
        assert mx > 0.1 and rms > 0.01, (
            f"{coord}'s gap came out at {mx:.4f} / {rms:.4f}; if the interpolation convention "
            f"was ever made consistent this test should be rewritten, not allowed to pass on zero")
