"""Two denominators, two numbers, and the difference between them is a real floor.

Measured 2026-09-21 (docs/ibi_loop_and_oxrna_findings.md Part 5): boltzmann_bonded stores
table["sigma"] as the PLAIN standard deviation of every database observation, while the table's
support is built from a ROBUST sigma that excludes the non-physical tail. For bb_bond that makes
sigma_data 0.0633 against the table's own 0.0540 -- a ratio of 0.853 -- so sim/ref for the bond
could never exceed 0.853 and every J quoted for it carried |ln 0.853| = 0.159 of floor. These tests
pin the second denominator, which is a REPORTING quantity: nothing in the update path reads it.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_core as IC                 # noqa: E402

KBT = B.KBT
GRID = np.linspace(0.001, 0.999, 1000)          # one grid everywhere: centre[i] pairs with U[i]


def _table(centre, p, sigma):
    binw = float(centre[1] - centre[0])
    return {"lo": float(centre[0] - binw / 2), "hi": float(centre[-1] + binw / 2), "binw": binw,
            "U": -KBT * np.log(p / p.max()), "centre": centre, "sigma": float(sigma)}


def _acc(q, p):
    """A moments accumulator for the distribution p on the bin centres q."""
    p = p / p.sum()
    n = 1_000_000
    return [float((p * q).sum()) * n, float((p * q ** 2).sum()) * n, n]


def _gauss(q, width, centre=0.5):
    p = np.exp(-0.5 * ((q - centre) / width) ** 2)
    return p / p.sum()


def _sigma(q, p):
    m = float((p * q).sum())
    return float(np.sqrt((p * q ** 2).sum() - m * m))


def test_implied_sigma_is_the_tables_own_width():
    p = _gauss(GRID, 0.08)
    assert abs(IC.implied_sigma(_table(GRID, p, sigma=0.1)) - _sigma(GRID, p)) < 2e-3


def test_a_matching_simulation_scores_one_against_the_table_and_below_it_against_the_data():
    """The floor in one assertion: sample exactly what the table implies."""
    p = _gauss(GRID, 0.03)
    sigma_table = IC.implied_sigma(_table(GRID, p, sigma=0.1))
    tab = {"bb_bond": _table(GRID, p, sigma=sigma_table / 0.85)}   # a sigma 15 percent too large
    acc = {"bb_bond": _acc(GRID, p)}
    r_data = IC.sim_ref_ratio(acc, "bb_bond", tab)
    r_table = IC.sim_ref_ratio_table(acc, "bb_bond", tab)
    assert abs(r_table - 1.0) < 1e-2, f"a matching sample must score 1.000, got {r_table}"
    assert abs(r_data - 0.85) < 1e-2, f"against the inflated sigma it scores 0.85, got {r_data}"


def test_the_two_metrics_differ_by_exactly_the_sigmas_ratio():
    for name, width in (("bb_bond", 0.04), ("angle", 0.10), ("dihedral", 0.20), ("stack", 0.06)):
        p = _gauss(GRID, width)
        sigma_data = 0.3
        tab = {name: _table(GRID, p, sigma=sigma_data)}
        acc = {name: _acc(GRID, p)}
        a = IC.sim_ref_ratio(acc, name, tab)
        b = IC.sim_ref_ratio_table(acc, name, tab)
        sigma_table = IC.implied_sigma(tab[name])
        assert a > 0 and b > 0
        assert abs((b / a) - (sigma_data / sigma_table)) < 1e-9, (name, a, b, sigma_table)


def test_simref_table_has_the_same_shape_as_simref_and_a_smaller_j():
    p = _gauss(GRID, 0.08)
    tab = {c: _table(GRID, p, sigma=0.2) for c in B.COORDS}
    acc = {c: _acc(GRID, p) for c in B.COORDS}
    v1, j1 = IC.simref(acc, tab)
    v2, j2 = IC.simref_table(acc, tab)
    assert len(v1) == len(v2) == len(B.COORDS)
    assert all(b > a for a, b in zip(v1, v2)), "the table's sigma is smaller here, so its ratio is larger"
    assert j2 < j1, "and its J is smaller -- that is the floor being removed"
    v3, _ = IC.simref_table(acc, tab, skip=("bb_bond",))
    assert v3[0] != v3[0], "a skipped coordinate stays nan"
