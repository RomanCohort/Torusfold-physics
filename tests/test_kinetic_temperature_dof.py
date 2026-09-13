"""The kinetic-temperature degree-of-freedom count, and the trap it sets.

Every temperature estimator in this repo divided by 3N. That is the count of independent
velocity components only when nothing is constrained. A distance constraint removes one, so a
system with C of them has 3N - C, and evaluating the 3N form anyway does not fail loudly -- it
reports a temperature low by (3N-C)/3N.

The arithmetic that matters here: this model's intra-residue rigidification adds 2L
constraints to a 3L-residue chain, i.e. C = 2L on 3N = 9L beads. The naive estimator would
then read 7/9 = 0.778 of the true temperature, so a run overshooting at 440 K would appear to
"improve" to 342 K with no physical change at all. That is the failure this file exists to
prevent from coming back, so both numbers are pinned, not just the corrected one.

Run: python -m pytest tests/test_kinetic_temperature_dof.py
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

C = pytest.importorskip("torusfold.scheme2.torch_cgsim")

MASS = 110.0
# The chain lengths the pipeline actually runs, plus both dtypes: the batched production path
# is float32 (BatchedREMD2D converts on the way in) while the diagnostics use float64.
LENGTHS = [27, 200, 2013]
DTYPES = [torch.float32, torch.float64]


def _vel(L, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(4, 3 * L, 3, dtype=dtype, generator=g) * 0.15


@pytest.mark.parametrize("L", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_constraints_is_bit_identical_to_the_bare_three_n_form(L, dtype):
    """The correction must cost nothing when there is nothing to correct.

    Without this the change is not landable: every number in the diagnostics would move for
    a reason unrelated to any physics, and no before/after comparison would mean anything.
    """
    v = _vel(L, dtype)
    bare = MASS * (v ** 2).sum(dim=-1) / (3.0 * C.KB_KJ)
    assert torch.equal(C.kinetic_temperature(v, MASS, 0), bare), (
        f"L={L} {dtype}: n_constraints=0 is not bit-identical to the 3N form")


@pytest.mark.parametrize("L", LENGTHS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_constraints_scale_the_temperature_by_the_dof_ratio(L, dtype):
    v = _vel(L, dtype)
    n_beads = 3 * L
    for c in (1, L, 2 * L):
        want = n_beads * 3 / (n_beads * 3 - c)
        got = float((C.kinetic_temperature(v, MASS, c) /
                     C.kinetic_temperature(v, MASS, 0)).mean())
        assert abs(got - want) < 1e-5, f"L={L} C={c}: scaling {got} != {want}"


def test_the_naive_estimator_reads_0_778_of_the_true_temperature():
    """Pin BOTH numbers, so the artefact cannot return disguised as a fix.

    If a future change makes the naive 3N value agree with the corrected one here, that is
    not a rounding improvement -- it means the constraint count stopped being subtracted.
    """
    L = 100
    v = _vel(L, torch.float64)
    n_constraints = 2 * L                      # P-C4' and C4'-N, one pair each per residue
    corrected = float(C.kinetic_temperature(v, MASS, n_constraints).mean())
    naive = float(C.kinetic_temperature(v, MASS, 0).mean())

    assert abs(naive / corrected - 7.0 / 9.0) < 1e-9, (
        "the naive 3N estimator no longer reads 7/9 of the corrected one; the DOF correction "
        "is not being applied")
    assert naive < corrected, "the uncorrected estimator must read LOW, never high"


def test_a_wrong_constraint_count_is_not_silently_tolerated():
    """Guard the boundary: 3N - C must stay positive."""
    v = _vel(2, torch.float64)
    with pytest.raises(ZeroDivisionError):
        C.kinetic_temperature(v, MASS, 3 * v.shape[-2])
