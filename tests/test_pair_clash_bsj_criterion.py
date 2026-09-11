r"""Lock what the structure database can and cannot say about the four uncalibrated
constants K_PAIR, K_CLASH, K_BSJ and K_BSJ_GUIDE.

scripts/measure_pair_clash_bsj_constants.py measures D:\torusfold-cgdata\rsRNASP\
Training_set for the coordinates those four constants restrain.  The numbers here are its
output; that script is the authority.  These tests exist so a constant cannot be moved
away from the measurement without the suite saying so -- the same role
tests/test_cpu_force_constants.py plays for the constants that do have a criterion -- and
so that the negative results are locked as facts rather than as prose.

What was measured, one line per constant:

  K_PAIR       the criterion applies.  sigma_NN = 0.1103 nm over 3059 accepted
               Watson-Crick pairs in 126 gap-free chains, so kBT/sigma^2 = 204.8
               kJ/mol/nm^2.  That is a LOWER BOUND (each entry is one conformation, so the
               pooled spread mixes sequence and conformer variation into the thermal
               width), not a value.  A second bound runs the other way and is also
               measured: the pair term's own force at native geometry (mean |d - 1.0 nm| =
               0.1063) must stay under the 200 kJ/mol/nm cap that cg_energy_forces applies,
               which gives k <= 1881 kJ/mol/nm^2.
  K_CLASH      no criterion.  Over 2022024 bead pairs with |bead index gap| >= 3 -- the
               field's own clash set, verified pair-for-pair against its neighbor list --
               the closest approach is 0.3092 nm and nothing lies inside the 0.30 nm
               cutoff.  P(data | K_CLASH) is therefore constant for every K_CLASH >= 0.
               The database bounds the cutoff, not the stiffness.
  K_BSJ        no independent criterion.  No deposited chain is covalently closed (1 of
               126 has ends within 0.7 nm), so the coordinate the term restrains is a free
               end-to-end distance (sd 2.467 nm) whose kBT/sd^2 = 0.41 means nothing.  The
               database gives the target -- mean P(i)-P(i+1) = 0.5923 nm over 6638 bonds --
               and by the phosphodiester identity a transferable stiffness of 1122.
  K_BSJ_GUIDE  no criterion.  E(d) = -K * softplus(-(r0 - d)/0.2) is strictly monotone in
               d, so there is no equilibrium point and no curvature for kBT/sigma^2 to
               match; its scale is an annealing-schedule choice.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C  # noqa: E402

KBT = 2.494                # kJ/mol at 300 K
FORCE_CAP = 200.0          # kJ/mol/nm, cg_energy_forces force_cap
SIGMA_NN = 0.1103          # nm, 3059 accepted WC pairs over 99 chains
MEAN_ABS_OFFSET = 0.1063   # nm, mean |d - PAIR_NN| over the same 3059 pairs
CLASH_CLOSEST = 0.3092     # nm, closest of 2022024 bead pairs with |i-j| >= 3, 126 chains
PP_MEAN = 0.5923           # nm, 6638 P(i)-P(i+1) bonds
PP_SIGMA = 0.0471          # nm, same 6638 bonds
E2E_SD = 2.467             # nm, |P(0)-P(L-1)| over 126 free-ended chains


def test_pair_spring_lies_between_the_spread_bound_and_the_cap_bound():
    """K_PAIR is bracketed by two measured bounds, but nothing measured selects 600."""
    k_lower = KBT / SIGMA_NN ** 2
    k_cap = FORCE_CAP / MEAN_ABS_OFFSET
    assert k_lower == pytest.approx(205.0, rel=5e-3), (
        f"sigma_NN = {SIGMA_NN} nm should imply kBT/sigma^2 = 204.8; got {k_lower:.1f}")
    assert k_cap == pytest.approx(1881.0, rel=5e-3), (
        f"mean |d - 1.0| = {MEAN_ABS_OFFSET} nm against a {FORCE_CAP} kJ/mol/nm cap "
        f"should imply k <= 1881; got {k_cap:.1f}")
    assert k_lower <= C.K_PAIR <= k_cap, (
        f"K_PAIR = {C.K_PAIR} is outside [{k_lower:.1f}, {k_cap:.1f}].  It is not what the "
        f"criterion gives (that is {k_lower:.1f}, a bound); it is only allowed inside the "
        f"bracket.  Moving it out needs a fold/ranking measurement, not a new prefactor.")
    # and it is not the criterion value itself: 600 = 2.9x the bound
    assert C.K_PAIR != pytest.approx(k_lower, rel=0.10)


def test_clash_cutoff_sits_below_the_closest_native_contact():
    """CLASH_DIST is bounded above by the closest native approach; K_CLASH is not bounded
    at all, because the term is identically zero over the whole database."""
    margin = CLASH_CLOSEST - C.CLASH_DIST
    assert margin > 0.0, (
        f"the measured closest bead approach is {CLASH_CLOSEST} nm against a "
        f"{C.CLASH_DIST} nm cutoff: native geometry now sits on the clash wall, and "
        f"K_CLASH would have to be re-derived from a distribution it does not have")
    assert margin < 0.02, (
        f"margin is {margin:.4f} nm -- larger than the measured 0.0092 nm, which means "
        f"either the database or the cutoff changed and the earlier statement that the "
        f"term never fires needs re-measuring")
    # 0.30 < 0.3092 < 0.310: exactly one pair in the database is within 0.01 nm of the wall
    assert C.CLASH_DIST == pytest.approx(0.30)


def test_bsj_guide_is_monotone_so_kbt_over_sigma_squared_cannot_price_it():
    """The guide's functional form is what makes it unmeasurable from a spread."""
    d = torch.linspace(0.4, 3.0, 256, dtype=torch.float64)
    e = -C.K_BSJ_GUIDE * C._stable_softplus(-(C.PAIR_NN - d) / 0.2)
    assert bool((e[1:] < e[:-1]).all()), (
        "the BSJ guide is no longer strictly decreasing in d; if it has gained an "
        "equilibrium point then kBT/sigma^2 may apply to it and this test is stale")


def test_bsj_stiffness_is_a_transfer_from_the_phosphodiester_bond():
    """The database sets the BSJ target; the stiffness can only be transferred."""
    assert C.BOND_P_NEXT == pytest.approx(PP_MEAN, abs=0.01), (
        f"BOND_P_NEXT = {C.BOND_P_NEXT} is not the measured P-P bond {PP_MEAN} nm")
    k_transfer = KBT / PP_SIGMA ** 2
    assert k_transfer == pytest.approx(1122.0, rel=5e-3)
    assert KBT / E2E_SD ** 2 < 1.0, (
        "the free end-to-end coordinate no longer gives a value far below the transfer; "
        "if the deposited chains became closed, K_BSJ could be measured directly")
    assert k_transfer > 10.0 * (KBT / E2E_SD ** 2), (
        "the answer must depend on which coordinate is chosen -- that is the finding")
