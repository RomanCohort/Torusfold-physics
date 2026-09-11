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
  K_CLASH      no criterion for the stiffness.  Over 2022024 bead pairs with |bead index
               gap| >= 3 -- the field's own clash set, verified pair-for-pair against its
               neighbor list -- the closest approach is 0.3092 nm.  The 0.30 nm linear
               spring that used to be here sat below it and was identically zero over the
               whole database, so P(data | K_CLASH) was constant; the range is now 0.3975,
               which is above that closest approach, so 167 of the 2022024 pairs do sit
               inside it.  The database still bounds the range and not the stiffness
               (scripts/measure_type_pair_excluded_volume.py).
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

import inspect

KBT = 2.494                # kJ/mol at 300 K
# read live rather than pinned: the cap was raised from 200 to its measured physical value, and
# a hardcoded 200 here would have gone stale silently
FORCE_CAP = inspect.signature(
    C.cg_energy_forces).parameters["force_cap"].default
SIGMA_NN = 0.1103          # nm, 3059 accepted WC pairs over 99 chains
MEAN_ABS_OFFSET = 0.1063   # nm, mean |d - PAIR_NN| over the same 3059 pairs
CLASH_CLOSEST = 0.3092     # nm, closest of 2022024 bead pairs with |i-j| >= 3, 126 chains
CLASH_SET_B_N = 2022024    # pairs in that set
CLASH_INSIDE = 167         # of them, how many are inside the shipped CLASH_SIGMA = 0.3975 (0.00826%)
RANK_UPPER = 2173.9        # kJ/mol/nm^2, tightest K* over 35 structures on the register-shift-by-1
                           # decoy; above it the decoy starts preferring the wrong register
                           # (scripts/select_k_pair_by_ranking.py)
PP_MEAN = 0.5923           # nm, 6638 P(i)-P(i+1) bonds
PP_SIGMA = 0.0471          # nm, same 6638 bonds
E2E_SD = 2.467             # nm, |P(0)-P(L-1)| over 126 free-ended chains


def test_pair_spring_lies_between_the_spread_bound_and_the_ranking_bound():
    """K_PAIR has a measured LOWER bound, a measured UPPER bound, and still no measured value.

    The upper bound used to come from the force cap: at 200 kJ/mol/nm with a mean |d - 1.0| of
    0.1063 nm the pair term alone could not exceed 1881 without saturating it. Raising the cap to
    the value the field's own undamaged forces require made that bound vacuous -- 5000/0.1063 is
    47035, forty-seven times the criterion value -- and this test used to assert the vacuity.

    scripts/select_k_pair_by_ranking.py replaces it with a ranking measurement. On a register-shift
    decoy -- the same residues paired to members of the same helix, slid one position, so every
    distance stays plausible and only the chemistry changes -- the correct register wins for all 35
    usable structures while K_PAIR is below 2173.9, and structures start to drop above it: 32/35 at
    2500, 21/35 at 4000, 10/35 at 8000. The decoy's own N-N mean is 1.1426 nm, closer to PAIR_NN
    than the native's 0.9402 nm, which is exactly why it turns over instead of being monotone.

    What the decoy does NOT do, and this is the honest part: it separates neither shipped candidate
    from the other. 600 and 1500 both win 35/35. The random re-pairing decoy bounds nothing from
    above at all -- its coefficient A is positive for all 40 structures, so its win count rises with
    K_PAIR without limit. The bracket is therefore [204.8, 2173.9] and choosing inside it still
    needs a measurement this suite does not have.
    """
    k_lower = KBT / SIGMA_NN ** 2
    k_cap = FORCE_CAP / MEAN_ABS_OFFSET
    assert k_lower == pytest.approx(205.0, rel=5e-3), (
        f"sigma_NN = {SIGMA_NN} nm should imply kBT/sigma^2 = 204.8; got {k_lower:.1f}")
    assert k_cap > RANK_UPPER, (
        f"the cap bound ({k_cap:.0f}) is no longer looser than the ranking bound "
        f"({RANK_UPPER:.1f}), so the cap has become the operative constraint again. That is not "
        f"wrong, but it means the vacuity this test used to record is back in the other "
        f"direction and the ranking measurement has stopped being the thing that bounds K_PAIR")
    assert k_lower <= C.K_PAIR <= RANK_UPPER, (
        f"K_PAIR = {C.K_PAIR} is outside [{k_lower:.1f}, {RANK_UPPER:.1f}]. The lower end is the "
        f"spread criterion (kBT/sigma_NN^2, a bound and not a value); the upper end is where the "
        f"register-shift decoy starts losing structures. Moving it out needs a new measurement.")
    # and it is not the criterion value itself: 600 = 2.9x the bound
    assert C.K_PAIR != pytest.approx(k_lower, rel=0.10)
    # the decoy rules out NEITHER shipped candidate, which is the claim that matters: both 600 and
    # 1500 sit under 2173.9. The margin for 1500 is only 1.45x, so it is pinned both ways.
    assert 1500.0 < RANK_UPPER, (
        f"the ranking bound {RANK_UPPER:.1f} is at or below the predecessor value 1500, so this "
        f"decoy DOES separate 600 from 1500 and the bracket above is narrower than claimed")
    assert RANK_UPPER / 1500.0 == pytest.approx(1.45, abs=0.05), (
        f"the bound is {RANK_UPPER / 1500.0:.2f} times the predecessor value; the margin for 1500 "
        f"has moved and the prose about it needs re-reading")


def test_the_excluded_volume_range_and_what_it_costs_on_native_geometry():
    """The range is set by the P-P minimum, and the set it acts on reaches below that.

    This replaces an assertion that the clash term was identically zero over the whole database.
    That was true of the retired 0.30 nm linear spring -- set B's closest approach is 0.3092 nm --
    and it stopped being true when the range became 0.3975. The cost is real and is pinned here so
    it cannot grow unnoticed: 167 of the 2022024 pairs in set B sit inside the range, all of them
    C4'-C4' (own minimum 0.3092), C4'-N9/N1 (0.3673) or N9/N1-N9/N1 (0.3411).

    A per-type range was considered against this number and rejected. It would remove 0.008
    percent of contacts, and the Boltzmann criterion that set K_CLASH cannot supply a per-type
    stiffness at all, because sigma is defined as the type's minimum and the bin below it is
    therefore empty for every one of the six type pairs
    (scripts/measure_type_pair_excluded_volume.py prints that result per type).
    """
    assert C.CLASH_SIGMA == pytest.approx(0.3975), (
        "the range moved; the cost below is measured against 0.3975")
    assert CLASH_CLOSEST == 0.3092, "set B's closest approach moved; re-measure the cost"
    assert CLASH_CLOSEST < C.CLASH_SIGMA, (
        f"set B's closest ({CLASH_CLOSEST}) is no longer inside the range "
        f"({C.CLASH_SIGMA}), so the cost this test pins does not exist")
    assert CLASH_INSIDE == 167, (
        f"{CLASH_INSIDE} pairs of set B are inside {C.CLASH_SIGMA} nm; the measurement said 167")
    assert CLASH_INSIDE / CLASH_SET_B_N < 1e-4, (
        f"the share inside the range is {CLASH_INSIDE / CLASH_SET_B_N:.6f}, above the 1e-4 the "
        f"single-range decision was taken on; a per-type range may now be worth its cost")
    # CLASH_DIST is retired and nothing reads it. tests/test_clash_single_potential.py owns the
    # check that no code outside the four call sites names it, so this only pins the value.
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
