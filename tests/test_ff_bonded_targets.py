"""The bonded targets the force field ships with, and the freeze that let them diverge.

Three constants in torch_cgsim.py did not match native RNA geometry: STACK_R0 restrained
P(i)-P(i+2) to 0.505 nm when that pair is 1.12 nm apart, and DIH_PPPP restrained the
P-P-P-P pseudo-torsion to 180 deg when native RNA sits near 0. They cost the field its
ability to prefer a native structure: over 32 held-out structures the native ranked 6.75th
of 7 candidates and the best random decoy scored 19592.6 kJ/mol LOWER.

The third bug was structural rather than numerical. STACK_R0 was frozen into a module
constant _R0_STACK at import, and cg_forces_explicit_batched read the frozen copy while
cg_energy_forces read the live global -- so changing the constant would have fixed one path
and silently left the other. That freeze is gone and these tests keep it gone, because the
failure mode is invisible: nothing raises, the two paths just disagree.

Run: python tests/test_ff_bonded_targets.py
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C


def _chain(L=8):
    """A straight, evenly spaced backbone in nm, enough to exercise every bonded term."""
    pos = np.zeros((L, 3, 3))
    for i in range(L):
        base = np.array([0.59 * i, 0.0, 0.0])
        pos[i, 0] = base
        pos[i, 1] = base + np.array([0.39 * 0.3, 0.39 * 0.95, 0.0])
        pos[i, 2] = pos[i, 1] + np.array([0.335 * 0.2, 0.335 * 0.9, 0.335 * 0.4])
    tensor = torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)
    pairs = torch.zeros((0, 2), dtype=torch.long)
    return tensor, pairs


def _energies(pos, pairs):
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    e_unified = float(C.cg_energy_forces(pos, pairs, None, cell_list=cl)[0])
    e_batched = float(C.cg_forces_explicit_batched(pos, pairs, None, cell_list=cl)[0])
    return e_unified, e_batched


def test_shipped_targets_match_measurement():
    assert abs(C.STACK_R0 - 1.125) < 1e-12, (
        f"STACK_R0 is {C.STACK_R0}, expected 1.125 nm -- the mode of P(i)-P(i+2) over "
        f"10874 observations from 191 deposited structures")
    assert abs(math.cos(C.DIH_PPPP) - 0.975) < 1e-12, (
        f"cos(DIH_PPPP) is {math.cos(C.DIH_PPPP):.6f}, expected +0.975 -- the mode of the "
        f"P-P-P-P pseudo-torsion cosine")
    # The angle was measured too and needed no change: the native mode is cos -0.875.
    assert abs(math.cos(C.ANGLE_PPP) - (-0.866)) < 0.02, (
        f"cos(ANGLE_PPP) is {math.cos(C.ANGLE_PPP):.6f}; the native mode is -0.875, so it "
        f"should still be near -0.866 and should not have been moved to the mean")


def test_stacking_target_survives_the_pp_scale_check():
    # A floor, not a tight bound: if anyone re-derives STACK_R0 from a P(i)-P(i+1)-scale
    # quantity (0.59 nm) or copies the OpenMM N(i)-N(i+1) value (0.505 nm) back in, this
    # fails loudly instead of quietly collapsing the backbone again.
    assert C.STACK_R0 > 1.0, (
        f"STACK_R0 is {C.STACK_R0} nm, which is a P(i)-P(i+1)-scale distance; P(i)-P(i+2) "
        f"is about 1.12 nm")


def test_backbone_stiffness_matches_the_observed_spread():
    # k = kBT/(sigma^2) with sigma from 96 gap-free chains: 0.2978 for the angle cosine and
    # 0.5880 for the dihedral cosine. These were 600.0 and 500.0, which by the ablation in
    # ablate_backbone_terms.py did no fold work while putting 27 percent of beads over the
    # force cap.
    assert abs(C.K_ANGLE - 28.1) < 1e-9, (
        f"K_ANGLE is {C.K_ANGLE}, expected kBT/sigma^2 = 28.1 kJ/mol/nm")
    assert abs(C.K_DIH - 7.2) < 1e-9, (
        f"K_DIH is {C.K_DIH}, expected kBT/sigma^2 = 7.2 kJ/mol/nm")


def test_maxwell_boltzmann_check_is_not_thirty_times_off():
    # A guard against a silent return to the tuned values. The exact numbers matter less
    # than the order of magnitude, which is what the cap saturation depended on.
    assert C.K_ANGLE < 100.0, f"K_ANGLE {C.K_ANGLE} is back in the tuned range"
    assert C.K_DIH < 100.0, f"K_DIH {C.K_DIH} is back in the tuned range"


def test_stacking_term_is_disabled_as_redundant():
    # |P(i)-P(i+2)|^2 = |b_i|^2 + |b_{i+1}|^2 - 2|b_i||b_{i+1}|cos_a, exactly, with R^2 of
    # 1.000000 over 1278 windows. Both bonds are restrained by K_BB and cos_a by K_ANGLE, so
    # this term was a third spring on a derived quantity: ablating it left the funnel rank
    # and gap untouched at 1.000 and 0.0 and took cap saturation from 4.02 to 0.58 percent.
    assert C.K_STACK == 0.0, (
        f"K_STACK is {C.K_STACK}; it is redundant with K_BB and K_ANGLE and was set to zero")
    # the target is kept only as a record of the coordinate's mode
    assert abs(C.STACK_R0 - 1.125) < 1e-12


def test_bond_stiffness_was_not_moved_to_the_variance_matching_value():
    # kBT/sigma^2 would be 1076 for the P-P bond, but applying it raises cap saturation from
    # 0.58 to 1.61 percent with stacking off. The variance-matching criterion that worked for
    # the angle and dihedral does not transfer to a coordinate with a heavy-tailed spread.
    assert C.K_BB < 800.0, (
        f"K_BB is {C.K_BB}; the variance-matching value 1076 makes cap saturation worse")


def test_bpp_stiffness_is_not_five_times_the_force_cap():
    # At the target distance the softplus derivative is sigmoid(0) = 0.5, so this term
    # applies K_BPP / 0.6 to every pair regardless of geometry. The cap in cg_energy_forces
    # is 200, and 600/0.6 = 1000, which is why bpp alone saturated 22.31 percent of beads.
    force_at_target = C.K_BPP / 0.6
    assert force_at_target < 200.0, (
        f"bpp applies {force_at_target:.0f} kJ/mol/nm at its own target distance, above the "
        f"200 cap")
    # 0.6 * kBT / sigma_NN with sigma_NN = 0.112 nm over 561 observed pairs
    assert abs(C.K_BPP - 13.4) < 1e-9, (
        f"K_BPP is {C.K_BPP}, expected 0.6 * kBT / sigma_NN = 13.4")


def test_no_frozen_snapshot_of_the_stacking_target():
    assert not hasattr(C, "_R0_STACK"), (
        "torch_cgsim exports _R0_STACK again; a frozen copy of a live constant is how the "
        "two force paths came to disagree")


def test_both_paths_read_the_live_stacking_target():
    """The anti-regression test. Fails if any path goes back to a frozen copy.

    K_STACK ships at zero because the term is redundant with K_BB and K_ANGLE, so the term
    has to be switched back on for this test to have anything to respond to. That is the
    point: the property under test is that neither path reads a copy of STACK_R0 taken at
    import, and with the term disabled the test would pass vacuously.
    """
    pos, pairs = _chain()
    try:
        C.K_STACK = 500.0
        C._K_STACK = 500.0
        base_unified, base_batched = _energies(pos, pairs)
        C.STACK_R0 = 1.0
        alt_unified, alt_batched = _energies(pos, pairs)
    finally:
        C.STACK_R0 = 1.125
        C.K_STACK = 0.0
        C._K_STACK = 0.0
    assert base_unified != alt_unified, "cg_energy_forces ignored a change to STACK_R0"
    assert base_batched != alt_batched, (
        "cg_forces_explicit_batched ignored a change to STACK_R0 -- it is reading a frozen "
        "copy, so the two force paths would use different stacking targets")


def test_both_paths_read_the_live_dihedral_target():
    pos, pairs = _chain()
    try:
        base_unified, base_batched = _energies(pos, pairs)
        C.DIH_PPPP = math.acos(0.5)
        alt_unified, alt_batched = _energies(pos, pairs)
    finally:
        C.DIH_PPPP = math.acos(0.975)
    assert base_unified != alt_unified, "cg_energy_forces ignored a change to DIH_PPPP"
    assert base_batched != alt_batched, (
        "cg_forces_explicit_batched ignored a change to DIH_PPPP")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
