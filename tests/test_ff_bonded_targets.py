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

It was not one constant. Twenty frozen copies lived in a block of their own -- _K_BOND_BB,
_K_PAIR, _R0_BB and seventeen more, each carrying a comment saying it was "kept in step" with
a live global that nothing kept it in step with. Removing _R0_STACK left the rest of that
apparatus in place. It is all gone now: every term in cg_energy_forces,
cg_forces_explicit_batched, cg_forces_explicit and cg_energy_3bead reads the live global.
The tests below reject any module-level alias reappearing (a bare "X = Y" evaluates once at
import, which IS the freeze) and perturb every constant in the block to check that each path
that implements the term moves with it.

Run: python tests/test_ff_bonded_targets.py
"""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import torusfold.scheme2.torch_cgsim as C


def _chain(L=8, pair_ij=()):
    """A straight, evenly spaced backbone in nm, enough to exercise every bonded term."""
    pos = np.zeros((L, 3, 3))
    for i in range(L):
        base = np.array([0.59 * i, 0.0, 0.0])
        pos[i, 0] = base
        pos[i, 1] = base + np.array([0.39 * 0.3, 0.39 * 0.95, 0.0])
        pos[i, 2] = pos[i, 1] + np.array([0.335 * 0.2, 0.335 * 0.9, 0.335 * 0.4])
    tensor = torch.tensor(pos.reshape(1, 3 * L, 3), dtype=torch.float64)
    if len(pair_ij):
        pairs = torch.tensor(list(pair_ij), dtype=torch.long).reshape(-1, 2)
    else:
        pairs = torch.zeros((0, 2), dtype=torch.long)
    return tensor, pairs


def _folded_chain(L=8):
    """A compact, slightly non-planar chain.

    Two terms need geometry rather than a generic chain to be non-negligible: the BSJ guide
    acts on P(0)-P(L-1), which is 4.13 nm in _chain() and makes it exp(-21) small, and the
    dihedral term needs the P-P-P-P windows off 0 and 180 degrees or it sits on its own
    target by accident. Compressing x puts the ends 0.9 nm apart; the z zigzag breaks the
    planar symmetry.
    """
    pos, pairs = _chain(L)
    pos[0, :, 0] *= 0.9 / (0.59 * (L - 1))
    for i in range(1, L - 1):
        pos[0, 3 * i, 2] += 0.06 * (i % 2)
    return pos, pairs


def _clash_chain(L=8):
    """_folded_chain with one N bead inside CLASH_SIGMA of a C4' more than two residues away
    (the neighbour list excludes |i - j| <= 2, so a nearer residue would not see it)."""
    pos, pairs = _folded_chain(L)
    near = pos[0, 3 * 0 + 1] + torch.tensor([0.0, 0.0, 0.25], dtype=torch.float64)
    pos[0, 3 * (L - 2) + 2] = near
    return pos, pairs


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
        base_unified, base_batched = _energies(pos, pairs)
        C.STACK_R0 = 1.0
        alt_unified, alt_batched = _energies(pos, pairs)
    finally:
        C.STACK_R0 = 1.125
        C.K_STACK = 0.0
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


def test_the_two_intra_bonds_are_separate_live_constants():
    """P-C4' and C4'-N are not the same spring and must not share one number.

    Their reference spreads over 126 gap-free chains are 0.010963 and 0.008278 nm, which under
    k = kBT/sigma^2 is 20752.7 against 36399.2. The single K_INTRA of 400 they replaced was
    52x and 91x too soft. Both coordinates are one-dimensional -- C4' appears only in these two
    bonds and the clash term, and the clash repulsion does not fire at native geometry -- so
    the criterion is exact here rather than an extrapolation.
    """
    assert not hasattr(C, "K_INTRA"), (
        "K_INTRA is back; one value cannot serve two coordinates whose reference spreads differ "
        "by 1.3x, which is a factor 1.75 in stiffness")
    npz = Path(__file__).resolve().parent.parent / "results" / "boltzmann_tables_clean.npz"
    if npz.exists():
        z = np.load(npz)
        for name, value in (("intra_pc", C.K_INTRA_PC), ("intra_cn", C.K_INTRA_CN)):
            want = 2.494 / float(z[f"{name}__sigma"]) ** 2
            assert abs(value / want - 1.0) < 0.01, (
                f"{name}: the constant is {value} but kBT/sigma^2 is {want}")


def test_both_paths_read_the_live_intra_constants():
    """Either force path silently ignoring a change here is the frozen-copy bug again."""
    pos, pairs = _chain()
    for name in ("K_INTRA_PC", "K_INTRA_CN"):
        try:
            base_unified, base_batched = _energies(pos, pairs)
            setattr(C, name, getattr(C, name) * 1.5)
            alt_unified, alt_batched = _energies(pos, pairs)
        finally:
            setattr(C, name, getattr(C, name) / 1.5)
        assert base_unified != alt_unified, f"cg_energy_forces ignored a change to {name}"
        assert base_batched != alt_batched, (
            f"cg_forces_explicit_batched ignored a change to {name} -- it is reading a frozen "
            f"copy, so the two paths would use different intra-residue bonds")


# ── every frozen copy in the block, not just _R0_STACK ─────────────────────────────────────
# The names the explicit-force snapshot block defined (torch_cgsim.py, just above
# _explicit_forces_bonds). _K_GB and _K_SASA are on the list even though there was never a live
# K_GB or K_SASA: their numbers were written inline in the GB/SA term, so those two copies were
# dead code waiting for a reader.
SNAPSHOT_NAMES = (
    "_K_BOND_BB", "_K_BOND_INTRA", "_K_PAIR", "_K_STACK", "_K_ANGLE", "_K_DIH",
    "_K_CLASH", "_K_BSJ", "_K_BSJ_GUIDE", "_K_PAIR_GUIDE", "_K_BSJ_CONTACT",
    "_K_BPP", "_K_MG", "_K_GB", "_K_SASA",
    "_R0_BB", "_R0_INTRA_PC", "_R0_INTRA_CN", "_R0_PAIR", "_R0_CLASH",
)


def _three_paths(pos, pairs, weights):
    """Energy from all three implementations on the same geometry."""
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(pos)
    return (
        float(C.cg_energy_forces(pos, pairs, weights, cell_list=cl)[0]),
        float(C.cg_forces_explicit_batched(pos, pairs, weights, cell_list=cl)[0]),
        float(C.cg_forces_explicit(pos, pairs, weights, cell_list=cl)[0]),
    )


def test_no_frozen_snapshot_of_any_live_constant():
    present = [name for name in SNAPSHOT_NAMES if hasattr(C, name)]
    assert not present, (
        f"torch_cgsim exports {present} again; a module-level copy of a live constant is how "
        f"one force path came to read a stale value while another read the global")


def test_no_module_level_alias_in_the_force_field_source():
    """A bare "X = Y" at module level is a freeze, whatever the two names are.

    It is evaluated once, at import, so a later retune of Y is invisible to every use of X and
    nothing raises. ANGLE_K and DIH_K were exactly that. The AST check below cannot be fooled
    by a name this test file does not know about.
    """
    tree = ast.parse(Path(C.__file__).read_text(encoding="utf-8"))
    aliases = [f"line {node.lineno}: {target.id} = {node.value.id}"
               for node in tree.body if isinstance(node, ast.Assign)
               for target in node.targets
               if isinstance(target, ast.Name) and isinstance(node.value, ast.Name)]
    assert not aliases, (
        f"module-level aliases in {Path(C.__file__).name}: {aliases}. Each one binds at import "
        f"and silently stops tracking the constant it copies")


def test_every_live_constant_reaches_all_three_force_paths():
    """Perturb a live constant; every implementation that has the term must move with it.

    cg_energy_forces, cg_forces_explicit_batched and cg_forces_explicit are three separate
    implementations of the same field. The snapshot block existed only for the two explicit
    paths, which is precisely how a retune could move one and not the others.

    The geometries are picked so each term is actually firing -- see _folded_chain for the BSJ
    guide and the dihedral, _clash_chain for K_CLASH and CLASH_SIGMA, and the pair list for
    K_PAIR and PAIR_NN. K_STACK ships at zero, so it is switched on for its case and off
    again, the same way test_both_paths_read_the_live_stacking_target does it.
    """
    folded, _ = _folded_chain()
    paired_pairs = torch.tensor([[0, 2]], dtype=torch.long)
    weight = torch.ones(1, dtype=torch.float64)
    clash, _ = _clash_chain()
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    cases = (
        ("K_BB", folded, paired_pairs, weight, {}),
        ("K_INTRA_PC", folded, paired_pairs, weight, {}),
        ("K_INTRA_CN", folded, paired_pairs, weight, {}),
        ("K_BSJ", folded, paired_pairs, weight, {}),
        ("K_ANGLE", folded, paired_pairs, weight, {}),
        ("K_DIH", folded, paired_pairs, weight, {}),
        ("K_BSJ_GUIDE", folded, paired_pairs, weight, {}),
        ("K_PAIR", folded, paired_pairs, weight, {}),
        ("PAIR_NN", folded, paired_pairs, weight, {}),
        ("BOND_P_NEXT", folded, paired_pairs, weight, {}),
        ("BOND_P_C4", folded, paired_pairs, weight, {}),
        ("BOND_C4_N", folded, paired_pairs, weight, {}),
        ("K_STACK", folded, paired_pairs, weight, {"K_STACK": 500.0}),
        ("K_CLASH", clash, no_pairs, None, {}),
        ("CLASH_SIGMA", clash, no_pairs, None, {}),
    )
    paths = ("cg_energy_forces", "cg_forces_explicit_batched", "cg_forces_explicit")
    failures = []
    for name, pos, pair_ij, weights, enable in cases:
        saved = {key: getattr(C, key) for key in enable}
        try:
            for key, value in enable.items():
                setattr(C, key, value)
            before = _three_paths(pos, pair_ij, weights)
            setattr(C, name, getattr(C, name) * 1.5)
            after = _three_paths(pos, pair_ij, weights)
        finally:
            setattr(C, name, getattr(C, name) / 1.5)
            for key, value in saved.items():
                setattr(C, key, value)
        for path, was, now in zip(paths, before, after):
            if was == now:
                failures.append(f"{path} ignored a change to {name}")
    assert not failures, (
        "a frozen copy is back: " + "; ".join(failures) +
        ". A term's constant has to reach every implementation that has the term")


def test_the_terms_only_cg_energy_forces_has_read_live_constants():
    """K_PAIR_GUIDE, K_BPP and K_BSJ_CONTACT have no term in either explicit path.

    cg_forces_explicit_batched and cg_forces_explicit implement bonds, BSJ, angle, dihedral,
    WC pairing, stacking, clash, BSJ guide and GB/SA/Mg -- there is no pair-guide, BPP or
    BSJ-contact term in them at all, so they cannot respond to these three however the
    constants are read. The path that does implement them is what has to read the live global:
    _K_PAIR_GUIDE, _K_BPP and _K_BSJ_CONTACT were harmless only as long as nothing read them.
    """
    folded, _ = _folded_chain()
    paired_pairs = torch.tensor([[0, 2]], dtype=torch.long)
    weight = torch.ones(1, dtype=torch.float64)
    long_chain, _ = _chain(20)
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    for name, pos, pair_ij, weights in (
            ("K_PAIR_GUIDE", folded, paired_pairs, weight),
            ("K_BPP", folded, paired_pairs, weight),
            ("K_BSJ_CONTACT", long_chain, no_pairs, None)):
        cl = C.GPUCellList(cell_size=1.5)
        cl.build(pos)
        before = float(C.cg_energy_forces(pos, pair_ij, weights, cell_list=cl)[0])
        try:
            setattr(C, name, getattr(C, name) * 1.5)
            after = float(C.cg_energy_forces(pos, pair_ij, weights, cell_list=cl)[0])
        finally:
            setattr(C, name, getattr(C, name) / 1.5)
        assert after != before, f"cg_energy_forces ignored a change to {name}"


def test_kmg_reaches_the_explicit_paths_that_use_it():
    """K_MG has no term in cg_energy_forces, so "both paths respond" is impossible for it.

    The unified path's Mg pair energy is -c_mg * exp(-r/0.3), written with the concentration
    argument rather than K_MG; the two explicit paths use -K_MG * exp(-softmin/0.3). That
    asymmetry predates the snapshot block, but _K_MG was a copy of K_MG, so what can break here
    is "the paths that have the term read the live one".
    """
    folded, _ = _folded_chain()
    no_pairs = torch.zeros((0, 2), dtype=torch.long)
    cl = C.GPUCellList(cell_size=1.5)
    cl.build(folded)
    before = (
        float(C.cg_forces_explicit_batched(folded, no_pairs, None, cell_list=cl)[0]),
        float(C.cg_forces_explicit(folded, no_pairs, None, cell_list=cl)[0]),
    )
    try:
        C.K_MG = C.K_MG * 1.5
        after = (
            float(C.cg_forces_explicit_batched(folded, no_pairs, None, cell_list=cl)[0]),
            float(C.cg_forces_explicit(folded, no_pairs, None, cell_list=cl)[0]),
        )
    finally:
        C.K_MG = C.K_MG / 1.5
    assert after[0] != before[0], "cg_forces_explicit_batched ignored a change to K_MG"
    assert after[1] != before[1], "cg_forces_explicit ignored a change to K_MG"


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
