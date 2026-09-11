"""The stratified table builder: does it leave the pooled path alone, and does it report gaps?

fit() is the entry point every existing script uses; its output goes through prepare() and
into the energy. Adding a stratified mode to the same module must not move a single number on
that path, and the stratified mode must not be able to hand back a table fitted on too few
observations without saying so.

These use synthetic chains. The properties under test -- labelling alignment, shared support,
fallback reporting -- are properties of the code, not of the PDB database, and a test that
loads 96 files to check an off-by-one is a test nobody runs.

Run: python tests/test_boltzmann_stratified.py    (or python -m pytest tests/)
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import boltzmann_bonded as B          # noqa: E402


def _chain(names, seed=0):
    """A synthetic bead chain: (L, 3, 3) nm, P / C4-prime / N per residue, plus the letters."""
    rng = np.random.default_rng(seed)
    L = len(names)
    pos = np.zeros((L, 3, 3))
    for i in range(L):
        base = np.array([0.59 * i, 0.0, 0.0]) + rng.normal(0, 0.01, 3)
        pos[i, 0] = base
        pos[i, 1] = base + np.array([0.39 * 0.3, 0.39 * 0.95, 0.0]) + rng.normal(0, 0.01, 3)
        pos[i, 2] = pos[i, 1] + np.array([0.335 * 0.2, 0.335 * 0.9, 0.335 * 0.4]) \
            + rng.normal(0, 0.01, 3)
    return {"name": f"synth{seed}", "pos": pos, "pairs": [], "names": list(names)}


SEQ = "ACGUACGUACGUACGU"          # all four bases, 16 residues


def _structs(n=6, seq=SEQ):
    return [_chain(seq, seed=i) for i in range(n)]


def _coords(struct, name):
    pos = torch.tensor(struct["pos"].reshape(1, -1, 3), dtype=torch.float64)
    return B.coords_of(pos, name).reshape(-1).numpy()


# ------------------------------------------------------------- pooled path
def test_pooled_table_keys_are_exactly_what_callers_read():
    """prepare() and mixed_energy read these keys. A new key is not harmless: prepare()
    injects sigma=0.0 only when sigma is absent, and mixed_energy then defaults k_local to
    kBT/sigma^2, so an accidentally added sigma silently changes the energy."""
    t = B.fit(_structs())
    assert set(t) == set(B.COORDS)
    for name in B.COORDS:
        assert set(t[name]) == {"lo", "hi", "binw", "U", "centre", "n", "empty"}, name


def test_pooled_fit_is_identical_with_and_without_the_flag():
    """The regression guard for the refactor that moved the binning into a helper."""
    s = _structs()
    a, b = B.fit(s), B.fit(s, stratify=False)
    for name in B.COORDS:
        for field in ("lo", "hi", "binw", "n", "empty"):
            assert a[name][field] == b[name][field], (name, field)
        assert np.array_equal(a[name]["U"], b[name]["U"]), name
        assert np.array_equal(a[name]["centre"], b[name]["centre"]), name
    # the pooled table the stratified fit reports is bit-identical to the standalone one
    st = B.fit(s, stratify=True)
    for name in B.COORDS:
        assert np.array_equal(st[name]["pooled"]["U"], a[name]["U"]), name


def test_pooled_u_is_shifted_to_its_own_minimum_and_finite():
    t = B.fit(_structs())
    for name in B.COORDS:
        assert np.all(np.isfinite(t[name]["U"])), name
        assert t[name]["U"].min() == 0.0, name
        assert math.isclose(t[name]["binw"] * len(t[name]["U"]),
                            t[name]["hi"] - t[name]["lo"], rel_tol=1e-12)


# -------------------------------------------------------------- labelling
def test_labelling_lines_up_with_the_coordinate_windows():
    """The off-by-one that would matter: element k of the label array must describe window
    k of the coordinate array. Checked against the exact letters, not just the length."""
    names = list("ACGUAC")
    s = _chain(names)
    expected = {
        "bb_bond": ["AC", "CG", "GU", "UA", "AC"],
        "intra_pc": ["A", "C", "G", "U", "A", "C"],
        "intra_cn": ["A", "C", "G", "U", "A", "C"],
        "angle": ["C", "G", "U", "A"],            # default scheme is the middle residue
        "dihedral": ["CG", "GU", "UA"],           # the two residues of the central bond
        "stack": ["AG", "CU", "GA", "UC"],        # names[i] + names[i+2]
    }
    for name, want in expected.items():
        lab = B.labels_for(names, name)
        assert list(lab) == want, (name, list(lab), want)
        assert len(lab) == len(_coords(s, name)), name


def test_labelling_matches_the_coordinate_count_on_every_coordinate():
    for seq in ("ACGUACGUACGUACGU", "AAAAAAAAUUUU", "GCGCGCGC"):
        s = _chain(seq)
        for name in B.COORDS:
            assert len(B.labels_for(s["names"], name)) == len(_coords(s, name)), (seq, name)


def test_a_scheme_reaching_past_the_span_is_rejected():
    """A pair label on a one-residue coordinate would give L labels for L-1 observations."""
    with pytest.raises(ValueError):
        B.labels_for(list("ACGUAC"), "intra_pc", scheme="pair")


def test_an_unknown_scheme_is_rejected():
    with pytest.raises(KeyError):
        B.labels_for(list("ACGUAC"), "bb_bond", scheme="quadruplet")


def test_a_scalar_string_is_rejected_rather_than_read_as_one_residue():
    """np.asarray("ACGU") is a 0-d array of one string, which would make every downstream
    length check raise with a message that says nothing about the cause."""
    with pytest.raises(ValueError):
        B.labels_for("ACGUAC", "bb_bond")


# ------------------------------------------------------------ stratification
def test_group_tables_share_the_pooled_support():
    """Same lo, hi and binw as the pooled table, so the two are directly comparable and a
    caller can swap between them without a jump at the edges."""
    # 6 synthetic chains give 90 pair observations over 16 possible pairs, so the real
    # floor would put every group into fallback; this test is about the support, not the size
    st = B.fit(_structs(), stratify=True, min_obs=1)
    for name in B.COORDS:
        p = st[name]["pooled"]
        assert st[name]["groups"], name
        for g, t in st[name]["groups"].items():
            assert t["lo"] == p["lo"] and t["hi"] == p["hi"], (name, g)
            assert t["binw"] == p["binw"], (name, g)
            assert len(t["U"]) == len(p["U"]), (name, g)


def test_sparse_groups_fall_back_to_pooled_and_the_fallback_is_visible():
    """One dominant group and one rare one, so the partition is unambiguous."""
    structs = [_chain("A" * 39 + "U", seed=i) for i in range(4)]
    st = B.fit(structs, stratify=True, min_obs=100)
    bond = st["bb_bond"]
    assert bond["n"] == {"AA": 152, "AU": 4}, bond["n"]
    assert bond["fallback"] == ["AU"]
    assert list(bond["groups"]) == ["AA"]
    assert bond["groups"]["AA"]["n"] == 152
    assert bond["min_obs"] == 100
    # the count of a fallen-back group is kept, so the reader sees how far short it fell
    assert bond["n"]["AU"] == 4
    assert bond["n_obs"] == 156
    # and the residue labelling of the same structures is gated the same way
    assert st["intra_pc"]["n"] == {"A": 156, "U": 4}
    assert st["intra_pc"]["fallback"] == ["U"]


def test_a_floor_above_every_group_leaves_the_coordinate_fully_pooled():
    st = B.fit(_structs(), stratify=True, min_obs=10 ** 9)
    for name in B.COORDS:
        assert st[name]["groups"] == {}, name
        assert len(st[name]["fallback"]) == st[name]["n_groups"], name
        assert sum(st[name]["n"].values()) == st[name]["n_obs"], name


def test_pseudo_fraction_is_reported_and_matches_its_definition():
    st = B.fit(_structs(), stratify=True)
    for name in B.COORDS:
        for g, t in st[name]["groups"].items():
            want = 0.5 * 120 / (t["n"] + 0.5 * 120)
            assert math.isclose(t["pseudo_frac"], want, rel_tol=1e-12), (name, g)


def test_group_sigma_and_k_are_measured_from_the_groups_own_values():
    structs = _structs()
    st = B.fit(structs, stratify=True)
    name = "intra_cn"
    v = np.concatenate([_coords(s, name) for s in structs])
    lab = np.concatenate([B.labels_for(s["names"], name) for s in structs])
    for g, t in st[name]["groups"].items():
        vg = v[lab == g]
        assert math.isclose(t["sigma"], float(vg.std()), rel_tol=1e-12), g
        assert math.isclose(t["k"], B.KBT / float(vg.std()) ** 2, rel_tol=1e-12), g
    assert math.isclose(st[name]["pooled_k"], B.KBT / float(v.std()) ** 2, rel_tol=1e-12)


def test_structs_without_names_are_rejected_rather_than_silently_pooled():
    bare = [{"name": "x", "pos": _chain(SEQ)["pos"], "pairs": []}]
    with pytest.raises(ValueError):
        B.fit(bare, stratify=True)


def test_stratify_values_agrees_with_fit_stratified():
    """The null control in refit_tables_stratified.py calls stratify_values directly, so it
    has to be the same computation the builder performs, not a second implementation."""
    structs = _structs()
    st = B.fit(structs, stratify=True)
    name = "angle"
    v = np.concatenate([_coords(s, name) for s in structs])
    lab = np.concatenate([B.labels_for(s["names"], name) for s in structs])
    groups, counts, fallback = B.stratify_values(v, lab, st[name]["pooled"])
    assert sorted(groups) == sorted(st[name]["groups"])
    assert counts == st[name]["n"]
    assert sorted(fallback) == st[name]["fallback"]
    for g in groups:
        assert np.array_equal(groups[g]["U"], st[name]["groups"][g]["U"]), g


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as exc:                      # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}: {exc!r}")
    print()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)