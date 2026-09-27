"""What plan_c_loop's pure parts must do, checked without sampling anything.

The three arms are an experiment, not a library, so the tests here are about the pieces that a
silent change would corrupt WITHOUT failing loudly: the field file round-trip (a table written
wrong is still a readable npz), the two fits (a Chebyshev projection off by one basis function is
still smooth), the stationarity ratio (a wrong weighting is still a number), and the pool (a
loader that reorders changes the experiment and nothing else would notice -- the A/B arms and this
run must be the same seven chains, or the comparison to their numbers is between two pools).

Nothing here samples: run_round is minutes, and these are the checks that would catch a mistake in
seconds.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import boltzmann_bonded as B          # noqa: E402
import ibi_bonded as I                # noqa: E402
import plan_c_loop as L               # noqa: E402


def _table(nbins=40, lo=-1.0, hi=1.0, U=None):
    """A table in the stored format, on an explicit grid so the arithmetic below is checkable."""
    edges = np.linspace(lo, hi, nbins + 1)
    centre = 0.5 * (edges[:-1] + edges[1:])
    return {"lo": lo, "hi": hi, "binw": (hi - lo) / nbins, "centre": centre,
            "U": np.zeros(nbins) if U is None else np.asarray(U, dtype=float),
            "sigma": 0.1}


# ------------------------------------------------------------------ the Boltzmann inverse
def test_target_from_counts_uniform_is_flat():
    """Equal counts mean no potential: this is the one case where the answer is known exactly."""
    U = L._target_from_counts(np.full(50, 7, dtype=np.int64))
    assert np.allclose(U, 0.0)


def test_target_from_counts_is_kbt_times_ln_ratio():
    """U_i - U_0 = -kBT ln(p_i/p_0) with the pseudocount inside p, and nothing else."""
    counts = np.array([99, 49, 9, 1], dtype=np.int64)
    p = I.probability_from_counts(counts, pseudo=I.DEFAULT_PSEUDO)
    U = L._target_from_counts(counts)
    assert U.min() == pytest.approx(0.0)
    shifted = U - U.min()
    assert np.allclose(shifted, -B.KBT * np.log(p) - (-B.KBT * np.log(p)).min())
    # monotone: fewer observations is a higher potential, and the spacing is kBT*ln of the ratio
    assert np.all(np.diff(U) > 0)
    assert U[1] - U[0] == pytest.approx(B.KBT * np.log(p[0] / p[1]))


def test_target_counts_of_one_bin_are_finite():
    """A bin nobody visited gets the pseudocount, not -ln(0) = inf: the interpolator needs a value."""
    U = L._target_from_counts(np.array([1000, 0, 0], dtype=np.int64))
    assert np.all(np.isfinite(U))


# ------------------------------------------------------------------ C2's fit
def test_chebyshev_fit_recovers_a_low_order_target_exactly():
    """A target INSIDE the span comes back unchanged (up to the min shift the tables all carry).

    This is the check that the design matrix and the coefficient vector are used consistently: an
    off-by-one in the basis index, or a transpose, would still return a smooth potential -- and it
    would not recover a target that is exactly representable.
    """
    tab = _table(nbins=64)
    A, _x = I._chebyshev_design(tab["centre"], tab["lo"], tab["hi"], 5)
    coef = np.array([1.0, -2.0, 0.5, 0.25, -0.1])
    # NO OFFSET, deliberately: the basis is T_1..T_K and k=0 is absent (ibi_bonded._chebyshev_design
    # says why -- a constant has no force), so a target carrying a large arbitrary constant is NOT
    # representable. Measured while writing this test: adding 137 kJ/mol to a target of 0-5 kJ/mol
    # range turned the recovered coefficients into [1, -125, 0.5, -64, -0.1] and the fit oscillated
    # over 260 kJ/mol -- the constants are absorbed by the higher functions on a discretised grid.
    # The loop never feeds one: _target_from_counts subtracts its own minimum first.
    target = A @ coef
    U_fit, diag = L._chebyshev_fit(tab, target, np.full(64, 10, dtype=np.int64), K=5)
    assert diag["K"] == 5
    assert np.allclose(U_fit, target - target.min(), atol=1e-8)


def test_chebyshev_fit_K1_is_linear_in_the_coordinate():
    """K=1 is one Chebyshev function, which is affine in x: a second difference of zero."""
    tab = _table(nbins=64)
    rng = np.random.default_rng(0)
    U_fit, _diag = L._chebyshev_fit(tab, rng.normal(size=64), np.full(64, 5, dtype=np.int64), K=1)
    d2 = np.diff(U_fit, 2)
    assert np.allclose(d2, 0.0, atol=1e-9)


def test_chebyshev_fit_is_weighted_by_the_ensemble():
    """A bin the ensemble never visits must not drag the fit: its U is a pseudocount artefact.

    Measured shape of the problem (the smoke run, 2026-09-22): with 32 frames over 1000 bins the
    unweighted target reached 52 kJ/mol in bins holding no observations, and the sampler feels
    force, not probability. The test is the extreme case -- one bin, one observation, and a target
    that says +500 kJ/mol there.
    """
    tab = _table(nbins=64)
    counts = np.full(64, 1000, dtype=np.int64)
    counts[7] = 1
    target = np.zeros(64)
    target[7] = 500.0
    U_fit, diag = L._chebyshev_fit(tab, target, counts, K=4)
    assert np.abs(U_fit).max() < 1.0, "one unvisited bin moved the potential"
    assert np.all(np.isfinite(diag["coef"]))


# ------------------------------------------------------------------ stationarity
def test_stationarity_is_zero_for_an_ensemble_that_did_not_move():
    p = I.probability_from_counts(np.array([10, 20, 30, 40], dtype=np.int64))
    st = L._stationarity(p, p)
    assert st["mass_weighted_mean"] == pytest.approx(0.0)
    assert st["max"] == pytest.approx(0.0)
    assert st["n_bins"] == 4


def test_stationarity_is_the_mass_weighted_ln_ratio():
    prev = np.array([0.5, 0.5])
    now = np.array([0.25, 0.75])
    st = L._stationarity(prev, now)
    w = 0.5 * (prev + now)
    lr = np.abs(np.log(now / prev))
    assert st["mass_weighted_mean"] == pytest.approx(float((w * lr).sum() / w.sum()))
    assert st["max"] == pytest.approx(float(lr.max()))


def test_stationarity_of_the_first_round_is_not_a_number():
    """Round 1 has no previous ensemble, and returning 0.0 there would read as convergence."""
    assert L._stationarity(None, np.array([0.5, 0.5])) is None


# ------------------------------------------------------------------ retention summary
def test_summarize_retention_counts_the_chains_that_left():
    rows = [{"name": "a", "L": 26, "rmsd_dep_mean": 0.5, "rmsd_frame_spread": 0.2},
            {"name": "b", "L": 30, "rmsd_dep_mean": 1.5, "rmsd_frame_spread": 0.4},
            {"name": "c", "L": 34, "rmsd_dep_mean": 12.0, "rmsd_frame_spread": 3.0}]
    s = L.summarize_retention(rows)
    assert s["n"] == 3
    assert s["median_dep_mean"] == pytest.approx(1.5)
    assert s["n_moved_over_10A"] == 1
    assert s["moved"] == ["c"]
    assert s["median_spread"] == pytest.approx(0.4)
    assert s["worst"][0][1] == "c"
    assert L.summarize_retention([]) is None


# ------------------------------------------------------------------ field files
def test_write_field_round_trips_through_load_clean_tables(tmp_path):
    """The file has to satisfy BOTH readers: cg_potentials.use_table_file and ibi_core.load_tables.

    load_tables demands all six keys for every coordinate it is not told to skip, which is why the
    writer carries the two rigid coordinates rather than only the four the loop samples.
    """
    tabs = {c: _table(nbins=8, U=np.arange(8, dtype=float) + i)
            for i, c in enumerate(B.COORDS)}
    path = L.write_field(tmp_path / "f.npz", tabs)
    back = L.load_field(path)
    assert set(back) == set(B.COORDS)
    for c in B.COORDS:
        assert np.allclose(back[c]["U"], tabs[c]["U"])
        assert back[c]["lo"] == tabs[c]["lo"] and back[c]["hi"] == tabs[c]["hi"]
        assert back[c]["sigma"] == tabs[c]["sigma"]
    # the skips the sampler uses must also succeed (that is the path run_round takes)
    assert set(I.load_clean_tables(str(path))) == set(B.COORDS)


# ------------------------------------------------------------------ C0's refusal path
def test_c0_refuses_when_the_reference_support_was_never_visited():
    """plan_update's unvisited_reference_support refusal has to reach the caller as a refusal.

    Measured 2026-09-22 on the wiring smoke run (2 chains x 1 replica x 32 frames): 677 of 1000
    dihedral bins holding 0.34 of the reference probability were never visited, and the correction
    there would have been the pseudocount. The arm must record that, not apply it.
    """
    tab = _table(nbins=100)
    p_ref = np.zeros(100)
    p_ref[10:60] = 1.0 / 50.0                     # half the support holds all the reference mass
    counts = np.zeros(100, dtype=np.int64)
    counts[10:35] = 5                             # the simulation covers the first half of it
    with pytest.raises(I.IBIRefusal) as exc:
        L._fit_coord("C0", tab, counts, 125, 0, p_ref, K=8)
    assert exc.value.reason == I.REFUSE_UNSUPPORTED


def test_order_key_handles_both_worker_shapes():
    """The two workers disagree on their return shape; a sort that assumed one would KeyError."""
    assert L._order_key((3, {"a": 1})) == 3
    assert L._order_key({"idx": 5}) == 5


# ------------------------------------------------------------------ the pool
def test_pool_rule_selects_the_ab_band_and_its_holdout():
    """The rule on a synthetic list: pairs >= 8, 24 <= L <= 34, and the HEAD of the loader's order.

    Two off-by-ones are silent in a real run and both matter: an inclusive 35-residue bound would
    admit a chain the A/B arms never had, and taking the tail instead of the head changes which
    seven chains the experiment is about -- while every number downstream still looks plausible.
    """
    def s(name, L, pairs):
        return {"name": name, "pos": np.zeros((L, 3)), "pairs": [(0, 1)] * pairs}

    structs = [s("too_short", 23, 10),      # below the band
               s("bar_b", 24, 8),           # the band's lower edge, and exactly 8 pairs
               s("foo_a", 30, 20),
               s("skip_pairs", 30, 7),      # inside the band, too few pairs
               s("bar_c", 34, 40),          # the band's upper edge
               s("too_long", 35, 40),       # above the band
               s("bar_d", 26, 12)]
    pool, hold = L.build_pool(2, 2, structs=structs)
    assert [x["name"] for x in pool] == ["bar_b", "foo_a"]         # loader order, head
    assert [x["name"] for x in hold] == ["bar_c", "bar_d"]         # the next two, disjoint
    for name in ("too_short", "skip_pairs", "too_long"):
        assert name not in [x["name"] for x in pool + hold]


def test_pool_refuses_when_the_band_is_too_small():
    """Asking for more than the band holds must stop the run, not silently shrink the pool.

    Measured 2026-09-22: without TORUSFOLD_RSRNASP the loader extracts eight chains in the band, so
    a bare checkout cannot run the seven-chain pool plus a four-chain holdout -- and a loop that
    quietly took what was there would report a two-chain holdout as if it were the plan's four.
    """
    only = [{"name": "x", "pos": np.zeros((30, 3)), "pairs": [(0, 1)] * 9}]
    with pytest.raises(SystemExit):
        L.build_pool(2, 2, structs=only)


def test_the_pool_is_the_ab_pool():
    """Seven chains, the same seven the A/B arms ran: the comparison depends on it.

    ibi_loop.py:662's rule (>= 8 pairs, 24 <= L <= 34, loader order, first seven). The A/B logs
    (results/ab_operator/*.log) print the same head: 10ZT_1(29), 1L2X(27), 1Q96(26), 2IL9(34),
    3BO1_2(27), 3MJA(29) and one more, which is 4PCJ(30).
    """
    try:
        pool, holdout = L.build_pool(7, 4)
    except (Exception, SystemExit) as exc:        # the full database is not always mounted
        pytest.skip(f"the full structure database is not mounted here ({exc}); the RULE is pinned "
                    f"by test_pool_rule_selects_the_ab_band_and_its_holdout")
    assert [s["name"] for s in pool] == ["10ZT_1", "1L2X", "1Q96", "2IL9", "3BO1_2", "3MJA",
                                         "4PCJ"]
    assert [len(s["pos"]) for s in pool] == [29, 27, 26, 34, 27, 29, 30]
    assert [s["name"] for s in holdout] == ["4RZD", "6D6V_5", "6PMO", "6R47"]
    assert not set(s["name"] for s in holdout) & set(s["name"] for s in pool)

# ------------------------------------------------------------------ the stabilised Chebyshev arm
def test_arm_names_split_into_stable_and_K():
    """C2s4 is the stabilised fit at K=4; everything else takes --K. A silent misread here would
    run a different experiment under the same name."""
    assert L._is_stable_c2("C2s") and L._is_stable_c2("C2s2")
    assert L._is_stable_c2("C2s4") and L._is_stable_c2("C2s8")
    assert not L._is_stable_c2("C2") and not L._is_stable_c2("C1") and not L._is_stable_c2("C0")
    assert L._arm_K("C2s4", 8) == 4 and L._arm_K("C2s2", 8) == 2
    assert L._arm_K("C2s", 8) == 8 and L._arm_K("C2", 8) == 8 and L._arm_K("C0", 3) == 3


def test_stable_fit_has_no_offset_under_the_ensemble_mass():
    """The gauge the plain fit got wrong: measured 2026-09-22, its first round came back as a
    +49 kJ/mol constant over the occupied bins (spread 1.5), which has no force anywhere and made
    max|dU| report where the polynomial bottomed out instead of what the sampler would feel."""
    tab = _table(nbins=64)
    rng = np.random.default_rng(3)
    counts = (1000 * np.exp(-0.5 * ((np.arange(64) - 32) / 8.0) ** 2)).astype(np.int64)
    target = 137.0 + rng.normal(scale=2.0, size=64)      # a big constant on top of a small shape
    U_fit, diag = L._chebyshev_fit_stable(tab, target, counts, K=8)
    p = I.probability_from_counts(counts.astype(float), pseudo=I.DEFAULT_PSEUDO)
    assert (p * U_fit).sum() / p.sum() == pytest.approx(0.0, abs=1e-9)
    assert diag["fit"] == "chebyshev_ridge" and diag["K"] == 8
    assert abs(diag["mass_offset_removed"]) > 10.0, "the constant was not there to remove"


def test_stable_fit_support_cut_drops_pseudocount_bins():
    """The support mask is the fix for the bins whose target is the logarithm of a pseudocount: at
    1e-3 of the mode they are out of the fit entirely, and the diagnostic says how many survived."""
    tab = _table(nbins=64)
    counts = np.full(64, 10_000, dtype=np.int64)
    counts[7] = 1                                        # 1e-4 of the mode: below the cut
    target = np.zeros(64)
    target[7] = 500.0
    U_fit, diag = L._chebyshev_fit_stable(tab, target, counts, K=4)
    assert diag["support_bins"] == 64 - 1
    assert np.abs(U_fit).max() < 1.0, "an excluded bin still moved the potential"
    assert np.all(np.isfinite(diag["coef"]))


def test_stable_fit_ridge_keeps_a_rank_deficient_design_finite():
    """One visited bin makes the weighted design rank 1: without the ridge the solve is singular,
    with it the coefficients shrink as the ridge grows (the same relative form
    ibi_bonded.moment_correction uses on the same basis)."""
    tab = _table(nbins=64)
    counts = np.zeros(64, dtype=np.int64)
    counts[31] = 500
    target = np.zeros(64)
    small, d_small = L._chebyshev_fit_stable(tab, target, counts, K=8, ridge=1e-6)
    big, d_big = L._chebyshev_fit_stable(tab, target, counts, K=8, ridge=1e0)
    assert np.all(np.isfinite(small)) and np.all(np.isfinite(big))
    assert np.abs(big).max() <= np.abs(small).max()
    assert d_big["ridge_lambda"] > d_small["ridge_lambda"]


def test_stable_fit_gain_damps_the_step_one_for_one():
    """gain is the loop's own damping instrument (ibi_loop:862); at 0.5 the step is half of what
    the undamped fit asked for, measured against the old table."""
    tab = _table(nbins=64, U=np.linspace(0.0, 5.0, 64))
    rng = np.random.default_rng(11)
    counts = (500 * np.exp(-0.5 * ((np.arange(64) - 20) / 6.0) ** 2)).astype(np.int64)
    target = 20.0 * np.sin(np.linspace(0, 2.0, 64))
    full, _ = L._chebyshev_fit_stable(tab, target, counts, K=6, gain=1.0)
    half, _ = L._chebyshev_fit_stable(tab, target, counts, K=6, gain=0.5)
    old = tab["U"]
    assert np.allclose(half - old, 0.5 * (full - old), atol=1e-9)


def test_fit_coord_refuses_an_unrepresentable_step_instead_of_crashing():
    """Measured 2026-09-22 on the smoke protocol at gain 30: the fitted U spanned more than
    exp(-U/kBT) holds, ibi_bonded raised, and the whole run died before writing a record. The fit
    now refuses -- the arm records the wanted step and stops."""
    tab = _table(nbins=64)
    counts = (1000 * np.exp(-0.5 * ((np.arange(64) - 32) / 8.0) ** 2)).astype(np.int64)
    with pytest.raises(I.IBIRefusal) as exc:
        L._fit_coord("C2s", tab, counts, int(counts.sum()), 0, np.ones(64) / 64.0, K=4,
                     gain=1e4)
    assert exc.value.status == "unrepresentable"
    assert "max|dU|" in str(exc.value)


def test_stationarity_reports_total_variation():
    """The ln-ratio mean is the instrument the plan is written in; TV says how much probability
    moved. Two distributions that differ by a quarter of their mass: TV = 0.25 by hand."""
    st = L._stationarity(np.array([0.5, 0.5]), np.array([0.25, 0.75]))
    assert st["total_variation"] == pytest.approx(0.25)
    same = L._stationarity(np.array([0.5, 0.5]), np.array([0.5, 0.5]))
    assert same["total_variation"] == pytest.approx(0.0)


# ------------------------------------------------------------------ the dihedral arms
def test_dihedral_arm_specs_change_only_the_dihedral():
    """D02/D05 damp the dihedral's refit; Dtbl hands the dihedral to the production table inversion.
    bb_bond and angle must keep gain 1.0 in all three, or the arms differ in two things at once."""
    assert L._is_dihedral_arm("D02") and L._is_dihedral_arm("D05") and L._is_dihedral_arm("Dtbl")
    assert not L._is_dihedral_arm("C2s8") and not L._is_dihedral_arm("C0")
    assert L._spec_for("D02", "dihedral", 1.0) == ("chebyshev_ridge", 0.2)
    assert L._spec_for("D05", "dihedral", 1.0) == ("chebyshev_ridge", 0.5)
    assert L._spec_for("Dtbl", "dihedral", 1.0) == ("plan_update", 1.0)
    for arm in ("D02", "D05", "Dtbl"):
        for coord in ("bb_bond", "angle"):
            assert L._spec_for(arm, coord, 1.0) == ("chebyshev_ridge", 1.0), (arm, coord)


def test_angle_arms_change_only_the_angle():
    """CA16 moves the angle to a local basis; CBdep moves it to the production rule (deposited
    target). bb_bond and the dihedral must stay the C2s refit, or the C comparison is two variables."""
    assert L._spec_for("CA16", "angle", 1.0) == ("bspline", 16, 1e-3)
    assert L._spec_for("CBdep", "angle", 1.0) == ("plan_update", 1.0)
    for arm in ("CA16", "CBdep"):
        for coord in ("bb_bond", "dihedral"):
            assert L._spec_for(arm, coord, 1.0) == ("chebyshev_ridge", 1.0), (arm, coord)
    assert L._uses_refit("CA16") and L._uses_refit("CBdep")
    assert not L._uses_refit("C0")


def test_dihedral_arms_are_guarded_and_plain_c0_is_not_rerouted():
    """The guard and the stop-on-refusal apply to any arm whose step is a fit -- including the D
    arms -- while a plain C0 must keep the production rule at gain 1.0 whatever else is set."""
    assert L._uses_refit("D02") and L._uses_refit("Dtbl") and L._uses_refit("C2s8")
    assert not L._uses_refit("C0") and not L._uses_refit("C1")


def test_dtbl_dihedral_takes_the_plan_update_path_with_the_arm_gain(monkeypatch):
    """One call of _fit_coord with coord='dihedral' must reach plan_update, not the refit, and must
    not carry a global --c2-gain into it: the D arms' damping is per coordinate by construction."""
    seen = {}

    class _Res:
        ok = False
        status = "refused"
        reason = "test"
        diagnostics = {}
        table = None

    def fake_plan_update(table, hist, p_ref, **kw):
        seen.update(kw)
        return _Res()

    monkeypatch.setattr(L.I, "plan_update", fake_plan_update)
    tab = _table(nbins=32)
    counts = (100 * np.exp(-0.5 * ((np.arange(32) - 16) / 3.0) ** 2)).astype(np.int64)
    p_ref = np.ones(32) / 32.0
    # Dtbl's dihedral goes through plan_update, with the ARM's gain and not a global 0.3
    with pytest.raises(L.I.IBIRefusal):
        L._fit_coord("Dtbl", tab, counts, int(counts.sum()), 0, p_ref, K=8, gain=0.3,
                     coord="dihedral")
    assert seen.get("gain") == 1.0, f"the arm gain did not reach plan_update: {seen}"
    # ... and its bb_bond keeps the refit, at the global gain
    _u, d_bb = L._fit_coord("Dtbl", tab, counts, int(counts.sum()), 0, p_ref, K=8, gain=0.3,
                            coord="bb_bond")
    assert d_bb["fit"] == "chebyshev_ridge" and d_bb["gain"] == pytest.approx(0.3)
    # D02's dihedral is the refit, damped to 0.2 whatever the global gain says
    _u2, d_dih = L._fit_coord("D02", tab, counts, int(counts.sum()), 0, p_ref, K=8, gain=1.0,
                              coord="dihedral")
    assert d_dih["fit"] == "chebyshev_ridge" and d_dih["gain"] == pytest.approx(0.2)

